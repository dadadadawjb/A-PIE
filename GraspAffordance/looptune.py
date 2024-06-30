import hydra
import logging
import os
from pathlib import Path
import json
import itertools
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from warmup_scheduler import GradualWarmupScheduler

from dataset import LoopDataset
from models.model import grasp_embedding_network
from inference import inference
from test import count_hit, count_affordable
from utils import AverageMeter, random_point_dropout, random_scale_point_cloud, shift_point_cloud, \
    random_scale_grasp, shift_grasp, random_scale_joint, shift_joint


# `config_looptune.yaml` should match with original `config.yaml`
@hydra.main(config_path='./', config_name='config_looptune', version_base='1.2')
def main(cfg):
    logger = logging.getLogger(__name__)
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']

    # load network
    logger.info('load network')
    model = grasp_embedding_network(cfg).to(cfg.device)
    model.load_state_dict(torch.load(f'{cfg.weight_path}/model_latest.pth'))

    # prepare data
    logger.info('prepare data')
    dataset = LoopDataset(cfg, True)
    dataloader = torch.utils.data.DataLoader(dataset, pin_memory=True, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    
    # optimize
    logger.info('start loop-tuning')
    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epoch, eta_min=cfg.lr/10.0)
    scheduler_warmup = GradualWarmupScheduler(opt, multiplier=1, total_epoch=10, after_scheduler=scheduler)
    if cfg.embedding_net.classification:
        criterion = nn.CrossEntropyLoss(reduction="sum")
    else:
        criterion = nn.BCEWithLogitsLoss(reduction="sum")
    for epoch in range(cfg.max_epoch):
        if epoch == 0:
            opt.zero_grad()
            opt.step()
            scheduler_warmup.step()
        
        loss_meter = AverageMeter()

        # train
        model.train()
        logger.info("epoch: " + str(epoch) + " lr: " + str(scheduler_warmup.get_last_lr()[0]))
        with tqdm(dataloader) as t:
            for pcs, normals, grasps, joints, affordances, grasp_num in t:
                pcs, normals, grasps, joints, affordances = \
                    pcs.cuda(cfg.device), normals.cuda(cfg.device), grasps.cuda(cfg.device), joints.cuda(cfg.device), affordances.cuda(cfg.device)
                # (B, N, 3), (B, N, 3), (B, 100*J, 16), (B, 100*J, 8), (B, 100*J), (B,)
                
                opt.zero_grad()

                if cfg.point_encoder.normal_channel:
                    pcs = torch.cat([pcs, normals], dim=-1)     # (B, N, 3/6)
                if not cfg.joint_encoder.state_channel:
                    joints = joints[..., :-1]                   # (B, 100*J, 7/8)
                
                # data augmentation
                if cfg.augmentation:
                    pcs = pcs.cpu().numpy()
                    pcs = random_point_dropout(pcs, cfg.point_dropout_ratio, cfg.point_dropout_prob)
                    pcs[:, :, 0:3], point_scales = random_scale_point_cloud(pcs[:, :, 0:3], cfg.point_scale_low, cfg.point_scale_high)  # (B,)
                    pcs[:, :, 0:3], point_shifts = shift_point_cloud(pcs[:, :, 0:3], cfg.point_shift_range) # (B, 3)
                    pcs = torch.Tensor(pcs).cuda(cfg.device)
                    grasps = grasps.cpu().numpy()
                    grasps = random_scale_grasp(grasps, point_scales)
                    grasps = shift_grasp(grasps, point_shifts)
                    grasps = torch.Tensor(grasps).cuda(cfg.device)
                    joints = joints.cpu().numpy()
                    joints = random_scale_joint(joints, point_scales)
                    joints = shift_joint(joints, point_shifts)
                    joints = torch.Tensor(joints).cuda(cfg.device)
                
                prediction = model(pcs, grasps, joints)         # (B, 100*J, c)
                if not cfg.embedding_net.classification:
                    prediction = prediction.squeeze(-1)         # (B, 100*J(, c))

                loss = 0
                for b in range(grasp_num.shape[0]):
                    loss += criterion(prediction[b, :int(grasp_num[b])], affordances[b, :int(grasp_num[b])])
                loss /= int(torch.sum(grasp_num))
                
                loss.backward(retain_graph=False)
                opt.step()
                loss_meter.update(loss.item())
                t.set_postfix(epoch=epoch, loss=loss_meter.avg)
            scheduler_warmup.step()
            logger.info("training loss: " + str(loss_meter.avg))
            
            # validation
            model.eval()

            test_fns = sorted(list(itertools.chain(*[list(Path(cfg.dataset.test_path).glob('{}/*/*.npz'.format(instance))) for instance in cfg.dataset.test_instances])))
            idx = np.random.randint(len(test_fns))

            data = np.load(test_fns[idx], allow_pickle=True)
            pc = data['point_cloud']
            grasp = data['grasp']
            assert data['joint_pose'].shape[0] == cfg.joints
            joint_pose = data['joint_pose']
            config_path = os.path.join(os.path.dirname(test_fns[idx]), 'config.json')
            with open(config_path, 'r') as f:
                config = json.load(f)
            joint_axis_which = config["link_axis"]
            joint_type = config["link_type"]
            joint_state = config["link_state"]

            joint_feat = []
            for j in range(cfg.joints):
                translation = joint_pose[j, :3, -1]
                if joint_axis_which[j] == 'x':
                    rotation = joint_pose[j, :3, 0]
                elif joint_axis_which[j] == 'y':
                    rotation = joint_pose[j, :3, 1]
                elif joint_axis_which[j] == 'z':
                    rotation = joint_pose[j, :3, 2]
                else:
                    raise ValueError('Invalid joint_axis_which: {}'.format(joint_axis_which[j]))
                assert joint_type[j] == cfg.types[j]
                if joint_type[j] == 'revolute':
                    type_feat = 0
                    state_feat = joint_state[j] - cfg.states[j] / 180.0 * np.pi / 2
                elif joint_type[j] == 'prismatic':
                    type_feat = 1
                    state_feat = joint_state[j] - cfg.states[j] / 100.0 / 2
                else:
                    raise ValueError('Invalid joint_type: {}'.format(joint_type[j]))
                joint_feat.append(np.concatenate([translation, rotation, [type_feat, state_feat]]))
            joint_feat = np.stack(joint_feat, axis=0)                       # (J, 8)
            if not cfg.joint_encoder.state_channel:
                joint_feat = joint_feat[..., :-1]                           # (J, 7/8)
            
            contained_grasp_items = []
            for level in cfg.embedding_net.levels:
                contained_grasp_items.extend(level[0])
            grasp = grasp[np.isin(grasp[:, 7].astype(np.int32), contained_grasp_items)]     # (G, 8)
            
            grasp_feat = np.concatenate([grasp[:, 0:4], np.stack(grasp[:, 4], axis=0), 
                                         np.stack(grasp[:, 5], axis=0).reshape((-1, 9))], axis=-1)  # (G, 16)
            grasp_joint = grasp[:, 6].astype(np.int32)                      # (G,)
            grasp_affordance_ = grasp[:, 7].astype(np.int32)
            if cfg.embedding_net.classification:
                grasp_affordance = np.zeros_like(grasp_affordance_, dtype=np.int32)         # (G,)
                for level_idx in range(len(cfg.embedding_net.levels)):
                    for level_item in cfg.embedding_net.levels[level_idx][0]:
                        grasp_affordance[grasp_affordance_ == level_item] = level_idx
            else:
                grasp_affordance = np.zeros_like(grasp_affordance_, dtype=np.float32)
                for level_idx in range(len(cfg.embedding_net.levels)):
                    for level_item in cfg.embedding_net.levels[level_idx][0]:
                        grasp_affordance[grasp_affordance_ == level_item] = cfg.embedding_net.levels[level_idx][1]
            grasp_affordance_affordable = np.zeros_like(grasp_affordance_, dtype=bool)      # (G,)
            grasp_affordance_affordable[grasp_affordance_ == 1] = True
            grasp_affordance_affordable[grasp_affordance_ == 2] = True

            prediction = inference(cfg.embedding_net.classification, pc, cfg.point_encoder.normal_channel, joint_feat, grasp_feat, 
                                   cfg.shot.res, cfg.shot.receptive_field, cfg.test_samples, cfg.normalization, model, cfg.device, cfg.cache, test_fns[idx])   # (G, J(, c))
            if cfg.embedding_net.classification:
                prediction = np.argmax(prediction, axis=-1)                 # (G, J)
            pred_affordance = []
            for g in range(prediction.shape[0]):
                pred_affordance.append(prediction[g, grasp_joint[g]])
            pred_affordance = np.stack(pred_affordance, axis=0)             # (G,)

            if cfg.embedding_net.classification:
                hit = np.sum(pred_affordance == grasp_affordance)
                pred_affordance_affordable = np.zeros_like(pred_affordance, dtype=bool)
                for level_idx in range(len(cfg.embedding_net.levels)):
                    for level_item in cfg.embedding_net.levels[level_idx][0]:
                        if level_item == 1 or level_item == 2:
                            pred_affordance_affordable[pred_affordance == level_idx] = True
                true_affordable = np.sum(np.logical_and(pred_affordance_affordable, grasp_affordance_affordable))
                false_affordable = np.sum(np.logical_and(pred_affordance_affordable, np.logical_not(grasp_affordance_affordable)))
            else:
                hit = count_hit(pred_affordance, grasp_affordance, [level[2] for level in cfg.embedding_net.levels])
                true_affordable, false_affordable = count_affordable(pred_affordance, grasp_affordance_affordable, [level[2] if 1 in level[0] or 2 in level[0] else None for level in cfg.embedding_net.levels])
            acc = hit / grasp_affordance.shape[0]
            if true_affordable + false_affordable != 0:
                precision = true_affordable / (true_affordable + false_affordable)
            else:
                precision = 1.0
            
            logger.info("validation acc: " + str(acc) + " precision: " + str(precision))

            # import pdb; pdb.set_trace()
            torch.save(model.state_dict(), os.path.join(output_dir, 'model_latest.pth'))
    
    logger.info('done loop-tuning')


if __name__ == '__main__':
    main()
