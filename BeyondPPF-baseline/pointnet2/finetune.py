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

from dataset import ArticulationDataset
from utils import inplace_relu, AverageMeter, \
    random_point_dropout, random_scale_point_cloud, random_scale_joint, shift_point_cloud, shift_joint
from models.model import PointNet2
from inference import inference
from test import calculate_plane_err


# `config_finetune.yaml` should match with original `config.yaml`
@hydra.main(config_path='./', config_name='config_finetune', version_base='1.2')
def main(cfg):
    logger = logging.getLogger(__name__)
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']

    # load network
    logger.info('load network')
    regressor = PointNet2(cfg).cuda(cfg.device)
    regressor.apply(inplace_relu)
    regressor.load_state_dict(torch.load(f'{cfg.weight_path}/pointnet2_latest.pth'))

    # prepare data
    logger.info('prepare data')
    dataset = ArticulationDataset(cfg, True)
    dataloader = torch.utils.data.DataLoader(dataset, pin_memory=True, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    
    # optimize
    logger.info('start fine-tuning')
    tr_criterion = nn.MSELoss()
    axis_criterion = nn.MSELoss()
    if cfg.pointnet.state:
        state_criterion = nn.MSELoss()
    opt = optim.Adam(regressor.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epoch, eta_min=cfg.lr/100.0)
    for epoch in range(cfg.max_epoch):
        loss_meter = AverageMeter()
        loss_tr_meter = AverageMeter()
        loss_axis_meter = AverageMeter()
        if cfg.pointnet.state:
            loss_state_meter = AverageMeter()

        # train
        regressor.train()
        logger.info("epoch: " + str(epoch) + " lr: " + str(scheduler.get_last_lr()[0]))
        with tqdm(dataloader) as t:
            for pcs, translations, rotations, states in t:
                opt.zero_grad()

                # data augmentation
                pcs = pcs.cpu().numpy()
                pcs = random_point_dropout(pcs, cfg.point_dropout_ratio, cfg.point_dropout_prob)
                pcs[:, :, 0:3], point_scales = random_scale_point_cloud(pcs[:, :, 0:3], cfg.point_scale_low, cfg.point_scale_high)  # (B,)
                pcs[:, :, 0:3], point_shifts = shift_point_cloud(pcs[:, :, 0:3], cfg.point_shift_range) # (B, 3)
                pcs = torch.Tensor(pcs)
                pcs = pcs.transpose(2, 1)
                pcs = pcs.cuda(cfg.device)      # (B, 3/6, N)
                translations = translations.cpu().numpy()
                rotations = rotations.cpu().numpy()
                states = states.cpu().numpy()
                translations, rotations, states = random_scale_joint(translations, rotations, states, point_scales)
                translations, rotations, states = shift_joint(translations, rotations, states, point_shifts)
                translations = torch.Tensor(translations)
                rotations = torch.Tensor(rotations)
                states = torch.Tensor(states)
                translations = translations.cuda(cfg.device)    # (B, J, 3)
                rotations = rotations.cuda(cfg.device)          # (B, J, 3)
                states = states.cuda(cfg.device)                # (B, J)

                if cfg.pointnet.state:
                    preds_tr, preds_axis, preds_state, _trans_feat = regressor(pcs)
                else:
                    preds_tr, preds_axis, _trans_feat = regressor(pcs)

                loss = 0
                # regression loss for translation
                preds_tr = preds_tr.view(-1, translations.shape[1], 3)
                loss_tr = tr_criterion(preds_tr, translations)
                loss += loss_tr
                loss_tr_meter.update(loss_tr.item())

                # regression loss for rotation
                preds_axis = preds_axis.view(-1, rotations.shape[1], 3)
                loss_axis = axis_criterion(preds_axis, rotations)
                loss_axis *= cfg.lambda_axis
                loss += loss_axis
                loss_axis_meter.update(loss_axis.item())

                # regression loss for state
                if cfg.pointnet.state:
                    loss_state = state_criterion(preds_state, states)
                    loss_state *= cfg.lambda_state
                    loss += loss_state
                    loss_state_meter.update(loss_state.item())
                
                loss.backward(retain_graph=False)
                opt.step()
                
                loss_meter.update(loss.item())
                
                if cfg.pointnet.state:
                    t.set_postfix(epoch=epoch, loss=loss_meter.avg, tr=loss_tr_meter.avg, axis=loss_axis_meter.avg, state=loss_state_meter.avg)
                else:
                    t.set_postfix(epoch=epoch, loss=loss_meter.avg, tr=loss_tr_meter.avg, axis=loss_axis_meter.avg)
            scheduler.step()
            if cfg.pointnet.state:
                logger.info("training loss: " + str(loss_tr_meter.avg) + " + " + str(loss_axis_meter.avg) + " + " + str(loss_state_meter.avg) + " = " + str(loss_meter.avg))
            else:
                logger.info("training loss: " + str(loss_tr_meter.avg) + " + " + str(loss_axis_meter.avg) + " = " + str(loss_meter.avg))
            
            # validation
            regressor.eval()

            test_fns = sorted(list(itertools.chain(*[list(Path(cfg.dataset.dataset_path).glob('{}/*/*.npz'.format(instance))) for instance in cfg.dataset.test_instances])))
            idx = np.random.randint(len(test_fns))

            data = np.load(test_fns[idx])
            pc = data['point_cloud']
            assert data['joint_pose'].shape[0] == cfg.joints
            joint_pose = data['joint_pose']
            config_path = os.path.join(os.path.dirname(test_fns[idx]), 'config.json')
            with open(config_path, 'r') as f:
                config = json.load(f)
            joint_axis_which = config["link_axis"]
            joint_type = config["link_type"]
            joint_state = config["link_state"]

            results = inference(pc, cfg.pointnet.normal_channel, cfg.pointnet.state, cfg.types, cfg.states, cfg.shot.res, cfg.shot.receptive_field, cfg.test_sample_points, regressor, cfg.device, cfg.cache, test_fns[idx])
            if cfg.pointnet.state:
                pred_translation, pred_direction, pred_state = results
            else:
                pred_translation, pred_direction = results
            
            for j in range(pred_translation.shape[0]):
                gt_translation = joint_pose[j, :3, -1]
                if joint_axis_which[j] == 'x':
                    gt_direction = joint_pose[j, :3, 0]
                elif joint_axis_which[j] == 'y':
                    gt_direction = joint_pose[j, :3, 1]
                elif joint_axis_which[j] == 'z':
                    gt_direction = joint_pose[j, :3, 2]
                else:
                    raise ValueError('Invalid joint_axis_which: {}'.format(joint_axis_which[j]))
                assert joint_type[j] == cfg.types[j]
                if joint_type[j] == 'revolute':
                    gt_state = joint_state[j] * 180.0 / np.pi
                elif joint_type[j] == 'prismatic':
                    gt_state = joint_state[j] * 100.0
                else:
                    raise ValueError('Invalid joint_type: {}'.format(joint_type[j]))
                tr_err = np.linalg.norm(pred_translation[j] - gt_translation)
                tr_along_err = abs(np.dot(pred_translation[j] - gt_translation, gt_direction))
                tr_perp_err = np.sqrt(tr_err**2 - tr_along_err**2)
                tr_plane_err = calculate_plane_err(pred_translation[j], pred_direction[j], gt_translation, gt_direction)
                rot_err = np.arccos(np.dot(pred_direction[j], gt_direction))
                if cfg.pointnet.state:
                    state_err = abs(pred_state[j] - gt_state)
                if cfg.pointnet.state:
                    if joint_type[j] == 'revolute':
                        logger.info("validation error: " + str(tr_err*100) + "cm " + str(tr_along_err*100) + "cm " 
                                + str(tr_perp_err*100) + "cm " + str(tr_plane_err*100) + "cm " + str(rot_err/np.pi*180) + "deg " + str(state_err) + "deg")
                    elif joint_type[j] == 'prismatic':
                        logger.info("validation error: " + str(tr_err*100) + "cm " + str(tr_along_err*100) + "cm " 
                                + str(tr_perp_err*100) + "cm " + str(tr_plane_err*100) + "cm " + str(rot_err/np.pi*180) + "deg " + str(state_err) + "cm")
                    else:
                        raise ValueError('Invalid joint_type: {}'.format(joint_type[j]))
                else:
                    logger.info("validation error: " + str(tr_err*100) + "cm " + str(tr_along_err*100) + "cm " 
                            + str(tr_perp_err*100) + "cm " + str(tr_plane_err*100) + "cm " + str(rot_err/np.pi*180) + "deg")

            # import pdb; pdb.set_trace()
            torch.save(regressor.state_dict(), os.path.join(output_dir, 'pointnet2_latest.pth'))
    
    logger.info('done fine-tuning')


if __name__ == '__main__':
    main()
