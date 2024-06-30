import hydra
import logging
import os
from pathlib import Path
import json
import itertools
from itertools import combinations
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from warmup_scheduler import GradualWarmupScheduler

from dataset import LoopDataset
from utils import real2prob, AverageMeter
from models.model import create_shot_encoder, create_encoder
from inference import inference
from test import calculate_plane_err


# `config_looptune.yaml` should match with original `config.yaml`
@hydra.main(config_path='./', config_name='config_looptune', version_base='1.2')
def main(cfg):
    logger = logging.getLogger(__name__)
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']

    # load network
    logger.info('load network')
    shot_encoder = create_shot_encoder(cfg)
    encoder = create_encoder(cfg)
    shot_encoder.load_state_dict(torch.load(f'{cfg.weight_path}/shot_encoder_latest.pth'))
    encoder.load_state_dict(torch.load(f'{cfg.weight_path}/encoder_latest.pth'))

    # prepare data
    logger.info('prepare data')
    dataset = LoopDataset(cfg, True)
    dataloader = torch.utils.data.DataLoader(dataset, pin_memory=True, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    
    # optimize
    logger.info('start loop-tuning')
    opt = optim.Adam([*encoder.parameters(), *shot_encoder.parameters()], lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.step == 0:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epoch, eta_min=cfg.lr/100.0)
        if cfg.warmup:
            scheduler_warmup = GradualWarmupScheduler(opt, multiplier=1, total_epoch=5, after_scheduler=scheduler)
        else:
            scheduler_warmup = scheduler
    else:
        scheduler = optim.lr_scheduler.StepLR(opt, cfg.step, 0.5)
        if cfg.warmup:
            scheduler_warmup = GradualWarmupScheduler(opt, multiplier=1, total_epoch=5, after_scheduler=scheduler)
        else:
            scheduler_warmup = scheduler
    for epoch in range(cfg.max_epoch):
        if epoch == 0 and cfg.warmup:
            opt.zero_grad()
            opt.step()
            scheduler_warmup.step()
        
        loss_meter = AverageMeter()
        loss_tr_meter = AverageMeter()
        loss_axis_meter = AverageMeter()
        if cfg.encoder.state != 0:
            loss_state_meter = AverageMeter()
        loss_conf_meter = AverageMeter()

        # train
        shot_encoder.train()
        encoder.train()
        logger.info("epoch: " + str(epoch) + " lr: " + str(scheduler_warmup.get_last_lr()[0]))
        with tqdm(dataloader) as t:
            for pcs, pc_feats, normals, targets_tr, targets_rot, targets_state, point_idxs in t:
                pcs, pc_feats, normals, targets_tr, targets_rot, targets_state, point_idxs = \
                    pcs.cuda(cfg.device), pc_feats.cuda(cfg.device), normals.cuda(cfg.device), targets_tr.cuda(cfg.device), targets_rot.cuda(cfg.device), targets_state.cuda(cfg.device), point_idxs.cuda(cfg.device)
                # (B, N, 3), (B, N, 352), (B, N, 3), (B, J, sample_points, 2), (B, J, sample_points), (B, J) (B, sample_points, 2)
                
                opt.zero_grad()

                # sample more points for point tuples
                point_idx_more = torch.randint(0, pcs.shape[1], (point_idxs.shape[0], point_idxs.shape[1], cfg.num_more)).cuda(cfg.device)  # (B, sample_points, num_more)
                point_idx_all = torch.cat([point_idxs, point_idx_more], dim=-1)                                                             # (B, sample_points, 2 + num_more)
                
                # shot encoder for every point
                shot_feat = shot_encoder(pc_feats)       # (B, N, feature_dim)
                
                # encoder for sampled point tuples
                # shot_inputs = torch.cat([shot_feat[point_idx_all[:, i]] for i in range(0, point_idx_all.shape[-1])], -1)        # (sample_points, feature_dim * (2 + num_more))
                # normal_inputs = torch.cat([torch.max(torch.sum(normal[point_idx_all[:, i]] * normal[point_idx_all[:, j]], dim=-1, keepdim=True), 
                #                                      torch.sum(-normal[point_idx_all[:, i]] * normal[point_idx_all[:, j]], dim=-1, keepdim=True))
                #                                      for (i, j) in combinations(np.arange(point_idx_all.shape[-1]), 2)], -1)    # (sample_points, (2+num_more \choose 2))
                # coord_inputs = torch.cat([pc[point_idx_all[:, i]] - pc[point_idx_all[:, j]] for (i, j) in combinations(np.arange(point_idx_all.shape[-1]), 2)], -1) # (sample_points, 3 * (2+num_more \choose 2))
                # shot_inputs = []
                # normal_inputs = []
                # coord_inputs = []
                # for b in range(pcs.shape[0]):
                #     shot_inputs.append(torch.cat([shot_feat[b][point_idx_all[b, :, i]] for i in range(0, point_idx_all.shape[-1])], dim=-1))    # (sample_points, feature_dim * (2 + num_more))
                #     normal_inputs.append(torch.cat([torch.max(torch.sum(normals[b][point_idx_all[b, :, i]] * normals[b][point_idx_all[b, :, j]], dim=-1, keepdim=True), 
                #                                      torch.sum(-normals[b][point_idx_all[b, :, i]] * normals[b][point_idx_all[b, :, j]], dim=-1, keepdim=True))
                #                                      for (i, j) in combinations(np.arange(point_idx_all.shape[-1]), 2)], dim=-1))   # (sample_points, (2+num_more \choose 2))
                #     coord_inputs.append(torch.cat([pcs[b][point_idx_all[b, :, i]] - pcs[b][point_idx_all[b, :, j]] for (i, j) in combinations(np.arange(point_idx_all.shape[-1]), 2)], dim=-1)) # (sample_points, 3 * (2+num_more \choose 2))
                # shot_inputs = torch.stack(shot_inputs, dim=0)     # (B, sample_points, feature_dim * (2 + num_more))
                # normal_inputs = torch.stack(normal_inputs, dim=0) # (B, sample_points, (2+num_more \choose 2))
                # coord_inputs = torch.stack(coord_inputs, dim=0)   # (B, sample_points, 3 * (2+num_more \choose 2))
                shot_inputs = torch.cat([
                    torch.gather(shot_feat, 1, 
                                 point_idx_all[:, :, i:i+1].expand(
                                 (point_idx_all.shape[0], point_idx_all.shape[1], shot_feat.shape[-1]))) 
                    for i in range(point_idx_all.shape[-1])], dim=-1)   # (B, sample_points, feature_dim * (2 + num_more))
                normal_inputs = torch.cat([torch.max(
                    torch.sum(torch.gather(normals, 1, 
                                           point_idx_all[:, :, i:i+1].expand(
                                           (point_idx_all.shape[0], point_idx_all.shape[1], normals.shape[-1]))) * 
                              torch.gather(normals, 1, 
                                           point_idx_all[:, :, j:j+1].expand(
                                           (point_idx_all.shape[0], point_idx_all.shape[1], normals.shape[-1]))), 
                              dim=-1, keepdim=True), 
                    torch.sum(-torch.gather(normals, 1, 
                                           point_idx_all[:, :, i:i+1].expand(
                                           (point_idx_all.shape[0], point_idx_all.shape[1], normals.shape[-1]))) * 
                              torch.gather(normals, 1, 
                                           point_idx_all[:, :, j:j+1].expand(
                                           (point_idx_all.shape[0], point_idx_all.shape[1], normals.shape[-1]))), 
                              dim=-1, keepdim=True)) 
                    for (i, j) in combinations(np.arange(point_idx_all.shape[-1]), 2)], dim=-1)     # (B, sample_points, (2+num_more \choose 2))
                coord_inputs = torch.cat([
                    torch.gather(pcs, 1, 
                                 point_idx_all[:, :, i:i+1].expand(
                                 (point_idx_all.shape[0], point_idx_all.shape[1], pcs.shape[-1]))) - 
                    torch.gather(pcs, 1, 
                                 point_idx_all[:, :, j:j+1].expand(
                                 (point_idx_all.shape[0], point_idx_all.shape[1], pcs.shape[-1]))) 
                    for (i, j) in combinations(np.arange(point_idx_all.shape[-1]), 2)], dim=-1)     # (B, sample_points, 3 * (2+num_more \choose 2))
                inputs = torch.cat([coord_inputs, normal_inputs, shot_inputs], dim=-1)
                preds = encoder(inputs)                 # (B, sample_points, (2 + rot_num_bins + 1( + 1)) * J)

                num_top = int(cfg.topk * preds.shape[1])
                loss = 0
                # regression loss for translation for topk
                preds_tr = preds[:, :, 0:(2 * targets_tr.shape[1])]             # (B, sample_points, 2*J)
                preds_tr = preds_tr.reshape((preds_tr.shape[0], preds_tr.shape[1], targets_tr.shape[1], 2)) # (B, sample_points, J, 2)
                preds_tr = preds_tr.transpose(1, 2)                             # (B, J, sample_points, 2)
                loss_tr_ = torch.mean((preds_tr - targets_tr) ** 2, dim=-1)     # (B, J, sample_points)
                loss_tr, best_idx_tr = torch.topk(loss_tr_, num_top, dim=-1, largest=False, sorted=False)   # (B, J, num_top)
                loss_tr = torch.mean(loss_tr)
                loss += loss_tr
                loss_tr_meter.update(loss_tr.item())

                # classification loss for rotation for topk
                preds_axis = preds[:, :, (2 * targets_rot.shape[1]):(2 * targets_rot.shape[1] + cfg.encoder.rot_num_bins * targets_rot.shape[1])]   # (B, sample_points, rot_num_bins*J)
                preds_axis = preds_axis.reshape((preds_axis.shape[0], preds_axis.shape[1], targets_rot.shape[1], cfg.encoder.rot_num_bins))         # (B, sample_points, J, rot_num_bins)
                preds_axis = preds_axis.transpose(1, 2)             # (B, J, sample_points, rot_num_bins)
                preds_axis_ = F.log_softmax(preds_axis, dim=-1)     # (B, J, sample_points, rot_num_bins)
                targets_rot_ = real2prob(targets_rot, np.pi, cfg.encoder.rot_num_bins, circular=False)          # (B, J, sample_points, rot_num_bins)
                loss_axis_ = torch.sum(F.kl_div(preds_axis_, targets_rot_, reduction='none'), dim=-1)           # (B, J, sample_points)
                loss_axis, best_idx_axis = torch.topk(loss_axis_, num_top, dim=-1, largest=False, sorted=False) # (B, J, num_top)
                loss_axis = torch.mean(loss_axis)
                loss_axis *= cfg.lambda_axis
                loss += loss_axis
                loss_axis_meter.update(loss_axis.item())

                # regression loss for state
                if cfg.encoder.state:
                    preds_state = preds[:, :, -2*targets_state.shape[1]:-1*targets_state.shape[1]]  # (B, sample_points, J)
                    preds_state = preds_state.transpose(1, 2)                       # (B, J, sample_points)
                    if cfg.goodness == 'tr':
                        preds_state = torch.gather(preds_state, -1, best_idx_tr)    # (B, J, num_top)
                    elif cfg.goodness == 'axis':
                        preds_state = torch.gather(preds_state, -1, best_idx_axis)  # (B, J, num_top)
                    elif cfg.goodness == 'both':
                        preds_state = torch.cat([torch.gather(preds_state, -1, best_idx_tr),
                                                    torch.gather(preds_state, -1, best_idx_axis)], dim=-1)  # (B, J, 2 * num_top)
                    else:
                        raise ValueError('Unknown goodness type')
                    preds_state = torch.mean(preds_state, dim=-1)   # (B, J)
                    loss_state = F.mse_loss(preds_state, targets_state, reduction='mean')
                    loss_state *= cfg.lambda_state
                    loss += loss_state
                    loss_state_meter.update(loss_state.item())
                
                # classification loss for goodness
                preds_conf = preds[:, :, -1*targets_state.shape[1]:]        # (B, sample_points, J)
                preds_conf = preds_conf.transpose(1, 2)                     # (B, J, sample_points)
                target_conf = torch.zeros_like(preds_conf).float().cuda(cfg.device)
                if cfg.goodness == 'tr':
                    for b in range(target_conf.shape[0]):
                        for j in range(target_conf.shape[1]):
                            target_conf[b, j, best_idx_tr[b, j]] = 1.
                elif cfg.goodness == 'axis':
                    for b in range(target_conf.shape[0]):
                        for j in range(target_conf.shape[1]):
                            target_conf[b, j, best_idx_axis[b, j]] = 1.
                elif cfg.goodness == 'both':
                    for b in range(target_conf.shape[0]):
                        for j in range(target_conf.shape[1]):
                            target_conf[b, j, best_idx_tr[b, j]] = 1.
                            target_conf[b, j, best_idx_axis[b, j]] = 1.
                else:
                    raise ValueError('Unknown goodness type')
                loss_conf = F.binary_cross_entropy_with_logits(preds_conf, target_conf, reduction='none')   # (B, J, sample_points)
                loss_conf = torch.mean(loss_conf)
                loss_conf *= cfg.lambda_conf
                loss += loss_conf
                loss_conf_meter.update(loss_conf.item())
                
                loss.backward(retain_graph=False)
                # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.)
                # torch.nn.utils.clip_grad_norm_(shot_encoder.parameters(), 1.)
                opt.step()
                
                loss_meter.update(loss.item())
                
                if cfg.encoder.state:
                    t.set_postfix(epoch=epoch, loss=loss_meter.avg, tr=loss_tr_meter.avg, axis=loss_axis_meter.avg, state=loss_state_meter.avg, conf=loss_conf_meter.avg)
                else:
                    t.set_postfix(epoch=epoch, loss=loss_meter.avg, tr=loss_tr_meter.avg, axis=loss_axis_meter.avg, conf=loss_conf_meter.avg)
            scheduler_warmup.step()
            if cfg.encoder.state:
                logger.info("training loss: " + str(loss_tr_meter.avg) + " + " + str(loss_axis_meter.avg) + " + " + 
                            str(loss_state_meter.avg) + " + " + str(loss_conf_meter.avg) + " = " + str(loss_meter.avg))
            else:
                logger.info("training loss: " + str(loss_tr_meter.avg) + " + " + str(loss_axis_meter.avg) + " + " + 
                            str(loss_conf_meter.avg) + " = " + str(loss_meter.avg))
            
            # validation
            shot_encoder.eval()
            encoder.eval()

            test_fns = sorted(list(itertools.chain(*[list(Path(cfg.dataset.test_path).glob('{}/*/*.npz'.format(instance))) for instance in cfg.dataset.test_instances])))
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

            results = inference(pc, cfg.shot.res, cfg.shot.receptive_field, cfg.test_samples, cfg.test_sample_points, cfg.num_more, 
                                cfg.encoder.rot_num_bins, cfg.encoder.state, cfg.types, cfg.states, cfg.topk, shot_encoder, encoder, cfg.device, cfg.cache, test_fns[idx])
            if cfg.encoder.state:
                pred_translations, pred_directions, pred_states = results
            else:
                pred_translations, pred_directions = results
            
            for j in range(pred_translations.shape[0]):
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
                tr_err = np.linalg.norm(pred_translations[j] - gt_translation)
                tr_along_err = abs(np.dot(pred_translations[j] - gt_translation, gt_direction))
                tr_perp_err = np.sqrt(tr_err**2 - tr_along_err**2)
                tr_plane_err = calculate_plane_err(pred_translations[j], pred_directions[j], gt_translation, gt_direction)
                rot_err = np.arccos(np.dot(pred_directions[j], gt_direction))
                if cfg.encoder.state:
                    state_err = abs(pred_states[j] - gt_state)
                if cfg.encoder.state:
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
            torch.save(encoder.state_dict(), os.path.join(output_dir, 'encoder_latest.pth'))
            torch.save(shot_encoder.state_dict(), os.path.join(output_dir, 'shot_encoder_latest.pth'))
    
    logger.info('done loop-tuning')


if __name__ == '__main__':
    main()
