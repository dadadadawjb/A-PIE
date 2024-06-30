import omegaconf
from tqdm import tqdm
import argparse
import numpy as np
import torch

from dataset import ArticulationDataset
from utils import inplace_relu, joint_denormalize_batch
from models.model import PointNet2

def calculate_plane_err(pred_translation:np.ndarray, pred_direction:np.ndarray, 
                        gt_translation:np.ndarray, gt_direction:np.ndarray) -> float:
    if abs(np.dot(pred_direction, gt_direction)) < 1e-3:
        # parallel to the plane
        # point-to-line distance
        dist = np.linalg.norm(np.cross(pred_direction, gt_translation - pred_translation))
        return dist
    # gt_direction \dot (x - gt_translation) = 0
    # x = pred_translation + t * pred_direction
    t = np.dot(gt_translation - pred_translation, gt_direction) / np.dot(pred_direction, gt_direction)
    x = pred_translation + t * pred_direction
    dist = np.linalg.norm(x - gt_translation)
    return dist

def calculate_plane_err_batch(pred_translations:torch.Tensor, pred_directions:torch.Tensor, 
                              gt_translations:torch.Tensor, gt_directions:torch.Tensor) -> torch.Tensor:
    dist = torch.zeros(*pred_translations.shape[:-1], device=pred_translations.device)
    flag = torch.abs(torch.sum(pred_directions * gt_directions, dim=-1)) < 1e-3
    not_flag = torch.logical_not(flag)

    dist[flag] = torch.linalg.norm(torch.linalg.cross(pred_translations[flag], gt_translations[flag] - pred_translations[flag]), dim=-1)

    t = torch.sum((gt_translations[not_flag] - pred_translations[not_flag]) * gt_directions[not_flag], dim=-1) / torch.sum(pred_directions[not_flag] * gt_directions[not_flag], dim=-1)
    x = pred_translations[not_flag] + t.unsqueeze(-1) * pred_directions[not_flag]
    dist[not_flag] = torch.linalg.norm(x - gt_translations[not_flag], dim=-1)

    return dist


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, help='the path to the weight directory')
    args = parser.parse_args()
    cfg = omegaconf.OmegaConf.load(f"{args.weight_path}/.hydra/config.yaml")
    
    # load network
    regressor = PointNet2(cfg).cuda(cfg.device)
    regressor.apply(inplace_relu)
    regressor.load_state_dict(torch.load(f'{args.weight_path}/pointnet2_latest.pth'))
    num_params = sum(param.numel() for param in regressor.parameters() if param.requires_grad)
    print('num_params: {}'.format(num_params))
    regressor.eval()

    # load data
    dataset = ArticulationDataset(cfg, False)
    dataloader = torch.utils.data.DataLoader(dataset, pin_memory=True, batch_size=cfg.batch_size, shuffle=False, num_workers=10)
    
    # test
    tr_errs = []
    tr_along_errs = []
    tr_perp_errs = []
    tr_plane_errs = []
    rot_errs = []
    if cfg.pointnet.state:
        state_errs = []
    with tqdm(dataloader) as t:
        for pcs, real_translations, real_rotations, real_states, offsets, scales in t:
            pcs, real_translations, real_rotations, real_states, offsets, scales = \
                pcs.cuda(cfg.device), real_translations.cuda(cfg.device), real_rotations.cuda(cfg.device), real_states.cuda(cfg.device), offsets.cuda(cfg.device), scales.cuda(cfg.device)
            # (B, N, 3/6), (B, J, 3), (B, J, 3), (B, J), (B, 3), (B, 1)

            pcs = pcs.transpose(2, 1)                               # (B, 3/6, N)

            with torch.no_grad():
                if cfg.pointnet.state:
                    preds_tr, preds_axis, preds_state, _trans_feat = regressor(pcs) # (B, 3*J), (B, 3*J), (B, 1*J)
                    preds_tr = preds_tr.view(-1, real_translations.shape[1], 3)     # (B, J, 3)
                    preds_axis = preds_axis.view(-1, real_rotations.shape[1], 3)    # (B, J, 3)
                else:
                    preds_tr, preds_axis, _trans_feat = regressor(pcs)
                    preds_tr = preds_tr.view(-1, real_translations.shape[1], 3)
                    preds_axis = preds_axis.view(-1, real_rotations.shape[1], 3)
                    preds_state = None

            # denormalize
            translation, direction, state = joint_denormalize_batch(preds_tr, preds_axis, preds_state, offsets, scales)     # (B, J, 3), (B, J, 3), (B, J)

            # compute error
            tr_err = torch.norm(translation - real_translations, dim=-1).cpu().numpy()                                          # (B, J)
            tr_along_err = torch.abs(torch.sum((translation - real_translations) * real_rotations, dim=-1)).cpu().numpy()       # (B, J)
            tr_perp_err = np.sqrt(tr_err * tr_err - tr_along_err * tr_along_err)                                                # (B, J)
            tr_plane_err = calculate_plane_err_batch(translation, direction, real_translations, real_rotations).cpu().numpy()   # (B, J)
            tr_errs.append(tr_err)
            tr_along_errs.append(tr_along_err)
            tr_perp_errs.append(tr_perp_err)
            tr_plane_errs.append(tr_plane_err)
            rot_errs.append(torch.acos(torch.sum(direction * real_rotations, dim=-1)).cpu().numpy() / np.pi * 180)          # (B, J)
            if cfg.pointnet.state:
                state_err_array = []
                for j in range(real_translations.shape[1]):
                    if cfg.types[j] == 'revolute':
                        state_ = state[:, j] * (cfg.states[j] / 180.0 * np.pi)
                        state_ *= 180.0 / np.pi
                        gt_state = real_states[:, j] * (cfg.states[j] / 180.0 * np.pi)
                        gt_state *= 180.0 / np.pi
                    elif cfg.types[j] == 'prismatic':
                        state_ = state[:, j] * (cfg.states[j] / 100.0)
                        state_ *= 100.0
                        gt_state = real_states[:, j] * (cfg.states[j] / 100.0)
                        gt_state *= 100.0
                    else:
                        raise ValueError('Invalid joint_type: {}'.format(cfg.types[j]))
                    state_err = torch.abs(state_ - gt_state).cpu().numpy()  # (B,)
                    state_err_array.append(state_err)
                state_errs.append(np.stack(state_err_array, axis=-1))       # (B, J)
    
    tr_errs = np.concatenate(tr_errs, axis=0)
    tr_along_errs = np.concatenate(tr_along_errs, axis=0)
    tr_perp_errs = np.concatenate(tr_perp_errs, axis=0)
    tr_plane_errs = np.concatenate(tr_plane_errs, axis=0)
    rot_errs = np.concatenate(rot_errs, axis=0)
    if cfg.pointnet.state:
        state_errs = np.concatenate(state_errs, axis=0)

    for j in range(cfg.joints):
        print('avg:', np.mean(tr_errs[:, j])*100, 'cm')
        print('1cm:', np.mean(tr_errs[:, j] < 1e-2))
        print('2cm:', np.mean(tr_errs[:, j] < 2e-2))
        print('5cm:', np.mean(tr_errs[:, j] < 5e-2))
        print('10cm:', np.mean(tr_errs[:, j] < 0.1))
        print('15cm:', np.mean(tr_errs[:, j] < 0.15))
        print('avg along:', np.mean(tr_along_errs[:, j])*100, 'cm')
        print('1cm along:', np.mean(tr_along_errs[:, j] < 1e-2))
        print('2cm along:', np.mean(tr_along_errs[:, j] < 2e-2))
        print('5cm along:', np.mean(tr_along_errs[:, j] < 5e-2))
        print('10cm along:', np.mean(tr_along_errs[:, j] < 0.1))
        print('15cm along:', np.mean(tr_along_errs[:, j] < 0.15))
        print('avg perp:', np.mean(tr_perp_errs[:, j])*100, 'cm')
        print('1cm perp:', np.mean(tr_perp_errs[:, j] < 1e-2))
        print('2cm perp:', np.mean(tr_perp_errs[:, j] < 2e-2))
        print('5cm perp:', np.mean(tr_perp_errs[:, j] < 5e-2))
        print('10cm perp:', np.mean(tr_perp_errs[:, j] < 0.1))
        print('15cm perp:', np.mean(tr_perp_errs[:, j] < 0.15))
        print('avg plane:', np.mean(tr_plane_errs[:, j])*100, 'cm')
        print('1cm plane:', np.mean(tr_plane_errs[:, j] < 1e-2))
        print('2cm plane:', np.mean(tr_plane_errs[:, j] < 2e-2))
        print('5cm plane:', np.mean(tr_plane_errs[:, j] < 5e-2))
        print('10cm plane:', np.mean(tr_plane_errs[:, j] < 0.1))
        print('15cm plane:', np.mean(tr_plane_errs[:, j] < 0.15))

        print('avg:', np.mean(rot_errs[:, j]), 'deg')
        print('1deg:', np.mean(rot_errs[:, j] < 1))
        print('2deg:', np.mean(rot_errs[:, j] < 2))
        print('5deg:', np.mean(rot_errs[:, j] < 5))
        print('10deg:', np.mean(rot_errs[:, j] < 10))
        print('15deg:', np.mean(rot_errs[:, j] < 15))

        if cfg.pointnet.state:
            if cfg.types[j] == 'revolute':
                print('avg:', np.mean(state_errs[:, j]), 'deg')
                print('1deg:', np.mean(state_errs[:, j] < 1))
                print('2deg:', np.mean(state_errs[:, j] < 2))
                print('5deg:', np.mean(state_errs[:, j] < 5))
                print('10deg:', np.mean(state_errs[:, j] < 10))
                print('15deg:', np.mean(state_errs[:, j] < 15))
            elif cfg.types[j] == 'prismatic':
                print('avg:', np.mean(state_errs[:, j]), 'cm')
                print('1cm:', np.mean(state_errs[:, j] < 1))
                print('2cm:', np.mean(state_errs[:, j] < 2))
                print('5cm:', np.mean(state_errs[:, j] < 5))
                print('10cm:', np.mean(state_errs[:, j] < 10))
                print('15cm:', np.mean(state_errs[:, j] < 15))
            else:
                raise ValueError('Invalid joint_type: {}'.format(cfg.types[j]))
