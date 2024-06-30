from pathlib import Path
import omegaconf
import itertools
from tqdm import tqdm
import argparse
import os
import json
import numpy as np
import torch

from models.model import create_shot_encoder, create_encoder
from inference import inference

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, help='the path to the weight directory')
    args = parser.parse_args()
    cfg = omegaconf.OmegaConf.load(f"{args.weight_path}/.hydra/config.yaml")
    
    # load network
    shot_encoder = create_shot_encoder(cfg)
    encoder = create_encoder(cfg)
    shot_encoder.load_state_dict(torch.load(f'{args.weight_path}/shot_encoder_latest.pth'))
    encoder.load_state_dict(torch.load(f'{args.weight_path}/encoder_latest.pth'))
    num_params = sum(param.numel() for param in shot_encoder.parameters() if param.requires_grad) + \
                    sum(param.numel() for param in encoder.parameters() if param.requires_grad)
    print('num_params: {}'.format(num_params))
    shot_encoder.eval()
    encoder.eval()

    # load data
    # TODO: in looptune, you need to change `dataset_path` to `test_path`
    test_fns = sorted(list(itertools.chain(*[list(Path(cfg.dataset.dataset_path).glob('{}/*/*.npz'.format(instance))) for instance in cfg.dataset.test_instances])))
    
    tr_errs = []
    tr_along_errs = []
    tr_perp_errs = []
    tr_plane_errs = []
    rot_errs = []
    if cfg.encoder.state:
        state_errs = []
    for idx in tqdm(range(len(test_fns))):
        # load data
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

        # inference
        results = inference(pc, cfg.shot.res, cfg.shot.receptive_field, cfg.test_samples, cfg.test_sample_points, cfg.num_more, 
                            cfg.encoder.rot_num_bins, cfg.encoder.state, cfg.types, cfg.states, cfg.topk, shot_encoder, encoder, cfg.device, cfg.cache, test_fns[idx])
        if cfg.encoder.state:
            pred_translations, pred_directions, pred_states = results
        else:
            pred_translations, pred_directions = results
        
        tr_err_array = []
        tr_along_err_array = []
        tr_perp_err_array = []
        tr_plane_err_array = []
        rot_err_array = []
        if cfg.encoder.state:
            state_err_array = []
        for j in range(pred_translations.shape[0]):
            # load ground truth
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
            
            # compute error
            tr_err = np.linalg.norm(pred_translations[j] - gt_translation)
            tr_along_err = abs(np.dot(pred_translations[j] - gt_translation, gt_direction))
            tr_perp_err = np.sqrt(tr_err**2 - tr_along_err**2)
            tr_plane_err = calculate_plane_err(pred_translations[j], pred_directions[j], gt_translation, gt_direction)
            rot_err = np.arccos(np.dot(pred_directions[j], gt_direction))
            if cfg.encoder.state:
                state_err = abs(pred_states[j] - gt_state)
            tr_err_array.append(tr_err)
            tr_along_err_array.append(tr_along_err)
            tr_perp_err_array.append(tr_perp_err)
            tr_plane_err_array.append(tr_plane_err)
            rot_err_array.append(rot_err / np.pi * 180)
            if cfg.encoder.state:
                state_err_array.append(state_err)
        tr_errs.append(tr_err_array)
        tr_along_errs.append(tr_along_err_array)
        tr_perp_errs.append(tr_perp_err_array)
        tr_plane_errs.append(tr_plane_err_array)
        rot_errs.append(rot_err_array)
        if cfg.encoder.state:
            state_errs.append(state_err_array)
    
    tr_errs = np.stack(tr_errs)
    tr_along_errs = np.stack(tr_along_errs)
    tr_perp_errs = np.stack(tr_perp_errs)
    tr_plane_errs = np.stack(tr_plane_errs)
    rot_errs = np.stack(rot_errs)
    if cfg.encoder.state:
        state_errs = np.stack(state_errs)

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

        if cfg.encoder.state:
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
