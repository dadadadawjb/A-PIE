from typing import Tuple, List, Optional
import os
from pathlib import Path
import omegaconf
import time
import argparse
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import MinkowskiEngine as ME

from src_shot.build import shot
from utils import inplace_relu, pc_normalize, joint_denormalize, farthest_point_sample
from models.model import PointNet2


# real point cloud -> real estimated joint
def inference(pc:np.ndarray, has_normal:bool, state_channel:bool, joint_types:List[str], joint_states:List[float], 
              res:float, receptive_field:float, num_sample_points:int, 
              regressor:nn.Module, device:int, cache:bool, data_path:Optional[Path]) -> Tuple[np.ndarray, np.ndarray]:
    # sparse quantize
    if data_path is not None:
        possible_quantize_path = data_path.parent / f"{data_path.name.replace('.npz', '')}_quantize_{res}.npy"
        if os.path.exists(possible_quantize_path):
            pc = np.load(possible_quantize_path)
        else:
            indices = ME.utils.sparse_quantize(np.ascontiguousarray(pc), return_index=True, quantization_size=res)[1]
            pc = np.ascontiguousarray(pc[indices].astype(np.float32))
            if cache:
                np.save(possible_quantize_path, pc)
    else:
        indices = ME.utils.sparse_quantize(np.ascontiguousarray(pc), return_index=True, quantization_size=res)[1]
        pc = np.ascontiguousarray(pc[indices].astype(np.float32))   # (N', 3)

    # estimate normal
    if has_normal:
        if data_path is not None:
            possible_normal_path = data_path.parent / f"{data_path.name.replace('.npz', '')}_normal_{res * receptive_field}_{res}.npy"
            if os.path.exists(possible_normal_path):
                pc_normal = np.load(possible_normal_path)
            else:
                pc_normal = shot.estimate_normal(pc, res * receptive_field).reshape(-1, 3).astype(np.float32)
                pc_normal[~np.isfinite(pc_normal)] = 0
                if cache:
                    np.save(possible_normal_path, pc_normal)
        else:
            pc_normal = shot.estimate_normal(pc, res * receptive_field).reshape(-1, 3).astype(np.float32)
            pc_normal[~np.isfinite(pc_normal)] = 0                  # (N', 3)
        pc = np.concatenate([pc, pc_normal], axis=1)                # (N', 6)
    
    # sample points
    if data_path is not None:
        possible_sample_path = data_path.parent / f"{data_path.name.replace('.npz', '')}_sample_{num_sample_points}_{res}.npy"
        if os.path.exists(possible_sample_path):
            indices = np.load(possible_sample_path)
            pc = pc[indices]
        else:
            pc, indices = farthest_point_sample(pc, num_sample_points)
            if cache:
                np.save(possible_sample_path, indices)
    else:
        pc, indices = farthest_point_sample(pc, num_sample_points)  # (N, 3/6)
    
    # normalize
    pc[:, 0:3], offset, scale = pc_normalize(pc[:, 0:3])            # (3,), (1,)

    with torch.no_grad():
        pcs = torch.from_numpy(pc.astype(np.float32)).cuda(device)  # (N, 3/6)
        pcs = pcs.unsqueeze(0).transpose(2, 1)                      # (1, 3/6, N)

        if state_channel:
            preds_tr, preds_axis, preds_state, _trans_feat = regressor(pcs)
        else:
            preds_tr, preds_axis, _trans_feat = regressor(pcs)

        # denormalize
        J = len(joint_types)
        preds_tr = preds_tr.squeeze(0).reshape(J, 3).cpu().numpy()              # (J, 3)
        preds_axis = preds_axis.squeeze(0).reshape(J, 3).cpu().numpy()          # (J, 3)
        if state_channel:
            preds_state = preds_state.squeeze(0).cpu().numpy()                  # (J,)
        else:
            preds_state = None
        translation, direction, state = joint_denormalize(preds_tr, preds_axis, preds_state, offset, scale)
        if state_channel:
            states = []
            for j in range(J):
                if joint_types[j] == 'revolute':
                    s = state[j] * (joint_states[j] / 180.0 * np.pi)
                    s *= 180.0 / np.pi
                elif joint_types[j] == 'prismatic':
                    s = state[j] * (joint_states[j] / 100.0)
                    s *= 100.0
                else:
                    raise ValueError
                states.append(s)
            state = np.array(states)
        
    return (translation, direction, state) if state_channel else (translation, direction)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='the path to the point cloud data')
    parser.add_argument('--weight_path', type=str, help='the path to the weight directory')
    args = parser.parse_args()
    cfg = omegaconf.OmegaConf.load(f"{args.weight_path}/.hydra/config.yaml")

    # load network
    regressor = PointNet2(cfg).cuda(cfg.device)
    regressor.apply(inplace_relu)
    regressor.load_state_dict(torch.load(f'{args.weight_path}/pointnet2_latest.pth'))
    regressor.eval()

    # load data
    point_cloud = np.load(args.data_path)

    # inference
    start_time = time.time()
    results = inference(point_cloud, cfg.pointnet.normal_channel, cfg.pointnet.state, cfg.types, cfg.states, cfg.shot.res, cfg.shot.receptive_field, cfg.test_sample_points, regressor, cfg.device, False, None)
    if cfg.pointnet.state:
        translation, direction, state = results
    else:
        translation, direction = results
    end_time = time.time()
    print('estimated translation:', translation)
    print('estimated direction:', direction)
    if cfg.pointnet.state:
        print('estimated state:', state)
    print('estimation time:', end_time - start_time)

    # visualize
    geometries = []
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    geometries.append(frame)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    geometries.append(pcd)
    for j in range(translation.shape[0]):
        joint_axis = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.01, cone_radius=0.02, cylinder_height=0.4, cone_height=0.1, resolution=20, cylinder_split=4, cone_split=1)
        rotation = np.zeros((3, 3))
        temp2 = np.cross(direction[j], np.array([1., 0., 0.]))
        if np.linalg.norm(temp2) < 1e-6:
            temp1 = np.cross(np.array([0., 1., 0.]), direction[j])
            temp1 /= np.linalg.norm(temp1)
            temp2 = np.cross(direction[j], temp1)
            temp2 /= np.linalg.norm(temp2)
        else:
            temp2 /= np.linalg.norm(temp2)
            temp1 = np.cross(temp2, direction[j])
            temp1 /= np.linalg.norm(temp1)
        rotation[:, 0] = temp1
        rotation[:, 1] = temp2
        rotation[:, 2] = direction[j]
        joint_axis.rotate(rotation, np.array([[0], [0], [0]]))
        joint_axis.translate(translation[j].reshape((3, 1)))
        joint_axis.paint_uniform_color([204/255, 204/255, 204/255])
        geometries.append(joint_axis)
        joint_point = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
        joint_point = joint_point.translate(translation[j].reshape((3, 1)))
        joint_point.paint_uniform_color([179/255, 179/255, 179/255])
        geometries.append(joint_point)
    o3d.visualization.draw_geometries(geometries)
