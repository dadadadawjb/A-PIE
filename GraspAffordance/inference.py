from typing import Optional
import os
import json
from pathlib import Path
import omegaconf
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
import open3d as o3d

from src_shot.build import shot
from utils import farthest_point_sample, pc_normalize, joint_normalize, grasp_normalize
from models.model import grasp_embedding_network


def inference(classification:bool, pc:np.ndarray, has_normal:bool, joints:np.ndarray, grasps:np.ndarray, 
              res:float, receptive_field:float, num_sample_points:int, normalization:bool, 
              model:nn.Module, device:int, cache:bool, data_path:Optional[Path]) -> np.ndarray:
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
    if normalization:
        pc[:, 0:3], offset, scale = pc_normalize(pc[:, 0:3])                        # (3,), (1,)
        joints = joint_normalize(joints, offset, scale)                             # (J, 7/8)
        grasps = grasp_normalize(grasps, offset, scale)                             # (G, 16)

    with torch.no_grad():
        J = joints.shape[0]
        G = grasps.shape[0]
        pcs = torch.from_numpy(pc.astype(np.float32)).cuda(device)                  # (N, 3/6)
        joints = torch.from_numpy(joints.astype(np.float32)).cuda(device)           # (J, 7/8)
        grasps = torch.from_numpy(grasps.astype(np.float32)).cuda(device)           # (G, 16)
        pcs = pcs.unsqueeze(0)                                                      # (1, N, 3/6)
        grasps = grasps[:, None, :].repeat(1, J, 1)                                 # (G, J, 16)
        grasps = grasps.reshape(-1, grasps.shape[-1])                               # (G*J, 16)
        grasps = grasps.unsqueeze(0)                                                # (1, G*J, 16)
        joints = joints[None, :, :].expand(G, -1, -1)                               # (G, J, 7/8)
        joints = joints.reshape(-1, joints.shape[-1])                               # (G*J, 7/8)
        joints = joints.unsqueeze(0)                                                # (1, G*J, 7/8)

        prediction = model(pcs, grasps, joints)         # (1, G*J, c)
        if classification:
            prediction = F.softmax(prediction, dim=-1)
            prediction = prediction.squeeze(0).reshape((G, J, -1))          # (G, J, c)
        else:
            prediction = torch.sigmoid(prediction)
            prediction = prediction.squeeze(0).squeeze(-1).reshape((G, J))  # (G, J)

        prediction = prediction.cpu().numpy()
    
    return prediction


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='the path to the point cloud data')
    parser.add_argument('--joints_path', type=str, help='the path to the joints data')
    parser.add_argument('--grasps_path', type=str, help='the path to the grasps data')
    parser.add_argument('--weight_path', type=str, help='the path to the weight directory')
    parser.add_argument('--dataset', action='store_true', help='whether the data is from the dataset')
    args = parser.parse_args()
    cfg = omegaconf.OmegaConf.load(f"{args.weight_path}/.hydra/config.yaml")

    # load network
    model = grasp_embedding_network(cfg).to(cfg.device)
    model.load_state_dict(torch.load(f'{args.weight_path}/model_latest.pth'))
    model.eval()

    # load data
    if args.dataset:
        data = np.load(args.data_path, allow_pickle=True)
        point_cloud = data['point_cloud']
        joint_pose = data['joint_pose']
        config_path = os.path.join(os.path.dirname(args.data_path), 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        joint_axis_which = config["link_axis"]
        joint_type = config["link_type"]
        joint_state = config["link_state"]
        joint_feat_ = []
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
            joint_feat_.append(np.concatenate([translation, rotation, [type_feat, state_feat]]))
        joints = np.stack(joint_feat_, axis=0)                      # (J, 8)
        if not cfg.joint_encoder.state_channel:
            joints = joints[:, :-1]                                 # (J, 7/8)
        grasp = data['grasp']
        contained_grasp_items = []
        for level in cfg.embedding_net.levels:
            contained_grasp_items.extend(level[0])
        grasp = grasp[np.isin(grasp[:, 7].astype(np.int32), contained_grasp_items)]         # (G, 8)
        grasps = np.concatenate([grasp[:, 0:4], np.stack(grasp[:, 4], axis=0), 
                                     np.stack(grasp[:, 5], axis=0).reshape((-1, 9))], axis=-1)  # (G, 16)
    else:
        point_cloud = np.load(args.data_path)   # (N, 3)
        joints = np.load(args.joints_path)      # (J, 7/8)
        grasps = np.load(args.grasps_path)      # (G, 16)

    # inference
    start_time = time.time()
    affordances = inference(cfg.embedding_net.classification, np.copy(point_cloud), cfg.point_encoder.normal_channel, np.copy(joints), np.copy(grasps), 
                            cfg.shot.res, cfg.shot.receptive_field, cfg.test_samples, cfg.normalization, model, cfg.device, False, None)
    if cfg.embedding_net.classification:
        affordances_ = np.zeros(affordances.shape[:-1], dtype=np.float32)
        for c, level in enumerate(cfg.embedding_net.levels):
            affordances_ += affordances[..., c] * level[1]
        affordances = affordances_
    end_time = time.time()
    print('estimated affordances:', affordances)
    print('estimation time:', end_time - start_time)

    # visualize
    for j in range(joints.shape[0]):
        geometries = []

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        geometries.append(frame)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        pcd.paint_uniform_color([126/255, 208/255, 248/255])
        geometries.append(pcd)

        joint_axis = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.01, cone_radius=0.02, cylinder_height=0.4, cone_height=0.1, resolution=20, cylinder_split=4, cone_split=1)
        rotation = np.zeros((3, 3))
        temp2 = np.cross(joints[j, 3:6], np.array([1., 0., 0.]))
        if np.linalg.norm(temp2) < 1e-6:
            temp1 = np.cross(np.array([0., 1., 0.]), joints[j, 3:6])
            temp1 /= np.linalg.norm(temp1)
            temp2 = np.cross(joints[j, 3:6], temp1)
            temp2 /= np.linalg.norm(temp2)
        else:
            temp2 /= np.linalg.norm(temp2)
            temp1 = np.cross(temp2, joints[j, 3:6])
            temp1 /= np.linalg.norm(temp1)
        rotation[:, 0] = temp1
        rotation[:, 1] = temp2
        rotation[:, 2] = joints[j, 3:6]
        joint_axis.rotate(rotation, np.array([[0], [0], [0]]))
        joint_axis.translate(joints[j, 0:3].reshape((3, 1)))
        joint_axis.paint_uniform_color([255/255, 220/255, 126/255])
        geometries.append(joint_axis)
        joint_point = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
        joint_point = joint_point.translate(joints[j, 0:3].reshape((3, 1)))
        joint_point.paint_uniform_color([250/255, 170/255, 137/255])
        geometries.append(joint_point)

        finger_width = 0.004
        tail_length = 0.04
        depth_base = 0.02
        for i in range(grasps.shape[0]):
            gg = grasps[i]
            gg_affordance = affordances[i, j]
            gg_score = gg[0]
            gg_width = gg[1]
            gg_depth = gg[3]
            gg_translation = gg[4:-9]
            gg_rotation = gg[-9:].reshape((3, 3))

            left = np.zeros((2, 3))
            left[0] = np.array([-depth_base - finger_width, -gg_width / 2, 0])
            left[1] = np.array([gg_depth, -gg_width / 2, 0])

            right = np.zeros((2, 3))
            right[0] = np.array([-depth_base - finger_width, gg_width / 2, 0])
            right[1] = np.array([gg_depth, gg_width / 2, 0])

            bottom = np.zeros((2, 3))
            bottom[0] = np.array([-finger_width - depth_base, -gg_width / 2, 0])
            bottom[1] = np.array([-finger_width - depth_base, gg_width / 2, 0])

            tail = np.zeros((2, 3))
            tail[0] = np.array([-(tail_length + finger_width + depth_base), 0, 0])
            tail[1] = np.array([-(finger_width + depth_base), 0, 0])

            vertices = np.vstack([left, right, bottom, tail])
            vertices = np.dot(gg_rotation, vertices.T).T + gg_translation

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(vertices)
            line_set.lines = o3d.utility.Vector2iVector([[0, 1], [2, 3], [4, 5], [6, 7]])
            if gg_affordance < 0.5:
                line_set.paint_uniform_color([1, 2*gg_affordance, 0])
            else:
                line_set.paint_uniform_color([-2*gg_affordance+2, 1, 0])
            geometries.append(line_set)
        
        o3d.visualization.draw_geometries(geometries, window_name=f"joint {j}")
