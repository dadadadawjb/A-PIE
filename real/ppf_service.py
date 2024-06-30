from typing import Tuple, List, Optional
import os
from pathlib import Path
import omegaconf
from itertools import combinations
import time
import argparse
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import MinkowskiEngine as ME
import cupy as cp

from src_shot.build import shot
from utils import fibonacci_sphere, farthest_point_sample
from models.model import create_shot_encoder, create_encoder
from models.voting import ppf_kernel, rot_voting_kernel

import rospy
from msg_srv.srv import PPF,PPFResponse
from sensor_msgs import point_cloud2

import os
root_dir = os.path.dirname(__file__)

weight_path = os.path.join(root_dir,'weights/04-21-13-58')
data_path = os.path.join(root_dir,'0001.npy')

cfg = omegaconf.OmegaConf.load(f"{weight_path}/.hydra/config.yaml")

# load network
shot_encoder = create_shot_encoder(cfg)
encoder = create_encoder(cfg)
shot_encoder.load_state_dict(torch.load(f'{weight_path}/shot_encoder_latest.pth', map_location='cuda:0'))
encoder.load_state_dict(torch.load(f'{weight_path}/encoder_latest.pth', map_location='cuda:0'))
shot_encoder.eval()
encoder.eval()

from inference import inference
# def inference(point_cloud:np.ndarray, res:float, receptive_field:float, test_samples:int, test_sample_points:int, num_more:int, 
#               rot_num_bins:int, state_channel:bool, joint_types:List[str], joint_states:List[float], topk:float, shot_encoder:nn.Module, encoder:nn.Module, 
#               device:int, cache:bool, data_path:Optional[Path]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
#     angle_tol = 1.5
#     num_samples = int(4 * np.pi / (angle_tol / 180 * np.pi))
#     sphere_pts = np.array(fibonacci_sphere(num_samples))
#     bmm_size = 100000
#     num_rots = 120

#     # sparse quantize
#     if data_path is not None:
#         possible_quantize_path = data_path.parent / f"{data_path.name.replace('.npz', '')}_quantize_{res}.npy"
#         if os.path.exists(possible_quantize_path):
#             pc = np.load(possible_quantize_path)
#         else:
#             indices = ME.utils.sparse_quantize(np.ascontiguousarray(point_cloud), return_index=True, quantization_size=res)[1]
#             pc = point_cloud[indices]
#             if cache:
#                 np.save(possible_quantize_path, pc)
#     else:
#         indices = ME.utils.sparse_quantize(np.ascontiguousarray(point_cloud), return_index=True, quantization_size=res)[1]
#         pc = point_cloud[indices]               # (N', 3)

#     # compute SHOT352 features
#     if data_path is not None:
#         possible_feature_path = data_path.parent / f"{data_path.name.replace('.npz', '')}_shot_{res * receptive_field}_{res * receptive_field}_{res}.npy"
#         if os.path.exists(possible_feature_path):
#             pc_feat = np.load(possible_feature_path)
#         else:
#             pc_feat = shot.compute(pc, res * receptive_field, res * receptive_field).reshape(-1, 352).astype(np.float32)
#             pc_feat[~np.isfinite(pc_feat)] = 0
#             if cache:
#                 np.save(possible_feature_path, pc_feat)
#     else:
#         pc_feat = shot.compute(pc, res * receptive_field, res * receptive_field).reshape(-1, 352).astype(np.float32)
#         pc_feat[~np.isfinite(pc_feat)] = 0      # (N', 352)

#     # estimate normal
#     if data_path is not None:
#         possible_normal_path = data_path.parent / f"{data_path.name.replace('.npz', '')}_normal_{res * receptive_field}_{res}.npy"
#         if os.path.exists(possible_normal_path):
#             normal = np.load(possible_normal_path)
#         else:
#             normal = shot.estimate_normal(pc, res * receptive_field).reshape(-1, 3).astype(np.float32)
#             normal[~np.isfinite(normal)] = 0
#             if cache:
#                 np.save(possible_normal_path, normal)
#     else:
#         normal = shot.estimate_normal(pc, res * receptive_field).reshape(-1, 3).astype(np.float32)
#         normal[~np.isfinite(normal)] = 0        # (N', 3)
    
#     # sample points like training
#     if data_path is not None:
#         possible_sample_path = data_path.parent / f"{data_path.name.replace('.npz', '')}_sample_{test_samples}_{res}.npy"
#         if os.path.exists(possible_sample_path):
#             indices = np.load(possible_sample_path)
#             pc = pc[indices]
#         else:
#             pc, indices = farthest_point_sample(pc, test_samples)
#             if cache:
#                 np.save(possible_sample_path, indices)
#     else:
#         pc, indices = farthest_point_sample(pc, test_samples)       # (N, 3)
#     pc_feat = pc_feat[indices]                                      # (N, 352)
#     normal = normal[indices]                                        # (N, 3)

#     with torch.no_grad():
#         pcs = torch.from_numpy(pc.astype(np.float32)).cuda(device)              # (N, 3)
#         normal = torch.from_numpy(normal.astype(np.float32)).cuda(device)       # (N, 3)
#         pc_feat = torch.from_numpy(pc_feat.astype(np.float32)).cuda(device)     # (N, 352)

#         # sample point tuples
#         point_idxs = torch.randint(0, pc.shape[0], (test_sample_points, 2)).cuda(device)                # (test_sample_points, 2)
#         point_idx_more = torch.randint(0, pc.shape[0], (test_sample_points, num_more)).cuda(device)     # (test_sample_points, num_more)
#         point_idx_all = torch.cat([point_idxs, point_idx_more], -1)                                     # (test_sample_points, 2 + num_more)

#         # forward SHOT352 features to compact features
#         shot_feat = shot_encoder(pc_feat)   # (N, feature_dim)

#         # forward concat point tuple features to estimate
#         shot_inputs = torch.cat([shot_feat[point_idx_all[:, i]] for i in range(0, point_idx_all.shape[-1])], -1)    # (test_sample_points, feature_dim * (2 + num_more))
#         normal_inputs = torch.cat([torch.max(torch.sum(normal[point_idx_all[:, i]] * normal[point_idx_all[:, j]], dim=-1, keepdim=True),
#                                             torch.sum(-normal[point_idx_all[:, i]] * normal[point_idx_all[:, j]], dim=-1, keepdim=True))
#                                     for (i, j) in combinations(np.arange(point_idx_all.shape[-1]), 2)], -1)         # (test_sample_points, (2+num_more \choose 2))
#         coord_inputs = torch.cat([pcs[point_idx_all[:, i]] - pcs[point_idx_all[:, j]] for (i, j) in combinations(np.arange(point_idx_all.shape[-1]), 2)], -1)   # (test_sample_points, 3 * (2+num_more \choose 2))
#         inputs = torch.cat([coord_inputs, normal_inputs, shot_inputs], -1)
#         preds = encoder(inputs)     # (test_sample_points, (2 + rot_num_bins + 1( + 1)) * J)

#         J = len(joint_types)
#         translations, directions = [], []
#         if state_channel:
#             states = []
#         for j in range(J):
#             # vote translation
#             preds_tr = preds[:, 2*j:2*(j+1)]
            
#             block_size = (preds.shape[0] + 512 - 1) // 512

#             corners = np.stack([np.min(pc, 0), np.max(pc, 0)])
#             grid_res = ((corners[1] - corners[0]) / res).astype(np.int32) + 1

#             with cp.cuda.Device(device):
#                 grid_obj = cp.asarray(np.zeros(grid_res, dtype=np.float32))
                
#                 probs = np.ones((preds_tr.shape[0],))
#                 ppf_kernel(
#                     (block_size, 1, 1),
#                     (512, 1, 1),
#                     (
#                         cp.ascontiguousarray(cp.asarray(pc).astype(cp.float32)),
#                         cp.ascontiguousarray(cp.asarray(preds_tr).astype(cp.float32)),
#                         cp.ascontiguousarray(cp.asarray(probs).astype(cp.float32)),
#                         cp.ascontiguousarray(cp.asarray(point_idxs).astype(cp.int32)),
#                         grid_obj,
#                         cp.ascontiguousarray(cp.asarray(corners[0]).astype(cp.float32)),
#                         cp.float32(res),
#                         cp.int32(preds.shape[0]),
#                         cp.int32(num_rots),
#                         grid_obj.shape[0],
#                         grid_obj.shape[1],
#                         grid_obj.shape[2]
#                     )
#                 )
            
#                 grid_obj = grid_obj.get()
#             cand = np.array(np.unravel_index([np.argmax(grid_obj, axis=None)], grid_obj.shape)).T[::-1]
#             cand_world = corners[0] + cand * res

#             best_idx = np.linalg.norm(pc - cand_world, axis=-1).argmin()
#             translation = pc[best_idx]
#             translations.append(translation)
            
#             # vote direction
#             preds_axis = preds[:, (2*J+rot_num_bins*j):(2*J+rot_num_bins*(j+1))]
#             preds_axis = torch.softmax(preds_axis, -1)
#             preds_axis = torch.multinomial(preds_axis, 1).float()
#             preds_axis = preds_axis / (rot_num_bins - 1) * np.pi

#             preds_conf = torch.sigmoid(preds[:, -1*J+j])
#             num_top = int((1 - topk) * preds_conf.shape[0])

#             preds_conf[torch.topk(preds_conf, num_top, largest=False, sorted=False)[1]] = 0
#             preds_conf[preds_conf > 0] = 1

#             with cp.cuda.Device(device):
#                 candidates = cp.zeros((point_idxs.shape[0], num_rots, 3), cp.float32)

#                 rot_voting_kernel(
#                     (block_size, 1, 1),
#                     (512, 1, 1),
#                     (
#                         cp.ascontiguousarray(cp.asarray(pc).astype(cp.float32)),
#                         cp.ascontiguousarray(cp.asarray(preds_axis).astype(cp.float32)),
#                         candidates,
#                         cp.ascontiguousarray(cp.asarray(point_idxs).astype(cp.int32)),
#                         point_idxs.shape[0],
#                         num_rots
#                     )
#                 )
                
#                 candidates = candidates.get().reshape(-1, 3)
#             conf = preds_conf[:, None].expand(-1, num_rots).reshape(-1, 1)
            
#             with torch.no_grad():
#                 sph_cp = torch.tensor(sphere_pts.T, dtype=torch.float32).cuda(device)
#                 candidates = torch.from_numpy(candidates).cuda(device)

#                 counts = torch.zeros((sphere_pts.shape[0],), dtype=torch.float32).cuda(device)
#                 for i in range((candidates.shape[0] - 1) // bmm_size + 1):
#                     cos = candidates[i * bmm_size:(i + 1) * bmm_size].mm(sph_cp)
#                     counts += torch.sum((cos > np.cos(2 * angle_tol / 180 * np.pi)).float() * conf[i * bmm_size:(i + 1) * bmm_size], 0)

#             direction = np.array(sphere_pts[np.argmax(counts.cpu().numpy())])
#             directions.append(direction)

#             # vote state
#             if state_channel:
#                 preds_state = preds[:, -2*J+j]
#                 if joint_types[j] == 'revolute':
#                     state = (torch.mean(preds_state) + joint_states[j] / 180.0 * np.pi / 2).cpu().numpy()
#                     state *= 180.0 / np.pi
#                 elif joint_types[j] == 'prismatic':
#                     state = (torch.mean(preds_state) + joint_states[j] / 100.0 / 2).cpu().numpy()
#                     state *= 100.0
#                 else:
#                     raise ValueError
#                 states.append(state)
#         translations = np.stack(translations)
#         directions = np.stack(directions)
#         if state_channel:
#             states = np.stack(states)
        
#     return (translations, directions, states) if state_channel else (translations, directions)


def ppf_handle(req):
    pc_msg = req.pointcloud
    
    try:
        pc = point_cloud2.read_points_list(pc_msg, field_names=("x", "y", "z"))
        pc = np.array(pc)
    except:
        print("point_cloud2 read error.")
        import pdb;pdb.set_trace()
     # inference
    start_time = time.time()
    results = inference(pc, cfg.shot.res, cfg.shot.receptive_field, cfg.test_samples, cfg.test_sample_points, cfg.num_more, 
                        cfg.encoder.rot_num_bins, cfg.encoder.state, cfg.types, cfg.states, cfg.topk, shot_encoder, encoder, cfg.device, False, None)
    
    if cfg.encoder.state:
        translations, directions, states = results
    else:
        translations, directions = results
    end_time = time.time()
    print('estimated translations:', translations)
    print('estimated directions:', directions)
    if cfg.encoder.state:
        print('estimated states:', states)
    print('estimation time:', end_time - start_time)
    axis = []
    for j in range(translations.shape[0]):
        translation = translations[j]
        direction = directions[j]
        axis.extend(translation.tolist())
        axis.extend(direction.tolist())
    return PPFResponse(axis=axis)
    
    
    

if __name__ == '__main__':
    import threading,rospy,rosgraph,socket
    rospy.init_node("ppf_service")
    rospy.Service("ppf_service",PPF,ppf_handle)
    rospy.loginfo("PPF service ready")
    # threading.Thread(target=lambda:rospy.spin()).start()
    rospy.spin()
