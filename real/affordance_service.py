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
from pyquaternion import Quaternion as Quat

root_dir = os.path.dirname(__file__)
import rospy
from msg_srv.srv import Affordance,AffordanceResponse
from msg_srv.msg import FloatList
from sensor_msgs import point_cloud2

weight_path = os.path.join(root_dir,"weights","04-23-20-09")
cfg = omegaconf.OmegaConf.load(f"{weight_path}/.hydra/config.yaml")

# load network
model = grasp_embedding_network(cfg).to(cfg.device)
model.load_state_dict(torch.load(f'{weight_path}/model_latest.pth',map_location='cuda:0'))
model.eval()


def convertGraspMsgtoNumpy(gg_list,to_affordance=False):
    # grasp_score, grasp_width, grasp_height, grasp_depth, rotation_matrix, grasp_center, obj_ids
    gg_array = []
    for gg in gg_list:
        grasp_score = gg.grasp_score
        grasp_width = gg.grasp_width
        grasp_height = gg.grasp_height
        grasp_depth = gg.grasp_depth
        pre_ = [grasp_score, grasp_width, grasp_height, grasp_depth]
        rotation = gg.rotation
        grasp_center = gg.grasp_center
        obj_ids = gg.obj_ids
        quat = Quat(rotation.w, rotation.x, rotation.y, rotation.z)
        rotation_matrix = quat.rotation_matrix.flatten().tolist()
        grasp_center = [grasp_center.x, grasp_center.y, grasp_center.z]
        if to_affordance:
            gg_array.append(pre_ + grasp_center + rotation_matrix)
        else:
            gg_array.append(pre_ + rotation_matrix + grasp_center + [obj_ids])
    gg_array = np.array(gg_array)
    return gg_array

from inference import inference
# def inference(classification:bool, pc:np.ndarray, has_normal:bool, joints:np.ndarray, grasps:np.ndarray, 
#               res:float, receptive_field:float, num_sample_points:int, normalization:bool, 
#               model:nn.Module, device:int, cache:bool, data_path:Optional[Path]) -> np.ndarray:
#     # sparse quantize
#     if data_path is not None:
#         possible_quantize_path = data_path.parent / f"{data_path.name.replace('.npz', '')}_quantize_{res}.npy"
#         if os.path.exists(possible_quantize_path):
#             pc = np.load(possible_quantize_path)
#         else:
#             indices = ME.utils.sparse_quantize(np.ascontiguousarray(pc), return_index=True, quantization_size=res)[1]
#             pc = np.ascontiguousarray(pc[indices].astype(np.float32))
#             if cache:
#                 np.save(possible_quantize_path, pc)
#     else:
#         indices = ME.utils.sparse_quantize(np.ascontiguousarray(pc), return_index=True, quantization_size=res)[1]
#         pc = np.ascontiguousarray(pc[indices].astype(np.float32))   # (N', 3)

#     # estimate normal
#     if has_normal:
#         if data_path is not None:
#             possible_normal_path = data_path.parent / f"{data_path.name.replace('.npz', '')}_normal_{res * receptive_field}_{res}.npy"
#             if os.path.exists(possible_normal_path):
#                 pc_normal = np.load(possible_normal_path)
#             else:
#                 pc_normal = shot.estimate_normal(pc, res * receptive_field).reshape(-1, 3).astype(np.float32)
#                 pc_normal[~np.isfinite(pc_normal)] = 0
#                 if cache:
#                     np.save(possible_normal_path, pc_normal)
#         else:
#             pc_normal = shot.estimate_normal(pc, res * receptive_field).reshape(-1, 3).astype(np.float32)
#             pc_normal[~np.isfinite(pc_normal)] = 0                  # (N', 3)
#         pc = np.concatenate([pc, pc_normal], axis=1)                # (N', 6)
    
#     # sample points
#     if data_path is not None:
#         possible_sample_path = data_path.parent / f"{data_path.name.replace('.npz', '')}_sample_{num_sample_points}_{res}.npy"
#         if os.path.exists(possible_sample_path):
#             indices = np.load(possible_sample_path)
#             pc = pc[indices]
#         else:
#             pc, indices = farthest_point_sample(pc, num_sample_points)
#             if cache:
#                 np.save(possible_sample_path, indices)
#     else:
#         pc, indices = farthest_point_sample(pc, num_sample_points)  # (N, 3/6)
    
#     # normalize
#     if normalization:
#         pc[:, 0:3], offset, scale = pc_normalize(pc[:, 0:3])                        # (3,), (1,)
#         joints = joint_normalize(joints, offset, scale)                             # (J, 7/8)
#         grasps = grasp_normalize(grasps, offset, scale)                             # (G, 16)

#     with torch.no_grad():
#         J = joints.shape[0]
#         G = grasps.shape[0]
#         pcs = torch.from_numpy(pc.astype(np.float32)).cuda(device)                  # (N, 3/6)
#         joints = torch.from_numpy(joints.astype(np.float32)).cuda(device)           # (J, 7/8)
#         grasps = torch.from_numpy(grasps.astype(np.float32)).cuda(device)           # (G, 16)
#         pcs = pcs.unsqueeze(0)                                                      # (1, N, 3/6)
#         grasps = grasps[:, None, :].repeat(1, J, 1)                                 # (G, J, 16)
#         grasps = grasps.reshape(-1, grasps.shape[-1])                               # (G*J, 16)
#         grasps = grasps.unsqueeze(0)                                                # (1, G*J, 16)
#         joints = joints[None, :, :].expand(G, -1, -1)                               # (G, J, 7/8)
#         joints = joints.reshape(-1, joints.shape[-1])                               # (G*J, 7/8)
#         joints = joints.unsqueeze(0)                                                # (1, G*J, 7/8)
#         prediction = model(pcs, grasps, joints)         # (1, G*J, c)
#         if classification:
#             prediction = F.softmax(prediction, dim=-1)
#             prediction = prediction.squeeze(0).reshape((G, J, -1))          # (G, J, c)
#         else:
#             prediction = torch.sigmoid(prediction)
#             prediction = prediction.squeeze(0).squeeze(-1).reshape((G, J))  # (G, J)

#         prediction = prediction.cpu().numpy()
    
#     return prediction


def affordance_handle(req):
    GraspMsg_list = req.gg
    axis = req.axis
    pc_msg = req.pointcloud
    
    try:
        pc = point_cloud2.read_points_list(pc_msg, field_names=("x", "y", "z"))
        point_cloud = np.array(pc)
    except:
        print("point_cloud2 read error.")
        import pdb;pdb.set_trace()
    # [grasp_score, grasp_width, grasp_height, grasp_depth] + rotation_matrix + grasp_center + [obj_ids]
    grasps = convertGraspMsgtoNumpy(GraspMsg_list,to_affordance=True)
    num = int(len(axis)/7)
    axis = list(axis)
    axis_list = []
    for j in range(num):
        axis_list.append(axis[j*7:j*7+7])
    joints = np.array(axis_list)
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
    
    result = []
    for j in range(num):
        floatlist = FloatList()
        floatlist.floatlist = affordances[:,j].tolist()
        result.append(floatlist)
    return AffordanceResponse(result=result)
    
    

if __name__ == '__main__':
    rospy.init_node("affordance_service")
    rospy.Service("affordance_service",Affordance,affordance_handle)
    rospy.loginfo("Affordance service ready")
    rospy.spin()
    
