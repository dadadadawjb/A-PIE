from typing import Tuple, List, Optional, Union
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import MinkowskiEngine as ME

from src_shot.build import shot
from utils import pc_normalize, farthest_point_sample

def joint_denormalize_prime(translation:torch.Tensor, rotation:torch.Tensor, state:Optional[torch.Tensor], 
                      centroid:np.ndarray, scale:np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    translation: (J, 3)
    rotation: (J, 3)
    state: (J,)
    centroid: (3,)
    scale: (1,)
    """
    real_translation = translation * scale[0]
    real_translation = real_translation + torch.from_numpy(centroid[np.newaxis, :]).cuda(translation.device)
    real_rotation = rotation
    real_state = state
    return (real_translation, real_rotation, real_state)


# real point cloud -> real estimated joint
def grad_inference(pc:np.ndarray, has_normal:bool, state_channel:bool, joint_types:List[str], joint_states:List[float], 
              res:float, receptive_field:float, num_sample_points:int, 
              regressor:nn.Module, device:int, cache:bool, data_path:Optional[Path]) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray], 
                                                                                              Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]]:
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

    # start different with original inference
    pcs = torch.from_numpy(pc.astype(np.float32)).cuda(device)      # (N, 3/6)
    pcs = pcs.unsqueeze(0).transpose(2, 1)                          # (1, 3/6, N)

    if state_channel:
        preds_tr, preds_axis, preds_state, _trans_feat = regressor(pcs)
    else:
        preds_tr, preds_axis, _trans_feat = regressor(pcs)

    # denormalize
    J = len(joint_types)
    preds_tr_prime = preds_tr.reshape((J, 3))                   # (J, 3)
    preds_axis_prime = preds_axis.reshape((J, 3))               # (J, 3)
    if state_channel:
        preds_state_prime = preds_state.reshape((J,))           # (J,)
    else:
        preds_state_prime = None
    translation, direction, state = joint_denormalize_prime(preds_tr_prime, preds_axis_prime, preds_state_prime, offset, scale)
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
        state = torch.tensor(states)
        
    return (translation, direction, state, preds_tr, preds_axis, preds_state, offset, scale) if state_channel \
            else (translation, direction, preds_tr, preds_axis, offset, scale)
