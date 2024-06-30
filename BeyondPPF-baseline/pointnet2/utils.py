"""
Modified from Pointnet_Pointnet2_pytorch:
https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/provider.py#L246
"""

from typing import Tuple, Optional
import numpy as np
import torch


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def pc_normalize(pc:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return (pc, centroid, np.array([m]))

def joint_normalize(real_translation:np.ndarray, real_rotation:np.ndarray, real_state:np.ndarray, 
                    centroid:np.ndarray, scale:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    translation = real_translation - centroid
    translation = translation / scale[0]
    rotation = real_rotation
    state = real_state
    return (translation, rotation, state)

def joint_denormalize(translation:np.ndarray, rotation:np.ndarray, state:Optional[np.ndarray], 
                      centroid:np.ndarray, scale:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    translation: (J, 3)
    rotation: (J, 3)
    state: (J,)
    centroid: (3,)
    scale: (1,)
    """
    real_translation = translation * scale[0]
    real_translation = real_translation + centroid[np.newaxis, :]
    real_rotation = rotation
    real_state = state
    return (real_translation, real_rotation, real_state)

def joint_denormalize_batch(translation:torch.Tensor, rotation:torch.Tensor, state:Optional[torch.Tensor], 
                      centroid:torch.Tensor, scale:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    translation: (B, J, 3)
    rotation: (B, J, 3)
    state: (B, J)
    centroid: (B, 3)
    scale: (B, 1)
    """
    real_translation = translation * scale[..., None]
    real_translation = real_translation + centroid[:, None, :]
    real_rotation = rotation
    real_state = state
    return (real_translation, real_rotation, real_state)


def random_point_dropout(batch_pc:np.ndarray, max_dropout_ratio:float, dropout_prob:float) -> np.ndarray:
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio =  np.random.random() * max_dropout_ratio if np.random.random() <= dropout_prob else -1e-6
        drop_idx = np.where(np.random.random((batch_pc.shape[1])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            batch_pc[b, drop_idx, :] = batch_pc[b, 0, :]    # set to the first point
    return batch_pc

def random_scale_point_cloud(batch_data:np.ndarray, scale_low:float, scale_high:float) -> Tuple[np.ndarray, np.ndarray]:
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index,:,:] *= scales[batch_index]
    return (batch_data, scales)

def random_scale_joint(translations:np.ndarray, rotations:np.ndarray, states:np.ndarray, 
                       point_scales:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Randomly scale the joint according to the point cloud. Scale is per joint.
        Input:
            translations: BxJx3 array, original batch of joints' translations
            rotations: BxJx3 array, original batch of joints' rotations
            states: BxJ array, original batch of joints' states
            point_scales: B array, original batch of point cloud scales
        Return:
            translations: BxJx3 array, scaled batch of joints' translations
            rotations: BxJx3 array, scaled batch of joints' rotations
            states: BxJ array, scaled batch of joints' states
    """
    translations *= point_scales[:, np.newaxis, np.newaxis]
    return (translations, rotations, states)

def shift_point_cloud(batch_data:np.ndarray, shift_range:float) -> Tuple[np.ndarray, np.ndarray]:
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B,3))
    for batch_index in range(B):
        batch_data[batch_index,:,:] += shifts[batch_index,:]
    return (batch_data, shifts)

def shift_joint(translations:np.ndarray, rotations:np.ndarray, states:np.ndarray, 
                point_shifts:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Randomly shift the joint according to the point cloud. Shift is per joint.
        Input:
            translations: BxJx3 array, original batch of joints' translations
            rotations: BxJx3 array, original batch of joints' rotations
            states: BxJ array, original batch of joints' states
            point_shifts: Bx3 array, original batch of point cloud shifts
        Return:
            translations: BxJx3 array, shifted batch of joints' translations
            rotations: BxJx3 array, shifted batch of joints' rotations
            states: BxJ array, shifted batch of joints' states
    """
    translations += point_shifts[:, np.newaxis, :]
    return (translations, rotations, states)


def farthest_point_sample(point:np.ndarray, npoint:int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        point: sampled pointcloud, [npoint, D]
        centroids: sampled pointcloud index
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    centroids = centroids.astype(np.int32)
    point = point[centroids]
    return (point, centroids)


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
