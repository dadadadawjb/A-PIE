"""
Modified from Pointnet_Pointnet2_pytorch:
https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/provider.py#L246
"""

from typing import Tuple
import numpy as np


def pc_normalize(pc:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return (pc, centroid, np.array([m]))

def grasp_normalize(grasps:np.ndarray, centroid:np.ndarray, scale:np.ndarray) -> np.ndarray:
    """
    grasps: (G, 16)
    centroid: (3,)
    scale: (1,)
    """
    grasps[:, [4, 5, 6]] = grasps[:, [4, 5, 6]] - centroid[None, ...]           # translation
    grasps[:, [1, 2, 3, 4, 5, 6]] = grasps[:, [1, 2, 3, 4, 5, 6]] / scale[0]    # width, height, depth, translation
    return grasps

def joint_normalize(joints:np.ndarray, centroid:np.ndarray, scale:np.ndarray) -> np.ndarray:
    """
    joints: (J, 7/8)
    centroid: (3,)
    scale: (1,)
    """
    joints[:, [0, 1, 2]] = joints[:, [0, 1, 2]] - centroid[None, ...]           # translation
    joints[:, [0, 1, 2]] = joints[:, [0, 1, 2]] / scale[0]                      # translation
    return joints


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

def random_scale_grasp(grasps:np.ndarray, point_scales:np.ndarray) -> np.ndarray:
    """ Randomly scale the grasps according to the point cloud. Scale is per grasp.
        Input:
            grasps: BxGx16 array, original batch of grasps
            point_scales: B array, original batch of point cloud scales
        Return:
            grasps: BxGx16 array, scaled batch of grasps
    """
    grasps[:, :, [1, 2, 3, 4, 5, 6]] *= point_scales[:, np.newaxis, np.newaxis]     # width, height, depth, translation
    return grasps

def random_scale_joint(joints:np.ndarray, point_scales:np.ndarray) -> np.ndarray:
    """ Randomly scale the joints according to the point cloud. Scale is per joint.
        Input:
            joints: BxJx(7/8) array, original batch of joints
            point_scales: B array, original batch of point cloud scales
        Return:
            joints: BxJx(7/8) array, scaled batch of joints
    """
    joints[:, :, [0, 1, 2]] *= point_scales[:, np.newaxis, np.newaxis]      # translation
    return joints

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

def shift_grasp(grasps:np.ndarray, point_shifts:np.ndarray) -> np.ndarray:
    """ Randomly shift the grasps according to the point cloud. Shift is per grasp.
        Input:
            grasps: BxGx16 array, original batch of grasps
            point_shifts: Bx3 array, original batch of point cloud shifts
        Return:
            grasps: BxGx16 array, shifted batch of grasps
    """
    grasps[:, :, [4, 5, 6]] += point_shifts[:, np.newaxis, :]       # translation
    return grasps

def shift_joint(joints:np.ndarray, point_shifts:np.ndarray) -> np.ndarray:
    """ Randomly shift the joints according to the point cloud. Shift is per joint.
        Input:
            joints: BxJx(7/8) array, original batch of joints
            point_shifts: Bx3 array, original batch of point cloud shifts
        Return:
            joints: BxJx(7/8) array, shifted batch of joints
    """
    joints[:, :, [0, 1, 2]] += point_shifts[:, np.newaxis, :]       # translation
    return joints


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
