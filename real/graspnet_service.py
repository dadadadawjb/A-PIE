#!/usr/bin/env python
import csv
import pdb
from graspnetAPI.grasp import Grasp
from gsnet import AnyGrasp
import os
import gsnet_cfg as grasp_detector_cfg
import open3d as o3d
from src.utilt import *
from pyquaternion import Quaternion as Quat
from scipy.spatial.transform import Rotation as Rot
import time
import torch
from src.robots import PadGripperFloating
from random import choice
from model.grasp_pose_affordance_model import online_pointnet_grasp_affordance
from model.CVAE_manipulation_model import online_pointnet_trajectory_predict, TrajectoryNetwork
from src.grasp_utilt import view_pre_grasp_trajectory
# ros
# from utilt import *
#
# from collections import deque
# from functools import partial
# import threading

from pyquaternion import Quaternion as Quat
from scipy.spatial.transform import Rotation as Rot
# ros sth import
import rospy
from geometry_msgs.msg import PoseArray
from graspnet_pkg.srv import GraspNetList,GraspNetListResponse
from graspnet_pkg.msg import GraspMsg
from msg_srv.srv import GraspAffordance, GraspTrajectory,GraspAffordanceResponse,GraspTrajectoryResponse
from sensor_msgs import point_cloud2
from geometry_msgs.msg import Point, Quaternion,Pose

_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0

grasp_detector = AnyGrasp(grasp_detector_cfg)
grasp_detector.load_net()

# model
# online_grasp_affordance_model = online_pointnet_grasp_affordance("sem", "pointnet2_ssg", 3, 16,
#                                                                  "./src/model/grasp_attention/best.pth", "cuda", attention=True)
# model = TrajectoryNetwork(pointnet_type="pointnet2_ssg", attention=True).to("cuda")
# online_pointnet_trajectory_predict_model = online_pointnet_trajectory_predict(model, "./src/model/trajectory_attention/best.pth",
#                                                                               4096, "cuda")
print("model load done")

def sample_points(points, method='voxel', num_points=4096, voxel_size=0.005):
    ''' points: numpy.ndarray, [N,3]
        method: 'voxel'/'random'
        num_points: output point number
        voxel_size: grid size used in voxel_down_sample
    '''
    assert (method in ['voxel', 'random'])
    if method == 'voxel':
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud = cloud.voxel_down_sample(voxel_size)
        points = np.array(cloud.points)

    if len(points) >= num_points:
        idxs = np.random.choice(len(points), num_points, replace=False)
    else:
        idxs1 = np.arange(len(points))
        idxs2 = np.random.choice(len(points), num_points - len(points), replace=True)
        idxs = np.concatenate([idxs1, idxs2])
    points = points[idxs]

    return points


def farthest_point_sample(point:np.ndarray, npoint:int):
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
    import tqdm
    for i in tqdm.trange(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    centroids = centroids.astype(np.int32)
    point = point[centroids]
    return point


def point_cloud_center_and_scale(point_could):
    center = np.mean(point_could, axis=0)
    point_could = point_could - center
    scale = np.max(np.sqrt(np.sum(point_could**2, axis=1)))
    point_could = point_could/scale
    return point_could, center, scale

def getGraspNetResult(point_cloud,vis=False,num=100):
    # point_cloud = farthest_point_sample(point_cloud, npoint=4096)
    point_cloud = point_cloud.astype(np.float32)
    lims = [-1, 1, -1, 1, -1, 1]
    # print(points.shape)
    grasp_group = grasp_detector.get_grasp(point_cloud, lims)
    if type(grasp_group) is tuple:
        print("no grasp")
        if vis:
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(point_cloud)
            o3d.visualization.draw_geometries([cloud])
    else:
        grasp_group.nms()
        grasp_group = grasp_group.sort_by_score()
        grasp_group = grasp_group[:num]

        if vis:
            axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(point_cloud)
            grippers = grasp_group.to_open3d_geometry_list()
            o3d.visualization.draw_geometries([cloud, axis_pcd, *grippers])
        # cloud = o3d.geometry.PointCloud()
        # cloud.points = o3d.utility.Vector3dVector(points)
        # o3d.visualization.draw_geometries([cloud])
        # obs = grasp_group.grasp_group_array
        # grasp_pose = get_xyz_rpy(obs)
    return grasp_group

def graspnet_service_handle(req):
    pc_msg = req.pointcloud
    try:
        pc = point_cloud2.read_points_list(pc_msg, field_names=("x", "y", "z"))
        pc = np.array(pc)
    except:
        print("point_cloud2 read error.")
        import pdb;pdb.set_trace()
    # sample points
    grasp_group = getGraspNetResult(pc)

    # grasp_score, grasp_width, grasp_height, grasp_depth, rotation_matrix, grasp_center, obj_ids
    grasp_group = grasp_group.grasp_group_array
    gg_list = []
    for _gg in grasp_group:
        grasp_score, grasp_width, grasp_height, grasp_depth = _gg[:4]
        rotation_matrix = _gg[4:13]
        grasp_center = _gg[13:16]
        obj_ids = _gg[16]
        gg_msg = GraspMsg()
        gg_msg.grasp_score, gg_msg.grasp_width, gg_msg.grasp_height, gg_msg.grasp_depth = \
            grasp_score, grasp_width, grasp_height, grasp_depth
        gg_msg.obj_ids = obj_ids
        grasp_center_msg = Point()
        grasp_center_msg.x, grasp_center_msg.y, grasp_center_msg.z = grasp_center
        gg_msg.grasp_center = grasp_center_msg
        rotation = Quaternion()
        quat = Quat._from_matrix(matrix=rotation_matrix.reshape((3, 3)), rtol=1e-03, atol=1e-03)
        rotation.w, rotation.x, rotation.y, rotation.z = quat.q
        gg_msg.rotation = rotation
        gg_list.append(gg_msg)
    # gg = GraspGroup(gg_array)
    print("Return %d Grasp." % len(gg_list))
    return GraspNetListResponse(gg=gg_list)

def grasp_affordance_service_handle(req):
    pc_msg = req.pointcloud
    gg_srv = req.gg
    try:
        pc = point_cloud2.read_points_list(pc_msg, field_names=("x", "y", "z"))
        pc = np.array(pc)
    except:
        print("point_cloud2 read error.")
        import pdb;pdb.set_trace()
    # sample points
    pointcloud = sample_points(pc, num_points=30000)
    pointcloud = farthest_point_sample(pointcloud, 4096)
    pointcloud_center, center, scale = point_cloud_center_and_scale(pointcloud)
    result = []
    for index, gg_ in enumerate(gg_srv):
        position = [gg_.grasp_center.x,gg_.grasp_center.y,gg_.grasp_center.z]
        rotation = Quat([gg_.rotation.w,gg_.rotation.x,gg_.rotation.y,gg_.rotation.z]).rotation_matrix.reshape(-1)
        grasp_pose = np.concatenate((np.array([gg_.grasp_score, gg_.grasp_width, gg_.grasp_height, gg_.grasp_depth]),
                                     (position-center)/scale,
                                     rotation
                                     ),axis=0)

        if gg_.grasp_width > 0.1:
            # print("width error: ", gg_.width)
            continue
            # pass
        # view_grasp(pointcloud_center, grasp_pose)
        affordance_label, sem_label = online_grasp_affordance_model.predict_grasp_label(pointcloud_center, grasp_pose)
        pre_grasp = affordance_label.argmax(dim=1)[0].item()
        if pre_grasp == 1:
            result.append(int(index))
    print("Return %d Grasps with Affordance."%len(result))
    return GraspAffordanceResponse(result=result)

def grasp_trajectory_service_handle(req):
    pc_msg = req.pointcloud
    gg_ = req.gg
    try:
        pc = point_cloud2.read_points_list(pc_msg, field_names=("x", "y", "z"))
        pc = np.array(pc)
    except:
        print("point_cloud2 read error.")
        import pdb;pdb.set_trace()
    # sample points
    pointcloud = sample_points(pc, num_points=30000)
    pointcloud = farthest_point_sample(pointcloud, 4096)
    pointcloud_center, center, scale = point_cloud_center_and_scale(pointcloud)

    pcd = pointcloud_center.transpose(1, 0)
    pcd = torch.FloatTensor(pcd).to("cuda", dtype=torch.float)
    pcd = pcd.unsqueeze(0)

    # grasp
    position = [gg_.grasp_center.x, gg_.grasp_center.y, gg_.grasp_center.z]
    rotation = Quat([gg_.rotation.w, gg_.rotation.x, gg_.rotation.y, gg_.rotation.z]).rotation_matrix.reshape(-1)
    grasp_pose = np.concatenate((np.array([gg_.grasp_score, gg_.grasp_width, gg_.grasp_height, gg_.grasp_depth]),
                                 (position - center) / scale,
                                 rotation
                                 ), axis=0)

    grasp_pose = torch.FloatTensor(grasp_pose).to("cuda", dtype=torch.float)
    grasp_pose = grasp_pose.unsqueeze(0)
    pre_delta_trajectory = online_pointnet_trajectory_predict_model.sample_trajectory(pcd, grasp_pose)

    real_grasp_pose = grasp_pose.cpu().numpy()[0]
    real_grasp_pose[4:7] = real_grasp_pose[4:7] * scale + center
    # grasp_rpy = mat2euler(real_grasp_pose[7:].reshape(3, 3))
    grasp_rpy = Rot.from_matrix(matrix=real_grasp_pose[7:].reshape(3, 3)).as_euler('xyz')
    grasp_position = real_grasp_pose[4:7]
    trajectory0 = np.concatenate([grasp_position, grasp_rpy])

    trajectory_length = pre_delta_trajectory.shape[0]
    pre_delta_trajectory = pre_delta_trajectory.detach().cpu().numpy()
    pre_delta_trajectory[:, :3] = pre_delta_trajectory[:, :3] * scale

    trajectory_pre = trajectory0
    trajectory = PoseArray()
    trajectory.header.stamp = rospy.Time.now()
    trajectory.header.frame_id = 'pc_base'

    pose = Pose()
    position = Point()
    position.x, position.y, position.z = trajectory_pre[:3]
    orientation = Quaternion()
    orientation.x, orientation.y, orientation.z, orientation.w = pb.getQuaternionFromEuler(trajectory_pre[3:])
    pose.position = position
    pose.orientation = orientation

    trajectory.poses.append(pose)
    for i in range(trajectory_length):

        trajectory_pre = trajectory_pre + pre_delta_trajectory[i]

        pose = Pose()
        position = Point()
        position.x, position.y, position.z = trajectory_pre[:3]
        orientation = Quaternion()
        orientation.x,orientation.y,orientation.z,orientation.w = pb.getQuaternionFromEuler(trajectory_pre[3:])
        pose.position = position
        pose.orientation = orientation
        trajectory.poses.append(pose)
    print("Return Trajectory with %d poses." % len(trajectory.poses))
    return GraspTrajectoryResponse(trajectory=trajectory)





if __name__ == '__main__':
    # rospy.init_node("affordance_trajectory_service")
    rospy.init_node("graspnet_service")
    GraspNetSrv = rospy.Service("GraspNet", GraspNetList, graspnet_service_handle)
    print("Ready to get Grasp List.")
    # GraspAffordanceSrv = rospy.Service("Affordance",GraspAffordance,grasp_affordance_service_handle)
    # print("Ready to get Affordance.")
    # GraspTrajectorySrv = rospy.Service("Trajectory",GraspTrajectory,grasp_trajectory_service_handle)
    # print("Ready to get Trajectory.")
    rospy.spin()
