from typing import Tuple
import pybullet as pb
import pybullet_data as pbd
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import transformations as tf
import configargparse
import os
import json
import tqdm
import open3d as o3d
from munch import DefaultMunch

from utils import get_camera, generate_camera, \
    get_point_cloud, get_base_pose, get_joint_poses, get_link_pos, \
    draw_link_coord, draw_gripper
from visualize_data import visualize

def calculate_transformation_diff(pose1:np.ndarray, pose2:np.ndarray) -> Tuple[float, float]:
    tr_diff = np.linalg.norm(pose1[:3, 3] - pose2[:3, 3])
    rot_diff = np.arccos(np.clip((np.trace(pose1[:3, :3].T @ pose2[:3, :3]) - 1) / 2, -1, 1))
    return (tr_diff, rot_diff)


def config_parse() -> configargparse.Namespace:
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True, help='config file path')
    # general config
    parser.add_argument('--object', type=str, help='object class')
    parser.add_argument('--id', type=str, help='id')
    parser.add_argument('--log_path', type=str, default="logs", help='log path')
    parser.add_argument('--object_path', type=str, help='model path')
    parser.add_argument('--seed', type=int, default=100, help='random seed')
    parser.add_argument('--num_samples', type=int, default=100, help='number of samples, for random object poses and joint poses')
    parser.add_argument('--gui', action='store_true', help='enable gui')
    parser.add_argument('--vis', action='store_true', help='whether to visualize')
    parser.add_argument('--pause', action='store_true', help='whether to pause after each step')
    # camera config
    parser.add_argument('--auto_camera', type=int, help='whether to use auto camera, 0 for no, positive for number, for random camera poses')
    parser.add_argument('--auto_camera_distance_min', type=float, help='auto camera distance min')
    parser.add_argument('--auto_camera_distance_max', type=float, help='auto camera distance max')
    parser.add_argument('--auto_camera_fov', type=float, help='auto camera fov')
    parser.add_argument('--auto_camera_near', type=float, help='auto camera near')
    parser.add_argument('--auto_camera_far', type=float, help='auto camera far')
    parser.add_argument('--auto_camera_height', type=int, help='auto camera height')
    parser.add_argument('--auto_camera_width', type=int, help='auto camera width')
    parser.add_argument('--auto_camera_cone_direction', nargs=3, type=float, help='auto camera cone direction in object local frame')
    parser.add_argument('--auto_camera_cone_angle', type=float, help='auto camera cone angle in degree')
    parser.add_argument('--auto_camera_up_axis', nargs=3, type=float, help='auto camera up axis coplanar in object local frame')
    parser.add_argument('--camera_target_x', nargs='+', type=float, help='camera target x world coordinate')
    parser.add_argument('--camera_target_y', nargs='+', type=float, help='camera target y world coordinate')
    parser.add_argument('--camera_target_z', nargs='+', type=float, help='camera target z world coordinate')
    parser.add_argument('--camera_distance', nargs='+', type=float, help='camera distance to target')
    parser.add_argument('--camera_yaw', nargs='+', type=float, help='camera yaw degree')
    parser.add_argument('--camera_pitch', nargs='+', type=float, help='camera pitch degree')
    parser.add_argument('--camera_roll', nargs='+', type=float, help='camera roll degree')
    parser.add_argument('--camera_up', nargs='+', type=str, choices=['y', 'z'], help='camera z axis')
    parser.add_argument('--camera_fov', nargs='+', type=float, help='camera fov')
    parser.add_argument('--camera_near', nargs='+', type=float, help='camera near')
    parser.add_argument('--camera_far', nargs='+', type=float, help='camera far')
    parser.add_argument('--camera_height', nargs='+', type=int, help='camera height')
    parser.add_argument('--camera_width', nargs='+', type=int, help='camera width')
    # object config
    parser.add_argument('--object_scale', type=float, help='object scale')
    parser.add_argument('--link_id', nargs='+', type=int, help='link id')
    parser.add_argument('--link_type', nargs='+', type=str, choices=['revolute', 'prismatic'], help='link type')
    parser.add_argument('--link_axis', nargs='+', type=str, choices=['x', 'y', 'z'], help='link axis')
    parser.add_argument('--link_pos', nargs='+', type=str, help='link position')
    parser.add_argument('--link_offset', nargs='+', type=float, help='link offset')
    # grasp config
    parser.add_argument('--grasp', action='store_true', default=False, help='whether to collect grasp affordance')
    parser.add_argument('--gsnet_weight', type=str, help='the path to the gsnet weight for grasp affordance')
    parser.add_argument('--angle_threshold', type=float, help='angle threshold in degree for grasp affordance')
    parser.add_argument('--distance_threshold', type=float, help='distance threshold in cm for grasp affordance')
    parser.add_argument('--gripper_scale', type=float, help='grasp gripper scale for grasp affordance')
    parser.add_argument('--allow', action='store_true', default=False, help='whether to allow tiny move for grasp')
    parser.add_argument('--allow_distance', type=float, help='allow tiny move distance in cm for grasp')
    parser.add_argument('--allow_angle', type=float, help='allow tiny move angle in degree for grasp')
    parser.add_argument('--direct', action='store_true', default=False, help='whether to collect directly grasping instead of approaching')
    parser.add_argument('--contact', action='store_true', default=False, help='whether to collect directly using contact instead of actually manipulating')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # common config
    args = config_parse()
    print('===>', args.object + '_' + args.id)
    os.makedirs(args.log_path, exist_ok=True)
    os.makedirs(os.path.join(args.log_path, args.object), exist_ok=True)
    os.makedirs(os.path.join(args.log_path, args.object, args.id), exist_ok=True)
    np.random.seed(args.seed)
    if (len(args.link_id) == 0) or (len(args.link_type) == 0) or (len(args.link_axis) == 0) or (len(args.link_pos) == 0) or (len(args.link_offset) == 0):
        print('===> you should set `link_id` and `link_type` and `link_axis` and `link_pos` and `link_offset` manually')

    # pybullet setting
    if args.gui:
        uid = pb.connect(pb.GUI)
    else:
        uid = pb.connect(pb.DIRECT)
    pb.setAdditionalSearchPath(pbd.getDataPath())

    # set camera
    if args.auto_camera > 0:
        num_camera = args.auto_camera
        view_matrix = None
        proj_matrix = None
        extrinsic = None
    else:
        num_camera, view_matrix, proj_matrix, extrinsic = get_camera(args.camera_target_x, args.camera_target_y, args.camera_target_z, 
                                                    args.camera_distance, args.camera_yaw, args.camera_pitch, args.camera_roll, args.camera_up,
                                                    args.camera_fov, args.camera_near, args.camera_far, args.camera_height, args.camera_width)

    # set grasp
    if args.grasp:
        from gsnet import AnyGrasp
        grasp_detector_cfg = {
            'checkpoint_path': args.gsnet_weight,
            'max_gripper_width': 0.1,
            'gripper_height': 0.03,
            'top_down_grasp': False,
            'add_vdistance': True
        }
        grasp_detector_cfg = DefaultMunch.fromDict(grasp_detector_cfg)
        grasp_detector = AnyGrasp(grasp_detector_cfg)
        grasp_detector.load_net()

        from models.gripper import PadGripperFloating

    for i in tqdm.trange(args.num_samples):
        pb.resetSimulation(physicsClientId=uid)
        os.makedirs(os.path.join(args.log_path, args.object, args.id, '%04d' % i), exist_ok=True)
        
        # load background
        # plane = pb.loadURDF('plane.urdf')
        # table = pb.loadURDF('table/table.urdf', 
        #                     useFixedBase=True, globalScaling=0.6)
        # pb.resetBasePositionAndOrientation(
        #     table, [0.7, 0., 0.0], [0, 0, -0.707, 0.707])
        # table_plane_z = (0.6 + 0.05 / 2) * 0.6

        # load object
        object_model = pb.loadURDF(args.object_path, useFixedBase=True, globalScaling=args.object_scale)
        object_x = np.random.randn()
        object_x = np.clip(object_x,-2, 2) * 0.1
        object_y = np.random.randn()
        object_y = np.clip(object_y, -2, 2) * 0.1
        object_z = np.random.randn()
        object_z = np.clip(object_z, -2, 2) * 0.045
        object_rz = np.random.randn()
        object_rz = np.clip(object_rz, -1, 1) * 30/180.*np.pi
        pb.resetBasePositionAndOrientation(object_model,
                                           [0.7+object_x, 0.0+object_y, 0.7+object_z],
                                           pb.getQuaternionFromEuler((0, 0, 0.0+object_rz)))
        states = []
        real_states = []
        limit_states = []
        for j, link_id in enumerate(args.link_id):
            if args.link_type[j] == 'revolute':
                joint_info = pb.getJointInfo(object_model, link_id)
                assert joint_info[2] == pb.JOINT_REVOLUTE
                min_limit = joint_info[8]
                max_limit = joint_info[9]
                if abs(max_limit - min_limit - 90/180*np.pi) > 1/180*np.pi:
                    if abs(max_limit - min_limit - np.pi) > 1/180*np.pi:
                        if (max_limit - min_limit) > np.pi:
                            limit_state = np.pi
                            limit_states.append(limit_state)
                            print('===> warning: >180 deg limit, use 180 deg limit instead')
                        elif (max_limit - min_limit) > 90/180*np.pi:
                            limit_state = np.pi/2
                            limit_states.append(limit_state)
                            print('===> warning: >90 deg limit, use 90 deg limit instead')
                        else:
                            limit_state = max_limit - min_limit
                            limit_states.append(limit_state)
                            print('===> warning: <90 deg limit, use itself instead')
                    else:
                        limit_state = np.pi
                        limit_states.append(limit_state)
                        print('===> warning: 180 deg limit')
                else:
                    limit_state = np.pi/2
                    limit_states.append(limit_state)
                closed = np.random.randn() > 0
                if closed:
                    real_angle = min_limit
                else:
                    real_angle = np.random.uniform(min_limit, min_limit + limit_state)
                pb.resetJointState(object_model, link_id, real_angle)
                pb.setJointMotorControl2(object_model, link_id, pb.VELOCITY_CONTROL, targetVelocity=0., force=0.)
                real_states.append(real_angle)
                states.append(real_angle - min_limit)
            elif args.link_type[j] == 'prismatic':
                joint_info = pb.getJointInfo(object_model, link_id)
                assert joint_info[2] == pb.JOINT_PRISMATIC
                min_limit = joint_info[8]
                max_limit = joint_info[9]
                limit_state = max_limit - min_limit
                limit_states.append(limit_state)
                closed = np.random.randn() > 0
                if closed:
                    real_distance = min_limit
                else:
                    real_distance = np.random.uniform(min_limit, max_limit)
                pb.resetJointState(object_model, link_id, real_distance)
                pb.setJointMotorControl2(object_model, link_id, pb.VELOCITY_CONTROL, targetVelocity=0., force=0.)
                real_states.append(real_distance)
                states.append(real_distance - min_limit)
            else:
                raise ValueError
        
        # auto camera
        if args.auto_camera > 0:
            view_matrix, proj_matrix, extrinsic = generate_camera(args.auto_camera, object_model, 
                args.auto_camera_distance_min, args.auto_camera_distance_max, 
                args.auto_camera_fov, args.auto_camera_width, args.auto_camera_height, 
                args.auto_camera_near, args.auto_camera_far, 
                np.array(args.auto_camera_cone_direction), 
                args.auto_camera_cone_angle/180 * np.pi, 
                np.array(args.auto_camera_up_axis))

        # record
        with open(os.path.join(args.log_path, args.object, args.id, '%04d' % i, 'config.json'), 'w') as f:
            object_pos, object_ori = pb.getBasePositionAndOrientation(object_model)
            conf_dict = {
                'position': object_pos, 
                'orientation': object_ori, 
                'scale': args.object_scale, 
                'link_id': args.link_id, 
                'link_type': args.link_type, 
                'link_state': states, 
                'link_state_limit': limit_states, 
                'link_axis': args.link_axis, 
                'link_pos': args.link_pos, 
                'link_offset': args.link_offset, 
                'camera_view_matrix': [np.array(view_mat).reshape((4, 4)).tolist() for view_mat in view_matrix], 
                'camera_proj_matrix': [np.array(proj_mat).reshape((4, 4)).tolist() for proj_mat in proj_matrix], 
                'camera_extrinsic': [np.array(ext_mat).reshape((4, 4)).tolist() for ext_mat in extrinsic], 
            }
            if args.grasp:
                conf_dict['grasp'] = True
                conf_dict['angle_threshold'] = args.angle_threshold
                conf_dict['distance_threshold'] = args.distance_threshold
                conf_dict['gripper_scale'] = args.gripper_scale
            else:
                conf_dict['grasp'] = False
            json.dump(conf_dict, f, indent=4)

        # get pose
        base_pose = get_base_pose(object_model)
        joint_poses = get_joint_poses(object_model, args.link_id)
        for joint_id in range(len(joint_poses)):
            if args.link_axis[joint_id] == 'x':
                link_axis = np.array([1, 0, 0])
            elif args.link_axis[joint_id] == 'y':
                link_axis = np.array([0, 1, 0])
            elif args.link_axis[joint_id] == 'z':
                link_axis = np.array([0, 0, 1])
            else:
                raise ValueError
            link_axis = np.dot(joint_poses[joint_id][:3, :3], link_axis)
            joint_poses[joint_id][:3, 3] += link_axis * args.link_offset[joint_id] * args.object_scale

        # draw pose
        draw_link_coord(pose=base_pose)
        for joint_pose in joint_poses:
            draw_link_coord(pose=joint_pose)
        for ext_mat in extrinsic:
            draw_link_coord(pose=ext_mat)

        # deal with per-camera
        for camera_id in range(num_camera):
            # get point cloud
            if args.auto_camera > 0:
                point_cloud = get_point_cloud(args.auto_camera_width, args.auto_camera_height, view_matrix[camera_id], proj_matrix[camera_id], obj_id=0)
            else:
                point_cloud = get_point_cloud(args.camera_width[camera_id], args.camera_height[camera_id], view_matrix[camera_id], proj_matrix[camera_id], obj_id=0)
            
            # transform
            view_matrix_inv = np.linalg.inv(extrinsic[camera_id])
            base_pose_cam = view_matrix_inv @ base_pose
            joint_poses_cam = []
            for joint_pose in joint_poses:
                joint_poses_cam.append(view_matrix_inv @ joint_pose)
            point_cloud_cam = (view_matrix_inv[:3, :3] @ point_cloud.T + view_matrix_inv[:3, 3].reshape(3, 1)).T
            
            # grasp affordance
            if args.grasp:
                # grasp detection without semantics
                gg_cam = grasp_detector.get_grasp(point_cloud_cam.astype(np.float32), colors=None, lims=[-2, 2, -2, 2, -2, 2])
                if gg_cam is None:
                    gg_cam = []
                else:
                    if len(gg_cam) != 2:
                        gg_cam = []
                    else:
                        gg_cam, _ = gg_cam
                collect_gg_array = []
                both_success_grasp_num = 0
                half_success_grasp_num = 0
                none_success_grasp_num = 0
                fail_grasp_num = 0
                if len(gg_cam) > 0:
                    gg_cam.nms()
                    gg_cam.sort_by_score()
                    gg_cam = gg_cam[:100]
                    drawn_items = []
                    if args.allow:
                        # build point cloud search tree in advance
                        pcd_cam = o3d.geometry.PointCloud()
                        pcd_cam.points = o3d.utility.Vector3dVector(point_cloud_cam)
                        pcd_tree_cam = o3d.geometry.KDTreeFlann(pcd_cam)
                        # estimate point cloud normals in advance
                        pcd_cam.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(100))
                    for gg_cam_ in gg_cam:
                        # if collected
                        collect_grasp_ = [gg_cam_.score, gg_cam_.width, gg_cam_.height, gg_cam_.depth, 
                                          np.copy(gg_cam_.translation), np.copy(gg_cam_.rotation_matrix)]   # grasp information
                        
                        if args.allow:
                            # translate grasp pose to the nearest point
                            [k, idx, _] = pcd_tree_cam.search_knn_vector_3d(gg_cam_.translation, 1)
                            nearest_point_cam = np.asarray(pcd_cam.points)[idx, :].squeeze(axis=0)
                            if np.linalg.norm(nearest_point_cam - gg_cam_.translation) < args.allow_distance / 100.0:
                                gg_cam_.translation = nearest_point_cam
                            
                            # rotate grasp pose to the nearest point's normal
                            normal_cam = np.asarray(pcd_cam.normals)[idx, :].squeeze(axis=0)
                            normal_cam /= np.linalg.norm(normal_cam)
                            if abs(np.dot(normal_cam, gg_cam_.rotation_matrix[:, 0])) > np.cos(args.allow_angle / 180.0 * np.pi):
                                # x axis is the grasp depth
                                gg_cam_.rotation_matrix[:, 0] = normal_cam if np.dot(normal_cam, gg_cam_.rotation_matrix[:, 0]) > 0 else (-1 * normal_cam)
                                gg_cam_.rotation_matrix[:, 1] = np.cross(gg_cam_.rotation_matrix[:, 2], gg_cam_.rotation_matrix[:, 0])
                                gg_cam_.rotation_matrix[:, 1] /= np.linalg.norm(gg_cam_.rotation_matrix[:, 1])
                                gg_cam_.rotation_matrix[:, 2] = np.cross(gg_cam_.rotation_matrix[:, 0], gg_cam_.rotation_matrix[:, 1])
                                gg_cam_.rotation_matrix[:, 2] /= np.linalg.norm(gg_cam_.rotation_matrix[:, 2])
                        for j, link_id in enumerate(args.link_id):
                            # if collected
                            collect_grasp = collect_grasp_ + [j]    # against link

                            collect_two = 0
                            grasp_two = 0
                            success_two = 0
                            last_orientation = False    # False for negative, True for positive
                            for grasp_orientation in [-1, 1]:
                                # initialize gripper
                                gripper = PadGripperFloating(uid=uid, useFixedBase=True, globalScaling=args.gripper_scale)

                                grasp_pose_cam = np.identity(4)
                                grasp_pose_cam[:3, :3] = gg_cam_.rotation_matrix
                                grasp_pose_cam[:3, 3] = gg_cam_.translation
                                
                                offset = np.identity(4)
                                offset[:3, 3] = np.array([gg_cam_.depth - 0.035 * args.gripper_scale, 0, 0])
                                target_gripper_pose = extrinsic[camera_id] @ (grasp_pose_cam @ offset)
                                offset = np.identity(4)
                                offset[0, 3] = -0.05 * args.gripper_scale
                                pre_gripper_pose = target_gripper_pose @ offset
                                rpy = Rot.from_matrix(target_gripper_pose[:3, :3]).as_euler('ZYX', degrees=False)

                                # gripper.move_gripper(gg_cam_.width, wait_done=True)
                                pb.resetJointState(gripper._id, gripper.mimic_parent_id, gg_cam_.width/2., 0)
                                pb.resetJointState(gripper._id, gripper.mimic_parent_id+1, gg_cam_.width/2., 0)
                                pb.resetJointState(gripper._id, 3, rpy[0], 0)
                                pb.resetJointState(gripper._id, 4, rpy[1], 0)
                                pb.resetJointState(gripper._id, 5, rpy[2], 0)
                                if args.direct:
                                    pb.resetJointState(gripper._id, 0, pre_gripper_pose[0, 3], 0)
                                    pb.resetJointState(gripper._id, 1, pre_gripper_pose[1, 3], 0)
                                    pb.resetJointState(gripper._id, 2, pre_gripper_pose[2, 3], 0)
                                    pb.resetJointState(gripper._id, 0, target_gripper_pose[0, 3], 0)
                                    pb.resetJointState(gripper._id, 1, target_gripper_pose[1, 3], 0)
                                    pb.resetJointState(gripper._id, 2, target_gripper_pose[2, 3], 0)
                                else:
                                    pb.resetJointState(gripper._id, 0, pre_gripper_pose[0, 3], 0)
                                    pb.resetJointState(gripper._id, 1, pre_gripper_pose[1, 3], 0)
                                    pb.resetJointState(gripper._id, 2, pre_gripper_pose[2, 3], 0)

                                # first check grasp valid for gripper
                                if gripper.gripper_range[0] <= gg_cam_.width <= gripper.gripper_range[1]:
                                    # collect
                                    collect_two += 1
                                    # dynamics for grasping
                                    for all_link_id in range(pb.getNumJoints(object_model)):
                                        pb.changeDynamics(object_model, all_link_id, lateralFriction=50000)
                                    pb.setJointMotorControl2(object_model, link_id, pb.VELOCITY_CONTROL, force=1)
                                    # grasp
                                    if not args.direct:
                                        gripper.move_pose(target_gripper_pose[:3, 3].tolist() + rpy.tolist())
                                        has_contact = False
                                        move_done = False
                                        time_count = 0
                                        while not has_contact and not move_done:
                                            pb.stepSimulation()
                                            gripper_pose = get_link_pos(gripper._id, 6)
                                            has_contact = len(pb.getContactPoints(object_model, gripper._id)) > 0
                                            tr_diff, rot_diff = calculate_transformation_diff(gripper_pose, target_gripper_pose)
                                            move_done = (tr_diff < 0.001) and (rot_diff < 1 * np.pi / 180)      # 0.1cm and 1deg err
                                            time_count += 1
                                            if time_count > 100:
                                                break   # 100 steps timeout

                                            if args.allow:
                                                if has_contact and move_done:
                                                    # no need to move anymore
                                                    pass
                                                elif has_contact and not move_done:
                                                    # keep moving
                                                    joint_prime = pb.getJointState(object_model, link_id)[0]
                                                    while abs(joint_prime - real_states[j]) < 1 * np.pi / 180 and not move_done:
                                                        pb.stepSimulation()
                                                        gripper_pose = get_link_pos(gripper._id, 6)
                                                        tr_diff, rot_diff = calculate_transformation_diff(gripper_pose, target_gripper_pose)
                                                        move_done = (tr_diff < 0.001) and (rot_diff < 1 * np.pi / 180)
                                                        joint_prime = pb.getJointState(object_model, link_id)[0]
                                                        time_count += 1
                                                        if time_count > 100:
                                                            break
                                                elif not has_contact and move_done:
                                                    # deeper for grasping more easily
                                                    offset_dis = 0
                                                    while move_done:
                                                        offset_dis += 0.005
                                                        if offset_dis > 0.03:
                                                            # cannot move much deeper
                                                            break
                                                        offset = np.identity(4)
                                                        offset[:3, 3] = np.array([gg_cam_.depth - 0.035 * args.gripper_scale + offset_dis, 0,  0])
                                                        new_target_gripper_pose = extrinsic[camera_id] @ (grasp_pose_cam @ offset)
                                                        new_rpy = Rot.from_matrix(new_target_gripper_pose[:3, :3]).as_euler('ZYX', degrees=False)
                                                        gripper.move_pose(new_target_gripper_pose[:3, 3].tolist() + new_rpy.tolist())
                                                        for _ in range(10):
                                                            has_contact = len(pb.getContactPoints(object_model, gripper._id)) > 0
                                                            if has_contact:
                                                                break
                                                            pb.stepSimulation()
                                                        gripper_pose = get_link_pos(gripper._id, 6)
                                                        tr_diff, rot_diff = calculate_transformation_diff(gripper_pose, new_target_gripper_pose)
                                                        move_done = (tr_diff < 0.001) and (rot_diff < 1 * np.pi / 180)
                                                        if has_contact:
                                                            break
                                                    move_done = True
                                                else:
                                                    # normal case
                                                    pass
                                    else:
                                        before_gripper_pose = get_link_pos(gripper._id, 6)
                                        before_gripper_width = pb.getJointState(gripper._id, gripper.mimic_parent_id)[0] + \
                                                                pb.getJointState(gripper._id, gripper.mimic_parent_id+1)[0]
                                        for _ in range(10):
                                            pb.stepSimulation()
                                        after_gripper_pose = get_link_pos(gripper._id, 6)
                                        after_gripper_width = pb.getJointState(gripper._id, gripper.mimic_parent_id)[0] + \
                                                                pb.getJointState(gripper._id, gripper.mimic_parent_id+1)[0]
                                        tr_diff, rot_diff = calculate_transformation_diff(before_gripper_pose, after_gripper_pose)
                                        width_diff = abs(after_gripper_width - before_gripper_width)
                                        move_done = (tr_diff < 0.05) and (rot_diff < 5 * np.pi / 180) and (width_diff < 0.02)
                                    if args.pause:
                                        import pdb; pdb.set_trace()
                                    # second check grasp actually
                                    if move_done:
                                        # close gripper
                                        gripper.close_gripper(wait_done=False)
                                        for _ in range(10):
                                            pb.stepSimulation()
                                        # grasp_done = len(pb.getContactPoints(object_model, gripper._id)) >= 2 and \
                                        grasp_done = len(pb.getContactPoints(object_model, gripper._id)) > 0 and \
                                                    abs(pb.getJointState(object_model, link_id)[0] - real_states[j]) < 5 / 180.0 * np.pi
                                        grasp_done = grasp_done or (len(pb.getContactPoints(bodyA=object_model, bodyB=gripper._id, linkIndexA=link_id, linkIndexB=gripper.mimic_parent_id)) > 0 and \
                                                                    len(pb.getContactPoints(bodyA=object_model, bodyB=gripper._id, linkIndexA=link_id, linkIndexB=gripper.mimic_parent_id+1)) > 0)
                                        if grasp_done:
                                            # grasp
                                            grasp_two += 1
                                            if args.contact:
                                                # grasp_correct = len(pb.getContactPoints(bodyA=object_model, bodyB=gripper._id, linkIndexA=link_id, linkIndexB=gripper.mimic_parent_id)) > 0 and \
                                                #                 len(pb.getContactPoints(bodyA=object_model, bodyB=gripper._id, linkIndexA=link_id, linkIndexB=gripper.mimic_parent_id+1)) > 0
                                                grasp_correct = len(pb.getContactPoints(bodyA=object_model, bodyB=gripper._id, linkIndexA=link_id, linkIndexB=gripper.mimic_parent_id)) + \
                                                                len(pb.getContactPoints(bodyA=object_model, bodyB=gripper._id, linkIndexA=link_id, linkIndexB=gripper.mimic_parent_id+1)) == \
                                                                len(pb.getContactPoints(object_model, gripper._id))
                                                if grasp_correct:
                                                    # joint_state = pb.getJointState(object_model, link_id)[0]
                                                    joint_state = states[j]
                                                    if args.link_type[j] == 'revolute':
                                                        residual = abs(joint_state - 0) if grasp_orientation == 1 else abs(joint_state - limit_states[j])
                                                        if residual >= args.angle_threshold / 180.0 * np.pi:
                                                            success_two += 1
                                                        else:
                                                            pass
                                                    elif args.link_type[j] == 'prismatic':
                                                        residual = abs(joint_state - 0) if grasp_orientation == 1 else abs(joint_state - limit_states[j])
                                                        if residual >= args.distance_threshold / 100.0:
                                                            success_two += 1
                                                        else:
                                                            pass
                                                    else:
                                                        raise ValueError
                                                else:
                                                    pass
                                            else:
                                                # prepare move
                                                joint1 = pb.getJointState(object_model, link_id)[0]
                                                joint_poses_temp = get_joint_poses(object_model, args.link_id)
                                                
                                                if args.link_type[j] == 'revolute':
                                                    if args.link_axis[j] == 'x':
                                                        rotation_direction = joint_poses_temp[j][:3, 0]
                                                    elif args.link_axis[j] == 'y':
                                                        rotation_direction = joint_poses_temp[j][:3, 1]
                                                    elif args.link_axis[j] == 'z':
                                                        rotation_direction = joint_poses_temp[j][:3, 2]
                                                    else:
                                                        raise ValueError
                                                    rotation_point = joint_poses_temp[j][:3, 3]
                                                    rotation_angle = args.angle_threshold / 180.0 * np.pi * grasp_orientation
                                                    move_gripper_pose_delta = tf.rotation_matrix(angle=rotation_angle, 
                                                                                            direction=rotation_direction, point=rotation_point)
                                                elif args.link_type[j] == 'prismatic':
                                                    if args.link_axis[j] == 'x':
                                                        move_direction = joint_poses_temp[j][:3, 0]
                                                    elif args.link_axis[j] == 'y':
                                                        move_direction = joint_poses_temp[j][:3, 1]
                                                    elif args.link_axis[j] == 'z':
                                                        move_direction = joint_poses_temp[j][:3, 2]
                                                    else:
                                                        raise ValueError
                                                    move_distance = args.distance_threshold / 100.0 * grasp_orientation
                                                    move_gripper_pose_delta = tf.translation_matrix(move_direction * move_distance)
                                                else:
                                                    raise ValueError
                                                
                                                start_gripper_pose = gripper.gripper_pose(return_list=False)
                                                move_gripper_pose = move_gripper_pose_delta @ start_gripper_pose
                                                gripper.move_pose(
                                                    move_gripper_pose[:3, 3].tolist() + Rot.from_matrix(move_gripper_pose[:3, :3]).as_euler('ZYX',
                                                                                                                            degrees=False).tolist()
                                                )
                                                for _ in range(1000):
                                                    pb.stepSimulation()
                                                
                                                end_gripper_pose = gripper.gripper_pose(return_list=False)
                                                joint2 = pb.getJointState(object_model, link_id)[0]
                                                
                                                # check grasp success
                                                if args.link_type[j] == 'revolute':
                                                    if abs(joint2 - joint1) > args.angle_threshold / 180 * np.pi / 2 \
                                                        and len(pb.getContactPoints(object_model, gripper._id)) > 0:
                                                        # success
                                                        if success_two == 1:
                                                            if (joint2 - joint1 > 0) == last_orientation:   # may be pushing instead of grasping
                                                                pass
                                                            else:   # ensure both success need different orientation
                                                                success_two += 1
                                                        elif success_two == 0:
                                                            success_two += 1
                                                        else:
                                                            raise ValueError
                                                        last_orientation = joint2 - joint1 > 0
                                                    else:
                                                        pass
                                                elif args.link_type[j] == 'prismatic':
                                                    if abs(joint2 - joint1) > args.distance_threshold * 0.01 / 2 \
                                                        and len(pb.getContactPoints(object_model, gripper._id)) > 0:
                                                        # success
                                                        if success_two == 1:
                                                            if (joint2 - joint1 > 0) == last_orientation:
                                                                pass
                                                            else:
                                                                success_two += 1
                                                        elif success_two == 0:
                                                            success_two += 1
                                                        else:
                                                            raise ValueError
                                                        last_orientation = joint2 - joint1 > 0
                                                    else:
                                                        pass
                                                else:
                                                    raise ValueError
                                        else:
                                            pass
                                    else:
                                        pass
                                else:
                                    pass

                                # reset
                                del gripper
                                for k, state in enumerate(real_states):
                                    pb.resetJointState(object_model, args.link_id[k], state)
                            
                            if collect_two == 2:
                                if grasp_two == 2:
                                    if success_two == 2:
                                        collect_grasp.append(2)     # success both orientation, mainly because of space enough and tight grasp
                                        both_success_grasp_num += 1
                                        collect_gg_array.append(collect_grasp)
                                        drawn_ids = draw_gripper(collect_grasp[4], collect_grasp[5], collect_grasp[1], collect_grasp[3], extrinsic[camera_id], color=[0, 1, 0])
                                        drawn_items.extend(list(drawn_ids))
                                    elif success_two == 1:
                                        collect_grasp.append(1)     # success half orientation, mainly because of lack of space or loose grasp or unequal grasp or just bug
                                        half_success_grasp_num += 1
                                        collect_gg_array.append(collect_grasp)
                                        drawn_ids = draw_gripper(collect_grasp[4], collect_grasp[5], collect_grasp[1], collect_grasp[3], extrinsic[camera_id], color=[0, 0, 1])
                                        drawn_items.extend(list(drawn_ids))
                                    else:
                                        collect_grasp.append(0)     # success none orientation, mainly because of lack articulation semantics or grasp too loose
                                        none_success_grasp_num += 1
                                        collect_gg_array.append(collect_grasp)
                                        drawn_ids = draw_gripper(collect_grasp[4], collect_grasp[5], collect_grasp[1], collect_grasp[3], extrinsic[camera_id], color=[1, 1, 0])
                                        drawn_items.extend(list(drawn_ids))
                                elif grasp_two == 1:
                                    # in theory should not happen
                                    pass
                                else:
                                    collect_grasp.append(-1)        # grasp early fail, mainly because of collision due to partial observation or collision of pre-grasp
                                    fail_grasp_num += 1
                                    collect_gg_array.append(collect_grasp)
                                    drawn_ids = draw_gripper(collect_grasp[4], collect_grasp[5], collect_grasp[1], collect_grasp[3], extrinsic[camera_id], color=[1, 0, 0])
                                    drawn_items.extend(list(drawn_ids))
                            elif collect_two == 1:
                                # in theory should not happen
                                pass
                            else:
                                # we do not collect such grasp that cannot satisfy gripper range itself
                                pass
                    
                    # remove this camera view
                    for drawn_item in drawn_items:
                        pb.removeUserDebugItem(drawn_item, uid)
                else:
                    pass

            # save
            print('===>', i, camera_id)
            if args.grasp:
                success_grasp_num_whole = both_success_grasp_num + half_success_grasp_num / 2
                print('===>', len(gg_cam), len(args.link_id), 
                      str(fail_grasp_num) + ' + ' + str(none_success_grasp_num) + ' + ' + str(half_success_grasp_num) + ' + ' 
                      + str(both_success_grasp_num) + ' = ' + str(len(collect_gg_array)))
            if not args.grasp:
                np.savez(os.path.join(args.log_path, args.object, args.id, '%04d' % i, '%04d.npz' % camera_id),
                    joint_pose=joint_poses_cam, object_pose=base_pose_cam, point_cloud=point_cloud_cam)
            else:
                if args.contact:
                    if both_success_grasp_num + half_success_grasp_num + none_success_grasp_num > 0:
                        if both_success_grasp_num + half_success_grasp_num > 0:
                            np.savez(os.path.join(args.log_path, args.object, args.id, '%04d' % i, '%04d.npz' % camera_id),
                                joint_pose=joint_poses_cam, object_pose=base_pose_cam, point_cloud=point_cloud_cam, grasp=np.array(collect_gg_array, dtype=object))
                        else:
                            whether = np.random.randn() > 0
                            if whether:
                                np.savez(os.path.join(args.log_path, args.object, args.id, '%04d' % i, '%04d.npz' % camera_id),
                                    joint_pose=joint_poses_cam, object_pose=base_pose_cam, point_cloud=point_cloud_cam, grasp=np.array(collect_gg_array, dtype=object))
                            else:
                                pass
                    else:
                        pass
                else:
                    if success_grasp_num_whole > 1:
                        np.savez(os.path.join(args.log_path, args.object, args.id, '%04d' % i, '%04d.npz' % camera_id),
                            joint_pose=joint_poses_cam, object_pose=base_pose_cam, point_cloud=point_cloud_cam, grasp=np.array(collect_gg_array, dtype=object))
                    else:
                        pass
            
            # visualize
            if args.vis:
                visualize(point_cloud_cam, base_pose_cam, np.array(joint_poses_cam), args.link_axis, np.array(collect_gg_array, dtype=object) if (args.grasp and success_grasp_num_whole > 1) else None)
            
            # pause
            if args.pause:
                import pdb;pdb.set_trace()
