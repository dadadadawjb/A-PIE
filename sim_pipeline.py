from typing import List, Tuple, Optional
import configargparse
import time
import importlib
import os
import tqdm
import numpy as np
import pybullet as pb
import pybullet_data as pbd
import torch
import torch.nn as nn
import torch.optim as optim
import random
import omegaconf
import transformations as tf
from munch import DefaultMunch
import open3d as o3d

manipulation_control = importlib.import_module("ManiControl")
articulation_generation = importlib.import_module("articulation-generation")
assert articulation_generation.AnyGrasp is not None
beyondppf = importlib.import_module("BeyondPPF")
pointnet2 = importlib.import_module("BeyondPPF-baseline.pointnet2")
grasp_affordance = importlib.import_module("GraspAffordance")


def setup_seed(seed:int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def config_parse() -> configargparse.Namespace:
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True, help='config file path')
    # general config
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--object_path', type=str, help='model path')
    parser.add_argument('--gui', action='store_true', help='enable gui')
    parser.add_argument('--gt', action='store_true', help='whether to show ground truth')
    parser.add_argument('--test', action='store_true', help='whether to test instead of demo')
    parser.add_argument('--num_samples', type=int, help='number of samples, only valid when test')
    parser.add_argument('--log_path', type=str, help='save path, only valid when test')
    # object config
    parser.add_argument('--object_scale', type=float, help='object scale')
    parser.add_argument('--link_id', nargs='+', type=int, help='link id')
    parser.add_argument('--link_type', nargs='+', type=str, choices=['revolute', 'prismatic'], help='link type')
    parser.add_argument('--link_axis', nargs='+', type=str, choices=['x', 'y', 'z'], help='link axis')
    parser.add_argument('--link_offset', nargs='+', type=float, help='link offset, only valid when gt')
    # gripper config
    parser.add_argument('--gripper_scale', type=float, help='gripper scale')
    # camera config
    parser.add_argument('--auto_camera', action='store_true', help='whether to use auto camera, for random camera poses')
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
    # perception config
    parser.add_argument('--perception_model', type=str, choices=['BeyondPPF', 'PointNet++'], help='perception model')
    parser.add_argument('--perception_weight', type=str, help='the path to the perception weight')
    # imagination config
    parser.add_argument('--imagination_weight', type=str, help='the path to the imagination weight')
    # anygrasp config
    parser.add_argument('--gsnet_weight', type=str, help='the path to the gsnet weight')
    # manipulation config
    parser.add_argument('--affordance_threshold', type=float, help='affordance threshold for manipulation')
    parser.add_argument('--k', type=int, help='k tries')
    parser.add_argument('--t', type=float, help='t seconds for each try')
    parser.add_argument('--angle_threshold', type=float, help='angle threshold for manipulation success in deg')
    parser.add_argument('--distance_threshold', type=float, help='distance threshold for manipulation success in cm')
    parser.add_argument('--allow', action='store_true', help='whether to allow tiny move')
    parser.add_argument('--allow_distance', type=float, help='allow tiny move distance in cm for grasp')
    parser.add_argument('--allow_angle', type=float, help='allow tiny move angle in degree for grasp')
    parser.add_argument('--loop_mode', type=str, choices=['none', 'offline_joint', 'offline_trajectory', 'online_joint', 'online_trajectory'], help='loop mode from manipulation to perception')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # common config
    args = config_parse()
    setup_seed(args.seed)

    # pybullet setting
    if args.gui:
        uid = pb.connect(pb.GUI)
    else:
        uid = pb.connect(pb.DIRECT)
    pb.setAdditionalSearchPath(pbd.getDataPath())

    if args.test:
        num_samples = args.num_samples
        os.makedirs(os.path.join(args.log_path), exist_ok=True)
    else:
        num_samples = 1
    manipulation_num = 0
    success_num = 0
    for i in tqdm.trange(num_samples):
        pb.resetSimulation(physicsClientId=uid)

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
        gt_states = []
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
                            print('===> warning: >180 deg limit, use 180 deg limit instead')
                        elif (max_limit - min_limit) > 90/180*np.pi:
                            limit_state = np.pi/2
                            print('===> warning: >90 deg limit, use 90 deg limit instead')
                        else:
                            limit_state = max_limit - min_limit
                            print('===> warning: <90 deg limit, use itself instead')
                    else:
                        limit_state = np.pi
                        print('===> warning: 180 deg limit')
                else:
                    limit_state = np.pi/2
                closed = np.random.randn() > 0
                if closed:
                    angle = min_limit
                else:
                    angle = np.random.uniform(min_limit, min_limit + limit_state/2)
                pb.resetJointState(object_model, link_id, angle)
                pb.setJointMotorControl2(object_model, link_id, pb.VELOCITY_CONTROL, targetVelocity=0., force=1.)
                gt_states.append(angle)
            elif args.link_type[j] == 'prismatic':
                joint_info = pb.getJointInfo(object_model, link_id)
                assert joint_info[2] == pb.JOINT_PRISMATIC
                min_limit = joint_info[8]
                max_limit = joint_info[9]
                limit_state = max_limit - min_limit
                closed = np.random.randn() > 0
                if closed:
                    distance = min_limit
                else:
                    distance = np.random.uniform(min_limit, min_limit + limit_state/2)
                pb.resetJointState(object_model, link_id, distance)
                pb.setJointMotorControl2(object_model, link_id, pb.VELOCITY_CONTROL, targetVelocity=0., force=1.)
                gt_states.append(distance)
            else:
                raise ValueError
        for all_link_id in range(pb.getNumJoints(object_model)):
            pb.changeDynamics(object_model, all_link_id, lateralFriction=50000)
        if args.gt:
            gt_joint_poses = articulation_generation.get_joint_poses(object_model, args.link_id)
            for joint_id in range(len(gt_joint_poses)):
                if args.link_axis[joint_id] == 'x':
                    gt_link_axis = np.array([1, 0, 0])
                elif args.link_axis[joint_id] == 'y':
                    gt_link_axis = np.array([0, 1, 0])
                elif args.link_axis[joint_id] == 'z':
                    gt_link_axis = np.array([0, 0, 1])
                else:
                    raise ValueError
                gt_link_axis = np.dot(gt_joint_poses[joint_id][:3, :3], gt_link_axis)
                gt_joint_poses[joint_id][:3, 3] += gt_link_axis * args.link_offset[joint_id] * args.object_scale
        
        # load gripper
        gripper = manipulation_control.PandaGripper(uid=uid, gravity=False, useFixedBase=False, globalScaling=args.gripper_scale)

        # set camera
        if args.auto_camera:
            view_matrix, proj_matrix, extrinsic = articulation_generation.generate_camera(1, object_model, args.auto_camera_distance_min, args.auto_camera_distance_max, 
                                                                args.auto_camera_fov, args.auto_camera_width, args.auto_camera_height, args.auto_camera_near, args.auto_camera_far, 
                                                                np.array(args.auto_camera_cone_direction), args.auto_camera_cone_angle/180 * np.pi, np.array(args.auto_camera_up_axis))
        else:
            _, view_matrix, proj_matrix, extrinsic = articulation_generation.get_camera(args.camera_target_x, args.camera_target_y, args.camera_target_z, 
                                                                args.camera_distance, args.camera_yaw, args.camera_pitch, args.camera_roll, args.camera_up,
                                                                args.camera_fov, args.camera_near, args.camera_far, args.camera_height, args.camera_width)
        view_matrix = view_matrix[0]
        proj_matrix = proj_matrix[0]
        extrinsic = extrinsic[0]
        articulation_generation.draw_link_coord(pose=extrinsic)

        # move gripper to camera, camera frame as (x, y, z) = (right, down, in), gripper frame as (x, y, z) = (up, right, in)
        initial_gripper_pose = np.copy(extrinsic)
        initial_gripper_pose[:3, 0] = -extrinsic[:3, 1]
        initial_gripper_pose[:3, 1] = extrinsic[:3, 0]
        initial_gripper_pose[:3, 2] = extrinsic[:3, 2]
        ee2gripper = np.identity(4)
        ee2gripper[2, 3] = 0.105 * args.gripper_scale
        initial_gripper_pose = initial_gripper_pose @ np.linalg.inv(ee2gripper)
        pb.resetBasePositionAndOrientation(gripper._id, initial_gripper_pose[:3, 3], pb.getQuaternionFromEuler(tf.euler_from_matrix(initial_gripper_pose)))
        
        # load perception
        if i == 0:
            if args.perception_model == 'BeyondPPF':
                perception_cfg = omegaconf.OmegaConf.load(f"{args.perception_weight}/.hydra/config.yaml")
                shot_encoder = beyondppf.create_shot_encoder(perception_cfg)
                encoder = beyondppf.create_encoder(perception_cfg)
                shot_encoder.load_state_dict(torch.load(f'{args.perception_weight}/shot_encoder_latest.pth'))
                encoder.load_state_dict(torch.load(f'{args.perception_weight}/encoder_latest.pth'))
                if args.loop_mode == 'none' or args.loop_mode == 'offline_joint' or args.loop_mode == 'offline_trajectory':
                    shot_encoder.eval()
                    encoder.eval()
                elif args.loop_mode == 'online_joint' or args.loop_mode == 'online_trajectory':
                    raise NotImplementedError
                else:
                    raise NotImplementedError
            elif args.perception_model == 'PointNet++':
                perception_cfg = omegaconf.OmegaConf.load(f"{args.perception_weight}/.hydra/config.yaml")
                regressor = pointnet2.PointNet2(perception_cfg).cuda(perception_cfg.device)
                regressor.apply(pointnet2.inplace_relu)
                if args.loop_mode == 'none' or args.loop_mode == 'offline_joint' or args.loop_mode == 'offline_trajectory':
                    if os.path.exists(f'{args.perception_weight}/pointnet2_loop.pth'):
                        regressor.load_state_dict(torch.load(f'{args.perception_weight}/pointnet2_loop.pth'))
                    else:
                        regressor.load_state_dict(torch.load(f'{args.perception_weight}/pointnet2_latest.pth'))
                    regressor.eval()
                elif args.loop_mode == 'online_joint':
                    if os.path.exists(f'{args.perception_weight}/pointnet2_loop.pth'):
                        regressor.load_state_dict(torch.load(f'{args.perception_weight}/pointnet2_loop.pth'))
                    else:
                        regressor.load_state_dict(torch.load(f'{args.perception_weight}/pointnet2_latest.pth'))
                    regressor.train()
                    # freeze batchnorm and dropout
                    for m in regressor.modules():
                        if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Dropout):
                            m.eval()
                    
                    tr_criterion = nn.MSELoss()
                    axis_criterion = nn.MSELoss()
                    if perception_cfg.pointnet.state:
                        state_criterion = nn.MSELoss()
                    perception_opt = optim.Adam(regressor.parameters(), lr=perception_cfg.lr/100, weight_decay=perception_cfg.weight_decay)
                    perception_opt.zero_grad()
                elif args.loop_mode == 'online_trajectory':
                    regressor.train()
                    # freeze batchnorm and dropout
                    for m in regressor.modules():
                        if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Dropout):
                            m.eval()
                    
                    traj_criterion = nn.MSELoss()
                    perception_opt = optim.Adam(regressor.parameters(), lr=perception_cfg.lr/100, weight_decay=perception_cfg.weight_decay)
                    perception_opt.zero_grad()
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

        # load imagination
        if i == 0:
            imagination_cfg = omegaconf.OmegaConf.load(f"{args.imagination_weight}/.hydra/config.yaml")
            model = grasp_affordance.grasp_embedding_network(imagination_cfg).to(imagination_cfg.device)
            model.load_state_dict(torch.load(f'{args.imagination_weight}/model_latest.pth'))
            model.eval()

        # load execution
        controller = manipulation_control.PIDController(gripper)

        # load anygrasp, belong to imagination
        if i == 0:
            grasp_detector_cfg = {
                'checkpoint_path': args.gsnet_weight,
                'max_gripper_width': gripper.gripper_range[1] * 2,
                'gripper_height': 0.03,
                'top_down_grasp': False,
                'add_vdistance': True
            }
            grasp_detector_cfg = DefaultMunch.fromDict(grasp_detector_cfg)
            grasp_detector = articulation_generation.AnyGrasp(grasp_detector_cfg)
            grasp_detector.load_net()

        if not args.test:
            print("===> prepare done")
            import pdb; pdb.set_trace()

        # get point cloud
        if args.auto_camera:
            point_cloud = articulation_generation.get_point_cloud(args.auto_camera_width, args.auto_camera_height, view_matrix, proj_matrix, obj_id=0)
        else:
            point_cloud = articulation_generation.get_point_cloud(args.camera_width[0], args.camera_height[0], view_matrix, proj_matrix, obj_id=0)
        view_matrix_inv = np.linalg.inv(extrinsic)
        point_cloud_cam = (view_matrix_inv[:3, :3] @ point_cloud.T + view_matrix_inv[:3, 3].reshape(3, 1)).T

        # estimate joint pose
        if args.perception_model == 'BeyondPPF':
            start_time = time.time()
            results = beyondppf.inference(np.copy(point_cloud_cam), perception_cfg.shot.res, perception_cfg.shot.receptive_field, perception_cfg.test_samples, perception_cfg.test_sample_points, perception_cfg.num_more, 
                    perception_cfg.encoder.rot_num_bins, perception_cfg.encoder.state, perception_cfg.types, perception_cfg.states, perception_cfg.topk, shot_encoder, encoder, perception_cfg.device, False, None)
            if perception_cfg.encoder.state:
                translations_cam, directions_cam, states = results
            else:
                translations_cam, directions_cam = results
            end_time = time.time()
            translations = (extrinsic[:3, :3] @ translations_cam.T + extrinsic[:3, 3].reshape(3, 1)).T
            directions = (extrinsic[:3, :3] @ directions_cam.T).T
            if not args.test:
                print('estimated translations:', translations)
                print('estimated directions:', directions)
                if perception_cfg.encoder.state:
                    print('estimated states:', states)
            if args.gt:
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
                tr_errs, tr_along_errs, tr_perp_errs, tr_plane_errs = [], [], [], []
                rot_errs = []
                if perception_cfg.encoder.state:
                    state_errs = []
                for j in range(translations.shape[0]):
                    gt_translation = gt_joint_poses[j][:3, 3]
                    if args.link_axis[joint_id] == 'x':
                        gt_direction = gt_joint_poses[j][:3, 0]
                    elif args.link_axis[joint_id] == 'y':
                        gt_direction = gt_joint_poses[j][:3, 1]
                    elif args.link_axis[joint_id] == 'z':
                        gt_direction = gt_joint_poses[j][:3, 2]
                    else:
                        raise ValueError
                    tr_err = np.linalg.norm(translations[j] - gt_translation) * 100
                    tr_along_err = abs(np.dot(translations[j] - gt_translation, gt_direction)) * 100
                    tr_perp_err = np.sqrt(tr_err**2 - tr_along_err**2)
                    tr_plane_err = calculate_plane_err(translations[j], directions[j], gt_translation, gt_direction) * 100
                    rot_err = np.arccos(np.dot(directions[j], gt_direction)) / np.pi * 180
                    if perception_cfg.encoder.state:
                        state_err = abs(states[j] - gt_states[j])
                    tr_errs.append(tr_err)
                    tr_along_errs.append(tr_along_err)
                    tr_perp_errs.append(tr_perp_err)
                    tr_plane_errs.append(tr_plane_err)
                    rot_errs.append(rot_err)
                    if perception_cfg.encoder.state:
                        state_errs.append(state_err)
                if not args.test:
                    print('estimated translation errors:', np.array(tr_errs))
                    print('estimated translation along errors:', np.array(tr_along_errs))
                    print('estimated translation perpendicular errors:', np.array(tr_perp_errs))
                    print('estimated translation plane errors:', np.array(tr_plane_errs))
                    print('estimated rotation errors:', np.array(rot_errs))
                    if perception_cfg.encoder.state:
                        print('estimated state errors:', np.array(state_errs))
            if not args.test:
                print('estimation time:', end_time - start_time)
        elif args.perception_model == 'PointNet++':
            start_time = time.time()
            if args.loop_mode == 'none' or args.loop_mode == 'offline_joint' or args.loop_mode == 'offline_trajectory':
                results = pointnet2.inference(np.copy(point_cloud_cam), perception_cfg.pointnet.normal_channel, perception_cfg.pointnet.state, perception_cfg.types, perception_cfg.states, 
                                                    perception_cfg.shot.res, perception_cfg.shot.receptive_field, perception_cfg.test_sample_points, regressor, perception_cfg.device, False, None)
                if perception_cfg.pointnet.state:
                    translations_cam, directions_cam, states = results
                else:
                    translations_cam, directions_cam = results
                translations = (extrinsic[:3, :3] @ translations_cam.T + extrinsic[:3, 3].reshape(3, 1)).T
                directions = (extrinsic[:3, :3] @ directions_cam.T).T
            elif args.loop_mode == 'online_joint' or args.loop_mode == 'online_trajectory':
                results = pointnet2.grad_inference(np.copy(point_cloud_cam), perception_cfg.pointnet.normal_channel, perception_cfg.pointnet.state, perception_cfg.types, perception_cfg.states, 
                                                    perception_cfg.shot.res, perception_cfg.shot.receptive_field, perception_cfg.test_sample_points, regressor, perception_cfg.device, False, None)
                if perception_cfg.pointnet.state:
                    translations_cam_tensor, directions_cam_tensor, states_tensor, preds_tr, preds_axis, preds_state, joint_offset, joint_scale = results
                    translations_cam = translations_cam_tensor.cpu().detach().numpy()
                    directions_cam = directions_cam_tensor.cpu().detach().numpy()
                    states = states_tensor.cpu().detach().numpy()
                else:
                    translations_cam_tensor, directions_cam_tensor, preds_tr, preds_axis, joint_offset, joint_scale = results
                    translations_cam = translations_cam_tensor.cpu().detach().numpy()
                    directions_cam = directions_cam_tensor.cpu().detach().numpy()
                extrinsic_tensor = torch.from_numpy(extrinsic).cuda(translations_cam_tensor.device).to(translations_cam_tensor.dtype)
                translations_tensor = (extrinsic_tensor[:3, :3] @ translations_cam_tensor.T + extrinsic_tensor[:3, 3].reshape(3, 1)).T
                directions_tensor = (extrinsic_tensor[:3, :3] @ directions_cam_tensor.T).T
                translations = translations_tensor.cpu().detach().numpy()
                directions = directions_tensor.cpu().detach().numpy()
            else:
                raise NotImplementedError
            end_time = time.time()
            if not args.test:
                print('estimated translations:', translations)
                print('estimated directions:', directions)
                if perception_cfg.pointnet.state:
                    print('estimated states:', states)
            if args.gt:
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
                tr_errs, tr_along_errs, tr_perp_errs, tr_plane_errs = [], [], [], []
                rot_errs = []
                if perception_cfg.pointnet.state:
                    state_errs = []
                for j in range(translations.shape[0]):
                    gt_translation = gt_joint_poses[j][:3, 3]
                    if args.link_axis[joint_id] == 'x':
                        gt_direction = gt_joint_poses[j][:3, 0]
                    elif args.link_axis[joint_id] == 'y':
                        gt_direction = gt_joint_poses[j][:3, 1]
                    elif args.link_axis[joint_id] == 'z':
                        gt_direction = gt_joint_poses[j][:3, 2]
                    else:
                        raise ValueError
                    tr_err = np.linalg.norm(translations[j] - gt_translation) * 100
                    tr_along_err = abs(np.dot(translations[j] - gt_translation, gt_direction)) * 100
                    tr_perp_err = np.sqrt(tr_err**2 - tr_along_err**2)
                    tr_plane_err = calculate_plane_err(translations[j], directions[j], gt_translation, gt_direction) * 100
                    rot_err = np.arccos(np.dot(directions[j], gt_direction)) / np.pi * 180
                    if perception_cfg.pointnet.state:
                        state_err = abs(states[j] - gt_states[j])
                    tr_errs.append(tr_err)
                    tr_along_errs.append(tr_along_err)
                    tr_perp_errs.append(tr_perp_err)
                    tr_plane_errs.append(tr_plane_err)
                    rot_errs.append(rot_err)
                    if perception_cfg.pointnet.state:
                        state_errs.append(state_err)
                if not args.test:
                    print('estimated translation errors:', np.array(tr_errs))
                    print('estimated translation along errors:', np.array(tr_along_errs))
                    print('estimated translation perpendicular errors:', np.array(tr_perp_errs))
                    print('estimated translation plane errors:', np.array(tr_plane_errs))
                    print('estimated rotation errors:', np.array(rot_errs))
                    if perception_cfg.pointnet.state:
                        print('estimated state errors:', np.array(state_errs))
            if not args.test:
                print('estimation time:', end_time - start_time)
        else:
            raise NotImplementedError
        
        # draw estimated joint
        for j in range(translations.shape[0]):
            joint_pos = pb.createVisualShape(pb.GEOM_SPHERE, radius=0.015)
            joint_pos = pb.createMultiBody(baseMass=0, baseVisualShapeIndex=joint_pos, basePosition=translations[j])
            pb.changeVisualShape(joint_pos, -1, rgbaColor=[0.5, 0.5, 0.5, 1])
            joint_axis = pb.addUserDebugLine(translations[j], translations[j] + 0.3 * directions[j], [0.5, 0.5, 0.5], lineWidth=5)

            if args.gt:
                gt_translation = gt_joint_poses[j][:3, 3]
                if args.link_axis[joint_id] == 'x':
                    gt_direction = gt_joint_poses[j][:3, 0]
                elif args.link_axis[joint_id] == 'y':
                    gt_direction = gt_joint_poses[j][:3, 1]
                elif args.link_axis[joint_id] == 'z':
                    gt_direction = gt_joint_poses[j][:3, 2]
                else:
                    raise ValueError
                gt_joint_pos = pb.createVisualShape(pb.GEOM_SPHERE, radius=0.015)
                gt_joint_pos = pb.createMultiBody(baseMass=0, baseVisualShapeIndex=gt_joint_pos, basePosition=gt_translation)
                pb.changeVisualShape(gt_joint_pos, -1, rgbaColor=[0, 0, 1, 1])
                gt_joint_axis = pb.addUserDebugLine(gt_translation, gt_translation + 0.3 * gt_direction, [0, 0, 1], lineWidth=5)

        if not args.test:
            print("===> perception done")
            import pdb; pdb.set_trace()

        # get grasps
        lims = [-2, 2, -2, 2, -2, 2]
        gg_cloud_cam = grasp_detector.get_grasp(point_cloud_cam.astype(np.float32), colors=None, lims=lims)
        if gg_cloud_cam is None:
            gg_cam = []
        else:
            if len(gg_cloud_cam) != 2:
                gg_cam = []
            else:
                gg_cam, _ = gg_cloud_cam
        if len(gg_cam) > 0:
            gg_cam.nms()
            gg_cam.sort_by_score()
            gg_cam = gg_cam[:100]
        grasps_cam_ = []
        for gg_cam_ in gg_cam:
            grasps_cam_.append([gg_cam_.score, gg_cam_.width, gg_cam_.height, gg_cam_.depth, 
                            np.copy(gg_cam_.translation), np.copy(gg_cam_.rotation_matrix)])
        if len(grasps_cam_) == 0:
            continue
        grasps_cam = np.array(grasps_cam_, dtype=object)
        
        # estimate grasp affordance
        joint_feat_cam_ = []
        for j in range(translations_cam.shape[0]):
            translation_cam = translations_cam[j]
            rotation_cam = directions_cam[j]
            assert args.link_type[j] == imagination_cfg.types[j]
            if args.link_type[j] == 'revolute':
                type_feat = 0
                if imagination_cfg.joint_encoder.state_channel:
                    state_feat = states[j] - imagination_cfg.states[j] / 180.0 * np.pi / 2
                else:
                    state_feat = 0
            elif args.link_type[j] == 'prismatic':
                type_feat = 1
                if imagination_cfg.joint_encoder.state_channel:
                    state_feat = states[j] - imagination_cfg.states[j] / 100.0 / 2
                else:
                    state_feat = 0
            else:
                raise ValueError('Invalid joint_type: {}'.format(args.link_type[j]))
            joint_feat_cam_.append(np.concatenate([translation_cam, rotation_cam, [type_feat, state_feat]]))
        joints_feat_cam = np.stack(joint_feat_cam_, axis=0)         # (J, 8)
        if not imagination_cfg.joint_encoder.state_channel:
            joints_feat_cam = joints_feat_cam[:, :-1]               # (J, 7/8)
        grasps_feat_cam = np.concatenate([grasps_cam[:, 0:4], np.stack(grasps_cam[:, 4], axis=0), 
                                        np.stack(grasps_cam[:, 5], axis=0).reshape((-1, 9))], axis=-1)    # (G, 16)
        
        start_time = time.time()
        affordances = grasp_affordance.inference(imagination_cfg.embedding_net.classification, np.copy(point_cloud_cam), imagination_cfg.point_encoder.normal_channel, 
                                                np.copy(joints_feat_cam), np.copy(grasps_feat_cam), imagination_cfg.shot.res, imagination_cfg.shot.receptive_field, 
                                                imagination_cfg.test_samples, imagination_cfg.normalization, model, imagination_cfg.device, False, None)
        if imagination_cfg.embedding_net.classification:
            affordances_ = np.zeros(affordances.shape[:-1], dtype=np.float32)
            for c, level in enumerate(imagination_cfg.embedding_net.levels):
                affordances_ += affordances[..., c] * level[1]
            affordances = affordances_
        end_time = time.time()
        if not args.test:
            print('estimated affordances:', affordances)
            print('estimation time:', end_time - start_time)

        if args.allow:
            # build point cloud search tree in advance
            pcd_cam = o3d.geometry.PointCloud()
            pcd_cam.points = o3d.utility.Vector3dVector(point_cloud_cam)
            pcd_tree_cam = o3d.geometry.KDTreeFlann(pcd_cam)
            # estimate point cloud normals in advance
            pcd_cam.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(100))
        
        # for each joint
        if args.loop_mode == 'offline_joint' or args.loop_mode == 'online_joint':
            fit_joints = []
            tried_grasps = []
            tried_affordances = []
        elif args.loop_mode == 'offline_trajectory' or args.loop_mode == 'online_trajectory':
            trajectories = []
            tried_grasps = []
            tried_affordances = []
        elif args.loop_mode == 'none':
            pass
        else:
            raise NotImplementedError
        for j in range(joints_feat_cam.shape[0]):
            # draw estimated grasps
            def draw_grasp(grasp:np.ndarray, affordance:float) -> List[int]:
                finger_width = 0.004
                tail_length = 0.04
                depth_base = 0.02
                gg_width = grasp[1]
                gg_depth = grasp[3]
                gg_translation = grasp[4]
                gg_rotation = grasp[5]

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

                if affordance < 0.5:
                    color = [1, 2*affordance, 0]
                else:
                    color = [-2*affordance+2, 1, 0]
                left_id = pb.addUserDebugLine(vertices[0], vertices[1], color, lineWidth=1)
                right_id = pb.addUserDebugLine(vertices[2], vertices[3], color, lineWidth=1)
                bottom_id = pb.addUserDebugLine(vertices[4], vertices[5], color, lineWidth=1)
                tail_id = pb.addUserDebugLine(vertices[6], vertices[7], color, lineWidth=1)
                return [left_id, right_id, bottom_id, tail_id]
            affordance = affordances[:, j]
            grasps_world = []
            drawn_grasp_ids = []
            for g in range(grasps_feat_cam.shape[0]):
                if args.allow:
                    # translate grasp pose to the nearest point
                    [k, idx, _] = pcd_tree_cam.search_knn_vector_3d(grasps_cam[g][4], 1)
                    nearest_point = np.asarray(pcd_cam.points)[idx, :].squeeze(axis=0)
                    if np.linalg.norm(nearest_point - grasps_cam[g][4]) < args.allow_distance / 100.0:
                        grasps_cam[g][4] = nearest_point
                    
                    # rotate grasp pose to the nearest point's normal
                    normal = np.asarray(pcd_cam.normals)[idx, :].squeeze(axis=0)
                    if abs(np.dot(normal, grasps_cam[g][5][:, 0])) > np.cos(args.allow_angle / 180.0 * np.pi):
                        grasps_cam[g][5][:, 0] = normal if np.dot(normal, grasps_cam[g][5][:, 0]) > 0 else (-1 * normal)
                        grasps_cam[g][5][:, 1] = np.cross(grasps_cam[g][5][:, 2], grasps_cam[g][5][:, 0])
                        grasps_cam[g][5][:, 2] = np.cross(grasps_cam[g][5][:, 0], grasps_cam[g][5][:, 1])
                    
                grasp_world = np.copy(grasps_cam[g])
                grasp_world[4] = extrinsic[:3, :3] @ grasp_world[4] + extrinsic[:3, 3]
                grasp_world[5] = extrinsic[:3, :3] @ grasp_world[5]
                grasps_world.append(grasp_world)

                grasp_ids = draw_grasp(grasp_world, affordance[g])
                drawn_grasp_ids.append(grasp_ids)
            grasps_world = np.array(grasps_world, dtype=object)

            if not args.test:
                print("===> imagination done for joint {}".format(j))
                import pdb; pdb.set_trace()

            # select top k grasps within gripper range to try
            valid_grasps_mask = np.logical_and(grasps_feat_cam[:, 1] >= 2*gripper.gripper_range[0], grasps_feat_cam[:, 1] <= 2*gripper.gripper_range[1])
            grasps_world = grasps_world[valid_grasps_mask]
            affordance = affordance[valid_grasps_mask]
            for g in range(valid_grasps_mask.shape[0]):
                if not valid_grasps_mask[g]:
                    for grasp_id in drawn_grasp_ids[g]:
                        pb.removeUserDebugItem(grasp_id)
            drawn_grasp_ids = [drawn_grasp_ids[g] for g in range(valid_grasps_mask.shape[0]) if valid_grasps_mask[g]]
            top_grasp_ids = np.argsort(affordance)[::-1][:args.k]
            non_top_grasp_ids = np.argsort(affordance)[::-1][args.k:]
            grasps_world = grasps_world[top_grasp_ids]
            affordance = affordance[top_grasp_ids]
            for g in non_top_grasp_ids:
                for grasp_id in drawn_grasp_ids[g]:
                    pb.removeUserDebugItem(grasp_id)
            drawn_grasp_ids = [drawn_grasp_ids[g] for g in top_grasp_ids]
            if (affordance >= args.affordance_threshold).any():
                pass
            else:
                print("===> no grasp affordable found for joint {}".format(j))
                if args.loop_mode == 'offline_joint' or args.loop_mode == 'online_joint':
                    fit_joints.append(None)
                    tried_grasps.append(None)
                    tried_affordances.append(None)
                elif args.loop_mode == 'offline_trajectory' or args.loop_mode == 'online_trajectory':
                    trajectories.append(None)
                    tried_grasps.append(None)
                    tried_affordances.append(None)
                elif args.loop_mode == 'none':
                    pass
                else:
                    raise NotImplementedError
                for grasp_id in drawn_grasp_ids:
                    for _drawn_id in grasp_id:
                        pb.removeUserDebugItem(_drawn_id)
                continue
            success_until = -1
            for g, grasp_world in enumerate(grasps_world):
                # move gripper to grasp, grasp frame as (x, y, z) = (in, right, down), gripper frame as (x, y, z) = (up, right, in)
                grasp_pose = np.eye(4)
                grasp_pose[:3, 0] = -grasp_world[5][:3, 2]
                grasp_pose[:3, 1] = grasp_world[5][:3, 1]
                grasp_pose[:3, 2] = grasp_world[5][:3, 0]
                grasp_pose[:3, 3] = grasp_world[4]
                ee2gripper = np.identity(4)
                ee2gripper[2, 3] = 0.105 * args.gripper_scale
                grasp_pose = grasp_pose @ np.linalg.inv(ee2gripper)
                
                pb.resetJointState(gripper._id, 1, grasp_world[1]/2)
                pb.resetJointState(gripper._id, 2, grasp_world[1]/2)
                pb.resetBasePositionAndOrientation(gripper._id, grasp_pose[:3, 3], pb.getQuaternionFromEuler(tf.euler_from_matrix(grasp_pose)))
                gripper.close_gripper()
                joint1 = pb.getJointState(object_model, args.link_id[j])[0]

                # manipulation starts
                if args.loop_mode == 'offline_joint' or args.loop_mode == 'offline_trajectory' or args.loop_mode == 'online_joint' or args.loop_mode == 'online_trajectory':
                    current_joint_state = joint1
                    ee_pose = articulation_generation.get_base_pose(gripper._id)
                    trajectory = [(ee_pose @ ee2gripper)[:3, 3]]
                elif args.loop_mode == 'none':
                    pass
                else:
                    raise NotImplementedError
                controller.start_controller_thread()
                start_time = time.time()
                while time.time() - start_time < args.t:
                    ee_pose = articulation_generation.get_base_pose(gripper._id)
                    if args.link_type[j] == 'revolute':
                        if args.loop_mode == 'offline_joint' or args.loop_mode == 'offline_trajectory' or args.loop_mode == 'online_joint' or args.loop_mode == 'online_trajectory':
                            current_joint_state_prime = pb.getJointState(object_model, args.link_id[j])[0]
                            if abs(current_joint_state_prime - current_joint_state) >= 5.0 / 180.0 * np.pi:
                                trajectory.append((ee_pose @ ee2gripper)[:3, 3])
                                current_joint_state = current_joint_state_prime
                        elif args.loop_mode == 'none':
                            pass
                        else:
                            raise NotImplementedError
                        # TODO: for our cases, y up axis uses negative to open, x right axis uses positive to open, x left axis uses negative to open
                        # if j == 0:
                        #     rotation_angle = 1.0 / 180.0 * np.pi
                        # elif j == 1:
                        #     rotation_angle = -1.0 / 180.0 * np.pi
                        # elif j == 2:
                        #     rotation_angle = -1.0 / 180.0 * np.pi
                        # elif j == 3:
                        #     rotation_angle = 1.0 / 180.0 * np.pi
                        rotation_angle = -1.0 / 180.0 * np.pi
                        delta_pose = tf.rotation_matrix(angle=rotation_angle, direction=directions[j], point=translations[j])
                    elif args.link_type[j] == 'prismatic':
                        if args.loop_mode == 'offline_joint' or args.loop_mode == 'offline_trajectory' or args.loop_mode == 'online_joint' or args.loop_mode == 'online_trajectory':
                            current_joint_state_prime = pb.getJointState(object_model, args.link_id[j])[0]
                            if abs(current_joint_state_prime - current_joint_state) >= 1 / 100.0:
                                trajectory.append((ee_pose @ ee2gripper)[:3, 3])
                                current_joint_state = current_joint_state_prime
                        elif args.loop_mode == 'none':
                            pass
                        else:
                            raise NotImplementedError
                        translation_distance = 0.3 / 100.0
                        delta_pose = tf.translation_matrix(directions[j] * translation_distance)
                    else:
                        raise ValueError('Invalid joint_type: {}'.format(args.link_type[j]))
                    ee_pose_next = delta_pose @ ee_pose
                    goal_tr = ee_pose_next[:3, 3]
                    goal_ori = tf.quaternion_from_matrix(ee_pose_next)
                    goal_ori = np.quaternion(goal_ori[0], goal_ori[1], goal_ori[2], goal_ori[3])
                    controller.update_goal(goal_tr, goal_ori)
                controller.stop_controller_thread()
                
                # check success
                joint2 = pb.getJointState(object_model, args.link_id[j])[0]
                if args.link_type[j] == 'revolute':
                    if not args.test:
                        print('manipulation:', abs(joint1 - joint2) / np.pi * 180.0, 'deg')
                    if abs(joint1 - joint2) / np.pi * 180.0 >= args.angle_threshold:
                        success_until = g
                        break
                elif args.link_type[j] == 'prismatic':
                    if not args.test:
                        print('manipulation:', abs(joint1 - joint2) * 100.0, 'cm')
                    if abs(joint1 - joint2) * 100.0 >= args.distance_threshold:
                        success_until = g
                        break
                else:
                    raise ValueError('Invalid joint_type: {}'.format(args.link_type[j]))

            if success_until >= 0:
                manipulation_num += 1
                success_num += 1
                if not args.test:
                    print("===> execution done and success for joint {}".format(j))

                if args.loop_mode == 'none':
                    pass
                elif args.loop_mode == 'offline_joint' or args.loop_mode == 'online_joint':
                    pb.addUserDebugPoints(trajectory, [[0, 1, 0] for _ in trajectory], pointSize=2.0)

                    if args.link_type[j] == 'revolute':
                        fit_joint = manipulation_control.fit_arc(np.array(trajectory))
                    elif args.link_type[j] == 'prismatic':
                        fit_joint = manipulation_control.fit_line(np.array(trajectory))
                    else:
                        raise ValueError('Invalid joint_type: {}'.format(args.link_type[j]))
                    fit_joint = manipulation_control.refit(fit_joint[0], fit_joint[1], translations[j], directions[j])
                    fit_joints.append(fit_joint)
                    if args.gt:
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
                        gt_translation = gt_joint_poses[j][:3, 3]
                        if args.link_axis[joint_id] == 'x':
                            gt_direction = gt_joint_poses[j][:3, 0]
                        elif args.link_axis[joint_id] == 'y':
                            gt_direction = gt_joint_poses[j][:3, 1]
                        elif args.link_axis[joint_id] == 'z':
                            gt_direction = gt_joint_poses[j][:3, 2]
                        else:
                            raise ValueError
                        tr_err = np.linalg.norm(fit_joint[0] - gt_translation) * 100
                        tr_along_err = abs(np.dot(fit_joint[0] - gt_translation, gt_direction)) * 100
                        tr_perp_err = np.sqrt(tr_err**2 - tr_along_err**2)
                        tr_plane_err = calculate_plane_err(fit_joint[0], fit_joint[1], gt_translation, gt_direction) * 100
                        rot_err = np.arccos(np.dot(fit_joint[1], gt_direction)) / np.pi * 180
                        if not args.test:
                            print('fit translation error:', tr_err)
                            print('fit translation along error:', tr_along_err)
                            print('fit translation perpendicular error:', tr_perp_err)
                            print('fit translation plane error:', tr_plane_err)
                            print('fit rotation error:', rot_err)

                    fit_joint_pos = pb.createVisualShape(pb.GEOM_SPHERE, radius=0.015)
                    fit_joint_pos = pb.createMultiBody(baseMass=0, baseVisualShapeIndex=fit_joint_pos, basePosition=fit_joint[0])
                    pb.changeVisualShape(fit_joint_pos, -1, rgbaColor=[0, 1, 0, 1])
                    pb.addUserDebugLine(fit_joint[0], fit_joint[0] + 0.3 * fit_joint[1], [0, 1, 0], lineWidth=5)

                    tried_grasps.append([grasps_world[g] for g in range(success_until + 1)])
                    tried_affordances.append([0 for _ in range(success_until)] + [1])
                elif args.loop_mode == 'offline_trajectory' or args.loop_mode == 'online_trajectory':
                    pb.addUserDebugPoints(trajectory, [[0, 1, 0] for _ in trajectory], pointSize=2.0)

                    trajectories.append(np.vstack(trajectory))

                    tried_grasps.append([grasps_world[g] for g in range(success_until + 1)])
                    tried_affordances.append([0 for _ in range(success_until)] + [1])
                else:
                    raise NotImplementedError
            else:
                manipulation_num += 1
                if not args.test:
                    print("===> execution done and fail for joint {}".format(j))

                if args.loop_mode == 'none':
                    pass
                elif args.loop_mode == 'offline_joint' or args.loop_mode == 'online_joint':
                    fit_joints.append(None)

                    tried_grasps.append([grasps_world[g] for g in range(grasps_world.shape[0])])
                    tried_affordances.append([0 for _ in range(grasps_world.shape[0])])
                elif args.loop_mode == 'offline_trajectory' or args.loop_mode == 'online_trajectory':
                    trajectories.append(None)

                    tried_grasps.append([grasps_world[g] for g in range(grasps_world.shape[0])])
                    tried_affordances.append([0 for _ in range(grasps_world.shape[0])])
                else:
                    raise NotImplementedError
            
            pb.resetBasePositionAndOrientation(gripper._id, initial_gripper_pose[:3, 3], pb.getQuaternionFromEuler(tf.euler_from_matrix(initial_gripper_pose)))
            for k, gt_state in enumerate(gt_states):
                pb.resetJointState(object_model, args.link_id[k], gt_state)
            
            if not args.test:
                import pdb; pdb.set_trace()
        
        # loop
        if args.loop_mode == 'none':
            pass
        elif args.loop_mode == 'offline_joint':
            input_perception = np.copy(point_cloud_cam)
            output_perception = (np.vstack([fit_joint[0] if fit_joint is not None else np.array([np.nan, np.nan, np.nan]) for fit_joint in fit_joints]), 
                                np.vstack([fit_joint[1] if fit_joint is not None else np.array([np.nan, np.nan, np.nan]) for fit_joint in fit_joints]))
            output_perception = ((np.linalg.inv(extrinsic[:3, :3]) @ (output_perception[0].T - extrinsic[:3, 3].reshape((3, 1)))).T, 
                                (np.linalg.inv(extrinsic[:3, :3]) @ output_perception[1].T).T)
            if not args.test:
                np.savez('perception.npz', 
                        point_cloud=input_perception, joint_translation=output_perception[0], joint_direction=output_perception[1])

            for j in range(len(tried_grasps)):
                if tried_grasps[j] is None:
                    continue
                for g in range(len(tried_grasps[j])):
                    tried_grasps[j][g][4] = (np.linalg.inv(extrinsic[:3, :3]) @ (tried_grasps[j][g][4] - extrinsic[:3, 3]).reshape((3, 1))).reshape((3,))
                    tried_grasps[j][g][5] = np.linalg.inv(extrinsic[:3, :3]) @ tried_grasps[j][g][5]
            input_imagination = (np.copy(point_cloud_cam), np.hstack([translations_cam, directions_cam]), np.array(tried_grasps))
            output_imagination = np.array(tried_affordances)
            if not args.test:
                np.savez('imagination.npz', 
                        point_cloud=input_imagination[0], joint=input_imagination[1], grasp=input_imagination[2], affordance=output_imagination)
            
            if args.test:
                if not np.isnan(output_perception[0]).any():
                    np.savez(os.path.join(args.log_path, 'loop%04d.npz' % i), 
                             point_cloud=input_perception, joint_translation=output_perception[0], joint_direction=output_perception[1], 
                             grasp=input_imagination[2], affordance=output_imagination)
        elif args.loop_mode == 'offline_trajectory':
            input_perception = np.copy(point_cloud_cam)
            output_perception = [(np.linalg.inv(extrinsic[:3, :3]) @ (trajectory.T - extrinsic[:3, 3].reshape((3, 1)))).T if trajectory is not None else np.nan for trajectory in trajectories]
            output_perception = np.array(output_perception, dtype=object)
            np.savez('perception.npz', 
                    point_cloud=input_perception, trajectory=output_perception)
            
            for j in range(len(tried_grasps)):
                for g in range(len(tried_grasps[j])):
                    tried_grasps[j][g][4] = (np.linalg.inv(extrinsic[:3, :3]) @ (tried_grasps[j][g][4] - extrinsic[:3, 3]).reshape((3, 1))).reshape((3,))
                    tried_grasps[j][g][5] = np.linalg.inv(extrinsic[:3, :3]) @ tried_grasps[j][g][5]
            input_imagination = (np.copy(point_cloud_cam), np.hstack([translations_cam, directions_cam]), np.array(tried_grasps))
            output_imagination = np.array(tried_affordances)
            np.savez('imagination.npz', 
                    point_cloud=input_imagination[0], joint=input_imagination[1], grasp=input_imagination[2], affordance=output_imagination)
        elif args.loop_mode == 'online_joint':
            def joint_normalize(real_translation:np.ndarray, real_rotation:np.ndarray, real_state:Optional[np.ndarray], 
                                centroid:np.ndarray, scale:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
                translation = real_translation - centroid.reshape((1, 3))
                translation = translation / scale[0]
                rotation = real_rotation
                state = real_state
                return (translation, rotation, state)
            output_perception = (np.vstack([fit_joint[0] if fit_joint is not None else np.array([np.nan, np.nan, np.nan]) for fit_joint in fit_joints]), 
                                np.vstack([fit_joint[1] if fit_joint is not None else np.array([np.nan, np.nan, np.nan]) for fit_joint in fit_joints]), None)
            output_perception = ((np.linalg.inv(extrinsic[:3, :3]) @ (output_perception[0].T - extrinsic[:3, 3].reshape((3, 1)))).T, 
                                (np.linalg.inv(extrinsic[:3, :3]) @ output_perception[1].T).T, None)
            tr_label, rot_label, _state_label = joint_normalize(output_perception[0], output_perception[1], output_perception[2], joint_offset, joint_scale)
            tr_label = torch.from_numpy(tr_label).cuda(preds_tr.device).to(preds_tr.dtype)
            rot_label = torch.from_numpy(rot_label).cuda(preds_axis.device).to(preds_axis.dtype)
            
            perception_loss = 0
            # regression loss for translation
            preds_tr = preds_tr.view(len(fit_joints), 3)
            perception_loss_tr = tr_criterion(preds_tr[torch.logical_not(torch.isnan(tr_label).any(dim=-1))], 
                                            tr_label[torch.logical_not(torch.isnan(tr_label).any(dim=-1))])
            perception_loss += perception_loss_tr

            # regression loss for rotation
            preds_axis = preds_axis.view(len(fit_joints), 3)
            perception_loss_axis = axis_criterion(preds_axis[torch.logical_not(torch.isnan(rot_label).any(dim=-1))], 
                                                rot_label[torch.logical_not(torch.isnan(rot_label).any(dim=-1))])
            perception_loss_axis *= perception_cfg.lambda_axis
            perception_loss += perception_loss_axis

            # regression loss for state
            if perception_cfg.pointnet.state:
                pass
            
            perception_loss.backward(retain_graph=False)
            perception_opt.step()
            if not args.test:
                print("===> perception loss: {} + {} + 0 = {}".format(perception_loss_tr.item(), perception_loss_axis.item(), perception_loss.item()))
            torch.save(regressor.state_dict(), f'{args.perception_weight}/pointnet2_loop.pth')
        elif args.loop_mode == 'online_trajectory':
            output_perception = [(np.linalg.inv(extrinsic[:3, :3]) @ (trajectory.T - extrinsic[:3, 3].reshape((3, 1)))).T if trajectory is not None else np.nan for trajectory in trajectories]
            starters, labels_traj = [], []
            for j in range(len(output_perception)):
                if output_perception[j] is np.nan:
                    starters.append(np.nan)
                    labels_traj.append(np.nan)
                else:
                    starters.append(torch.from_numpy(output_perception[j][0]).cuda(translations_cam_tensor[j].device).to(translations_cam_tensor[j].dtype))
                    labels_traj.append(torch.from_numpy(output_perception[j][1:]).cuda(translations_cam_tensor[j].device).to(translations_cam_tensor[j].dtype))
            
            preds_traj = []
            for j in range(len(output_perception)):
                if output_perception[j] is np.nan:
                    preds_traj.append(np.nan)
                else:
                    if args.link_type[j] == 'revolute':
                        preds_traj.append(manipulation_control.generate_arc_grad(translations_cam_tensor[j], directions_cam_tensor[j], starters[j], 5, labels_traj[j].shape[0]))
                    elif args.link_type[j] == 'prismatic':
                        preds_traj.append(manipulation_control.generate_line_grad(translations_cam_tensor[j], directions_cam_tensor[j], starters[j], 0.1, labels_traj[j].shape[0]))
                    else:
                        raise ValueError('Invalid joint_type: {}'.format(args.link_type[j]))
            
            perception_loss = 0
            for j in range(len(output_perception)):
                if output_perception[j] is np.nan:
                    continue
                else:
                    perception_loss += traj_criterion(preds_traj[j], labels_traj[j]) / len(output_perception)
            
            perception_loss.backward(retain_graph=False)
            perception_opt.step()
            if not args.test:
                print("===> perception loss: {}".format(perception_loss.item()))
            torch.save(regressor.state_dict(), f'{args.perception_weight}/pointnet2_loop.pth')
        else:
            raise NotImplementedError
        if not args.test:
            print("===> execution back to perception and imagination done")
            import pdb; pdb.set_trace()
    print("===>", success_num, manipulation_num)
    if args.test:
        # write log into file
        with open(os.path.join(args.log_path, 'log.log'), 'w') as f:
            print("===>", success_num, manipulation_num, file=f)
