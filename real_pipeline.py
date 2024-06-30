from typing import Optional, List
import configargparse
import time
import numpy as np
import cv2
import transformations as tf

from real.l515_get_data import CameraL515
from real.robot import Panda
from real.get_perception_service import xyzl_array_to_pointcloud2, get_axis_ppf
from real.get_imagination_service import getGraspNetService, convertGraspMsgtoNumpy, getAffordanceService


def config_parse() -> configargparse.Namespace:
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True, help='config file path')
    # general config
    parser.add_argument('--load', action='store_true', help='whether directly load stored results')
    parser.add_argument('--pause', action='store_true', help='whether pause every stage')
    # perception config
    parser.add_argument('--remove', action='store_true', help='whether remove table')
    parser.add_argument('--marker_height', type=int, help='the num of chessboard marker blocks in width minus 1, only valid when remove=True')
    parser.add_argument('--marker_width', type=int, help='the num of chessboard marker blocks in height minus 1, only valid when remove=True')
    parser.add_argument('--marker_size', type=float, help='the size of chessboard marker block in meter, only valid when remove=True')
    parser.add_argument('--table_color', nargs='+', type=int, help='optionally the color of table in RGB in range [0, 255], only valid when remove=True')
    # manipulation config
    parser.add_argument('--initial_joints', nargs='+', type=float, help='robot initial 7 joints')
    parser.add_argument('--cam2EE_path', type=str, help='the path to the calibrated cam2EE 4x4 pose')
    parser.add_argument('--flip', action='store_true', help='whether to flip the gripper')
    parser.add_argument('--affordance_threshold', type=float, help='affordance threshold for manipulation')
    parser.add_argument('--t', type=int, help='the time steps to manipulate')
    parser.add_argument('--sleep', type=float, help='the sleep time between each action in second')

    args = parser.parse_args()
    return args


def vis_pc(xyzrgb:np.ndarray, marker_pose:Optional[np.ndarray]=None, 
           joint_pose:Optional[np.ndarray]=None, 
           grasp:Optional[np.ndarray]=None, affordance:Optional[np.ndarray]=None, 
           grasp_pose:Optional[List[np.ndarray]]=None, 
           window_name:Optional[str]=None) -> None:
    import open3d as o3d
    geometries = []
    
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    geometries.append(camera_frame)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyzrgb[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(xyzrgb[:, 3:])
    geometries.append(pcd)
    
    if marker_pose is not None:
        marker_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        marker_frame.transform(marker_pose)
        geometries.append(marker_frame)
    
    if grasp_pose is not None:
        for grasp_pose_item in grasp_pose:
            grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            grasp_frame.transform(grasp_pose_item)
            geometries.append(grasp_frame)
    
    if joint_pose is not None:
        joint_num = int(len(joint_pose)/6)
        for j in range(joint_num):
            direction = np.asarray(joint_pose[j*6+3:j*6+6])
            translation = np.asarray(joint_pose[j*6:j*6+3])
            joint_axis = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.01, cone_radius=0.02, cylinder_height=0.4, cone_height=0.1, resolution=20, cylinder_split=4, cone_split=1)
            rotation = np.zeros((3, 3))
            temp2 = np.cross(direction, np.array([1., 0., 0.]))
            if np.linalg.norm(temp2) < 1e-6:
                temp1 = np.cross(np.array([0., 1., 0.]), direction)
                temp1 /= np.linalg.norm(temp1)
                temp2 = np.cross(direction, temp1)
                temp2 /= np.linalg.norm(temp2)
            else:
                temp2 /= np.linalg.norm(temp2)
                temp1 = np.cross(temp2, direction)
                temp1 /= np.linalg.norm(temp1)
            rotation[:, 0] = temp1
            rotation[:, 1] = temp2
            rotation[:, 2] = direction
            joint_axis.rotate(rotation, np.array([[0], [0], [0]]))
            joint_axis.translate(translation.reshape((3, 1)))
            joint_axis.paint_uniform_color([204/255, 204/255, 204/255])
            geometries.append(joint_axis)
            joint_point = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
            joint_point = joint_point.translate(translation.reshape((3, 1)))
            joint_point.paint_uniform_color([179/255, 179/255, 179/255])
            geometries.append(joint_point)
    
    if grasp is not None:
        finger_width = 0.004
        tail_length = 0.04
        depth_base = 0.02
        for i in range(grasp.shape[0]):
            gg = grasp[i]
            gg_affordance = affordance[i]
            gg_score = gg[0]
            gg_width = gg[1]
            gg_depth = gg[3]
            gg_translation = gg[13:16]
            gg_rotation = gg[4:13].reshape((3, 3))

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
    
    o3d.visualization.draw_geometries(geometries, window_name=window_name if window_name is not None else "visualization")


if __name__ == '__main__':
    args = config_parse()
    print("make sure services all running, include roscore, perception, imagination, graspnet, camera and robot")
    cam2EE_pose = np.load(args.cam2EE_path)     # cam2EE
    
    try:
        # prepare camera
        if not args.load:
            print("initialize camera")
            camera = CameraL515()
            print("initialize camera done")
        else:
            pass
        
        # prepare robot
        print("initialize robot")
        panda = Panda()
        panda.gripper_open()
        print("initialize robot done")
        
        # move to initial pose
        if args.initial_joints is not None:
            panda.moveJoint(args.initial_joints)
        print("===> prepare done")
        if args.pause:
            import pdb; pdb.set_trace()
        
        # get observation
        if not args.load:
            print("get observation")
            color, depth = camera.get_data(hole_filling=False)
            depth_sensor = camera.pipeline_profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            xyzrgb = camera.getXYZRGB(color, depth, np.identity(4), np.identity(4), camera.getIntrinsics(), inpaint=False, depth_scale=depth_scale)
            xyzrgb = xyzrgb[xyzrgb[:, 2] <= 1.5, :]
            if args.pause:
                vis_pc(xyzrgb, window_name="observation")
                cv2.imshow('color', color)
                while True:
                    if cv2.getWindowProperty('color', cv2.WND_PROP_VISIBLE) <= 0:
                        break
                    cv2.waitKey(1)
                cv2.destroyAllWindows()
                cmd = input("whether continue? (y/n): ")
                if cmd == 'y':
                    cv2.imwrite("rgb.png", color)
                    np.savez("xyzrgb.npz", point_cloud=xyzrgb[:, :3], rgb=xyzrgb[:, 3:])
                    print("save observation")
                elif cmd == 'n':
                    exit(1)
                else:
                    raise ValueError
            print("get observation done")
        else:
            color = cv2.imread("rgb.png")
            data = np.load("xyzrgb.npz")
            xyzrgb = np.hstack([data['point_cloud'], data['rgb']])
            print("load observation")
        
        # remove table
        if args.remove:
            print("remove table")
            gray_image = cv2.cvtColor(cv2.imread("rgb.png"), cv2.COLOR_BGR2GRAY)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            ret, corners = cv2.findChessboardCorners(gray_image, (args.marker_width, args.marker_height), None)
            if ret:
                corners2 = cv2.cornerSubPix(gray_image, corners, (5,5), (-1,-1), criteria)
                cv2.drawChessboardCorners(color, (args.marker_width, args.marker_height), corners2, ret)
                if args.pause:
                    cv2.imshow('marker', color)
                    cv2.waitKey(0)

                objp = np.zeros((args.marker_width*args.marker_height, 3), np.float32)
                objp[:, :2] = np.mgrid[0:args.marker_width, 0:args.marker_height].T.reshape(-1,2) * args.marker_size
                objpoints = []
                objpoints.append(objp)
                imgpoints = []
                imgpoints.append(corners2)

                # calibrate camera
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_image.shape[::-1], None, None)
                intrinsic_matrix = mtx
                distortion_coefficients = dist
                rotation_matrix = cv2.Rodrigues(rvecs[0])[0]
                translation_vector = tvecs[0]
                extrinsic_matrix = np.concatenate((rotation_matrix, translation_vector), axis=1)
                extrinsic_matrix = np.concatenate((extrinsic_matrix, np.array([[0, 0, 0, 1]])), axis=0)

                if args.pause:
                    vis_pc(xyzrgb, marker_pose=extrinsic_matrix, window_name="marker calibration")

                # remove z>0 points in marker frame
                point_cloud_marker = np.concatenate((xyzrgb[:, :3], np.ones((xyzrgb[:, :3].shape[0], 1))), axis=1)
                point_cloud_marker = np.matmul(point_cloud_marker, np.linalg.inv(extrinsic_matrix).T)
                xyzrgb = xyzrgb[np.where(point_cloud_marker[:,2] < 0)]

                if args.pause:
                    vis_pc(xyzrgb, marker_pose=extrinsic_matrix, window_name="remove calibration")

                # remove according to optional table color
                if args.table_color is not None:
                    assert len(args.table_color) % 3 == 0
                    for sample_idx in range(len(args.table_color) // 3):
                        table_color = np.array(args.table_color[3*sample_idx: 3*sample_idx+3]) / 255.0
                        color_diff = np.sum(np.abs(xyzrgb[:, 3:] - table_color), axis=1)
                        xyzrgb = xyzrgb[np.where(color_diff > 0.1)]

                    if args.pause:
                        vis_pc(xyzrgb, marker_pose=extrinsic_matrix, window_name="remove color")
            else:
                print('no chessboard marker found')
                exit(2)

            # continue
            if args.pause:
                cmd = input("whether continue? (y/n): ")
                if cmd == 'y':
                    pass
                elif cmd == 'n':
                    exit(3)
                else:
                    raise ValueError
                cv2.destroyAllWindows()
            print("remove table done")
        else:
            pass
        
        # perception service
        if not args.load:
            print("get perception service")
            pc_msg = xyzl_array_to_pointcloud2(xyzrgb[:, :3])
            axis = get_axis_ppf(pc_msg)
            if args.pause:
                vis_pc(xyzrgb, joint_pose=axis, window_name="joint")
                # continue
                cmd = input("whether continue? (y/n): ")
                if cmd == 'y':
                    np.save("axis.npy", axis)
                    print("save perception")
                elif cmd == 'n':
                    exit(4)
                else:
                    raise ValueError
            print("perception service done")
        else:
            axis = tuple(np.load("axis.npy"))
            print("load perception")
        print("===> perception done")
        if args.pause:
            import pdb; pdb.set_trace()
        
        if not args.load:
            # graspnet service
            print("get anygrasp service")
            GraspMsg_list = getGraspNetService(pc_msg)
            grasp = convertGraspMsgtoNumpy(GraspMsg_list)
            print("anygrasp service done")
            
            # imagination service
            print("get imagination service")
            joint_num = int(len(axis)/6)
            axises = []
            for i in range(joint_num):
                axises.extend(list(axis)[i*6:i*6+6] + [0])
            affordance = getAffordanceService(pc_msg, GraspMsg_list, axises)
            affordance = np.array([a for a in affordance[0].floatlist])     # TODO: only support 1 joint case
            if args.pause:
                vis_pc(xyzrgb, joint_pose=axis, grasp=grasp, affordance=affordance, window_name="all grasps in camera")
                # continue
                cmd = input("whether continue? (y/n): ")
                if cmd == 'y':
                    pass
                elif cmd == 'n':
                    exit(5)
                else:
                    raise ValueError
            print("imagination service done")
            
            # select top 1 grasps within gripper range to try
            valid_grasps_mask = grasp[:, 1] <= panda.gripper.read_once().max_width
            grasp = grasp[valid_grasps_mask]
            affordance = affordance[valid_grasps_mask]
            top_grasp_ids = np.argsort(affordance)[::-1][:1]
            grasp = grasp[top_grasp_ids]
            affordance = affordance[top_grasp_ids]
            if (affordance >= args.affordance_threshold).any():
                np.save("grasp.npy", grasp)
                np.save("affordance.npy", affordance)
                print("save imagination")
            else:
                print("no grasp affordable found")
                exit(6)
        else:
            grasp = np.load("grasp.npy")
            affordance = np.load("affordance.npy")
            print("load imagination")
        if args.pause:
            vis_pc(xyzrgb, joint_pose=axis, grasp=grasp, affordance=affordance, window_name="single grasp in camera")
        print("===> imagination done")
        if args.pause:
            import pdb; pdb.set_trace()
        
        # transform grasp pose to robot frame
        panda.move_gripper(grasp[0][1])     # gripper
        
        current_pose = np.array(panda.robot.read_once().O_T_EE).reshape(4, 4).T     # EE2robot
        EE2robot_pose = np.copy(current_pose)
        grasp_pose_cam = np.eye(4)
        grasp_pose_cam[:3, 3] = grasp[0][13:16]
        grasp_pose_cam[:3, :3] = grasp[0][4:13].reshape((3, 3))
        
        if args.pause:
            vis_pc(xyzrgb, grasp_pose=[grasp_pose_cam], window_name="single grasp frame in camera")
        
        cam2robot_pose = EE2robot_pose @ cam2EE_pose
        xyzrgb[:, :3] = (cam2robot_pose[:3, :3] @ xyzrgb[:, :3].T + cam2robot_pose[:3, 3].reshape(3, 1)).T
        translation = np.asarray(axis[0:3])
        direction = np.asarray(axis[3:6])
        translation_robot = cam2robot_pose[:3, :3] @ translation + cam2robot_pose[:3, 3]
        direction_robot = cam2robot_pose[:3, :3] @ direction
        grasp_pose_robot = cam2robot_pose @ grasp_pose_cam      # grasp (x, y, z) as (in, right, down), panda (x, y, z) as (up, left, out), robot (x, y, z) as (in, left, up)
        gripper_pose_robot = np.eye(4)
        gripper_pose_robot[:3, 0] = -grasp_pose_robot[:3, 2]
        gripper_pose_robot[:3, 1] = -grasp_pose_robot[:3, 1]
        gripper_pose_robot[:3, 2] = -grasp_pose_robot[:3, 0]
        gripper_pose_robot[:3, 3] = grasp_pose_robot[:3, 3]
        
        if args.flip:
            gripper_pose_robot = gripper_pose_robot @ np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        
        if args.pause:
            vis_pc(xyzrgb, joint_pose=translation_robot.tolist()+direction_robot.tolist(), 
                   grasp_pose=[gripper_pose_robot], window_name="single gripper frame in robot")
        
        # orientation
        R_AB = current_pose[:3, :3].T @ gripper_pose_robot[:3, :3]
        R_AB_Transformation = np.vstack([R_AB, np.array([[0, 0, 0]])])
        R_AB_Transformation = np.hstack([R_AB_Transformation, np.array([[0], [0], [0], [1]])])
        euler_angle = tf.euler_from_matrix(R_AB_Transformation, axes='rzyx')
        for _ in range(8):
            panda.moveRelativePose([0.0, 0.0, 0.0, euler_angle[0]/8, euler_angle[1]/8, euler_angle[2]/8])
            time.sleep(args.sleep)
        
        # translation
        current_pose = np.array(panda.robot.read_once().O_T_EE).reshape(4, 4).T     # EE2robot
        tr_robot = gripper_pose_robot[:3, 3] - current_pose[:3, 3]
        tr_rel = np.matmul(current_pose[:3, :3].T, tr_robot)
        for _ in range(8):
            panda.moveRelativePose([tr_rel[0]/8, tr_rel[1]/8, tr_rel[2]/8])
            time.sleep(args.sleep)
        
        panda.gripper_close()
        for time_step in range(args.t):
            rotation_angle = -1.0 / 180.0 * np.pi
            delta_pose_robot = tf.rotation_matrix(angle=rotation_angle, direction=direction_robot, point=translation_robot)
            current_pose_robot = np.array(panda.robot.read_once().O_T_EE).reshape(4, 4).T
            target_pose_robot = delta_pose_robot @ current_pose_robot
            R_AB = current_pose_robot[:3, :3].T @ target_pose_robot[:3, :3]
            R_AB_Transformation = np.vstack([R_AB, np.array([[0, 0, 0]])])
            R_AB_Transformation = np.hstack([R_AB_Transformation, np.array([[0], [0], [0], [1]])])
            euler_angle = tf.euler_from_matrix(R_AB_Transformation, axes='rzyx')
            tr_robot = target_pose_robot[:3, 3] - current_pose_robot[:3, 3]
            tr_rel = np.matmul(current_pose_robot[:3, :3].T, tr_robot)
            panda.moveRelativePose([tr_rel[0], tr_rel[1], tr_rel[2], euler_angle[0], euler_angle[1], euler_angle[2]])
            time.sleep(args.sleep)
            if (time_step+1) % (args.t//20) == 0:
                current_width = panda.gripper.read_once().width
                panda.move_gripper(current_width+0.01)
                panda.gripper_close()
        print("===> manipulation done")
        import pdb; pdb.set_trace()
    except:
        del camera
        del panda
