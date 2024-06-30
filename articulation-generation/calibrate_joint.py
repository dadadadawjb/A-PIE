import argparse
import numpy as np
import transformations as tf
import open3d as o3d

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pc_path', type=str, help='the path to the point cloud, need contain `point_cloud` and `rgb`')
    parser.add_argument('--translation', nargs='+', type=float, help='translation vector in meter')
    parser.add_argument('--rotation', nargs='+', type=float, help='rotation vector in degree')
    args = parser.parse_args()

    # load point cloud
    pc = np.load(args.pc_path)
    point_cloud = pc['point_cloud']
    pc_rgb = pc['rgb']

    # joint poses
    num_joint = len(args.translation) // 3
    joint_poses = []
    for i in range(num_joint):
        translation = args.translation[i*3:i*3+3]
        joint_pose = np.eye(4)
        joint_pose[:3, 3] = translation
        rotation = args.rotation[i*3:i*3+3]
        rotation1 = tf.rotation_matrix(rotation[0]/180*np.pi, [1, 0, 0], point=translation)
        rotation2 = tf.rotation_matrix(rotation[1]/180*np.pi, [0, 1, 0], point=translation)
        rotation3 = tf.rotation_matrix(rotation[2]/180*np.pi, [0, 0, 1], point=translation)
        joint_pose[:3, :3] = np.matmul(np.matmul(rotation1[:3, :3], rotation2[:3, :3]), rotation3[:3, :3])
        joint_poses.append(joint_pose)

    geometries = []
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    geometries.append(camera_frame)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(pc_rgb)
    geometries.append(pcd)
    for joint_pose in joint_poses:
        joint_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        joint_frame.transform(joint_pose)
        geometries.append(joint_frame)
    o3d.visualization.draw_geometries(geometries, window_name='calibration')

    cmd = input('Is the calibration correct? (y/n): ')
    if cmd == 'y':
        np.savez(args.pc_path, point_cloud=point_cloud, rgb=pc_rgb, joint_pose=np.stack(joint_poses))
        print('Calibration saved.')
    elif cmd == 'n':
        print('Calibration not saved.')
    else:
        print('Invalid input, calibration not saved.')
