import argparse
import os
import numpy as np
import cv2
import open3d as o3d

def visualize(pc:np.ndarray, pc_rgb:np.ndarray, marker_pose:np.ndarray, joint_poses:np.ndarray, window_name:str) -> None:
    geometries = []
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    geometries.append(camera_frame)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.colors = o3d.utility.Vector3dVector(pc_rgb)
    for j in range(joint_poses.shape[0]):
        joint_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        joint_frame.transform(joint_poses[j])
        geometries.append(joint_frame)
    geometries.append(pcd)
    marker_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    marker_frame.transform(marker_pose)
    geometries.append(marker_frame)
    o3d.visualization.draw_geometries(geometries, window_name=window_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='the path to the RGB image')
    parser.add_argument('--pc_path', type=str, help='the path to the point cloud, need contain `point_cloud` and `rgb` and `joint_pose`')
    parser.add_argument('--marker_width', type=int, help='the num of chessboard marker blocks in width minus 1')
    parser.add_argument('--marker_height', type=int, help='the num of chessboard marker blocks in height minus 1')
    parser.add_argument('--marker_size', type=float, default=0.01, help='the size of chessboard marker block in meter')
    parser.add_argument('--table_color', nargs='+', type=int, help='optionally the color of table in RGB in range [0, 255]')
    args = parser.parse_args()

    # load RGB image
    image = cv2.imread(args.image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # load point cloud
    pc = np.load(args.pc_path)
    point_cloud = pc['point_cloud']
    pc_rgb = pc['rgb']
    joint_pose = pc['joint_pose']

    # find chessboard marker
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    ret, corners = cv2.findChessboardCorners(gray_image, (args.marker_width, args.marker_height), None)
    if ret:
        corners2 = cv2.cornerSubPix(gray_image, corners, (5,5), (-1,-1), criteria)
        cv2.drawChessboardCorners(image, (args.marker_width, args.marker_height), corners2, ret)
        cv2.imshow('marker', image)
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

        visualize(point_cloud, pc_rgb, extrinsic_matrix, joint_pose, 'before remove')

        # remove z>0 points in marker frame
        point_cloud_marker = np.concatenate((point_cloud, np.ones((point_cloud.shape[0], 1))), axis=1)
        point_cloud_marker = np.matmul(point_cloud_marker, np.linalg.inv(extrinsic_matrix).T)
        point_cloud = point_cloud[np.where(point_cloud_marker[:,2] < 0)]
        pc_rgb = pc_rgb[np.where(point_cloud_marker[:,2] < 0)]

        visualize(point_cloud, pc_rgb, extrinsic_matrix, joint_pose, 'after remove')

        # remove according to optional table color
        if args.table_color is not None:
            assert len(args.table_color) % 3 == 0
            for sample_idx in range(len(args.table_color) // 3):
                table_color = np.array(args.table_color[3*sample_idx: 3*sample_idx+3]) / 255.0
                color_diff = np.sum(np.abs(pc_rgb - table_color), axis=1)
                point_cloud = point_cloud[np.where(color_diff > 0.1)]
                pc_rgb = pc_rgb[np.where(color_diff > 0.1)]

            visualize(point_cloud, pc_rgb, extrinsic_matrix, joint_pose, 'after remove color')

        # save
        cmd = input('Is the removal correct? (y/n): ')
        if cmd == 'y':
            save_path = os.path.join(os.path.split(args.pc_path)[0], os.path.splitext(os.path.split(args.pc_path)[1])[0] + '_remove.npz')
            np.savez(save_path, point_cloud=point_cloud, rgb=pc_rgb, joint_pose=pc['joint_pose'])
            print('Removal saved.')
        elif cmd == 'n':
            print('Removal not saved.')
        else:
            print('Invalid input, removal not saved.')

        cv2.destroyAllWindows()
    else:
        print('no chessboard marker found')
        exit()
