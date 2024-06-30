import argparse
import numpy as np
import open3d as o3d

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pc_path', type=str, help='the path to the point cloud, need contain `point_cloud`, `rgb` and `grasp`')
    args = parser.parse_args()

    # load point cloud
    pc = np.load(args.pc_path, allow_pickle=True)
    point_cloud = pc['point_cloud']
    pc_rgb = pc['rgb']
    joint_pose = pc['joint_pose']
    grasp = pc['grasp']

    for g in range(grasp.shape[0]):
        gg = grasp[g]
        against_joint_index = gg[-2]
        this_joint_pose = joint_pose[against_joint_index]

        geometries = []
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        geometries.append(camera_frame)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        pcd.colors = o3d.utility.Vector3dVector(pc_rgb)
        geometries.append(pcd)
        joint_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        joint_frame.transform(this_joint_pose)
        geometries.append(joint_frame)
        finger_width = 0.004
        tail_length = 0.04
        depth_base = 0.02
        gg_score = gg[0]
        gg_width = gg[1]
        gg_depth = gg[3]
        gg_translation = gg[4]
        gg_rotation = gg[5]

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
        geometries.append(line_set)
        o3d.visualization.draw_geometries(geometries, window_name="visualization")
        
        score = input(f"{g}/{grasp.shape[0]} Score: (-1/0/1/2): ")
        if score == '-1':
            score = -1
        elif score == '0':
            score = 0
        elif score == '1':
            score = 1
        elif score == '2':
            score = 2
        else:
            raise ValueError('Invalid score')
        grasp[g][-1] = score
    
    np.savez(args.pc_path, point_cloud=point_cloud, rgb=pc_rgb, joint_pose=joint_pose, grasp=grasp)
