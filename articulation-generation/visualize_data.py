from typing import List, Optional
import argparse
import os
import json
import numpy as np
import open3d as o3d

def visualize(point_cloud:np.ndarray, color:Optional[np.ndarray], object_pose:Optional[np.ndarray], joint_poses:np.ndarray, joint_axes_which:List[str], 
              grasps:Optional[np.ndarray]) -> None:
    geometries = []

    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    geometries.append(camera_frame)

    if object_pose is not None:
        object_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        object_frame = object_frame.transform(object_pose)
        geometries.append(object_frame)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    else:
        pcd.paint_uniform_color([126/255, 208/255, 248/255])
    geometries.append(pcd)

    for i in range(joint_poses.shape[0]):
        joint_pose = joint_poses[i]
        joint_axis = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.008, cone_radius=0.02, cylinder_height=0.2, cone_height=0.05)
        joint_axis_which = joint_axes_which[i]
        if joint_axis_which == 'x':
            joint_axis = joint_axis.rotate(np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]), center=np.array([[0], [0], [0]]))
        elif joint_axis_which == 'y':
            joint_axis = joint_axis.rotate(np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]), center=np.array([[0], [0], [0]]))
        elif joint_axis_which == 'z':
            joint_axis = joint_axis.rotate(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), center=np.array([[0], [0], [0]]))
        else:
            raise ValueError('joint axis must be x, y or z')
        joint_axis = joint_axis.transform(joint_pose)
        joint_axis.paint_uniform_color([255/255, 220/255, 126/255])
        geometries.append(joint_axis)
        joint_point = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        joint_point = joint_point.transform(joint_pose)
        joint_point.paint_uniform_color([250/255, 170/255, 137/255])
        geometries.append(joint_point)
    
    if grasps is not None:
        finger_width = 0.004
        tail_length = 0.04
        depth_base = 0.02
        for i in range(grasps.shape[0]):
            gg = grasps[i]
            if gg[-2] != 0:
                # TODO: this code is used to filter out only the grasps against the specific joint
                continue
            gg_success = gg[-1]
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
            if gg_success == 2:
                line_set.paint_uniform_color([0, 1, 0])     # green
            elif gg_success == 1:
                line_set.paint_uniform_color([0, 0, 1])     # blue
            elif gg_success == 0:
                line_set.paint_uniform_color([1, 1, 0])     # yellow
            elif gg_success == -1:
                line_set.paint_uniform_color([1, 0, 0])     # red
            else:
                raise ValueError('grasp success must be 2, 1, 0, -1')
            geometries.append(line_set)
    
    o3d.visualization.draw_geometries(geometries, window_name="visualization")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='the path to the collected data')
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(args.data_path), 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    if config['grasp']:
        data = np.load(args.data_path, allow_pickle=True)
    else:
        data = np.load(args.data_path)
    pc = data['point_cloud']
    if 'object_pose' in list(data.keys()):
        pose = data['object_pose']
    else:
        pose = None
    if 'rgb' in list(data.keys()):
        pc_rgb = data['rgb']
    else:
        pc_rgb = None
    joint_poses = data['joint_pose']
    print('points num:', pc.shape[0])
    print('joints num:', joint_poses.shape[0])
    if config['grasp']:
        grasp = data['grasp']
        print('grasps num:', grasp.shape[0])
        print('both successful grasps num:', grasp[grasp[:, -1] == 2].shape[0])
        print('half successful grasps num:', grasp[grasp[:, -1] == 1].shape[0])
        print('none successful grasps num:', grasp[grasp[:, -1] == 0].shape[0])
        print('fail grasp num:', grasp[grasp[:, -1] == -1].shape[0])
        print('avg grasp score:', np.mean(grasp[:, 0]) if grasp.shape[0] > 0 else 'nan')
        print('avg both successful grasp score:', np.mean(grasp[grasp[:, -1] == 2, 0]) if grasp[grasp[:, -1] == 2].shape[0] > 0 else 'nan')
        print('avg half successful grasp score:', np.mean(grasp[grasp[:, -1] == 1, 0]) if grasp[grasp[:, -1] == 1].shape[0] > 0 else 'nan')
        print('avg none successful grasp score:', np.mean(grasp[grasp[:, -1] == 0, 0]) if grasp[grasp[:, -1] == 0].shape[0] > 0 else 'nan')
        print('avg fail grasp score:', np.mean(grasp[grasp[:, -1] == -1, 0]) if grasp[grasp[:, -1] == -1].shape[0] > 0 else 'nan')
    else:
        grasp = None

    visualize(pc, pc_rgb, pose, joint_poses, config["link_axis"], grasp)
