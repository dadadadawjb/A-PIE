import argparse
import numpy as np
import open3d as o3d

def visualize(point_cloud:np.ndarray, joint_translations:np.ndarray, joint_directions:np.ndarray, 
              grasps:np.ndarray, affordances:np.ndarray) -> None:
    geometries = []

    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    geometries.append(camera_frame)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.paint_uniform_color([126/255, 208/255, 248/255])
    geometries.append(pcd)

    for j in range(joint_translations.shape[0]):
        joint_axis = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.008, cone_radius=0.02, cylinder_height=0.2, cone_height=0.05)
        rotation = np.zeros((3, 3))
        temp2 = np.cross(joint_directions[j], np.array([1., 0., 0.]))
        if np.linalg.norm(temp2) < 1e-6:
            temp1 = np.cross(np.array([0., 1., 0.]), joint_directions[j])
            temp1 /= np.linalg.norm(temp1)
            temp2 = np.cross(joint_directions[j], temp1)
            temp2 /= np.linalg.norm(temp2)
        else:
            temp2 /= np.linalg.norm(temp2)
            temp1 = np.cross(temp2, joint_directions[j])
            temp1 /= np.linalg.norm(temp1)
        rotation[:, 0] = temp1
        rotation[:, 1] = temp2
        rotation[:, 2] = joint_directions[j]
        joint_axis.rotate(rotation, np.array([[0], [0], [0]]))
        joint_axis.translate(joint_translations[j].reshape((3, 1)))
        joint_axis.paint_uniform_color([255/255, 220/255, 126/255])
        geometries.append(joint_axis)
        joint_point = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        joint_point = joint_point.translate(joint_translations[j].reshape((3, 1)))
        joint_point.paint_uniform_color([250/255, 170/255, 137/255])
        geometries.append(joint_point)
    
    finger_width = 0.004
    tail_length = 0.04
    depth_base = 0.02
    for j in range(grasps.shape[0]):
        for g in range(grasps.shape[1]):
            gg = grasps[j][g]
            gg_success = affordances[j][g]
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
            if gg_success == 1:
                line_set.paint_uniform_color([0, 1, 0])     # green
            elif gg_success == 0:
                line_set.paint_uniform_color([1, 0, 0])     # red
            else:
                raise ValueError('grasp success must be 1, 0')
            geometries.append(line_set)
    
    o3d.visualization.draw_geometries(geometries, window_name="visualization")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='the path to the loop data')
    args = parser.parse_args()

    data = np.load(args.data_path, allow_pickle=True)
    pc = data['point_cloud']
    joint_translations = data['joint_translation']
    joint_directions = data['joint_direction']
    grasps = data['grasp']
    affordances = data['affordance']
    print('points num:', pc.shape[0])
    print('joints num:', joint_translations.shape[0])
    print('grasps num:', grasps.shape[0:2])

    visualize(pc, joint_translations, joint_directions, grasps, affordances)
