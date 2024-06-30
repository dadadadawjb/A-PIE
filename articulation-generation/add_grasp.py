import argparse
import numpy as np
from munch import DefaultMunch
from gsnet import AnyGrasp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='the path to the point cloud data, need contain `point_cloud` and `rgb` and `joint_pose`')
    parser.add_argument('--gsnet_weight', type=str, help='the path to the gsnet weight')
    args = parser.parse_args()

    # load data
    data = np.load(args.data_path)
    point_cloud = data['point_cloud']
    joint_pose = data['joint_pose']
    pc_rgb = data['rgb']

    # load model
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

    # detect
    gg = grasp_detector.get_grasp(point_cloud.astype(np.float32), colors=None, lims=[-2, 2, -2, 2, -2, 2])
    if gg is None:
        gg = []
    else:
        if len(gg) != 2:
            gg = []
        else:
            gg, _ = gg
    if len(gg) > 0:
        gg.nms()
        gg.sort_by_score()
        gg = gg[:100]
        collect_gg_array = []
        for gg_ in gg:
            for j in range(joint_pose.shape[0]):
                # if collected
                collect_grasp = [gg_.score, gg_.width, gg_.height, gg_.depth, 
                                    np.copy(gg_.translation), np.copy(gg_.rotation_matrix), j, 1]
                collect_gg_array.append(collect_grasp)
    else:
        collect_gg_array = []
    
    save_path = args.data_path.replace('.npz', '_grasp.npz')
    np.savez(save_path, joint_pose=joint_pose, point_cloud=point_cloud, rgb=pc_rgb, grasp=np.array(collect_gg_array, dtype=object))
