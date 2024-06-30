import os
import json
import omegaconf
import itertools
from pathlib import Path
import tqdm
import numpy as np
import torch
import MinkowskiEngine as ME

from utils import farthest_point_sample, pc_normalize, grasp_normalize, joint_normalize
from src_shot.build import shot


class GraspDataset(torch.utils.data.Dataset):
    def __init__(self, cfg:omegaconf.dictconfig.DictConfig, is_train:bool):
        super().__init__()
        self.cfg = cfg
        self.is_train = is_train
        if is_train:
            instances = cfg.dataset.train_instances
        else:
            instances = cfg.dataset.test_instances
        # TODO: in looptune, you need to change `dataset_path` to `test_path`
        self.fns = sorted(list(itertools.chain(*[list(Path(cfg.dataset.dataset_path).glob('{}/*/*.npz'.format(instance))) for instance in instances])))

    def __len__(self):
        return len(self.fns)
    
    def __getitem__(self, idx:int):
        data = np.load(self.fns[idx], allow_pickle=True)
        pc = data['point_cloud']
        grasp = data['grasp']
        assert data['joint_pose'].shape[0] == self.cfg.joints
        joint_pose = data['joint_pose']
        config_path = os.path.join(os.path.dirname(self.fns[idx]), 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        joint_axis_which = config["link_axis"]
        joint_type = config["link_type"]
        joint_state = config["link_state"]

        # sparse quantize
        possible_quantize_path = self.fns[idx].parent / f"{self.fns[idx].name.replace('.npz', '')}_quantize_{self.cfg.shot.res}.npy"
        if os.path.exists(possible_quantize_path):
            pc = np.load(possible_quantize_path)
        else:
            indices = ME.utils.sparse_quantize(np.ascontiguousarray(pc), return_index=True, quantization_size=self.cfg.shot.res)[1]
            pc = np.ascontiguousarray(pc[indices].astype(np.float32))   # (N', 3)
            if self.cfg.cache:
                np.save(possible_quantize_path, pc)

        # estimate normal
        possible_normal_path = self.fns[idx].parent / f"{self.fns[idx].name.replace('.npz', '')}_normal_{self.cfg.shot.res * self.cfg.shot.receptive_field}_{self.cfg.shot.res}.npy"
        if os.path.exists(possible_normal_path):
            pc_normal = np.load(possible_normal_path)
        else:
            pc_normal = shot.estimate_normal(pc, self.cfg.shot.res * self.cfg.shot.receptive_field).reshape(-1, 3).astype(np.float32)
            pc_normal[~np.isfinite(pc_normal)] = 0                      # (N', 3)
            if self.cfg.cache:
                np.save(possible_normal_path, pc_normal)

        # sample points to make batch regular
        num_sample_points = self.cfg.samples if self.is_train else self.cfg.test_samples
        possible_sample_path = self.fns[idx].parent / f"{self.fns[idx].name.replace('.npz', '')}_sample_{num_sample_points}_{self.cfg.shot.res}.npy"
        if os.path.exists(possible_sample_path):
            indices = np.load(possible_sample_path)
            pc = pc[indices]
        else:
            pc, indices = farthest_point_sample(pc, num_sample_points)  # (N, 3)
            if self.cfg.cache:
                np.save(possible_sample_path, indices)
        pc_normal = pc_normal[indices]                                  # (N, 3)

        # normalize
        if self.cfg.normalization:
            pc, offset, scale = pc_normalize(pc)                        # (3,), (1,)

        # grasp features
        contained_grasp_items = []
        for level in self.cfg.embedding_net.levels:
            contained_grasp_items.extend(level[0])
        grasp = grasp[np.isin(grasp[:, 7].astype(np.int32), contained_grasp_items)]     # (G, 8)

        grasp_num = grasp.shape[0]
        grasp_feat = np.concatenate([grasp[:, 0:4], np.stack(grasp[:, 4], axis=0), 
                                     np.stack(grasp[:, 5], axis=0).reshape((-1, 9))], axis=-1)  # (G, 16)
        if self.cfg.normalization:
            grasp_feat = grasp_normalize(grasp_feat, offset, scale)
        padding = np.zeros((100*self.cfg.joints - grasp_num, 16), dtype=np.float32)
        grasp_feat = np.concatenate([grasp_feat, padding], axis=0)      # (100*J, 16)

        grasp_joint = grasp[:, 6].astype(np.int32)                      # (G,)
        grasp_affordance_ = grasp[:, 7].astype(np.int32)
        if self.cfg.embedding_net.classification:
            grasp_affordance = np.zeros_like(grasp_affordance_, dtype=np.int64)     # (G,)
            for level_idx in range(len(self.cfg.embedding_net.levels)):
                for level_item in self.cfg.embedding_net.levels[level_idx][0]:
                    grasp_affordance[grasp_affordance_ == level_item] = level_idx
            padding = np.zeros(100*self.cfg.joints - grasp_num, dtype=np.int64)
            grasp_affordance = np.concatenate([grasp_affordance, padding], axis=0)  # (100*J,)
        else:
            grasp_affordance = np.zeros_like(grasp_affordance_, dtype=np.float32)
            for level_idx in range(len(self.cfg.embedding_net.levels)):
                for level_item in self.cfg.embedding_net.levels[level_idx][0]:
                    grasp_affordance[grasp_affordance_ == level_item] = self.cfg.embedding_net.levels[level_idx][1]
            padding = np.zeros(100*self.cfg.joints - grasp_num, dtype=np.float32)
            grasp_affordance = np.concatenate([grasp_affordance, padding], axis=0)

        # joint features
        joint_feat_ = []
        for j in range(self.cfg.joints):
            translation = joint_pose[j, :3, -1]
            if joint_axis_which[j] == 'x':
                rotation = joint_pose[j, :3, 0]
            elif joint_axis_which[j] == 'y':
                rotation = joint_pose[j, :3, 1]
            elif joint_axis_which[j] == 'z':
                rotation = joint_pose[j, :3, 2]
            else:
                raise ValueError('Invalid joint_axis_which: {}'.format(joint_axis_which[j]))
            assert joint_type[j] == self.cfg.types[j]
            if joint_type[j] == 'revolute':
                type_feat = 0
                state_feat = joint_state[j] - self.cfg.states[j] / 180.0 * np.pi / 2
            elif joint_type[j] == 'prismatic':
                type_feat = 1
                state_feat = joint_state[j] - self.cfg.states[j] / 100.0 / 2
            else:
                raise ValueError('Invalid joint_type: {}'.format(joint_type[j]))
            joint_feat_.append(np.concatenate([translation, rotation, [type_feat, state_feat]]))
        joint_feat_ = np.stack(joint_feat_, axis=0)                     # (J, 8)
        if self.cfg.normalization:
            joint_feat_ = joint_normalize(joint_feat_, offset, scale)
        joint_feat = joint_feat_[grasp_joint, :]                        # (G, 8)
        padding = np.zeros((100*self.cfg.joints - grasp_num, 8), dtype=np.float32)
        joint_feat = np.concatenate([joint_feat, padding], axis=0)      # (100*J, 8)

        return pc.astype(np.float32), \
            pc_normal.astype(np.float32), \
            grasp_feat.astype(np.float32), \
            joint_feat.astype(np.float32), \
            grasp_affordance, \
            grasp_num


class LoopDataset(torch.utils.data.Dataset):
    def __init__(self, cfg:omegaconf.dictconfig.DictConfig, is_train:bool):
        super().__init__()
        self.cfg = cfg
        self.is_train = is_train
        if is_train:
            dataset_path = cfg.dataset.train_path
            instances = cfg.dataset.train_instances
        else:
            dataset_path = cfg.dataset.test_path
            instances = cfg.dataset.test_instances
        self.fns = sorted(list(itertools.chain(*[list(Path(dataset_path).glob('{}/*.npz'.format(instance))) for instance in instances])))

    def __len__(self):
        return len(self.fns)
    
    def __getitem__(self, idx:int):
        data = np.load(self.fns[idx], allow_pickle=True)
        pc = data['point_cloud']
        assert data['joint_translation'].shape[0] == self.cfg.joints and data['joint_direction'].shape[0] == self.cfg.joints
        joint_translation = data['joint_translation']
        joint_direction = data['joint_direction']
        grasp = data['grasp']
        affordance = data['affordance']

        # sparse quantize
        possible_quantize_path = self.fns[idx].parent / f"{self.fns[idx].name.replace('.npz', '')}_quantize_{self.cfg.shot.res}.npy"
        if os.path.exists(possible_quantize_path):
            pc = np.load(possible_quantize_path)
        else:
            indices = ME.utils.sparse_quantize(np.ascontiguousarray(pc), return_index=True, quantization_size=self.cfg.shot.res)[1]
            pc = np.ascontiguousarray(pc[indices].astype(np.float32))   # (N', 3)
            if self.cfg.cache:
                np.save(possible_quantize_path, pc)

        # estimate normal
        possible_normal_path = self.fns[idx].parent / f"{self.fns[idx].name.replace('.npz', '')}_normal_{self.cfg.shot.res * self.cfg.shot.receptive_field}_{self.cfg.shot.res}.npy"
        if os.path.exists(possible_normal_path):
            pc_normal = np.load(possible_normal_path)
        else:
            pc_normal = shot.estimate_normal(pc, self.cfg.shot.res * self.cfg.shot.receptive_field).reshape(-1, 3).astype(np.float32)
            pc_normal[~np.isfinite(pc_normal)] = 0                      # (N', 3)
            if self.cfg.cache:
                np.save(possible_normal_path, pc_normal)

        # sample points to make batch regular
        num_sample_points = self.cfg.samples if self.is_train else self.cfg.test_samples
        possible_sample_path = self.fns[idx].parent / f"{self.fns[idx].name.replace('.npz', '')}_sample_{num_sample_points}_{self.cfg.shot.res}.npy"
        if os.path.exists(possible_sample_path):
            indices = np.load(possible_sample_path)
            pc = pc[indices]
        else:
            pc, indices = farthest_point_sample(pc, num_sample_points)  # (N, 3)
            if self.cfg.cache:
                np.save(possible_sample_path, indices)
        pc_normal = pc_normal[indices]                                  # (N, 3)

        # normalize
        if self.cfg.normalization:
            pc, offset, scale = pc_normalize(pc)                        # (3,), (1,)

        # grasp features
        grasp_ = []
        for j in range(grasp.shape[0]):
            for g in range(grasp[j].shape[0]):
                grasp_jg = np.append(grasp[j][g], j)
                grasp_jg = np.append(grasp_jg, affordance[j][g])
                grasp_.append(grasp_jg)
        grasp_ = np.stack(grasp_, axis=0)                               # (G, 8)
        grasp_num = grasp_.shape[0]
        grasp_feat = np.concatenate([grasp_[:, 0:4], np.stack(grasp_[:, 4], axis=0), 
                                     np.stack(grasp_[:, 5], axis=0).reshape((-1, 9))], axis=-1) # (G, 16)
        if self.cfg.normalization:
            grasp_feat = grasp_normalize(grasp_feat, offset, scale)
        padding = np.zeros((5*self.cfg.joints - grasp_num, 16), dtype=np.float32)
        grasp_feat = np.concatenate([grasp_feat, padding], axis=0)      # (5*J, 16)

        grasp_joint = grasp_[:, 6].astype(np.int32)                     # (G,)
        grasp_affordance_ = grasp_[:, 7].astype(np.int32)
        if self.cfg.embedding_net.classification:
            grasp_affordance = np.zeros_like(grasp_affordance_, dtype=np.int64)     # (G,)
            grasp_affordance[grasp_affordance_ == 0] = 0
            grasp_affordance[grasp_affordance_ == 1] = len(self.cfg.embedding_net.levels) - 1   # TODO: currently only support hard code
            padding = np.zeros(5*self.cfg.joints - grasp_num, dtype=np.int64)
            grasp_affordance = np.concatenate([grasp_affordance, padding], axis=0)  # (5*J,)
        else:
            grasp_affordance = np.zeros_like(grasp_affordance_, dtype=np.float32)
            grasp_affordance = grasp_affordance_
            padding = np.zeros(5*self.cfg.joints - grasp_num, dtype=np.float32)
            grasp_affordance = np.concatenate([grasp_affordance, padding], axis=0)

        # joint features
        joint_feat_ = []
        for j in range(self.cfg.joints):
            translation = joint_translation[j]
            rotation = joint_direction[j]
            type_feat = 0                   # TODO: currently only support `revolute` joint
            state_feat = 0
            joint_feat_.append(np.concatenate([translation, rotation, [type_feat, state_feat]]))
        joint_feat_ = np.stack(joint_feat_, axis=0)                     # (J, 8)
        if self.cfg.normalization:
            joint_feat_ = joint_normalize(joint_feat_, offset, scale)
        joint_feat = joint_feat_[grasp_joint, :]                        # (G, 8)
        padding = np.zeros((5*self.cfg.joints - grasp_num, 8), dtype=np.float32)
        joint_feat = np.concatenate([joint_feat, padding], axis=0)      # (5*J, 8)

        return pc.astype(np.float32), \
            pc_normal.astype(np.float32), \
            grasp_feat.astype(np.float32), \
            joint_feat.astype(np.float32), \
            grasp_affordance, \
            grasp_num


if __name__ == '__main__':
    cfg = omegaconf.OmegaConf.load('config.yaml')
    dataset = GraspDataset(cfg, is_train=True)
    dataloader = torch.utils.data.DataLoader(dataset, pin_memory=True, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    for pcs, normals, grasps, joints, affordances, grasp_num in tqdm.tqdm(dataloader):
        import pdb; pdb.set_trace()
