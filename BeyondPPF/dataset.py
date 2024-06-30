import os
import json
import omegaconf
import itertools
from pathlib import Path
import tqdm
import numpy as np
import torch
import MinkowskiEngine as ME

from utils import farthest_point_sample
from src_shot.build import shot


class ArticulationDataset(torch.utils.data.Dataset):
    def __init__(self, cfg:omegaconf.dictconfig.DictConfig, is_train:bool):
        super().__init__()
        self.cfg = cfg
        self.is_train = is_train
        if is_train:
            instances = cfg.dataset.train_instances
        else:
            instances = cfg.dataset.test_instances
        self.fns = sorted(list(itertools.chain(*[list(Path(cfg.dataset.dataset_path).glob('{}/*/*.npz'.format(instance))) for instance in instances])))

    def __len__(self):
        return len(self.fns)
    
    def __getitem__(self, idx:int):
        data = np.load(self.fns[idx])
        pc = data['point_cloud']
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

        # compute SHOT352 features
        possible_feature_path = self.fns[idx].parent / f"{self.fns[idx].name.replace('.npz', '')}_shot_{self.cfg.shot.res * self.cfg.shot.receptive_field}_{self.cfg.shot.res * self.cfg.shot.receptive_field}_{self.cfg.shot.res}.npy"
        if os.path.exists(possible_feature_path):
            pc_feat = np.load(possible_feature_path)
        else:
            pc_feat = shot.compute(pc, self.cfg.shot.res * self.cfg.shot.receptive_field, self.cfg.shot.res * self.cfg.shot.receptive_field).reshape(-1, 352)
            pc_feat[~np.isfinite(pc_feat)] = 0                          # (N', 352)
            if self.cfg.cache:
                np.save(possible_feature_path, pc_feat)

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
        pc_feat = pc_feat[indices]                                      # (N, 352)
        pc_normal = pc_normal[indices]                                  # (N, 3)

        # sample point pairs
        num_sample_pairs = self.cfg.sample_points if self.is_train else self.cfg.test_sample_points
        point_idxs = np.random.randint(0, pc.shape[0], size=(num_sample_pairs, 2))  # (sample_points, 2)

        # generate targets
        targets_tr, targets_rot, targets_state = [], [], []
        translations, rotations = [], []
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
            target_tr = self.generate_target_tr(pc, translation, point_idxs)           # (sample_points, 2)
            target_rot = self.generate_target_rot(pc, rotation, point_idxs)            # (sample_points,)
            assert joint_type[j] == self.cfg.types[j]
            if joint_type[j] == 'revolute':
                target_state = np.array(joint_state[j] - self.cfg.states[j] / 180.0 * np.pi / 2)    # ()
            elif joint_type[j] == 'prismatic':
                target_state = np.array(joint_state[j] - self.cfg.states[j] / 100.0 / 2)
            else:
                raise ValueError('Invalid joint_type: {}'.format(joint_type[j]))
            targets_tr.append(target_tr)
            targets_rot.append(target_rot)
            targets_state.append(target_state)
            translations.append(translation)
            rotations.append(rotation)
        targets_tr = np.stack(targets_tr, axis=0)                                     # (joints, sample_points, 2)
        targets_rot = np.stack(targets_rot, axis=0)                                   # (joints, sample_points)
        targets_state = np.stack(targets_state, axis=0)                               # (joints,)
        translations = np.stack(translations, axis=0)                                 # (joints, 3)
        rotations = np.stack(rotations, axis=0)                                       # (joints, 3)
    
        if self.is_train:
            return pc.astype(np.float32), \
                pc_feat.astype(np.float32), \
                pc_normal.astype(np.float32), \
                targets_tr.astype(np.float32), \
                targets_rot.astype(np.float32), \
                targets_state.astype(np.float32), \
                point_idxs.astype(int)
        else:
            return pc.astype(np.float32), \
                pc_feat.astype(np.float32), \
                pc_normal.astype(np.float32), \
                translations.astype(np.float32), \
                rotations.astype(np.float32), \
                targets_state.astype(np.float32), \
                point_idxs.astype(int)

    def generate_target_tr(self, pc:np.ndarray, o:np.ndarray, point_idxs:np.ndarray) -> np.ndarray:
        a = pc[point_idxs[:, 0]]    # (sample_points, 3)
        b = pc[point_idxs[:, 1]]    # (sample_points, 3)
        pdist = a - b
        pdist_unit = pdist / (np.linalg.norm(pdist, axis=-1, keepdims=True) + 1e-7)
        proj_len = np.sum((a - o) * pdist_unit, -1)
        oc = a - o - proj_len[..., None] * pdist_unit
        dist2o = np.linalg.norm(oc, axis=-1)
        target_tr = np.stack([proj_len, dist2o], -1)
        return target_tr.astype(np.float32).reshape((-1, 2))
    
    def generate_target_rot(self, pc:np.ndarray, axis:np.ndarray, point_idxs:np.ndarray) -> np.ndarray:
        a = pc[point_idxs[:, 0]]    # (sample_points, 3)
        b = pc[point_idxs[:, 1]]    # (sample_points, 3)
        pdist = a - b
        pdist_unit = pdist / (np.linalg.norm(pdist, axis=-1, keepdims=True) + 1e-7)
        target_rot = np.arccos(np.sum(pdist_unit * axis, -1))
        return target_rot.astype(np.float32).reshape((-1,))


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

        # sparse quantize
        possible_quantize_path = self.fns[idx].parent / f"{self.fns[idx].name.replace('.npz', '')}_quantize_{self.cfg.shot.res}.npy"
        if os.path.exists(possible_quantize_path):
            pc = np.load(possible_quantize_path)
        else:
            indices = ME.utils.sparse_quantize(np.ascontiguousarray(pc), return_index=True, quantization_size=self.cfg.shot.res)[1]
            pc = np.ascontiguousarray(pc[indices].astype(np.float32))   # (N', 3)
            if self.cfg.cache:
                np.save(possible_quantize_path, pc)

        # compute SHOT352 features
        possible_feature_path = self.fns[idx].parent / f"{self.fns[idx].name.replace('.npz', '')}_shot_{self.cfg.shot.res * self.cfg.shot.receptive_field}_{self.cfg.shot.res * self.cfg.shot.receptive_field}_{self.cfg.shot.res}.npy"
        if os.path.exists(possible_feature_path):
            pc_feat = np.load(possible_feature_path)
        else:
            pc_feat = shot.compute(pc, self.cfg.shot.res * self.cfg.shot.receptive_field, self.cfg.shot.res * self.cfg.shot.receptive_field).reshape(-1, 352)
            pc_feat[~np.isfinite(pc_feat)] = 0                          # (N', 352)
            if self.cfg.cache:
                np.save(possible_feature_path, pc_feat)

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
        pc_feat = pc_feat[indices]                                      # (N, 352)
        pc_normal = pc_normal[indices]                                  # (N, 3)

        # sample point pairs
        num_sample_pairs = self.cfg.sample_points if self.is_train else self.cfg.test_sample_points
        point_idxs = np.random.randint(0, pc.shape[0], size=(num_sample_pairs, 2))  # (sample_points, 2)

        # generate targets
        targets_tr, targets_rot, targets_state = [], [], []
        translations, rotations = [], []
        for j in range(self.cfg.joints):
            translation = joint_translation[j]
            rotation = joint_direction[j]
            target_tr = self.generate_target_tr(pc, translation, point_idxs)           # (sample_points, 2)
            target_rot = self.generate_target_rot(pc, rotation, point_idxs)            # (sample_points,)
            target_state = np.array(0)    # ()
            targets_tr.append(target_tr)
            targets_rot.append(target_rot)
            targets_state.append(target_state)
            translations.append(translation)
            rotations.append(rotation)
        targets_tr = np.stack(targets_tr, axis=0)                                     # (joints, sample_points, 2)
        targets_rot = np.stack(targets_rot, axis=0)                                   # (joints, sample_points)
        targets_state = np.stack(targets_state, axis=0)                               # (joints,)
        translations = np.stack(translations, axis=0)                                 # (joints, 3)
        rotations = np.stack(rotations, axis=0)                                       # (joints, 3)
    
        if self.is_train:
            return pc.astype(np.float32), \
                pc_feat.astype(np.float32), \
                pc_normal.astype(np.float32), \
                targets_tr.astype(np.float32), \
                targets_rot.astype(np.float32), \
                targets_state.astype(np.float32), \
                point_idxs.astype(int)
        else:
            return pc.astype(np.float32), \
                pc_feat.astype(np.float32), \
                pc_normal.astype(np.float32), \
                translations.astype(np.float32), \
                rotations.astype(np.float32), \
                targets_state.astype(np.float32), \
                point_idxs.astype(int)

    def generate_target_tr(self, pc:np.ndarray, o:np.ndarray, point_idxs:np.ndarray) -> np.ndarray:
        a = pc[point_idxs[:, 0]]    # (sample_points, 3)
        b = pc[point_idxs[:, 1]]    # (sample_points, 3)
        pdist = a - b
        pdist_unit = pdist / (np.linalg.norm(pdist, axis=-1, keepdims=True) + 1e-7)
        proj_len = np.sum((a - o) * pdist_unit, -1)
        oc = a - o - proj_len[..., None] * pdist_unit
        dist2o = np.linalg.norm(oc, axis=-1)
        target_tr = np.stack([proj_len, dist2o], -1)
        return target_tr.astype(np.float32).reshape((-1, 2))
    
    def generate_target_rot(self, pc:np.ndarray, axis:np.ndarray, point_idxs:np.ndarray) -> np.ndarray:
        a = pc[point_idxs[:, 0]]    # (sample_points, 3)
        b = pc[point_idxs[:, 1]]    # (sample_points, 3)
        pdist = a - b
        pdist_unit = pdist / (np.linalg.norm(pdist, axis=-1, keepdims=True) + 1e-7)
        target_rot = np.arccos(np.sum(pdist_unit * axis, -1))
        return target_rot.astype(np.float32).reshape((-1,))


if __name__ == '__main__':
    cfg = omegaconf.OmegaConf.load('config.yaml')
    dataset = ArticulationDataset(cfg, is_train=True)
    dataloader = torch.utils.data.DataLoader(dataset, pin_memory=True, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    for pcs, pc_feats, normals, targets_tr, targets_rot, targets_state, point_idxs in tqdm.tqdm(dataloader):
        import pdb; pdb.set_trace()
    
    cfg = omegaconf.OmegaConf.load('config_looptune.yaml')
    dataset = LoopDataset(cfg, is_train=True)
    dataloader = torch.utils.data.DataLoader(dataset, pin_memory=True, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    for pcs, pc_feats, normals, targets_tr, targets_rot, targets_state, point_idxs in tqdm.tqdm(dataloader):
        import pdb; pdb.set_trace()
