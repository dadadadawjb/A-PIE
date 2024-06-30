import os
import json
import omegaconf
import itertools
from pathlib import Path
import numpy as np
import torch
import MinkowskiEngine as ME

from src_shot.build import shot
from utils import pc_normalize, joint_normalize, farthest_point_sample


# (normalized point cloud, normalized joint)
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

        # estimate normal
        if self.cfg.pointnet.normal_channel:
            possible_normal_path = self.fns[idx].parent / f"{self.fns[idx].name.replace('.npz', '')}_normal_{self.cfg.shot.res * self.cfg.shot.receptive_field}_{self.cfg.shot.res}.npy"
            if os.path.exists(possible_normal_path):
                pc_normal = np.load(possible_normal_path)
            else:
                pc_normal = shot.estimate_normal(pc, self.cfg.shot.res * self.cfg.shot.receptive_field).reshape(-1, 3).astype(np.float32)
                pc_normal[~np.isfinite(pc_normal)] = 0                  # (N', 3)
                if self.cfg.cache:
                    np.save(possible_normal_path, pc_normal)
            pc = np.concatenate([pc, pc_normal], axis=1)                # (N', 6)
        
        # sample points
        num_sample_points = self.cfg.sample_points if self.is_train else self.cfg.test_sample_points
        possible_sample_path = self.fns[idx].parent / f"{self.fns[idx].name.replace('.npz', '')}_sample_{num_sample_points}_{self.cfg.shot.res}.npy"
        if os.path.exists(possible_sample_path):
            indices = np.load(possible_sample_path)
            pc = pc[indices]
        else:
            pc, indices = farthest_point_sample(pc, num_sample_points)  # (N, 3/6)
            if self.cfg.cache:
                np.save(possible_sample_path, indices)
        
        # normalize
        pc[:, 0:3], offset, scale = pc_normalize(pc[:, 0:3])            # (3,), (1,)

        # generate targets
        translations, rotations, states = [], [], []
        real_translations, real_rotations, real_states = [], [], []
        for j in range(self.cfg.joints):
            real_translation = joint_pose[j, :3, -1]
            if joint_axis_which[j] == 'x':
                real_rotation = joint_pose[j, :3, 0]
            elif joint_axis_which[j] == 'y':
                real_rotation = joint_pose[j, :3, 1]
            elif joint_axis_which[j] == 'z':
                real_rotation = joint_pose[j, :3, 2]
            else:
                raise ValueError('Invalid joint_axis_which: {}'.format(joint_axis_which[j]))
            assert joint_type[j] == self.cfg.types[j]
            if self.cfg.pointnet.state:
                if joint_type[j] == 'revolute':
                    real_state = np.array(joint_state[j] / (self.cfg.states[j] / 180.0 * np.pi))
                elif joint_type[j] == 'prismatic':
                    real_state = np.array(joint_state[j] / (self.cfg.states[j] / 100.0))
                else:
                    raise ValueError('Invalid joint_type: {}'.format(joint_type[j]))
            else:
                real_state = np.array(0.0)
            translation, rotation, state = joint_normalize(real_translation, real_rotation, real_state, offset, scale)
            translations.append(translation)
            rotations.append(rotation)
            states.append(state)
            real_translations.append(real_translation)
            real_rotations.append(real_rotation)
            real_states.append(real_state)
        translations = np.stack(translations, axis=0)               # (joints, 3)
        rotations = np.stack(rotations, axis=0)                     # (joints, 3)
        states = np.stack(states, axis=0)                           # (joints,)
        real_translations = np.stack(real_translations, axis=0)     # (joints, 3)
        real_rotations = np.stack(real_rotations, axis=0)           # (joints, 3)
        real_states = np.stack(real_states, axis=0)                 # (joints,)

        if self.is_train:
            return pc.astype(np.float32), \
                translations.astype(np.float32), \
                rotations.astype(np.float32), \
                states.astype(np.float32)
        else:
            return pc.astype(np.float32), \
                real_translations.astype(np.float32), \
                real_rotations.astype(np.float32), \
                real_states.astype(np.float32), \
                offset.astype(np.float32), \
                scale.astype(np.float32)


if __name__ == '__main__':
    config = omegaconf.OmegaConf.load('config.yaml')
    dataset = ArticulationDataset(config, True)
    for d in dataset:
        import pdb; pdb.set_trace()
