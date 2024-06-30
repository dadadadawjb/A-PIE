from typing import List, Tuple, Union, Optional
import omegaconf
from tqdm import tqdm
import argparse
import torch
import torch.nn.functional as F
import numpy as np

from models.model import grasp_embedding_network
from dataset import GraspDataset

def count_hit(prediction:Union[torch.Tensor, np.ndarray], label:Union[torch.Tensor, np.ndarray], 
              levels:List[List[float]]) -> int:
    """
    prediction: (G,) with value in [0, 1]
    label: (G,) with value in (levels[:, 0], levels[:, 1]]
    levels: [[level_min, level_max], ...] with length >= 2
    hit: [0, G]
    """
    hit = 0
    if isinstance(prediction, np.ndarray):
        prediction = torch.from_numpy(prediction)
    if isinstance(label, np.ndarray):
        label = torch.from_numpy(label)
    for level_idx, level in enumerate(levels):
        if level_idx == 0:
            pred_level_mask = torch.logical_and((prediction >= level[0]), (prediction <= level[1]))
            label_level_mask = torch.logical_and((label >= level[0]), (label <= level[1]))
        else:
            pred_level_mask = torch.logical_and((prediction > level[0]), (prediction <= level[1]))
            label_level_mask = torch.logical_and((label > level[0]), (label <= level[1]))
        hit += torch.sum(torch.logical_and(pred_level_mask, label_level_mask))
    return int(hit)

def count_affordable(prediction:Union[torch.Tensor, np.ndarray], label:Union[torch.Tensor, np.ndarray], 
              levels:List[Optional[List[float]]]) -> Tuple[int, int]:
    """
    prediction: (G,) with value in [0, 1]
    label: (G,) with value as True or False
    levels: [[level_min, level_max], ...]
    true_affordable, false_affordable: [0, G]
    """
    if isinstance(prediction, np.ndarray):
        prediction = torch.from_numpy(prediction)
    if isinstance(label, np.ndarray):
        label = torch.from_numpy(label)
    pred_affordance_affordable = torch.zeros_like(prediction, dtype=bool)
    for level in levels:
        if level is not None:
            pred_affordance_affordable[torch.logical_and((prediction > level[0]), (prediction <= level[1]))] = True
    true_affordable = torch.sum(torch.logical_and(pred_affordance_affordable, label)).item()
    false_affordable = torch.sum(torch.logical_and(pred_affordance_affordable, torch.logical_not(label))).item()
    return (true_affordable, false_affordable)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, required=True, help='the path to the weight directory')
    parser.add_argument('--device', type=int, help='the device to use for testing')
    args = parser.parse_args()
    cfg = omegaconf.OmegaConf.load(f"{args.weight_path}/.hydra/config.yaml")
    if args.device is not None:
        cfg.device = args.device
    f = open(f"{args.weight_path}/test.log", "w", encoding="utf-8")
    
    # load network
    model = grasp_embedding_network(cfg).to(cfg.device)
    model.load_state_dict(torch.load(f'{args.weight_path}/model_latest.pth'))
    model.eval()

    # load data
    for train_test in [False, True]:
        # TODO: in looptune, you need to only `False` for `train_test`
        dataset = GraspDataset(cfg, train_test)
        dataloader = torch.utils.data.DataLoader(dataset, pin_memory=True, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
        
        hit = 0
        num = 0
        true_affordable, false_affordable = 0, 0
        with tqdm(dataloader) as t:
            for pcs, normals, grasps, joints, affordances, grasp_num in t:
                pcs, normals, grasps, joints, affordances = \
                    pcs.cuda(cfg.device), normals.cuda(cfg.device), grasps.cuda(cfg.device), joints.cuda(cfg.device), affordances.cuda(cfg.device)
                # (B, N, 3), (B, N, 3), (B, 100*J, 16), (B, 100*J, 8), (B, 100*J), (B,)
                
                if cfg.point_encoder.normal_channel:
                    pcs = torch.cat([pcs, normals], dim=-1)             # (B, N, 3/6)
                if not cfg.joint_encoder.state_channel:
                    joints = joints[..., :-1]                           # (B, 100*J, 7/8)
                
                with torch.no_grad():
                    prediction = model(pcs, grasps, joints)             # (B, 100*J, c)
                    if cfg.embedding_net.classification:
                        prediction = F.softmax(prediction, dim=-1)      # (B, 100*J, c)
                        prediction = torch.argmax(prediction, dim=-1)   # (B, 100*J)
                    else:
                        prediction = prediction.squeeze(-1)             # (B, 100*J)
                        prediction = torch.sigmoid(prediction)          # (B, 100*J)
                
                for b in range(grasp_num.shape[0]):
                    if cfg.embedding_net.classification:
                        hit += torch.sum(prediction[b, :int(grasp_num[b])] == affordances[b, :int(grasp_num[b])]).item()
                        pred_affordance = prediction[b, :int(grasp_num[b])].cpu().numpy()
                        pred_affordance_affordable = np.zeros_like(pred_affordance, dtype=bool)
                        grasp_affordance = affordances[b, :int(grasp_num[b])].cpu().numpy()
                        grasp_affordance_affordable = np.zeros_like(grasp_affordance, dtype=bool)
                        for level_idx in range(len(cfg.embedding_net.levels)):
                            for level_item in cfg.embedding_net.levels[level_idx][0]:
                                if level_item == 1 or level_item == 2:
                                    pred_affordance_affordable[pred_affordance == level_idx] = True
                                    grasp_affordance_affordable[grasp_affordance == level_idx] = True
                        true_affordable += np.sum(np.logical_and(pred_affordance_affordable, grasp_affordance_affordable))
                        false_affordable += np.sum(np.logical_and(pred_affordance_affordable, np.logical_not(grasp_affordance_affordable)))
                    else:
                        hit += count_hit(prediction[b, :int(grasp_num[b])], affordances[b, :int(grasp_num[b])], 
                                        [level[2] for level in cfg.embedding_net.levels])
                        grasp_affordance = affordances[b, :int(grasp_num[b])].cpu().numpy()
                        grasp_affordance_affordable = np.zeros_like(grasp_affordance, dtype=bool)
                        for level_idx in range(len(cfg.embedding_net.levels)):
                            for level_item in cfg.embedding_net.levels[level_idx][0]:
                                if level_item == 1 or level_item == 2:
                                    grasp_affordance_affordable[np.abs(grasp_affordance - cfg.embedding_net.levels[level_idx][1]) < 0.1] = True
                        this_true_affordable, this_false_affordable = count_affordable(prediction[b, :int(grasp_num[b])].cpu(), grasp_affordance_affordable, 
                                                                                       [level[2] if 1 in level[0] or 2 in level[0] else None for level in cfg.embedding_net.levels])
                        true_affordable += this_true_affordable
                        false_affordable += this_false_affordable
                num += int(torch.sum(grasp_num))
        
        if train_test:
            print("train acc:", hit/num)
            print("train precision:", true_affordable/(true_affordable+false_affordable))
            print("train acc:", hit/num, file=f)
            print("train precision:", true_affordable/(true_affordable+false_affordable), file=f)
        else:
            print("test acc:", hit/num)
            print("test precision:", true_affordable/(true_affordable+false_affordable))
            print("test acc:", hit/num, file=f)
            print("test precision:", true_affordable/(true_affordable+false_affordable), file=f)
