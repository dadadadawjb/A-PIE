import omegaconf
from itertools import combinations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResLayer(nn.Module):
    def __init__(self, dim_in:int, dim_out:int, bn:bool=False, ln:bool=False, dropout:float=0.):
        super().__init__()
        self.is_bn = bn
        self.is_ln = ln
        self.fc1 = nn.Linear(dim_in, dim_out)
        if bn:
            self.bn1 = nn.BatchNorm1d(dim_out)
        else:
            self.bn1 = lambda x: x
        if ln:
            self.ln1 = nn.LayerNorm(dim_out)
        else:
            self.ln1 = lambda x: x
        self.fc2 = nn.Linear(dim_out, dim_out)
        if bn:
            self.bn2 = nn.BatchNorm1d(dim_out)
        else:
            self.bn2 = lambda x: x
        if ln:
            self.ln2 = nn.LayerNorm(dim_out)
        else:
            self.ln2 = lambda x: x
        if dim_in != dim_out:
            self.fc0 = nn.Linear(dim_in, dim_out)
        else:
            self.fc0 = None
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
    
    def forward(self, x):
        x_res = x if self.fc0 is None else self.fc0(x)
        x = self.fc1(x)
        if len(x.shape) > 3 or len(x.shape) < 2:
            raise ValueError("x.shape should be (B, N, D) or (N, D)")
        elif len(x.shape) == 3 and self.is_bn:
            x = x.permute(0, 2, 1)      # from (B, N, D) to (B, D, N)
            x = self.bn1(x)
            x = x.permute(0, 2, 1)      # from (B, D, N) to (B, N, D)
        elif len(x.shape) == 2 and self.is_bn:
            x = self.bn1(x)
        elif self.is_ln:
            x = self.ln1(x)
        else:
            x = self.bn1(x)             # actually self.bn1 is identity function
        x = F.relu(x)

        x = self.fc2(x)
        if len(x.shape) > 3 or len(x.shape) < 2:
            raise ValueError("x.shape should be (B, N, D) or (N, D)")
        elif len(x.shape) == 3 and self.is_bn:
            x = x.permute(0, 2, 1)      # from (B, N, D) to (B, D, N)
            x = self.bn2(x)
            x = x.permute(0, 2, 1)      # from (B, D, N) to (B, N, D)
        elif len(x.shape) == 2 and self.is_bn:
            x = self.bn2(x)
        elif self.is_ln:
            x = self.ln2(x)
        else:
            x = self.bn2(x)             # actually self.bn2 is identity function
        x = self.dropout(x + x_res)
        return x


def create_shot_encoder(cfg:omegaconf.dictconfig.DictConfig) -> nn.Module:
    fcs_shot = cfg.shot_encoder.hidden_dims
    fcs_shot.insert(0, 352)
    fcs_shot.append(cfg.shot_encoder.feature_dim)
    shot_encoder = nn.Sequential(
        *[ResLayer(fcs_shot[i], fcs_shot[i + 1], bn=cfg.shot_encoder.bn, ln=cfg.shot_encoder.ln, dropout=cfg.shot_encoder.dropout) 
          for i in range(len(fcs_shot) - 1)]
    ).cuda(cfg.device)
    return shot_encoder

def create_encoder(cfg:omegaconf.dictconfig.DictConfig) -> nn.Module:
    fcs = cfg.encoder.hidden_dims
    # input order: (coords, normals, shots)
    fcs.insert(0, len(list(combinations(np.arange(cfg.num_more + 2), 2))) * 4 + (cfg.num_more + 2) * cfg.shot_encoder.feature_dim)
    # output order: (J*tr, J*rot(, J*state), J*conf)
    fcs.append(((2 + cfg.encoder.rot_num_bins + 1 + 1) * cfg.joints) if cfg.encoder.state else ((2 + cfg.encoder.rot_num_bins + 1) * cfg.joints))
    encoder = nn.Sequential(
        *[ResLayer(fcs[i], fcs[i + 1], bn=cfg.encoder.bn, ln=cfg.encoder.ln, dropout=cfg.encoder.dropout) 
          for i in range(len(fcs) - 1)]
    ).cuda(cfg.device)
    return encoder
