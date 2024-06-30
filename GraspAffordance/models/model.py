import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction


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


class Pointfeat(nn.Module):
    """
    Modified from Pointnet_Pointnet2_pytorch:
    https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_cls_msg.py
    """
    def __init__(self, cfg:omegaconf.dictconfig.DictConfig):
        super(Pointfeat, self).__init__()
        
        in_channel = 3 if cfg.point_encoder.normal_channel else 0
        self.normal_channel = cfg.point_encoder.normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, cfg.point_encoder.feature_dim)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x


class Graspfeat(nn.Module):
    def __init__(self, cfg:omegaconf.dictconfig.DictConfig):
        super(Graspfeat, self).__init__()
        
        dims = cfg.grasp_encoder.hidden_dims
        dims.insert(0, cfg.grasp_encoder.grasp_dim)
        dims.append(cfg.grasp_encoder.feature_dim)
        self.linears = nn.Sequential(
            *[ResLayer(dims[i], dims[i + 1], bn=cfg.grasp_encoder.bn, ln=cfg.grasp_encoder.ln, dropout=cfg.grasp_encoder.dropout) 
            for i in range(len(dims) - 1)]
        )

    def forward(self, x):
        x = self.linears(x)
        return x


class Jointfeat(nn.Module):
    def __init__(self, cfg:omegaconf.dictconfig.DictConfig) -> None:
        super(Jointfeat, self).__init__()
        
        dims = cfg.joint_encoder.hidden_dims
        dims.insert(0, 8 if cfg.joint_encoder.state_channel else 7)
        dims.append(cfg.joint_encoder.feature_dim)
        self.linears = nn.Sequential(
            *[ResLayer(dims[i], dims[i + 1], bn=cfg.joint_encoder.bn, ln=cfg.joint_encoder.ln, dropout=cfg.joint_encoder.dropout) 
            for i in range(len(dims) - 1)]
        )
    
    def forward(self, x):
        x = self.linears(x)
        return x


class grasp_embedding_network(nn.Module):
    def __init__(self, cfg:omegaconf.dictconfig.DictConfig):
        super(grasp_embedding_network, self).__init__()
        self.point_feat = Pointfeat(cfg)
        self.grasp_feat = Graspfeat(cfg)
        self.joint_feat = Jointfeat(cfg)

        dims = cfg.embedding_net.hidden_dims
        dims.insert(0, cfg.point_encoder.feature_dim + cfg.grasp_encoder.feature_dim + cfg.joint_encoder.feature_dim)
        dims.append(len(cfg.embedding_net.levels) if cfg.embedding_net.classification else 1)
        self.linears = nn.Sequential(
            *[ResLayer(dims[i], dims[i + 1], bn=cfg.embedding_net.bn, ln=cfg.embedding_net.ln, dropout=cfg.embedding_net.dropout) 
            for i in range(len(dims) - 1)]
        )

    def forward(self, pcs, grasps, joints):
        assert (len(pcs.shape) == 3) and (len(grasps.shape) == 3) and (len(joints.shape) == 3)
        grasp_num = grasps.shape[1]
        assert grasps.shape[1] == joints.shape[1]

        pcs = pcs.transpose(1, 2).contiguous()
        point_feat = self.point_feat(pcs)       # (B, N, 3/6) -> (B, dim1)
        grasp_feat = self.grasp_feat(grasps)    # (B, G, 16) -> (B, G, dim2)
        joint_feat = self.joint_feat(joints)    # (B, G, 7/8) -> (B, G, dim3)

        point_feat = point_feat[:, None, :].repeat(1, grasp_num, 1)
        x = torch.cat([point_feat, grasp_feat, joint_feat], dim=-1) # (B, G, dim1+dim2+dim3)

        x = self.linears(x) # (B, G, c)

        return x
