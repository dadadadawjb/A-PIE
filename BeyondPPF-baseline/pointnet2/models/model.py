"""
Modified from Pointnet_Pointnet2_pytorch:
https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_cls_msg.py
"""

import omegaconf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction


# normalized point cloud -> normalized joint
class PointNet2(nn.Module):
    def __init__(self, cfg:omegaconf.dictconfig.DictConfig):
        super(PointNet2, self).__init__()
        in_channel = 3 if cfg.pointnet.normal_channel else 0
        self.normal_channel = cfg.pointnet.normal_channel
        self.state = cfg.pointnet.state
        self.joints = cfg.joints
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)

        self.fc3_tr = nn.Linear(256, 3 * self.joints)
        self.fc3_axis = nn.Linear(256, 2 * self.joints)
        self.sigmoid_axis = nn.Sigmoid()
        if self.state:
            self.fc3_state = nn.Linear(256, 1 * self.joints)
            self.sigmoid_state = nn.Sigmoid()

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
        
        trs = self.fc3_tr(x)                        # (B, 3 * joints)
        theta_phi = self.fc3_axis(x)
        theta_phi = self.sigmoid_axis(theta_phi) * 2 * np.pi
        axes = []
        for j in range(self.joints):
            x_axis = torch.sin(theta_phi[:, 0+2*j]) * torch.cos(theta_phi[:, 1+2*j])
            y_axis = torch.sin(theta_phi[:, 0+2*j]) * torch.sin(theta_phi[:, 1+2*j])
            z_axis = torch.cos(theta_phi[:, 0+2*j])
            axis = torch.stack([x_axis, y_axis, z_axis], dim=-1)
            axes.append(axis)
        axes = torch.cat(axes, dim=-1)              # (B, 3 * joints)
        if self.state:
            states = self.fc3_state(x)
            states = self.sigmoid_state(states)     # (B, joints)

        if self.state:
            return trs, axes, states, l3_points
        else:
            return trs, axes, l3_points
