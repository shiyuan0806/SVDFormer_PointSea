from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointCloudDiscriminator(nn.Module):
    """A simple PointNet-style discriminator for point sets.

    Input: (B, N, 3)
    Output: (B, 1) logits
    """

    def __init__(self, feat_dim: int = 3):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Conv1d(feat_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x_bpn: torch.Tensor) -> torch.Tensor:
        # x_bpn: (B, N, 3)
        x = x_bpn.transpose(1, 2).contiguous()  # (B, 3, N)
        f = self.point_mlp(x)  # (B, 256, N)
        g = torch.max(f, dim=2, keepdim=False)[0]  # (B, 256)
        logits = self.global_mlp(g)  # (B, 1)
        return logits
