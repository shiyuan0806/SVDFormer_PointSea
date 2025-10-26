import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from pointnet2_ops.pointnet2_utils import gather_operation as gather_points
from models.model_utils import (
    PointNet_SA_Module_KNN,
    MLP_CONV,
    EdgeConv,
    furthest_point_sample,
    get_nearest_index,
    group_local,
    index_points,
    self_attention,
    cross_attention,
    SDG_Decoder,
)
from models.SVDFormer import SVFNet as SVFNet_Base, SDG, local_encoder


def _create_dct_matrix(k: int, device: torch.device) -> torch.Tensor:
    """Create a DCT-II transform matrix of size (k, k)."""
    n = torch.arange(k, dtype=torch.float32, device=device).view(-1, 1)
    m = torch.arange(k, dtype=torch.float32, device=device).view(1, -1)
    coef = torch.cos(torch.pi * (n + 0.5) * m / k)
    coef[:, 0] = coef[:, 0] / torch.sqrt(torch.tensor(2.0, device=device))
    coef = coef * torch.sqrt(torch.tensor(2.0 / k, device=device))
    return coef  # (k, k)


class SpectralAdapter(nn.Module):
    """
    PCSA-like spectral adapter on local KNN patches using a fixed DCT basis
    and learnable frequency gates. Operates along neighbor axis (K) efficiently.
    """

    def __init__(self, in_channels: int, out_channels: int, k_neighbors: int = 16, reduction: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k_neighbors

        # Frequency gates per channel and frequency
        self.freq_gate = nn.Parameter(torch.randn(in_channels, self.k) * 0.02)

        # Geometric attention over neighbors (softmax of learned projection of relative distances)
        self.geo_proj = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)
        )

        # Channel mixing after spectral pooling
        hidden = max(in_channels // reduction, 16)
        self.proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, out_channels, kernel_size=1)
        )

        self.register_buffer('dct_mat', torch.empty(0), persistent=False)

    def _ensure_dct(self, device: torch.device):
        if self.dct_mat.numel() == 0 or self.dct_mat.shape[0] != self.k or self.dct_mat.device != device:
            self.dct_mat = _create_dct_matrix(self.k, device)

    def forward(self, xyz: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: (B, 3, N)
            feats: (B, C, N)
        Returns:
            out: (B, out_channels, N)
        """
        B, C, N = feats.shape
        device = feats.device
        self._ensure_dct(device)

        # KNN group positions and gather features: group_local returns (B, 3, N, K)
        group_xyz, idx = group_local(xyz, k=self.k, return_idx=True)
        # feats -> (B, N, C) for indexing
        feats_bnC = feats.transpose(1, 2).contiguous()  # (B, N, C)
        neigh_feats = index_points(feats_bnC, idx)  # (B, N, K, C)
        neigh_feats = neigh_feats.permute(0, 3, 1, 2).contiguous()  # (B, C, N, K)

        # Geometry attention weights from relative distances (B, 1, N, K)
        dists = torch.norm(group_xyz, dim=1, keepdim=True)  # (B, 1, N, K)
        attn_logits = self.geo_proj(dists)
        attn = torch.softmax(-attn_logits, dim=-1)

        # DCT transform along neighbor axis K
        W = self.dct_mat  # (K, K)
        X = neigh_feats.view(B * C * N, self.k)
        X_hat = torch.matmul(X, W)  # (B*C*N, K)
        # Apply learnable frequency gates
        gamma = self.freq_gate.view(1, C, self.k).repeat(B, 1, 1).view(B * C, self.k)
        X_hat = X_hat.view(B, C, N, self.k)
        X_hat = X_hat * gamma.view(B, C, 1, self.k)
        # Inverse DCT (using transpose since matrix is orthonormal)
        X_filt = torch.matmul(X_hat.view(B * C * N, self.k), W.t())
        X_filt = X_filt.view(B, C, N, self.k)

        # Pool back to per-point with geometry attention
        out = torch.sum(X_filt * attn, dim=-1)  # (B, C, N)

        # Channel projection
        out = self.proj(out)
        return out


class MSGSpecConv(nn.Module):
    """Multi-scale spectral conv over K in {k_list} with fusion."""

    def __init__(self, in_channels: int, out_channels: int, k_list: List[int]):
        super().__init__()
        self.branches = nn.ModuleList([
            SpectralAdapter(in_channels, out_channels, k) for k in k_list
        ])
        self.fuse = nn.Sequential(
            nn.Conv1d(out_channels * len(k_list), out_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, xyz: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        outs = [b(xyz, feats) for b in self.branches]
        out = torch.cat(outs, dim=1)
        out = self.fuse(out)
        return out


class SpectralFeatureExtractor(nn.Module):
    """
    Point feature extractor enhanced by spectral multi-scale graph conv (MSGSpecConv).
    Output is a global feature (B, out_dim, 1) akin to PointNet++ SA encoder.
    """

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.sa_module_2 = PointNet_SA_Module_KNN(128, 16, 128, [128, 256], group_all=False, if_bn=False, if_idx=True)
        self.msg_spec = MSGSpecConv(in_channels=256, out_channels=256, k_list=[16, 32])
        self.sa_module_3 = PointNet_SA_Module_KNN(None, None, 256, [512, out_dim], group_all=True, if_bn=False)

    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        # point_cloud: (B, 3, N)
        l1_xyz, l1_points, idx1 = self.sa_module_1(point_cloud, point_cloud)
        l2_xyz, l2_points, idx2 = self.sa_module_2(l1_xyz, l1_points)  # (B, 3, 128), (B, 256, 128)

        # Spectral refinement at the mid-scale
        spec_refined = self.msg_spec(l2_xyz, l2_points)  # (B, 256, 128)
        l2_points = l2_points + spec_refined

        l3_xyz, l3_points = self.sa_module_3(l2_xyz, l2_points)  # (B, 3, 1), (B, out_dim, 1)
        return l3_points


class SVFNetGS(nn.Module):
    """SVFNet encoder that fuses spectral-enhanced point features and multi-view depth features."""

    def __init__(self, cfg):
        super().__init__()
        self.channel = 64
        self.point_feature_extractor = SpectralFeatureExtractor()
        self.view_distance = cfg.NETWORK.view_distance
        self.relu = nn.GELU()
        self.sa = self_attention(self.channel * 8, self.channel * 8, dropout=0.0)
        self.viewattn = self_attention(128 + 256, 256)

        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)
        self.conv_out1 = nn.Conv1d(512 + self.channel * 4, 64, kernel_size=1)
        self.ps = nn.ConvTranspose1d(512, self.channel, 128, bias=True)
        self.ps_refuse = nn.Conv1d(512 + self.channel, self.channel * 8, kernel_size=1)

        img_layers, in_features = SVFNet_Base.get_img_layers('resnet18', feat_size=16)
        self.img_feature_extractor = nn.Sequential(*img_layers)
        self.posmlp = MLP_CONV(3, [64, 256])

    def forward(self, points: torch.Tensor, depth: torch.Tensor):
        # points: (B, 3, N); depth: (B*V, H, W) with V=3
        batch_size, _, N = points.size()
        f_v = self.img_feature_extractor(depth).view(batch_size, 3, -1).transpose(1, 2).contiguous()
        f_p = self.point_feature_extractor(points)  # (B, 256, 1)

        # View augment and fusion
        view_point = torch.tensor(
            [0, 0, -self.view_distance, -self.view_distance, 0, 0, 0, self.view_distance, 0],
            dtype=torch.float32,
        ).view(-1, 3, 3).permute(0, 2, 1).expand(batch_size, 3, 3).to(depth.device)
        view_feature = self.posmlp(view_point).permute(2, 0, 1)
        f_v_ = self.viewattn(torch.cat([f_v, f_p.repeat(1, 1, f_v.size(2))], 1), view_feature)
        f_v_ = F.adaptive_max_pool1d(f_v_, 1)
        f_g = torch.cat([f_p, f_v_], 1)  # (B, 512, 1)

        x = self.relu(self.ps(f_g))
        x = self.relu(self.ps_refuse(torch.cat([x, f_g.repeat(1, 1, x.size(2))], 1)))
        x2_d = (self.sa(x)).reshape(batch_size, self.channel * 4, N // 8)
        coarse = self.conv_out(self.relu(self.conv_out1(torch.cat([x2_d, f_g.repeat(1, 1, x2_d.size(2))], 1))))

        return f_g, coarse


class Model(nn.Module):
    """GeoSpecNet: spectral + geometry encoder-decoder with DRSN (SDG) refinement."""

    def __init__(self, cfg):
        super().__init__()
        self.encoder = SVFNetGS(cfg)
        self.localencoder = local_encoder(cfg)  # returns (B, 256, M)
        self.merge_points = cfg.NETWORK.merge_points
        self.refine1 = SDG(ratio=cfg.NETWORK.step1, hidden_dim=768, dataset=cfg.DATASET.TEST_DATASET)
        self.refine2 = SDG(ratio=cfg.NETWORK.step2, hidden_dim=512, dataset=cfg.DATASET.TEST_DATASET)

    def forward(self, partial: torch.Tensor, depth: torch.Tensor):
        # partial: (B, N, 3)
        partial = partial.transpose(1, 2).contiguous()
        feat_g, coarse = self.encoder(partial, depth)
        local_feat = self.localencoder(partial)

        coarse_merge = torch.cat([partial, coarse], dim=2)
        coarse_merge = gather_points(
            coarse_merge, furthest_point_sample(coarse_merge.transpose(1, 2).contiguous(), self.merge_points)
        )

        fine1 = self.refine1(local_feat, coarse_merge, feat_g, partial)
        fine2 = self.refine2(local_feat, fine1, feat_g, partial)

        return (
            coarse.transpose(1, 2).contiguous(),
            fine1.transpose(1, 2).contiguous(),
            fine2.transpose(1, 2).contiguous(),
        )


class Discriminator(nn.Module):
    """Point-cloud discriminator using PointNet global features."""

    def __init__(self, feat_size: int = 256):
        super().__init__()
        # A minimal PointNet-like stem
        self.stem = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, feat_size, 1), nn.BatchNorm1d(feat_size), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(feat_size, feat_size // 2), nn.ReLU(inplace=True),
            nn.Linear(feat_size // 2, 1)
        )

    def forward(self, pcd: torch.Tensor):
        # pcd: (B, N, 3)
        x = pcd.transpose(1, 2).contiguous()
        x = self.stem(x)
        x = torch.max(x, dim=2, keepdim=False)[0]
        logit = self.head(x)
        return logit.squeeze(-1)
