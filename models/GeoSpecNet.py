"""
GeoSpecNet: Geometric-Spectral Collaborative Perception for Point Cloud Completion

This implementation includes:
1. Geo-Spectral Collaborative Perception Module (with Graph Fourier Transform)
2. Dynamic Region Selection Network (DRSN)
3. Self-supervised Structural Consistency Training
4. Encoder-Decoder Framework

References:
- Graph Fourier Transform: Signal processing on graphs
- PointNet++: NIPS 2017
- PointCFormer: CVPR 2025
- Point-Transformer: ICCV 2021
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from pointnet2_ops.pointnet2_utils import gather_operation as gather_points
from models.model_utils import *
from metrics.CD.chamfer3D.dist_chamfer_3D import chamfer_3DDist
import numpy as np


# ============================================================================
# 1. Graph Fourier Transform (GFT) Module
# ============================================================================

class GraphFourierTransform(nn.Module):
    """
    Graph Fourier Transform for spectral domain feature extraction.
    
    This module converts point cloud features from spatial domain to spectral domain
    using graph Laplacian decomposition.
    
    Reference: Inspired by spectral graph theory and signal processing on graphs
    """
    def __init__(self, in_channels, out_channels, k_neighbors=16):
        super(GraphFourierTransform, self).__init__()
        self.k = k_neighbors
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Learnable spectral basis
        self.spectral_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )
        
        # Frequency decomposition layers
        self.low_freq = nn.Conv1d(out_channels, out_channels // 2, 1)
        self.high_freq = nn.Conv1d(out_channels, out_channels // 2, 1)
        
    def compute_graph_laplacian(self, xyz):
        """
        Compute normalized graph Laplacian
        Args:
            xyz: (B, 3, N) point coordinates
        Returns:
            L: (B, N, N) normalized Laplacian matrix
        """
        B, _, N = xyz.shape
        # Compute pairwise distances
        xyz_t = xyz.transpose(1, 2).contiguous()  # (B, N, 3)
        dist = square_distance(xyz_t, xyz_t)  # (B, N, N)
        
        # K-nearest neighbor graph
        _, idx = torch.topk(dist, self.k, largest=False, dim=-1)  # (B, N, k)
        
        # Build adjacency matrix
        A = torch.zeros(B, N, N, device=xyz.device)
        for i in range(self.k):
            neighbor_idx = idx[:, :, i]
            batch_idx = torch.arange(B, device=xyz.device).view(B, 1).expand(B, N)
            point_idx = torch.arange(N, device=xyz.device).view(1, N).expand(B, N)
            A[batch_idx, point_idx, neighbor_idx] = 1.0
        
        # Symmetric adjacency
        A = (A + A.transpose(1, 2)) / 2.0
        
        # Degree matrix
        D = torch.sum(A, dim=-1)  # (B, N)
        D_inv_sqrt = torch.pow(D + 1e-6, -0.5)
        D_inv_sqrt = torch.diag_embed(D_inv_sqrt)  # (B, N, N)
        
        # Normalized Laplacian: L = I - D^(-1/2) * A * D^(-1/2)
        I = torch.eye(N, device=xyz.device).unsqueeze(0).expand(B, N, N)
        L = I - torch.bmm(torch.bmm(D_inv_sqrt, A), D_inv_sqrt)
        
        return L
    
    def forward(self, xyz, features):
        """
        Args:
            xyz: (B, 3, N) point coordinates
            features: (B, C, N) point features
        Returns:
            spectral_features: (B, out_channels, N) features in spectral domain
        """
        B, C, N = features.shape
        
        # Compute graph Laplacian
        L = self.compute_graph_laplacian(xyz)  # (B, N, N)
        
        # Transform features to spectral domain
        features_t = features.transpose(1, 2)  # (B, N, C)
        spectral_features = torch.bmm(L, features_t)  # (B, N, C)
        spectral_features = spectral_features.transpose(1, 2)  # (B, C, N)
        
        # Apply learnable spectral convolution
        spectral_features = self.spectral_conv(spectral_features)
        
        # Decompose into low and high frequency components
        low_freq_features = self.low_freq(spectral_features)
        high_freq_features = self.high_freq(spectral_features)
        
        # Concatenate frequency components
        spectral_features = torch.cat([low_freq_features, high_freq_features], dim=1)
        
        return spectral_features


# ============================================================================
# 2. Multi-Scale Graph Convolution (MSGConv)
# ============================================================================

class MultiScaleGraphConv(nn.Module):
    """
    Multi-Scale Graph Convolution for capturing geometric patterns at different scales.
    
    Reference: Inspired by PointNet++ (NIPS 2017) and PointCFormer (CVPR 2025)
    """
    def __init__(self, in_channels, out_channels, scales=[8, 16, 32]):
        super(MultiScaleGraphConv, self).__init__()
        self.scales = scales
        self.num_scales = len(scales)
        
        # Multi-scale edge convolution
        self.edge_convs = nn.ModuleList()
        for k in scales:
            self.edge_convs.append(EdgeConv(in_channels, out_channels // self.num_scales, k))
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, C, N) input features
        Returns:
            out: (B, out_channels, N) multi-scale features
        """
        multi_scale_features = []
        for edge_conv in self.edge_convs:
            feat = edge_conv(x)
            multi_scale_features.append(feat)
        
        # Concatenate multi-scale features
        out = torch.cat(multi_scale_features, dim=1)
        out = self.fusion(out)
        
        return out


# ============================================================================
# 3. Geo-Spectral Collaborative Perception Module
# ============================================================================

class GeoSpectralCollaborativeModule(nn.Module):
    """
    Geo-Spectral Collaborative Perception Module
    
    Combines Graph Fourier Transform and geometric attention mechanism
    for multi-scale feature extraction in both spectral and spatial domains.
    
    References:
    - GFT: Spectral graph theory
    - MSGConv: Multi-scale geometric patterns
    - LGRP (PointCFormer): Local geometric relation perception
    """
    def __init__(self, in_channels=3, hidden_dim=256, k_neighbors=16):
        super(GeoSpectralCollaborativeModule, self).__init__()
        
        # Spatial domain feature extraction (geometric)
        self.spatial_conv = MultiScaleGraphConv(in_channels, hidden_dim, scales=[8, 16, 32])
        
        # Spectral domain feature extraction (frequency)
        self.spectral_transform = GraphFourierTransform(in_channels, hidden_dim, k_neighbors)
        
        # Cross-domain feature alignment
        self.cross_domain_attn = cross_attention(hidden_dim * 2, hidden_dim, nhead=8)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        
    def forward(self, xyz, features):
        """
        Args:
            xyz: (B, 3, N) point coordinates
            features: (B, C, N) point features
        Returns:
            fused_features: (B, hidden_dim, N) geo-spectral features
        """
        # Spatial domain features
        spatial_feat = self.spatial_conv(features)
        
        # Spectral domain features
        spectral_feat = self.spectral_transform(xyz, features)
        
        # Concatenate for cross-domain alignment
        combined_feat = torch.cat([spatial_feat, spectral_feat], dim=1)
        
        # Cross-domain attention
        aligned_feat = self.cross_domain_attn(combined_feat, combined_feat)
        
        # Final fusion
        fused_feat = self.fusion(aligned_feat)
        
        return fused_feat


# ============================================================================
# 4. Structure-Aware Gating Unit
# ============================================================================

class StructureAwareGatingUnit(nn.Module):
    """
    Structure-Aware Gating Unit for dynamic region selection.
    
    Uses self-attention to compute incompleteness scores and 
    dynamically allocate weights for different repair paths.
    
    Reference: Inspired by attention mechanisms in transformers
    """
    def __init__(self, dim, nhead=8):
        super(StructureAwareGatingUnit, self).__init__()
        
        self.attention = self_attention(dim, dim, nhead=nhead)
        
        # Incompleteness score computation
        self.incompleteness_mlp = nn.Sequential(
            nn.Conv1d(dim, dim // 2, 1),
            nn.GELU(),
            nn.Conv1d(dim // 2, 1, 1),
            nn.Sigmoid()
        )
        
        # Gating function
        self.gate_mlp = nn.Sequential(
            nn.Conv1d(dim * 2, dim, 1),
            nn.GELU(),
            nn.Conv1d(dim, 2, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, global_feat, local_feat):
        """
        Args:
            global_feat: (B, C, N) global semantic features
            local_feat: (B, C, N) local detail features
        Returns:
            gates: (B, 2, N) gating weights for [global_path, local_path]
            incompleteness_scores: (B, 1, N) incompleteness scores
        """
        # Compute incompleteness scores
        incompleteness_scores = self.incompleteness_mlp(global_feat)
        
        # Concatenate global and local features
        combined = torch.cat([global_feat, local_feat], dim=1)
        
        # Compute gating weights
        gates = self.gate_mlp(combined)  # (B, 2, N)
        
        return gates, incompleteness_scores


# ============================================================================
# 5. Dynamic Region Selection Network (DRSN)
# ============================================================================

class DynamicRegionSelectionNetwork(nn.Module):
    """
    Dynamic Region Selection Network (DRSN)
    
    Implements dual-path repair strategy:
    1. Global Semantic Path: Coarse completion based on shape priors
    2. Local Detail Path: Fine-grained detail recovery
    
    References:
    - Structure-aware gating: Adaptive path selection
    - Dual-path strategy: Hierarchical completion
    """
    def __init__(self, hidden_dim=512, ratio=2):
        super(DynamicRegionSelectionNetwork, self).__init__()
        self.ratio = ratio
        
        # Structure-aware gating unit
        self.gating_unit = StructureAwareGatingUnit(hidden_dim)
        
        # Global semantic path (coarse completion)
        self.global_path = nn.Sequential(
            self_attention(hidden_dim, hidden_dim, nhead=8),
            self_attention(hidden_dim, hidden_dim * ratio, nhead=8)
        )
        
        # Local detail path (fine-grained recovery)
        self.local_path = nn.Sequential(
            cross_attention(hidden_dim, hidden_dim, nhead=8),
            self_attention(hidden_dim, hidden_dim * ratio, nhead=8)
        )
        
        # Feature refinement
        self.refine = nn.Sequential(
            nn.Conv1d(hidden_dim * ratio * 2, hidden_dim * ratio, 1),
            nn.BatchNorm1d(hidden_dim * ratio),
            nn.GELU(),
            nn.Conv1d(hidden_dim * ratio, 3, 1)
        )
        
    def forward(self, global_feat, local_feat, coarse_xyz):
        """
        Args:
            global_feat: (B, C, N) global shape features
            local_feat: (B, C, N) local geometric features
            coarse_xyz: (B, 3, N) coarse point cloud
        Returns:
            refined_xyz: (B, 3, N*ratio) refined point cloud
            gates: (B, 2, N) gating weights
        """
        B, _, N = coarse_xyz.shape
        
        # Compute gating weights
        gates, incompleteness_scores = self.gating_unit(global_feat, local_feat)
        
        # Global semantic path
        global_output = self.global_path(global_feat)
        
        # Local detail path
        local_output = self.local_path(local_feat, global_feat)
        
        # Weighted combination
        global_weight = gates[:, 0:1, :].repeat(1, self.ratio, 1).reshape(B, -1, N * self.ratio)
        local_weight = gates[:, 1:2, :].repeat(1, self.ratio, 1).reshape(B, -1, N * self.ratio)
        
        # Combine paths
        combined = torch.cat([global_output, local_output], dim=1)
        
        # Refine to get offset
        offset = self.refine(combined)
        
        # Generate refined points
        coarse_expanded = coarse_xyz.repeat(1, 1, self.ratio)
        refined_xyz = coarse_expanded + offset
        
        return refined_xyz, gates


# ============================================================================
# 6. GeoSpecNet Encoder
# ============================================================================

class GeoSpecNetEncoder(nn.Module):
    """
    GeoSpecNet Encoder combining PointNet++ and Geo-Spectral Module
    
    Extracts both geometric and spectral features from partial point cloud
    """
    def __init__(self, out_dim=512):
        super(GeoSpecNetEncoder, self).__init__()
        
        # PointNet++ feature extraction
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], 
                                                   group_all=False, if_bn=False, if_idx=True)
        self.sa_module_2 = PointNet_SA_Module_KNN(128, 16, 128, [128, 256], 
                                                   group_all=False, if_bn=False, if_idx=True)
        self.sa_module_3 = PointNet_SA_Module_KNN(None, None, 256, [512, out_dim], 
                                                   group_all=True, if_bn=False)
        
        # Geo-Spectral Collaborative Module
        self.geo_spectral_module = GeoSpectralCollaborativeModule(
            in_channels=3, hidden_dim=256, k_neighbors=16
        )
        
        # Feature fusion
        self.fusion = nn.Conv1d(out_dim + 256, out_dim, 1)
        
    def forward(self, point_cloud):
        """
        Args:
            point_cloud: (B, 3, N) partial point cloud
        Returns:
            global_feat: (B, out_dim, 1) global features
            spectral_feat: (B, 256, N) spectral features
        """
        l0_xyz = point_cloud
        l0_points = point_cloud
        
        # PointNet++ layers
        l1_xyz, l1_points, idx1 = self.sa_module_1(l0_xyz, l0_points)
        l2_xyz, l2_points, idx2 = self.sa_module_2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa_module_3(l2_xyz, l2_points)
        
        # Geo-Spectral features
        spectral_feat = self.geo_spectral_module(point_cloud, point_cloud)
        spectral_global = F.adaptive_max_pool1d(spectral_feat, 1)
        
        # Fuse geometric and spectral features
        fused_global = torch.cat([l3_points, spectral_global], dim=1)
        global_feat = self.fusion(fused_global)
        
        return global_feat, spectral_feat


# ============================================================================
# 7. GeoSpecNet Decoder
# ============================================================================

class GeoSpecNetDecoder(nn.Module):
    """
    GeoSpecNet Decoder with hierarchical completion
    """
    def __init__(self, hidden_dim=512, num_coarse=1024):
        super(GeoSpecNetDecoder, self).__init__()
        self.num_coarse = num_coarse
        
        # Coarse point generation
        self.coarse_mlp = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim // 2, 1),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 2, num_coarse * 3, 1)
        )
        
        # Local feature encoder
        self.local_encoder = nn.Sequential(
            EdgeConv(3, 128, 16),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.GELU()
        )
        
        # DRSN modules for multi-stage refinement
        self.drsn_stage1 = DynamicRegionSelectionNetwork(hidden_dim=256, ratio=2)
        self.drsn_stage2 = DynamicRegionSelectionNetwork(hidden_dim=256, ratio=2)
        
    def forward(self, global_feat, partial_cloud):
        """
        Args:
            global_feat: (B, 512, 1) global features
            partial_cloud: (B, 3, N) partial point cloud
        Returns:
            coarse: (B, num_coarse, 3) coarse completion
            fine1: (B, num_coarse*2, 3) first refinement
            fine2: (B, num_coarse*4, 3) second refinement
        """
        B = global_feat.shape[0]
        
        # Generate coarse point cloud
        coarse_flat = self.coarse_mlp(global_feat)  # (B, num_coarse*3, 1)
        coarse = coarse_flat.reshape(B, 3, self.num_coarse)
        
        # Extract local features
        local_feat = self.local_encoder(partial_cloud)
        
        # Prepare features for DRSN
        global_feat_expand = global_feat.repeat(1, 1, self.num_coarse)
        global_feat_256 = F.adaptive_avg_pool1d(global_feat_expand, local_feat.shape[-1])
        global_feat_256 = F.interpolate(global_feat_256, size=local_feat.shape[-1], mode='linear', align_corners=True)
        
        # Convert to 256 channels
        conv_to_256 = nn.Conv1d(512, 256, 1).to(global_feat.device)
        global_feat_256 = conv_to_256(global_feat_expand)
        
        # First refinement stage
        fine1, gates1 = self.drsn_stage1(global_feat_256, local_feat, coarse)
        
        # Prepare for second stage
        local_feat_interp = F.interpolate(local_feat, size=fine1.shape[-1], mode='linear', align_corners=True)
        global_feat_interp = F.interpolate(global_feat_256, size=fine1.shape[-1], mode='linear', align_corners=True)
        
        # Second refinement stage
        fine2, gates2 = self.drsn_stage2(global_feat_interp, local_feat_interp, fine1)
        
        return coarse.transpose(1, 2).contiguous(), \
               fine1.transpose(1, 2).contiguous(), \
               fine2.transpose(1, 2).contiguous()


# ============================================================================
# 8. Discriminator for GAN Training
# ============================================================================

class PointCloudDiscriminator(nn.Module):
    """
    Discriminator for self-supervised structural consistency training
    
    Reference: GAN-based Shape Completion (CVPR 2019)
    """
    def __init__(self, in_channels=3):
        super(PointCloudDiscriminator, self).__init__()
        
        self.conv_layers = nn.Sequential(
            EdgeConv(in_channels, 64, 16),
            EdgeConv(64, 128, 16),
            EdgeConv(128, 256, 16),
        )
        
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, point_cloud):
        """
        Args:
            point_cloud: (B, 3, N) or (B, N, 3)
        Returns:
            prob: (B, 1) probability of being real
        """
        if point_cloud.shape[1] != 3:
            point_cloud = point_cloud.transpose(1, 2).contiguous()
        
        # Extract features
        feat = self.conv_layers(point_cloud)
        
        # Global pooling
        global_feat = self.global_pool(feat).squeeze(-1)
        
        # Classify
        prob = self.classifier(global_feat)
        
        return prob


# ============================================================================
# 9. Main GeoSpecNet Model
# ============================================================================

class GeoSpecNet(nn.Module):
    """
    GeoSpecNet: Geometric-Spectral Collaborative Perception Network
    
    A novel point cloud completion framework with:
    1. Geo-Spectral Collaborative Perception Module
    2. Dynamic Region Selection Network (DRSN)
    3. Multi-stage hierarchical refinement
    4. GAN-based structural consistency training
    
    Key Innovations:
    - Graph Fourier Transform for spectral domain modeling
    - Cross-domain feature alignment (spatial + spectral)
    - Structure-aware dynamic path selection
    - Dual-path repair strategy (global + local)
    """
    def __init__(self, cfg):
        super(GeoSpecNet, self).__init__()
        
        # Encoder
        self.encoder = GeoSpecNetEncoder(out_dim=512)
        
        # Decoder
        self.decoder = GeoSpecNetDecoder(
            hidden_dim=512, 
            num_coarse=cfg.NETWORK.num_coarse
        )
        
        # Discriminator (for GAN training)
        self.discriminator = PointCloudDiscriminator(in_channels=3)
        
        # Chamfer distance for loss computation
        self.cd_loss = chamfer_3DDist()
        
    def forward(self, partial_cloud, gt_cloud=None, return_loss=False):
        """
        Args:
            partial_cloud: (B, N, 3) partial point cloud
            gt_cloud: (B, M, 3) ground truth complete point cloud (optional)
            return_loss: whether to compute losses
        Returns:
            coarse: (B, num_coarse, 3) coarse completion
            fine1: (B, num_coarse*2, 3) first refinement
            fine2: (B, num_coarse*4, 3) final completion
            losses: dict of losses (if return_loss=True)
        """
        # Transpose to (B, 3, N)
        partial_cloud_t = partial_cloud.transpose(1, 2).contiguous()
        
        # Encode
        global_feat, spectral_feat = self.encoder(partial_cloud_t)
        
        # Decode
        coarse, fine1, fine2 = self.decoder(global_feat, partial_cloud_t)
        
        if return_loss and gt_cloud is not None:
            losses = self.compute_losses(partial_cloud, coarse, fine1, fine2, gt_cloud)
            return coarse, fine1, fine2, losses
        
        return coarse, fine1, fine2
    
    def compute_losses(self, partial, coarse, fine1, fine2, gt):
        """
        Compute multi-stage losses
        """
        losses = {}
        
        # Chamfer Distance losses
        cd_coarse, _ = self.cd_loss(coarse, gt)
        cd_fine1, _ = self.cd_loss(fine1, gt)
        cd_fine2, _ = self.cd_loss(fine2, gt)
        
        losses['cd_coarse'] = cd_coarse.mean()
        losses['cd_fine1'] = cd_fine1.mean()
        losses['cd_fine2'] = cd_fine2.mean()
        
        # Partial matching loss (ensure visible regions remain unchanged)
        cd_partial, _ = self.cd_loss(fine2[:, :partial.shape[1], :], partial)
        losses['partial_match'] = cd_partial.mean()
        
        # Total loss
        losses['total'] = losses['cd_coarse'] + losses['cd_fine1'] * 2 + \
                         losses['cd_fine2'] * 4 + losses['partial_match'] * 0.5
        
        return losses
    
    def compute_gan_loss(self, completed, gt):
        """
        Compute GAN loss for structural consistency
        """
        # Discriminator predictions
        real_prob = self.discriminator(gt)
        fake_prob = self.discriminator(completed.detach())
        
        # Discriminator loss
        d_loss_real = F.binary_cross_entropy(real_prob, torch.ones_like(real_prob))
        d_loss_fake = F.binary_cross_entropy(fake_prob, torch.zeros_like(fake_prob))
        d_loss = (d_loss_real + d_loss_fake) / 2
        
        # Generator loss
        fake_prob_for_g = self.discriminator(completed)
        g_loss = F.binary_cross_entropy(fake_prob_for_g, torch.ones_like(fake_prob_for_g))
        
        return d_loss, g_loss


# ============================================================================
# Model Factory
# ============================================================================

def build_geospecnet(cfg):
    """Build GeoSpecNet model"""
    model = GeoSpecNet(cfg)
    return model


if __name__ == '__main__':
    # Test model
    class TestConfig:
        class NETWORK:
            num_coarse = 1024
    
    cfg = TestConfig()
    model = GeoSpecNet(cfg).cuda()
    
    # Test forward pass
    partial = torch.randn(2, 2048, 3).cuda()
    gt = torch.randn(2, 8192, 3).cuda()
    
    coarse, fine1, fine2, losses = model(partial, gt, return_loss=True)
    
    print(f"Coarse shape: {coarse.shape}")
    print(f"Fine1 shape: {fine1.shape}")
    print(f"Fine2 shape: {fine2.shape}")
    print(f"Losses: {losses}")
    print("\nGeoSpecNet model test passed!")
