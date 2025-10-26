"""
GeoSpecNet: Point Cloud Completion with Spectral Domain Enhancement
Combines:
1. Point Cloud Spectral Adapter (PCSA) from PointGST
2. Geo-Spectral Collaborative Perception Module with Graph Fourier Transform
3. Dynamic Region Selection Network (DRSN)
4. Multi-view fusion from PointSea
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pointnet2_ops.pointnet2_utils import gather_operation as gather_points, furthest_point_sample
from models.model_utils import *
from models_PointSea.model_utils import *
from metrics.CD.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from torchvision.models import resnet18, ResNet18_Weights
from einops import rearrange


# ========== Point Cloud Spectral Adapter (PCSA) ==========
class SpectralGraphConv(nn.Module):
    """Spectral Graph Convolution using Graph Fourier Transform"""
    def __init__(self, in_dim, out_dim, k=16):
        super(SpectralGraphConv, self).__init__()
        self.k = k
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Learnable spectral filter
        self.spectral_filter = nn.Parameter(torch.randn(k, in_dim, out_dim))
        self.bn = nn.BatchNorm1d(out_dim)
        
    def forward(self, x, pos):
        """
        Args:
            x: (B, C, N) - point features
            pos: (B, 3, N) - point positions
        Returns:
            out: (B, out_dim, N) - transformed features
        """
        B, C, N = x.shape
        
        # Build graph adjacency using k-NN
        idx = knn(pos, self.k)  # (B, N, k)
        
        # Get neighbor features
        neighbor_features = indexing_neighbor(x, idx)  # (B, C, N, k)
        
        # Compute graph Laplacian approximation
        center_features = x.unsqueeze(-1).expand_as(neighbor_features)
        edge_features = neighbor_features - center_features
        
        # Apply spectral filtering
        # Approximate GFT using local neighborhood
        edge_features = edge_features.permute(0, 2, 3, 1)  # (B, N, k, C)
        
        # Apply learnable spectral filter
        out = torch.einsum('bnkc,kcd->bnd', edge_features, self.spectral_filter)
        out = out.permute(0, 2, 1)  # (B, out_dim, N)
        out = self.bn(out)
        
        return out


class PCSA(nn.Module):
    """Point Cloud Spectral Adapter - adapts spatial features to spectral domain"""
    def __init__(self, dim, k=16, reduction=4):
        super(PCSA, self).__init__()
        self.k = k
        self.dim = dim
        
        # Spectral transformation
        self.spectral_conv = SpectralGraphConv(dim, dim, k)
        
        # Channel attention for spectral-spatial fusion
        self.channel_attn = nn.Sequential(
            nn.Conv1d(dim, dim // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim // reduction, dim, 1),
            nn.Sigmoid()
        )
        
        # Spatial-spectral fusion
        self.fusion = nn.Conv1d(dim * 2, dim, 1)
        
    def forward(self, x, pos):
        """
        Args:
            x: (B, C, N) - spatial features
            pos: (B, 3, N) - point positions
        Returns:
            out: (B, C, N) - spectral-enhanced features
        """
        # Transform to spectral domain
        x_spectral = self.spectral_conv(x, pos)
        
        # Channel attention for adaptive fusion
        attn = self.channel_attn(x_spectral)
        x_spectral = x_spectral * attn
        
        # Fuse spatial and spectral features
        x_fused = torch.cat([x, x_spectral], dim=1)
        out = self.fusion(x_fused)
        
        return out


# ========== Multi-Scale Graph Convolution (MSGConv) ==========
class MSGConv(nn.Module):
    """Multi-Scale Graph Convolution for extracting multi-frequency geometric patterns"""
    def __init__(self, in_dim, out_dim, scales=[8, 16, 32]):
        super(MSGConv, self).__init__()
        self.scales = scales
        self.num_scales = len(scales)
        
        # Multi-scale spectral convolutions
        self.spectral_convs = nn.ModuleList([
            SpectralGraphConv(in_dim, out_dim // self.num_scales, k=k)
            for k in scales
        ])
        
        # Feature aggregation
        self.aggregation = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, 1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, pos):
        """
        Args:
            x: (B, C, N) - input features
            pos: (B, 3, N) - point positions
        Returns:
            out: (B, out_dim, N) - multi-scale features
        """
        multi_scale_feats = []
        for conv in self.spectral_convs:
            feat = conv(x, pos)
            multi_scale_feats.append(feat)
        
        # Concatenate multi-scale features
        out = torch.cat(multi_scale_feats, dim=1)
        out = self.aggregation(out)
        
        return out


# ========== Geo-Spectral Collaborative Perception Module ==========
class GeoSpectralModule(nn.Module):
    """
    Combines Graph Fourier Transform with geometric attention
    for multi-scale feature extraction in both spectral and spatial domains
    """
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GeoSpectralModule, self).__init__()
        
        # Spectral branch with PCSA
        self.pcsa = PCSA(in_dim, k=16)
        self.msg_conv = MSGConv(in_dim, hidden_dim)
        
        # Geometric branch with local attention
        self.geo_conv1 = EdgeConv(in_dim, hidden_dim, k=16)
        self.geo_attn = self_attention(hidden_dim, hidden_dim, nhead=4)
        
        # Cross-modal fusion
        self.cross_attn = cross_attention(hidden_dim, hidden_dim, nhead=4)
        
        # Output projection
        self.out_proj = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, out_dim, 1),
            nn.BatchNorm1d(out_dim),
            nn.GELU()
        )
        
    def forward(self, x, pos):
        """
        Args:
            x: (B, C, N) - input features
            pos: (B, 3, N) - point positions
        Returns:
            out: (B, out_dim, N) - geo-spectral features
        """
        # Spectral pathway
        x_spatial_enhanced = self.pcsa(x, pos)
        x_spectral = self.msg_conv(x_spatial_enhanced, pos)
        
        # Geometric pathway
        x_geo = self.geo_conv1(x)
        x_geo = self.geo_attn(x_geo)
        
        # Cross-modal interaction
        x_spectral_enhanced = self.cross_attn(x_spectral, x_geo)
        
        # Fusion
        out = torch.cat([x_spectral_enhanced, x_geo], dim=1)
        out = self.out_proj(out)
        
        return out


# ========== Structure-Aware Gating Unit ==========
class StructureAwareGating(nn.Module):
    """Gate unit for dynamic region selection based on geometric complexity"""
    def __init__(self, dim):
        super(StructureAwareGating, self).__init__()
        
        self.complexity_estimator = nn.Sequential(
            nn.Conv1d(dim, dim // 2, 1),
            nn.BatchNorm1d(dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim // 2, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, C, N) - input features
        Returns:
            gate: (B, 1, N) - gating scores
        """
        gate = self.complexity_estimator(x)
        return gate


# ========== Dynamic Region Selection Network (DRSN) ==========
class DRSN(nn.Module):
    """
    Dynamic Region Selection Network with dual-path refinement
    - Global semantic path for coarse geometry
    - Local detail path for fine structures
    """
    def __init__(self, channel=128, ratio=1, hidden_dim=768):
        super(DRSN, self).__init__()
        self.channel = channel
        self.hidden = hidden_dim
        self.ratio = ratio
        
        # Input projection
        self.conv_x = nn.Conv1d(3, 64, kernel_size=1)
        self.conv_x1 = nn.Conv1d(64, channel, kernel_size=1)
        self.conv_1 = nn.Conv1d(256, channel, kernel_size=1)
        self.conv_11 = nn.Conv1d(512, 256, kernel_size=1)
        
        # Structure-aware gating
        self.structure_gate = StructureAwareGating(channel * 2)
        
        # Global semantic path
        self.global_attn = self_attention(channel * 2, hidden_dim, dropout=0.0, nhead=8)
        self.global_decoder = self_attention(hidden_dim, channel * ratio, dropout=0.0, nhead=8)
        
        # Local detail path with geo-spectral enhancement
        self.local_geo_spectral = GeoSpectralModule(832, hidden_dim, hidden_dim)
        self.local_attn = cross_attention(hidden_dim, hidden_dim, dropout=0.0, nhead=8)
        self.local_decoder = self_attention(hidden_dim, channel * ratio, dropout=0.0, nhead=8)
        
        # Path selection and fusion
        self.path_selector = nn.Sequential(
            nn.Conv1d(hidden_dim * 2 + channel, hidden_dim, 1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, 1, 1),
            nn.Sigmoid()
        )
        
        # Output layers
        self.relu = nn.GELU()
        self.conv_delta = nn.Conv1d(channel, channel, kernel_size=1)
        self.conv_out1 = nn.Conv1d(channel, 64, kernel_size=1)
        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)
        
        # Distance calculation
        self.cd_distance = chamfer_3DDist()
        self.sigma_d = 0.2
        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        
    def forward(self, local_feat, coarse, f_g, partial):
        """
        Args:
            local_feat: (B, 832, N_local) - local geometric features
            coarse: (B, 3, N_coarse) - coarse point cloud
            f_g: (B, 512, 1) - global feature
            partial: (B, 3, N_partial) - partial input
        Returns:
            fine: (B, 3, N_fine) - refined point cloud
            F_L: (B, C, N_fine) - refined features
        """
        batch_size, _, N = coarse.size()
        
        # Feature preparation
        F = self.conv_x1(self.relu(self.conv_x(coarse)))
        f_g = self.conv_1(self.relu(self.conv_11(f_g)))
        F = torch.cat([F, f_g.repeat(1, 1, F.shape[-1])], dim=1)
        
        # Structure-aware gating for region identification
        structure_score = self.structure_gate(F)
        
        # Positional embedding based on chamfer distance
        half_cd = self.cd_distance(
            coarse.transpose(1, 2).contiguous(),
            partial.transpose(1, 2).contiguous()
        )[0] / self.sigma_d
        embd = self.embedding(half_cd).reshape(batch_size, self.hidden, -1).permute(2, 0, 1)
        
        # Global semantic path
        F_global = self.global_attn(F, embd)
        F_global_decoded = self.global_decoder(F_global)
        
        # Local detail path with geo-spectral processing
        # First extract geo-spectral features from local features
        local_feat_pos = coarse[:, :3, :local_feat.size(2)] if local_feat.size(2) < coarse.size(2) else coarse
        local_feat_enhanced = self.local_geo_spectral(local_feat, local_feat_pos)
        
        # Cross-attention between coarse features and enhanced local features
        F_local = self.local_attn(F_global, local_feat_enhanced)
        F_local_decoded = self.local_decoder(F_local)
        
        # Dynamic path selection based on structure complexity
        path_score = self.path_selector(
            torch.cat([F_global_decoded, F_local_decoded, f_g.repeat(1, 1, F_global_decoded.size(2))], 1)
        )
        
        # Adaptive fusion of two paths
        F_L = path_score * F_global_decoded + (1 - path_score) * F_local_decoded
        
        # Generate output
        F_L = self.conv_delta(F_L.reshape(batch_size, -1, N * self.ratio))
        O_L = self.conv_out(self.relu(self.conv_out1(F_L)))
        fine = coarse.repeat(1, 1, self.ratio) + O_L
        
        return fine, F_L


# ========== Multi-View Feature Extractor ==========
class ResEncoder(nn.Module):
    """ResNet-based encoder for multi-view depth images"""
    def __init__(self):
        super(ResEncoder, self).__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
    def forward(self, input_view):
        feat0 = self.relu(self.bn1(self.conv1(input_view)))
        x = self.maxpool(feat0)
        
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)
        
        return feat4


# ========== Point Cloud Feature Extractor ==========
class PointCloudEncoder(nn.Module):
    """Encoder for extracting global features from partial point cloud"""
    def __init__(self, out_dim=256):
        super(PointCloudEncoder, self).__init__()
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.sa_module_2 = PointNet_SA_Module_KNN(128, 16, 128, [128, 256], group_all=False, if_bn=False, if_idx=True)
        self.sa_module_3 = PointNet_SA_Module_KNN(None, None, 256, [512, out_dim], group_all=True, if_bn=False)
        
    def forward(self, point_cloud):
        """
        Args:
            point_cloud: (B, 3, N)
        Returns:
            l3_points: (B, out_dim, 1)
        """
        l0_xyz = point_cloud
        l0_points = point_cloud
        
        l1_xyz, l1_points, idx1 = self.sa_module_1(l0_xyz, l0_points)
        l2_xyz, l2_points, idx2 = self.sa_module_2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa_module_3(l2_xyz, l2_points)
        
        return l3_points


# ========== Multi-View Fusion Encoder ==========
class MultiViewFusionEncoder(nn.Module):
    """Encoder combining point cloud and multi-view features"""
    def __init__(self, cfg):
        super(MultiViewFusionEncoder, self).__init__()
        self.channel = 64
        self.view_distance = cfg.NETWORK.view_distance
        
        # Feature extractors
        self.point_encoder = PointCloudEncoder(out_dim=512)
        self.view_encoder = ResEncoder()
        
        # Multi-view attention
        self.viewattn1 = self_attention(256 + 512, 512, nhead=4)
        self.viewattn2 = self_attention(256 + 512, 256, nhead=4)
        
        # Position encoding for views
        self.posmlp = MLP_CONV(3, [64, 256])
        
        # Coarse generation
        self.relu = nn.GELU()
        self.sa = self_attention(self.channel * 8, self.channel * 8, dropout=0.0)
        self.ps = nn.ConvTranspose1d(512, self.channel, 128, bias=True)
        self.ps_refuse = nn.Conv1d(512 + self.channel, self.channel * 8, kernel_size=1)
        self.conv_out1 = nn.Conv1d(512 + self.channel * 4, 64, kernel_size=1)
        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)
        
    def forward(self, points, depth):
        """
        Args:
            points: (B, 3, N) - partial point cloud
            depth: (BV, 3, H, W) - multi-view depth images
        Returns:
            f_g: (B, 512, 1) - global feature
            coarse: (B, 3, N_coarse) - coarse completion
        """
        batch_size, _, N = points.size()
        
        # Extract point cloud features
        f_p = self.point_encoder(points)
        
        # Extract multi-view features
        f_v = self.view_encoder(depth)
        f_v = rearrange(f_v, 'bv c h w -> bv c (h w)')
        
        # Multi-view fusion
        view_point = torch.tensor(
            [0, 0, -self.view_distance, -self.view_distance, 0, 0, 0, self.view_distance, 0],
            dtype=torch.float32
        ).view(-1, 3, 3).permute(0, 2, 1).repeat(batch_size, 1, 1).to(depth.device)
        view_feature_1 = self.posmlp(view_point)
        
        # Aggregate multi-view features
        f_v_ = self.viewattn1(torch.cat([f_v, f_p.repeat(3, 1, f_v.size(2))], 1))
        f_v_ = rearrange(f_v_, '(b v) c n -> b c v n', b=batch_size)
        f_v_ = torch.max(f_v_, dim=3)[0]
        f_v_ = self.viewattn2(torch.cat([f_v_, f_p.repeat(1, 1, f_v_.size(2))], dim=1), view_feature_1.permute(2, 0, 1))
        f_v_ = F.adaptive_max_pool1d(f_v_, 1)
        
        # Global feature
        f_g = torch.cat([f_p, f_v_], 1)
        
        # Generate coarse point cloud
        x = self.relu(self.ps(f_g))
        x = self.relu(self.ps_refuse(torch.cat([x, f_g.repeat(1, 1, x.size(2))], 1)))
        x2_d = self.sa(x).reshape(batch_size, self.channel * 4, N // 8)
        coarse = self.conv_out(self.relu(self.conv_out1(torch.cat([x2_d, f_g.repeat(1, 1, x2_d.size(2))], 1))))
        
        return f_g, coarse


# ========== Local Geometric Feature Extractor ==========
class LocalGeometricEncoder(nn.Module):
    """Extract local geometric features using EdgeConv"""
    def __init__(self, cfg):
        super(LocalGeometricEncoder, self).__init__()
        self.gcn_1 = EdgeConv(3, 64, 16)
        self.gcn_2 = EdgeConv(64, 256, 8)
        self.gcn_3 = EdgeConv(256, 512, 4)
        self.local_number = cfg.NETWORK.local_points
        
    def forward(self, input):
        """
        Args:
            input: (B, 3, N) - input point cloud
        Returns:
            features: (B, 832, N_local) - local features
        """
        x1 = self.gcn_1(input)
        idx = furthest_point_sample(input.transpose(1, 2).contiguous(), self.local_number)
        x1 = gather_points(x1, idx)
        
        x2 = self.gcn_2(x1)
        x3 = self.gcn_3(x2)
        
        return torch.cat([x1, x2, x3], 1)


# ========== GeoSpecNet Main Model ==========
class GeoSpecNet(nn.Module):
    """
    GeoSpecNet: Point Cloud Completion with Spectral Domain Enhancement
    
    Architecture:
    1. Multi-view fusion encoder (from PointSea)
    2. Local geometric encoder
    3. Geo-spectral collaborative perception
    4. Dynamic region selection network for iterative refinement
    """
    def __init__(self, cfg):
        super(GeoSpecNet, self).__init__()
        
        # Encoders
        self.encoder = MultiViewFusionEncoder(cfg)
        self.local_encoder = LocalGeometricEncoder(cfg)
        
        # Refinement parameters
        self.merge_points = cfg.NETWORK.merge_points
        
        # Two-stage DRSN refinement
        self.refine1 = DRSN(channel=128, ratio=cfg.NETWORK.step1, hidden_dim=768)
        self.refine2 = DRSN(channel=128, ratio=cfg.NETWORK.step2, hidden_dim=512)
        
    def forward(self, partial, depth):
        """
        Args:
            partial: (B, N, 3) - partial input point cloud
            depth: (BV, 3, H, W) - multi-view depth images
        Returns:
            coarse: (B, N_coarse, 3) - coarse completion
            fine1: (B, N_fine1, 3) - first refinement
            fine2: (B, N_fine2, 3) - second refinement (final output)
        """
        # Transpose to (B, 3, N)
        partial = partial.transpose(1, 2).contiguous()
        
        # Multi-view fusion encoding
        feat_g, coarse = self.encoder(partial, depth)
        
        # Local geometric feature extraction
        local_feat = self.local_encoder(partial)
        
        # Merge partial and coarse for refinement input
        coarse_merge = torch.cat([partial, coarse], dim=2)
        coarse_merge = gather_points(
            coarse_merge,
            furthest_point_sample(coarse_merge.transpose(1, 2).contiguous(), self.merge_points)
        )
        
        # Two-stage refinement with DRSN
        fine1, F_L_1 = self.refine1(local_feat, coarse_merge, feat_g, partial)
        fine2, F_L_2 = self.refine2(local_feat, fine1, feat_g, partial)
        
        # Transpose back to (B, N, 3)
        return (
            coarse.transpose(1, 2).contiguous(),
            fine1.transpose(1, 2).contiguous(),
            fine2.transpose(1, 2).contiguous()
        )


# Model alias for compatibility
Model = GeoSpecNet


if __name__ == '__main__':
    # Test the model
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Mock config
    class Config:
        class NETWORK:
            view_distance = 1.5
            local_points = 512
            merge_points = 1024
            step1 = 4
            step2 = 2
    
    cfg = Config()
    
    model = GeoSpecNet(cfg).cuda()
    model.eval()
    
    # Create dummy inputs
    from models_PointSea.mv_utils_zs import PCViews_Real
    render = PCViews_Real(TRANS=-cfg.NETWORK.view_distance)
    
    input_pc = torch.rand(2, 2048, 3).cuda()
    depth = render.get_img(input_pc)
    
    with torch.no_grad():
        outputs = model(input_pc, depth)
    
    print("Model output shapes:")
    for i, output in enumerate(outputs):
        print(f"  Stage {i}: {output.shape}")
