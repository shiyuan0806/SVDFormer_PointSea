from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2_ops.pointnet2_utils import (
    furthest_point_sample,
    gather_operation as gather_points,
)

from models.model_utils import (
    EdgeConv,
    self_attention,
    cross_attention,
    SDG_Decoder,
    SinusoidalPositionalEmbedding,
    square_distance,
    query_knn_point,
    index_points,
)


class GraphBuilder(nn.Module):
    """Utility to build k-NN graph and Laplacian for a set of points.

    Computes a symmetric normalized Laplacian L = I - D^{-1/2} A D^{-1/2}
    where A_ij = exp(-||xi-xj||^2 / (2*sigma^2)) if j in kNN(i), else 0.
    """

    def __init__(self, k: int = 16, sigma: float = 1.0):
        super().__init__()
        self.k = k
        self.sigma2 = sigma * sigma

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """Build Laplacian for each batch sample.

        Args:
            xyz: (B, 3, N)
        Returns:
            L: (B, N, N)
        """
        B, _, N = xyz.shape
        xyz_nchw = xyz.transpose(1, 2).contiguous()  # (B, N, 3)
        idx = query_knn_point(self.k, xyz_nchw, xyz_nchw)  # (B, N, k)

        # Gather neighbor coords
        neighbors = index_points(xyz_nchw, idx)  # (B, N, k, 3)
        central = xyz_nchw.unsqueeze(2).expand(-1, -1, self.k, -1)  # (B, N, k, 3)
        diff = central - neighbors  # (B, N, k, 3)
        dist2 = (diff * diff).sum(dim=3)  # (B, N, k)
        weights = torch.exp(-dist2 / (2.0 * self.sigma2))  # (B, N, k)

        # Build sparse-like adjacency then dense mat by scatter
        device = xyz.device
        A = torch.zeros(B, N, N, device=device)
        arange_b = torch.arange(B, device=device)[:, None, None]
        arange_i = torch.arange(N, device=device)[None, :, None]
        A[arange_b.expand_as(idx), arange_i.expand_as(idx), idx] = weights
        # Symmetrize
        A = 0.5 * (A + A.transpose(1, 2))

        # Degree and Laplacian
        D = torch.clamp_min(A.sum(dim=-1), 1e-6)  # (B, N)
        D_inv_sqrt = D.pow(-0.5)
        D_inv_sqrt = torch.diag_embed(D_inv_sqrt)  # (B, N, N)
        I = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)
        L = I - D_inv_sqrt @ A @ D_inv_sqrt
        return L


class GFTLayer(nn.Module):
    """Graph Fourier Transform layer with learnable spectral filtering.

    For each sample, computes eigen-decomposition of Laplacian and applies
    learnable spectral filters to features, then transforms back to spatial domain.
    """

    def __init__(self, in_channels: int, out_channels: int, k_eig: int = 48):
        super().__init__()
        self.k_eig = k_eig
        # Spectral filter implemented as a small MLP over eigen-coefficient dimension
        self.spectral_mlp = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=True),
        )

    @torch.enable_grad()  # keep gradients for backprop through eig if needed
    def forward(self, L: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        """Apply GFT filter.

        Args:
            L: (B, N, N) symmetric normalized Laplacian
            feats: (B, C_in, N)
        Returns:
            out: (B, C_out, N)
        """
        B, _, N = feats.shape
        device = feats.device
        out = []
        for b in range(B):
            # eigh returns ascending eigenvalues
            evals, evecs = torch.linalg.eigh(L[b])  # (N,), (N, N)
            # Select k smallest (low-to-mid frequency) eigenvectors (excluding trivial if exists)
            k = min(self.k_eig, N)
            U = evecs[:, :k]  # (N, k)
            X = feats[b]  # (C_in, N)
            X_hat = torch.matmul(X, U)  # (C_in, k)
            X_hat = X_hat.unsqueeze(0)  # (1, C_in, k)
            X_hat = self.spectral_mlp(X_hat.squeeze(0)).unsqueeze(0)  # (1, C_out, k)
            Y = torch.matmul(X_hat.squeeze(0), U.transpose(0, 1))  # (C_out, N)
            out.append(Y.unsqueeze(0))
        return torch.cat(out, dim=0)  # (B, C_out, N)


class MSGConv(nn.Module):
    """Multi-Scale Graph Convolution combining spatial EdgeConv and spectral GFT.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 k_neighbor_small: int = 16, k_neighbor_large: int = 32, k_eig: int = 48):
        super().__init__()
        self.edge_small = EdgeConv(in_channels, hidden_channels, k_neighbor_small)
        self.edge_large = EdgeConv(hidden_channels, hidden_channels, k_neighbor_large)
        self.graph_builder = GraphBuilder(k=k_neighbor_small)
        self.gft = GFTLayer(hidden_channels, hidden_channels, k_eig=k_eig)
        self.proj = nn.Sequential(
            nn.Conv1d(hidden_channels * 3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, xyz: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        """Args:
            xyz: (B, 3, N)
            feats: (B, C_in, N)
        Returns:
            (B, out_channels, N)
        """
        f1 = self.edge_small(feats)  # (B, H, N)
        f2 = self.edge_large(f1)     # (B, H, N)
        L = self.graph_builder(xyz)  # (B, N, N)
        fspec = self.gft(L, f2)      # (B, H, N)
        f = torch.cat([f1, f2, fspec], dim=1)
        return self.proj(f)


class StructureAwareGatingUnit(nn.Module):
    """Compute gate per point from global and local features.
    G = sigma(MLP([F_global; F_local])) in [0,1].
    """

    def __init__(self, in_global: int, in_local: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_global + in_local, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(128, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, Fg: torch.Tensor, Fl: torch.Tensor) -> torch.Tensor:
        x = torch.cat([Fg, Fl], dim=1)
        return self.mlp(x)  # (B, 1, N)


class DRSN(nn.Module):
    """Dynamic Region Selection Network with dual-path refinement.

    - Global semantic path: self-attention + decoder
    - Local detail path: cross-attention with local features + decoder
    - Structure-aware gating blends both paths per-point.
    """

    def __init__(self, in_channels: int, hidden_dim: int, up_ratio: int):
        super().__init__()
        self.up_ratio = up_ratio
        self.embed_pos = SinusoidalPositionalEmbedding(hidden_dim)
        self.self_attn = self_attention(d_model=in_channels, d_model_out=hidden_dim, nhead=8, dropout=0.0)
        self.global_decoder = SDG_Decoder(hidden_dim=hidden_dim, channel=64, ratio=up_ratio)

        self.cross_attn = cross_attention(d_model=in_channels, d_model_out=hidden_dim, nhead=8, dropout=0.0)
        self.local_decoder = SDG_Decoder(hidden_dim=hidden_dim, channel=64, ratio=up_ratio)

        self.to_offset = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(64, 3, kernel_size=1),
        )

        self.gate = StructureAwareGatingUnit(in_global=hidden_dim, in_local=hidden_dim)

    def forward(self, coarse_xyz: torch.Tensor, global_feats: torch.Tensor, local_feats: torch.Tensor,
                partial_xyz: torch.Tensor) -> torch.Tensor:
        """Refine and upsample points.

        Args:
            coarse_xyz: (B, 3, Nc)
            global_feats: (B, Cg, Nc) repeated global embedding aligned to points
            local_feats: (B, Cl, Nc) local features interpolated/aligned to coarse points
            partial_xyz: (B, 3, Np)
        Returns:
            refined: (B, 3, Nc * up_ratio)
        """
        B, _, Nc = coarse_xyz.shape
        # Incompleteness score via nearest distance to partial
        d_chamfer = square_distance(coarse_xyz.transpose(1, 2).contiguous(),
                                    partial_xyz.transpose(1, 2).contiguous())  # (B, Nc, Np)
        nearest = torch.sqrt(d_chamfer.min(dim=-1).values)  # (B, Nc)
        embed = self.embed_pos(nearest).permute(1, 0, 2)  # (Nc, B, hidden)

        # Global path
        Fg = self.self_attn(global_feats, pos=embed)  # (B, hidden, Nc)
        G_out = self.global_decoder(Fg)  # (B, channel=64, Nc*ratio)
        G_delta = self.to_offset(G_out)   # (B, 3, Nc*ratio)

        # Local path
        Fl = self.cross_attn(global_feats, local_feats)  # (B, hidden, Nc)
        L_out = self.local_decoder(Fl)  # (B, 64, Nc*ratio)
        L_delta = self.to_offset(L_out)  # (B, 3, Nc*ratio)

        # Gate
        gate = self.gate(Fg, Fl)  # (B,1,Nc)
        gate_rep = gate.repeat(1, 1, self.up_ratio)  # (B,1,Nc*ratio)

        # Upsample base by repeating coarse
        base = coarse_xyz.repeat(1, 1, self.up_ratio)
        delta = gate_rep * G_delta + (1.0 - gate_rep) * L_delta
        refined = base + delta
        return refined


class GeoSpectralEncoder(nn.Module):
    """Encoder combining spatial EdgeConv and spectral GFT on downsampled points.
    Produces a global shape embedding and per-point features for decoding.
    """

    def __init__(self, hidden_dim: int = 256, spectral_points: int = 512, k_eig: int = 48):
        super().__init__()
        self.spectral_points = spectral_points
        self.edge1 = EdgeConv(3, 64, k=16)
        self.edge2 = EdgeConv(64, 128, k=16)
        self.msg = MSGConv(in_channels=128, hidden_channels=128, out_channels=hidden_dim, k_eig=k_eig)
        self.proj_global = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)

    def forward(self, xyz_full: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Args:
            xyz_full: (B, 3, N)
        Returns:
            global_feat: (B, hidden, 1)
            point_feats: (B, hidden, Ns) features on downsampled points
            xyz_ds: (B, 3, Ns) downsampled coordinates
        """
        B, _, N = xyz_full.shape
        # Downsample for spectral efficiency
        if self.spectral_points is not None and self.spectral_points < N:
            idx = furthest_point_sample(xyz_full.transpose(1, 2).contiguous(), self.spectral_points)
            xyz_ds = gather_points(xyz_full, idx)  # (B, 3, Ns)
        else:
            xyz_ds = xyz_full

        f = self.edge1(xyz_ds)
        f = self.edge2(f)
        f = self.msg(xyz_ds, f)
        g = F.adaptive_max_pool1d(f, 1)
        g = self.proj_global(g)
        return g, f, xyz_ds


class CoarseGenerator(nn.Module):
    """Generate a coarse point cloud from a global embedding and a learnable grid.
    """

    def __init__(self, hidden_dim: int = 256, num_points: int = 512, grid_dim: int = 16):
        super().__init__()
        self.num_points = num_points
        self.grid = nn.Parameter(torch.randn(num_points, grid_dim))
        self.mlp = nn.Sequential(
            nn.Conv1d(hidden_dim + grid_dim, 256, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(256, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(128, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(64, 3, kernel_size=1),
        )

    def forward(self, global_feat: torch.Tensor) -> torch.Tensor:
        """Args:
            global_feat: (B, hidden, 1)
        Returns:
            coarse: (B, 3, P)
        """
        B = global_feat.shape[0]
        grid = self.grid.unsqueeze(0).expand(B, -1, -1)  # (B, P, G)
        grid = grid.permute(0, 2, 1)  # (B, G, P)
        g = global_feat.expand(-1, -1, self.num_points)  # (B, H, P)
        x = torch.cat([g, grid], dim=1)
        coarse = self.mlp(x)
        return coarse


class LocalEncoderForDRSN(nn.Module):
    """Local feature encoder to provide high-resolution local cues to DRSN.
    Matches interface used inside DRSN for cross-attention.
    """

    def __init__(self, local_points: int = 1024):
        super().__init__()
        self.gcn1 = EdgeConv(3, 64, k=16)
        self.gcn2 = EdgeConv(64, 256, k=8)
        self.local_points = local_points

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        # xyz: (B, 3, N)
        x1 = self.gcn1(xyz)
        idx = furthest_point_sample(xyz.transpose(1, 2).contiguous(), self.local_points)
        x1_ds = gather_points(x1, idx)
        x2 = self.gcn2(x1_ds)
        return x2  # (B, 256, local_points)


class GeoSpecNet(nn.Module):
    """GeoSpecNet: Geo-Spectral Collaborative Perception + DRSN for completion.

    Forward signature matches existing training: returns (Pc, P1, P2).
    """

    def __init__(self, cfg):
        super().__init__()
        net_cfg = cfg.NETWORK
        self.step1 = getattr(net_cfg, 'step1', 2)
        self.step2 = getattr(net_cfg, 'step2', 4)
        self.merge_points = getattr(net_cfg, 'merge_points', 1024)
        self.local_points = getattr(net_cfg, 'local_points', 1024)
        self.view_distance = getattr(net_cfg, 'view_distance', 1.5)
        self.spectral_points = getattr(net_cfg, 'spectral_points', 512)
        self.spectral_k = getattr(net_cfg, 'spectral_k', 48)
        self.coarse_points = getattr(net_cfg, 'coarse_points', 512)

        hidden = 256
        self.encoder = GeoSpectralEncoder(hidden_dim=hidden, spectral_points=self.spectral_points, k_eig=self.spectral_k)
        self.local_encoder = LocalEncoderForDRSN(local_points=self.local_points)
        self.coarse_gen = CoarseGenerator(hidden_dim=hidden, num_points=self.coarse_points)

        # Projectors to align channel dims for attentions
        self.proj_global_to_attn = nn.Conv1d(hidden, hidden, kernel_size=1)
        self.proj_local_to_attn = nn.Conv1d(256, hidden, kernel_size=1)
        self.proj_xyz_to_feat = nn.Conv1d(3, 64, kernel_size=1)
        self.to_global_feats_per_point = nn.Conv1d(hidden, hidden, kernel_size=1)

        self.refine1 = DRSN(in_channels=hidden, hidden_dim=512, up_ratio=self.step1)
        self.refine2 = DRSN(in_channels=hidden, hidden_dim=384, up_ratio=self.step2)

    def forward(self, partial_xyz_bnc: torch.Tensor, depth_imgs: Optional[torch.Tensor] = None):
        # partial_xyz_bnc: (B, N, 3)
        # We ignore depth_imgs in this implementation; it's accepted for API compatibility.
        xyz = partial_xyz_bnc.transpose(1, 2).contiguous()  # (B, 3, N)

        # Encode global + spectral features on downsampled points
        g, f_points, xyz_ds = self.encoder(xyz)  # g: (B, H, 1), f_points: (B, H, Ns)
        coarse = self.coarse_gen(g)  # (B, 3, Pc)

        # Build local features from original partial
        local_feats_full = self.local_encoder(xyz)  # (B, 256, local_points)

        # Merge partial and coarse for denser seed before refinement (FPS)
        coarse_merge = torch.cat([xyz, coarse], dim=2)  # (B,3,N+Pc)
        idx = furthest_point_sample(coarse_merge.transpose(1, 2).contiguous(), self.merge_points)
        coarse_merge = gather_points(coarse_merge, idx)  # (B,3,merge_points)

        # Prepare attention features aligned to coarse_merge points by simple nearest interpolation from encoder points
        # Compute nearest indices from coarse_merge to xyz_ds
        dist = square_distance(coarse_merge.transpose(1, 2).contiguous(), xyz_ds.transpose(1, 2).contiguous())  # (B, M, Ns)
        idx_nn = dist.argmin(dim=-1)  # (B, M)
        B, M = idx_nn.shape
        arange_b = torch.arange(B, device=xyz.device)[:, None]
        f_points_t = f_points.transpose(1, 2).contiguous()  # (B, Ns, H)
        f_sel = f_points_t[arange_b, idx_nn]  # (B, M, H)
        f_sel = f_sel.transpose(1, 2).contiguous()  # (B, H, M)

        # Project to attention dims
        global_feats = self.proj_global_to_attn(f_sel)  # (B, H, M)
        local_feats = self.proj_local_to_attn(local_feats_full)  # (B, H, local_points)

        # First refinement
        fine1 = self.refine1(coarse_merge, global_feats, local_feats, xyz)
        # Second refinement
        fine2 = self.refine2(fine1, global_feats, local_feats, xyz)

        # Return in (B, P, 3)
        Pc = coarse.transpose(1, 2).contiguous()
        P1 = fine1.transpose(1, 2).contiguous()
        P2 = fine2.transpose(1, 2).contiguous()
        return (Pc, P1, P2)
