"""
Loss functions for GeoSpecNet
Includes Chamfer Distance, Partial Matching Loss, and GAN losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics.CD.chamfer3D.dist_chamfer_3D import chamfer_3DDist


class GeoSpecNetLoss(nn.Module):
    """
    Comprehensive loss function for GeoSpecNet
    
    Components:
    1. Chamfer Distance (multi-stage)
    2. Partial Matching Loss
    3. Smoothness Loss
    4. GAN Loss (optional)
    """
    
    def __init__(self, cfg):
        super(GeoSpecNetLoss, self).__init__()
        
        self.cfg = cfg
        self.cd_loss = chamfer_3DDist()
        
        # Loss weights
        self.w_cd_coarse = cfg.TRAIN.LOSS_WEIGHTS.CD_COARSE
        self.w_cd_fine1 = cfg.TRAIN.LOSS_WEIGHTS.CD_FINE1
        self.w_cd_fine2 = cfg.TRAIN.LOSS_WEIGHTS.CD_FINE2
        self.w_partial = cfg.TRAIN.LOSS_WEIGHTS.PARTIAL_MATCH
        
    def forward(self, partial, coarse, fine1, fine2, gt):
        """
        Compute total loss
        
        Args:
            partial: (B, N, 3) partial point cloud
            coarse: (B, M, 3) coarse completion
            fine1: (B, M1, 3) first refinement
            fine2: (B, M2, 3) final completion
            gt: (B, G, 3) ground truth
        
        Returns:
            total_loss: scalar tensor
            loss_dict: dictionary of individual losses
        """
        
        losses = {}
        
        # 1. Chamfer Distance losses
        cd_coarse = self.chamfer_distance_loss(coarse, gt)
        cd_fine1 = self.chamfer_distance_loss(fine1, gt)
        cd_fine2 = self.chamfer_distance_loss(fine2, gt)
        
        losses['cd_coarse'] = cd_coarse
        losses['cd_fine1'] = cd_fine1
        losses['cd_fine2'] = cd_fine2
        
        # 2. Partial Matching Loss
        partial_loss = self.partial_matching_loss(fine2, partial)
        losses['partial_match'] = partial_loss
        
        # 3. Smoothness Loss
        smooth_loss = self.smoothness_loss(fine2)
        losses['smoothness'] = smooth_loss
        
        # Compute total loss
        total_loss = (
            self.w_cd_coarse * cd_coarse +
            self.w_cd_fine1 * cd_fine1 +
            self.w_cd_fine2 * cd_fine2 +
            self.w_partial * partial_loss +
            0.1 * smooth_loss
        )
        
        losses['total'] = total_loss
        
        return total_loss, losses
    
    def chamfer_distance_loss(self, pred, gt):
        """
        Compute Chamfer Distance loss
        
        Args:
            pred: (B, N, 3) predicted point cloud
            gt: (B, M, 3) ground truth point cloud
        
        Returns:
            loss: scalar tensor
        """
        dist1, dist2 = self.cd_loss(pred, gt)
        loss = (dist1.mean(1) + dist2.mean(1)).mean()
        return loss
    
    def partial_matching_loss(self, completed, partial):
        """
        Partial Matching Loss: ensures visible regions remain unchanged
        
        Args:
            completed: (B, N, 3) completed point cloud
            partial: (B, M, 3) partial point cloud
        
        Returns:
            loss: scalar tensor
        """
        # Find nearest neighbors in completed cloud for each partial point
        B, M, _ = partial.shape
        N = completed.shape[1]
        
        # Compute pairwise distances
        # (B, M, N)
        dist = torch.cdist(partial, completed)
        
        # Find minimum distance for each partial point
        min_dist, _ = torch.min(dist, dim=2)  # (B, M)
        
        # Loss is the average minimum distance
        loss = min_dist.mean()
        
        return loss
    
    def smoothness_loss(self, point_cloud, k=10):
        """
        Smoothness Loss: encourages local smoothness
        
        Args:
            point_cloud: (B, N, 3) point cloud
            k: number of neighbors
        
        Returns:
            loss: scalar tensor
        """
        B, N, _ = point_cloud.shape
        
        # Compute pairwise distances
        dist = torch.cdist(point_cloud, point_cloud)  # (B, N, N)
        
        # Find k-nearest neighbors
        _, idx = torch.topk(dist, k + 1, largest=False, dim=-1)  # (B, N, k+1)
        idx = idx[:, :, 1:]  # Exclude self (B, N, k)
        
        # Gather neighbor points
        batch_idx = torch.arange(B, device=point_cloud.device).view(B, 1, 1).expand(B, N, k)
        neighbors = point_cloud[batch_idx, idx]  # (B, N, k, 3)
        
        # Compute local variance
        center = point_cloud.unsqueeze(2)  # (B, N, 1, 3)
        diff = neighbors - center  # (B, N, k, 3)
        variance = (diff ** 2).sum(dim=-1).mean()  # Scalar
        
        return variance


class PartialMatchingLoss(nn.Module):
    """
    Partial Matching Loss
    Ensures that the visible region in partial cloud is preserved in completion
    
    Reference: Inspired by PointOutNet (ECCV 2020)
    """
    
    def __init__(self, threshold=0.01):
        super(PartialMatchingLoss, self).__init__()
        self.threshold = threshold
    
    def forward(self, completed, partial):
        """
        Args:
            completed: (B, N, 3) completed point cloud
            partial: (B, M, 3) partial point cloud
        
        Returns:
            loss: scalar tensor
        """
        # For each point in partial, find closest point in completed
        dist = torch.cdist(partial, completed)  # (B, M, N)
        min_dist, _ = torch.min(dist, dim=2)  # (B, M)
        
        # Loss is average distance
        loss = min_dist.mean()
        
        return loss


class StructuralConsistencyLoss(nn.Module):
    """
    Structural Consistency Loss using feature matching
    Ensures completed cloud has similar structural patterns as GT
    """
    
    def __init__(self):
        super(StructuralConsistencyLoss, self).__init__()
        
        # Feature extractor (simple MLP)
        self.feature_extractor = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )
    
    def forward(self, completed, gt):
        """
        Args:
            completed: (B, N, 3) completed point cloud
            gt: (B, M, 3) ground truth point cloud
        
        Returns:
            loss: scalar tensor
        """
        # Extract features
        feat_completed = self.feature_extractor(completed)  # (B, N, 256)
        feat_gt = self.feature_extractor(gt)  # (B, M, 256)
        
        # Global features
        global_completed = feat_completed.mean(dim=1)  # (B, 256)
        global_gt = feat_gt.mean(dim=1)  # (B, 256)
        
        # L2 distance between global features
        loss = F.mse_loss(global_completed, global_gt)
        
        return loss


class DensityLoss(nn.Module):
    """
    Density Loss: penalizes non-uniform point distribution
    Encourages points to be evenly distributed on the surface
    """
    
    def __init__(self, k=10):
        super(DensityLoss, self).__init__()
        self.k = k
    
    def forward(self, point_cloud):
        """
        Args:
            point_cloud: (B, N, 3) point cloud
        
        Returns:
            loss: scalar tensor
        """
        B, N, _ = point_cloud.shape
        
        # Compute k-nearest neighbor distances
        dist = torch.cdist(point_cloud, point_cloud)  # (B, N, N)
        
        # Get k-nearest distances (excluding self)
        knn_dist, _ = torch.topk(dist, self.k + 1, largest=False, dim=-1)  # (B, N, k+1)
        knn_dist = knn_dist[:, :, 1:]  # (B, N, k)
        
        # Average distance to k neighbors for each point
        avg_dist = knn_dist.mean(dim=-1)  # (B, N)
        
        # Variance of these distances
        density_variance = avg_dist.var(dim=-1).mean()  # Scalar
        
        return density_variance


class RepulsionLoss(nn.Module):
    """
    Repulsion Loss: prevents points from clustering too tightly
    Encourages minimum distance between points
    
    Reference: Inspired by 3D point cloud generation papers
    """
    
    def __init__(self, radius=0.07, k=10):
        super(RepulsionLoss, self).__init__()
        self.radius = radius
        self.k = k
    
    def forward(self, point_cloud):
        """
        Args:
            point_cloud: (B, N, 3) point cloud
        
        Returns:
            loss: scalar tensor
        """
        B, N, _ = point_cloud.shape
        
        # Compute pairwise distances
        dist = torch.cdist(point_cloud, point_cloud)  # (B, N, N)
        
        # Get k-nearest distances (excluding self)
        knn_dist, _ = torch.topk(dist, self.k + 1, largest=False, dim=-1)  # (B, N, k+1)
        knn_dist = knn_dist[:, :, 1:]  # (B, N, k)
        
        # Penalize distances less than radius
        repulsion = torch.clamp(self.radius - knn_dist, min=0.0)
        loss = repulsion.mean()
        
        return loss


class GANLoss(nn.Module):
    """
    GAN Loss for structural consistency training
    
    Reference: GAN-based Shape Completion (CVPR 2019)
    """
    
    def __init__(self, discriminator):
        super(GANLoss, self).__init__()
        self.discriminator = discriminator
        self.bce_loss = nn.BCELoss()
    
    def discriminator_loss(self, real, fake):
        """
        Compute discriminator loss
        
        Args:
            real: (B, N, 3) real point clouds
            fake: (B, N, 3) generated point clouds
        
        Returns:
            d_loss: discriminator loss
        """
        # Predictions
        real_pred = self.discriminator(real)
        fake_pred = self.discriminator(fake.detach())
        
        # Losses
        real_loss = self.bce_loss(real_pred, torch.ones_like(real_pred))
        fake_loss = self.bce_loss(fake_pred, torch.zeros_like(fake_pred))
        
        d_loss = (real_loss + fake_loss) / 2
        
        return d_loss
    
    def generator_loss(self, fake):
        """
        Compute generator loss
        
        Args:
            fake: (B, N, 3) generated point clouds
        
        Returns:
            g_loss: generator loss
        """
        fake_pred = self.discriminator(fake)
        g_loss = self.bce_loss(fake_pred, torch.ones_like(fake_pred))
        
        return g_loss


def compute_gradient_penalty(discriminator, real_data, fake_data):
    """
    Compute gradient penalty for WGAN-GP
    
    Args:
        discriminator: discriminator network
        real_data: (B, N, 3) real point clouds
        fake_data: (B, N, 3) fake point clouds
    
    Returns:
        gradient_penalty: scalar tensor
    """
    B = real_data.shape[0]
    
    # Random weight for interpolation
    alpha = torch.rand(B, 1, 1, device=real_data.device)
    
    # Interpolate between real and fake
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates.requires_grad_(True)
    
    # Discriminator output
    d_interpolates = discriminator(interpolates)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Flatten gradients
    gradients = gradients.view(B, -1)
    
    # Compute gradient penalty
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty
