"""
Discriminator Networks for Self-supervised Structural Consistency Training
用于自监督结构一致性训练的判别器网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_utils import PointNet_SA_Module_KNN, EdgeConv, self_attention


class PointCloudDiscriminator(nn.Module):
    """
    点云判别器 - 判断点云是真实的完整点云还是生成的补全结果
    
    设计思想：
    1. 使用多尺度特征提取捕捉不同层次的结构信息
    2. 结合局部几何特征和全局形状特征
    3. 输出真实性分数 [0, 1]
    """
    
    def __init__(self, input_dim=3, hidden_dim=256):
        super(PointCloudDiscriminator, self).__init__()
        
        # 多尺度特征提取 (类似 PointNet++)
        self.sa1 = PointNet_SA_Module_KNN(
            npoint=512, nsample=16, in_channel=input_dim,
            mlp=[64, 128], group_all=False, if_bn=True, if_idx=False
        )
        
        self.sa2 = PointNet_SA_Module_KNN(
            npoint=128, nsample=16, in_channel=128,
            mlp=[128, 256], group_all=False, if_bn=True, if_idx=False
        )
        
        self.sa3 = PointNet_SA_Module_KNN(
            npoint=None, nsample=None, in_channel=256,
            mlp=[256, hidden_dim], group_all=True, if_bn=True, if_idx=False
        )
        
        # 全局判别层
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(128, 1)
        )
        
    def forward(self, points):
        """
        Args:
            points: (B, N, 3) - 输入点云
        Returns:
            validity: (B, 1) - 真实性分数 (未经sigmoid)
        """
        # 转换为 (B, 3, N)
        if points.dim() == 3 and points.size(-1) == 3:
            points = points.transpose(1, 2).contiguous()
        
        # 多尺度特征提取
        l1_xyz, l1_points = self.sa1(points, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # 全局特征: (B, hidden_dim, 1) -> (B, hidden_dim)
        global_feat = l3_points.squeeze(-1)
        
        # 判别
        validity = self.fc_layers(global_feat)
        
        return validity


class LocalGlobalDiscriminator(nn.Module):
    """
    局部-全局判别器 - 同时判断局部结构和全局形状的真实性
    
    设计思想：
    1. 局部分支：判断局部几何结构的连贯性
    2. 全局分支：判断整体形状的合理性
    3. 融合两个分支的判别结果
    """
    
    def __init__(self, input_dim=3, hidden_dim=256):
        super(LocalGlobalDiscriminator, self).__init__()
        
        # 局部几何特征提取
        self.local_encoder = nn.ModuleList([
            EdgeConv(input_dim, 64, k=16),
            EdgeConv(64, 128, k=16),
            EdgeConv(128, 256, k=8)
        ])
        
        # 全局形状特征提取
        self.global_sa1 = PointNet_SA_Module_KNN(
            npoint=512, nsample=32, in_channel=input_dim,
            mlp=[64, 128], group_all=False, if_bn=True
        )
        
        self.global_sa2 = PointNet_SA_Module_KNN(
            npoint=128, nsample=32, in_channel=128,
            mlp=[256, hidden_dim], group_all=False, if_bn=True
        )
        
        self.global_sa3 = PointNet_SA_Module_KNN(
            npoint=None, nsample=None, in_channel=hidden_dim,
            mlp=[hidden_dim, hidden_dim], group_all=True, if_bn=True
        )
        
        # 局部判别分支
        self.local_discriminator = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 64, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 1)
        )
        
        # 全局判别分支
        self.global_discriminator = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(2, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1)
        )
        
    def forward(self, points, return_features=False):
        """
        Args:
            points: (B, N, 3) or (B, 3, N) - 输入点云
            return_features: 是否返回中间特征
        Returns:
            validity: (B, 1) - 真实性分数
            features: (可选) 中间特征字典
        """
        # 确保输入格式为 (B, 3, N)
        if points.dim() == 3 and points.size(1) != 3:
            points = points.transpose(1, 2).contiguous()
        
        # 局部特征提取
        local_feat = points
        for conv in self.local_encoder:
            local_feat = conv(local_feat)
        
        # 全局特征提取
        global_xyz, global_feat1 = self.global_sa1(points, points)
        global_xyz, global_feat2 = self.global_sa2(global_xyz, global_feat1)
        _, global_feat3 = self.global_sa3(global_xyz, global_feat2)
        global_feat = global_feat3.squeeze(-1)
        
        # 局部判别
        local_validity = self.local_discriminator(local_feat)
        
        # 全局判别
        global_validity = self.global_discriminator(global_feat)
        
        # 融合判别结果
        combined = torch.cat([local_validity, global_validity], dim=1)
        validity = self.fusion(combined)
        
        if return_features:
            features = {
                'local_feat': local_feat,
                'global_feat': global_feat,
                'local_validity': local_validity,
                'global_validity': global_validity
            }
            return validity, features
        
        return validity


class SpectralDiscriminator(nn.Module):
    """
    频谱域判别器 - 在频谱域判断点云的真实性
    
    设计思想：
    1. 真实点云和生成点云在频谱域有不同的特征分布
    2. 使用图傅里叶变换提取频谱特征
    3. 判断频谱特征的真实性
    """
    
    def __init__(self, input_dim=3, hidden_dim=256, spectral_k=16):
        super(SpectralDiscriminator, self).__init__()
        
        self.spectral_k = spectral_k
        
        # 频谱特征提取
        from models.GeoSpecNet import SpectralGraphConv, PCSA
        
        self.spectral_conv1 = SpectralGraphConv(input_dim, 64, k=spectral_k)
        self.spectral_conv2 = SpectralGraphConv(64, 128, k=spectral_k)
        self.spectral_conv3 = SpectralGraphConv(128, 256, k=spectral_k)
        
        # 全局聚合
        self.global_pool = nn.Sequential(
            nn.Conv1d(256, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.AdaptiveMaxPool1d(1)
        )
        
        # 判别层
        self.discriminator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )
        
    def forward(self, points):
        """
        Args:
            points: (B, N, 3) or (B, 3, N) - 输入点云
        Returns:
            validity: (B, 1) - 真实性分数
        """
        # 确保格式为 (B, 3, N)
        if points.dim() == 3 and points.size(1) != 3:
            pos = points.transpose(1, 2).contiguous()
            points = pos
        else:
            pos = points
        
        # 频谱域特征提取
        feat1 = self.spectral_conv1(points, pos)
        feat2 = self.spectral_conv2(feat1, pos)
        feat3 = self.spectral_conv3(feat2, pos)
        
        # 全局聚合
        global_feat = self.global_pool(feat3)
        
        # 判别
        validity = self.discriminator(global_feat)
        
        return validity


def get_discriminator(disc_type='local_global', **kwargs):
    """
    工厂函数：根据类型创建判别器
    
    Args:
        disc_type: 判别器类型
            - 'simple': 简单的PointNet++判别器
            - 'local_global': 局部-全局判别器 (推荐)
            - 'spectral': 频谱域判别器
        **kwargs: 判别器的参数
    
    Returns:
        discriminator: 判别器实例
    """
    if disc_type == 'simple':
        return PointCloudDiscriminator(**kwargs)
    elif disc_type == 'local_global':
        return LocalGlobalDiscriminator(**kwargs)
    elif disc_type == 'spectral':
        return SpectralDiscriminator(**kwargs)
    else:
        raise ValueError(f"Unknown discriminator type: {disc_type}")


if __name__ == '__main__':
    # 测试判别器
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # 创建测试数据
    batch_size = 4
    num_points = 2048
    points = torch.rand(batch_size, num_points, 3).cuda()
    
    print("Testing Discriminators:")
    print("=" * 50)
    
    # 测试简单判别器
    print("\n1. Simple Discriminator:")
    disc_simple = PointCloudDiscriminator().cuda()
    validity_simple = disc_simple(points)
    print(f"   Input: {points.shape}")
    print(f"   Output: {validity_simple.shape}")
    print(f"   Parameters: {sum(p.numel() for p in disc_simple.parameters()):,}")
    
    # 测试局部-全局判别器
    print("\n2. Local-Global Discriminator:")
    disc_lg = LocalGlobalDiscriminator().cuda()
    validity_lg, features = disc_lg(points, return_features=True)
    print(f"   Input: {points.shape}")
    print(f"   Output: {validity_lg.shape}")
    print(f"   Local validity: {features['local_validity'].shape}")
    print(f"   Global validity: {features['global_validity'].shape}")
    print(f"   Parameters: {sum(p.numel() for p in disc_lg.parameters()):,}")
    
    # 测试频谱判别器
    print("\n3. Spectral Discriminator:")
    disc_spectral = SpectralDiscriminator().cuda()
    validity_spectral = disc_spectral(points)
    print(f"   Input: {points.shape}")
    print(f"   Output: {validity_spectral.shape}")
    print(f"   Parameters: {sum(p.numel() for p in disc_spectral.parameters()):,}")
    
    print("\n" + "=" * 50)
    print("All discriminators tested successfully!")
