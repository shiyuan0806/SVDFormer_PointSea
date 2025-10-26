"""
GAN Loss Functions and Partial Matching Loss
用于自监督结构一致性训练的损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics.CD.chamfer3D.dist_chamfer_3D import chamfer_3DDist


class PartialMatchingLoss(nn.Module):
    """
    部分匹配损失 (Partial Matching Loss)
    
    目标：确保输入点云的可见区域在补全过程中保持不变
    
    实现思路：
    1. 对于补全结果中的每个点，找到输入部分点云中最近的点
    2. 计算这些最近点的距离
    3. 只对距离小于阈值的点计算损失
    4. 这样可以确保补全结果包含输入的可见部分
    """
    
    def __init__(self, threshold=0.05, weight=1.0):
        """
        Args:
            threshold: 距离阈值，用于判断点是否属于可见区域
            weight: 损失权重
        """
        super(PartialMatchingLoss, self).__init__()
        self.threshold = threshold
        self.weight = weight
        self.chamfer_dist = chamfer_3DDist()
        
    def forward(self, completed, partial):
        """
        Args:
            completed: (B, N_complete, 3) - 补全结果
            partial: (B, N_partial, 3) - 输入部分点云
        Returns:
            loss: 部分匹配损失
        """
        # 计算从补全结果到部分点云的最近距离
        # dist_complete_to_partial: (B, N_complete)
        dist_complete_to_partial = self.chamfer_dist(
            completed.contiguous(),
            partial.contiguous()
        )[0]
        
        # 只对距离小于阈值的点计算损失（这些点应该属于可见区域）
        mask = (dist_complete_to_partial < self.threshold).float()
        
        # 计算被遮罩的距离的平均值
        if mask.sum() > 0:
            loss = (dist_complete_to_partial * mask).sum() / (mask.sum() + 1e-6)
        else:
            loss = torch.tensor(0.0, device=completed.device)
        
        return self.weight * loss
    
    def compute_matching_accuracy(self, completed, partial):
        """
        计算匹配准确率：有多少比例的可见点被正确保留
        
        Args:
            completed: (B, N_complete, 3) - 补全结果
            partial: (B, N_partial, 3) - 输入部分点云
        Returns:
            accuracy: 匹配准确率 (0-1)
        """
        with torch.no_grad():
            dist_partial_to_complete = self.chamfer_dist(
                partial.contiguous(),
                completed.contiguous()
            )[0]
            
            # 计算有多少输入点在补全结果中有近邻（距离小于阈值）
            matched = (dist_partial_to_complete < self.threshold).float()
            accuracy = matched.mean()
            
        return accuracy


class ConsistencyLoss(nn.Module):
    """
    一致性损失 - 确保补全结果的局部结构与输入一致
    
    通过比较局部邻域的几何特征来确保结构一致性
    """
    
    def __init__(self, k=16, weight=1.0):
        """
        Args:
            k: 邻域大小
            weight: 损失权重
        """
        super(ConsistencyLoss, self).__init__()
        self.k = k
        self.weight = weight
        
    def compute_local_geometry(self, points, k):
        """
        计算局部几何特征
        
        Args:
            points: (B, N, 3) - 点云
            k: 邻域大小
        Returns:
            features: (B, N, k*3) - 局部几何特征
        """
        from models.model_utils import knn
        
        # 转换为 (B, 3, N)
        if points.size(1) != 3:
            points_t = points.transpose(1, 2).contiguous()
        else:
            points_t = points
            points = points.transpose(1, 2).contiguous()
        
        # 找k近邻
        idx = knn(points_t, k)  # (B, N, k)
        
        # 获取邻居点
        batch_size, num_points, _ = points.size()
        idx_base = torch.arange(0, batch_size, device=points.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        
        points_flat = points.view(batch_size * num_points, 3)
        neighbors = points_flat[idx].view(batch_size, num_points, k, 3)
        
        # 计算相对位置
        center = points.unsqueeze(2).expand_as(neighbors)
        relative_pos = neighbors - center  # (B, N, k, 3)
        
        # 展平为特征
        features = relative_pos.view(batch_size, num_points, k * 3)
        
        return features
    
    def forward(self, completed, partial):
        """
        Args:
            completed: (B, N_complete, 3) - 补全结果
            partial: (B, N_partial, 3) - 输入部分点云
        Returns:
            loss: 一致性损失
        """
        # 对部分点云的每个点，找到补全结果中最近的点
        from metrics.CD.chamfer3D.dist_chamfer_3D import chamfer_3DDist
        
        chamfer_dist = chamfer_3DDist()
        dist, idx = chamfer_dist(
            partial.contiguous(),
            completed.contiguous()
        )[0:2]
        
        # 选择补全结果中对应的点
        batch_size = completed.size(0)
        idx = idx.long()
        
        # 收集对应点
        matched_points = torch.stack([
            completed[i][idx[i]] for i in range(batch_size)
        ])
        
        # 计算局部几何特征
        partial_features = self.compute_local_geometry(partial, self.k)
        matched_features = self.compute_local_geometry(matched_points, self.k)
        
        # 计算特征差异
        loss = F.mse_loss(matched_features, partial_features)
        
        return self.weight * loss


class GANLoss(nn.Module):
    """
    GAN损失函数
    
    支持多种GAN损失：
    - vanilla: 原始GAN损失 (BCE)
    - lsgan: 最小二乘GAN
    - wgan: Wasserstein GAN
    """
    
    def __init__(self, gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0):
        """
        Args:
            gan_mode: GAN损失类型 ('vanilla' | 'lsgan' | 'wgan')
            target_real_label: 真实标签的目标值
            target_fake_label: 假标签的目标值
        """
        super(GANLoss, self).__init__()
        self.gan_mode = gan_mode
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'wgan':
            self.loss = None
        else:
            raise NotImplementedError(f'GAN mode {gan_mode} not implemented')
    
    def get_target_tensor(self, prediction, target_is_real):
        """
        创建目标张量
        
        Args:
            prediction: 判别器的输出
            target_is_real: 是真实样本还是假样本
        Returns:
            target: 目标张量
        """
        if target_is_real:
            target = self.real_label
        else:
            target = self.fake_label
        
        return target.expand_as(prediction)
    
    def __call__(self, prediction, target_is_real):
        """
        计算GAN损失
        
        Args:
            prediction: (B, 1) - 判别器输出
            target_is_real: 是否为真实样本
        Returns:
            loss: GAN损失
        """
        if self.gan_mode == 'wgan':
            # Wasserstein距离
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        else:
            target = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target)
        
        return loss


class StructuralConsistencyLoss(nn.Module):
    """
    结构一致性损失 - 组合多个损失以确保结构一致性
    
    包括：
    1. 部分匹配损失：确保可见区域保持不变
    2. 一致性损失：确保局部结构一致
    3. GAN损失：确保整体分布真实
    """
    
    def __init__(self, 
                 use_partial_matching=True,
                 use_consistency=True,
                 partial_matching_weight=1.0,
                 consistency_weight=0.5,
                 gan_weight=0.1,
                 gan_mode='lsgan'):
        """
        Args:
            use_partial_matching: 是否使用部分匹配损失
            use_consistency: 是否使用一致性损失
            partial_matching_weight: 部分匹配损失权重
            consistency_weight: 一致性损失权重
            gan_weight: GAN损失权重
            gan_mode: GAN损失类型
        """
        super(StructuralConsistencyLoss, self).__init__()
        
        self.use_partial_matching = use_partial_matching
        self.use_consistency = use_consistency
        
        # 初始化各个损失
        if use_partial_matching:
            self.partial_matching_loss = PartialMatchingLoss(weight=partial_matching_weight)
        
        if use_consistency:
            self.consistency_loss = ConsistencyLoss(weight=consistency_weight)
        
        self.gan_loss = GANLoss(gan_mode=gan_mode)
        self.gan_weight = gan_weight
    
    def compute_generator_loss(self, completed, partial, discriminator):
        """
        计算生成器的总损失
        
        Args:
            completed: (B, N, 3) - 补全结果
            partial: (B, N_partial, 3) - 输入部分点云
            discriminator: 判别器网络
        Returns:
            loss_dict: 包含各项损失的字典
        """
        loss_dict = {}
        total_loss = 0.0
        
        # 部分匹配损失
        if self.use_partial_matching:
            pm_loss = self.partial_matching_loss(completed, partial)
            loss_dict['partial_matching'] = pm_loss.item()
            total_loss += pm_loss
        
        # 一致性损失
        if self.use_consistency:
            cons_loss = self.consistency_loss(completed, partial)
            loss_dict['consistency'] = cons_loss.item()
            total_loss += cons_loss
        
        # GAN损失（生成器希望判别器认为生成的点云是真的）
        pred_fake = discriminator(completed)
        gan_loss = self.gan_loss(pred_fake, target_is_real=True) * self.gan_weight
        loss_dict['gan'] = gan_loss.item()
        total_loss += gan_loss
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def compute_discriminator_loss(self, completed, real, discriminator):
        """
        计算判别器的损失
        
        Args:
            completed: (B, N, 3) - 补全结果（假样本）
            real: (B, N, 3) - 真实完整点云
            discriminator: 判别器网络
        Returns:
            loss: 判别器损失
            loss_dict: 包含各项损失的字典
        """
        loss_dict = {}
        
        # 真实样本的判别
        pred_real = discriminator(real)
        loss_real = self.gan_loss(pred_real, target_is_real=True)
        
        # 假样本的判别（detach以避免梯度传播到生成器）
        pred_fake = discriminator(completed.detach())
        loss_fake = self.gan_loss(pred_fake, target_is_real=False)
        
        # 总判别器损失
        disc_loss = (loss_real + loss_fake) * 0.5
        
        loss_dict['disc_real'] = loss_real.item()
        loss_dict['disc_fake'] = loss_fake.item()
        loss_dict['disc_total'] = disc_loss.item()
        
        # 计算判别准确率
        with torch.no_grad():
            pred_real_prob = torch.sigmoid(pred_real)
            pred_fake_prob = torch.sigmoid(pred_fake)
            acc_real = (pred_real_prob > 0.5).float().mean()
            acc_fake = (pred_fake_prob < 0.5).float().mean()
            loss_dict['acc_real'] = acc_real.item()
            loss_dict['acc_fake'] = acc_fake.item()
        
        return disc_loss, loss_dict


if __name__ == '__main__':
    # 测试损失函数
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    print("Testing GAN Loss Functions:")
    print("=" * 60)
    
    # 创建测试数据
    batch_size = 4
    n_complete = 2048
    n_partial = 1024
    
    completed = torch.rand(batch_size, n_complete, 3).cuda()
    partial = torch.rand(batch_size, n_partial, 3).cuda()
    real = torch.rand(batch_size, n_complete, 3).cuda()
    
    # 测试部分匹配损失
    print("\n1. Partial Matching Loss:")
    pm_loss_fn = PartialMatchingLoss().cuda()
    pm_loss = pm_loss_fn(completed, partial)
    accuracy = pm_loss_fn.compute_matching_accuracy(completed, partial)
    print(f"   Loss: {pm_loss.item():.6f}")
    print(f"   Matching Accuracy: {accuracy:.4f}")
    
    # 测试一致性损失
    print("\n2. Consistency Loss:")
    cons_loss_fn = ConsistencyLoss().cuda()
    cons_loss = cons_loss_fn(completed, partial)
    print(f"   Loss: {cons_loss.item():.6f}")
    
    # 测试GAN损失
    print("\n3. GAN Loss:")
    for gan_mode in ['vanilla', 'lsgan', 'wgan']:
        gan_loss_fn = GANLoss(gan_mode=gan_mode)
        pred = torch.rand(batch_size, 1).cuda()
        loss_real = gan_loss_fn(pred, target_is_real=True)
        loss_fake = gan_loss_fn(pred, target_is_real=False)
        print(f"   {gan_mode:8s} - Real: {loss_real.item():.6f}, Fake: {loss_fake.item():.6f}")
    
    # 测试结构一致性损失
    print("\n4. Structural Consistency Loss:")
    from models.discriminator import LocalGlobalDiscriminator
    
    discriminator = LocalGlobalDiscriminator().cuda()
    sc_loss_fn = StructuralConsistencyLoss().cuda()
    
    # 生成器损失
    gen_loss, gen_loss_dict = sc_loss_fn.compute_generator_loss(
        completed, partial, discriminator
    )
    print(f"   Generator Loss: {gen_loss.item():.6f}")
    for k, v in gen_loss_dict.items():
        print(f"      {k}: {v:.6f}")
    
    # 判别器损失
    disc_loss, disc_loss_dict = sc_loss_fn.compute_discriminator_loss(
        completed, real, discriminator
    )
    print(f"   Discriminator Loss: {disc_loss.item():.6f}")
    for k, v in disc_loss_dict.items():
        print(f"      {k}: {v:.6f}")
    
    print("\n" + "=" * 60)
    print("All loss functions tested successfully!")
