"""
GeoSpecNet Training Example
示例训练脚本，展示如何使用GeoSpecNet进行训练
"""

import sys
sys.path.append('..')

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.GeoSpecNet import GeoSpecNet
from config_geospecnet import cfg
from utils.loss_geospecnet import GeoSpecNetLoss
from metrics.CD.chamfer3D.dist_chamfer_3D import chamfer_3DDist


def simple_training_example():
    """
    简单的训练示例
    """
    print("="*80)
    print("GeoSpecNet Training Example")
    print("="*80)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 创建模型
    print("\n创建模型...")
    model = GeoSpecNet(cfg).to(device)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 创建优化器
    optimizer = Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.999))
    
    # 创建损失函数
    criterion = GeoSpecNetLoss(cfg)
    cd_loss = chamfer_3DDist()
    
    # 模拟数据加载器（实际使用时应替换为真实数据）
    print("\n模拟训练数据...")
    batch_size = 4
    num_iterations = 10
    
    # 训练循环
    model.train()
    print("\n开始训练...")
    
    for iteration in tqdm(range(num_iterations), desc="Training"):
        # 模拟批次数据
        partial = torch.randn(batch_size, 2048, 3).to(device)
        gt = torch.randn(batch_size, 8192, 3).to(device)
        
        # 前向传播
        coarse, fine1, fine2 = model(partial)
        
        # 计算损失
        cd_coarse, _ = cd_loss(coarse, gt)
        cd_fine1, _ = cd_loss(fine1, gt)
        cd_fine2, _ = cd_loss(fine2, gt)
        
        # 总损失
        loss = cd_coarse.mean() + cd_fine1.mean() * 2 + cd_fine2.mean() * 4
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 打印损失
        if iteration % 2 == 0:
            print(f"\nIteration {iteration}: Loss = {loss.item():.4f}, "
                  f"CD_Coarse = {cd_coarse.mean().item():.4f}, "
                  f"CD_Fine2 = {cd_fine2.mean().item():.4f}")
    
    print("\n训练完成！")
    
    # 保存模型
    checkpoint_path = 'geospecnet_example.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"\n模型已保存到: {checkpoint_path}")


def test_model_forward():
    """
    测试模型前向传播
    """
    print("\n" + "="*80)
    print("Testing Model Forward Pass")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = GeoSpecNet(cfg).to(device)
    model.eval()
    
    # 测试输入
    batch_size = 2
    num_partial_points = 2048
    
    print(f"\n输入:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Partial points: {num_partial_points}")
    
    partial = torch.randn(batch_size, num_partial_points, 3).to(device)
    
    # 前向传播
    print(f"\n前向传播...")
    with torch.no_grad():
        coarse, fine1, fine2 = model(partial)
    
    # 输出形状
    print(f"\n输出:")
    print(f"  - Coarse: {coarse.shape}")
    print(f"  - Fine1: {fine1.shape}")
    print(f"  - Fine2: {fine2.shape}")
    
    print(f"\n✓ 前向传播测试成功!")


def test_individual_modules():
    """
    测试各个模块
    """
    print("\n" + "="*80)
    print("Testing Individual Modules")
    print("="*80)
    
    from models.GeoSpecNet import (
        GraphFourierTransform,
        MultiScaleGraphConv,
        GeoSpectralCollaborativeModule,
        StructureAwareGatingUnit,
        DynamicRegionSelectionNetwork
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    num_points = 1024
    
    # 测试图傅里叶变换
    print("\n1. 测试图傅里叶变换 (GFT)...")
    gft = GraphFourierTransform(in_channels=3, out_channels=256, k_neighbors=16).to(device)
    xyz = torch.randn(batch_size, 3, num_points).to(device)
    features = torch.randn(batch_size, 3, num_points).to(device)
    spectral_feat = gft(xyz, features)
    print(f"   输入: {features.shape} -> 输出: {spectral_feat.shape}")
    print("   ✓ GFT 测试通过")
    
    # 测试多尺度图卷积
    print("\n2. 测试多尺度图卷积 (MSGConv)...")
    msgconv = MultiScaleGraphConv(in_channels=3, out_channels=256, scales=[8, 16, 32]).to(device)
    x = torch.randn(batch_size, 3, num_points).to(device)
    out = msgconv(x)
    print(f"   输入: {x.shape} -> 输出: {out.shape}")
    print("   ✓ MSGConv 测试通过")
    
    # 测试几何-频谱协同模块
    print("\n3. 测试几何-频谱协同模块...")
    geo_spectral = GeoSpectralCollaborativeModule(in_channels=3, hidden_dim=256).to(device)
    fused_feat = geo_spectral(xyz, features)
    print(f"   输入: {features.shape} -> 输出: {fused_feat.shape}")
    print("   ✓ 几何-频谱协同模块 测试通过")
    
    # 测试结构感知门控单元
    print("\n4. 测试结构感知门控单元...")
    gating = StructureAwareGatingUnit(dim=256).to(device)
    global_feat = torch.randn(batch_size, 256, num_points).to(device)
    local_feat = torch.randn(batch_size, 256, num_points).to(device)
    gates, incompleteness = gating(global_feat, local_feat)
    print(f"   输入: {global_feat.shape}, {local_feat.shape}")
    print(f"   输出: gates={gates.shape}, incompleteness={incompleteness.shape}")
    print("   ✓ 结构感知门控单元 测试通过")
    
    # 测试动态区域选择网络
    print("\n5. 测试动态区域选择网络 (DRSN)...")
    drsn = DynamicRegionSelectionNetwork(hidden_dim=256, ratio=2).to(device)
    coarse_xyz = torch.randn(batch_size, 3, num_points).to(device)
    refined_xyz, gates = drsn(global_feat, local_feat, coarse_xyz)
    print(f"   输入: coarse={coarse_xyz.shape}")
    print(f"   输出: refined={refined_xyz.shape}, gates={gates.shape}")
    print("   ✓ DRSN 测试通过")
    
    print("\n所有模块测试通过! ✓")


if __name__ == '__main__':
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                  GeoSpecNet Training Example                   ║
    ║         Geometric-Spectral Point Cloud Completion              ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    # 测试模型前向传播
    test_model_forward()
    
    # 测试各个模块
    test_individual_modules()
    
    # 简单训练示例
    print("\n是否运行训练示例? (可能需要几分钟)")
    response = input("输入 'yes' 继续: ").strip().lower()
    if response in ['yes', 'y']:
        simple_training_example()
    else:
        print("跳过训练示例")
    
    print("\n" + "="*80)
    print("示例脚本运行完成!")
    print("="*80)
