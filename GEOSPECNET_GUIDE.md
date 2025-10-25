# GeoSpecNet 项目完整指南

## 📁 已创建的文件列表

### 核心模型文件
```
models/
└── GeoSpecNet.py                      # 主模型实现（包含所有核心模块）
    ├── GraphFourierTransform          # 图傅里叶变换
    ├── MultiScaleGraphConv            # 多尺度图卷积
    ├── GeoSpectralCollaborativeModule # 几何-频谱协同感知模块
    ├── StructureAwareGatingUnit       # 结构感知门控单元
    ├── DynamicRegionSelectionNetwork  # 动态区域选择网络
    ├── GeoSpecNetEncoder              # 编码器
    ├── GeoSpecNetDecoder              # 解码器
    ├── PointCloudDiscriminator        # 判别器（GAN训练）
    └── GeoSpecNet                     # 完整模型
```

### 配置文件
```
config_geospecnet.py                   # 模型配置（网络参数、训练参数等）
```

### 训练和测试脚本
```
core/
├── train_geospecnet.py                # 训练脚本
└── test_geospecnet.py                 # 测试脚本
```

### 主入口文件
```
main_geospecnet.py                     # 主程序入口（支持训练、测试、推理）
```

### 损失函数
```
utils/
└── loss_geospecnet.py                 # 损失函数模块
    ├── GeoSpecNetLoss                 # 综合损失函数
    ├── PartialMatchingLoss            # 部分匹配损失
    ├── StructuralConsistencyLoss      # 结构一致性损失
    ├── DensityLoss                    # 密度损失
    ├── RepulsionLoss                  # 排斥损失
    └── GANLoss                        # GAN损失
```

### 文档和示例
```
README_GeoSpecNet.md                   # 项目说明文档（中英双语）
GEOSPECNET_GUIDE.md                    # 本文件（完整指南）
requirements_geospecnet.txt            # Python依赖列表

examples/
├── train_example.py                   # 训练示例脚本
└── visualize_completion.py            # 可视化工具
```

---

## 🚀 快速开始指南

### 1. 环境准备

#### 安装依赖
```bash
# 基础依赖
pip install -r requirements_geospecnet.txt

# 安装PointNet++ CUDA操作
cd pointnet2_ops_lib
pip install -e .

# 安装Chamfer Distance
cd ../metrics/CD/chamfer3D
python setup.py install

# 安装EMD
cd ../../EMD
python setup.py install
cd ../../..
```

### 2. 数据准备

```bash
# ShapeNet-55数据集结构
shapenet_pc/
├── 02691156/        # 飞机类别
│   ├── model1.pcd
│   ├── model2.pcd
│   └── ...
├── 02933112/        # 柜子类别
└── ...

# KITTI数据集结构
kitti/
├── train/
│   ├── partial/
│   └── complete/
└── test/
    ├── partial/
    └── complete/
```

### 3. 训练模型

#### 基础训练
```bash
python main_geospecnet.py --train
```

#### 自定义训练
```bash
python main_geospecnet.py --train \
    --gpu 0,1,2,3 \
    --batch-size 32 \
    --epochs 400 \
    --lr 0.0002 \
    --output ./output/experiment1
```

#### 继续训练（从checkpoint恢复）
```bash
python main_geospecnet.py --train \
    --weights path/to/checkpoint.pth
```

### 4. 测试模型

```bash
python main_geospecnet.py --test \
    --weights path/to/best_model.pth \
    --dataset ShapeNet55
```

### 5. 推理模式

```python
import torch
import numpy as np
from models.GeoSpecNet import GeoSpecNet
from config_geospecnet import cfg

# 加载模型
model = GeoSpecNet(cfg)
checkpoint = torch.load('path/to/checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval().cuda()

# 加载部分点云
partial = np.load('partial_cloud.npy')  # (N, 3)
partial = torch.from_numpy(partial).float().unsqueeze(0).cuda()

# 补全
with torch.no_grad():
    coarse, fine1, fine2 = model(partial)

# 保存结果
completion = fine2.squeeze(0).cpu().numpy()
np.save('completion.npy', completion)
```

---

## 🔬 核心模块详解

### 1. 图傅里叶变换（Graph Fourier Transform, GFT）

**位置**: `models/GeoSpecNet.py` -> `GraphFourierTransform`

**功能**: 将点云从空间域转换到频谱域

**来源**: 频谱图理论，用于信号处理

**关键代码**:
```python
gft = GraphFourierTransform(in_channels=3, out_channels=256, k_neighbors=16)
spectral_features = gft(xyz, features)
```

**论文依据**:
- 图信号处理（Graph Signal Processing）
- 频谱域建模可以消除特征冗余
- 捕捉高频细节信息

### 2. 多尺度图卷积（Multi-Scale Graph Convolution, MSGConv）

**位置**: `models/GeoSpecNet.py` -> `MultiScaleGraphConv`

**功能**: 在多个尺度上提取几何特征

**来源**: PointNet++ (NIPS 2017)

**关键代码**:
```python
msgconv = MultiScaleGraphConv(in_channels=3, out_channels=256, scales=[8, 16, 32])
multi_scale_features = msgconv(input_features)
```

### 3. 几何-频谱协同感知模块

**位置**: `models/GeoSpecNet.py` -> `GeoSpectralCollaborativeModule`

**功能**: 融合空间域和频谱域特征

**来源**: 
- GFT: 频谱域建模
- LGRP (PointCFormer CVPR 2025): 局部几何感知
- Point-Transformer (ICCV 2021): 注意力机制

**关键代码**:
```python
geo_spectral = GeoSpectralCollaborativeModule(in_channels=3, hidden_dim=256)
fused_features = geo_spectral(xyz, features)
```

### 4. 动态区域选择网络（DRSN）

**位置**: `models/GeoSpecNet.py` -> `DynamicRegionSelectionNetwork`

**功能**: 通过双路径策略实现自适应补全

**来源**:
- 结构感知门控: 注意力机制
- 双路径策略: SnowflakeNet启发

**关键代码**:
```python
drsn = DynamicRegionSelectionNetwork(hidden_dim=512, ratio=2)
refined_xyz, gates = drsn(global_feat, local_feat, coarse_xyz)
```

**双路径说明**:
- **全局语义路径**: 基于形状先验，生成粗略结构
- **局部细节路径**: 基于局部相似性，恢复精细细节

### 5. GAN训练机制

**位置**: `models/GeoSpecNet.py` -> `PointCloudDiscriminator`

**功能**: 通过对抗训练提升结构一致性

**来源**:
- GAN-based Shape Completion (CVPR 2019)
- PointOutNet (ECCV 2020)

**训练流程**:
```python
# 判别器损失
d_loss, g_loss = model.compute_gan_loss(completed, ground_truth)

# 生成器损失
total_loss = cd_loss + gan_g_loss * weight
```

---

## 📊 配置参数详解

### 网络参数 (`cfg.NETWORK`)

| 参数 | 默认值 | 说明 |
|-----|--------|-----|
| `num_coarse` | 1024 | 粗略补全的点数 |
| `stage1_ratio` | 2 | 第一阶段细化比例（1024 -> 2048） |
| `stage2_ratio` | 4 | 第二阶段细化比例（1024 -> 4096） |
| `hidden_dim` | 512 | 隐藏层特征维度 |
| `spectral_dim` | 256 | 频谱特征维度 |
| `k_neighbors` | 16 | K近邻图的邻居数 |
| `msg_conv_scales` | [8,16,32] | 多尺度图卷积的尺度 |
| `drsn_nhead` | 8 | DRSN注意力头数 |
| `use_gan` | True | 是否使用GAN训练 |

### 训练参数 (`cfg.TRAIN`)

| 参数 | 默认值 | 说明 |
|-----|--------|-----|
| `BATCH_SIZE` | 32 | 批次大小 |
| `N_EPOCHS` | 400 | 训练轮数 |
| `LEARNING_RATE` | 0.0002 | 初始学习率 |
| `WARMUP_STEPS` | 500 | 预热步数 |
| `LR_MILESTONES` | [100,200,300] | 学习率衰减里程碑 |
| `GAMMA` | 0.5 | 学习率衰减因子 |

### 损失权重 (`cfg.TRAIN.LOSS_WEIGHTS`)

| 参数 | 默认值 | 说明 |
|-----|--------|-----|
| `CD_COARSE` | 1.0 | 粗略补全的Chamfer Distance权重 |
| `CD_FINE1` | 2.0 | 第一次细化的CD权重 |
| `CD_FINE2` | 4.0 | 最终补全的CD权重 |
| `PARTIAL_MATCH` | 0.5 | 部分匹配损失权重 |
| `GAN_G` | 0.1 | GAN生成器损失权重 |
| `GAN_D` | 0.05 | GAN判别器损失权重 |

---

## 🧪 测试和验证

### 运行单元测试

```bash
# 测试模型前向传播
cd examples
python train_example.py
```

### 可视化补全结果

```bash
cd examples
python visualize_completion.py
```

### 评估指标

**Chamfer Distance (CD)**:
```python
from metrics.CD.chamfer3D.dist_chamfer_3D import chamfer_3DDist

cd_loss = chamfer_3DDist()
dist1, dist2 = cd_loss(pred, gt)
cd = (dist1.mean() + dist2.mean()).item()
```

**F-Score**:
```python
# 在 core/test_geospecnet.py 中实现
f_score = compute_f_score(pred, gt, threshold=0.01)
```

---

## 🎨 可视化工具

### 使用Matplotlib

```python
from examples.visualize_completion import visualize_completion_stages

visualize_completion_stages(partial, coarse, fine1, fine2, gt)
```

### 使用Open3D（交互式）

```python
from examples.visualize_completion import compare_completion_open3d

compare_completion_open3d(partial, completion, gt)
```

---

## 🐛 常见问题

### Q1: CUDA out of memory

**解决方案**:
- 减小 `batch_size`
- 减小 `num_coarse` 点数
- 使用梯度累积

```python
# 在train_geospecnet.py中添加梯度累积
accumulation_steps = 4
if (idx + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

### Q2: 训练不收敛

**解决方案**:
- 检查学习率（可能太大或太小）
- 增加预热步数（`WARMUP_STEPS`）
- 调整损失权重
- 延迟GAN训练（`GAN_START_EPOCH`）

### Q3: 补全结果有孔洞

**解决方案**:
- 增加 `stage2_ratio` 生成更多点
- 调整 `PARTIAL_MATCH` 损失权重
- 增加训练轮数

### Q4: 模型训练速度慢

**解决方案**:
- 使用多GPU训练
- 启用 `torch.backends.cudnn.benchmark = True`
- 减少 `k_neighbors` 数量
- 使用混合精度训练

---

## 📈 性能优化建议

### 1. 数据加载优化
```python
# 在DataLoader中设置
num_workers = 8
pin_memory = True
prefetch_factor = 2
```

### 2. 混合精度训练
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    coarse, fine1, fine2 = model(partial)
    loss = criterion(partial, coarse, fine1, fine2, gt)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3. 分布式训练
```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    main_geospecnet.py --train
```

---

## 📝 引用

如果您使用了本项目，请引用：

```bibtex
@article{geospecnet2025,
  title={GeoSpecNet: Geometric-Spectral Collaborative Perception for Point Cloud Completion},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

---

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

---

## 📧 联系方式

- Issue: GitHub Issues
- Email: your.email@example.com

---

## 📜 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 🎉 致谢

感谢以下项目和论文的启发：

- **PointNet++** (Qi et al., NIPS 2017)
- **PointCFormer** (CVPR 2025)
- **Point-Transformer** (Zhao et al., ICCV 2021)
- **SnowflakeNet** (Xiang et al.)
- **GAN-based Shape Completion** (CVPR 2019)
- **PointOutNet** (ECCV 2020)

---

**最后更新**: 2025-10-25
