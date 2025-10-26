# GeoSpecNet 快速设置指南

## 项目概述

GeoSpecNet 是一个结合频谱域处理和几何特征学习的点云补全模型。该实现整合了以下核心思想:

### 核心模块

1. **点云频谱适配器 (PCSA)** - 源自 PointGST
   - 使用图傅里叶变换(GFT)将空间特征转换到频谱域
   - 通过正交基分解消除冗余特征
   - 捕捉高频细节信息

2. **几何-频谱协同感知模块 (Geo-Spectral Collaborative Perception)**
   - 频谱分支: PCSA + 多尺度图卷积(MSGConv)
   - 几何分支: EdgeConv + 自注意力机制
   - 跨模态融合增强特征表达

3. **动态区域选择网络 (DRSN)**
   - 结构感知门控单元识别几何复杂度
   - 全局语义路径生成粗略几何分布
   - 局部细节路径精细修复细节
   - 动态路径选择实现自适应补全

4. **多视角融合编码器** - 保留 PointSea 核心思想
   - 从多个视角渲染深度图
   - 使用 ResNet18 提取视图特征
   - 与点云特征融合生成全局特征

## 文件结构

```
workspace/
├── models/
│   └── GeoSpecNet.py          # 主模型实现
├── core/
│   ├── train_geospec.py       # 训练脚本
│   ├── test_geospec.py        # 测试脚本
│   └── eval_geospec.py        # 评估脚本
├── config_geospec.py          # 配置文件
├── main_geospec.py            # 主入口
└── README_GeoSpecNet.md       # 详细文档
```

## 快速开始

### 1. 环境配置

```bash
# 安装依赖
pip install torch torchvision
pip install einops tensorboardX tqdm easydict torch-scatter

# 编译 PointNet2 算子
cd pointnet2_ops_lib
pip install .

# 编译 Chamfer Distance
cd metrics/CD/chamfer3D
python setup.py install
```

### 2. 数据集准备

更新 `config_geospec.py` 中的数据集路径:

```python
__C.DATASET.SHAPENET.PARTIAL_POINTS_PATH = '/your/path/to/ShapeNet55/%s/partial/%s/%s/%02d.pcd'
__C.DATASET.SHAPENET.COMPLETE_POINTS_PATH = '/your/path/to/ShapeNet55/%s/complete/%s/%s.pcd'
```

### 3. 训练

```bash
# 基础训练
python main_geospec.py --mode train \
    --gpu 0,1,2,3 \
    --batch_size 32 \
    --epochs 300 \
    --shapenet_path /path/to/ShapeNet55

# 从检查点恢复训练
python main_geospec.py --mode train \
    --resume ./output/GeoSpecNet/checkpoints/ckpt-best.pth \
    --gpu 0,1,2,3
```

### 4. 评估

```bash
# 在测试集上评估
python main_geospec.py --mode eval \
    --weights ./output/GeoSpecNet/checkpoints/ckpt-best.pth \
    --test_dataset ShapeNet \
    --gpu 0
```

## 模型架构详解

### 前向传播流程

1. **输入**: 部分点云 (B, N, 3)

2. **多视角渲染**: 
   - 从3个视角生成深度图
   - 视角距离: 1.5 (可配置)

3. **特征提取**:
   - **点云特征**: 通过 PointNet++ 提取全局特征
   - **多视角特征**: 通过 ResNet18 提取视图特征  
   - **局部几何特征**: 通过 EdgeConv 提取局部特征

4. **粗略补全**:
   - 融合点云和多视角特征
   - 生成初步的粗略点云

5. **第一阶段精细化 (DRSN)**:
   - 应用几何-频谱协同感知模块
   - 全局语义路径 + 局部细节路径
   - 结构感知门控动态选择路径
   - 4倍上采样

6. **第二阶段精细化 (DRSN)**:
   - 进一步频谱域增强
   - 2倍上采样

7. **输出**: 三个预测结果 (coarse, fine1, fine2)

### 损失函数

多阶段 Chamfer Distance 损失:

```
L = 0.5 × CD(coarse, GT) + 0.3 × CD(fine1, GT) + 1.0 × CD(fine2, GT)
```

## 关键超参数

```python
# 网络架构
view_distance = 1.5      # 多视角渲染距离
local_points = 512       # 局部特征提取点数
merge_points = 1024      # 合并后的点数
step1 = 4                # 第一阶段上采样比例
step2 = 2                # 第二阶段上采样比例
spectral_k = 16          # 频谱卷积的k值
msg_scales = [8, 16, 32] # 多尺度图卷积的尺度

# 训练参数
batch_size = 32
epochs = 300
learning_rate = 0.0001
lr_decay_step = [150, 200, 250]
warmup_steps = 200
```

## 技术亮点

### 1. 频谱域处理
- **图傅里叶变换**: 将点云从空间域转换到频谱域
- **多尺度频谱卷积**: 在不同尺度(k=8,16,32)提取特征
- **频谱-空间融合**: 通过通道注意力自适应融合

### 2. 几何-频谱协同
- **双分支架构**: 频谱分支 + 几何分支并行处理
- **跨模态注意力**: 频谱特征和几何特征交互增强
- **协同建模**: 同步捕捉全局结构和局部细节

### 3. 动态区域选择
- **结构复杂度估计**: 通过门控单元评估每个区域的几何复杂度
- **双路径策略**: 
  - 全局路径: 适合简单几何,生成粗略结构
  - 局部路径: 适合复杂几何,精细化细节
- **自适应融合**: 基于复杂度动态选择路径权重

### 4. 多视角增强
- **深度图渲染**: 从多个视角投影点云到深度图
- **视图特征提取**: 使用预训练ResNet提取2D特征
- **跨域融合**: 结合3D点云特征和2D视图特征

## 预期性能

在 ShapeNet55 测试集上的预期性能:

| 阶段 | Chamfer Distance (×10⁻³) |
|------|--------------------------|
| Coarse | 6-7 |
| Fine1  | 4-5 |
| Fine2  | 3-4 |

## 调试和优化

### 常见问题

1. **内存不足**
   - 减小 batch_size
   - 减小 merge_points 或 local_points
   - 使用梯度累积

2. **训练不稳定**
   - 检查学习率设置
   - 增加 warmup_steps
   - 检查数据归一化

3. **性能不佳**
   - 增加训练轮数
   - 调整 step1 和 step2 的上采样比例
   - 尝试不同的 spectral_k 值

### 可视化

训练过程中可以通过 TensorBoard 监控:

```bash
tensorboard --logdir ./output/GeoSpecNet/logs
```

## 扩展和改进

### 可能的改进方向

1. **自监督结构一致性训练**
   - 添加 GAN 约束增强结构一致性
   - 使用部分匹配损失保持可见区域不变

2. **更复杂的频谱变换**
   - 使用更高阶的图拉普拉斯算子
   - 尝试学习性的频谱基函数

3. **多尺度融合**
   - 在更多尺度上提取特征
   - 使用金字塔式的特征融合

4. **注意力机制增强**
   - 使用 Transformer 架构
   - 添加空间-频谱联合注意力

## 引用

如果您使用此代码,请引用相关论文:

- **PointGST**: 频谱域变换的灵感来源
- **PointSea**: 多视角融合的核心思想
- **PointNet++**: 分层特征提取
- **PointCFormer**: 局部几何感知

## 联系方式

如有问题或建议,请在 GitHub 上提交 issue。
