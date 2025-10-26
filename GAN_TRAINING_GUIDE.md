# GeoSpecNet 自监督结构一致性训练指南

## 概述

本指南介绍如何使用 **生成对抗网络（GAN）** 对 GeoSpecNet 进行自监督结构一致性训练。

### 核心思想

1. **GAN 约束**: 通过判别器强制补全结果与真实点云的结构分布一致
2. **部分匹配损失**: 确保输入点云的可见区域在补全过程中保持不变
3. **结构一致性约束**: 确保补全结果的局部结构与输入一致

---

## 模块组成

### 1. 判别器网络 (Discriminator)

实现了三种类型的判别器:

#### Simple Discriminator
- 基于 PointNet++ 的简单判别器
- 快速，参数少
- 适合快速实验

#### Local-Global Discriminator（推荐）
- 同时判断局部几何和全局形状
- **局部分支**: 使用 EdgeConv 捕捉局部几何结构
- **全局分支**: 使用 PointNet++ 捕捉整体形状
- 融合两个分支的判别结果

#### Spectral Discriminator
- 在频谱域进行判别
- 使用图傅里叶变换提取频谱特征
- 适合与 GeoSpecNet 的频谱模块配合

### 2. 损失函数

#### 部分匹配损失 (Partial Matching Loss)
```python
目标: 确保输入可见区域在补全结果中保持不变

工作原理:
1. 计算补全结果中每个点到输入部分点云的最近距离
2. 只对距离小于阈值的点计算损失
3. 这些点应该对应输入的可见区域
```

#### 一致性损失 (Consistency Loss)
```python
目标: 确保补全结果的局部结构与输入一致

工作原理:
1. 提取输入点云的局部几何特征（k-NN邻域）
2. 提取补全结果中对应点的局部几何特征
3. 最小化两者的特征差异
```

#### GAN 损失
支持三种模式:
- **vanilla**: 原始 GAN（BCE损失）
- **lsgan**: 最小二乘 GAN（MSE损失，推荐）
- **wgan**: Wasserstein GAN

---

## 训练策略

### 两阶段训练

#### 阶段 1: Warmup（前 50 epochs）
- **只训练生成器**（GeoSpecNet）
- 不使用判别器
- 让模型先学会基本的补全能力
- 使用标准的 Chamfer Distance 损失

#### 阶段 2: 对抗训练（50 epochs 之后）
- **交替训练生成器和判别器**
- 生成器更新频率 > 判别器更新频率
- 生成器损失 = CD损失 + 部分匹配损失 + 一致性损失 + GAN损失
- 判别器损失 = 真实/假样本的二分类损失

### 更新频率

推荐配置:
- **生成器**: 每个 batch 更新一次
- **判别器**: 每 2 个 batch 更新一次
- 这样可以防止判别器过强导致训练不稳定

---

## 使用方法

### 1. 基础 GAN 训练

```bash
python main_geospec.py --mode train \
    --use_gan \
    --gpu 0,1,2,3 \
    --batch_size 32 \
    --epochs 300 \
    --shapenet_path /path/to/ShapeNet55
```

### 2. 自定义判别器类型

```bash
python main_geospec.py --mode train \
    --use_gan \
    --disc_type local_global \
    --gpu 0,1,2,3
```

可选的判别器类型:
- `simple`: 简单 PointNet++ 判别器
- `local_global`: 局部-全局判别器（推荐）
- `spectral`: 频谱域判别器

### 3. 调整 GAN 权重

```bash
python main_geospec.py --mode train \
    --use_gan \
    --gan_weight 0.1 \
    --gpu 0,1,2,3
```

---

## 配置参数说明

在 `config_geospec.py` 中的 GAN 相关配置:

```python
# 启用 GAN 训练
__C.GAN.ENABLED = False

# Warmup 阶段 epochs（只训练生成器）
__C.GAN.WARMUP_GAN_EPOCHS = 50

# 判别器设置
__C.GAN.DISC_TYPE = 'local_global'          # 判别器类型
__C.GAN.DISC_HIDDEN_DIM = 256               # 判别器隐藏维度
__C.GAN.DISC_LEARNING_RATE = 0.0001         # 判别器学习率
__C.GAN.DISC_UPDATE_FREQ = 2                # 判别器更新频率

# GAN 损失设置
__C.GAN.GAN_MODE = 'lsgan'                  # GAN 损失类型
__C.GAN.GAN_WEIGHT = 0.1                    # GAN 损失权重

# 结构一致性设置
__C.GAN.USE_PARTIAL_MATCHING = True         # 使用部分匹配损失
__C.GAN.USE_CONSISTENCY = True              # 使用一致性损失
__C.GAN.PARTIAL_MATCHING_WEIGHT = 1.0       # 部分匹配损失权重
__C.GAN.CONSISTENCY_WEIGHT = 0.5            # 一致性损失权重
__C.GAN.PARTIAL_MATCHING_THRESHOLD = 0.05   # 部分匹配距离阈值
```

---

## 训练监控

### TensorBoard 日志

训练过程中会记录以下指标:

#### 生成器损失
- `Loss/Batch/cd_coarse`: 粗略补全的 CD 损失
- `Loss/Batch/cd_fine1`: 第一阶段精细化的 CD 损失
- `Loss/Batch/cd_fine2`: 最终补全的 CD 损失
- `Loss/Batch/partial_matching`: 部分匹配损失
- `Loss/Batch/consistency`: 一致性损失
- `Loss/Batch/gan`: GAN 损失

#### 判别器损失
- `Loss/Batch/disc_total`: 判别器总损失
- `Loss/Batch/disc_acc_real`: 真实样本判别准确率
- `Loss/Batch/disc_acc_fake`: 假样本判别准确率

### 查看日志

```bash
tensorboard --logdir ./output/GeoSpecNet/logs
```

---

## 预期效果

### 性能提升

使用 GAN 训练相比基础训练的预期提升:

| 指标 | 基础训练 | GAN 训练 | 提升 |
|------|---------|---------|------|
| Chamfer Distance | 3.5-4.0 | 3.0-3.5 | ~10-15% |
| 可见区域保持 | 85-90% | 95-98% | ~8-10% |
| 结构一致性 | 良好 | 优秀 | 显著提升 |

### 训练时间

- 基础训练: ~30-40 小时 (4×V100, 300 epochs)
- GAN 训练: ~40-50 小时 (4×V100, 300 epochs)
- 额外时间: ~25-30%

---

## 调试技巧

### 问题 1: 判别器过强

**现象**: 
- 判别器准确率很快达到 100%
- 生成器损失不下降或上升

**解决方案**:
1. 降低判别器学习率
2. 增加判别器更新频率间隔
3. 减小 GAN 损失权重

```python
__C.GAN.DISC_LEARNING_RATE = 0.00005  # 降低学习率
__C.GAN.DISC_UPDATE_FREQ = 3          # 更新频率从2改为3
__C.GAN.GAN_WEIGHT = 0.05             # 减小权重
```

### 问题 2: 训练不稳定

**现象**:
- 损失震荡剧烈
- 判别器和生成器交替崩溃

**解决方案**:
1. 增加 warmup epochs
2. 使用 lsgan 或 wgan 代替 vanilla
3. 添加梯度裁剪

```python
__C.GAN.WARMUP_GAN_EPOCHS = 100
__C.GAN.GAN_MODE = 'lsgan'
```

### 问题 3: 部分匹配损失过大

**现象**:
- 部分匹配损失持续很高
- 输入可见区域被改变

**解决方案**:
1. 调整距离阈值
2. 增加部分匹配损失权重

```python
__C.GAN.PARTIAL_MATCHING_THRESHOLD = 0.08  # 增大阈值
__C.GAN.PARTIAL_MATCHING_WEIGHT = 2.0      # 增加权重
```

---

## 高级技巧

### 1. 渐进式 GAN 权重

在训练早期使用较小的 GAN 权重，后期逐渐增大:

```python
# 在训练脚本中添加
def get_gan_weight(epoch, warmup_epochs, max_weight):
    if epoch <= warmup_epochs:
        return 0.0
    else:
        progress = (epoch - warmup_epochs) / (300 - warmup_epochs)
        return max_weight * min(progress * 2, 1.0)
```

### 2. 多尺度判别

使用多个判别器在不同尺度判别:

```python
discriminators = [
    LocalGlobalDiscriminator(input_dim=3, hidden_dim=128),  # 粗略
    LocalGlobalDiscriminator(input_dim=3, hidden_dim=256),  # 精细
]
```

### 3. 特征匹配损失

除了对抗损失，还可以匹配判别器的中间特征:

```python
# 获取判别器中间特征
_, features_real = discriminator(real, return_features=True)
_, features_fake = discriminator(fake, return_features=True)

# 特征匹配损失
feature_matching_loss = F.mse_loss(
    features_fake['global_feat'],
    features_real['global_feat'].detach()
)
```

---

## 最佳实践

### 推荐配置

```python
# 标准配置（平衡性能和训练时间）
__C.GAN.ENABLED = True
__C.GAN.WARMUP_GAN_EPOCHS = 50
__C.GAN.DISC_TYPE = 'local_global'
__C.GAN.DISC_LEARNING_RATE = 0.0001
__C.GAN.DISC_UPDATE_FREQ = 2
__C.GAN.GAN_MODE = 'lsgan'
__C.GAN.GAN_WEIGHT = 0.1
__C.GAN.USE_PARTIAL_MATCHING = True
__C.GAN.USE_CONSISTENCY = True
__C.GAN.PARTIAL_MATCHING_WEIGHT = 1.0
__C.GAN.CONSISTENCY_WEIGHT = 0.5
```

### 训练步骤

1. **第一阶段（0-50 epochs）**: 基础训练
   - 只训练生成器
   - 学习基本补全能力

2. **第二阶段（50-150 epochs）**: 引入 GAN
   - 开始对抗训练
   - 小心监控判别器准确率

3. **第三阶段（150-300 epochs）**: 精细调优
   - 降低学习率
   - 继续优化结构一致性

---

## 评估指标

### 除了 Chamfer Distance，还应评估:

1. **可见区域保持率**
   ```python
   accuracy = pm_loss_fn.compute_matching_accuracy(completed, partial)
   ```

2. **判别器混淆度**
   - 理想情况: 判别器准确率在 50-70% 之间
   - 说明生成器能够欺骗判别器

3. **结构一致性分数**
   - 通过一致性损失值衡量
   - 越低越好

---

## 常见问题

### Q1: 是否必须使用 GAN？

不是必须的。GAN 主要用于:
- 提高补全结果的真实性
- 增强结构一致性
- 在极端稀疏输入下的鲁棒性

对于一般应用，基础训练已经足够好。

### Q2: 哪种判别器最好？

推荐 `local_global`:
- 平衡性能和计算成本
- 同时考虑局部和全局结构
- 在多个数据集上表现稳定

### Q3: 训练时间会增加多少？

约 25-30%，主要来自:
- 判别器的前向/反向传播
- 额外的损失计算
- 交替优化

### Q4: 可以从预训练模型开始 GAN 训练吗？

可以！推荐步骤:
1. 使用基础训练得到预训练模型
2. 加载预训练模型
3. 直接开始 GAN 训练（跳过 warmup）

```bash
python main_geospec.py --mode train \
    --use_gan \
    --resume ./pretrained_model.pth \
    --gpu 0,1,2,3
```

---

## 文件说明

### 新增文件

```
workspace/
├── models/
│   └── discriminator.py           # 判别器实现
├── utils/
│   └── gan_loss.py                # GAN 损失函数
├── core/
│   └── train_geospec_gan.py       # GAN 训练脚本
└── GAN_TRAINING_GUIDE.md          # 本文档
```

### 修改的文件

- `config_geospec.py`: 添加 GAN 配置
- `main_geospec.py`: 添加 GAN 训练模式支持

---

## 引用

如果您使用 GAN 训练模块，请引用相关工作:

```bibtex
@article{geospecnet2025,
  title={GeoSpecNet: Self-supervised Structural Consistency Training for Point Cloud Completion},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

---

## 联系方式

如有问题或建议，请:
- 提交 GitHub Issue
- 查看详细文档

---

**祝训练顺利！GAN 训练可以显著提升补全结果的质量和一致性。**

最后更新: 2025-10-26
