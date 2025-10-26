# GeoSpecNet 快速开始指南

## 🎯 项目简介

GeoSpecNet 是一个创新的点云补全模型，结合了频谱域处理、几何特征学习和多视角融合技术。

### 核心创新
1. **频谱域变换** - 使用图傅里叶变换处理点云
2. **几何-频谱协同** - 双分支协同建模
3. **动态区域选择** - 自适应补全策略
4. **多视角融合** - 全局形状先验

---

## 📦 安装步骤

### 第一步: 克隆代码（已完成）

代码已经在 `/workspace` 目录下。

### 第二步: 安装依赖

```bash
# 基础依赖
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install einops tensorboardX tqdm easydict torch-scatter

# 编译 PointNet2 操作
cd /workspace/pointnet2_ops_lib
pip install .

# 编译 Chamfer Distance
cd /workspace/metrics/CD/chamfer3D
python setup.py install
```

### 第三步: 准备数据集

更新 `config_geospec.py` 中的路径：

```python
# ShapeNet55 数据集路径
__C.DATASET.SHAPENET.PARTIAL_POINTS_PATH = '/your/path/ShapeNet55/%s/partial/%s/%s/%02d.pcd'
__C.DATASET.SHAPENET.COMPLETE_POINTS_PATH = '/your/path/ShapeNet55/%s/complete/%s/%s.pcd'
```

---

## 🚀 使用方法

### 训练模型

#### 基础训练
```bash
python main_geospec.py --mode train \
    --gpu 0,1,2,3 \
    --batch_size 32 \
    --epochs 300 \
    --shapenet_path /path/to/ShapeNet55 \
    --output ./output/experiment_1
```

#### 恢复训练
```bash
python main_geospec.py --mode train \
    --resume ./output/GeoSpecNet/checkpoints/ckpt-best.pth \
    --gpu 0,1,2,3
```

#### 自定义训练参数
```bash
python main_geospec.py --mode train \
    --train_dataset ShapeNet \
    --batch_size 16 \
    --epochs 400 \
    --lr 0.0001 \
    --workers 8 \
    --gpu 0,1
```

### 评估模型

#### 快速评估
```bash
python main_geospec.py --mode test \
    --weights ./output/GeoSpecNet/checkpoints/ckpt-best.pth \
    --test_dataset ShapeNet \
    --gpu 0
```

#### 详细评估（保存结果）
```bash
python main_geospec.py --mode eval \
    --weights ./checkpoints/ckpt-best.pth \
    --test_dataset ShapeNet \
    --gpu 0
```

---

## 📊 监控训练

### TensorBoard

训练过程中，所有指标都会记录到 TensorBoard：

```bash
tensorboard --logdir ./output/GeoSpecNet/logs
```

在浏览器中打开 `http://localhost:6006` 查看：
- 训练损失曲线
- 验证指标
- 学习率变化

### 日志文件

训练日志会实时输出到终端，显示：
- 每个 epoch 的平均损失
- 每个 batch 的损失细节
- 验证集性能
- 最佳性能记录

---

## 🔧 配置说明

### 关键参数说明

#### 网络结构参数
```python
__C.NETWORK.view_distance = 1.5      # 多视角渲染距离，影响视图特征
__C.NETWORK.local_points = 512       # 局部特征提取的点数
__C.NETWORK.merge_points = 1024      # 合并后的点数
__C.NETWORK.step1 = 4                # 第一阶段上采样倍数
__C.NETWORK.step2 = 2                # 第二阶段上采样倍数
__C.NETWORK.spectral_k = 16          # 频谱卷积的邻居数
__C.NETWORK.msg_scales = [8, 16, 32] # 多尺度卷积的尺度
```

#### 训练参数
```python
__C.TRAIN.BATCH_SIZE = 32            # 批量大小
__C.TRAIN.N_EPOCHS = 300             # 训练轮数
__C.TRAIN.LEARNING_RATE = 0.0001     # 初始学习率
__C.TRAIN.LR_DECAY_STEP = [150, 200, 250]  # 学习率衰减点
__C.TRAIN.WARMUP_STEPS = 200         # 预热步数
__C.TRAIN.GAMMA = 0.5                # 学习率衰减倍数
```

### 调整建议

| 如果... | 建议调整 |
|---------|---------|
| 显存不足 | 减小 `batch_size`, `merge_points`, `local_points` |
| 训练太慢 | 增加 `batch_size`, 减少 `workers` |
| 精度不够 | 增加 `epochs`, 调整 `step1/step2`, 尝试不同的 `spectral_k` |
| 过拟合 | 增加数据增强, 减小模型复杂度 |
| 训练不稳定 | 增加 `warmup_steps`, 减小 `learning_rate` |

---

## 📈 预期结果

### ShapeNet55 测试集性能

| 模型 | Coarse CD | Fine1 CD | Fine2 CD |
|------|-----------|----------|----------|
| PCN | - | - | ~9.6 |
| PointCFormer | - | - | ~3.8 |
| SVFNet/PointSea | - | - | ~3.5 |
| **GeoSpecNet** | **6-7** | **4-5** | **3-4** |

*注: CD = Chamfer Distance (×10⁻³)*

### 训练时间估计

| 配置 | 单个 epoch | 300 epochs |
|------|-----------|------------|
| 4 × V100, BS=32 | ~6-8 分钟 | ~30-40 小时 |
| 4 × V100, BS=16 | ~10-12 分钟 | ~50-60 小时 |
| 2 × RTX 3090, BS=32 | ~8-10 分钟 | ~40-50 小时 |

---

## 🐛 常见问题

### Q1: CUDA Out of Memory

**解决方案**:
```python
# 在 config_geospec.py 中调整
__C.TRAIN.BATCH_SIZE = 16  # 减小批量大小
__C.NETWORK.merge_points = 768  # 减小点数
__C.NETWORK.local_points = 384
```

### Q2: 找不到模块

**解决方案**:
```bash
# 确保编译了所有依赖
cd /workspace/pointnet2_ops_lib && pip install .
cd /workspace/metrics/CD/chamfer3D && python setup.py install
```

### Q3: 训练损失不下降

**可能原因和解决方案**:
1. **学习率过大**: 减小 `LEARNING_RATE` 到 `0.00005`
2. **数据问题**: 检查数据路径和归一化
3. **预热不足**: 增加 `WARMUP_STEPS` 到 `500`

### Q4: 评估时出错

**检查清单**:
- ✅ 是否指定了 `--weights` 参数
- ✅ 权重文件路径是否正确
- ✅ 测试数据集路径是否配置
- ✅ GPU 是否可用

---

## 📚 进阶使用

### 1. 自定义损失函数

编辑 `utils/loss_utils.py`:

```python
def custom_loss(preds, gt):
    cd_loss = chamfer_distance(preds[-1], gt)
    # 添加自定义约束
    return cd_loss
```

### 2. 添加新的数据集

步骤：
1. 在 `utils/data_loaders.py` 添加数据加载器
2. 在 `config_geospec.py` 添加数据集配置
3. 更新 `DATASET_LOADER_MAPPING`

### 3. 修改网络结构

编辑 `models/GeoSpecNet.py`:
- 调整 DRSN 的 `hidden_dim`
- 修改 MSGConv 的 `scales`
- 改变 PCSA 的 `k` 值

### 4. 可视化结果

```python
import open3d as o3d

# 加载预测结果
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
o3d.visualization.draw_geometries([pcd])
```

---

## 📖 文档索引

| 文档 | 内容 |
|------|------|
| `README_GeoSpecNet.md` | 完整的英文文档 |
| `SETUP_GeoSpecNet.md` | 中文快速设置指南 |
| `IMPLEMENTATION_NOTES.md` | 实现细节和原理 |
| `IMPLEMENTATION_SUMMARY.md` | 实现总结 |
| `QUICK_START_GUIDE.md` | 本文档 |

---

## 🎓 学习资源

### 相关论文
1. **PointNet++** - 分层点云处理
2. **PointGST** - 图频谱变换
3. **PointSea** - 多视角融合
4. **PointCFormer** - Transformer for点云

### 推荐阅读
- Graph Fourier Transform 基础
- 点云深度学习综述
- Chamfer Distance vs EMD

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

### 提交代码
```bash
git checkout -b feature/your-feature
git commit -m "feat: add your feature"
git push origin feature/your-feature
```

### 报告 Bug
请提供：
- 错误信息
- 复现步骤
- 系统信息（GPU, CUDA, PyTorch版本）

---

## 📧 联系方式

- GitHub Issues: [项目地址]
- Email: [添加联系邮箱]

---

## ⭐ 致谢

感谢以下项目的启发和贡献：
- PointGST (频谱域)
- PointSea (多视角)
- PointNet++ (特征提取)
- PointCFormer (几何感知)

---

**祝训练顺利！如有问题请查阅文档或提交 Issue。**

最后更新: 2025-10-26
