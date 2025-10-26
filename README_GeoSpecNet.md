# GeoSpecNet: Point Cloud Completion with Spectral Domain Enhancement

GeoSpecNet is a state-of-the-art point cloud completion model that combines spectral domain processing with geometric feature learning. The model integrates ideas from multiple sources:

1. **Point Cloud Spectral Adapter (PCSA)** from PointGST - for spectral domain transformation
2. **Geo-Spectral Collaborative Perception Module** - combining Graph Fourier Transform with geometric attention
3. **Dynamic Region Selection Network (DRSN)** - with structure-aware gating for adaptive refinement
4. **Multi-view Fusion** from PointSea - using self-projected depth maps from multiple viewpoints

## Architecture Overview

### Key Components

1. **Spectral Graph Convolution (SpectralGraphConv)**
   - Transforms spatial features to spectral domain using Graph Fourier Transform
   - Captures multi-frequency geometric patterns
   - Reduces redundancy through orthogonal basis decomposition

2. **Point Cloud Spectral Adapter (PCSA)**
   - Adapts spatial features to spectral domain
   - Channel attention for adaptive fusion
   - Combines spatial and spectral features

3. **Multi-Scale Graph Convolution (MSGConv)**
   - Extracts multi-scale features at different scales (k=8, 16, 32)
   - Captures geometric patterns at different frequencies
   - Aggregates multi-scale information

4. **Geo-Spectral Collaborative Perception Module**
   - Dual-branch architecture (spectral + geometric)
   - Cross-modal fusion with attention mechanisms
   - Synchronized modeling in both domains

5. **Dynamic Region Selection Network (DRSN)**
   - Structure-aware gating for complexity estimation
   - Dual-path refinement:
     - Global semantic path for coarse geometry
     - Local detail path for fine structures
   - Dynamic path selection based on geometric complexity

6. **Multi-View Fusion Encoder**
   - Renders depth images from multiple viewpoints
   - Extracts features using ResNet18
   - Fuses multi-view features with point cloud features
   - Generates coarse completion

## Installation

### Prerequisites
- Python 3.6+
- PyTorch 1.7+
- CUDA 10.2+ (for GPU support)

### Dependencies
```bash
pip install torch torchvision
pip install einops
pip install tensorboardX
pip install tqdm
pip install easydict
pip install torch-scatter
```

### Build PointNet2 Operations
```bash
cd pointnet2_ops_lib
pip install .
```

### Build Chamfer Distance
```bash
cd metrics/CD/chamfer3D
python setup.py install
```

### Build EMD (Optional)
```bash
cd metrics/EMD
bash run_build.sh
```

## Dataset Preparation

### ShapeNet55
Download and organize the ShapeNet dataset as follows:
```
ShapeNet55/
├── train/
│   ├── partial/
│   └── complete/
└── test/
    ├── partial/
    └── complete/
```

Update paths in `config_geospec.py`:
```python
__C.DATASET.SHAPENET.PARTIAL_POINTS_PATH = '/path/to/ShapeNet55/%s/partial/%s/%s/%02d.pcd'
__C.DATASET.SHAPENET.COMPLETE_POINTS_PATH = '/path/to/ShapeNet55/%s/complete/%s/%s.pcd'
```

### KITTI
Download KITTI dataset and update paths in config.

### Completion3D
Download Completion3D dataset and update paths in config.

## Usage

### Training

**Basic training:**
```bash
python main_geospec.py --mode train \
    --gpu 0,1,2,3 \
    --batch_size 32 \
    --epochs 300 \
    --shapenet_path /path/to/ShapeNet55
```

**Resume training from checkpoint:**
```bash
python main_geospec.py --mode train \
    --resume ./output/GeoSpecNet/checkpoints/ckpt-best.pth \
    --gpu 0,1,2,3
```

**Training with custom configuration:**
```bash
python main_geospec.py --mode train \
    --train_dataset ShapeNet \
    --test_dataset ShapeNet \
    --batch_size 32 \
    --epochs 300 \
    --lr 0.0001 \
    --workers 8 \
    --output ./output/my_experiment
```

### Evaluation

**Evaluate on test set:**
```bash
python main_geospec.py --mode eval \
    --weights ./output/GeoSpecNet/checkpoints/ckpt-best.pth \
    --test_dataset ShapeNet \
    --gpu 0
```

**Test mode (more detailed metrics):**
```bash
python main_geospec.py --mode test \
    --weights ./output/GeoSpecNet/checkpoints/ckpt-best.pth \
    --test_dataset ShapeNet \
    --gpu 0
```

## Model Configuration

Key hyperparameters in `config_geospec.py`:

```python
# Network architecture
__C.NETWORK.view_distance = 1.5      # Multi-view rendering distance
__C.NETWORK.local_points = 512       # Local feature extraction points
__C.NETWORK.merge_points = 1024      # Points after merging
__C.NETWORK.step1 = 4                # First upsampling ratio
__C.NETWORK.step2 = 2                # Second upsampling ratio
__C.NETWORK.spectral_k = 16          # K for spectral convolution
__C.NETWORK.msg_scales = [8, 16, 32] # Multi-scale graph convolution

# Training
__C.TRAIN.BATCH_SIZE = 32
__C.TRAIN.N_EPOCHS = 300
__C.TRAIN.LEARNING_RATE = 0.0001
__C.TRAIN.LR_DECAY_STEP = [150, 200, 250]
__C.TRAIN.WARMUP_STEPS = 200
```

## Model Architecture Details

### Forward Pass
1. **Input**: Partial point cloud (B, N, 3)
2. **Multi-view Rendering**: Generate depth images from 3 viewpoints
3. **Feature Extraction**:
   - Point cloud features via PointNet++
   - Multi-view features via ResNet18
   - Local geometric features via EdgeConv
4. **Coarse Generation**: Fuse features and generate coarse completion
5. **Refinement Stage 1 (DRSN)**:
   - Geo-spectral enhancement
   - Dual-path processing (global + local)
   - Dynamic path selection
   - Upsample 4x
6. **Refinement Stage 2 (DRSN)**:
   - Further refinement with spectral features
   - Upsample 2x
7. **Output**: Three predictions (coarse, fine1, fine2)

### Loss Function
Multi-stage Chamfer Distance loss:
```python
L = λ₁ * CD(coarse, GT) + λ₂ * CD(fine1, GT) + λ₃ * CD(fine2, GT)
```
where λ₁ = 0.5, λ₂ = 0.3, λ₃ = 1.0

## Performance

Expected performance on ShapeNet55 test set:
- Coarse CD: ~6-7 (×10⁻³)
- Fine1 CD: ~4-5 (×10⁻³)
- Fine2 CD: ~3-4 (×10⁻³)

## Citation

If you use this code in your research, please cite the relevant papers:

```bibtex
@article{geospecnet2025,
  title={GeoSpecNet: Point Cloud Completion with Spectral Domain Enhancement},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

## Acknowledgments

This implementation builds upon ideas from:
- **PointGST**: For PCSA module and spectral domain transformation
- **PointSea/SVFNet**: For multi-view fusion approach
- **PointNet++**: For hierarchical feature extraction
- **PointCFormer**: For local geometric relation perception

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.
