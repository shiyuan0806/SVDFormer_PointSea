#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Haozhe Xie, Modified for GeoSpecNet
# @Email   : cshzxie@gmail.com

from easydict import EasyDict as edict

__C = edict()
cfg = __C

#
# Common
#
__C.CONST = edict()
__C.CONST.DEVICE = '0,1,2,3'
__C.CONST.NUM_WORKERS = 8
# __C.CONST.WEIGHTS = ''  # Uncomment to resume training

#
# Datasets
#
__C.DATASET = edict()
__C.DATASET.TRAIN_DATASET = 'ShapeNet'
__C.DATASET.TEST_DATASET = 'ShapeNet'

#
# Dataset - ShapeNet55
#
__C.DATASET.SHAPENET = edict()
__C.DATASET.SHAPENET.PARTIAL_POINTS_PATH = '/path/to/ShapeNet55/ShapeNet-55/%s/partial/%s/%s/%02d.pcd'
__C.DATASET.SHAPENET.COMPLETE_POINTS_PATH = '/path/to/ShapeNet55/ShapeNet-55/%s/complete/%s/%s.pcd'
__C.DATASET.SHAPENET.CATEGORY_FILE_PATH = './datasets/ShapeNet55.json'
__C.DATASET.SHAPENET.N_RENDERINGS = 8
__C.DATASET.SHAPENET.N_POINTS = 2048
__C.DATASET.SHAPENET.PARTIAL_N_POINTS = 2048

#
# Dataset - KITTI
#
__C.DATASET.KITTI = edict()
__C.DATASET.KITTI.PARTIAL_POINTS_PATH = '/path/to/KITTI/cars/%s.pcd'
__C.DATASET.KITTI.BOUNDING_BOX_FILE_PATH = '/path/to/KITTI/bboxes/%s.txt'
__C.DATASET.KITTI.CATEGORY_FILE_PATH = './datasets/KITTI.json'
__C.DATASET.KITTI.N_POINTS = 2048
__C.DATASET.KITTI.PARTIAL_N_POINTS = 2048

#
# Dataset - Completion3D
#
__C.DATASET.COMPLETION3D = edict()
__C.DATASET.COMPLETION3D.PARTIAL_POINTS_PATH = '/path/to/Completion3D/%s/partial/%s/%s.h5'
__C.DATASET.COMPLETION3D.COMPLETE_POINTS_PATH = '/path/to/Completion3D/%s/gt/%s/%s.h5'
__C.DATASET.COMPLETION3D.CATEGORY_FILE_PATH = './datasets/Completion3D.json'
__C.DATASET.COMPLETION3D.N_POINTS = 2048
__C.DATASET.COMPLETION3D.PARTIAL_N_POINTS = 2048

#
# Network
#
__C.NETWORK = edict()
__C.NETWORK.view_distance = 1.5  # Distance for multi-view rendering
__C.NETWORK.local_points = 512   # Number of points for local feature extraction
__C.NETWORK.merge_points = 1024  # Number of points after merging partial and coarse
__C.NETWORK.step1 = 4            # Upsampling ratio for first refinement stage
__C.NETWORK.step2 = 2            # Upsampling ratio for second refinement stage

# GeoSpecNet specific parameters
__C.NETWORK.spectral_k = 16      # K for spectral graph convolution
__C.NETWORK.msg_scales = [8, 16, 32]  # Multi-scale graph convolution scales
__C.NETWORK.hidden_dim_stage1 = 768   # Hidden dimension for stage 1 DRSN
__C.NETWORK.hidden_dim_stage2 = 512   # Hidden dimension for stage 2 DRSN

#
# GAN (Self-supervised Structural Consistency Training)
#
__C.GAN = edict()
__C.GAN.ENABLED = False          # Enable GAN training
__C.GAN.WARMUP_GAN_EPOCHS = 50   # Train generator only for first N epochs

# Discriminator settings
__C.GAN.DISC_TYPE = 'local_global'  # Discriminator type: 'simple', 'local_global', 'spectral'
__C.GAN.DISC_HIDDEN_DIM = 256       # Hidden dimension for discriminator
__C.GAN.DISC_LEARNING_RATE = 0.0001 # Learning rate for discriminator
__C.GAN.DISC_UPDATE_FREQ = 2        # Update discriminator every N batches (slower than generator)

# GAN loss settings
__C.GAN.GAN_MODE = 'lsgan'       # GAN loss type: 'vanilla', 'lsgan', 'wgan'
__C.GAN.GAN_WEIGHT = 0.1         # Weight for GAN loss

# Structural consistency settings
__C.GAN.USE_PARTIAL_MATCHING = True   # Use partial matching loss
__C.GAN.USE_CONSISTENCY = True        # Use consistency loss
__C.GAN.PARTIAL_MATCHING_WEIGHT = 1.0 # Weight for partial matching loss
__C.GAN.CONSISTENCY_WEIGHT = 0.5      # Weight for consistency loss
__C.GAN.PARTIAL_MATCHING_THRESHOLD = 0.05  # Distance threshold for partial matching

#
# Training
#
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 32
__C.TRAIN.N_EPOCHS = 300
__C.TRAIN.SAVE_FREQ = 10
__C.TRAIN.LEARNING_RATE = 0.0001
__C.TRAIN.LR_DECAY_STEP = [150, 200, 250]
__C.TRAIN.WARMUP_STEPS = 200
__C.TRAIN.GAMMA = 0.5
__C.TRAIN.BETAS = (0.9, 0.999)
__C.TRAIN.WEIGHT_DECAY = 0

#
# Testing
#
__C.TEST = edict()
__C.TEST.METRIC_NAME = 'ChamferDistance'

#
# Directories
#
__C.DIR = edict()
__C.DIR.OUT_PATH = './output/GeoSpecNet'
