from easydict import EasyDict as edict

__C                                              = edict()
cfg                                              = __C

#
# Dataset Config
#
__C.DATASETS                                     = edict()
__C.DATASETS.SHAPENET55                          = edict()
__C.DATASETS.SHAPENET55.CATEGORY_FILE_PATH       = 'datasets/ShapeNet55'
__C.DATASETS.SHAPENET55.N_POINTS                 = 2048
__C.DATASETS.SHAPENET55.COMPLETE_POINTS_PATH     = './shapenet_pc/%s'

__C.DATASETS.KITTI                               = edict()
__C.DATASETS.KITTI.CATEGORY_FILE_PATH            = 'datasets/KITTI.json'
__C.DATASETS.KITTI.PARTIAL_POINTS_PATH           = './kitti/%s/partial/%s/%s.dat'
__C.DATASETS.KITTI.COMPLETE_POINTS_PATH          = './kitti/%s/complete/%s/%s.dat'

#
# Dataset
#
__C.DATASET                                      = edict()
__C.DATASET.TRAIN_DATASET                        = 'ShapeNet55'
__C.DATASET.TEST_DATASET                         = 'ShapeNet55'

#
# Constants
#
__C.CONST                                        = edict()
__C.CONST.NUM_WORKERS                            = 8
__C.CONST.N_INPUT_POINTS                         = 2048
__C.CONST.mode = 'easy'

#
# Directories
#
__C.DIR                                          = edict()
__C.DIR.OUT_PATH                                 = 'GeoSpecNet_output'
__C.CONST.DEVICE                                 = '0,1,2,3'
# __C.CONST.WEIGHTS                                = ''  # Uncomment and set for testing

# Memcached
#
__C.MEMCACHED                                    = edict()
__C.MEMCACHED.ENABLED                            = False
__C.MEMCACHED.LIBRARY_PATH                       = '/mnt/lustre/share/pymc/py3'
__C.MEMCACHED.SERVER_CONFIG                      = '/mnt/lustre/share/memcached_client/server_list.conf'
__C.MEMCACHED.CLIENT_CONFIG                      = '/mnt/lustre/share/memcached_client/client.conf'

#
# Network - GeoSpecNet specific configurations
#
__C.NETWORK                                      = edict()
# Coarse completion settings
__C.NETWORK.num_coarse                           = 1024
# Multi-stage refinement ratios
__C.NETWORK.stage1_ratio                         = 2  # 1024 -> 2048
__C.NETWORK.stage2_ratio                         = 4  # 1024 -> 4096
# Feature dimensions
__C.NETWORK.hidden_dim                           = 512
__C.NETWORK.spectral_dim                         = 256
# Graph settings
__C.NETWORK.k_neighbors                          = 16
# Multi-scale graph convolution scales
__C.NETWORK.msg_conv_scales                      = [8, 16, 32]
# DRSN settings
__C.NETWORK.drsn_nhead                           = 8
# GAN training
__C.NETWORK.use_gan                              = True
__C.NETWORK.gan_loss_weight                      = 0.1

#
# Train
#
__C.TRAIN                                        = edict()
__C.TRAIN.BATCH_SIZE                             = 32
__C.TRAIN.N_EPOCHS                               = 400
__C.TRAIN.SAVE_FREQ                              = 10
__C.TRAIN.LEARNING_RATE                          = 0.0002
__C.TRAIN.LR_MILESTONES                          = [100, 200, 300]
__C.TRAIN.LR_DECAY_STEP                          = 50
__C.TRAIN.WARMUP_STEPS                           = 500
__C.TRAIN.GAMMA                                  = 0.5
__C.TRAIN.BETAS                                  = (0.9, 0.999)
__C.TRAIN.WEIGHT_DECAY                           = 1e-6

# Loss weights
__C.TRAIN.LOSS_WEIGHTS                           = edict()
__C.TRAIN.LOSS_WEIGHTS.CD_COARSE                 = 1.0
__C.TRAIN.LOSS_WEIGHTS.CD_FINE1                  = 2.0
__C.TRAIN.LOSS_WEIGHTS.CD_FINE2                  = 4.0
__C.TRAIN.LOSS_WEIGHTS.PARTIAL_MATCH             = 0.5
__C.TRAIN.LOSS_WEIGHTS.GAN_G                     = 0.1
__C.TRAIN.LOSS_WEIGHTS.GAN_D                     = 0.05

# GAN training settings
__C.TRAIN.GAN_START_EPOCH                        = 50  # Start GAN training after this epoch
__C.TRAIN.DISCRIMINATOR_STEPS                    = 1   # Train discriminator every N steps

#
# Test
#
__C.TEST                                         = edict()
__C.TEST.METRIC_NAME                             = 'ChamferDistance'
__C.TEST.OUTPUT_DIR                              = './test_results'
__C.TEST.SAVE_COMPLETIONS                        = True
