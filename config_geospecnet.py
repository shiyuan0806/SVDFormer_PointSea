from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Dataset (reuse ShapeNet55 complete-only online partial generation)
__C.DATASETS = edict()
__C.DATASETS.SHAPENET55 = edict()
__C.DATASETS.SHAPENET55.CATEGORY_FILE_PATH = 'datasets/ShapeNet55'
__C.DATASETS.SHAPENET55.N_POINTS = 2048
__C.DATASETS.SHAPENET55.COMPLETE_POINTS_PATH = './shapenet_pc/%s'

__C.DATASET = edict()
__C.DATASET.TRAIN_DATASET = 'ShapeNet55'
__C.DATASET.TEST_DATASET = 'ShapeNet55'

# Constants
__C.CONST = edict()
__C.CONST.NUM_WORKERS = 4
__C.CONST.N_INPUT_POINTS = 2048
__C.CONST.mode = 'easy'
__C.CONST.DEVICE = '0,1'
# __C.CONST.WEIGHTS = ''

# Dirs
__C.DIR = edict()
__C.DIR.OUT_PATH = 'GeoSpecNet_55'

# Network
__C.NETWORK = edict()
__C.NETWORK.step1 = 2
__C.NETWORK.step2 = 4
__C.NETWORK.merge_points = 1024
__C.NETWORK.local_points = 1024
__C.NETWORK.view_distance = 1.5
# GeoSpectral specific
__C.NETWORK.spectral_points = 512
__C.NETWORK.spectral_k = 48
__C.NETWORK.coarse_points = 512

# Train
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 16
__C.TRAIN.N_EPOCHS = 300
__C.TRAIN.SAVE_FREQ = 5
__C.TRAIN.LEARNING_RATE = 1e-4
__C.TRAIN.LR_DECAY_STEP = 2
__C.TRAIN.WARMUP_STEPS = 300
__C.TRAIN.GAMMA = 0.98
__C.TRAIN.BETAS = (0.9, 0.999)
__C.TRAIN.WEIGHT_DECAY = 0.0
__C.TRAIN.GAN_WEIGHT = 0.1
__C.TRAIN.PM_WEIGHT = 1.0

# Test
__C.TEST = edict()
__C.TEST.METRIC_NAME = 'ChamferDistance'
