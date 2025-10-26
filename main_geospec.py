#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Main entry point for GeoSpecNet training and evaluation
"""

import logging
import os
import sys
import argparse
import random
import numpy as np
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_geospec import cfg
from core.train_geospec import train_net
from core.train_geospec_gan import train_net_with_gan
from core.eval_geospec import eval_net


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        format='[%(levelname)s] %(asctime)s %(message)s',
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True  # Slower but more reproducible
    # torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='GeoSpecNet: Point Cloud Completion with Spectral Domain Enhancement')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'test'],
                        help='Mode: train, eval, or test')
    
    # Dataset paths
    parser.add_argument('--shapenet_path', type=str, default=None,
                        help='Path to ShapeNet dataset')
    parser.add_argument('--kitti_path', type=str, default=None,
                        help='Path to KITTI dataset')
    parser.add_argument('--completion3d_path', type=str, default=None,
                        help='Path to Completion3D dataset')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of data loading workers')
    
    # Model parameters
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to pretrained weights')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training')
    
    # GPU settings
    parser.add_argument('--gpu', type=str, default=None,
                        help='GPU device IDs (e.g., "0,1,2,3")')
    
    # Output directory
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for checkpoints and logs')
    
    # GAN training
    parser.add_argument('--use_gan', action='store_true',
                        help='Use GAN for self-supervised structural consistency training')
    parser.add_argument('--disc_type', type=str, default=None,
                        choices=['simple', 'local_global', 'spectral'],
                        help='Discriminator type for GAN training')
    parser.add_argument('--gan_weight', type=float, default=None,
                        help='Weight for GAN loss')
    
    # Random seed
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Dataset selection
    parser.add_argument('--train_dataset', type=str, default=None,
                        choices=['ShapeNet', 'Completion3D', 'KITTI'],
                        help='Training dataset')
    parser.add_argument('--test_dataset', type=str, default=None,
                        choices=['ShapeNet', 'Completion3D', 'KITTI'],
                        help='Testing dataset')
    
    return parser.parse_args()


def update_config(args):
    """Update configuration based on command line arguments"""
    # Dataset paths
    if args.shapenet_path:
        cfg.DATASET.SHAPENET.PARTIAL_POINTS_PATH = os.path.join(
            args.shapenet_path, '%s/partial/%s/%s/%02d.pcd'
        )
        cfg.DATASET.SHAPENET.COMPLETE_POINTS_PATH = os.path.join(
            args.shapenet_path, '%s/complete/%s/%s.pcd'
        )
    
    if args.kitti_path:
        cfg.DATASET.KITTI.PARTIAL_POINTS_PATH = os.path.join(
            args.kitti_path, 'cars/%s.pcd'
        )
        cfg.DATASET.KITTI.BOUNDING_BOX_FILE_PATH = os.path.join(
            args.kitti_path, 'bboxes/%s.txt'
        )
    
    if args.completion3d_path:
        cfg.DATASET.COMPLETION3D.PARTIAL_POINTS_PATH = os.path.join(
            args.completion3d_path, '%s/partial/%s/%s.h5'
        )
        cfg.DATASET.COMPLETION3D.COMPLETE_POINTS_PATH = os.path.join(
            args.completion3d_path, '%s/gt/%s/%s.h5'
        )
    
    # Training parameters
    if args.batch_size:
        cfg.TRAIN.BATCH_SIZE = args.batch_size
    if args.epochs:
        cfg.TRAIN.N_EPOCHS = args.epochs
    if args.lr:
        cfg.TRAIN.LEARNING_RATE = args.lr
    if args.workers:
        cfg.CONST.NUM_WORKERS = args.workers
    
    # Model parameters
    if args.weights:
        cfg.CONST.WEIGHTS = args.weights
    if args.resume:
        cfg.CONST.WEIGHTS = args.resume
    
    # GPU settings
    if args.gpu:
        cfg.CONST.DEVICE = args.gpu
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Output directory
    if args.output:
        cfg.DIR.OUT_PATH = args.output
    
    # Dataset selection
    if args.train_dataset:
        cfg.DATASET.TRAIN_DATASET = args.train_dataset
    if args.test_dataset:
        cfg.DATASET.TEST_DATASET = args.test_dataset
    
    # GAN settings
    if args.use_gan:
        cfg.GAN.ENABLED = True
    if args.disc_type:
        cfg.GAN.DISC_TYPE = args.disc_type
    if args.gan_weight:
        cfg.GAN.GAN_WEIGHT = args.gan_weight


def print_config():
    """Print current configuration"""
    logging.info('=' * 80)
    logging.info('GeoSpecNet Configuration')
    logging.info('=' * 80)
    logging.info(f'Training Dataset: {cfg.DATASET.TRAIN_DATASET}')
    logging.info(f'Testing Dataset:  {cfg.DATASET.TEST_DATASET}')
    logging.info(f'Batch Size:       {cfg.TRAIN.BATCH_SIZE}')
    logging.info(f'Epochs:           {cfg.TRAIN.N_EPOCHS}')
    logging.info(f'Learning Rate:    {cfg.TRAIN.LEARNING_RATE}')
    logging.info(f'Workers:          {cfg.CONST.NUM_WORKERS}')
    logging.info(f'GPU Devices:      {cfg.CONST.DEVICE}')
    logging.info(f'Output Path:      {cfg.DIR.OUT_PATH}')
    logging.info('-' * 80)
    logging.info('Network Configuration:')
    logging.info(f'  View Distance:  {cfg.NETWORK.view_distance}')
    logging.info(f'  Local Points:   {cfg.NETWORK.local_points}')
    logging.info(f'  Merge Points:   {cfg.NETWORK.merge_points}')
    logging.info(f'  Upsampling Step1: {cfg.NETWORK.step1}x')
    logging.info(f'  Upsampling Step2: {cfg.NETWORK.step2}x')
    if hasattr(cfg.CONST, 'WEIGHTS') and cfg.CONST.WEIGHTS:
        logging.info(f'  Weights:        {cfg.CONST.WEIGHTS}')
    logging.info('=' * 80)


def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    setup_logging()
    
    # Set random seed
    set_seed(args.seed)
    logging.info(f'Random seed set to {args.seed}')
    
    # Update configuration
    update_config(args)
    
    # Set GPU devices
    if cfg.CONST.DEVICE:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.CONST.DEVICE
    
    # Print configuration
    print_config()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        logging.warning('CUDA is not available! Running on CPU will be very slow.')
    else:
        logging.info(f'CUDA is available. Using {torch.cuda.device_count()} GPU(s).')
    
    # Run the appropriate mode
    if args.mode == 'train':
        logging.info('Starting training...')
        if cfg.GAN.ENABLED:
            logging.info('Using GAN for self-supervised structural consistency training')
            train_net_with_gan(cfg)
        else:
            train_net(cfg)
        logging.info('Training completed!')
        
    elif args.mode == 'eval':
        logging.info('Starting evaluation...')
        if not hasattr(cfg.CONST, 'WEIGHTS') or not cfg.CONST.WEIGHTS:
            logging.error('Please specify weights for evaluation using --weights or --resume')
            sys.exit(1)
        eval_net(cfg)
        logging.info('Evaluation completed!')
        
    elif args.mode == 'test':
        logging.info('Starting testing...')
        if not hasattr(cfg.CONST, 'WEIGHTS') or not cfg.CONST.WEIGHTS:
            logging.error('Please specify weights for testing using --weights or --resume')
            sys.exit(1)
        from core.test_geospec import evaluate_net
        evaluate_net(cfg)
        logging.info('Testing completed!')
    
    else:
        logging.error(f'Unknown mode: {args.mode}')
        sys.exit(1)


if __name__ == '__main__':
    main()
