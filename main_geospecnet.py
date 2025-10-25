"""
Main entry point for GeoSpecNet
Geometric-Spectral Collaborative Perception Network for Point Cloud Completion
"""

import argparse
import logging
import os
import numpy as np
import sys
import torch
from pprint import pprint
from config_geospecnet import cfg
from core.train_geospecnet import train_net
from core.test_geospecnet import test_net

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE


def set_seed(seed):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args_from_command_line():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='GeoSpecNet: Geometric-Spectral Collaborative Perception for Point Cloud Completion',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on ShapeNet-55
  python main_geospecnet.py --train
  
  # Test with checkpoint
  python main_geospecnet.py --test --weights path/to/checkpoint.pth
  
  # Train with custom config
  python main_geospecnet.py --train --config custom_config.py
  
  # Inference mode
  python main_geospecnet.py --inference --weights path/to/checkpoint.pth
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--train', dest='train', help='Train the network', action='store_true')
    mode_group.add_argument('--test', dest='test', help='Test the network', action='store_true')
    mode_group.add_argument('--inference', dest='inference', help='Inference mode for deployment', action='store_true')
    
    # Optional arguments
    parser.add_argument('--weights', dest='weights', help='Path to checkpoint file', type=str, default=None)
    parser.add_argument('--config', dest='config', help='Path to config file', type=str, default=None)
    parser.add_argument('--gpu', dest='gpu', help='GPU devices to use (e.g., 0,1,2,3)', type=str, default=None)
    parser.add_argument('--batch-size', dest='batch_size', help='Batch size', type=int, default=None)
    parser.add_argument('--epochs', dest='epochs', help='Number of epochs', type=int, default=None)
    parser.add_argument('--lr', dest='learning_rate', help='Learning rate', type=float, default=None)
    parser.add_argument('--output', dest='output', help='Output directory', type=str, default=None)
    parser.add_argument('--dataset', dest='dataset', help='Dataset name (ShapeNet55, KITTI, etc.)', type=str, default=None)
    parser.add_argument('--seed', dest='seed', help='Random seed', type=int, default=42)
    
    args = parser.parse_args()
    return args


def update_config_from_args(cfg, args):
    """Update configuration from command line arguments"""
    
    if args.weights:
        cfg.CONST.WEIGHTS = args.weights
    
    if args.gpu:
        cfg.CONST.DEVICE = args.gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    if args.batch_size:
        cfg.TRAIN.BATCH_SIZE = args.batch_size
    
    if args.epochs:
        cfg.TRAIN.N_EPOCHS = args.epochs
    
    if args.learning_rate:
        cfg.TRAIN.LEARNING_RATE = args.learning_rate
    
    if args.output:
        cfg.DIR.OUT_PATH = args.output
    
    if args.dataset:
        cfg.DATASET.TRAIN_DATASET = args.dataset
        cfg.DATASET.TEST_DATASET = args.dataset
    
    return cfg


def print_model_info():
    """Print model information"""
    print("\n" + "="*80)
    print(" " * 20 + "GeoSpecNet: Point Cloud Completion")
    print("="*80)
    print("\n📝 Model Architecture:")
    print("  ├─ Encoder:")
    print("  │   ├─ PointNet++ Feature Extraction")
    print("  │   └─ Geo-Spectral Collaborative Module")
    print("  │       ├─ Graph Fourier Transform (GFT)")
    print("  │       ├─ Multi-Scale Graph Convolution (MSGConv)")
    print("  │       └─ Cross-Domain Feature Alignment")
    print("  │")
    print("  ├─ Decoder:")
    print("  │   ├─ Coarse Point Generation")
    print("  │   └─ Dynamic Region Selection Network (DRSN)")
    print("  │       ├─ Structure-Aware Gating Unit")
    print("  │       ├─ Global Semantic Path")
    print("  │       └─ Local Detail Path")
    print("  │")
    print("  └─ Training:")
    print("      ├─ Multi-stage Chamfer Distance Loss")
    print("      ├─ Partial Matching Loss")
    print("      └─ GAN-based Structural Consistency")
    print("\n🔬 Key Innovations:")
    print("  • Spectral domain modeling via Graph Fourier Transform")
    print("  • Dual-path repair strategy (global + local)")
    print("  • Structure-aware dynamic region selection")
    print("  • Multi-scale geometric pattern extraction")
    print("="*80 + "\n")


def main():
    """Main function"""
    
    # Get args from command line
    args = get_args_from_command_line()
    
    # Set random seed
    set_seed(args.seed)
    logging.info(f'Random seed set to {args.seed}')
    
    # Load custom config if specified
    if args.config:
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", args.config)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        global cfg
        cfg = config_module.cfg
    
    # Update config from args
    cfg = update_config_from_args(cfg, args)
    
    # Print model info
    print_model_info()
    
    # Print config
    print('📊 Configuration:')
    print('-' * 80)
    pprint(cfg)
    print('-' * 80 + '\n')
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f'✅ CUDA is available. Using GPU: {cfg.CONST.DEVICE}')
        print(f'   Number of GPUs: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'   GPU {i}: {torch.cuda.get_device_name(i)}')
    else:
        print('⚠️  CUDA is not available. Using CPU.')
    print()
    
    # Run training, testing, or inference
    if args.train:
        print('🚀 Starting training...\n')
        train_net(cfg)
        
    elif args.test:
        if not hasattr(cfg.CONST, 'WEIGHTS') or not cfg.CONST.WEIGHTS:
            raise ValueError('Please specify checkpoint path using --weights argument!')
        print('🔍 Starting testing...\n')
        test_net(cfg)
        
    elif args.inference:
        if not hasattr(cfg.CONST, 'WEIGHTS') or not cfg.CONST.WEIGHTS:
            raise ValueError('Please specify checkpoint path using --weights argument!')
        print('🎯 Starting inference...\n')
        run_inference(cfg)


def run_inference(cfg):
    """Run inference mode for deployment"""
    
    from models.GeoSpecNet import GeoSpecNet
    import torch
    
    # Build model
    model = GeoSpecNet(cfg)
    
    # Load weights
    logging.info(f'Loading weights from {cfg.CONST.WEIGHTS}...')
    checkpoint = torch.load(cfg.CONST.WEIGHTS)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    logging.info('Model loaded successfully. Ready for inference.')
    
    # Example: load a partial point cloud and complete it
    print("\n" + "="*80)
    print("Inference Mode - Model Ready")
    print("="*80)
    print("\nExample usage:")
    print("```python")
    print("import numpy as np")
    print("import torch")
    print()
    print("# Load your partial point cloud (N, 3)")
    print("partial_cloud = np.load('your_partial_cloud.npy')")
    print("partial_cloud = torch.from_numpy(partial_cloud).float().unsqueeze(0)")
    print()
    print("# Complete the point cloud")
    print("with torch.no_grad():")
    print("    _, _, completion = model(partial_cloud)")
    print()
    print("# Save result")
    print("np.save('completion.npy', completion.squeeze(0).cpu().numpy())")
    print("```")
    print("="*80 + "\n")
    
    return model


if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(
        format='[%(levelname)s] %(asctime)s - %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Check Python version
    if sys.version_info < (3, 6):
        logging.error('Python 3.6 or higher is required!')
        sys.exit(1)
    
    # Run main
    try:
        main()
    except KeyboardInterrupt:
        print('\n\n⚠️  Training interrupted by user')
        sys.exit(0)
    except Exception as e:
        logging.error(f'Error: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
