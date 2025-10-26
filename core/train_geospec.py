"""
Training script for GeoSpecNet
"""

import logging
import os
import torch
import utils.data_loaders
import utils.helpers
from datetime import datetime
from tqdm import tqdm
from time import time
from tensorboardX import SummaryWriter
from core.test_geospec import test_net
from utils.average_meter import AverageMeter
from torch.optim.lr_scheduler import MultiStepLR
from utils.schedular import GradualWarmupScheduler
from utils.loss_utils import get_loss
from models_PointSea.mv_utils_zs import PCViews_Real
from models.GeoSpecNet import GeoSpecNet


def train_net(cfg):
    """Main training function for GeoSpecNet"""
    torch.backends.cudnn.benchmark = True
    
    # Setup datasets
    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    test_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset_loader.get_dataset(utils.data_loaders.DatasetSubset.TRAIN),
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.CONST.NUM_WORKERS,
        collate_fn=utils.data_loaders.collate_fn,
        pin_memory=True,
        shuffle=True,
        drop_last=False
    )
    
    val_data_loader = torch.utils.data.DataLoader(
        dataset=test_dataset_loader.get_dataset(utils.data_loaders.DatasetSubset.TEST),
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.CONST.NUM_WORKERS // 2,
        collate_fn=utils.data_loaders.collate_fn,
        pin_memory=True,
        shuffle=False
    )
    
    # Set up output directories
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', datetime.now().isoformat())
    cfg.DIR.CHECKPOINTS = output_dir % 'checkpoints'
    cfg.DIR.LOGS = output_dir % 'logs'
    if not os.path.exists(cfg.DIR.CHECKPOINTS):
        os.makedirs(cfg.DIR.CHECKPOINTS)
    
    # Create tensorboard writers
    train_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
    val_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))
    
    # Initialize model
    model = GeoSpecNet(cfg)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    
    logging.info(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}')
    logging.info(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.TRAIN.LEARNING_RATE,
        weight_decay=cfg.TRAIN.WEIGHT_DECAY,
        betas=cfg.TRAIN.BETAS
    )
    
    # Learning rate scheduler
    scheduler_steplr = MultiStepLR(
        optimizer,
        milestones=cfg.TRAIN.LR_DECAY_STEP,
        gamma=cfg.TRAIN.GAMMA
    )
    lr_scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=1,
        total_epoch=cfg.TRAIN.WARMUP_STEPS,
        after_scheduler=scheduler_steplr
    )
    
    init_epoch = 0
    best_metrics = float('inf')
    steps = 0
    best_epoch = 0
    
    # Multi-view renderer
    render = PCViews_Real(TRANS=-cfg.NETWORK.view_distance)
    
    # Resume from checkpoint if specified
    if 'WEIGHTS' in cfg.CONST:
        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        steps = cfg.TRAIN.WARMUP_STEPS + 1
        lr_scheduler = MultiStepLR(
            optimizer,
            milestones=cfg.TRAIN.LR_DECAY_STEP,
            gamma=cfg.TRAIN.GAMMA
        )
        optimizer.param_groups[0]['lr'] = cfg.TRAIN.LEARNING_RATE
        logging.info('Recovery complete.')
    
    # Training loop
    for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):
        epoch_start_time = time()
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        
        model.train()
        
        total_cd_coarse = 0
        total_cd_fine1 = 0
        total_cd_fine2 = 0
        
        batch_end_time = time()
        n_batches = len(train_data_loader)
        
        print(f'Epoch: {epoch_idx}, Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        with tqdm(train_data_loader) as t:
            for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(t):
                data_time.update(time() - batch_end_time)
                
                # Move data to GPU
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)
                
                partial = data['partial_cloud']
                gt = data['gtcloud']
                
                # Render multi-view depth images
                partial_depth = render.get_img(partial)
                
                # Forward pass
                pcds_pred = model(partial, partial_depth)
                
                # Compute loss
                loss_total, losses = get_loss(pcds_pred, gt, sqrt=True)
                
                # Backward pass
                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()
                
                # Record losses
                cd_coarse_item = losses[0].item() * 1e3
                total_cd_coarse += cd_coarse_item
                cd_fine1_item = losses[1].item() * 1e3
                total_cd_fine1 += cd_fine1_item
                cd_fine2_item = losses[2].item() * 1e3
                total_cd_fine2 += cd_fine2_item
                
                # Log to tensorboard
                n_itr = (epoch_idx - 1) * n_batches + batch_idx
                train_writer.add_scalar('Loss/Batch/cd_coarse', cd_coarse_item, n_itr)
                train_writer.add_scalar('Loss/Batch/cd_fine1', cd_fine1_item, n_itr)
                train_writer.add_scalar('Loss/Batch/cd_fine2', cd_fine2_item, n_itr)
                
                batch_time.update(time() - batch_end_time)
                batch_end_time = time()
                
                # Update progress bar
                t.set_description(
                    f'[Epoch {epoch_idx}/{cfg.TRAIN.N_EPOCHS}]'
                    f'[Batch {batch_idx + 1}/{n_batches}]'
                )
                t.set_postfix(
                    coarse=f'{cd_coarse_item:.4f}',
                    fine1=f'{cd_fine1_item:.4f}',
                    fine2=f'{cd_fine2_item:.4f}'
                )
                
                # Learning rate warmup
                if steps <= cfg.TRAIN.WARMUP_STEPS:
                    lr_scheduler.step()
                    steps += 1
        
        # Epoch statistics
        avg_cd_coarse = total_cd_coarse / n_batches
        avg_cd_fine1 = total_cd_fine1 / n_batches
        avg_cd_fine2 = total_cd_fine2 / n_batches
        
        lr_scheduler.step()
        epoch_end_time = time()
        
        # Log epoch losses
        train_writer.add_scalar('Loss/Epoch/cd_coarse', avg_cd_coarse, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/cd_fine1', avg_cd_fine1, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/cd_fine2', avg_cd_fine2, epoch_idx)
        
        logging.info(
            f'[Epoch {epoch_idx}/{cfg.TRAIN.N_EPOCHS}] '
            f'EpochTime = {epoch_end_time - epoch_start_time:.3f}s '
            f'Losses = [Coarse: {avg_cd_coarse:.4f}, Fine1: {avg_cd_fine1:.4f}, Fine2: {avg_cd_fine2:.4f}]'
        )
        
        # Validation
        cd_eval = test_net(cfg, epoch_idx, val_data_loader, val_writer, model)
        
        # Save checkpoints
        if epoch_idx % cfg.TRAIN.SAVE_FREQ == 0 or cd_eval < best_metrics:
            if cd_eval < best_metrics:
                best_metrics = cd_eval
                best_epoch = epoch_idx
                file_name = 'ckpt-best.pth'
            else:
                file_name = f'ckpt-epoch-{epoch_idx:03d}.pth'
            
            output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
            torch.save({
                'epoch': epoch_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_metrics': best_metrics
            }, output_path)
            
            logging.info(f'Saved checkpoint to {output_path}')
        
        logging.info(f'Best Performance: Epoch {best_epoch} -- CD {best_metrics:.4f}')
    
    train_writer.close()
    val_writer.close()
