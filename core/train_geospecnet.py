"""
Training script for GeoSpecNet
Implements multi-stage training with GAN-based structural consistency
"""

import os
import torch
import torch.nn as nn
import torch.utils.data
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import logging
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

from models.GeoSpecNet import GeoSpecNet
from utils.average_meter import AverageMeter
from utils.schedular import GradualWarmupScheduler
from utils.loss_utils import *
from metrics.CD.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from metrics.EMD.emd_module import emdModule


def train_net(cfg):
    """Main training function for GeoSpecNet"""
    
    # Enable cudnn benchmark for better performance
    torch.backends.cudnn.benchmark = True
    
    # Set up output directories
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', '%s')
    log_dir = output_dir % ('logs', 'train')
    ckpt_dir = output_dir % ('checkpoints', 'train')
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Set up TensorBoard writer
    train_writer = SummaryWriter(log_dir)
    
    # Set up data loader
    train_data_loader = setup_dataloader(cfg, 'train')
    val_data_loader = setup_dataloader(cfg, 'val')
    
    # Build model
    model = GeoSpecNet(cfg)
    
    # Move model to GPU
    if torch.cuda.is_available():
        model = nn.DataParallel(model).cuda()
    
    # Set up optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.TRAIN.LEARNING_RATE,
        betas=cfg.TRAIN.BETAS,
        weight_decay=cfg.TRAIN.WEIGHT_DECAY
    )
    
    # Set up optimizer for discriminator
    if cfg.NETWORK.use_gan:
        discriminator_params = list(model.module.discriminator.parameters()) if hasattr(model, 'module') else list(model.discriminator.parameters())
        optimizer_d = torch.optim.Adam(
            discriminator_params,
            lr=cfg.TRAIN.LEARNING_RATE * 0.5,
            betas=cfg.TRAIN.BETAS,
            weight_decay=cfg.TRAIN.WEIGHT_DECAY
        )
    
    # Set up learning rate scheduler
    lr_scheduler = StepLR(
        optimizer,
        step_size=cfg.TRAIN.LR_DECAY_STEP,
        gamma=cfg.TRAIN.GAMMA
    )
    
    if cfg.TRAIN.WARMUP_STEPS > 0:
        lr_scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=1,
            total_epoch=cfg.TRAIN.WARMUP_STEPS,
            after_scheduler=lr_scheduler
        )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if hasattr(cfg.CONST, 'WEIGHTS') and cfg.CONST.WEIGHTS:
        logging.info(f'Loading weights from {cfg.CONST.WEIGHTS}...')
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logging.info(f'Resumed from epoch {start_epoch}')
    
    # Loss functions
    cd_loss = chamfer_3DDist()
    
    # Training loop
    logging.info('Start training...')
    for epoch in range(start_epoch, cfg.TRAIN.N_EPOCHS):
        
        # Training phase
        train_one_epoch(
            model=model,
            train_loader=train_data_loader,
            optimizer=optimizer,
            optimizer_d=optimizer_d if cfg.NETWORK.use_gan else None,
            cd_loss=cd_loss,
            epoch=epoch,
            cfg=cfg,
            train_writer=train_writer
        )
        
        # Update learning rate
        lr_scheduler.step()
        
        # Validation phase
        if (epoch + 1) % cfg.TRAIN.SAVE_FREQ == 0:
            validate(
                model=model,
                val_loader=val_data_loader,
                cd_loss=cd_loss,
                epoch=epoch,
                cfg=cfg,
                train_writer=train_writer
            )
            
            # Save checkpoint
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                ckpt_dir=ckpt_dir,
                cfg=cfg
            )
    
    logging.info('Training completed!')
    train_writer.close()


def train_one_epoch(model, train_loader, optimizer, optimizer_d, cd_loss, epoch, cfg, train_writer):
    """Train for one epoch"""
    
    model.train()
    
    # Metrics
    metrics = {
        'total_loss': AverageMeter(),
        'cd_coarse': AverageMeter(),
        'cd_fine1': AverageMeter(),
        'cd_fine2': AverageMeter(),
        'partial_match': AverageMeter(),
    }
    
    if cfg.NETWORK.use_gan and epoch >= cfg.TRAIN.GAN_START_EPOCH:
        metrics['gan_g'] = AverageMeter()
        metrics['gan_d'] = AverageMeter()
    
    # Progress bar
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch}')
    
    for idx, (taxonomy_ids, model_ids, data) in pbar:
        # Get data
        partial = data['partial_cloud'].cuda() if torch.cuda.is_available() else data['partial_cloud']
        gt = data['gtcloud'].cuda() if torch.cuda.is_available() else data['gtcloud']
        
        # Forward pass
        coarse, fine1, fine2, losses = model(partial, gt, return_loss=True)
        
        # Compute total loss
        total_loss = (
            losses['cd_coarse'] * cfg.TRAIN.LOSS_WEIGHTS.CD_COARSE +
            losses['cd_fine1'] * cfg.TRAIN.LOSS_WEIGHTS.CD_FINE1 +
            losses['cd_fine2'] * cfg.TRAIN.LOSS_WEIGHTS.CD_FINE2 +
            losses['partial_match'] * cfg.TRAIN.LOSS_WEIGHTS.PARTIAL_MATCH
        )
        
        # GAN training
        if cfg.NETWORK.use_gan and epoch >= cfg.TRAIN.GAN_START_EPOCH:
            # Train discriminator
            if idx % cfg.TRAIN.DISCRIMINATOR_STEPS == 0:
                optimizer_d.zero_grad()
                d_loss, g_loss = model.module.compute_gan_loss(fine2, gt) if hasattr(model, 'module') else model.compute_gan_loss(fine2, gt)
                d_loss.backward(retain_graph=True)
                optimizer_d.step()
                metrics['gan_d'].update(d_loss.item())
            
            # Add generator loss
            _, g_loss = model.module.compute_gan_loss(fine2, gt) if hasattr(model, 'module') else model.compute_gan_loss(fine2, gt)
            total_loss = total_loss + g_loss * cfg.TRAIN.LOSS_WEIGHTS.GAN_G
            metrics['gan_g'].update(g_loss.item())
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update metrics
        metrics['total_loss'].update(total_loss.item())
        metrics['cd_coarse'].update(losses['cd_coarse'].item())
        metrics['cd_fine1'].update(losses['cd_fine1'].item())
        metrics['cd_fine2'].update(losses['cd_fine2'].item())
        metrics['partial_match'].update(losses['partial_match'].item())
        
        # Update progress bar
        pbar_desc = f"Epoch {epoch} | Loss: {metrics['total_loss'].avg:.4f} | " \
                   f"CD_fine2: {metrics['cd_fine2'].avg:.4f}"
        pbar.set_description(pbar_desc)
    
    # Log to TensorBoard
    step = epoch * len(train_loader)
    train_writer.add_scalar('Train/Total_Loss', metrics['total_loss'].avg, step)
    train_writer.add_scalar('Train/CD_Coarse', metrics['cd_coarse'].avg, step)
    train_writer.add_scalar('Train/CD_Fine1', metrics['cd_fine1'].avg, step)
    train_writer.add_scalar('Train/CD_Fine2', metrics['cd_fine2'].avg, step)
    train_writer.add_scalar('Train/Partial_Match', metrics['partial_match'].avg, step)
    
    if 'gan_g' in metrics:
        train_writer.add_scalar('Train/GAN_Generator', metrics['gan_g'].avg, step)
        train_writer.add_scalar('Train/GAN_Discriminator', metrics['gan_d'].avg, step)
    
    logging.info(
        f'[Epoch {epoch}] Train Loss: {metrics["total_loss"].avg:.4f}, '
        f'CD_Fine2: {metrics["cd_fine2"].avg:.4f}'
    )


def validate(model, val_loader, cd_loss, epoch, cfg, train_writer):
    """Validation"""
    
    model.eval()
    
    cd_coarse_losses = AverageMeter()
    cd_fine1_losses = AverageMeter()
    cd_fine2_losses = AverageMeter()
    
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(tqdm(val_loader, desc='Validation')):
            partial = data['partial_cloud'].cuda() if torch.cuda.is_available() else data['partial_cloud']
            gt = data['gtcloud'].cuda() if torch.cuda.is_available() else data['gtcloud']
            
            # Forward pass
            coarse, fine1, fine2 = model(partial)
            
            # Compute losses
            cd_c, _ = cd_loss(coarse, gt)
            cd_f1, _ = cd_loss(fine1, gt)
            cd_f2, _ = cd_loss(fine2, gt)
            
            cd_coarse_losses.update(cd_c.mean().item())
            cd_fine1_losses.update(cd_f1.mean().item())
            cd_fine2_losses.update(cd_f2.mean().item())
    
    # Log to TensorBoard
    train_writer.add_scalar('Val/CD_Coarse', cd_coarse_losses.avg, epoch)
    train_writer.add_scalar('Val/CD_Fine1', cd_fine1_losses.avg, epoch)
    train_writer.add_scalar('Val/CD_Fine2', cd_fine2_losses.avg, epoch)
    
    logging.info(
        f'[Epoch {epoch}] Validation - '
        f'CD_Coarse: {cd_coarse_losses.avg:.4f}, '
        f'CD_Fine1: {cd_fine1_losses.avg:.4f}, '
        f'CD_Fine2: {cd_fine2_losses.avg:.4f}'
    )


def save_checkpoint(model, optimizer, epoch, ckpt_dir, cfg):
    """Save checkpoint"""
    
    ckpt_path = os.path.join(ckpt_dir, f'epoch_{epoch}.pth')
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, ckpt_path)
    
    logging.info(f'Checkpoint saved to {ckpt_path}')
    
    # Keep only the latest N checkpoints
    checkpoints = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pth')])
    if len(checkpoints) > 5:  # Keep only 5 latest
        for old_ckpt in checkpoints[:-5]:
            os.remove(os.path.join(ckpt_dir, old_ckpt))


def setup_dataloader(cfg, split='train'):
    """Set up data loader"""
    
    from utils.data_loaders import DATASET_LOADER_MAPPING
    
    dataset_name = cfg.DATASET.TRAIN_DATASET if split == 'train' else cfg.DATASET.TEST_DATASET
    dataset_loader = DATASET_LOADER_MAPPING[dataset_name](cfg)
    
    if split == 'train':
        dataset = dataset_loader.get_dataset(DATASET_LOADER_MAPPING[dataset_name].TRAIN_DATASET)
    else:
        dataset = dataset_loader.get_dataset(DATASET_LOADER_MAPPING[dataset_name].VAL_DATASET)
    
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.CONST.NUM_WORKERS,
        collate_fn=dataset_loader.collate_fn,
        pin_memory=True,
        shuffle=(split == 'train'),
        drop_last=(split == 'train')
    )
    
    return data_loader
