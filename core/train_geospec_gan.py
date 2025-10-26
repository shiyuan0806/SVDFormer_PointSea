"""
Training script for GeoSpecNet with Self-supervised Structural Consistency Training
使用自监督结构一致性训练的 GeoSpecNet 训练脚本

包括:
1. GAN 对抗训练
2. 部分匹配损失
3. 结构一致性约束
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
from utils.gan_loss import StructuralConsistencyLoss
from models_PointSea.mv_utils_zs import PCViews_Real
from models.GeoSpecNet import GeoSpecNet
from models.discriminator import get_discriminator


def train_net_with_gan(cfg):
    """
    使用 GAN 训练 GeoSpecNet
    
    训练策略：
    1. 前 warmup_gan_epochs 个 epoch 只训练生成器（不使用判别器）
    2. 之后交替训练生成器和判别器
    3. 生成器更新频率 > 判别器更新频率（通常 2:1 或 3:1）
    """
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
    
    # Initialize generator (GeoSpecNet)
    generator = GeoSpecNet(cfg)
    if torch.cuda.is_available():
        generator = torch.nn.DataParallel(generator).cuda()
    
    # Initialize discriminator
    discriminator = get_discriminator(
        disc_type=cfg.GAN.DISC_TYPE,
        input_dim=3,
        hidden_dim=cfg.GAN.DISC_HIDDEN_DIM
    )
    if torch.cuda.is_available():
        discriminator = torch.nn.DataParallel(discriminator).cuda()
    
    logging.info(f'Generator parameters: {sum(p.numel() for p in generator.parameters()):,}')
    logging.info(f'Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}')
    
    # Create optimizers
    optimizer_G = torch.optim.Adam(
        filter(lambda p: p.requires_grad, generator.parameters()),
        lr=cfg.TRAIN.LEARNING_RATE,
        weight_decay=cfg.TRAIN.WEIGHT_DECAY,
        betas=cfg.TRAIN.BETAS
    )
    
    optimizer_D = torch.optim.Adam(
        filter(lambda p: p.requires_grad, discriminator.parameters()),
        lr=cfg.GAN.DISC_LEARNING_RATE,
        weight_decay=cfg.TRAIN.WEIGHT_DECAY,
        betas=cfg.TRAIN.BETAS
    )
    
    # Learning rate schedulers
    scheduler_G_steplr = MultiStepLR(
        optimizer_G,
        milestones=cfg.TRAIN.LR_DECAY_STEP,
        gamma=cfg.TRAIN.GAMMA
    )
    lr_scheduler_G = GradualWarmupScheduler(
        optimizer_G,
        multiplier=1,
        total_epoch=cfg.TRAIN.WARMUP_STEPS,
        after_scheduler=scheduler_G_steplr
    )
    
    scheduler_D_steplr = MultiStepLR(
        optimizer_D,
        milestones=cfg.TRAIN.LR_DECAY_STEP,
        gamma=cfg.TRAIN.GAMMA
    )
    lr_scheduler_D = GradualWarmupScheduler(
        optimizer_D,
        multiplier=1,
        total_epoch=cfg.TRAIN.WARMUP_STEPS,
        after_scheduler=scheduler_D_steplr
    )
    
    # Initialize loss functions
    structural_consistency_loss = StructuralConsistencyLoss(
        use_partial_matching=cfg.GAN.USE_PARTIAL_MATCHING,
        use_consistency=cfg.GAN.USE_CONSISTENCY,
        partial_matching_weight=cfg.GAN.PARTIAL_MATCHING_WEIGHT,
        consistency_weight=cfg.GAN.CONSISTENCY_WEIGHT,
        gan_weight=cfg.GAN.GAN_WEIGHT,
        gan_mode=cfg.GAN.GAN_MODE
    )
    
    init_epoch = 0
    best_metrics = float('inf')
    steps = 0
    best_epoch = 0
    
    # Multi-view renderer
    render = PCViews_Real(TRANS=-cfg.NETWORK.view_distance)
    
    # Resume from checkpoint if specified
    if 'WEIGHTS' in cfg.CONST and cfg.CONST.WEIGHTS:
        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        generator.load_state_dict(checkpoint['generator'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        
        if 'discriminator' in checkpoint:
            discriminator.load_state_dict(checkpoint['discriminator'])
            optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        
        if 'epoch' in checkpoint:
            init_epoch = checkpoint['epoch']
        
        steps = cfg.TRAIN.WARMUP_STEPS + 1
        lr_scheduler_G = MultiStepLR(
            optimizer_G,
            milestones=cfg.TRAIN.LR_DECAY_STEP,
            gamma=cfg.TRAIN.GAMMA
        )
        lr_scheduler_D = MultiStepLR(
            optimizer_D,
            milestones=cfg.TRAIN.LR_DECAY_STEP,
            gamma=cfg.TRAIN.GAMMA
        )
        optimizer_G.param_groups[0]['lr'] = cfg.TRAIN.LEARNING_RATE
        optimizer_D.param_groups[0]['lr'] = cfg.GAN.DISC_LEARNING_RATE
        
        logging.info('Recovery complete.')
    
    # Training loop
    for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):
        epoch_start_time = time()
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        
        generator.train()
        discriminator.train()
        
        # Loss accumulators
        total_cd_coarse = 0
        total_cd_fine1 = 0
        total_cd_fine2 = 0
        total_gen_loss = 0
        total_disc_loss = 0
        total_pm_loss = 0
        total_cons_loss = 0
        total_gan_loss = 0
        
        batch_end_time = time()
        n_batches = len(train_data_loader)
        
        # 判断是否使用GAN训练
        use_gan = epoch_idx > cfg.GAN.WARMUP_GAN_EPOCHS
        
        print(f'Epoch: {epoch_idx}, LR_G: {optimizer_G.param_groups[0]["lr"]:.6f}, '
              f'LR_D: {optimizer_D.param_groups[0]["lr"]:.6f}, '
              f'GAN: {"ON" if use_gan else "OFF"}')
        
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
                
                # ==================== Train Generator ====================
                # Forward pass
                pcds_pred = generator(partial, partial_depth)
                coarse_pred, fine1_pred, fine2_pred = pcds_pred
                
                # Compute reconstruction loss (Chamfer Distance)
                recon_loss, cd_losses = get_loss(pcds_pred, gt, sqrt=True)
                
                # Compute structural consistency loss (if GAN is enabled)
                if use_gan:
                    gen_sc_loss, gen_sc_dict = structural_consistency_loss.compute_generator_loss(
                        fine2_pred, partial, discriminator
                    )
                    total_gen_loss_value = recon_loss + gen_sc_loss
                else:
                    gen_sc_loss = torch.tensor(0.0).cuda()
                    gen_sc_dict = {
                        'partial_matching': 0.0,
                        'consistency': 0.0,
                        'gan': 0.0,
                        'total': 0.0
                    }
                    total_gen_loss_value = recon_loss
                
                # Backward and optimize generator
                optimizer_G.zero_grad()
                total_gen_loss_value.backward()
                optimizer_G.step()
                
                # ==================== Train Discriminator ====================
                if use_gan and (batch_idx % cfg.GAN.DISC_UPDATE_FREQ == 0):
                    # 判别器更新频率可以低于生成器
                    disc_loss, disc_loss_dict = structural_consistency_loss.compute_discriminator_loss(
                        fine2_pred.detach(), gt, discriminator
                    )
                    
                    optimizer_D.zero_grad()
                    disc_loss.backward()
                    optimizer_D.step()
                    
                    total_disc_loss += disc_loss.item()
                else:
                    disc_loss_dict = {
                        'disc_total': 0.0,
                        'disc_real': 0.0,
                        'disc_fake': 0.0,
                        'acc_real': 0.0,
                        'acc_fake': 0.0
                    }
                
                # Record losses
                cd_coarse_item = cd_losses[0].item() * 1e3
                cd_fine1_item = cd_losses[1].item() * 1e3
                cd_fine2_item = cd_losses[2].item() * 1e3
                
                total_cd_coarse += cd_coarse_item
                total_cd_fine1 += cd_fine1_item
                total_cd_fine2 += cd_fine2_item
                total_pm_loss += gen_sc_dict.get('partial_matching', 0.0)
                total_cons_loss += gen_sc_dict.get('consistency', 0.0)
                total_gan_loss += gen_sc_dict.get('gan', 0.0)
                
                # Log to tensorboard
                n_itr = (epoch_idx - 1) * n_batches + batch_idx
                train_writer.add_scalar('Loss/Batch/cd_coarse', cd_coarse_item, n_itr)
                train_writer.add_scalar('Loss/Batch/cd_fine1', cd_fine1_item, n_itr)
                train_writer.add_scalar('Loss/Batch/cd_fine2', cd_fine2_item, n_itr)
                
                if use_gan:
                    train_writer.add_scalar('Loss/Batch/partial_matching', gen_sc_dict['partial_matching'], n_itr)
                    train_writer.add_scalar('Loss/Batch/consistency', gen_sc_dict['consistency'], n_itr)
                    train_writer.add_scalar('Loss/Batch/gan', gen_sc_dict['gan'], n_itr)
                    train_writer.add_scalar('Loss/Batch/disc_total', disc_loss_dict['disc_total'], n_itr)
                    train_writer.add_scalar('Loss/Batch/disc_acc_real', disc_loss_dict['acc_real'], n_itr)
                    train_writer.add_scalar('Loss/Batch/disc_acc_fake', disc_loss_dict['acc_fake'], n_itr)
                
                batch_time.update(time() - batch_end_time)
                batch_end_time = time()
                
                # Update progress bar
                desc = f'[Epoch {epoch_idx}/{cfg.TRAIN.N_EPOCHS}][Batch {batch_idx + 1}/{n_batches}]'
                postfix = {
                    'CD': f'{cd_fine2_item:.2f}',
                    'PM': f'{gen_sc_dict.get("partial_matching", 0.0):.4f}' if use_gan else '-',
                    'D': f'{disc_loss_dict["disc_total"]:.4f}' if use_gan else '-'
                }
                t.set_description(desc)
                t.set_postfix(postfix)
                
                # Learning rate warmup
                if steps <= cfg.TRAIN.WARMUP_STEPS:
                    lr_scheduler_G.step()
                    lr_scheduler_D.step()
                    steps += 1
        
        # Epoch statistics
        avg_cd_coarse = total_cd_coarse / n_batches
        avg_cd_fine1 = total_cd_fine1 / n_batches
        avg_cd_fine2 = total_cd_fine2 / n_batches
        avg_pm_loss = total_pm_loss / n_batches
        avg_cons_loss = total_cons_loss / n_batches
        avg_gan_loss = total_gan_loss / n_batches
        avg_disc_loss = total_disc_loss / max(n_batches // cfg.GAN.DISC_UPDATE_FREQ, 1)
        
        lr_scheduler_G.step()
        lr_scheduler_D.step()
        epoch_end_time = time()
        
        # Log epoch losses
        train_writer.add_scalar('Loss/Epoch/cd_coarse', avg_cd_coarse, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/cd_fine1', avg_cd_fine1, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/cd_fine2', avg_cd_fine2, epoch_idx)
        
        if use_gan:
            train_writer.add_scalar('Loss/Epoch/partial_matching', avg_pm_loss, epoch_idx)
            train_writer.add_scalar('Loss/Epoch/consistency', avg_cons_loss, epoch_idx)
            train_writer.add_scalar('Loss/Epoch/gan', avg_gan_loss, epoch_idx)
            train_writer.add_scalar('Loss/Epoch/discriminator', avg_disc_loss, epoch_idx)
        
        logging.info(
            f'[Epoch {epoch_idx}/{cfg.TRAIN.N_EPOCHS}] '
            f'Time={epoch_end_time - epoch_start_time:.2f}s '
            f'CD=[{avg_cd_coarse:.2f}, {avg_cd_fine1:.2f}, {avg_cd_fine2:.2f}] '
            f'PM={avg_pm_loss:.4f} D={avg_disc_loss:.4f}'
        )
        
        # Validation
        cd_eval = test_net(cfg, epoch_idx, val_data_loader, val_writer, generator)
        
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
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
                'best_metrics': best_metrics
            }, output_path)
            
            logging.info(f'Saved checkpoint to {output_path}')
        
        logging.info(f'Best Performance: Epoch {best_epoch} -- CD {best_metrics:.4f}')
    
    train_writer.close()
    val_writer.close()
