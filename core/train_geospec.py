import logging
import os
import torch
import utils.data_loaders
import utils.helpers
from tqdm import tqdm
from time import time
from tensorboardX import SummaryWriter
from core.test_pcn import test_net as test_pcn
from utils.average_meter import AverageMeter
from torch.optim.lr_scheduler import MultiStepLR
from utils.schedular import GradualWarmupScheduler
from utils.loss_utils import get_loss_PM
from models.model_utils import PCViews
from models.GeoSpecNet import Model as GeoSpecModel, Discriminator


def train_net(cfg):
    torch.backends.cudnn.benchmark = True

    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    test_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)

    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TRAIN),
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.CONST.NUM_WORKERS,
        collate_fn=utils.data_loaders.collate_fn, pin_memory=True,
        shuffle=True,
        drop_last=False)
    val_data_loader = torch.utils.data.DataLoader(dataset=test_dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TEST),
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.CONST.NUM_WORKERS//2,
        collate_fn=utils.data_loaders.collate_fn,
        pin_memory=True,
        shuffle=False)

    # Set up folders for logs and checkpoints
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', cfg.TRAIN.RUN_ID)
    cfg.DIR.CHECKPOINTS = output_dir % 'checkpoints'
    cfg.DIR.LOGS = output_dir % 'logs'
    if not os.path.exists(cfg.DIR.CHECKPOINTS):
        os.makedirs(cfg.DIR.CHECKPOINTS)

    # Create tensorboard writers
    train_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
    val_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))

    G = GeoSpecModel(cfg)
    D = Discriminator()
    if torch.cuda.is_available():
        G = torch.nn.DataParallel(G).cuda()
        D = torch.nn.DataParallel(D).cuda()

    # Optimizers
    g_optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, G.parameters()),
                                lr=cfg.TRAIN.LEARNING_RATE, weight_decay=0.0005)
    d_optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, D.parameters()),
                                lr=cfg.TRAIN.LEARNING_RATE, weight_decay=0.0005)

    # LR scheduler
    scheduler_steplr_g = MultiStepLR(g_optim, milestones=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.GAMMA)
    scheduler_steplr_d = MultiStepLR(d_optim, milestones=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.GAMMA)
    g_scheduler = GradualWarmupScheduler(g_optim, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_STEPS,
                                         after_scheduler=scheduler_steplr_g)
    d_scheduler = GradualWarmupScheduler(d_optim, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_STEPS,
                                         after_scheduler=scheduler_steplr_d)

    init_epoch = 0
    best_metrics = float('inf')
    steps = 0
    BestEpoch = 0
    render = PCViews(TRANS=-cfg.NETWORK.view_distance, RESOLUTION=224)

    bce = torch.nn.BCEWithLogitsLoss()

    # Training/Testing the network
    for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):
        epoch_start_time = time()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        G.train(); D.train()

        total_cd_pc = 0
        total_cd_p1 = 0
        total_cd_p2 = 0
        total_gan_d = 0
        total_gan_g = 0

        batch_end_time = time()
        n_batches = len(train_data_loader)
        print('epoch: ', epoch_idx, 'lr_g: ', g_optim.param_groups[0]['lr'], 'lr_d: ', d_optim.param_groups[0]['lr'])
        with tqdm(train_data_loader) as t:
            for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(t):
                data_time.update(time() - batch_end_time)
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)
                partial = data['partial_cloud']
                gt = data['gtcloud']

                partial_depth = torch.unsqueeze(render.get_img(partial), 1)
                pcds_pred = G(partial, partial_depth)

                # Reconstruction loss with partial matching
                loss_total, losses = get_loss_PM(pcds_pred, partial, gt, sqrt=True)

                # GAN losses
                P2 = pcds_pred[-1].detach()
                B = gt.shape[0]
                real_logit = D(gt)
                fake_logit = D(P2)
                d_loss_real = bce(real_logit, torch.ones_like(real_logit))
                d_loss_fake = bce(fake_logit, torch.zeros_like(fake_logit))
                d_loss = (d_loss_real + d_loss_fake) * 0.5

                d_optim.zero_grad()
                d_loss.backward()
                d_optim.step()

                # Generator adversarial loss
                fake_logit_g = D(pcds_pred[-1])
                g_gan = bce(fake_logit_g, torch.ones_like(fake_logit_g))
                total_g = loss_total + cfg.TRAIN.GAN_WEIGHT * g_gan

                g_optim.zero_grad()
                total_g.backward()
                g_optim.step()

                cd_pc_item = losses[0].item() * 1e3
                total_cd_pc += cd_pc_item
                cd_p1_item = losses[1].item() * 1e3
                total_cd_p1 += cd_p1_item
                cd_p2_item = losses[2].item() * 1e3
                total_cd_p2 += cd_p2_item
                total_gan_d += d_loss.item()
                total_gan_g += g_gan.item()

                n_itr = (epoch_idx - 1) * n_batches + batch_idx
                train_writer.add_scalar('Loss/Batch/cd_pc', cd_pc_item, n_itr)
                train_writer.add_scalar('Loss/Batch/cd_p1', cd_p1_item, n_itr)
                train_writer.add_scalar('Loss/Batch/cd_p2', cd_p2_item, n_itr)
                train_writer.add_scalar('Loss/Batch/gan_d', d_loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/gan_g', g_gan.item(), n_itr)

                batch_time.update(time() - batch_end_time)
                batch_end_time = time()
                t.set_description('[Epoch %d/%d][Batch %d/%d]' % (epoch_idx, cfg.TRAIN.N_EPOCHS, batch_idx + 1, n_batches))
                t.set_postfix(loss='%s' % ['%.4f' % l for l in [cd_pc_item, cd_p1_item, cd_p2_item, d_loss.item(), g_gan.item()]])

                if steps <= cfg.TRAIN.WARMUP_STEPS:
                    g_scheduler.step(); d_scheduler.step()
                    steps += 1

        avg_cdc = total_cd_pc / n_batches
        avg_cd1 = total_cd_p1 / n_batches
        avg_cd2 = total_cd_p2 / n_batches
        avg_d = total_gan_d / n_batches
        avg_g = total_gan_g / n_batches

        scheduler_steplr_g.step(); scheduler_steplr_d.step()
        epoch_end_time = time()
        train_writer.add_scalar('Loss/Epoch/cd_pc', avg_cdc, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/cd_p1', avg_cd1, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/cd_p2', avg_cd2, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/gan_d', avg_d, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/gan_g', avg_g, epoch_idx)
        logging.info(
            '[Epoch %d/%d] EpochTime = %.3f (s) Losses = %s' %
            (epoch_idx, cfg.TRAIN.N_EPOCHS, epoch_end_time - epoch_start_time, ['%.4f' % l for l in [avg_cdc, avg_cd1, avg_cd2, avg_d, avg_g]]))

        # Validate
        cd_eval = test_pcn(cfg, epoch_idx, val_data_loader, val_writer, G)
        if epoch_idx % cfg.TRAIN.SAVE_FREQ == 0 or cd_eval < best_metrics:
            if cd_eval < best_metrics:
                best_metrics = cd_eval
                BestEpoch = epoch_idx
                file_name = 'ckpt-best.pth'
            else:
                file_name = 'ckpt-epoch-%03d.pth' % epoch_idx
            output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
            torch.save({'G': G.state_dict(), 'D': D.state_dict(), 'g_optim': g_optim.state_dict(), 'd_optim': d_optim.state_dict()}, output_path)
            logging.info('Saved checkpoint to %s ...' % output_path)

        logging.info('Best Performance: Epoch %d -- CD %.4f' % (BestEpoch, best_metrics))

    train_writer.close(); val_writer.close()
