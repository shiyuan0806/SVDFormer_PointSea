import logging
import os
import torch
import utils.data_loaders
import utils.helpers
from datetime import datetime
from tqdm import tqdm
from time import time
from tensorboardX import SummaryWriter
from core.test_geospecnet import test_net
from utils.average_meter import AverageMeter
from torch.optim.lr_scheduler import StepLR
from utils.schedular import GradualWarmupScheduler
from utils.loss_utils import get_loss_PM
from utils.helpers import seprate_point_cloud
from models.model_utils import PCViews
from models.GeoSpecNet import GeoSpecNet
from models.discriminator import PointCloudDiscriminator


def train_net(cfg):
    torch.backends.cudnn.benchmark = True

    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg).get_dataset(
        utils.data_loaders.DatasetSubset.TRAIN)
    test_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)

    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader,
                                                    batch_size=cfg.TRAIN.BATCH_SIZE,
                                                    num_workers=cfg.CONST.NUM_WORKERS,
                                                    collate_fn=utils.data_loaders.collate_fn_55, pin_memory=True,
                                                    shuffle=True,
                                                    drop_last=False)
    val_data_loader = torch.utils.data.DataLoader(dataset=test_dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TEST),
        batch_size=2,
        num_workers=cfg.CONST.NUM_WORKERS // 2,
        collate_fn=utils.data_loaders.collate_fn_55,
        pin_memory=True,
        shuffle=False)

    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', datetime.now().isoformat())
    cfg.DIR.CHECKPOINTS = output_dir % 'checkpoints'
    cfg.DIR.LOGS = output_dir % 'logs'
    if not os.path.exists(cfg.DIR.CHECKPOINTS):
        os.makedirs(cfg.DIR.CHECKPOINTS)

    train_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
    val_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))

    model = GeoSpecNet(cfg)
    disc = PointCloudDiscriminator()

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        disc = torch.nn.DataParallel(disc).cuda()

    optimizer_G = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=cfg.TRAIN.LEARNING_RATE,
                                 weight_decay=0.0005)
    optimizer_D = torch.optim.AdamW(filter(lambda p: p.requires_grad, disc.parameters()),
                                    lr=1e-4, weight_decay=0.0005)

    scheduler_steplr_G = StepLR(optimizer_G, step_size=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.GAMMA)
    scheduler_steplr_D = StepLR(optimizer_D, step_size=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.GAMMA)

    lr_scheduler_G = GradualWarmupScheduler(optimizer_G, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_STEPS,
                                          after_scheduler=scheduler_steplr_G)
    lr_scheduler_D = GradualWarmupScheduler(optimizer_D, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_STEPS,
                                          after_scheduler=scheduler_steplr_D)

    init_epoch = 0
    best_metrics = float('inf')
    steps = 0
    BestEpoch = 0

    render = PCViews(TRANS=-cfg.NETWORK.view_distance, RESOLUTION=224)

    if 'WEIGHTS' in cfg.CONST:
        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        model.load_state_dict(checkpoint['model'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        disc.load_state_dict(checkpoint['disc'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        steps = cfg.TRAIN.WARMUP_STEPS + 1
        lr_scheduler_G = StepLR(optimizer_G, step_size=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.GAMMA)
        lr_scheduler_D = StepLR(optimizer_D, step_size=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.GAMMA)
        optimizer_G.param_groups[0]['lr'] = cfg.TRAIN.LEARNING_RATE

        logging.info('Recover complete.')

    for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):
        epoch_start_time = time()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        model.train()
        disc.train()

        total_cd_pc = 0
        total_cd_p1 = 0
        total_cd_p2 = 0
        total_gan = 0

        batch_end_time = time()
        n_batches = len(train_data_loader)
        print('epoch: ', epoch_idx, 'lr(G): ', optimizer_G.param_groups[0]['lr'])
        with tqdm(train_data_loader) as t:
            for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(t):
                data_time.update(time() - batch_end_time)
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)
                gt = data['gtcloud']
                batchsize, npoints, _ = gt.size()
                if batchsize % 2 != 0:
                    gt = torch.cat([gt, gt], 0)
                partial, _ = seprate_point_cloud(gt, npoints, [int(npoints * 1/4), int(npoints * 3/4)], fixed_points=None)
                partial_depth = torch.unsqueeze(render.get_img(partial), 1)

                # Generator forward
                Pc, P1, P2 = model(partial, partial_depth)

                # Chamfer + Partial Matching loss
                loss_rec, losses = get_loss_PM((Pc, P1, P2), partial, gt, sqrt=False)

                # GAN losses (WGAN-GP could be used; keep simple BCE)
                real_logits = disc(gt.detach())
                fake_logits = disc(P2.detach())
                d_real = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
                d_fake = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
                d_loss = (d_real + d_fake) * 0.5

                optimizer_D.zero_grad()
                d_loss.backward()
                optimizer_D.step()

                # Generator adversarial loss
                fake_logits_g = disc(P2)
                g_gan = F.binary_cross_entropy_with_logits(fake_logits_g, torch.ones_like(fake_logits_g))
                gan_weight = getattr(cfg.TRAIN, 'GAN_WEIGHT', 0.1)

                g_loss = loss_rec + gan_weight * g_gan

                optimizer_G.zero_grad()
                g_loss.backward()
                optimizer_G.step()

                cd_pc_item = losses[0].item() * 1e3
                total_cd_pc += cd_pc_item
                cd_p1_item = losses[1].item() * 1e3
                total_cd_p1 += cd_p1_item
                cd_p2_item = losses[2].item() * 1e3
                total_cd_p2 += cd_p2_item
                total_gan += g_gan.item()

                n_itr = (epoch_idx - 1) * n_batches + batch_idx
                train_writer.add_scalar('Loss/Batch/cd_pc', cd_pc_item, n_itr)
                train_writer.add_scalar('Loss/Batch/cd_p1', cd_p1_item, n_itr)
                train_writer.add_scalar('Loss/Batch/cd_p2', cd_p2_item, n_itr)
                train_writer.add_scalar('Loss/Batch/gan', g_gan.item(), n_itr)

                batch_time.update(time() - batch_end_time)
                batch_end_time = time()
                t.set_description('[Epoch %d/%d][Batch %d/%d]' % (epoch_idx, cfg.TRAIN.N_EPOCHS, batch_idx + 1, n_batches))
                t.set_postfix(loss='%s' % ['%.4f' % l for l in [cd_pc_item, cd_p1_item, cd_p2_item, g_gan.item()]])

                if steps <= cfg.TRAIN.WARMUP_STEPS:
                    lr_scheduler_G.step()
                    lr_scheduler_D.step()
                    steps += 1

        avg_cdc = total_cd_pc / n_batches
        avg_cd1 = total_cd_p1 / n_batches
        avg_cd2 = total_cd_p2 / n_batches
        avg_gan = total_gan / n_batches

        lr_scheduler_G.step()
        lr_scheduler_D.step()
        epoch_end_time = time()
        train_writer.add_scalar('Loss/Epoch/cd_pc', avg_cdc, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/cd_p1', avg_cd1, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/cd_p2', avg_cd2, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/gan', avg_gan, epoch_idx)
        logging.info('[Epoch %d/%d] EpochTime = %.3f (s) Losses = %s' % (
            epoch_idx, cfg.TRAIN.N_EPOCHS, epoch_end_time - epoch_start_time, ['%.4f' % l for l in [avg_cdc, avg_cd1, avg_cd2, avg_gan]]))

        if epoch_idx % 1 == 0:
            cd_eval = test_net(cfg, epoch_idx, val_data_loader, val_writer, model)
            if epoch_idx % cfg.TRAIN.SAVE_FREQ == 0 or cd_eval < best_metrics:
                if cd_eval < best_metrics:
                    best_metrics = cd_eval
                    BestEpoch = epoch_idx
                    file_name = 'ckpt-best.pth'
                else:
                    file_name = 'ckpt-epoch-%03d.pth' % epoch_idx
                output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
                torch.save({
                    'model': model.state_dict(),
                    'optimizer_G': optimizer_G.state_dict(),
                    'disc': disc.state_dict(),
                    'optimizer_D': optimizer_D.state_dict()
                }, output_path)
                logging.info('Saved checkpoint to %s ...' % output_path)

        logging.info('Best Performance: Epoch %d -- CD %.4f' % (BestEpoch, best_metrics))

    train_writer.close()
    val_writer.close()
