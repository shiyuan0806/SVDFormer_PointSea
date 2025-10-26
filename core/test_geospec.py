"""
Testing/Evaluation script for GeoSpecNet
"""

import logging
import torch
import utils.helpers
from time import time
from tqdm import tqdm
from metrics.CD.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from models_PointSea.mv_utils_zs import PCViews_Real


def test_net(cfg, epoch_idx=-1, test_data_loader=None, test_writer=None, model=None):
    """
    Test/Validate GeoSpecNet
    
    Args:
        cfg: Configuration object
        epoch_idx: Current epoch index
        test_data_loader: DataLoader for test/validation data
        test_writer: TensorBoard writer
        model: Model to evaluate
    
    Returns:
        avg_cd: Average Chamfer Distance
    """
    torch.backends.cudnn.benchmark = True
    
    # Multi-view renderer
    render = PCViews_Real(TRANS=-cfg.NETWORK.view_distance)
    
    # Chamfer Distance metric
    chamfer_distance = chamfer_3DDist()
    
    # Set model to evaluation mode
    if model is not None:
        model.eval()
    
    # Metrics accumulators
    total_cd_coarse = 0
    total_cd_fine1 = 0
    total_cd_fine2 = 0
    n_samples = 0
    
    # Evaluation loop
    with tqdm(test_data_loader) as t:
        for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(t):
            # Move data to GPU
            for k, v in data.items():
                data[k] = utils.helpers.var_or_cuda(v)
            
            partial = data['partial_cloud']
            gt = data['gtcloud']
            
            # Render multi-view depth images
            partial_depth = render.get_img(partial)
            
            # Forward pass (no gradients)
            with torch.no_grad():
                pcds_pred = model(partial, partial_depth)
            
            # Compute Chamfer Distances
            coarse_pred = pcds_pred[0]
            fine1_pred = pcds_pred[1]
            fine2_pred = pcds_pred[2]
            
            # CD for coarse prediction
            cd_coarse_forward = chamfer_distance(
                coarse_pred.contiguous(),
                gt.contiguous()
            )[0]
            cd_coarse_backward = chamfer_distance(
                gt.contiguous(),
                coarse_pred.contiguous()
            )[0]
            cd_coarse = (cd_coarse_forward.mean() + cd_coarse_backward.mean()) / 2
            
            # CD for fine1 prediction
            cd_fine1_forward = chamfer_distance(
                fine1_pred.contiguous(),
                gt.contiguous()
            )[0]
            cd_fine1_backward = chamfer_distance(
                gt.contiguous(),
                fine1_pred.contiguous()
            )[0]
            cd_fine1 = (cd_fine1_forward.mean() + cd_fine1_backward.mean()) / 2
            
            # CD for fine2 prediction (final)
            cd_fine2_forward = chamfer_distance(
                fine2_pred.contiguous(),
                gt.contiguous()
            )[0]
            cd_fine2_backward = chamfer_distance(
                gt.contiguous(),
                fine2_pred.contiguous()
            )[0]
            cd_fine2 = (cd_fine2_forward.mean() + cd_fine2_backward.mean()) / 2
            
            # Accumulate metrics
            batch_size = partial.size(0)
            total_cd_coarse += cd_coarse.item() * batch_size
            total_cd_fine1 += cd_fine1.item() * batch_size
            total_cd_fine2 += cd_fine2.item() * batch_size
            n_samples += batch_size
            
            # Update progress bar
            t.set_description(f'[Test][Epoch {epoch_idx}]')
            t.set_postfix(
                coarse=f'{cd_coarse.item() * 1e3:.4f}',
                fine1=f'{cd_fine1.item() * 1e3:.4f}',
                fine2=f'{cd_fine2.item() * 1e3:.4f}'
            )
    
    # Compute average metrics
    avg_cd_coarse = (total_cd_coarse / n_samples) * 1e3
    avg_cd_fine1 = (total_cd_fine1 / n_samples) * 1e3
    avg_cd_fine2 = (total_cd_fine2 / n_samples) * 1e3
    
    # Log to tensorboard
    if test_writer is not None and epoch_idx > 0:
        test_writer.add_scalar('Loss/Epoch/cd_coarse', avg_cd_coarse, epoch_idx)
        test_writer.add_scalar('Loss/Epoch/cd_fine1', avg_cd_fine1, epoch_idx)
        test_writer.add_scalar('Loss/Epoch/cd_fine2', avg_cd_fine2, epoch_idx)
    
    logging.info(
        f'[Epoch {epoch_idx}] Test Results: '
        f'Coarse CD = {avg_cd_coarse:.4f}, '
        f'Fine1 CD = {avg_cd_fine1:.4f}, '
        f'Fine2 CD = {avg_cd_fine2:.4f}'
    )
    
    # Return the final CD (fine2) as the primary metric
    return avg_cd_fine2


def evaluate_net(cfg):
    """
    Full evaluation mode (for testing without training)
    
    Args:
        cfg: Configuration object
    """
    import utils.data_loaders
    from models.GeoSpecNet import GeoSpecNet
    
    torch.backends.cudnn.benchmark = True
    
    # Setup test dataset
    test_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_dataset_loader.get_dataset(utils.data_loaders.DatasetSubset.TEST),
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.CONST.NUM_WORKERS,
        collate_fn=utils.data_loaders.collate_fn,
        pin_memory=True,
        shuffle=False
    )
    
    # Initialize model
    model = GeoSpecNet(cfg)
    
    # Load checkpoint
    if 'WEIGHTS' in cfg.CONST and cfg.CONST.WEIGHTS:
        logging.info(f'Loading weights from {cfg.CONST.WEIGHTS}')
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        model.load_state_dict(checkpoint['model'])
        logging.info('Weights loaded successfully')
    else:
        logging.error('No weights specified for evaluation!')
        return
    
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    
    # Evaluate
    avg_cd = test_net(cfg, epoch_idx=0, test_data_loader=test_data_loader, test_writer=None, model=model)
    
    logging.info(f'Final Evaluation Result: CD = {avg_cd:.4f}')
    
    return avg_cd
