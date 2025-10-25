"""
Testing script for GeoSpecNet
Evaluates the model on test datasets and saves completions
"""

import os
import torch
import torch.utils.data
import logging
import numpy as np
from tqdm import tqdm
import json

from models.GeoSpecNet import GeoSpecNet
from metrics.CD.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from metrics.EMD.emd_module import emdModule
from utils.average_meter import AverageMeter


def test_net(cfg):
    """Main testing function for GeoSpecNet"""
    
    # Set up output directory
    output_dir = cfg.TEST.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up data loader
    test_data_loader = setup_dataloader(cfg, 'test')
    
    # Build model
    model = GeoSpecNet(cfg)
    
    # Move model to GPU
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    
    # Load weights
    if not hasattr(cfg.CONST, 'WEIGHTS') or not cfg.CONST.WEIGHTS:
        raise ValueError('Please specify checkpoint path in cfg.CONST.WEIGHTS')
    
    logging.info(f'Loading weights from {cfg.CONST.WEIGHTS}...')
    checkpoint = torch.load(cfg.CONST.WEIGHTS)
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info('Weights loaded successfully')
    
    # Evaluate
    logging.info('Starting evaluation...')
    metrics = evaluate_model(
        model=model,
        test_loader=test_data_loader,
        cfg=cfg,
        output_dir=output_dir
    )
    
    # Print results
    print('\n' + '='*60)
    print('GeoSpecNet Evaluation Results')
    print('='*60)
    print(f"Dataset: {cfg.DATASET.TEST_DATASET}")
    print(f"Checkpoint: {cfg.CONST.WEIGHTS}")
    print('-'*60)
    print(f"Chamfer Distance (Coarse): {metrics['cd_coarse']:.6f}")
    print(f"Chamfer Distance (Fine1):  {metrics['cd_fine1']:.6f}")
    print(f"Chamfer Distance (Fine2):  {metrics['cd_fine2']:.6f}")
    if 'f_score' in metrics:
        print(f"F-Score @ 0.01:            {metrics['f_score']:.4f}")
    print('='*60)
    
    # Save metrics to file
    metrics_file = os.path.join(output_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    logging.info(f'Metrics saved to {metrics_file}')
    
    return metrics


def evaluate_model(model, test_loader, cfg, output_dir):
    """Evaluate model on test set"""
    
    model.eval()
    
    # Metrics
    cd_coarse_losses = AverageMeter()
    cd_fine1_losses = AverageMeter()
    cd_fine2_losses = AverageMeter()
    f_scores = AverageMeter()
    
    # Loss functions
    cd_loss = chamfer_3DDist()
    
    # Category-wise metrics
    category_metrics = {}
    
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(tqdm(test_loader, desc='Testing')):
            
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids, list) else taxonomy_ids
            model_id = model_ids[0] if isinstance(model_ids, list) else model_ids
            
            # Get data
            partial = data['partial_cloud']
            gt = data['gtcloud']
            
            if torch.cuda.is_available():
                partial = partial.cuda()
                gt = gt.cuda()
            
            # Forward pass
            coarse, fine1, fine2 = model(partial)
            
            # Compute Chamfer Distance
            cd_c, _ = cd_loss(coarse, gt)
            cd_f1, _ = cd_loss(fine1, gt)
            cd_f2, _ = cd_loss(fine2, gt)
            
            cd_coarse_losses.update(cd_c.mean().item())
            cd_fine1_losses.update(cd_f1.mean().item())
            cd_fine2_losses.update(cd_f2.mean().item())
            
            # Compute F-Score
            f_score = compute_f_score(fine2[0].cpu().numpy(), gt[0].cpu().numpy(), threshold=0.01)
            f_scores.update(f_score)
            
            # Update category metrics
            if taxonomy_id not in category_metrics:
                category_metrics[taxonomy_id] = {
                    'cd_coarse': AverageMeter(),
                    'cd_fine1': AverageMeter(),
                    'cd_fine2': AverageMeter(),
                    'f_score': AverageMeter()
                }
            
            category_metrics[taxonomy_id]['cd_coarse'].update(cd_c.mean().item())
            category_metrics[taxonomy_id]['cd_fine1'].update(cd_f1.mean().item())
            category_metrics[taxonomy_id]['cd_fine2'].update(cd_f2.mean().item())
            category_metrics[taxonomy_id]['f_score'].update(f_score)
            
            # Save completions if specified
            if cfg.TEST.SAVE_COMPLETIONS and idx < 100:  # Save first 100 samples
                save_completion(
                    partial=partial[0].cpu().numpy(),
                    coarse=coarse[0].cpu().numpy(),
                    fine1=fine1[0].cpu().numpy(),
                    fine2=fine2[0].cpu().numpy(),
                    gt=gt[0].cpu().numpy(),
                    taxonomy_id=taxonomy_id,
                    model_id=model_id,
                    output_dir=output_dir
                )
    
    # Aggregate metrics
    metrics = {
        'cd_coarse': cd_coarse_losses.avg,
        'cd_fine1': cd_fine1_losses.avg,
        'cd_fine2': cd_fine2_losses.avg,
        'f_score': f_scores.avg,
    }
    
    # Add category-wise metrics
    for taxonomy_id, cat_metrics in category_metrics.items():
        metrics[f'{taxonomy_id}_cd_fine2'] = cat_metrics['cd_fine2'].avg
        metrics[f'{taxonomy_id}_f_score'] = cat_metrics['f_score'].avg
    
    return metrics


def compute_f_score(pred, gt, threshold=0.01):
    """
    Compute F-Score between prediction and ground truth
    
    Args:
        pred: (N, 3) predicted point cloud
        gt: (M, 3) ground truth point cloud
        threshold: distance threshold
    
    Returns:
        f_score: F-Score value
    """
    # Compute pairwise distances
    pred_tensor = torch.from_numpy(pred).float().unsqueeze(0)
    gt_tensor = torch.from_numpy(gt).float().unsqueeze(0)
    
    # Precision: percentage of predicted points close to GT
    dist_pred_to_gt = torch.cdist(pred_tensor, gt_tensor).squeeze(0)
    precision = (dist_pred_to_gt.min(dim=1)[0] < threshold).float().mean().item()
    
    # Recall: percentage of GT points close to prediction
    recall = (dist_pred_to_gt.min(dim=0)[0] < threshold).float().mean().item()
    
    # F-Score
    if precision + recall > 0:
        f_score = 2 * precision * recall / (precision + recall)
    else:
        f_score = 0.0
    
    return f_score


def save_completion(partial, coarse, fine1, fine2, gt, taxonomy_id, model_id, output_dir):
    """Save completion results"""
    
    # Create directory
    save_dir = os.path.join(output_dir, 'completions', taxonomy_id)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save as .npy files
    np.save(os.path.join(save_dir, f'{model_id}_partial.npy'), partial)
    np.save(os.path.join(save_dir, f'{model_id}_coarse.npy'), coarse)
    np.save(os.path.join(save_dir, f'{model_id}_fine1.npy'), fine1)
    np.save(os.path.join(save_dir, f'{model_id}_fine2.npy'), fine2)
    np.save(os.path.join(save_dir, f'{model_id}_gt.npy'), gt)


def setup_dataloader(cfg, split='test'):
    """Set up data loader for testing"""
    
    from utils.data_loaders import DATASET_LOADER_MAPPING
    
    dataset_name = cfg.DATASET.TEST_DATASET
    dataset_loader = DATASET_LOADER_MAPPING[dataset_name](cfg)
    
    dataset = dataset_loader.get_dataset(DATASET_LOADER_MAPPING[dataset_name].TEST_DATASET)
    
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,  # Test with batch size 1
        num_workers=cfg.CONST.NUM_WORKERS,
        collate_fn=dataset_loader.collate_fn,
        pin_memory=True,
        shuffle=False
    )
    
    return data_loader


def inference_single(model, partial_cloud):
    """
    Run inference on a single partial point cloud
    
    Args:
        model: GeoSpecNet model
        partial_cloud: (N, 3) or (1, N, 3) partial point cloud
    
    Returns:
        completion: (M, 3) completed point cloud
    """
    model.eval()
    
    # Ensure correct shape
    if len(partial_cloud.shape) == 2:
        partial_cloud = partial_cloud.unsqueeze(0)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        partial_cloud = partial_cloud.cuda()
    
    with torch.no_grad():
        _, _, completion = model(partial_cloud)
    
    return completion.squeeze(0).cpu().numpy()
