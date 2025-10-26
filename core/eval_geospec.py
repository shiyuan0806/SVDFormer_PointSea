"""
Evaluation script for GeoSpecNet
"""

import logging
import os
import torch
import utils.data_loaders
import utils.helpers
from tqdm import tqdm
from metrics.CD.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from models_PointSea.mv_utils_zs import PCViews_Real
from models.GeoSpecNet import GeoSpecNet


def eval_net(cfg):
    """
    Evaluate GeoSpecNet on test dataset and save results
    
    Args:
        cfg: Configuration object
    """
    torch.backends.cudnn.benchmark = True
    
    # Setup test dataset
    test_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_dataset_loader.get_dataset(utils.data_loaders.DatasetSubset.TEST),
        batch_size=1,  # Evaluate one at a time for detailed metrics
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
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        logging.info('Weights loaded successfully')
    else:
        logging.error('No weights specified for evaluation!')
        return
    
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    
    model.eval()
    
    # Multi-view renderer
    render = PCViews_Real(TRANS=-cfg.NETWORK.view_distance)
    
    # Chamfer Distance metric
    chamfer_distance = chamfer_3DDist()
    
    # Metrics storage
    all_cd_coarse = []
    all_cd_fine1 = []
    all_cd_fine2 = []
    
    per_category_cd = {}
    
    # Create output directory for saving results
    output_dir = os.path.join(cfg.DIR.OUT_PATH, 'evaluation_results')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    logging.info(f'Starting evaluation on {len(test_data_loader)} samples...')
    
    # Evaluation loop
    with tqdm(test_data_loader, desc='Evaluating') as t:
        for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(t):
            taxonomy_id = taxonomy_ids[0]
            model_id = model_ids[0]
            
            # Move data to GPU
            for k, v in data.items():
                data[k] = utils.helpers.var_or_cuda(v)
            
            partial = data['partial_cloud']
            gt = data['gtcloud']
            
            # Render multi-view depth images
            partial_depth = render.get_img(partial)
            
            # Forward pass
            with torch.no_grad():
                pcds_pred = model(partial, partial_depth)
            
            coarse_pred = pcds_pred[0]
            fine1_pred = pcds_pred[1]
            fine2_pred = pcds_pred[2]
            
            # Compute Chamfer Distances
            # Coarse
            cd_coarse_forward = chamfer_distance(coarse_pred.contiguous(), gt.contiguous())[0]
            cd_coarse_backward = chamfer_distance(gt.contiguous(), coarse_pred.contiguous())[0]
            cd_coarse = ((cd_coarse_forward.mean() + cd_coarse_backward.mean()) / 2).item() * 1e3
            
            # Fine1
            cd_fine1_forward = chamfer_distance(fine1_pred.contiguous(), gt.contiguous())[0]
            cd_fine1_backward = chamfer_distance(gt.contiguous(), fine1_pred.contiguous())[0]
            cd_fine1 = ((cd_fine1_forward.mean() + cd_fine1_backward.mean()) / 2).item() * 1e3
            
            # Fine2 (final)
            cd_fine2_forward = chamfer_distance(fine2_pred.contiguous(), gt.contiguous())[0]
            cd_fine2_backward = chamfer_distance(gt.contiguous(), fine2_pred.contiguous())[0]
            cd_fine2 = ((cd_fine2_forward.mean() + cd_fine2_backward.mean()) / 2).item() * 1e3
            
            # Store results
            all_cd_coarse.append(cd_coarse)
            all_cd_fine1.append(cd_fine1)
            all_cd_fine2.append(cd_fine2)
            
            # Per-category statistics
            if taxonomy_id not in per_category_cd:
                per_category_cd[taxonomy_id] = {
                    'coarse': [],
                    'fine1': [],
                    'fine2': []
                }
            per_category_cd[taxonomy_id]['coarse'].append(cd_coarse)
            per_category_cd[taxonomy_id]['fine1'].append(cd_fine1)
            per_category_cd[taxonomy_id]['fine2'].append(cd_fine2)
            
            # Update progress bar
            t.set_postfix(
                coarse=f'{cd_coarse:.4f}',
                fine1=f'{cd_fine1:.4f}',
                fine2=f'{cd_fine2:.4f}'
            )
    
    # Compute overall statistics
    import numpy as np
    avg_cd_coarse = np.mean(all_cd_coarse)
    avg_cd_fine1 = np.mean(all_cd_fine1)
    avg_cd_fine2 = np.mean(all_cd_fine2)
    
    std_cd_coarse = np.std(all_cd_coarse)
    std_cd_fine1 = np.std(all_cd_fine1)
    std_cd_fine2 = np.std(all_cd_fine2)
    
    # Print overall results
    logging.info('=' * 80)
    logging.info('OVERALL EVALUATION RESULTS')
    logging.info('=' * 80)
    logging.info(f'Coarse CD:  {avg_cd_coarse:.4f} ± {std_cd_coarse:.4f}')
    logging.info(f'Fine1 CD:   {avg_cd_fine1:.4f} ± {std_cd_fine1:.4f}')
    logging.info(f'Fine2 CD:   {avg_cd_fine2:.4f} ± {std_cd_fine2:.4f}')
    logging.info('=' * 80)
    
    # Print per-category results
    logging.info('\nPER-CATEGORY RESULTS:')
    logging.info('-' * 80)
    for category, cds in sorted(per_category_cd.items()):
        cat_coarse = np.mean(cds['coarse'])
        cat_fine1 = np.mean(cds['fine1'])
        cat_fine2 = np.mean(cds['fine2'])
        logging.info(
            f'{category:20s} | '
            f'Coarse: {cat_coarse:6.4f} | '
            f'Fine1: {cat_fine1:6.4f} | '
            f'Fine2: {cat_fine2:6.4f}'
        )
    logging.info('-' * 80)
    
    # Save results to file
    results_file = os.path.join(output_dir, 'evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write('GeoSpecNet Evaluation Results\n')
        f.write('=' * 80 + '\n')
        f.write(f'Coarse CD:  {avg_cd_coarse:.4f} ± {std_cd_coarse:.4f}\n')
        f.write(f'Fine1 CD:   {avg_cd_fine1:.4f} ± {std_cd_fine1:.4f}\n')
        f.write(f'Fine2 CD:   {avg_cd_fine2:.4f} ± {std_cd_fine2:.4f}\n')
        f.write('=' * 80 + '\n\n')
        f.write('Per-Category Results:\n')
        f.write('-' * 80 + '\n')
        for category, cds in sorted(per_category_cd.items()):
            cat_coarse = np.mean(cds['coarse'])
            cat_fine1 = np.mean(cds['fine1'])
            cat_fine2 = np.mean(cds['fine2'])
            f.write(
                f'{category:20s} | '
                f'Coarse: {cat_coarse:6.4f} | '
                f'Fine1: {cat_fine1:6.4f} | '
                f'Fine2: {cat_fine2:6.4f}\n'
            )
        f.write('-' * 80 + '\n')
    
    logging.info(f'\nResults saved to {results_file}')
    
    return avg_cd_fine2
