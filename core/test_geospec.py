import logging
import torch
import utils.data_loaders
import utils.helpers
from tqdm import tqdm
from utils.average_meter import AverageMeter
from utils.loss_utils import calc_cd, calc_dcd
from models.GeoSpecNet import Model as GeoSpecModel
from models.model_utils import PCViews


def test_net(cfg, epoch_idx=-1, test_data_loader=None, test_writer=None, model=None):
    torch.backends.cudnn.benchmark = True

    if test_data_loader is None:
        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
            utils.data_loaders.DatasetSubset.TEST),
            batch_size=1,
            num_workers=cfg.CONST.NUM_WORKERS//2,
            collate_fn=utils.data_loaders.collate_fn,
            pin_memory=True,
            shuffle=False)

    if model is None:
        model = GeoSpecModel(cfg)
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()
        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        model.load_state_dict(checkpoint['G'] if 'G' in checkpoint else checkpoint['model'])

    model.eval()

    n_samples = len(test_data_loader)
    test_losses = AverageMeter(['CD','DCD','F1'])
    test_metrics = AverageMeter(['CD','DCD','F1'])
    category_metrics = dict()
    render = PCViews(TRANS=-cfg.NETWORK.view_distance, RESOLUTION=224)

    with tqdm(test_data_loader) as t:
        for model_idx, (taxonomy_id, model_id, data) in enumerate(t):
            taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
            model_id = model_id[0]

            with torch.no_grad():
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)
                partial = data['partial_cloud']
                gt = data['gtcloud']
                partial_depth = torch.unsqueeze(render.get_img(partial),1)
                pcds_pred = model(partial.contiguous(), partial_depth)
                cdl1,cdl2,f1 = calc_cd(pcds_pred[-1],gt,calc_f1=True)
                dcd,_,_ = calc_dcd(pcds_pred[-1],gt)

                cd = cdl1.mean().item() * 1e3
                dcd = dcd.mean().item()
                f1 = f1.mean().item()

                _metrics = [cd, dcd, f1]
                test_losses.update([cd, dcd, f1])
                test_metrics.update(_metrics)
                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(['CD','DCD','F1'])
                category_metrics[taxonomy_id].update(_metrics)

                t.set_description('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                             (model_idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()
                                                                                ], ['%.4f' % m for m in _metrics]))

    print('Overall', end='\t\t\t')
    for value in test_metrics.avg():
        print('%.4f' % value, end='\t')
    print('\n')
    return test_losses.avg(0)
