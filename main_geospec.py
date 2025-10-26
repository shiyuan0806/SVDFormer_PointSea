import argparse
import logging
import os
import numpy as np
import torch
from pprint import pprint
from config_geospec import cfg
from core.train_geospec import train_net
from core.test_geospec import test_net

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_args_from_command_line():
    parser = argparse.ArgumentParser(description='GeoSpecNet training/testing')
    parser.add_argument('--test', dest='test', help='Test neural networks', action='store_true')
    parser.add_argument('--inference', dest='inference', help='Inference for benchmark', action='store_true')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--run_id', type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    args = get_args_from_command_line()
    if args.run_id:
        cfg.TRAIN.RUN_ID = args.run_id
    if args.weights:
        cfg.CONST.WEIGHTS = args.weights

    print('cuda available ', torch.cuda.is_available())
    print('Use config:')
    pprint(cfg)

    if not args.test and not args.inference:
        train_net(cfg)
    else:
        if cfg.CONST.get('WEIGHTS', None) is None:
            raise Exception('Please specify the path to checkpoint in the configuration file or via --weights!')
        test_net(cfg)


if __name__ == '__main__':
    seed = 1
    set_seed(seed)
    logging.basicConfig(format='[%(levelname)s] %(asctime)s %(message)s', level=logging.DEBUG)
    main()
