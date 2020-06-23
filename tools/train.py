#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/6/3 16:15
# @Author : jj.wang
import anyconfig
import argparse
from addict import Dict


def init_args():
    parser = argparse.ArgumentParser(description='DBNet.pytorch')
    parser.add_argument('--config_file', default='config/open_dataset_dcn_resnet50_FPN_DBhead_polyLR.yaml', type=str)
    parser.add_argument('--local_rank', dest='local_rank', default=0, type=int, help='Use distributed training')

    args = parser.parse_args()
    return args


def main(config):
    import torch
    from model import get_model, get_loss, get_converter, get_post_processing
    from metric import get_metric
    from data_loader import get_dataloader
    from tools.rec_trainer import RecTrainer as rec
    from tools.det_trainer import DetTrainer as det
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=torch.cuda.device_count(),
                                             rank=args.local_rank)
        config['distributed'] = True
    else:
        config['distributed'] = False
    config['local_rank'] = args.local_rank
    train_loader = get_dataloader(config['dataset']['train'], config['distributed'])
    assert train_loader is not None
    if 'validate' in config['dataset']:
        validate_loader = get_dataloader(config['dataset']['validate'], False)
    else:
        validate_loader = None

    criterion = get_loss(config['loss']).cuda()

    if config.get('post_processing', None):
        post_p = get_post_processing(config['post_processing'])
    else:
        post_p = None

    metric = get_metric(config['metric'])

    if config['arch']['algorithm'] == 'rec':
        converter = get_converter(config['converter'])
        config['arch']['num_class'] = len(converter.character)
        model = get_model(config['arch'])
    else:
        converter = None
        model = get_model(config['arch'])

    trainer = eval(config['arch']['algorithm'])(config=config,
                         model=model,
                         criterion=criterion,
                         train_loader=train_loader,
                         post_process=post_p,
                         metric=metric,
                         validate_loader=validate_loader,
                         converter=converter)
    trainer.train()


if __name__ == '__main__':
    import os
    import sys

    project = 'tianrang-ocr'  # 工作项目根目录
    sys.path.append(os.getcwd().split(project)[0] + project)

    from utils.utils import parse_config

    args = init_args()
    assert os.path.exists(args.config_file)
    config = anyconfig.load(open(args.config_file, 'rb'))
    if 'base' in config:
        config = parse_config(config)
    mapping = Dict(config)
    main(mapping)
