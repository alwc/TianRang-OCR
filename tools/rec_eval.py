#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/6/8 16:00
# @Author : jj.wang


import os
import sys

project = 'tianrang-ocr'  # 工作项目根目录
sys.path.append(os.getcwd().split(project)[0] + project)

import argparse
import time
import torch
from tqdm.auto import tqdm
import torch.nn.functional as F


class EVAL():
    def __init__(self, model_path, gpu_id=0):
        from model import get_model, get_loss, get_converter
        from data_loader import get_dataloader
        from metric import get_metric
        self.gpu_id = gpu_id

        if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:%s" % self.gpu_id)
        else:
            self.device = torch.device("cpu")
        print('device:', self.device)
        checkpoint = torch.load(model_path, map_location=self.device)

        config = checkpoint['config']
        self.config = config
        self.model = get_model(config['arch'])
        # config['converter']['args']['character'] = 'license_plate'
        self.converter = get_converter(config['converter'])
        # self.post_process = get_post_processing(config['post_processing'])
        self.img_mode = config['dataset']['train']['dataset']['args']['img_mode']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()
        self.metric = get_metric(config['metric'])
        # config['dataset']['validate']['loader']['num_workers'] = 8
        # config['dataset']['validate']['dataset']['args']['pre_processes'] = [{'type': 'CropWordBox', 'args': [1, 1.2]}]
        if args.img_path is not None:
            config['dataset']['validate']['dataset']['args']['data_path'] = [args.img_path]
        self.validate_loader = get_dataloader(config['dataset']['validate'], config['distributed'])

    def eval(self):
        self.model.eval()
        total_frame = 0.0
        total_time = 0.0
        self.metric.reset()
        for i, batch in tqdm(enumerate(self.validate_loader), total=len(self.validate_loader), desc='test model'):
            with torch.no_grad():
                # 数据进行转换和丢到gpu
                for key, value in batch.items():
                    if value is not None:
                        if isinstance(value, torch.Tensor):
                            batch[key] = value.to(self.device)
                start = time.time()
                preds = self.model(batch['img'])
                preds_prob = F.softmax(preds, dim=2)
                preds_prob, pred_index = preds_prob.max(dim=2)
                pred_str = self.converter.decode(pred_index)
                self.metric.measure(pred_str, batch['labels'], preds_prob)
                total_frame += batch['img'].size()[0]
                total_time += time.time() - start
        acc = self.metric.avg['acc']['true']
        edit = self.metric.avg['edit']
        print('FPS:{}'.format(total_frame / total_time))

        return acc, edit


def init_args():
    parser = argparse.ArgumentParser(description='tianrang-ocr')
    parser.add_argument('--model_path', required=True, type=str)
    parser.add_argument('--img_path', default=None, type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = init_args()
    eval = EVAL(args.model_path)
    result = eval.eval()
    print(result)
