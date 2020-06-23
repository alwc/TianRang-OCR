#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/6/2 10:53
# @Author : jj.wang

import sys, os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
import torch.nn as nn
import backbone, neck, head, preprocess


class RecModel(nn.Module):
    def __init__(self, config):
        super(RecModel, self).__init__()
        # 前处理，目前仅支持tps
        if config.get('preprocess', None):
            self.preprocess = getattr(preprocess, config['preprocess']['type'])(**config['preprocess']['args'])
        # backbone需要algorithm参数，去调整stride；需要输出的类别参数，lprnet没有neck和head
        config['backbone']['args']['algorithm'] = config['algorithm']
        config['backbone']['args']['output_size'] = config['num_class']
        # backbone
        self.backnone = getattr(backbone, config['backbone']['type'])(**config['backbone']['args'])

        # neck 需要backbone的属性output_channel
        if config['neck']:
            config['neck']['args']['input_size'] = self.backnone.output_channel
            self.neck = getattr(neck, config['neck']['type'])(**config['neck']['args'])
        # head
        if config['head']:
            config['head']['args']['output_size'] = config['num_class']
            # 在没有neck的时候head以backnone.output_channel作为input channel，否则以neck.output_channel作为input channel
            config['head']['args']['input_size'] = self.neck.output_channel if self.neck.output_channel is not None else self.backnone.output_channel
            self.head = getattr(head, config['head']['type'])(**config['head']['args'])
        self.name = 'rec'

    def forward(self, x):
        if getattr(self, 'preprocess', None):
            x = self.preprocess(x)
        x = self.backnone(x)
        if getattr(self, 'neck', None):
            if isinstance(x, (list, tuple)):
                x = self.neck(x[-1])
            else:
                x = self.neck(x)
        if getattr(self, 'head', None):
            x = self.head(x)
        return x


if __name__ == '__main__':
    import torch
    from addict import Dict
    config = { 'algorithm': 'rec',
              'backbone':{'type':'shufflenet_v2_x0_5', 'args':{'pretrained': False}},
              'neck': {'type':'Squeeze', 'args':{'mode': 'max_pool'}},
              'head': {'type':'CTC_Head'},
              'num_class':80}
    config = Dict(config)
    rec = RecModel(config)
    device = torch.device('cpu')
    x = torch.zeros(1, 3, 32, 120).to(device)
    y = rec(x)
    print(y.shape)