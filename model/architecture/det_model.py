#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/6/5 10:42
# @Author : jj.wang

import sys, os
import torch.nn.functional as F
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
import torch.nn as nn
import backbone, neck, head

backbone_channel_dict = {
    'resnet18': [64, 128, 256, 512],
    'deformable_resnet18': [64, 128, 256, 512],
    'resnet34':  [64, 128, 256, 512],
    'resnet50':  [256, 512, 1024, 2048],
    'deformable_resnet50': [256, 512, 1024, 2048],
    'resnet101':  [256, 512, 1024, 2048],
    'resnet152': [256, 512, 1024, 2048],
    'shufflenetv2': [24, 116, 232, 464],
    'shufflenet_v2_x0_5': [24, 48, 96, 192]}



class DetModel(nn.Module):
    def __init__(self, config):
        super(DetModel, self).__init__()
        config['backbone']['args']['algorithm'] = config['algorithm']
        self.backnone = getattr(backbone, config['backbone']['type'])(**config['backbone']['args'])
        config['neck']['args']['backbone_out_channels'] = backbone_channel_dict[config['backbone']['type']]

        self.neck = getattr(neck, config['neck']['type'])(**config['neck']['args'])
        config['head']['args']['in_channels'] = self.neck.out_channels
        self.head = getattr(head, config['head']['type'])(**config['head']['args'])
        self.name = 'det'

    def forward(self, x):
        _, _, H, W = x.size()
        x = self.backnone(x)
        x = self.neck(x)
        x = self.head(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x


if __name__ == '__main__':
    import torch
    config = {'algorithm': 'rec',
              'backbone': {'type': 'resnet18', 'args': {'pretrained': False}},
              'neck': {'type': 'FPN', 'args': {'inner_channels': 256}},
              'head': {'type': 'DBHead', 'args': {'k': 50}}}
    det = DetModel(config)
    device = torch.device('cpu')
    x = torch.zeros(1, 3, 640, 640).to(device)
    y = det(x)
    print(y.shape)
