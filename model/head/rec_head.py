#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/6/1 17:57
# @Author : jj.wang

import torch.nn as nn


class CTC_Head(nn.Module):
    def __init__(self, input_size, output_size, **kwargs):
        super(CTC_Head, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc = nn.Linear(self.input_size, self.output_size)
        if 'dropout_rate' not in locals().keys():
            dropout_rate = 0
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc(x)
        x = self.dropout(x)
        return x

