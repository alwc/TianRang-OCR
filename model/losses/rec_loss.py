#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/6/3 15:38
# @Author : jj.wang

import torch
from torch import nn


class CTCLoss(nn.Module):

    def __init__(self, zero_infinity=True, blank=0, reduction='mean'):
        super(CTCLoss, self).__init__()
        self.criterion = nn.CTCLoss(zero_infinity=zero_infinity,
                                    blank=blank,
                                    reduction=reduction)

    def forward(self, pred, target, length, batch_size):
        pred = pred.log_softmax(2)
        preds_size = torch.IntTensor([pred.size(1)] * batch_size).to('cuda')
        pred_ = pred.permute(1, 0, 2)
        torch.backends.cudnn.enabled = False
        cost = self.criterion(pred_, target, preds_size, length)
        torch.backends.cudnn.enabled = True
        return {'loss': cost}
