#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/6/1 17:53
# @Author : jj.wang

import torch.nn as nn


class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output


class BiLSTM(nn.Module):
    def __init__(self, mode, input_size, hidden_size, **kwargs):
        super(BiLSTM, self).__init__()
        self.layer = nn.Sequential(Squeeze(mode),
            BidirectionalLSTM(input_size, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size))
        self.output_channel = hidden_size

    def forward(self, x):
        x = self.layer(x)
        return x



class Squeeze(nn.Module):
    def __init__(self, mode, **kwargs):
        super(Squeeze, self).__init__()
        # 卷积squeeze待实现
        mode_dict = {'conv': None, 'max_pool':nn.AdaptiveMaxPool2d, 'avg_pool': nn.AdaptiveAvgPool2d}
        self.squeeze = mode_dict[mode]((None, 1))
        self.output_channel = None


    def forward(self, x):
        x = self.squeeze(x.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        x = x.squeeze(3)
        return x
