#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/6/4 15:23
# @Author : jj.wang


from .icdar2015 import QuadMetric
from .rec_metrics import STRMeters


def get_metric(config):
    if 'args' not in config:
        args = {}
    else:
        args = config['args']
    if isinstance(args, dict):
        cls = eval(config['type'])(**args)
    else:
        cls = eval(config['type'])(args)
    return cls