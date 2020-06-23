#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/6/1 17:32
# @Author : jj.wang


from . import architecture, losses, converter, post_processing


def get_model(config):
    _model = getattr(architecture, config['algorithm'])(config)
    return _model


def get_loss(config):
    if 'args' not in config:
        args = {}
    else:
        args = config['args']
    return getattr(losses, config['type'])(**args)

def get_converter(config):
    return getattr(converter, config['type'])(**config['args'])


def get_post_processing(config):
    try:
        cls = getattr(post_processing, config['type'])(**config['args'])
        return cls
    except:
        return None