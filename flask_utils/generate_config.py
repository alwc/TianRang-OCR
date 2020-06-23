#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/6/22 10:33
# @Author : jj.wang

import json
from easydict import EasyDict as edict


def main():
    cfg = edict()
    cfg.host = '0.0.0.0'
    cfg.port = 8080
    cfg.device_id = None
    cfg.det_model_path = 'weights/det.pth'
    cfg.det_thre = 0.7
    cfg.det_short_size = 416
    cfg.rec_model_path = 'weights/rec.pth'
    cfg.rec_crop_ratio = 1.05
    cfg.file_record = True
    cfg.log_dir = 'output/log/lpr.log'
    cfg.debug = False
    cfg.use_hyperlpr = False

    return cfg


if __name__ == '__main__':
    cfg = main()
    with open('config.json', 'w') as f:
        f.write(json.dumps(cfg, sort_keys=True, indent=4, separators=(',', ': ')))
        print('generate config complete')
