#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/6/8 13:52
# @Author : jj.wang


import os
import sys

project = 'tianrang-ocr'  # 工作项目根目录
sys.path.append(os.getcwd().split(project)[0] + project)
import time
import cv2
import torch
import numpy as np
from model import get_model, get_converter
from data_loader import ResizeNormalize
from torchvision import transforms
import torch.nn.functional as F



class RecModel:
    def __init__(self, model_path, gpu_id=None):
        '''
        初始化pytorch模型
        :param model_path: 模型地址(可以是模型的参数或者参数和计算图一起保存的文件)
        :param gpu_id: 在哪一块gpu上运行
        '''
        self.gpu_id = gpu_id

        if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:%s" % self.gpu_id)
        else:
            self.device = torch.device("cpu")
        print('device:', self.device)
        checkpoint = torch.load(model_path, map_location=self.device)

        config = checkpoint['config']
        # print(config)
        self.model = get_model(config['arch'])
        # 临时
        # config['converter']['args']['character'] = 'license_plate'
        self.h = config['dataset']['validate']['loader']['collate_fn']['args']['imgH']
        self.w = config['dataset']['validate']['loader']['collate_fn']['args']['imgW']
        keep_ratio_with_pad = config['dataset']['validate']['loader']['collate_fn']['args']['keep_ratio_with_pad']
        if keep_ratio_with_pad:
            self.transform = self.resize_image
        else:
            self.transform = ResizeNormalize((self.w, self.h))
        self.converter = get_converter(config['converter'])
        self.img_mode = config['dataset']['train']['dataset']['args']['img_mode']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()


    def predict(self, data):
        '''
        对传入的图像进行预测，支持图像地址,opecv 读取图片，偏慢
        :param img_path: 图像地址
        :param is_numpy:
        :return:
        '''
        if isinstance(data, str):
            assert os.path.exists(data), 'file is not exists'
            img = cv2.imread(data, 1 if self.img_mode != 'GRAY' else 0)
            if self.img_mode == 'RGB':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif type(data) is np.ndarray:
            img = data.copy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError
        tensor = self.transform(img)
        tensor = tensor.unsqueeze_(0)
        tensor = tensor.to(self.device)
        with torch.no_grad():
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize(self.device)
            start = time.time()
            preds = self.model(tensor)
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize(self.device)
            preds_prob = F.softmax(preds, dim=2)
            preds_prob, pred_index = preds_prob.max(dim=2)
            pred_str = self.converter.decode(pred_index)[0]
            confidence = preds_prob[0].min().cpu().numpy().tolist()
        return pred_str, confidence, time.time() - start

    def resize_image(self, img, h=None):
        height, width, _ = img.shape
        if h is None:
            ratio = self.h/height
        else:
            ratio = h/height
        img = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
        method = transforms.ToTensor()
        img = method(img)
        img.sub_(0.5).div_(0.5)
        return img

def init_args():
    import argparse
    parser = argparse.ArgumentParser(description='tianrang-ocr')
    parser.add_argument('--model_path', default='model_best.pth', type=str)
    parser.add_argument('--input_folder', default='./test/input', type=str, help='img path for predict')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    import pathlib
    from tqdm import tqdm
    from utils.utils import get_file_list
    args = init_args()
    print(args)
    # 初始化网络
    model = RecModel(args.model_path, gpu_id=0)
    img_folder = pathlib.Path(args.input_folder)
    for img_path in tqdm(get_file_list(args.input_folder, p_postfix=['.jpg'])):
        preds, preds_prob, t = model.predict(img_path)
        print(f'{img_path} predict {preds}')

