#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/6/2 15:59
# @Author : jj.wang

import json
import pathlib
import time
import io
import os
import base64
import torch
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from natsort import natsorted


def parse_config(config: dict) -> dict:
    import anyconfig
    base_file_list = config.pop('base')
    base_config = {}
    for base_file in base_file_list:
        tmp_config = anyconfig.load(open(base_file, 'rb'))
        if 'base' in tmp_config:
            tmp_config = parse_config(tmp_config)
        anyconfig.merge(tmp_config, base_config)
        base_config = tmp_config
    anyconfig.merge(base_config, config)
    return base_config


def show_img(imgs: np.ndarray, title='img'):
    color = (len(imgs.shape) == 3 and imgs.shape[-1] == 3)
    imgs = np.expand_dims(imgs, axis=0)
    for i, img in enumerate(imgs):
        plt.figure()
        plt.title('{}_{}'.format(title, i))
        plt.imshow(img, cmap=None if color else 'gray')
    plt.show()


def setup_logger(log_file_path: str = None):
    import logging
    logging._warn_preinit_stderr = 0
    logger = logging.getLogger('tianrang-ocr')
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if log_file_path is not None:
        file_handle = logging.FileHandler(log_file_path)
        file_handle.setFormatter(formatter)
        logger.addHandler(file_handle)
    logger.setLevel(logging.DEBUG)
    return logger


def draw_bbox(img_path, result, color=(255, 0, 0), thickness=2):
    if isinstance(img_path, str):
        img_path = cv2.imread(img_path)
        # img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    img_path = img_path.copy()
    for point in result:
        point = point.astype(int)
        cv2.polylines(img_path, [point], True, color, thickness)
    return img_path


def cv2ImgAddText(img, text, pos, textColor=(255, 0, 0), textSize=12):
    if (isinstance(img, np.ndarray)):  # detect opencv format or not
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype('utils/STHeiti Medium.ttc', textSize, encoding="utf-8")
    draw.text(pos, text, textColor, font=fontText)

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def get_file_list(folder_path: str, p_postfix: list = None, sub_dir: bool = True) -> list:
    """
    获取所给文件目录里的指定后缀的文件,读取文件列表目前使用的是 os.walk 和 os.listdir ，这两个目前比 pathlib 快很多
    :param filder_path: 文件夹名称
    :param p_postfix: 文件后缀,如果为 [.*]将返回全部文件
    :param sub_dir: 是否搜索子文件夹
    :return: 获取到的指定类型的文件列表
    """
    assert os.path.exists(folder_path) and os.path.isdir(folder_path)
    if p_postfix is None:
        p_postfix = ['.jpg']
    if isinstance(p_postfix, str):
        p_postfix = [p_postfix]
    file_list = [x for x in glob.glob(folder_path + '/**/*.*', recursive=True) if
                 os.path.splitext(x)[-1] in p_postfix or '.*' in p_postfix]
    return natsorted(file_list)


def save_result(result_path, box_list, score_list, is_output_polygon):
    if is_output_polygon:
        with open(result_path, 'wt') as res:
            for i, box in enumerate(box_list):
                box = box.reshape(-1).tolist()
                result = ",".join([str(int(x)) for x in box])
                score = score_list[i]
                res.write(result + ',' + str(score) + "\n")
    else:
        with open(result_path, 'wt') as res:
            for i, box in enumerate(box_list):
                score = score_list[i]
                box = box.reshape(-1).tolist()
                result = ",".join([str(int(x)) for x in box])
                res.write(result + ',' + str(score) + "\n")


def cal_text_score(texts, gt_texts, training_masks, running_metric_text, thred=0.5):
    training_masks = training_masks.data.cpu().numpy()
    pred_text = texts.data.cpu().numpy() * training_masks
    pred_text[pred_text <= thred] = 0
    pred_text[pred_text > thred] = 1
    pred_text = pred_text.astype(np.int32)
    gt_text = gt_texts.data.cpu().numpy() * training_masks
    gt_text = gt_text.astype(np.int32)
    running_metric_text.update(gt_text, pred_text)
    score_text, _ = running_metric_text.get_scores()
    return score_text


def filter_params_assign_lr(model, filter_dict):
    '''
    不同的参数分配不同的学习率
    :param model: 模型对象
    :param filter_dict: {module_name: lr}， {'preprocess': 0.1}，表示preprocess部分学习率为0.1
    :return: list
    '''
    params_list = []
    name_lsit = []
    for name, p in model.named_parameters():
        name = name.replace('.module', '').split('.')[0]
        if name in name_lsit:
            continue
        if filter_dict.get(name, False):
            tensor = eval('model.' + name + '.parameters()')
            params_list.append({'params': tensor, 'lr': filter_dict[name]})
        else:
            tensor = eval('model.' + name + '.parameters()')
            params_list.append({'params': tensor})
        name_lsit.append(name)
    return params_list


def read_base64(data, mode):
    img_data = base64.b64decode(data)
    if mode == 'opencv':
        nparr = np.fromstring(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.COLOR_RGB2BGR)
    elif mode == 'pil':
        image = io.BytesIO(img_data)
        img = Image.open(image)
    else:
        raise ValueError
    return img


def makedirs(paths: list):
    for i in paths:
        if not os.path.isdir(i):
            os.makedirs(i)


def export_model(model_path, output):
    checkpoint = torch.load(model_path, map_location='cpu')
    state = {'state_dict':checkpoint['state_dict'], 'config':checkpoint['config'], 'metrics':checkpoint['metrics']}
    torch.save(state, output)
    print(f'export {model_path} to {output}')