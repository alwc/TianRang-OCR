#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/6/2 15:30
# @Author : jj.wang

import json
import pathlib
import time
import os
import glob
from natsort import natsorted
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance as dist


def get_datalist(train_data_path):
    """
    获取训练和验证的数据list
    :param train_data_path: 训练的dataset文件列表，每个文件内以如下格式存储 ‘path/to/img\tlabel’
    :return:
    """
    train_data = []
    for p in train_data_path:
        with open(p, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').replace('.jpg ', '.jpg\t').split('\t')
                if len(line) > 1:
                    img_path = pathlib.Path(line[0].strip(' '))
                    label_path = pathlib.Path(line[1].strip(' '))
                    if img_path.exists() and img_path.stat().st_size > 0 and label_path.exists() and label_path.stat().st_size > 0:
                        train_data.append((str(img_path), str(label_path)))
    return train_data



def expand_polygon(polygon):
    """
    对只有一个字符的框进行扩充
    """
    (x, y), (w, h), angle = cv2.minAreaRect(np.float32(polygon))
    if angle < -45:
        w, h = h, w
        angle += 90
    new_w = w + h
    box = ((x, y), (new_w, h), angle)
    points = cv2.boxPoints(box)
    return order_points_clockwise(points)


def order_points_clockwise(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _order_points(pts):
    # 根据x坐标对点进行排序
    """
    ---------------------
    作者：Tong_T
    来源：CSDN
    原文：https://blog.csdn.net/Tong_T/article/details/81907132
    版权声明：本文为博主原创文章，转载请附上博文链接！
    """
    x_sorted = pts[np.argsort(pts[:, 0]), :]

    # 从排序中获取最左侧和最右侧的点
    # x坐标点
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]

    # 现在，根据它们的y坐标对最左边的坐标进行排序，这样我们就可以分别抓住左上角和左下角
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    (tl, bl) = left_most

    # 现在我们有了左上角坐标，用它作为锚来计算左上角和右上角之间的欧氏距离;
    # 根据毕达哥拉斯定理，距离最大的点将是我们的右下角
    distance = dist.cdist(tl[np.newaxis], right_most, "euclidean")[0]
    (br, tr) = right_most[np.argsort(distance)[::-1], :]

    # 返回左上角，右上角，右下角和左下角的坐标
    return np.array([tl, tr, br, bl], dtype="float32")


def load(file_path: str):
    file_path = pathlib.Path(file_path)
    func_dict = {'.txt': _load_txt, '.json': _load_json, '.list': _load_txt}
    assert file_path.suffix in func_dict
    return func_dict[file_path.suffix](file_path)
