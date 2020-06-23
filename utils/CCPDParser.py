#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/5/22 11:47
# @Author : jj.wang

import os
import cv2
import glob
import random
import pprint
from multiprocessing.pool import ThreadPool
from queue import Queue, Empty

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']



def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    a = [0, 0, 0, 0]
    a[0] = 0 if box[0] < 0 else box[0]
    a[1] = box[1] if box[1] < size[0] else size[0]
    a[2] = 0 if box[2] < 0 else box[2]
    a[3] = box[3] if box[3] < size[1] else size[1]
    x = (a[0] + a[1]) / 2.0 - 1
    y = (a[2] + a[3]) / 2.0 - 1
    w = a[1] - a[0]
    h = a[3] - a[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


class Ann(object):
    def __init__(self):
        pass


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


class CCPDParser():
    def __init__(self, root_dir, trainval_split_ratio=0.1, train_val=['ccpd_base'],
                 test=['ccpd_challenge', 'ccpd_fn', 'ccpd_rotate', 'ccpd_weather', 'ccpd_db', 'ccpd_tilt', 'ccpd_blur']):
        self.ccpd_dir = root_dir
        # self.parser_annotation('025-95_113-154＆383_386＆473-386＆473_177＆454_154＆383_363＆402-0_0_22_27_27_33_16-37-15')
        self.trainval_img_list = self.get_image_path([os.path.join(root_dir, i) for i in train_val])
        self.test_img_list = self.get_image_path([os.path.join(root_dir, i) for i in test])
        print(f'generated test img list')
        self.train_img_list, self.val_img_list = self.split_train_val(trainval_split_ratio)
        print(f'generated train img list and val img list')

    def get_image_path(self, paths):
        if len(paths) == None:
            raise ValueError
        else:
            img_list = []
            for p in paths:
                img_list.extend(glob.glob(os.path.join(p, '*.jpg')))
            return img_list

    def split_train_val(self, ratio):
        if int(ratio * len(self.trainval_img_list)) > 0:
            random.seed(2020)
            train_img_list = []
            val_img_list = random.sample(self.trainval_img_list, int(ratio * len(self.trainval_img_list)))
            for i in self.trainval_img_list:
                if i not in val_img_list:
                    train_img_list.append(i)
        else:
            train_img_list = self.trainval_img_list
            val_img_list = []
        return train_img_list, val_img_list

    def parser_annotation(self, imgname):
        '''
        CCPD注释嵌入在文件名中。

        样本图像名称为“025-95_113-154＆383_386＆473-386＆473_177＆454_154＆383_363＆402-0_0_22_27_27_33_16-37-15.jpg”。每个名称可以分为七个字段。这些字段解释如下。
        面积：牌照面积与整个图片区域的面积比。
        倾斜度：水平倾斜程度和垂直倾斜度。
        边界框坐标：左上和右下顶点的坐标。
        四个顶点位置：整个图像中LP的四个顶点的精确（x，y）坐标。这些坐标从右下角顶点开始。
        车牌号：CCPD中的每个图像只有一个LP。每个LP号码由一个汉字，一个字母和五个字母或数字组成。有效的中文车牌由七个字符组成：省（1个字符），字母（1个字符），字母+数字（5个字符）。“ 0_0_22_27_27_33_16”是每个字符的索引。这三个数组定义如下。每个数组的最后一个字符是字母O，而不是数字0。我们将O用作“无字符”的符号，因为中文车牌字符中没有O。
        亮度：牌照区域的亮度。
        模糊度：车牌区域的模糊度
        :param imgname: 注释信息
        :return: label [x1,y1,x2,y2,x3,y3,x4,y4,label]
        '''
        imgname_split = imgname.split('-')
        # 水平倾斜度和垂直倾斜度
        horizontal_inclination, vertical_inclination = [int(i) for i in imgname_split[1].split('_')]
        # 正矩形坐标
        x1y1, x2y2 = imgname_split[2].split('_')
        x1, y1 = [int(i) for i in x1y1.split('＆').split('&')] if '＆' in x1y1 else [int(i) for i in x1y1.split('&')]
        x2, y2 = [int(i) for i in x1y1.split('＆').split('&')] if '＆' in x2y2 else [int(i) for i in x2y2.split('&')]
        rec = [x1, y1, x2, y2]
        # 四边形
        xy = imgname_split[3].split('_')
        polygon = []
        for i in xy:
            polygon.extend(i.split('＆') if '＆' in i else i.split('&'))
        polygon = [int(i) for i in polygon]
        # 车牌注释
        pre_label = imgname_split[4].split('_')
        lb = ''
        lb += provinces[int(pre_label[0])]
        lb += alphabets[int(pre_label[1])]
        for label in pre_label[2:]:
            lb += ads[int(label)]
        # 亮度
        britness = int(imgname_split[5])
        # 模糊度
        blur = int(imgname_split[6])
        annotation = Ann()
        annotation.horizontal_inclination = horizontal_inclination
        annotation.vertical_inclination = vertical_inclination
        annotation.rec = rec
        annotation.polygon = polygon
        annotation.label = lb
        annotation.britness = britness
        annotation.blur = blur
        # annotation.img_path = imgname + '.jpg'
        return annotation

    def get_name_suffix(self, path):
        basename = os.path.basename(path)
        name, suffix = os.path.splitext(basename)
        return name, suffix

    def generate_icdar(self, img_path, queue, output):
        '''
        生成icdar txt
        :param img_path:
        :param queue:
        :param output:
        :return:
        '''
        imgname, suffix = self.get_name_suffix(img_path)
        annotation_object = self.parser_annotation(imgname)
        str_label = ','.join([str(i) for i in annotation_object.polygon] + [annotation_object.label])
        path = os.path.join(output, imgname + '.txt')
        with open(path, 'w') as f:
            f.write(str_label)
        # file.write(img_path + '\t' + os.path.join(output, imgname+'.txt\n'))
        line_str = img_path + '\t' + os.path.join(output, imgname + '.txt') + '\n'
        queue.put(line_str)

    def generate_yolo(self, img_path, queue, output):
        '''
        生成yolo txt
        :param img_path: 图片路径
        :param queue: 线程队列
        :param output: 转化的label txt存放的路径
        :return:
        '''
        imgname, suffix = self.get_name_suffix(img_path)
        annotation_object = self.parser_annotation(imgname)
        img = cv2.imread(img_path)
        h, w, c = img.shape
        box = [annotation_object.rec[0], annotation_object.rec[2], annotation_object.rec[1], annotation_object.rec[3]]
        box = convert((w, h), box)
        path = os.path.join(output, imgname + '.txt')
        with open(path, 'w') as f:
            f.write(str(0) + ' ' + ' '.join([str(a) for a in box]) + '\n')
        queue.put(img_path + '\n')

    def generate_data(self, img_path_list, output, txt_file_name, mode, num_thread, root=None):
        '''
        :param img_path_list: ccpd 图片路径的列表
        :param output: 转化的label txt存放的路径
        :param txt_file_name: 'train.txt', 生成的txt文件名
        :param root: 生成标注文件的根目录
        :return:
        '''
        root = self.ccpd_dir if root is None else root
        mkdirs(output)
        print(f'output: {output}')
        txt_file = open(os.path.join(root, txt_file_name), 'w')
        print(f'txt file: {txt_file}')
        print(f'total {len(img_path_list)} images')
        # 创建线程池和队列
        pool = ThreadPool(processes=num_thread)
        print(f'start {num_thread} threads')
        q = Queue()
        c = 0
        for i in img_path_list:
            if c % 1000 == 0 and c != 0:
                print(f'generated {c} images')
            # getattr(self, 'generate_' + mode)(i, txt_file, output)
            pool.apply_async(getattr(self, 'generate_' + mode), (i, q, output,))
        pool.close()
        pool.join()
        print(f'closed {num_thread} threads')
        # 可以优化，创建一个线程边生产边消费
        while True:
            try:
                data = q.get(timeout=1)
                txt_file.write(data)
            except Empty:
                break
        txt_file.close()
        print(f'completed')

    @staticmethod
    def split_subtest(txt_path, names=['ccpd_challenge', 'ccpd_fn', 'ccpd_rotate', 'ccpd_weather', 'ccpd_db', 'ccpd_tilt', 'ccpd_blur']):
        root = os.path.dirname(txt_path)
        file_dict = {}
        count_dict = {}
        for i in names:
            file_dict[i] = open(os.path.join(root, i + '.txt'), 'w')
            count_dict[i] = 0
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                for name in names:
                    if name in line:
                        file_dict[name].write(line)
                        count_dict[name] += 1
        print('The number of each subset:')
        pprint.pprint(count_dict)
        for k, v in file_dict.items():
            file_dict[k].close()
        print('done')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ccpd_root', type=str, help='ccpd path')
    parser.add_argument('--type', type=str, choices=['yolo', 'icdar'], help='yolo or icdar')
    parser.add_argument('--num_threads', type=int, default=20, help='num threads')
    opt = parser.parse_args()
    ccpd_root = os.path.abspath(opt.ccpd_root)


    def process():
        import time
        st = time.time()
        paerser = CCPDParser(ccpd_root)
        output = opt.type + '_format'
        print(f'Convert CCPD format to {opt.type} format')
        print('start processing test')
        paerser.generate_data(paerser.test_img_list, os.path.join(ccpd_root, output, 'test'), 'test.txt', opt.type,
                              opt.num_threads, root=os.path.join(ccpd_root, output))
        print('====================================================================')
        print('start processing val')
        paerser.generate_data(paerser.val_img_list, os.path.join(ccpd_root, output, 'val'), 'val.txt', opt.type,
                              opt.num_threads, root=os.path.join(ccpd_root, output))
        print('====================================================================')
        print('start processing train')
        paerser.generate_data(paerser.train_img_list, os.path.join(ccpd_root, output, 'train'), 'train.txt', opt.type,
                              opt.num_threads, root=os.path.join(ccpd_root, output))
        et = time.time()
        print(f'total cost {et - st:.1f}s')


    process()
    if opt.type == 'icdar':
        output = opt.type + '_format'
        # 将test切分成不同场景的测试子集
        CCPDParser.split_subtest(os.path.join(ccpd_root, output, 'test.txt'))
