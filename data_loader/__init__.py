# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:52
# @Author  : zhoujun
import cv2
import copy
import math
import PIL
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


def get_dataset(data_path, module_name, transform, dataset_args):
    """
    获取训练dataset
    :param data_path: dataset文件列表，每个文件内以如下格式存储 ‘path/to/img\tlabel’
    :param module_name: 所使用的自定义dataset名称，目前只支持data_loaders.ImageDataset
    :param transform: 该数据集使用的transforms
    :param dataset_args: module_name的参数
    :return: 如果data_path列表不为空，返回对于的ConcatDataset对象，否则None
    """
    from . import dataset
    s_dataset = getattr(dataset, module_name)(transform=transform, data_path=data_path,
                                              **dataset_args)
    return s_dataset


def get_transforms(transforms_config):
    tr_list = []
    for item in transforms_config:
        if 'args' not in item:
            args = {}
        else:
            args = item['args']
        cls = getattr(transforms, item['type'])(**args)
        tr_list.append(cls)
    tr_list = transforms.Compose(tr_list)
    return tr_list


class ICDARCollectFN():
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, batch):
        data_dict = {}
        to_tensor_keys = []
        for sample in batch:
            for k, v in sample.items():
                if k not in data_dict:
                    data_dict[k] = []
                if isinstance(v, (np.ndarray, torch.Tensor, PIL.Image.Image)):
                    if k not in to_tensor_keys:
                        to_tensor_keys.append(k)
                data_dict[k].append(v)
        for k in to_tensor_keys:
            data_dict[k] = torch.stack(data_dict[k], 0)
        return data_dict



class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img


class ResizeNormalize(object):

    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = cv2.resize(img, self.size, interpolation=self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class AlignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=True):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, batch):
        images = []
        labels = []
        ratios = []
        for samples in batch:
            if isinstance(samples['word_imgs'], list):
                for i, sample in enumerate(samples['word_imgs']):
                    try:
                        h, w, c = sample.shape
                    except:
                        import pdb
                        pdb.set_trace()
                    ratios.append(w/h)
                    images.append(sample)
                    labels.append(samples['texts'][i])
            else:
                h, w, c = samples['word_imgs'].shape
                ratios.append(w/h)
                images.append(samples['word_imgs'])
                labels.append(samples['texts'][0])
        ratios.sort()
        max_ratio = ratios[-1]
        if self.keep_ratio_with_pad:
            resized_max_w = math.ceil(max_ratio * self.imgH)
            input_channel = images[0].shape[-1]
            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

            resized_images = []
            for image in images:
                h, w, c = image.shape
                # ratio = w / float(h)
                # if math.ceil(self.imgH * ratio) > resized_max_w:
                #     resized_w = resized_max_w
                # else:
                #     resized_w = math.ceil(self.imgH * ratio)

                resized_image = cv2.resize(image, None, fx=resized_max_w/w, fy=self.imgH/h,  interpolation=cv2.INTER_LINEAR)
                resized_images.append(transform(resized_image))
                # resized_image.save('./image_test/%d_test.jpg' % w)

            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        else:
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)
        data = {}
        data['img'] = image_tensors
        data['labels'] = labels
        return data



def get_dataloader(module_config, distributed=False):
    if module_config is None:
        return None
    config = copy.deepcopy(module_config)
    dataset_args = config['dataset']['args']
    if 'transforms' in dataset_args:
        img_transfroms = get_transforms(dataset_args.pop('transforms'))
    else:
        img_transfroms = None
    # 创建数据集
    dataset_name = config['dataset']['type']
    data_path = dataset_args.pop('data_path')
    if data_path == None:
        return None

    data_path = [x for x in data_path if x is not None]
    if len(data_path) == 0:
        return None
    if 'collate_fn' not in config['loader'] or config['loader']['collate_fn'] is None or len(config['loader']['collate_fn']) == 0:
        config['loader']['collate_fn'] = None
    else:
        config['loader']['collate_fn'] = eval(config['loader']['collate_fn']['type'])(**config['loader']['collate_fn'].get('args', {}))

    _dataset = get_dataset(data_path=data_path, module_name=dataset_name, transform=img_transfroms, dataset_args=dataset_args)
    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        # 3）使用DistributedSampler
        sampler = DistributedSampler(_dataset)
        config['loader']['shuffle'] = False
        config['loader']['pin_memory'] = True
    loader = DataLoader(dataset=_dataset, sampler=sampler, **config['loader'])
    return loader
