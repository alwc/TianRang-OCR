#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/6/23 16:49
# @Author : jj.wang

import os

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import cv2
import time
import json
import flask
import pprint
import traceback
from easydict import EasyDict as edict
from tools.det_predict import DetModel
from tools.rec_predict import RecModel
from data_loader.modules.random_crop_data import CropWordBox
from utils.utils import read_base64, draw_bbox, makedirs, cv2ImgAddText
from hyperlpr import PR
from flask import render_template, request, url_for, jsonify

app = flask.Flask(__name__)
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True


@app.route('/')
def index():
    htmlFileName = 'lpr.html'
    return render_template(htmlFileName)


@app.route("/predict", methods=["POST"])
def predict():
    time_str = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(time.time()))
    if flask.request.method == 'POST':
        start = time.time()
        received_file = request.files['input_image']
        imageFileName = received_file.filename
        if received_file:
            # 保存接收的图片到指定文件夹
            received_dirPath = 'static/images'
            if not os.path.isdir(received_dirPath):
                os.makedirs(received_dirPath)
            imageFilePath = os.path.join(received_dirPath, time_str + '_' + imageFileName)
            received_file.save(imageFilePath)
            print('receive image and save: %s' % imageFilePath)
            usedTime = time.time() - start
            print('receive image and save cost time: %f' % usedTime)
            preds, boxes_list, score_list, det_time = det_model.predict(imageFilePath, is_output_polygon=False,
                                                                        short_size=args.det_short_size)
            img = cv2.imread(imageFilePath)
            draw_img = draw_bbox(img, boxes_list)
            drawed_imageFileName = time_str + '_draw_' + os.path.splitext(imageFileName)[0] + '.jpg'
            drawed_imageFilePath = os.path.join('static', drawed_imageFileName)
            result = []
            for i, box in enumerate(boxes_list):
                rec_img = CropWordBox.crop_image_by_bbox(img, box, args.rec_crop_ratio)
                text, prob, t = rec_model.predict(rec_img)
                prob = round(prob, 3)
                draw_img = cv2ImgAddText(draw_img, text, (box[0][0], box[0][1] - 40), textColor=(255, 255, 0),
                                         textSize=40)
                draw_img = cv2ImgAddText(draw_img, f'{prob:.3f}', (box[3][0], box[3][1] + 5),
                                         textColor=(255, 255, 0), textSize=40)
                result.append(text)
            print(f'draw image save: {drawed_imageFilePath}')
            cv2.imwrite(drawed_imageFilePath, draw_img)
            image_source_url = url_for('static', filename=drawed_imageFileName)
            return jsonify(src=image_source_url, count=f'{result}')



def init_args():
    import argparse
    parser = argparse.ArgumentParser(description='tianrang-ocr')
    parser.add_argument('--host', type=str, default='0.0.0.0', help=' ')
    parser.add_argument('--port', type=int, default=8080, help=' ')
    parser.add_argument('--device_id', default=None, help='gpu device id, if None use cpu', type=int)
    parser.add_argument('--det_model_path', default='weights/det.pth', type=str)
    parser.add_argument('--det_thre', default=0.7, help='the thresh of post_processing', type=float)
    parser.add_argument('--det_short_size', default=736, help='det model input image short size', type=int)
    parser.add_argument('--rec_model_path', default='weights/rec.pth', type=str)
    parser.add_argument('--rec_crop_ratio', default=1.05, help='crop box ratio', type=float)
    parser.add_argument('--log_dir', default='static/lpr.log', help='save log file dir', type=str)
    parser.add_argument('--use_hyperlpr', action='store_true', help='debug mode')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    from flask_utils.log import Logger

    PATH = os.path.abspath(os.path.dirname(__file__))
    os.chdir(PATH)
    args = init_args()
    # log
    logger = Logger(args.log_dir).logger()

    logger.info(pprint.pformat(args))
    logger.info('========================================================================')
    if args.device_id is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
        logger.info('Use CPU')
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
        logger.info(f'Use GPU: {args.device_id}')


    det_model = DetModel(args.det_model_path, args.det_thre, gpu_id=0)
    logger.info('init det model')
    logger.info('========================================================================')

    if args.use_hyperlpr:
        rec_model = PR
        logger.info('use hyperlpr as rec model')
    else:
        rec_model = RecModel(args.rec_model_path, gpu_id=0)
    logger.info('init rec model')
    logger.info('========================================================================')

    app.run(host=args.host, port=args.port)
