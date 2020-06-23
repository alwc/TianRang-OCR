#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/6/17 11:29
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

app = flask.Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    time_str = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(time.time()))
    if flask.request.method == 'POST':
        try:
            start = time.time()
            # 入参 {"imgData": img_base64_str}
            data = json.loads(flask.request.data)
            t1 = time.time()
            # BGR
            img = read_base64(data['imgData'], mode='opencv')
            t2 = time.time()
            preds, boxes_list, score_list, det_time = det_model.predict(img, is_output_polygon=False, short_size=args.det_short_size)
            if args.debug:
                draw_img = draw_bbox(img, boxes_list)
            result = []
            rec_time = 0
            for i, box in enumerate(boxes_list):
                rec_img = CropWordBox.crop_image_by_bbox(img, box, args.rec_crop_ratio)
                text, prob, t = rec_model.predict(rec_img)
                prob = round(prob, 3)
                rec_time += t
                result.append({'id':time_str+ '_' + str(i), 'box':box.tolist(), 'recognition':text, 'prob':prob})
                if args.debug:
                    draw_img = cv2ImgAddText(draw_img, text, (box[0][0], box[0][1]-40), textColor=(255, 255, 0), textSize=40)
                    draw_img = cv2ImgAddText(draw_img, f'{prob:.3f}', (box[3][0], box[3][1]+5), textColor=(255, 255, 0), textSize=40)
            if args.debug:
                cv2.imwrite(os.path.join('debug/draw_img', 'draw_' + time_str + '.jpg'), draw_img)
                cv2.imwrite(os.path.join('debug/org_img', time_str + '.png'), img)
            logger.info(f'get img time: {(t1-start)*1000: .1f}ms \n'
                        f'read base64 img time: {(t2-t1)*1000: .1f}ms \n'
                        f'det preprocess time: {det_time[0]*1000: .1f}ms \n'
                        f'det inference time: {det_time[1]*1000: .1f}ms \n'
                        f'det postprocess time: {det_time[2]*1000: .1f}ms \n'
                        f'det total time: {det_time[3]*1000: .1f}ms \n'
                        f'rec total time: {rec_time*1000: .1f}ms \n')
            end = time.time()
            out = {'data':result, 'code':1, 'message':'', 'getImageTime':time_str}
            logger.info(f'total cost time: {(end - start)*1000: .1f}ms')
            logger.info(pprint.pformat(out))
            logger.info('========================================================================')
            if args.file_record:
                file_record.write(str(out))
            return json.dumps(out, ensure_ascii=False)
        except:
            out = {'code':0, 'message':traceback.format_exc(), 'getImageTime':time_str}
            logger.error(traceback.format_exc())
            logger.info('========================================================================')
            return json.dumps(out, ensure_ascii=False)
    else:
        out = {'code':0, 'message':'request method must be post', 'getImageTime':time_str}
        logger.error('request method must be post')
        logger.info('========================================================================')
        return json.dumps(out, ensure_ascii=False)


def init_args():
    import argparse
    parser = argparse.ArgumentParser(description='tianrang-ocr')
    parser.add_argument('--host', type=str, default='0.0.0.0', help=' ')
    parser.add_argument('--port', type=int, default=8080, help=' ')
    parser.add_argument('--device_id', default=None, help='gpu device id, if None use cpu', type=int)
    parser.add_argument('--det_model_path', default='weights/det.pth', type=str)
    parser.add_argument('--det_thre', default=0.7, help='the thresh of post_processing', type=float)
    parser.add_argument('--det_short_size', default=416, help='det model input image short size', type=int)
    parser.add_argument('--rec_model_path', default='weights/rec.pth', type=str)
    parser.add_argument('--rec_crop_ratio', default=1.05, help='crop box ratio', type=float)
    parser.add_argument('--file_record', action='store_true', help='Whether to save the recognition result')
    parser.add_argument('--log_dir', default='output/log/lpr.log', help='save log file dir', type=str)
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--use_hyperlpr', action='store_true', help='debug mode')
    parser.add_argument('--use_config_json', action='store_true', help='load config.json as config')
    args = parser.parse_args()
    if args.use_config_json:
        print('load flask_utils/config.json as config')
        with open('flask_utils/config.json', 'r') as f:
            config = edict(json.load(f))
        return config
    print('use command line as config')
    return args


if __name__ == '__main__':
    from flask_utils.log import Logger
    PATH = os.path.abspath(os.path.dirname(__file__))
    os.chdir(PATH)
    args = init_args()
    # record
    if args.file_record:
        from flask_utils.record import FileRecord
        file_record = FileRecord('output/result', 'result', True)
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

    if args.debug:
        makedirs(['debug/org_img', 'debug/draw_img', 'debug/txt'])
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