#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/6/22 13:22
# @Author : jj.wang

import os
import json
import base64
import requests


def decode2base64(path):
    with open(path, 'rb') as f:
        base64_data = base64.b64encode(f.read())
        s = base64_data.decode()
    return s


if __name__ == '__main__':
    os.environ['LANG'] = 'C.UTF-8'
    os.environ['LC_ALL'] = 'C.UTF-8'
    PATH = os.path.abspath(os.path.dirname(__file__))
    os.chdir(PATH)
    img_base64 = decode2base64('赣B2371B.jpg')
    result = requests.post('http://127.0.0.1:8080/predict', data=json.dumps({'imgData': img_base64})).json()
    print(result)
