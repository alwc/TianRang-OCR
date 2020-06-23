# Convert CCPD formatmd

# 一、数据集详情

## 1、数据集简介

CCPD是一个用于车牌识别的大型国内的数据集，由中科大的科研人员构建出来的。发表在ECCV2018论文Towards End-to-End License Plate Detection and Recognition: A Large Dataset and Baseline。

Github：[https://github.com/detectRecog/CCPD](https://github.com/detectRecog/CCPD)

该数据集在合肥市的停车场采集得来的，采集时间早上7:30到晚上10:00.涉及多种复杂环境。一共包含超多30万张图片，每种图片大小720x1160x3。根据文件夹分类，涉及不同的场景。

| CCPD-     | 数量/k | 描述                            |
| --------- | ------ | ------------------------------- |
| Base      | 200    | 正常车牌                        |
| FN        | 20.9   | 距离摄像头相当的远或者相当近    |
| DB        | 10     | 光线暗或者比较亮                |
| Rotate    | 10     | 水平倾斜20-25°，垂直倾斜-10-10° |
| Tilt      | 30     | 水平倾斜15-45°，垂直倾斜15-45°  |
| Weather   | 10     | 在雨天，雪天，或者雾天          |
| Blur      | 20.6   | 由于相机抖动造成的模糊          |
| Challenge | 50     | 其他的比较有挑战性的车牌        |
| NP        | 5      | 没有车牌的新车                  |

## 2、数据集标注格式

根据官方说明：每个名称可以分为七个字段。但是面积这个字段似乎不是每张图片都有，本次解析脚本中也未使用面积字段。不同字段间使用'-'作为分割符，相同字段间使用'_'作为分割符，xy坐标间使用('&'和'&')作为分隔符

如：**025-95_113-154＆383_386＆473-386＆473_177＆454_154＆383_363＆402-0_0_22_27_27_33_16-37-15.jpg**

1. 面积：牌照面积与整个图片区域的面积比。（025）
2. 倾斜度：水平倾斜程度和垂直倾斜度。（95_113）
3. 边界框坐标：左上和右下顶点的坐标。（154＆383_386＆473）
4. 四个顶点位置：整个图像中LP的四个顶点的精确（x，y）坐标。这些坐标从右下角顶点开始。（386＆473_177＆454_154＆383_363＆402）
5. 车牌号：CCPD中的每个图像只有一个LP。每个LP号码由一个汉字，一个字母和五个字母或数字组成。有效的中文车牌由七个字符组成：省（1个字符），字母（1个字符），字母+数字（5个字符）。“ 0_0_22_27_27_33_16”是每个字符的索引。这三个数组定义如下。每个数组的最后一个字符是字母O，而不是数字0。我们将O用作“无字符”的符号，因为中文车牌字符中没有O。
6. 亮度：牌照区域的亮度。（37）
7. 模糊度：车牌区域的模糊度。（15）



# 二、数据集格式转换

使用脚本CCPDParser.py。这一步最好在训练容器中做，这样生成的train.txt和test.txt中的图片路径就不需要在改动。参数说明：

| 参数名      | 说明               |      |
| ----------- | ------------------ | ---- |
| ccpd_root   | ccpd数据集的根目录 | str  |
| type        | 'yolo'或者'icdar'  | str  |
| num_threads | 线程数             | int  |

## 1、ccpd2yolo

```python
python3 CCPDParser.py --ccpd_root /home/wjj/src/ccpd_dataset_test/ --type yolo --num_threads 20
```

会在ccpd_root目录下生成yolo_format文件夹，只有一类，其中包括：

- train.txt
- val.txt
- test.txt
- train：目录下为yolo txt
- val：目录下为yolo txt
- test：目录下为yolo txt

train和val是ccpd_base中的图片按照9：1进行随机划分（已固定随机种子），test中的图片为除去ccpd_base和ccpd_np的图片。

## 2、ccpd2icdar

label中涉及中文，运行脚本前需要加上环境变量：

`export LANG=C.UTF-8 LC_ALL=C.UTF-8`

```python
LANG=C.UTF-8 LC_ALL=C.UTF-8 python3 CCPDParser.py --ccpd_root /home/wjj/src//decc--type icdar --num_threads 20
```

会在ccpd_root目录下生成icdar_format文件夹，其中包括：

- train.txt，每一行的格式为：`img_path \t icdar_txt_path`
- val.txt，每一行的格式为：`img_path \t icdar_txt_path`
- test.txt，每一行的格式为：`img_path \t icdar_txt_path`
- train：目录下为icdar txt
- val：目录下为icdar txt
- test：目录下为icdar txt

train和val是ccpd_base中的图片按照9：1进行随机划分（已固定随机种子），test中的图片为除去ccpd_base和ccpd_np的图片