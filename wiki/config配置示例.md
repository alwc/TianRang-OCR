# 识别模型Config配置示例

ResNet18+TPS+CTC识别模型示例，参考config/ccpd_res18.yaml。

会从base中加载基础配置，然后再加载config/ccpd_res18.yaml配置，如果重复配置，config/ccpd_res18.yaml中的配置项会覆盖基础配置项。训练结果会保存在{output_dir}/{name}_{algorithm}目录下。

## 基础配置

```yaml
name: resnet18_tps_small
base: ['config/ccpd_aug_base.yaml']
```



## 识别模型结构配置

模型结构采用arch进行配置，algorithm表示为识别算法，preprocess、backbone、neck、head子结构的type定义该组件的类型，args为对应参数。

```yaml
arch:
  preprocess:
    type: TPS
    args:
      F: 20
      I_size: [32, 120]
      I_r_size: [32, 120]
      I_channel_num: 3
      scale: 0.25
  algorithm: rec
  backbone:
    type: resnet18
  neck:
    type: Squeeze
    args:
      mode: max_pool
  head:
    type: CTC_Head
    args:
      dropout_rate: 0
```



## 识别模型的converter、metric、loss、optimizer、lr_scheduler配置

```yaml
converter:
  type: CTCLabelConverter 
  args:
    character: license_plate # 可以在model/converter/char.py中的CHAR类添加其他字符集
metric:
  type: STRMeters # 识别的评价指标，检测的为QuadMetric
loss:
  type: CTCLoss # 和converter一致
optimizer:
  type: Adadelta # 使用的是pytorch官方实现的optimizer
  args:
    lr: 0.1
lr_scheduler:
  type: MultiStepLR # 使用的是pytorch官方实现的lr_scheduler
  args:
    milestones: [60, 90]
```



##识别模型的trainer配置

```yaml
trainer:
  seed: 2
  epochs: 10 # 训练轮数
  log_iter: 100 # 多少个steps打印一次Log
  show_images_iter: 500 # 多少个steps将识别结果添加到tensorboard，暂时未实现
  resume_checkpoint: '' # resume模型的地址，
  finetune_checkpoint: '' # finetune模型的地址
  output_dir: output  # output的地址
  tensorboard: true # 暂未实现
```



## 识别模型的dataset配置

```yaml
dataset:
  train:
    dataset:
      args:
        data_path:
          - /home/wjj/src/new_ccpd/CCPD2019/icdar_format/train.txt
        pre_processes: # 数据的预处理过程，包含augment和标签制作
          - type: CropWordBox
            args:
              value: [1,1.2]
          - type: IaaAugment # 使用imgaug进行变换，参考imgaug库
            args:
              # - {'type':Resize, 'args':{'size': {'height': 32, 'width': 'keep-aspect-ratio'}}}
              - {'type':AddToHueAndSaturation,'args':{'value':[-50, 50]}}
              - {'type':Affine, 'args':{'rotate':[-10,10]}}
              - {'type':MultiplyBrightness, 'args':{'mul':[0.7, 1.3]}}
        img_mode: RGB
        filter_keys: [img_path,img_name,ignore_tags,shape] # 返回数据之前，从数据字典里删除的key
        ignore_tags: ['*', '###']
    loader:
      batch_size: 32
      shuffle: true
      pin_memory: true
      num_workers: 8
      collate_fn:
        type: AlignCollate
        args:
          imgH: 32
          imgW: 120
          keep_ratio_with_pad: false # 直接resize到相应大小
  validate:
    dataset:
      type: ICDAR2015Dataset
      args:
        data_path:
          - /home/wjj/src/new_ccpd/CCPD2019/icdar_format/test.txt
        pre_processes:
          - type: CropWordBox
            args:
              value: [1,1.2]
        img_mode: RGB
    loader:
      batch_size: 32
      shuffle: true
      pin_memory: false
      num_workers: 8
      collate_fn:
        type: AlignCollate
        args:
          imgH: 32
          imgW: 120
          keep_ratio_with_pad: false
```



# 检测模型Config配置示例

ccpd_det_dbnet_shufflenet.yaml

```yaml
name: DBNet_shufflenet_96
base: ['config/icdar2015.yaml']
arch:
  algorithm: det
  backbone:
    type: shufflenet_v2_x0_5
    args:
      pretrained: true
  neck:
    type: FPN
    args:
      inner_channels: 96
  head:
    type: DBHead
    args:
      k: 50
post_processing:
  type: DBPostProcess
  args:
    thresh: 0.3
    box_thresh: 0.7
    max_candidates: 1000
    unclip_ratio: 3.0
metric:
  type: QuadMetric
  args:
    is_output_polygon: false
loss:
  type: DBLoss
  args:
    alpha: 1
    beta: 10
    ohem_ratio: 3
optimizer:
  type: Adam
  args:
    lr: 0.001
    weight_decay: 0
    amsgrad: true
lr_scheduler:
  type: WarmupPolyLR
  args:
    warmup_epoch: 0.1
trainer:
  seed: 2
  epochs: 10
  log_iter: 100
  show_images_iter: 500
  resume_checkpoint: ''
  finetune_checkpoint: ''
  output_dir: output
  tensorboard: true
dataset:
  train:
    dataset:
      args:
        data_path:
          - /home/wjj/src/new_ccpd/CCPD2019/icdar_format/train_test.txt
        img_mode: RGB
    loader:
      batch_size: 16
      shuffle: true
      pin_memory: true
      num_workers: 6
      collate_fn: ''
  validate:
    dataset:
      args:
        data_path:
          - /home/wjj/src/new_ccpd/CCPD2019/icdar_format/val.txt
        pre_processes:
          - type: ResizeShortSize
            args:
              short_size: 416
              resize_text_polys: false
        img_mode: RGB
    loader:
      batch_size: 1
      shuffle: true
      pin_memory: false
      num_workers: 6
      collate_fn:
        type: ICDARCollectFN

```



基础配置config/icdar2015.yaml：

```yaml
name: tianrang-ocr
dataset:
  train:
    dataset:
      type: ICDAR2015Dataset # 数据集类型
      args:
        data_path: # 一个存放 img_path \t gt_path的文件
          - ''
        pre_processes: # 数据的预处理过程，包含augment和标签制作
          - type: IaaAugment # 使用imgaug进行变换
            args:
              - {'type':Fliplr, 'args':{'p':0.5}}
              - {'type': Affine, 'args':{'rotate':[-20,20]}}
              - {'type':Resize,'args':{'size':[0.5, 1.5]}}
              - {'type':AddToHueAndSaturation,'args':{'value':[-75, 75]}}
              - {'type':MultiplyBrightness, 'args':{'mul':[0.7, 1.3]}}
          - type: EastRandomCropData
            args:
              size: [416,416]
              max_tries: 50
              keep_ratio: true
          - type: MakeBorderMap
            args:
              shrink_ratio: 0.4
              thresh_min: 0.3
              thresh_max: 0.7
          - type: MakeShrinkMap
            args:
              shrink_ratio: 0
              min_text_size: 8
        transforms: # 对图片进行的变换方式
          - type: ToTensor
            args: {}
          - type: Normalize
            args:
              mean: [0.485, 0.456, 0.406]
              std: [0.229, 0.224, 0.225]
        img_mode: RGB
        filter_keys: [img_path,img_name,ignore_tags,shape] # 返回数据之前，从数据字典里删除的key
        ignore_tags: ['*', '###']
    loader:
      batch_size: 1
      shuffle: true
      pin_memory: false
      num_workers: 0
      collate_fn: ''
  validate:
    dataset:
      type: ICDAR2015Dataset
      args:
        data_path:
          - ''
        pre_processes:
          - type: ResizeShortSize
            args:
              short_size: 416
              resize_text_polys: false
        transforms:
          - type: ToTensor
            args: {}
          - type: Normalize
            args:
              mean: [0.485, 0.456, 0.406]
              std: [0.229, 0.224, 0.225]
        img_mode: RGB
        filter_keys: []
        ignore_tags: ['*', '###']
    loader:
      batch_size: 1
      shuffle: true
      pin_memory: false
      num_workers: 0
      collate_fn: ICDARCollectFN

```

