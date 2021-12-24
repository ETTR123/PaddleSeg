# AdvSemiSeg


## 1 简介
![images](images/network.png)  
本项目基于paddlepaddle框架复现了AdvSemiSeg半监督语义分割学习算法，基于Deeplabv2在Pascal VOC2012数据集上进行了实验。
**论文：**
- [1] Wei-Chih Hung, Yi-Hsuan Tsai, Yan-Ting Liou, Yen-Yu Lin, and Ming-Hsuan Yang
Proceedings of the British Machine Vision Conference (BMVC), 2018. [Adversarial Learning for Semi-Supervised Semantic Segmentation](https://arxiv.org/abs/1802.07934)

**项目参考：**
- [https://github.com/hfslyc/AdvSemiSeg](https://github.com/hfslyc/AdvSemiSeg)

## 2 复现精度
>在Pascal VOC2012 val数据集的测试效果如下表。

|NetWork |steps|opt|image_size|batch_size|dataset|memory|card|mIou|config|weight|log|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|Deeplabv2|20k|SGD|321x321|10|Pascal VOC2012+Aug|16G|1|72.66|[deeplabv2_resnet101_os8_voc_semi_321x321_20k.yml](configs/deeplabv2/deeplabv2_resnet101_os8_voc_semi_321x321_20k.yml)|(链接: https://pan.baidu.com/s/13bG-VGyW4VsD5iw3aJpJsQ 提取码: d3qy 复制这段内容后打开百度网盘手机App，操作更方便哦)|[log](-)|
注意：默认的配置是**ResNet101+Deeplabv2+VOC2012Aug+1/8Label**

## 3 数据集
[Pascal VOC 2012 + SBD dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html)

- 数据集大小:
    - 训练集: 10582
    - 验证集: 1449

应有的数据集结构：
```
pascalvoc/VOCdevkit/VOC2012
├── Annotations
├── ImageSets
├── JPEGImages
├── __MACOSX
├── SegmentationClass
├── SegmentationClassAug
└── SegmentationObject
```

## 4 环境依赖
- 硬件: Tesla V100 16G * 1

- 框架:
    - PaddlePaddle == 2.2.0



## 快速开始
**tipc**
[TIPC](test_tipc/docs/test_train_inference_python.md)

**模型动转静**

```python

python3.7 export_model.py --config test_tipc/configs/deeplabv2/deeplabv2_resnet101_os8_voc_semi_321x321_20k.yml --model_path=./test_tipc/output/deeplabv2_semi/norm_gpus_0_autocast_null/best_model/model.pdparams --save_dir=./test_tipc/output/deeplabv2_semi/norm_gpus_0_autocast_null
```

**预测**


```python

python3.7 infer.py --save_dir test_tipc/output/deeplabv2_semi/ --device=gpu --use_trt=False --precision=fp32 --config=./test_tipc/output/deeplabv2_semi/norm_gpus_0_autocast_null//deploy.yaml --batch_size=1 --image_path=test_tipc/data/pascalvoc/VOCdevkit/VOC2012/infer.txt --benchmark=True
```


```
#predict
Class IoU:
[0.823 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
 0.    0.    0.    0.    0.    0.    0.    0.    0.   ]
Class Acc:
[0.823 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
 0.    0.    0.    0.    0.    0.    0.    0.    0.   ]

 #inference
Class IoU:
[0.8228 0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.    ]
Class Acc:
[0.8232 0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.    ]
```

**tipc测试结果截图**
<div align="center">
    <img src="test_tipc\data\tipc_result.PNG" width="1000">
</div>


## 5 代码结构与说明
**代码结构**
```
├─configs  
├─images  
├─output  
├─paddleseg  
├─test_tipc  
│  export.py  
|  export_model.py  
|  infer.py  
|  infer_inference.py  
│  predict.py  
│  README.md  
│  requirements.txt  
│  setup.py  
│  train.py  
│  val.py  
```
**说明**
 感谢@CuberrChen复现AdvSemiSeg[https://github.com/CuberrChen/AdvSemiSeg-Paddle](https://github.com/CuberrChen/AdvSemiSeg-Paddle)
