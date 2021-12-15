# DMNet_paddle


## 1 简介
本项目基于paddlepaddle框架复现了DMNet语义分割模型，DMNet提出动态滤波器，核心在于DCM模块，该模块利用池化和卷积生成动态卷积核，与特征图进行卷积。

**论文：**
- [1] Junjun He, Zhongying Deng, Yu Qiao. [Dynamic Multi-scale Filters for Semantic Segmentation](https://openaccess.thecvf.com/content_ICCV_2019/papers/He_Dynamic_Multi-Scale_Filters_for_Semantic_Segmentation_ICCV_2019_paper.pdf)

**项目参考：**
- [https://github.com/open-mmlab/mmsegmentation/tree/master/configs/dmnet](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/dmnet)

## 2 复现精度
>在CityScapes val数据集的测试效果如下表。

|NetWork |steps|opt|image_size|batch_size|dataset|memory|card|mIou|config|weight|log|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|DMNet|80K|SGD|1024x512|8|CityScapes|32G|4|79.88|[dmnet_cityscapes_1024x512_80k.yml](configs/dmnet_cityscapes_1024x512_80k.yml)|[weight](https://bj.bcebos.com/v1/ai-studio-cluster-infinite-task/outputs/105098.tar?authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2021-11-25T02%3A08%3A27Z%2F-1%2F%2F8fd8238db80084be64ea3ae49ddb9ca0f3926a2b0d30dd9f81b5273b4927657a) |[-]()|


## 3 环境依赖
- 硬件: Tesla V100 * 4

- 框架:
    - PaddlePaddle == 2.2.0


## 快速开始
**tipc**
[tipc](test_tipc/docs/test_train_inference_python.md)

**模型动转静**

```python

python3.7 export_model.py --config test_tipc/configs/dmnet_small/dmnet_cityscapes_1024x512_100k.yml --input_shape 1 3 1024 2048 --model_path=./test_tipc/output/dmnet_small/norm_gpus_0_autocast_null/best_model/model.pdparams --save_dir=./test_tipc/output/dmnet_small/norm_gpus_0_autocast_null
```

**预测**

**原图**
![原图](test_tipc/data/origin.png)

```python

python3.7 infer.py --save_dir test_tipc/output/dmnet_small/ --device=gpu --use_trt=False --precision=fp32 --config=./test_tipc/output/dmnet_small/norm_gpus_0_autocast_null//deploy.yaml --batch_size=1 --image_path=test_tipc/data/cityscapes/infer.list --benchmark=True

```

**结果**
![结果](test_tipc/data/gt.png)

```
#predict
Class IoU:
[0.9928 0.9658 0.9803 0.     0.     0.7592 0.     0.8325 0.9707 0.
 0.8213 0.9261 0.     0.9527 0.     0.     0.     0.5342 0.7833]
Class Acc:
[0.9949 0.9861 0.9875 0.     0.     0.852  0.     0.8785 0.9855 0.
 0.9601 0.9849 0.     0.9802 0.     0.     0.     0.9808 0.8817]

#inference
Class IoU:
[0.9928 0.9658 0.9803 0.     0.     0.7592 0.     0.8325 0.9707 0.
 0.8213 0.9261 0.     0.9527 0.     0.     0.     0.5342 0.7833]
Class Acc:
[0.9949 0.9861 0.9875 0.     0.     0.852  0.     0.8785 0.9855 0.
 0.9601 0.9849 0.     0.9802 0.     0.     0.     0.9808 0.8817]
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
│  predict.py  
│  README.md
|  infer.py
|  infer_inference.py  
│  requirements.txt  
│  setup.py  
│  train.py  
│  val.py  
```
**说明**
 感谢朗督复现DMNet[https://github.com/justld/DMNet_paddle](https://github.com/justld/DMNet_paddle)
