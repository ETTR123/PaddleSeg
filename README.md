# ESPNet

## 1 简介
本项目基于paddlepaddle框架复现了ESPNet语义分割模型，该论文作者利用卷积因子分解原理设计了非常精巧的EESP模块，并基于次提出了一个轻量级、效率高的通用卷积神经网络模型ESPNet，能够大大减少模型的参数并且保持模型的性能。

### 论文:
[1] Mehta S ,  Rastegari M ,  Caspi A , et al. ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation

### 项目参考：
https://github.com/sacmehta/ESPNet

## 2 复现精度
>在CityScapes val数据集的测试效果如下表。


| |steps|opt|image_size|batch_size|dataset|memory|card|mIou|config|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|ESPNet|120k|adam|1024x512|4|CityScapes|32G|4|0.6365|[espnet_cityscapes_1024x512_120k.yml](configs/espnetv1/espnetv1_cityscapes_1024x512_120k.yml)|

## 3 数据集
[CityScapes dataset](https://www.cityscapes-dataset.com/)

- 数据集大小:
    - 训练集: 2975
    - 验证集: 500

## 4 环境依赖
- 硬件: Tesla V100 * 4

- 框架:
    - PaddlePaddle == develop

## 快速开始
**tipc**
[TIPC](test_tipc/docs/test_train_inference_python.md)

**模型动转静**

```python

python3.7 export_model.py --config test_tipc/configs/espnetv1/espnetv1_cityscapes_1024x512_120k.yml --input_shape 1 3 1024 2048 --model_path=./test_tipc/output/espnetv1/norm_gpus_0_autocast_null/best_model/model.pdparams --save_dir=./test_tipc/output/espnetv1/
```

**预测**

**原图**

```python

python3.7 infer.py --save_dir test_tipc/output/espnetv1/ --device=gpu --use_trt=False --precision=fp32 --config=./test_tipc/output/espnetv1/norm_gpus_0_autocast_null//deploy.yaml --batch_size=1 --image_path=test_tipc/data/cityscapes/infer.list --benchmark=True

```

**结果**

```
#predict
Class IoU:
[0.1137 0.0486 0.0033 0.     0.     0.0047 0.     0.0062 0.0359 0.
 0.0024 0.0062 0.     0.0187 0.     0.     0.     0.0036 0.0103]
Class Acc:
[0.3778 0.1131 0.5579 0.     0.     0.0073 0.     0.0067 0.0769 0.
 0.0026 0.0071 0.     0.0384 0.     0.     0.     0.0038 0.0117]

#inference
Class IoU:
[0.1137 0.0486 0.0033 0.     0.     0.0047 0.     0.0062 0.0359 0.
 0.0024 0.0062 0.     0.0187 0.     0.     0.     0.0036 0.0103]
Class Acc:
[0.3778 0.1131 0.5579 0.     0.     0.0073 0.     0.0067 0.0769 0.
 0.0026 0.0071 0.     0.0384 0.     0.     0.     0.0038 0.0117]
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
 感谢simuler[https://github.com/simuler/ESPNet](https://github.com/simuler/ESPNet)
