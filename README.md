#BiSeNetV1


## 1 简介
本项目基于paddlepaddle框架复现了BiSeNet语义分割模型，BiSeNet利用Attention Refinement Module 和 Feature Fusion Module来提升网络性能。

**论文：**
- [1] Changqian Yu, Jingbo Wang, Chao Peng, Changxin Gao, Gang Yu, and Nong Sang. [BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation](https://paperswithcode.com/paper/bisenet-bilateral-segmentation-network-for)

**项目参考：**
- [https://github.com/ycszen/TorchSeg](https://github.com/ycszen/TorchSeg)

## 2 复现精度
>在CityScapes val数据集的测试效果如下表。


|NetWork |steps|opt|image_size|batch_size|dataset|memory|card|mIou|config|weight|log|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|BiSeNet|160K|SGD|1024x512|4|CityScapes|32G|4|75.19|[bisenetv1_cityscapes_1024x512_160k.yml](configs/bisenetv1/bisenetv1_cityscapes_1024x512_160k.yml)|[link](https://bj.bcebos.com/v1/ai-studio-cluster-infinite-task/outputs/105278.tar?authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2021-11-25T19%3A25%3A13Z%2F-1%2F%2F3b5cf09d2869e0445166814922739cc648b95396256b7eb7f6a1e07cbcf01021)|[log]()|

## 3 数据集
[CityScapes dataset](https://www.cityscapes-dataset.com/)

- 数据集大小:
    - 训练集: 2975
    - 验证集: 500

## 4 环境依赖
- 硬件: Tesla V100 * 4

- 框架:
    - PaddlePaddle == 2.2.0

## 快速开始
**tipc**
[TIPC](test_tipc/docs/test_train_inference_python.md)

**模型动转静**

```python

python3.7 export_model.py --config test_tipc/configs/bisenetv1/bisenetv1_cityscapes_1024x512_160k.yml --model_path=./test_tipc/output/bisenetv1/norm_gpus_0_autocast_null/best_model/model.pdparams --save_dir=./test_tipc/output/bisenetv1/
```

**预测**

**原图**
![原图](test_tipc/data/origin.png)

```python

python3.7 infer.py --save_dir test_tipc/output/bisenetv1/ --device=gpu --use_trt=False --precision=fp32 --config=./test_tipc/output/bisenetv1/norm_gpus_0_autocast_null//deploy.yaml --batch_size=1 --image_path=test_tipc/data/cityscapes/infer.list --benchmark=True

```

**结果**
![结果](test_tipc/data/gt.png)

```
#predict
Class IoU:
[0.9832 0.9334 0.979  0.     0.     0.6956 0.     0.8139 0.9705 0.
 0.8294 0.9199 0.     0.9541 0.     0.     0.     0.7803 0.8127]
Class Acc:
[0.9852 0.9869 0.9881 0.     0.     0.8212 0.     0.8451 0.9818 0.
 0.9833 0.9834 0.     0.9754 0.     0.     0.     0.9607 0.9394]

#inference
Class IoU:
[0.9832 0.9334 0.979  0.     0.     0.6956 0.     0.8139 0.9705 0.
 0.8294 0.9199 0.     0.9541 0.     0.     0.     0.7803 0.8127]
Class Acc:
[0.9852 0.9869 0.9881 0.     0.     0.8212 0.     0.8451 0.9818 0.
 0.9833 0.9834 0.     0.9754 0.     0.     0.     0.9607 0.9394]
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
 感谢朗督复现BiSeNetV1[https://github.com/justld/BiSNetV1_paddle](https://github.com/justld/BiSNetV1_paddle)
