# FastFCN_paddle

## 1 简介

本项目基于paddlepaddle框架复现了FastFCN语义分割模型，使用的backbone是ENCNet_resnet50，FastFCN利用JPU模块来提升语义分割的效果。

**论文：**
- [1] Huikai Wu, Junge Zhang, Kaiqi Huang. [FastFCN: Rethinking Dilated Convolution in the Backbone for Semantic Segmentation](https://paperswithcode.com/paper/fastfcn-rethinking-dilated-convolution-in-the)

**项目参考：**
- [https://github.com/wuhuikai/FastFCN](https://github.com/wuhuikai/FastFCN)

## 2 复现精度
>ADE20K val数据集的测试效果如下表。

|NetWork |steps|opt|image_size|batch_size|dataset|memory|card|mIou|config|weight|log|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|FastFCN|120K|SGD|1024x512|4|ADE20K|32G|4|43.37|[fastfcn_ade20k_520x520_120k.yml](configs/fastfcn_ade20k_520x520_120k.yml)|[weight](https://bj.bcebos.com/v1/ai-studio-cluster-infinite-task/outputs/106456.tar?authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2021-12-02T04%3A11%3A57Z%2F-1%2F%2F37b0f7d7baf9e1c2275bdb3ed615295a34b2761de918da626fb9962cb0330c6c) |[-]()|


## 3 环境依赖
- 硬件: Tesla V100 * 4

- 框架:
    - PaddlePaddle == 2.2.0


## 快速开始
**tipc**
[TIPC](test_tipc/docs/test_train_inference_python.md)

**模型动转静**

```python

python3.7 export_model.py --config test_tipc/configs/fastfcn_small/fastfcn_ade20k_520x520_120k.yml --model_path=./test_tipc/output/fastfcn_small/norm_gpus_0_autocast_null/best_model/model.pdparams --save_dir=./test_tipc/output/fastfcn_small/norm_gpus_0_autocast_null
```

**预测**

**原图**

![原图](test_tipc/data/origin.jpg)

```python

python3.7 infer.py --save_dir test_tipc/output/fastfcn_small/ --device=gpu --use_trt=False --precision=fp32 --config=./test_tipc/output/fastfcn_small/norm_gpus_0_autocast_null//deploy.yaml --batch_size=1 --image_path=test_tipc/data/ADEChallengeData2016/validation.txt --benchmark=True


```

**结果**
![结果](test_tipc/data/gt.png)

```
#predict
Class IoU:
[0.4106 0.0389 0.9818 0.     0.5603 0.     0.5652 0.     0.     0.9602
 0.     0.     0.     0.     0.     0.     0.     0.2867 0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.    ]
Class Acc:
[0.98   0.7594 0.9878 0.     0.5686 0.     1.     0.     0.     0.974
 0.     0.     0.     0.     0.     0.     0.     0.8203 0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.    ]

 #inference
Class IoU:
[0.4107 0.0389 0.9818 0.     0.5603 0.     0.5652 0.     0.     0.9602
 0.     0.     0.     0.     0.     0.     0.     0.2867 0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.    ]
Class Acc:
[0.98   0.7594 0.9878 0.     0.5686 0.     1.     0.     0.     0.974
 0.     0.     0.     0.     0.     0.     0.     0.8203 0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.
 0.     0.     0.     0.     0.     0.     0.     0.     0.     0.    ]
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
 感谢朗督复现FastFCN[https://github.com/justld/FastFCN_paddle](https://github.com/justld/FastFCN_paddle)
