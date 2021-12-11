# ENCNet_paddle


## 1 简介

本项目基于paddlepaddle框架复现了ENCNet语义分割模型，ENCNet提取出特征图中的类别信息，同时引出注意力机制损失(se_loss)来提升网络性能。

**论文：**
- [1] Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi, Amit Agrawal. [Context Encoding for Semantic Segmentation](https://paperswithcode.com/paper/context-encoding-for-semantic-segmentation)

**项目参考：**
- [https://github.com/open-mmlab/mmsegmentation/tree/master/configs/encnet](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/encnet)

## 2 复现精度
>在CityScapes val数据集的测试效果如下表。

|NetWork |steps|opt|image_size|batch_size|dataset|memory|card|mIou|config|weight|log|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|DMNet|80K|SGD|1024x512|8|CityScapes|32G|4|78.70|[encnet_cityscapes_1024x512_80k.yml](configs/encnet/encnet_cityscapes_1024x512_80k.yml)|[link](https://bj.bcebos.com/v1/ai-studio-cluster-infinite-task/outputs/105022.tar?authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2021-11-24T10%3A37%3A31Z%2F-1%2F%2F1d0504cbdf4fac38dc60c1298e9b632e739d1a2f952485056a8f50ff45f3344b)|[-](-)|


## 3 环境依赖
- 硬件: Tesla V100 * 4

- 框架:
    - PaddlePaddle == 2.2.0


## 快速开始


## 5 代码结构与说明
**代码结构**
```
├─configs  
├─images  
├─output  
├─paddleseg  
├─test_tipc  
│  export.py  
│  predict.py  
│  README.md  
│  README_CN.md  
│  requirements.txt  
│  setup.py  
│  train.py  
│  val.py  
```
**说明**
 感谢朗督复现ENCNet[https://github.com/justld/ENCNet_paddle](https://github.com/justld/ENCNet_paddle)
