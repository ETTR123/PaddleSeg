# Linux端基础训练预测功能测试

Linux端基础训练预测功能测试的主程序为`test_train_inference_python.sh`，可以测试基于Python的模型训练、评估、推理等基本功能，包括裁剪、量化、蒸馏。

<!-- - Mac端基础训练预测功能测试参考[链接](./mac_test_train_inference_python.md)
- Windows端基础训练预测功能测试参考[链接](./win_test_train_inference_python.md) -->

## 1. 测试结论汇总

- 训练相关：

| 算法名称 | 模型名称 | 单机单卡 | 单机多卡 | 多机多卡 | 模型压缩（单机多卡） |
|  :----  |   :----  |    :----  |  :----   |  :----   |  :----   |
| DeepLabv3p     |PP-HumanSeg-Server (DeepLabv3p_resnet50)| 正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 |  |  |
|  HRNet     |PP-HumanSeg-mobile (HRNet_W18_small)  |  正常训练 <br> 混合精度 | 正常训练 <br> 混合精度 |  |  |
| ConnectNet | PP-HumanSeg-Lite| 正常训练  | 正常训练  |  |  |
| BiSeNetV2 | BiSeNetV2 | 正常训练  | 正常训练  |  |  |
| OCRNet | OCRNet_HRNetW18 | 正常训练  | 正常训练  |  |  |
| Segformer | Segformer_B0 | 正常训练  | 正常训练  |  |  |
| STDC | STDC_STDC1 | 正常训练  | 正常训练  |  |  |
| MODNet | PP-Matting | 正常训练  | 正常训练  |  |  |


- 预测相关：基于训练是否使用量化，可以将训练产出的模型可以分为`正常模型`和`量化模型`，这两类模型对应的预测功能汇总如下，

| 模型类型 |device | batchsize | tensorrt | mkldnn | cpu多线程 |
|  ----   |  ---- |   ----   |  :----:  |   :----:   |  :----:  |
| 正常模型 | GPU | 1/6 | fp32/fp16 | - | - |
| 正常模型 | CPU | 1/6 | - | fp32/fp16 | 支持 |
| 量化模型 | GPU | 1/6 | int8 | - | - |
| 量化模型 | CPU | 1/6 | - | int8 | 支持 |


## 2. 测试流程

### 2.1 环境配置
运行环境配置请参考[文档](./install.md)的内容配置TIPC的运行环境。


### 2.2 功能测试
先运行`prepare.sh`准备数据和模型，然后运行`test_train_inference_python.sh`进行测试，最终在```test_tipc/output```目录下生成`python_infer_*.log`格式的日志文件。


`test_train_inference_python.sh`包含4种运行模式，每种模式的运行数据不同，分别用于测试速度和精度，分别是：

- 模式1：lite_train_lite_infer，使用少量数据训练，用于快速验证训练到预测的走通流程，不验证精度和速度；
```shell
bash test_tipc/prepare.sh ./test_tipc/configs/fcn_hrnetw18_small/train_infer_python.txt 'lite_train_lite_infer'
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/fcn_hrnetw18_small/train_infer_python.txt 'lite_train_lite_infer'
```  

- 模式2：lite_train_whole_infer，使用少量数据训练，一定量数据预测，用于验证训练后的模型执行预测，预测速度是否合理；
```shell
bash test_tipc/prepare.sh ./test_tipc/configs/fcn_hrnetw18_small/train_infer_python.txt  'lite_train_whole_infer'
bash test_tipc/test_train_inference_python.sh ../test_tipc/configs/fcn_hrnetw18_small/train_infer_python.txt 'lite_train_whole_infer'
```  

- 模式3：whole_infer，不训练，全量数据预测，走通开源模型评估、动转静，检查inference model预测时间和精度;
```shell
bash test_tipc/prepare.sh ./test_tipc/configs/fcn_hrnetw18_small/train_infer_python.txt 'whole_infer'
# 用法1:
bash test_tipc/test_train_inference_python.sh ../test_tipc/configs/fcn_hrnetw18_small/train_infer_python.txt 'whole_infer'
# 用法2: 指定GPU卡预测，第三个传入参数为GPU卡号
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/fcn_hrnetw18_small/train_infer_python.txt 'whole_infer' '1'
```  

- 模式4：whole_train_whole_infer，CE： 全量数据训练，全量数据预测，验证模型训练精度，预测精度，预测速度；
```shell
bash test_tipc/prepare.sh ./test_tipc/configs/fcn_hrnetw18_small/train_infer_python.txt 'whole_train_whole_infer'
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/fcn_hrnetw18_small/train_infer_python.txt 'whole_train_whole_infer'
```  

<!-- - 模式5：klquant_whole_infer，测试离线量化；
```shell
bash test_tipc/prepare.sh ./test_tipc/configs/fcn_hrnetw18_small_KL/model_linux_gpu_normal_normal_infer_python_linux_gpu_cpu.txt  'klquant_whole_infer'
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/fcn_hrnetw18_small_KL/model_linux_gpu_normal_normal_infer_python_linux_gpu_cpu.txt  'klquant_whole_infer'
``` -->

运行相应指令后，在`test_tipc/output`文件夹下自动会保存运行日志。如'lite_train_lite_infer'模式下，会运行训练+inference的链条，因此，在`test_tipc/output`文件夹有以下文件：
```
test_tipc/output/[model name]/
|- results_python.log    # 运行指令状态的日志
|- norm_train_gpus_0_autocast_null/  # GPU 0号卡上正常训练的训练日志和模型保存文件夹
|- pact_train_gpus_0_autocast_null/  # GPU 0号卡上量化训练的训练日志和模型保存文件夹
......
|- python_infer_cpu_usemkldnn_True_threads_1_batchsize_1.log  # CPU上开启Mkldnn线程数设置为1，测试batch_size=1条件下的预测运行日志
|- python_infer_gpu_usetrt_True_precision_fp16_batchsize_1.log # GPU上开启TensorRT，测试batch_size=1的半精度预测日志
......
```

其中`results_python.log`中包含了每条指令的运行状态，如果运行成功会输出：
```
Run successfully with command - python3.7 train.py --config test_tipc/configs/fcn_hrnetw18_small/fcn_hrnetw18_small_mini_supervisely.yml --do_eval --save_interval 500 --seed 100    --save_dir=./test_tipc/output/fcn_hrnetw18_small/norm_gpus_0_autocast_null --iters=50     --batch_size=2   !
Run successfully with command - python3.7 export.py --config test_tipc/configs/fcn_hrnetw18_small/fcn_hrnetw18_small_mini_supervisely.yml --model_path=./test_tipc/output/fcn_hrnetw18_small/norm_gpus_0_autocast_null/best_model/model.pdparams --save_dir=./test_tipc/output/fcn_hrnetw18_small/norm_gpus_0_autocast_null!
......
```
如果运行失败，会输出：
```
Run failed with command - python3.7 train.py --config test_tipc/configs/fcn_hrnetw18_small/fcn_hrnetw18_small_mini_supervisely.yml --do_eval --save_interval 500 --seed 100    --save_dir=./test_tipc/output/fcn_hrnetw18_small/norm_gpus_0_autocast_null --iters=50     --batch_size=2   !
Run failed with command - python3.7 export.py --config test_tipc/configs/fcn_hrnetw18_small/fcn_hrnetw18_small_mini_supervisely.yml --model_path=./test_tipc/output/fcn_hrnetw18_small/norm_gpus_0_autocast_null/best_model/model.pdparams --save_dir=./test_tipc/output/fcn_hrnetw18_small/norm_gpus_0_autocast_null!
......
```
可以很方便的根据`results_python.log`中的内容判定哪一个指令运行错误。


### 2.3 精度测试

使用compare_results.py脚本比较模型预测的结果是否符合预期，步骤包括：
- 计算预测结果图片和label的精度指标；
- 从本地文件中提取保存好的精度指标结果；
- 比较上述两个结果是否符合精度预期，误差大于设置阈值时会报错。

【注意】仅需要验证`whole_infer`模式下预测结果是否正确。

#### 使用方式

```shell
python3.7 test_tipc/compare_results.py --metric_file=./test_tipc/results/*.txt  --predict_dir=./test_tipc/output/xxx/python_infer_*_results --gt_dir=./test_tipc/data/xxx --num_classes xxx
```

参数介绍：  
- metric_file： 指向事先保存好的预测结果路径，支持*.txt 结尾，会自动索引*.txt格式的文件，文件默认保存在test_tipc/result/ 文件夹下
- predict_dir: 指向运行test_tipc/test_train_inference_python.sh 脚本的infer模式保存的预测结果目录，同样支持python_infer_*_results格式传入
- gt_dir: 指向数据集label图片目录
- num_classes: 数据集的类别数，不包括ignore_index
- atol: 设置的绝对误差，默认为1e-3
- rtol: 设置的相对误差，默认为1e-3

以`fcn_hrnetw18_small`模型为例，运行命令如下：
```shell
python3.7 test_tipc/compare_results.py --metric_file=./test_tipc/results/*.txt  --predict_dir=./test_tipc/output/fcn_hrnetw18_small/python_infer_*_results --gt_dir=./test_tipc/data/mini_supervisely/Annotations --num_classes 2
```

#### 运行结果

正常运行效果如下：
```
Assert allclose passed! The results of python_infer_cpu_usemkldnn_False_threads_1_precision_fp32_batchsize_1_results and ./test_tipc/results/python_fcn_hrnetw18_small_results_fp32.txt are consistent!
Assert allclose passed! The results of python_infer_gpu_usetrt_False_precision_fp32_batchsize_1_results and ./test_tipc/results/python_fcn_hrnetw18_small_results_fp32.txt are consistent!
Assert allclose passed! The results of python_infer_cpu_usemkldnn_True_threads_1_precision_fp32_batchsize_1_results and ./test_tipc/results/python_fcn_hrnetw18_small_results_fp32.txt are consistent!
Assert allclose passed! The results of python_infer_gpu_usetrt_True_precision_fp32_batchsize_1_results and ./test_tipc/results/python_fcn_hrnetw18_small_results_fp32.txt are consistent!
Assert allclose passed! The results of python_infer_cpu_usemkldnn_True_threads_6_precision_fp32_batchsize_1_results and ./test_tipc/results/python_fcn_hrnetw18_small_results_fp32.txt are consistent!
Assert allclose passed! The results of python_infer_cpu_usemkldnn_False_threads_6_precision_fp32_batchsize_1_results and ./test_tipc/results/python_fcn_hrnetw18_small_results_fp32.txt are consistent!
Assert allclose passed! The results of python_infer_gpu_usetrt_True_precision_fp16_batchsize_1_results and ./test_tipc/results/python_fcn_hrnetw18_small_results_fp16.txt are consistent!
```

出现不一致结果时的运行输出：
```
Not equal to tolerance rtol=0.001, atol=0.001

Mismatched elements: 1 / 1 (100%)
Max absolute difference: 0.01
Max relative difference: 0.01146091
 x: array(0.882531)
 y: array(0.872531)
Assert allclose failed! The results of python_infer_cpu_usemkldnn_False_threads_1_precision_fp32_batchsize_1_results and the results of ./test_tipc/results/python_fcn_hrnetw18_small_results_fp32.txt are inconsistent!

Not equal to tolerance rtol=0.001, atol=0.001

Mismatched elements: 1 / 1 (100%)
Max absolute difference: 0.01
Max relative difference: 0.01146091
 x: array(0.882531)
 y: array(0.872531)
Assert allclose failed! The results of python_infer_gpu_usetrt_False_precision_fp32_batchsize_1_results and the results of ./test_tipc/results/python_fcn_hrnetw18_small_results_fp32.txt are inconsistent!

......
```

## 3. 更多教程
本文档为功能测试用，更丰富的训练预测使用教程请参考：  
[模型训练](../../docs/whole_process_cn.md)  
[基于Python预测引擎推理](../../docs/deployment/inference/python_inference.md)
