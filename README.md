# 2019 未来杯高校AI挑战赛 区域赛作品

* 战队编号: {690}
* 战队名称: {提交什么都}
* 战队成员: {huzi96，AstroJacobLi，杨礼铭}

## 概述

本方案使用res-block搭建了U-Net结构，总共进行3次降采样，每一级别采样均使用res-block。在相同的采样尺度上，使用跳跃连接(skip connection)，输入为b和c两张图像，输出为对于各个点可能存在目标的概率。

## 系统要求

### 硬件环境要求

* CPU: Intel i7-5930K
* GPU: NVIDIA GTX 1080
* 内存: 32G
* 硬盘: >= 50G
* 其他: 

### 软件环境要求

* 操作系统: Ubuntu 16.04
* CUDA 9.0
* TensorFlow 1.12
* astropy 2.0.2
* opencv-python 3.4.2

### 数据集

如使用官方提供的数据集之外的数据，请在此说明。

## 数据预处理

### 方法概述

用于训练的图像，随机选取数据集中的图像，随机选取128x128的区域，选入数据集。如果数据集中包含目标点，在用于训练的Ground Truth图上对应位置周围半径为7的位置上按照高斯分布放置一个响应点。

### 操作步骤

a. 使用 collect.py 从所有图像中产生打包的图像
b. 使用 make_dataset.py 产生hdf5文件，用于训练

### 模型

训练后的模型存储地址：./

模型文件大小：58M



## 训练

### 训练方法概述

自动训练

### 训练操作步骤

准备好h5文件
执行 python unify_v2.py

### 训练结果保存与获取

模型自动保存在 ./ckpt/

## 测试

### 方法概述

首先产生概率相应图，然后综合产生csv文件

### 操作步骤

第一步: python test.py {checkpoint name} {response_output_name} {'test_pickle_filename'} 
第二步: python jiaxuan_find_output.py {response_output_name} {csv_name}
