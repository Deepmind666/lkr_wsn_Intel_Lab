# 数据目录

本目录用于存储Intel Berkeley Lab数据集及其预处理后的数据。

## 数据集说明

Intel Berkeley Lab数据集包含54个静态节点，传感温度、湿度、光照、电压，每31秒采样一次。

## 数据集结构

- `raw/` - 原始数据集
  - `data.txt` - 原始数据文件
  - `connectivity.txt` - 节点连接信息
  - `topology.txt` - 网络拓扑信息
- `processed/` - 预处理后的数据
  - `cleaned_data.csv` - 清洗后的数据
  - `normalized_data.csv` - 归一化后的数据
  - `train_test_split/` - 训练集和测试集

## 数据集下载

可以通过以下链接下载Intel Berkeley Lab数据集：
http://db.csail.mit.edu/labdata/labdata.html

或者使用项目提供的下载脚本：
```bash
python ../src/utils/download_dataset.py
```

## 数据预处理

数据预处理步骤包括：
1. 清洗缺失值和异常值
2. 数据归一化
3. 特征工程
4. 训练集和测试集划分

可以使用以下脚本进行数据预处理：
```bash
python ../src/utils/preprocess_data.py
```