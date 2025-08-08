# WSN-Intel-Lab-Project

## 项目概述

本项目基于Intel Berkeley Lab数据集，设计并实现一套适合本地模拟的、高数学严谨性的、聚焦于能耗优化 + 时序预测 + 数据可靠性的融合算法框架。项目旨在解决无线传感器网络(WSN)中的能源效率、数据预测和可靠性问题。

## 数据集

本项目使用Intel Berkeley Lab数据集，该数据集包含54个静态节点，传感温度、湿度、光照、电压，每31秒采样一次。

数据集下载地址：http://db.csail.mit.edu/labdata/labdata.html

## 项目结构

```
├── data/                  # 数据集和预处理后的数据
├── src/                   # 源代码
│   ├── models/            # 算法模型实现
│   ├── utils/             # 工具函数
│   ├── simulation/        # 仿真环境
│   └── visualization/     # 可视化工具
├── experiments/           # 实验配置和结果
├── docs/                  # 文档
├── tests/                 # 测试代码
└── results/               # 实验结果和图表
```

## 主要功能

1. **能耗优化**：使用元启发式算法优化WSN路由，延长网络寿命
2. **时序预测**：实现轻量级时序预测模型，减少不必要的数据传输
3. **数据可靠性**：建立数据可靠性模型，提高网络的鲁棒性

## 技术栈

- Python 3.8+
- NumPy, Pandas, Matplotlib
- PyTorch/TensorFlow (用于时序预测模型)
- NetworkX (用于网络拓扑建模)

## 安装与使用

```bash
# 克隆仓库
git clone [repository-url]

# 安装依赖
pip install -r requirements.txt

# 下载数据集
python src/utils/download_dataset.py

# 运行仿真
python src/simulation/run_simulation.py
```

## 贡献指南

请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解如何为本项目做出贡献。

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。"# lkr_wsn_Intel_Lab" 
