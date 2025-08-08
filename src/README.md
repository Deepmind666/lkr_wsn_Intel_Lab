# 源代码目录

本目录包含项目的所有源代码，按功能模块划分为不同的子目录。

## 目录结构

- `models/` - 算法模型实现
  - `metaheuristic/` - 元启发式优化算法
  - `prediction/` - 时序预测模型
  - `reliability/` - 数据可靠性模型
  - `routing/` - 路由协议实现
- `utils/` - 工具函数
  - `data_utils.py` - 数据处理工具
  - `network_utils.py` - 网络相关工具
  - `visualization_utils.py` - 可视化工具
  - `download_dataset.py` - 数据集下载工具
  - `preprocess_data.py` - 数据预处理工具
- `simulation/` - 仿真环境
  - `simulator.py` - 仿真器主类
  - `node.py` - 节点类
  - `network.py` - 网络类
  - `energy_model.py` - 能量模型
  - `run_simulation.py` - 运行仿真的入口脚本
- `visualization/` - 可视化工具
  - `network_visualizer.py` - 网络拓扑可视化
  - `results_visualizer.py` - 结果可视化
  - `energy_visualizer.py` - 能量消耗可视化

## 主要模块说明

### 元启发式优化算法

实现了多种元启发式优化算法，用于优化WSN路由，包括：
- 粒子群优化算法 (PSO)
- 遗传算法 (GA)
- 蚁群优化算法 (ACO)
- 鲸鱼优化算法 (WOA)
- 混合优化算法

### 时序预测模型

实现了轻量级时序预测模型，用于减少不必要的数据传输，包括：
- LSTM模型
- GRU模型
- ESN模型（回声状态网络）
- 轻量级Transformer模型

### 数据可靠性模型

实现了数据可靠性评估模型，用于提高网络的鲁棒性，包括：
- 信任模型
- 模糊逻辑评估
- 置信区间估计
- 协同校验机制

### 路由协议

实现了多种路由协议，包括：
- LEACH协议及其变种
- 基于元启发式优化的路由协议
- 融合时序预测的路由协议
- 考虑数据可靠性的路由协议

### 仿真环境

提供了一个完整的WSN仿真环境，支持：
- 节点部署和网络拓扑构建
- 能量消耗模型
- 数据传输模拟
- 路由协议评估
- 性能指标计算