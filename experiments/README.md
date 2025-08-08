# 实验目录

本目录包含项目的实验配置和结果，用于评估不同算法和方法的性能。

## 目录结构

- `configs/` - 实验配置文件
  - `baseline/` - 基准实验配置
  - `metaheuristic/` - 元启发式算法实验配置
  - `prediction/` - 时序预测实验配置
  - `reliability/` - 数据可靠性实验配置
  - `hybrid/` - 混合方法实验配置
- `scripts/` - 实验脚本
  - `run_baseline.py` - 运行基准实验
  - `run_metaheuristic.py` - 运行元启发式算法实验
  - `run_prediction.py` - 运行时序预测实验
  - `run_reliability.py` - 运行数据可靠性实验
  - `run_hybrid.py` - 运行混合方法实验
  - `run_all.py` - 运行所有实验
- `results/` - 实验结果
  - `baseline/` - 基准实验结果
  - `metaheuristic/` - 元启发式算法实验结果
  - `prediction/` - 时序预测实验结果
  - `reliability/` - 数据可靠性实验结果
  - `hybrid/` - 混合方法实验结果
  - `comparative/` - 比较实验结果

## 实验配置

实验配置文件采用YAML或JSON格式，包含以下内容：

```yaml
# 实验基本信息
experiment_name: "LEACH基准实验"
description: "LEACH协议在Intel Lab数据集上的基准性能评估"
date: "2025-07-30"

# 网络配置
network:
  dataset: "intel_lab"
  num_nodes: 54
  area_size: [50, 50]  # 单位：米
  base_station_position: [25, 25]  # 单位：米
  initial_energy: 0.5  # 单位：焦耳

# 能量模型配置
energy_model:
  e_elec: 50.0e-9  # 单位：J/bit
  e_fs: 10.0e-12   # 单位：J/bit/m^2
  e_mp: 0.0013e-12  # 单位：J/bit/m^4
  e_da: 5.0e-9     # 单位：J/bit
  d0: 87.0         # 单位：米

# 数据配置
data:
  packet_size: 4000  # 单位：bit
  header_size: 200   # 单位：bit
  sensing_interval: 31  # 单位：秒

# 路由协议配置
routing_protocol:
  name: "LEACH"
  params:
    p: 0.1  # 簇头概率

# 仿真配置
simulation:
  num_rounds: 1000
  seed: 42
  save_interval: 10  # 每隔多少轮保存一次状态

# 评估指标
metrics:
  - "network_lifetime"
  - "energy_consumption"
  - "throughput"
  - "packet_delivery_ratio"
  - "latency"
```

## 实验脚本

实验脚本用于运行实验并收集结果，示例如下：

```python
# run_baseline.py

import os
import yaml
import argparse
import numpy as np
from datetime import datetime

from src.simulation.simulator import Simulator
from src.models.routing.leach import LEACH

def run_experiment(config_file):
    # 加载配置
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建结果目录
    result_dir = os.path.join('results', 'baseline', datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(result_dir, exist_ok=True)
    
    # 保存配置
    with open(os.path.join(result_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # 创建仿真器
    simulator = Simulator(config)
    
    # 设置仿真环境
    simulator.setup()
    
    # 创建路由协议
    protocol_config = config['routing_protocol']
    if protocol_config['name'] == 'LEACH':
        protocol = LEACH(simulator.network, protocol_config['params'])
    else:
        raise ValueError(f"不支持的路由协议: {protocol_config['name']}")
    
    # 运行仿真
    results = simulator.run(protocol, config['simulation']['num_rounds'])
    
    # 保存结果
    np.save(os.path.join(result_dir, 'results.npy'), results)
    
    # 生成报告
    generate_report(results, config, os.path.join(result_dir, 'report.md'))
    
    return results

def generate_report(results, config, save_path):
    # 生成实验报告
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='运行基准实验')
    parser.add_argument('--config', type=str, default='configs/baseline/leach.yaml', help='配置文件路径')
    args = parser.parse_args()
    
    run_experiment(args.config)
```

## 实验结果

实验结果包括以下内容：

1. **网络生命周期**：网络中第一个节点死亡的轮数、50%节点死亡的轮数、最后一个节点死亡的轮数
2. **能量消耗**：每轮的平均能量消耗、总能量消耗、能量消耗分布
3. **吞吐量**：成功传输到基站的数据包数量
4. **数据包传递率**：成功传输的数据包占总数据包的比例
5. **延迟**：数据包从源节点到基站的平均传输时间
6. **预测准确性**：时序预测模型的准确性指标（如RMSE、MAE等）
7. **可靠性**：数据可靠性评估指标

## 比较实验

比较实验用于对比不同方法的性能，包括：

1. **基准方法与元启发式优化方法的比较**：比较LEACH、PSO-LEACH、GA-LEACH等
2. **有无时序预测的比较**：比较使用时序预测和不使用时序预测的性能差异
3. **有无数据可靠性评估的比较**：比较考虑数据可靠性和不考虑数据可靠性的性能差异
4. **混合方法与单一方法的比较**：比较混合方法与各单一方法的性能差异

## 使用示例

```bash
# 运行LEACH基准实验
python experiments/scripts/run_baseline.py --config experiments/configs/baseline/leach.yaml

# 运行PSO优化的路由实验
python experiments/scripts/run_metaheuristic.py --config experiments/configs/metaheuristic/pso.yaml

# 运行所有实验
python experiments/scripts/run_all.py
```