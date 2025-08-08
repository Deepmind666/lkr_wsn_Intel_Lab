# 结果目录

本目录用于存储实验结果和可视化图表，包括网络性能指标、能量消耗数据、预测准确性评估和比较分析等。

## 目录结构

- `figures/` - 可视化图表
  - `network/` - 网络拓扑和状态图
  - `energy/` - 能量消耗图
  - `performance/` - 性能指标图
  - `prediction/` - 预测结果图
  - `comparison/` - 比较分析图
- `data/` - 结果数据
  - `baseline/` - 基准实验数据
  - `metaheuristic/` - 元启发式算法实验数据
  - `prediction/` - 时序预测实验数据
  - `reliability/` - 数据可靠性实验数据
  - `hybrid/` - 混合方法实验数据
- `reports/` - 实验报告
  - `baseline_report.md` - 基准实验报告
  - `metaheuristic_report.md` - 元启发式算法实验报告
  - `prediction_report.md` - 时序预测实验报告
  - `reliability_report.md` - 数据可靠性实验报告
  - `hybrid_report.md` - 混合方法实验报告
  - `comparative_report.md` - 比较分析报告

## 结果数据格式

结果数据以CSV、JSON或NumPy数组格式保存，包含以下内容：

### 网络性能数据

```json
{
  "experiment_info": {
    "name": "LEACH基准实验",
    "date": "2025-07-30",
    "protocol": "LEACH",
    "parameters": {
      "p": 0.1
    }
  },
  "network_lifetime": {
    "first_node_death": 245,
    "half_nodes_death": 487,
    "last_node_death": 892
  },
  "energy_consumption": {
    "total": 27.5,
    "average_per_round": 0.0275,
    "per_round": [0.03, 0.028, 0.027, ...]
  },
  "throughput": {
    "total_packets": 54000,
    "successful_packets": 52380,
    "packets_per_round": [54, 54, 53, ...]
  },
  "packet_delivery_ratio": 0.97,
  "latency": {
    "average": 0.015,
    "max": 0.045,
    "min": 0.005
  }
}
```

### 预测性能数据

```json
{
  "experiment_info": {
    "name": "LSTM预测实验",
    "date": "2025-07-30",
    "model": "LSTM",
    "parameters": {
      "hidden_size": 64,
      "num_layers": 2
    }
  },
  "prediction_accuracy": {
    "rmse": 0.15,
    "mae": 0.12,
    "r2": 0.85
  },
  "data_reduction": {
    "total_samples": 54000,
    "transmitted_samples": 12960,
    "reduction_ratio": 0.76
  },
  "energy_savings": {
    "with_prediction": 18.2,
    "without_prediction": 27.5,
    "savings_ratio": 0.34
  },
  "prediction_errors": {
    "temperature": 0.18,
    "humidity": 0.22,
    "light": 0.35,
    "voltage": 0.05
  }
}
```

### 可靠性评估数据

```json
{
  "experiment_info": {
    "name": "模糊信任评估实验",
    "date": "2025-07-30",
    "model": "FuzzyTrust",
    "parameters": {
      "threshold": 0.7
    }
  },
  "reliability_metrics": {
    "average_trust": 0.82,
    "min_trust": 0.45,
    "max_trust": 0.98
  },
  "attack_detection": {
    "true_positives": 45,
    "false_positives": 3,
    "true_negatives": 50,
    "false_negatives": 2,
    "accuracy": 0.95,
    "precision": 0.94,
    "recall": 0.96,
    "f1_score": 0.95
  },
  "data_quality": {
    "before_filtering": {
      "rmse": 0.35,
      "mae": 0.28
    },
    "after_filtering": {
      "rmse": 0.12,
      "mae": 0.09
    },
    "improvement_ratio": 0.66
  }
}
```

## 可视化图表

可视化图表以PNG、SVG或PDF格式保存，包含以下内容：

### 网络拓扑和状态图

- 网络拓扑图：显示节点位置和连接关系
- 簇形成图：显示簇头和簇成员
- 路由路径图：显示数据传输路径
- 节点状态图：显示节点的存活状态和能量水平

### 能量消耗图

- 能量消耗随时间变化图：显示网络总能量消耗随时间的变化
- 节点能量分布图：显示各节点的能量水平分布
- 能量消耗热力图：显示网络中能量消耗的空间分布

### 性能指标图

- 网络生命周期图：显示不同协议的网络生命周期比较
- 吞吐量图：显示不同协议的吞吐量比较
- 数据包传递率图：显示不同协议的数据包传递率比较
- 延迟图：显示不同协议的延迟比较

### 预测结果图

- 预测vs实际值图：显示预测值和实际值的比较
- 预测误差分布图：显示预测误差的分布
- 数据传输减少图：显示使用预测后数据传输的减少情况

### 比较分析图

- 综合性能雷达图：显示不同方法在多个指标上的比较
- 能量效率vs数据质量图：显示能量效率和数据质量之间的权衡
- 可靠性vs能量消耗图：显示可靠性和能量消耗之间的权衡

## 实验报告

实验报告以Markdown格式保存，包含以下内容：

### 基准实验报告

```markdown
# LEACH基准实验报告

## 实验信息

- 实验名称：LEACH基准实验
- 实验日期：2025-07-30
- 实验协议：LEACH
- 协议参数：p = 0.1

## 实验设置

- 网络规模：54个节点
- 区域大小：50m x 50m
- 基站位置：(25, 25)
- 初始能量：0.5J
- 数据包大小：4000bit
- 仿真轮数：1000轮

## 实验结果

### 网络生命周期

- 第一个节点死亡轮数：245轮
- 50%节点死亡轮数：487轮
- 最后一个节点死亡轮数：892轮

### 能量消耗

- 总能量消耗：27.5J
- 平均每轮能量消耗：0.0275J

### 吞吐量

- 总数据包数：54000个
- 成功传输数据包数：52380个
- 数据包传递率：97%

### 延迟

- 平均延迟：0.015s
- 最大延迟：0.045s
- 最小延迟：0.005s

## 结果分析

LEACH协议通过随机轮换簇头来平衡网络负载，但由于簇头选择没有考虑节点的剩余能量和位置，导致能量消耗不均衡。从结果可以看出，网络中第一个节点在245轮后就死亡了，而最后一个节点在892轮后才死亡，说明节点间的能量消耗差异较大。

数据包传递率达到了97%，说明LEACH协议在数据传输可靠性方面表现良好。但是，随着节点的死亡，网络的连通性会下降，导致数据包传递率在后期会有所下降。

## 结论与建议

LEACH协议作为一种经典的分层路由协议，在能量效率和数据传输可靠性方面表现不错，但仍有改进空间。建议在以下几个方面进行改进：

1. 考虑节点的剩余能量和位置信息来选择簇头，以平衡网络负载
2. 优化簇内和簇间通信，减少能量消耗
3. 引入数据预测机制，减少不必要的数据传输
4. 考虑数据可靠性，提高网络的鲁棒性

## 附图

![网络拓扑图](../figures/network/leach_topology.png)
![能量消耗图](../figures/energy/leach_energy_consumption.png)
![节点存活图](../figures/performance/leach_node_alive.png)
```

## 使用说明

1. 实验结果会自动保存在本目录中
2. 可以使用可视化工具查看和分析结果
3. 可以使用比较工具比较不同方法的性能

```python
# 查看实验结果示例
import json
import matplotlib.pyplot as plt

# 加载结果数据
with open('data/baseline/leach_results.json', 'r') as f:
    leach_results = json.load(f)

with open('data/metaheuristic/pso_results.json', 'r') as f:
    pso_results = json.load(f)

# 比较网络生命周期
protocols = ['LEACH', 'PSO-LEACH']
first_node_death = [leach_results['network_lifetime']['first_node_death'], 
                   pso_results['network_lifetime']['first_node_death']]
last_node_death = [leach_results['network_lifetime']['last_node_death'], 
                  pso_results['network_lifetime']['last_node_death']]

plt.figure(figsize=(10, 6))
plt.bar(protocols, first_node_death, label='First Node Death')
plt.bar(protocols, last_node_death, bottom=first_node_death, label='Network Lifetime')
plt.xlabel('Protocols')
plt.ylabel('Rounds')
plt.title('Network Lifetime Comparison')
plt.legend()
plt.savefig('figures/comparison/network_lifetime_comparison.png')
plt.show()
```