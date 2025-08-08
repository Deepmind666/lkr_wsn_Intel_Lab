# 可视化工具目录

本目录包含用于可视化WSN网络拓扑、能量消耗和实验结果的工具。

## 文件说明

### network_visualizer.py

网络拓扑可视化工具，用于可视化WSN的网络结构、节点状态和数据流。

```python
class NetworkVisualizer:
    def __init__(self, network, figsize=(10, 8)):
        """初始化网络可视化器
        
        Args:
            network: 网络实例
            figsize: 图形大小
        """
        pass
        
    def visualize_topology(self, save_path=None):
        """可视化网络拓扑
        
        Args:
            save_path: 保存路径
        """
        pass
        
    def visualize_clusters(self, clusters, save_path=None):
        """可视化网络簇
        
        Args:
            clusters: 簇信息
            save_path: 保存路径
        """
        pass
        
    def visualize_data_flow(self, data_flow, save_path=None):
        """可视化数据流
        
        Args:
            data_flow: 数据流信息
            save_path: 保存路径
        """
        pass
        
    def visualize_node_status(self, node_status, save_path=None):
        """可视化节点状态
        
        Args:
            node_status: 节点状态信息
            save_path: 保存路径
        """
        pass
        
    def create_animation(self, simulation_data, save_path=None):
        """创建网络动画
        
        Args:
            simulation_data: 仿真数据
            save_path: 保存路径
        """
        pass
```

### energy_visualizer.py

能量消耗可视化工具，用于可视化WSN中节点的能量消耗情况。

```python
class EnergyVisualizer:
    def __init__(self, figsize=(12, 8)):
        """初始化能量可视化器
        
        Args:
            figsize: 图形大小
        """
        pass
        
    def visualize_energy_consumption(self, energy_data, save_path=None):
        """可视化能量消耗
        
        Args:
            energy_data: 能量消耗数据
            save_path: 保存路径
        """
        pass
        
    def visualize_energy_distribution(self, network, save_path=None):
        """可视化能量分布
        
        Args:
            network: 网络实例
            save_path: 保存路径
        """
        pass
        
    def visualize_energy_over_time(self, energy_history, save_path=None):
        """可视化能量随时间变化
        
        Args:
            energy_history: 能量历史数据
            save_path: 保存路径
        """
        pass
        
    def visualize_comparative_energy(self, protocols_energy, save_path=None):
        """可视化不同协议的能量消耗比较
        
        Args:
            protocols_energy: 不同协议的能量消耗数据
            save_path: 保存路径
        """
        pass
```

### results_visualizer.py

结果可视化工具，用于可视化实验结果和性能指标。

```python
class ResultsVisualizer:
    def __init__(self, figsize=(12, 10)):
        """初始化结果可视化器
        
        Args:
            figsize: 图形大小
        """
        pass
        
    def visualize_network_lifetime(self, lifetime_data, save_path=None):
        """可视化网络生命周期
        
        Args:
            lifetime_data: 网络生命周期数据
            save_path: 保存路径
        """
        pass
        
    def visualize_throughput(self, throughput_data, save_path=None):
        """可视化网络吞吐量
        
        Args:
            throughput_data: 吞吐量数据
            save_path: 保存路径
        """
        pass
        
    def visualize_packet_delivery_ratio(self, pdr_data, save_path=None):
        """可视化数据包传递率
        
        Args:
            pdr_data: 数据包传递率数据
            save_path: 保存路径
        """
        pass
        
    def visualize_latency(self, latency_data, save_path=None):
        """可视化延迟
        
        Args:
            latency_data: 延迟数据
            save_path: 保存路径
        """
        pass
        
    def visualize_prediction_accuracy(self, accuracy_data, save_path=None):
        """可视化预测准确性
        
        Args:
            accuracy_data: 预测准确性数据
            save_path: 保存路径
        """
        pass
        
    def visualize_reliability(self, reliability_data, save_path=None):
        """可视化可靠性
        
        Args:
            reliability_data: 可靠性数据
            save_path: 保存路径
        """
        pass
        
    def visualize_comparative_results(self, results_data, metrics, save_path=None):
        """可视化比较结果
        
        Args:
            results_data: 结果数据
            metrics: 指标列表
            save_path: 保存路径
        """
        pass
        
    def create_dashboard(self, simulation_results, save_path=None):
        """创建仪表盘
        
        Args:
            simulation_results: 仿真结果
            save_path: 保存路径
        """
        pass
```

## 使用示例

```python
# 网络拓扑可视化示例
from visualization.network_visualizer import NetworkVisualizer
from simulation.network import Network

# 创建网络
network = Network(...)

# 创建可视化器
visualizer = NetworkVisualizer(network)

# 可视化网络拓扑
visualizer.visualize_topology(save_path='../../results/network_topology.png')

# 可视化簇
clusters = {...}  # 簇信息
visualizer.visualize_clusters(clusters, save_path='../../results/network_clusters.png')

# 能量消耗可视化示例
from visualization.energy_visualizer import EnergyVisualizer

# 创建能量可视化器
energy_visualizer = EnergyVisualizer()

# 可视化能量消耗
energy_data = {...}  # 能量消耗数据
energy_visualizer.visualize_energy_consumption(energy_data, save_path='../../results/energy_consumption.png')

# 可视化能量随时间变化
energy_history = {...}  # 能量历史数据
energy_visualizer.visualize_energy_over_time(energy_history, save_path='../../results/energy_over_time.png')

# 结果可视化示例
from visualization.results_visualizer import ResultsVisualizer

# 创建结果可视化器
results_visualizer = ResultsVisualizer()

# 可视化网络生命周期
lifetime_data = {...}  # 网络生命周期数据
results_visualizer.visualize_network_lifetime(lifetime_data, save_path='../../results/network_lifetime.png')

# 可视化比较结果
results_data = {...}  # 结果数据
metrics = ['network_lifetime', 'throughput', 'energy_efficiency']
results_visualizer.visualize_comparative_results(results_data, metrics, save_path='../../results/comparative_results.png')
```