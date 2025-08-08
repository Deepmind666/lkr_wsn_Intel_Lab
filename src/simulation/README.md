# 仿真环境目录

本目录包含WSN仿真环境的实现，用于模拟无线传感器网络的运行和评估不同路由协议的性能。

## 文件说明

### simulator.py

仿真器主类，负责协调整个仿真过程。

```python
class Simulator:
    def __init__(self, config):
        """初始化仿真器
        
        Args:
            config: 仿真配置参数
        """
        pass
        
    def setup(self):
        """设置仿真环境"""
        pass
        
    def run(self, num_rounds):
        """运行仿真
        
        Args:
            num_rounds: 仿真轮数
        """
        pass
        
    def collect_results(self):
        """收集仿真结果"""
        pass
        
    def reset(self):
        """重置仿真环境"""
        pass
```

### node.py

节点类，表示WSN中的传感器节点。

```python
class Node:
    def __init__(self, node_id, position, initial_energy, is_base_station=False):
        """初始化节点
        
        Args:
            node_id: 节点ID
            position: 节点位置坐标 (x, y)
            initial_energy: 初始能量
            is_base_station: 是否为基站
        """
        pass
        
    def sense(self):
        """感知数据"""
        pass
        
    def transmit(self, data, receiver):
        """发送数据
        
        Args:
            data: 要发送的数据
            receiver: 接收节点
        """
        pass
        
    def receive(self, data, sender):
        """接收数据
        
        Args:
            data: 接收到的数据
            sender: 发送节点
        """
        pass
        
    def predict(self, time_step):
        """预测数据
        
        Args:
            time_step: 时间步
        """
        pass
        
    def is_alive(self):
        """检查节点是否存活"""
        pass
```

### network.py

网络类，表示整个WSN网络。

```python
class Network:
    def __init__(self, nodes, base_station, connectivity=None):
        """初始化网络
        
        Args:
            nodes: 节点列表
            base_station: 基站节点
            connectivity: 连接矩阵
        """
        pass
        
    def setup_topology(self):
        """设置网络拓扑"""
        pass
        
    def apply_routing_protocol(self, protocol):
        """应用路由协议
        
        Args:
            protocol: 路由协议实例
        """
        pass
        
    def simulate_round(self):
        """模拟一轮网络运行"""
        pass
        
    def get_alive_nodes(self):
        """获取存活节点列表"""
        pass
        
    def get_energy_consumption(self):
        """获取能量消耗情况"""
        pass
        
    def get_network_lifetime(self):
        """获取网络生命周期"""
        pass
```

### energy_model.py

能量模型，用于计算节点的能量消耗。

```python
class EnergyModel:
    def __init__(self, params):
        """初始化能量模型
        
        Args:
            params: 能量模型参数
        """
        pass
        
    def calculate_tx_energy(self, packet_size, distance):
        """计算发送能量消耗
        
        Args:
            packet_size: 数据包大小
            distance: 传输距离
        """
        pass
        
    def calculate_rx_energy(self, packet_size):
        """计算接收能量消耗
        
        Args:
            packet_size: 数据包大小
        """
        pass
        
    def calculate_sensing_energy(self):
        """计算感知能量消耗"""
        pass
        
    def calculate_processing_energy(self, operation_type):
        """计算处理能量消耗
        
        Args:
            operation_type: 操作类型
        """
        pass
```

### run_simulation.py

运行仿真的入口脚本。

```python
# 主要功能
def load_config(config_file):
    """加载配置文件"""
    pass

def setup_simulation(config):
    """设置仿真环境"""
    pass

def run_experiments(simulator, protocols, num_rounds):
    """运行实验"""
    pass

def save_results(results, save_dir):
    """保存结果"""
    pass

if __name__ == '__main__':
    # 执行仿真
    pass
```

## 使用示例

```python
# 运行仿真示例
from simulation.simulator import Simulator
from simulation.network import Network
from simulation.node import Node
from simulation.energy_model import EnergyModel
from models.routing.leach import LEACH
from models.routing.meta_routing import MetaRouting

# 加载配置
config = {
    'area_size': (100, 100),
    'num_nodes': 100,
    'base_station_position': (50, 50),
    'initial_energy': 0.5,  # 单位：焦耳
    'energy_model_params': {
        'e_elec': 50e-9,  # 单位：J/bit
        'e_fs': 10e-12,   # 单位：J/bit/m^2
        'e_mp': 0.0013e-12,  # 单位：J/bit/m^4
        'e_da': 5e-9,     # 单位：J/bit
        'd0': 87,         # 单位：m
    },
    'packet_size': 4000,  # 单位：bit
}

# 创建仿真器
simulator = Simulator(config)

# 设置仿真环境
simulator.setup()

# 创建路由协议
leach = LEACH(simulator.network, {'p': 0.1})
meta_routing = MetaRouting(simulator.network, {'algorithm': 'pso'})

# 运行仿真
results_leach = simulator.run(leach, 1000)
simulator.reset()
results_meta = simulator.run(meta_routing, 1000)

# 比较结果
print(f'LEACH网络寿命: {results_leach["network_lifetime"]} 轮')
print(f'Meta-Routing网络寿命: {results_meta["network_lifetime"]} 轮')
```