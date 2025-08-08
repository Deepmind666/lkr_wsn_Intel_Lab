# 工具函数目录

本目录包含项目中使用的各种工具函数，用于数据处理、网络操作、可视化等。

## 文件说明

### data_utils.py

数据处理相关的工具函数，包括：
- 数据加载和保存
- 数据清洗（处理缺失值、异常值）
- 数据归一化和标准化
- 特征工程
- 数据集划分（训练集、验证集、测试集）

```python
# 示例函数
def load_data(file_path):
    """加载数据集"""
    pass

def clean_data(data):
    """清洗数据，处理缺失值和异常值"""
    pass

def normalize_data(data, method='minmax'):
    """数据归一化"""
    pass

def split_data(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """划分数据集"""
    pass
```

### network_utils.py

网络相关的工具函数，包括：
- 网络拓扑生成
- 距离计算
- 能量消耗计算
- 路径查找

```python
# 示例函数
def generate_topology(num_nodes, area_size, distribution='random'):
    """生成网络拓扑"""
    pass

def calculate_distance(node1, node2):
    """计算两个节点之间的距离"""
    pass

def calculate_energy_consumption(packet_size, distance, params):
    """计算能量消耗"""
    pass

def find_path(source, destination, network):
    """查找从源节点到目标节点的路径"""
    pass
```

### visualization_utils.py

可视化相关的工具函数，包括：
- 网络拓扑可视化
- 能量消耗可视化
- 性能指标可视化

```python
# 示例函数
def visualize_network(network, node_colors=None, edge_colors=None):
    """可视化网络拓扑"""
    pass

def visualize_energy(energy_data, nodes=None):
    """可视化能量消耗"""
    pass

def visualize_performance(metrics, methods=None):
    """可视化性能指标"""
    pass
```

### download_dataset.py

用于下载Intel Berkeley Lab数据集的脚本。

```python
# 主要功能
def download_intel_lab_dataset(save_dir='../../data/raw'):
    """下载Intel Berkeley Lab数据集"""
    pass

def extract_dataset(zip_file, extract_dir):
    """解压数据集"""
    pass

if __name__ == '__main__':
    # 执行下载和解压
    pass
```

### preprocess_data.py

用于预处理Intel Berkeley Lab数据集的脚本。

```python
# 主要功能
def preprocess_intel_lab_dataset(raw_dir='../../data/raw', processed_dir='../../data/processed'):
    """预处理Intel Berkeley Lab数据集"""
    pass

def generate_features(data):
    """生成特征"""
    pass

if __name__ == '__main__':
    # 执行预处理
    pass
```

## 使用示例

```python
# 数据处理示例
from utils.data_utils import load_data, clean_data, normalize_data, split_data

# 加载数据
data = load_data('../../data/raw/data.txt')

# 清洗数据
cleaned_data = clean_data(data)

# 归一化数据
normalized_data = normalize_data(cleaned_data)

# 划分数据集
train_data, val_data, test_data = split_data(normalized_data)

# 网络操作示例
from utils.network_utils import generate_topology, calculate_distance

# 生成网络拓扑
network = generate_topology(54, (50, 50))

# 计算节点间距离
distance = calculate_distance(network.nodes[0], network.nodes[1])

# 可视化示例
from utils.visualization_utils import visualize_network

# 可视化网络拓扑
visualize_network(network)
```