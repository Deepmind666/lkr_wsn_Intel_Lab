# 测试目录

本目录包含项目的测试代码，用于验证各个模块的功能和性能。

## 目录结构

- `unit/` - 单元测试
  - `models/` - 模型单元测试
    - `test_metaheuristic.py` - 元启发式优化算法测试
    - `test_prediction.py` - 时序预测模型测试
    - `test_reliability.py` - 数据可靠性模型测试
    - `test_routing.py` - 路由协议测试
  - `utils/` - 工具函数单元测试
    - `test_data_utils.py` - 数据处理工具测试
    - `test_network_utils.py` - 网络工具测试
    - `test_visualization_utils.py` - 可视化工具测试
  - `simulation/` - 仿真环境单元测试
    - `test_simulator.py` - 仿真器测试
    - `test_node.py` - 节点测试
    - `test_network.py` - 网络测试
    - `test_energy_model.py` - 能量模型测试
- `integration/` - 集成测试
  - `test_data_pipeline.py` - 数据处理流程测试
  - `test_simulation_pipeline.py` - 仿真流程测试
  - `test_visualization_pipeline.py` - 可视化流程测试
- `performance/` - 性能测试
  - `test_algorithm_performance.py` - 算法性能测试
  - `test_simulation_performance.py` - 仿真性能测试
- `fixtures/` - 测试数据和固定装置
  - `sample_data.py` - 样本数据
  - `sample_network.py` - 样本网络
- `conftest.py` - pytest配置文件

## 测试框架

本项目使用pytest作为测试框架，使用pytest-cov进行代码覆盖率分析。

## 单元测试

单元测试用于测试各个模块的独立功能，确保每个模块都能正确工作。

### 模型单元测试

模型单元测试用于测试各种算法模型的功能。

```python
# test_metaheuristic.py

import pytest
import numpy as np
from src.models.metaheuristic.pso import PSO

def test_pso_initialization():
    """测试PSO初始化"""
    pso = PSO(n_particles=10, dimensions=2)
    assert pso.n_particles == 10
    assert pso.dimensions == 2
    assert pso.particles.shape == (10, 2)

def test_pso_optimization():
    """测试PSO优化"""
    # 定义目标函数（Sphere函数）
    def sphere(x):
        return np.sum(x**2)
    
    pso = PSO(n_particles=20, dimensions=5)
    best_position, best_fitness = pso.optimize(sphere, max_iterations=100)
    
    # 检查最优解是否接近0
    assert np.all(np.abs(best_position) < 0.1)
    assert best_fitness < 0.01
```

### 工具函数单元测试

工具函数单元测试用于测试各种工具函数的功能。

```python
# test_data_utils.py

import pytest
import numpy as np
import pandas as pd
from src.utils.data_utils import load_data, clean_data, normalize_data, split_data

def test_load_data(tmp_path):
    """测试数据加载"""
    # 创建测试数据文件
    data = pd.DataFrame({
        'date': pd.date_range(start='2004-02-28', periods=10),
        'temperature': np.random.rand(10) * 30,
        'humidity': np.random.rand(10) * 100,
        'light': np.random.rand(10) * 1000,
        'voltage': np.random.rand(10) * 3
    })
    data_file = tmp_path / "test_data.csv"
    data.to_csv(data_file, index=False)
    
    # 测试加载数据
    loaded_data = load_data(data_file)
    assert isinstance(loaded_data, pd.DataFrame)
    assert loaded_data.shape == (10, 5)
    assert all(col in loaded_data.columns for col in ['date', 'temperature', 'humidity', 'light', 'voltage'])

def test_clean_data():
    """测试数据清洗"""
    # 创建带有缺失值和异常值的测试数据
    data = pd.DataFrame({
        'temperature': [25, np.nan, 30, 1000, 20],
        'humidity': [50, 60, np.nan, 70, 80]
    })
    
    # 测试清洗数据
    cleaned_data = clean_data(data)
    assert isinstance(cleaned_data, pd.DataFrame)
    assert cleaned_data.shape[0] <= data.shape[0]  # 可能会删除一些行
    assert not cleaned_data.isna().any().any()  # 没有缺失值
    assert cleaned_data['temperature'].max() < 100  # 异常值已处理
```

### 仿真环境单元测试

仿真环境单元测试用于测试仿真环境的各个组件。

```python
# test_node.py

import pytest
import numpy as np
from src.simulation.node import Node

def test_node_initialization():
    """测试节点初始化"""
    node = Node(node_id=1, position=(10, 20), initial_energy=0.5)
    assert node.node_id == 1
    assert node.position == (10, 20)
    assert node.energy == 0.5
    assert not node.is_base_station

def test_node_energy_consumption():
    """测试节点能量消耗"""
    node = Node(node_id=1, position=(10, 20), initial_energy=0.5)
    initial_energy = node.energy
    
    # 模拟发送数据
    node.transmit(data_size=1000, distance=50)
    assert node.energy < initial_energy
    
    # 模拟接收数据
    current_energy = node.energy
    node.receive(data_size=1000)
    assert node.energy < current_energy
```

## 集成测试

集成测试用于测试多个模块之间的交互，确保它们能够协同工作。

```python
# test_simulation_pipeline.py

import pytest
from src.simulation.simulator import Simulator
from src.models.routing.leach import LEACH

def test_simulation_pipeline():
    """测试仿真流程"""
    # 创建仿真配置
    config = {
        'area_size': (100, 100),
        'num_nodes': 20,
        'base_station_position': (50, 50),
        'initial_energy': 0.5,
        'energy_model_params': {
            'e_elec': 50e-9,
            'e_fs': 10e-12,
            'e_mp': 0.0013e-12,
            'e_da': 5e-9,
            'd0': 87,
        },
        'packet_size': 4000,
    }
    
    # 创建仿真器
    simulator = Simulator(config)
    simulator.setup()
    
    # 创建路由协议
    leach = LEACH(simulator.network, {'p': 0.1})
    
    # 运行仿真
    results = simulator.run(leach, num_rounds=10)
    
    # 检查结果
    assert 'network_lifetime' in results
    assert 'energy_consumption' in results
    assert 'throughput' in results
    assert len(results['energy_consumption']) == 10  # 10轮的能量消耗数据
```

## 性能测试

性能测试用于测试各个模块的性能，确保它们能够高效运行。

```python
# test_algorithm_performance.py

import pytest
import time
import numpy as np
from src.models.metaheuristic.pso import PSO
from src.models.metaheuristic.ga import GA

def test_pso_performance():
    """测试PSO性能"""
    # 定义目标函数（Sphere函数）
    def sphere(x):
        return np.sum(x**2)
    
    # 测试PSO性能
    pso = PSO(n_particles=30, dimensions=10)
    start_time = time.time()
    pso.optimize(sphere, max_iterations=100)
    end_time = time.time()
    
    # 检查运行时间
    assert end_time - start_time < 1.0  # 运行时间应小于1秒

def test_ga_performance():
    """测试GA性能"""
    # 定义目标函数（Sphere函数）
    def sphere(x):
        return np.sum(x**2)
    
    # 测试GA性能
    ga = GA(population_size=50, chromosome_length=10)
    start_time = time.time()
    ga.optimize(sphere, max_generations=100)
    end_time = time.time()
    
    # 检查运行时间
    assert end_time - start_time < 2.0  # 运行时间应小于2秒
```

## 运行测试

可以使用以下命令运行测试：

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/unit/models/test_metaheuristic.py

# 运行特定测试函数
pytest tests/unit/models/test_metaheuristic.py::test_pso_optimization

# 生成代码覆盖率报告
pytest --cov=src tests/
```

## 测试覆盖率

测试覆盖率报告可以帮助我们了解测试覆盖了多少代码，以及哪些部分需要更多的测试。

```bash
# 生成HTML格式的代码覆盖率报告
pytest --cov=src --cov-report=html tests/
```

生成的报告将保存在`htmlcov`目录中，可以在浏览器中打开`htmlcov/index.html`查看。