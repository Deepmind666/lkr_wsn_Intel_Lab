# 算法模型目录

本目录包含项目中使用的各种算法模型实现，按功能分为不同的子目录。

## 目录结构

- `metaheuristic/` - 元启发式优化算法
  - `pso.py` - 粒子群优化算法
  - `ga.py` - 遗传算法
  - `aco.py` - 蚁群优化算法
  - `woa.py` - 鲸鱼优化算法
  - `avoa.py` - 非洲秃鹫优化算法
  - `hybrid.py` - 混合优化算法
- `prediction/` - 时序预测模型
  - `lstm.py` - LSTM模型
  - `gru.py` - GRU模型
  - `esn.py` - 回声状态网络模型
  - `transformer_light.py` - 轻量级Transformer模型
  - `dual_prediction.py` - 双重预测框架
- `reliability/` - 数据可靠性模型
  - `trust_model.py` - 信任模型
  - `fuzzy_logic.py` - 模糊逻辑评估
  - `confidence_interval.py` - 置信区间估计
  - `collaborative_verification.py` - 协同校验机制
- `routing/` - 路由协议实现
  - `leach.py` - LEACH协议
  - `meta_routing.py` - 基于元启发式优化的路由协议
  - `prediction_routing.py` - 融合时序预测的路由协议
  - `reliability_routing.py` - 考虑数据可靠性的路由协议
  - `hybrid_routing.py` - 混合路由协议

## 模型接口

所有模型都应遵循统一的接口规范，便于集成和测试：

### 元启发式优化算法

```python
class MetaHeuristicAlgorithm:
    def __init__(self, params):
        # 初始化算法参数
        pass
        
    def optimize(self, objective_function, constraints, initial_solution=None):
        # 执行优化过程
        pass
        
    def get_best_solution(self):
        # 获取最优解
        pass
```

### 时序预测模型

```python
class PredictionModel:
    def __init__(self, params):
        # 初始化模型参数
        pass
        
    def train(self, X_train, y_train):
        # 训练模型
        pass
        
    def predict(self, X_test):
        # 预测
        pass
        
    def evaluate(self, X_test, y_test):
        # 评估模型性能
        pass
```

### 数据可靠性模型

```python
class ReliabilityModel:
    def __init__(self, params):
        # 初始化模型参数
        pass
        
    def evaluate_reliability(self, data, context=None):
        # 评估数据可靠性
        pass
        
    def update(self, new_data, feedback=None):
        # 更新模型
        pass
```

### 路由协议

```python
class RoutingProtocol:
    def __init__(self, network, params):
        # 初始化协议参数
        pass
        
    def setup_phase(self):
        # 设置阶段（如簇头选择）
        pass
        
    def routing_phase(self):
        # 路由阶段
        pass
        
    def get_routing_table(self):
        # 获取路由表
        pass
```