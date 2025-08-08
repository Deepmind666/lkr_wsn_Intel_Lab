# WSN-Intel-Lab-Project 运行指南

本文档提供了如何运行WSN混合智能节能路由协议和比较实验的详细说明。

## 环境准备

1. 确保已安装Python 3.7+
2. 安装所需依赖库：

```bash
pip install -r requirements.txt
```

## 项目结构

项目的主要组件包括：

- `src/models/routing/hybrid_routing.py`：混合智能节能路由协议实现
- `src/models/metaheuristic/pso.py`：粒子群优化算法
- `src/models/time_series/lstm.py`：LSTM时序预测模型
- `src/models/reliability/fuzzy_logic.py`：模糊逻辑可靠性评估模型
- `experiments/scripts/compare_routing_protocols.py`：路由协议比较实验脚本
- `experiments/configs/routing_comparison.yaml`：实验配置文件
- `main.py`：项目主脚本

## 运行步骤

### 1. 下载数据集

使用以下命令下载Intel Berkeley Lab数据集：

```bash
python main.py download
```

如果需要强制重新下载，可以添加`--force-download`参数：

```bash
python main.py download --force-download
```

### 2. 预处理数据

使用以下命令预处理数据集：

```bash
python main.py preprocess
```

可选参数：
- `--normalize`：归一化方法，可选值为`minmax`、`standard`或`none`，默认为`minmax`
- `--fill-missing`：缺失值填充方法，可选值为`interpolate`、`mean`、`median`或`none`，默认为`interpolate`
- `--generate-features`：是否生成特征
- `--visualize`：是否可视化数据

例如：

```bash
python main.py preprocess --normalize minmax --fill-missing interpolate --generate-features --visualize
```

### 3. 运行路由协议比较实验

使用以下命令运行路由协议比较实验：

```bash
python main.py experiment --experiment routing_comparison
```

可以通过`--config`参数指定自定义配置文件：

```bash
python main.py experiment --experiment routing_comparison --config path/to/your/config.yaml
```

默认配置文件位于`experiments/configs/routing_comparison.yaml`。

### 4. 可视化实验结果

使用以下命令可视化最新的实验结果：

```bash
python main.py visualize
```

可以通过`--result-file`参数指定特定的结果文件：

```bash
python main.py visualize --result-file path/to/your/result.json
```

## 自定义实验配置

您可以通过修改`experiments/configs/routing_comparison.yaml`文件来自定义实验配置，包括：

- 网络参数：节点数量、区域大小、传输范围
- 仿真参数：数据包数量、源节点列表、汇聚节点、随机种子
- 协议参数：LEACH、PSO、混合智能节能路由协议的参数
- 输出设置：结果保存目录、图表保存目录

## 测试单个路由协议

如果您只想测试混合智能节能路由协议，可以直接运行：

```bash
python -c "from src.models.routing.hybrid_routing import test_hybrid_routing; test_hybrid_routing()"
```

## 实验结果解释

实验结果将保存在`results/data`目录中，包含以下性能指标：

- 网络生命周期：网络中节点的最低能量水平
- 数据包传递率：成功传递的数据包比例
- 端到端延迟：数据包从源节点到汇聚节点的平均延迟
- 预测准确性：时序预测模型的准确性
- 可靠性：路由路径的可靠性评估

可视化结果将保存在`results/figures`目录中，包括：

- 各项性能指标的条形图比较
- 能量消耗随时间变化的折线图
- 综合性能的雷达图

## 故障排除

如果遇到问题，请检查：

1. Python版本是否为3.7+
2. 是否已安装所有依赖库
3. 项目路径是否正确
4. 数据集是否已正确下载和预处理

如有其他问题，请参考项目文档或提交issue。