# WSN-Intel-Lab-Project 重要信息记录

## 项目与入口
- **根目录**: `D:\lkr_wsn\WSN-Intel-Lab-Project`
- **主入口**: `main.py`
- **系统核心**: `src/enhanced_eehfr_system.py` (`EnhancedEEHFRSystem`, `SystemConfig`)
- **实验脚本**:
  - 路由比较: `experiments/routing_comparison.py`
  - 综合评估: `experiments/comprehensive_algorithm_evaluation.py`

## 数据集
- **来源**: [Intel Berkeley Lab Sensor Data](https://db.csail.mit.edu/labdata/labdata.html)
- **下载**: `python main.py download`
- **预处理**: `python main.py preprocess`

## 运行命令（PC仿真）
- 安装依赖: `pip install -r requirements.txt`
- 路由比较实验: `python main.py experiment --experiment routing_comparison`
- 结果可视化（自动回退 CSV/JSON）: `python main.py visualize`
- 综合评估（含 AFW-RL / ILMR / EEHFR，GNN-CTO按需启用）:
  - `python main.py experiment --experiment comprehensive`
  - 若缺少 `torch_geometric` 会自动跳过 GNN-CTO 分支

## 结果产出位置
- 路由比较（脚本自带）: `experiments/results/data/*.csv`, `experiments/results/figures/*.{png,pdf}`
- 可视化（主入口）: `results/figures/*`
- 综合评估 JSON（主入口保存）: `results/data/comprehensive_results_*.json`

## 近期修复（已完成）
- `experiments/comprehensive_algorithm_evaluation.py`:
  - 修正导入至 `from src.enhanced_eehfr_system import ...`
  - 重命名 `save_results` → `save_detailed_results`
  - GNN-CTO 可选导入 + 安全降级（无 `torch_geometric` 时跳过）
- `main.py`:
  - `experiment` 分支：支持 `routing_comparison` 与 `comprehensive`
  - `visualize_results`：创建结果目录、回退到 `experiments/results/data`、支持从 CSV 直接出图，避免无 JSON 报错

## 待办（高优先级）
- 跑通 `--experiment comprehensive` 并输出综合 JSON + 图（在无 PyG 时应打印跳过 GNN-CTO 的提示）
- 文档对齐：更新 `README_RUN.md` 的脚本与命令，使其与现状一致
- 能耗模型标定：在 `src/metrics/energy_model.py` 标明目标硬件平台参数，并在实验中固定
- 依赖说明：为 GNN-CTO 单列 `requirements-gnn.txt`（`torch` + `torch_geometric` 安装指引）

## 论文复现实验指标
- PDR、端到端延迟、能耗（总/均值/方差）、FND/HND/LND、吞吐量、路由开销、可信度统计
- 多随机种子统计与置信区间；SoD 阈值敏感性与消融（信任/能量/SoD 开关）

## 备注
- 当前为 PC 端 Python 仿真主线；TRIM-RPL（Contiki-NG/COOJA）为后续可并行推进的工程化方向。


