# WSN-Intel-Lab-Project

面向无线传感器网络（WSN）的能耗优化 + 时序预测 + 数据可靠性一体化研究与可复现实验平台。

- 聚焦：能源效率、预测/压缩上报、可信与鲁棒性
- 特性：模块化算法、批量实验、论文级可视化（SVG/PDF）、一键重现实验
- 数据：Intel Berkeley Lab（54个节点，温/湿/光/电，≈31s采样）

## 目录
- 项目亮点
- 快速开始（安装/运行）
- 核心目录结构
- 论文级可视化与“论文模式”
- 数据集与引用
- 故障排除
- 贡献与许可

## 项目亮点
- 一体化框架：能耗优化、时序预测、路径可信度评估协同设计
- 可复现性：固定随机种子、配置化实验、CSV结果、SVG/PDF图表
- 实证度量：PDR、端到端时延、Tx/Rx开销、能量分解、跳数等
- 论文输出：一键启用“论文模式”，固定场景顺序、默认白名单、导出PDF版本

## 快速开始
1) 环境与依赖（Python 3.8+，建议 3.10/3.11）：
```bash
pip install -r requirements.txt
```

2) 下载与预处理数据：
```bash
python main.py download
python main.py preprocess
```

3) 运行基线对比实验：
```bash
python main.py experiment --experiment routing_comparison
```

4) 可视化最近一次结果：
```bash
python main.py visualize
```

5) Augment 工作区（研究与论文复现实验）：
```bash
# 数据包级别仿真（示例）
python augment/run_packet_sim.py --help

# 生成概览图（启用论文模式：固定顺序+默认白名单+导出PDF）
python -m augment.plots.make_figures --overview --paper-mode
```

## 核心目录结构
```
augment/           # 研究增强工作区：仿真、批量运行、图表与报告
  ├─ experiments/  # 可复现实验与脚本
  ├─ plots/        # 论文级可视化与导出（含 paper-mode）
  ├─ results/      # 运行输出（CSV/JSON、SVG/PDF 图表）
  └─ simulation/   # 事件/时隙驱动的最小化数据包级仿真器
experiments/       # 主项目实验与配置
results/           # 主项目结果（data/、figures/ 等）
src/               # 模型与协议、仿真、可视化等模块
docs/              # 论文与技术文档
```

## 论文级可视化与“论文模式”
- 概览图指令：
```bash
python -m augment.plots.make_figures --overview --paper-mode
```
- 关键开关：
  - `--paper-mode`：启用论文模式（固定场景顺序、默认攻击白名单、额外导出 PDF）
  - `--attacks-whitelist`：手动指定展示的场景标签（逗号分隔）
  - `--attacks-whitelist-file`：从文件读取场景白名单（每行一个）
- 输出位置：`results/figures/`（包含 `attacks_pdr.*`、`attacks_delay.*`、`attacks_txrx.*`）

> 说明：论文模式默认与表格/正文一致的场景集合与排序，导出 SVG+PDF，便于投稿与后续编辑。

## 数据集与引用
- Intel Berkeley Lab 数据集（54节点，温/湿/光/电，≈31s）：
  - 下载页：http://db.csail.mit.edu/labdata/labdata.html

如在论文/报告中使用本项目，请引用本仓库并注明数据来源。

## 故障排除
- CRLF/LF 警告：属正常换行提示，不影响功能
- Git 提交失败（缺少用户名/邮箱）：
```bash
git config user.name "YourName"
git config user.email "you@example.com"
```
- 找不到输出图表：确认已运行可视化命令，并查看 `results/figures/`

## 贡献与许可
- 欢迎提交 Issue / PR，共建高质量可复现实验基座
- 许可协议：MIT（详见 LICENSE）
