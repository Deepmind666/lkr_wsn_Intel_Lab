# 增强研究工作区（augment/）

在不修改主项目模块的前提下，提供可复现、研究级别的数据包级仿真、批量实验与论文级图表导出能力，所有新增代码与报告均隔离在 augment/ 下，便于溯源与复现。

结构
- simulation/：最小化数据包级仿真器与度量工具（事件/时隙驱动）
- experiments/：批量运行与实验脚本
- plots/：论文级图表与导出（支持 --paper-mode）
- results/：实验输出（CSV/JSON、SVG/PDF 图表）
- configs/：YAML/JSON 配置，支持实验复现
- run_packet_sim.py：仿真入口（CLI）

原则
- 可复现：固定随机种子、配置化实验、CSV 输出
- 可度量：PDR、端到端时延、跳数、Tx/Rx 开销、能量分解
- 保守结论：不夸大效果，所有结论均有度量支撑
- 非侵入：不修改 augment/ 之外的模块

快速开始
- 查看仿真器参数选项：
  `python augment/run_packet_sim.py --help`
- 生成概览图（固定顺序 + 默认白名单 + 导出 PDF）：
  `python -m augment.plots.make_figures --overview --paper-mode`
- 自定义展示场景白名单：
  - `--attacks-whitelist`：逗号分隔的场景标签列表
  - `--attacks-whitelist-file`：从文件读取（每行一个标签）

输出位置
- 图表统一导出到 `augment/results/figures/` 或项目根 `results/figures/`（视脚本而定）
- 常见概览图文件名：`attacks_pdr.*`、`attacks_delay.*`、`attacks_txrx.*`

提示
- 论文模式（--paper-mode）会：
  - 固定场景顺序，确保与论文表格/正文一致
  - 在未显式提供白名单时应用默认白名单
  - 同步导出 PDF 版本，便于投稿与编辑

