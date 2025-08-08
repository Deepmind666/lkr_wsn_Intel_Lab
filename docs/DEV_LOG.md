WSN-Intel-Lab-Project — 开发变更记录

日期：2025-08-06

本次关键改动（面向 SoD/论文复现）：

1) 新增 SoD 工具模块
- 新文件：`src/utils/sod.py`
- 功能：
  - SoD 门控（Send-on-Delta）：仅在新读数与上次发送值差异超过阈值时触发“发送”。
  - 模式：fixed（`delta_day/delta_night`）与 adaptive（`delta = k * σ(window)`，并设白天/夜间最小阈值下限）。
  - 轻量状态：按节点维护滚动窗口与上次发送值。

2) 接入 SoD 到系统主循环
- 编辑：`src/enhanced_eehfr_system.py`
- 变更点：
  - 引入 `SoDController/SoDConfig`，在 `EnhancedEEHFRSystem` 初始化中维护 `self.sod_controllers`。
  - 在 `generate_sensor_data()` 中，用温度作为触发信号，对每个节点执行 SoD 判定；仅当 `should_send=True` 时将读数加入 `node['data_buffer']` 与 `self.sensor_data`。
  - 采用 `pseudo_hour = (round_num // 10) % 24` 粗略映射昼/夜（可根据真实时间戳替换）。

3) 目的与预期影响
- 目的：将“数据面”节能机制落地，减少上行报文与能耗，呼应论文 SoD/双预测思路。
- 预期：在不改 MAC/PHY 的前提下，数据包量下降、TX 占空比下降；后续可统计能耗差异与寿命指标（FND/HND/LND）。

4) 后续计划（下一步）
- 在能耗模型中计入“未触发发送”的缓存阶段成本（CPU/LPM），完善 mJ 估计。
- 扩展 SoD 触发到多通道（温/湿/光结合或选择最敏感通道）。
- 在 `experiments/` 中新增配置项：`sod.mode/k/window/delta_day/delta_night`，并输出 SoD 触发率与节能比。


