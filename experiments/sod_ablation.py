#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SoD 消融实验（最小可复现）

对比 SoD 开启/关闭两种设置在 EEHFR 系统中的效果：
- 能耗（每轮能耗累计）
- 存活节点数（最后一轮）
- SoD 触发率（平均）

输出：JSON 与 PNG 图表到 results 目录。
"""

import os
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt

# 确保项目根目录在 sys.path，便于以 "src.*" 方式导入
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.enhanced_eehfr_system import EnhancedEEHFRSystem, SystemConfig


def run_once(sod_enabled: bool, rounds: int = 60, num_nodes: int = 30) -> Dict[str, Any]:
    cfg = SystemConfig(
        num_nodes=num_nodes,
        simulation_rounds=rounds,
        sod_enabled=sod_enabled,
        sod_mode="adaptive",
        sod_k=1.5,
        sod_window=24,
        sod_delta_day=0.5,
        sod_delta_night=0.2,
        payload_bits=1024,
        idle_cpu_time_s=0.001,
        idle_lpm_time_s=0.004,
    )
    system = EnhancedEEHFRSystem(cfg)
    history = system.run_simulation()

    # 汇总能耗与 SoD 触发率
    total_energy = float(np.sum([h["energy_consumed"] for h in history]))
    avg_sod_ratio = float(np.mean([h["performance"].sod_trigger_ratio for h in history]))
    final_alive = int(history[-1]["alive_nodes"]) if history else 0

    # 也保留最后一轮完整指标
    final_metrics = asdict(history[-1]["performance"]) if history else {}

    # 导出每轮序列（便于论文图表）
    per_round = []
    for h in history:
        perf = asdict(h["performance"]) if hasattr(h["performance"], "__dict__") else h["performance"].__dict__
        per_round.append({
            "round": h["round"],
            "alive_nodes": h["alive_nodes"],
            "energy_consumed": h["energy_consumed"],
            "sod_trigger_ratio": perf.get("sod_trigger_ratio", 1.0),
            "energy_efficiency": perf.get("energy_efficiency", 0.0),
            "avg_trust": perf.get("average_trust_value", 0.0),
        })

    return {
        "config": {
            "sod_enabled": sod_enabled,
            "rounds": rounds,
            "num_nodes": num_nodes,
        },
        "summary": {
            "total_energy_consumed": total_energy,
            "avg_sod_trigger_ratio": avg_sod_ratio,
            "final_alive_nodes": final_alive,
        },
        "final_metrics": final_metrics,
        "per_round": per_round,
    }


def main():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_data_dir = Path("experiments/results/data")
    out_fig_dir = Path("experiments/results/figures")
    out_data_dir.mkdir(parents=True, exist_ok=True)
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    # 两组对照
    res_off = run_once(sod_enabled=False)
    res_on = run_once(sod_enabled=True)

    all_results = {"sod_off": res_off, "sod_on": res_on}

    # 保存 JSON
    json_path = out_data_dir / f"sod_ablation_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # 绘制简要对比图（能耗与触发率）
    labels = ["SoD Off", "SoD On"]
    energies = [res_off["summary"]["total_energy_consumed"], res_on["summary"]["total_energy_consumed"]]
    ratios = [res_off["summary"]["avg_sod_trigger_ratio"], res_on["summary"]["avg_sod_trigger_ratio"]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # 能耗
    axes[0].bar(labels, energies, color=["#999999", "#4CAF50"])
    axes[0].set_title("Total Energy Consumed (lower is better)")
    axes[0].set_ylabel("Energy (arbitrary)")
    # 触发率
    axes[1].bar(labels, ratios, color=["#999999", "#4CAF50"])
    axes[1].set_title("Average SoD Trigger Ratio")
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    fig_path = out_fig_dir / f"sod_ablation_{timestamp}.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"SoD 消融结果已保存:\n- JSON: {json_path}\n- FIG : {fig_path}")


if __name__ == "__main__":
    main()


