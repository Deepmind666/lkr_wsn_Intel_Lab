#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Repeated SoD ablation with multiple random seeds.
Outputs aggregated CSV with mean/std and generates CI plots.
"""

from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.enhanced_eehfr_system import EnhancedEEHFRSystem, SystemConfig


def run_once(seed: int, sod_enabled: bool, rounds: int = 60, num_nodes: int = 30):
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
        random_seed=seed,
    )
    sys = EnhancedEEHFRSystem(cfg)
    hist = sys.run_simulation()

    total_energy = float(np.sum([h["energy_consumed"] for h in hist]))
    avg_sod_ratio = float(np.mean([h["performance"].sod_trigger_ratio for h in hist]))
    final_alive = int(hist[-1]["alive_nodes"]) if hist else 0
    return total_energy, avg_sod_ratio, final_alive


def main():
    seeds = [0, 1, 2, 3, 4]
    rows = []
    for sod_enabled in [False, True]:
        for s in seeds:
            te, ar, fa = run_once(s, sod_enabled)
            rows.append({
                'sod_enabled': sod_enabled,
                'seed': s,
                'total_energy': te,
                'avg_sod_ratio': ar,
                'final_alive': fa,
            })

    df = pd.DataFrame(rows)
    out_dir = ROOT / 'experiments' / 'results' / 'data'
    fig_dir = ROOT / 'experiments' / 'results' / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime('%Y%m%d_%H%M%S')
    df_path = out_dir / f'sod_repeated_{ts}.csv'
    df.to_csv(df_path, index=False)

    # Aggregate
    agg = df.groupby('sod_enabled').agg(
        total_energy_mean=('total_energy', 'mean'),
        total_energy_std=('total_energy', 'std'),
        avg_sod_ratio_mean=('avg_sod_ratio', 'mean'),
        avg_sod_ratio_std=('avg_sod_ratio', 'std'),
    ).reset_index()

    # Plot with error bars
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = [0, 1]
    te_mean = agg['total_energy_mean'].values
    te_std = agg['total_energy_std'].values
    ar_mean = agg['avg_sod_ratio_mean'].values
    ar_std = agg['avg_sod_ratio_std'].values

    axes[0].bar(x, te_mean, yerr=te_std, capsize=4, color=['#888', '#4CAF50'])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['SoD Off', 'SoD On'])
    axes[0].set_title('Total Energy (mean ± std)')

    axes[1].bar(x, ar_mean, yerr=ar_std, capsize=4, color=['#888', '#4CAF50'])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['SoD Off', 'SoD On'])
    axes[1].set_ylim(0, 1)
    axes[1].set_title('Avg SoD Ratio (mean ± std)')

    plt.tight_layout()
    fig_path = fig_dir / f'sod_repeated_{ts}.png'
    fig_path_pdf = fig_dir / f'sod_repeated_{ts}.pdf'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path_pdf, dpi=300, bbox_inches='tight')
    plt.close()

    print('Aggregated results:')
    print(agg)
    print('Saved:', df_path, fig_path)


if __name__ == '__main__':
    main()


