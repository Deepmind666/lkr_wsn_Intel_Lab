#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Grid search over SoD parameters (k, window, delta_day, delta_night).
Outputs CSV and a heatmap for total energy.
"""

from pathlib import Path
import itertools
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.enhanced_eehfr_system import EnhancedEEHFRSystem, SystemConfig


def run_cfg(k, W, dd, dn, seed=0, rounds=60, num_nodes=30):
    cfg = SystemConfig(
        num_nodes=num_nodes,
        simulation_rounds=rounds,
        sod_enabled=True,
        sod_mode="adaptive",
        sod_k=k,
        sod_window=W,
        sod_delta_day=dd,
        sod_delta_night=dn,
        payload_bits=1024,
        idle_cpu_time_s=0.001,
        idle_lpm_time_s=0.004,
        random_seed=seed,
    )
    sys = EnhancedEEHFRSystem(cfg)
    hist = sys.run_simulation()
    total_energy = float(np.sum([h['energy_consumed'] for h in hist]))
    avg_ratio = float(np.mean([h['performance'].sod_trigger_ratio for h in hist]))
    return total_energy, avg_ratio


def main():
    ks = [0.5, 1.0, 1.5, 2.0]
    Ws = [12, 24, 48]
    dds = [0.2, 0.5]
    dns = [0.1, 0.2]
    seeds = [0, 1]

    rows = []
    for k, W, dd, dn, seed in itertools.product(ks, Ws, dds, dns, seeds):
        te, ar = run_cfg(k, W, dd, dn, seed)
        rows.append({'k': k, 'W': W, 'delta_day': dd, 'delta_night': dn, 'seed': seed,
                     'total_energy': te, 'avg_sod_ratio': ar})

    df = pd.DataFrame(rows)
    out_dir = ROOT / 'experiments' / 'results' / 'data'
    fig_dir = ROOT / 'experiments' / 'results' / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime('%Y%m%d_%H%M%S')
    csv_path = out_dir / f'sod_grid_{ts}.csv'
    df.to_csv(csv_path, index=False)

    # Heatmap: mean total energy vs (k, W)
    pivot = df.groupby(['k', 'W']).agg(mean_energy=('total_energy', 'mean')).reset_index()
    k_vals = sorted(pivot['k'].unique())
    W_vals = sorted(pivot['W'].unique())
    mat = np.zeros((len(W_vals), len(k_vals)))
    for i, W in enumerate(W_vals):
        for j, k in enumerate(k_vals):
            mat[i, j] = pivot[(pivot['k'] == k) & (pivot['W'] == W)]['mean_energy'].values[0]

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(mat, aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(k_vals))); ax.set_xticklabels(k_vals)
    ax.set_yticks(range(len(W_vals))); ax.set_yticklabels(W_vals)
    ax.set_xlabel('k'); ax.set_ylabel('window W')
    ax.set_title('Mean Total Energy')
    fig.colorbar(im, ax=ax)
    fig_path = fig_dir / f'sod_grid_heatmap_{ts}.png'
    plt.tight_layout(); plt.savefig(fig_path, dpi=300, bbox_inches='tight'); plt.close()

    print('Saved:', csv_path, fig_path)


if __name__ == '__main__':
    main()


