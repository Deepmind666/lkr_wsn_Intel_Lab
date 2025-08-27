#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compare ACO vs Baseline shortest-path routing with SoD on/off.
Outputs CSV and figures with mean±std over multiple seeds.
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.enhanced_eehfr_system import EnhancedEEHFRSystem, SystemConfig
from src.baseline_router import BaselineRouter


def run_once(seed: int, use_baseline: bool, sod_enabled: bool, rounds: int = 60, num_nodes: int = 30):
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
    sysm = EnhancedEEHFRSystem(cfg)
    # initial cluster selection to get tops and topology for baseline
    sysm.initialize_network()
    sysm.run_fuzzy_cluster_selection(1)
    if use_baseline:
        br = BaselineRouter()
        routes = br.find_routes(sysm.cluster_heads, sysm.nodes, sysm.network_topology, sysm.config.base_station_pos)
        # map routes to routing_paths format
        sysm.routing_paths = {}
        for ch, r in zip(sysm.cluster_heads, routes):
            sysm.routing_paths[ch] = r.path[1:]  # drop start node
    # continue simulation from round 1 with routing updates each loop
    hist = sysm.run_simulation()
    total_energy = float(np.sum([h['energy_consumed'] for h in hist]))
    final_alive = int(hist[-1]['alive_nodes']) if hist else 0
    avg_ratio = float(np.mean([h['performance'].sod_trigger_ratio for h in hist]))
    return total_energy, final_alive, avg_ratio


def main():
    seeds = [0, 1, 2]
    rows = []
    for use_baseline in [False, True]:
        for sod_enabled in [False, True]:
            for s in seeds:
                te, fa, ar = run_once(s, use_baseline, sod_enabled)
                rows.append({
                    'router': 'baseline' if use_baseline else 'aco',
                    'sod': 'on' if sod_enabled else 'off',
                    'seed': s,
                    'total_energy': te,
                    'final_alive': fa,
                    'avg_sod_ratio': ar,
                })

    df = pd.DataFrame(rows)
    out_dir = ROOT / 'experiments' / 'results' / 'data'
    fig_dir = ROOT / 'experiments' / 'results' / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    csv_path = out_dir / f'routing_comparison_{ts}.csv'
    df.to_csv(csv_path, index=False)

    # mean±std bar plots
    agg = df.groupby(['router', 'sod']).agg(
        te_mean=('total_energy', 'mean'), te_std=('total_energy', 'std'),
        alive_mean=('final_alive', 'mean'), alive_std=('final_alive', 'std')
    ).reset_index()
    routers = ['aco', 'baseline']
    sods = ['off', 'on']
    x = np.arange(len(routers))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, metric in enumerate([('te_mean','te_std','Total Energy'), ('alive_mean','alive_std','Final Alive')]):
        mean_col, std_col, title = metric
        vals_off = [agg[(agg.router==r) & (agg.sod=='off')][mean_col].values[0] for r in routers]
        err_off  = [agg[(agg.router==r) & (agg.sod=='off')][std_col].values[0] for r in routers]
        vals_on  = [agg[(agg.router==r) & (agg.sod=='on')][mean_col].values[0]  for r in routers]
        err_on   = [agg[(agg.router==r) & (agg.sod=='on')][std_col].values[0]  for r in routers]
        axes[i].bar(x-width/2, vals_off, width, yerr=err_off, capsize=4, label='SoD Off')
        axes[i].bar(x+width/2, vals_on,  width, yerr=err_on,  capsize=4, label='SoD On')
        axes[i].set_xticks(x); axes[i].set_xticklabels([r.upper() for r in routers])
        axes[i].set_title(title)
        axes[i].legend()
    plt.tight_layout()
    fig_path = fig_dir / f'routing_comparison_{ts}.png'
    fig_path_pdf = fig_dir / f'routing_comparison_{ts}.pdf'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path_pdf, dpi=300, bbox_inches='tight')
    plt.close()
    print('Saved:', csv_path, fig_path, fig_path_pdf)


if __name__ == '__main__':
    main()


