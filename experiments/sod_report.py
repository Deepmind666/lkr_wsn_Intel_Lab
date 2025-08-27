#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生成 SoD 实验高质量图表与 CSV
- 读取最新的 sod_ablation_*.json
- 导出每轮曲线（能耗、存活、SoD 触发率）与总览对比条形图
"""

from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / 'experiments' / 'results' / 'data'
    fig_dir = root / 'experiments' / 'results' / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(data_dir.glob('sod_ablation_*.json'))
    if not paths:
        print('No result json found')
        return
    p = paths[-1]
    d = json.loads(p.read_text(encoding='utf-8'))

    # 汇总表
    rows = []
    for key in ['sod_off', 'sod_on']:
        s = d[key]['summary']
        rows.append({
            'setting': key,
            'total_energy': s['total_energy_consumed'],
            'avg_sod_ratio': s['avg_sod_trigger_ratio'],
            'final_alive': s['final_alive_nodes'],
        })
    df_sum = pd.DataFrame(rows)
    df_sum.to_csv(data_dir / 'sod_ablation_summary_latest.csv', index=False)

    # 每轮曲线
    for key in ['sod_off', 'sod_on']:
        pr = pd.DataFrame(d[key].get('per_round', []))
        if not pr.empty:
            pr.to_csv(data_dir / f'{key}_per_round_latest.csv', index=False)
            # 画曲线
            fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
            axes[0].plot(pr['round'], pr['cum_energy'], label='Cumulative Energy')
            axes[0].set_ylabel('Cum Energy')
            axes[0].grid(True, alpha=0.3)
            axes[1].plot(pr['round'], pr['energy_consumed'], label='Energy per round')
            axes[1].set_ylabel('Energy/round')
            axes[1].grid(True, alpha=0.3)
            axes[2].plot(pr['round'], pr['alive_nodes'], label='Alive nodes', color='tab:green')
            axes[2].set_ylabel('Alive')
            axes[2].grid(True, alpha=0.3)
            axes[3].plot(pr['round'], pr['sod_trigger_ratio'], label='SoD ratio', color='tab:orange')
            axes[3].set_ylabel('SoD ratio')
            axes[3].set_xlabel('Round')
            axes[3].set_ylim(0, 1)
            axes[3].grid(True, alpha=0.3)

            # 叠加FND/HND/LND竖线（如可推断）
            for marker_name in ['FND', 'HND', 'LND']:
                col = marker_name.lower()
                if col in pr.columns and pr[col].notna().any():
                    r = int(pr[col].dropna().iloc[0])
                    for ax in axes:
                        ax.axvline(r, color='red', linestyle='--', alpha=0.3)
                        ax.text(r, ax.get_ylim()[1]*0.9, marker_name, color='red', fontsize=8)
            plt.tight_layout()
            png = fig_dir / f'{key}_per_round_latest.png'
            pdf = fig_dir / f'{key}_per_round_latest.pdf'
            plt.savefig(png, dpi=300, bbox_inches='tight')
            plt.savefig(pdf, dpi=300, bbox_inches='tight')
            plt.close()

    # 条形图对比
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(df_sum['setting'], df_sum['total_energy'], color=['#888', '#4CAF50'])
    axes[0].set_title('Total Energy (lower is better)')
    axes[1].bar(df_sum['setting'], df_sum['avg_sod_ratio'], color=['#888', '#4CAF50'])
    axes[1].set_title('Average SoD Trigger Ratio')
    axes[1].set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(fig_dir / 'sod_ablation_summary_latest.png', dpi=300, bbox_inches='tight')
    plt.savefig(fig_dir / 'sod_ablation_summary_latest.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    print('Report generated:')
    print('-', data_dir / 'sod_ablation_summary_latest.csv')
    print('-', fig_dir / 'sod_ablation_summary_latest.png')


if __name__ == '__main__':
    main()


