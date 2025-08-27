import argparse
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

PROTOS = ['greedy','leach','heed','etx','trusthr','ilmr']
LABELS = {'greedy':'Greedy','leach':'LEACH','heed':'HEED','etx':'ETX','trusthr':'TrustHR','ilmr':'ILMR'}
COLORS = {'greedy':'#4C78A8','leach':'#F58518','heed':'#E45756','etx':'#B279A2','trusthr':'#72B7B2','ilmr':'#E377C2'}

STAR_THRESH = [(0.001,'***'), (0.01,'**'), (0.05,'*')]

def star_from_p(p):
    for t, s in STAR_THRESH:
        if p < t:
            return s
    return 'n.s.'


def load_agg(root: Path):
    rows = []
    for p in PROTOS:
        with open(root / f'{p}_agg.json','r',encoding='utf-8') as f:
            d = json.load(f)
        d['proto'] = p
        rows.append(d)
    return rows


def load_sig(root: Path):
    fp = root / 'significance.csv'
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    return df

# 读取 per-mote 明细用于误差棒（CI）
# pdr_exp / delay_s 用节点分布自助法95%CI；能耗用节点求和的自助法95%CI
# 为了与聚合均值保持一致，误差棒以 per-mote CI 的左右偏差围绕现有聚合均值绘制

def load_per_mote(root: Path):
    dfs = {}
    for p in PROTOS:
        fp = root / f'{p}_per_mote.csv'
        if fp.exists():
            df = pd.read_csv(fp)
            if 'txrx_data' not in df.columns:
                df['txrx_data'] = df['tx_j_data'] + df['rx_j_data']
            if 'total_energy' not in df.columns:
                df['total_energy'] = df['tx_j_total'] + df['rx_j_total'] + df['cpu_j_total']
            dfs[p] = df
    return dfs


def bootstrap_ci(arr: np.ndarray, n_boot: int = 2000, ci: float = 95.0, agg='mean', random_state: int = 42):
    rng = np.random.default_rng(random_state)
    n = len(arr)
    if n == 0:
        return (np.nan, np.nan)
    stats = []
    for _ in range(n_boot):
        sample = arr[rng.integers(0, n, size=n)]
        if agg == 'mean':
            stats.append(np.mean(sample))
        elif agg == 'sum':
            stats.append(np.sum(sample))
        else:
            stats.append(np.mean(sample))
    low = (100 - ci) / 2
    high = 100 - low
    return (np.percentile(stats, low), np.percentile(stats, high))


def annotate(ax, metric_key: str, sig_df: pd.DataFrame):
    if sig_df is None:
        return
    sdf = sig_df[sig_df['metric'] == metric_key]
    lines = []
    for vs in ['greedy','leach','heed']:
        row = sdf[sdf['vs']==vs]
        if row.empty: continue
        p = float(row['p_value'].iloc[0])
        lines.append(f"TrustHR vs {LABELS[vs]}: {star_from_p(p)} (p={p:.2g})")
    if lines:
        ax.text(0.02, 0.98, '\n'.join(lines), transform=ax.transAxes, va='top', ha='left', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='none'))


def main():
    ap = argparse.ArgumentParser(description='Paper-ready multi-panel plot for real-data evaluation')
    ap.add_argument('--root', type=str, default='WSN-Intel-Lab-Project/augment/results/real')
    ap.add_argument('--out_png', type=str, default='WSN-Intel-Lab-Project/augment/results/real/figures/real_overview_paper.png')
    ap.add_argument('--out_pdf', type=str, default='WSN-Intel-Lab-Project/augment/results/real/figures/real_overview_paper.pdf')
    ap.add_argument('--out_svg', type=str, default='WSN-Intel-Lab-Project/augment/results/real/figures/real_overview_paper.svg')
    args = ap.parse_args()

    # 专业风格
    mpl.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
    })

    root = Path(args.root)
    rows = load_agg(root)
    sig = load_sig(root)
    per_mote = load_per_mote(root)

    xs = [LABELS[r['proto']] for r in rows]
    cs = [COLORS[r['proto']] for r in rows]

    # 聚合均值
    pdr_vals = [r['pdr_mean'] for r in rows]
    delay_vals = [r['delay_mean'] for r in rows]
    data_energy_vals = [r['tx_j_sum_data'] + r['rx_j_sum_data'] for r in rows]
    ctrl_energy_vals = [r['ctrl_tx_j_sum'] + r['ctrl_rx_j_sum'] + r['ctrl_cpu_j_sum'] for r in rows]

    # 误差棒（per-mote bootstrap 95% CI，围绕聚合均值对齐）
    pdr_err_low, pdr_err_high = [], []
    delay_err_low, delay_err_high = [], []
    energy_err_low, energy_err_high = [], []
    for r in rows:
        p = r['proto']
        df = per_mote.get(p, None)
        if df is None or df.empty:
            pdr_err_low.append(0.0); pdr_err_high.append(0.0)
            delay_err_low.append(0.0); delay_err_high.append(0.0)
            energy_err_low.append(0.0); energy_err_high.append(0.0)
            continue
        # PDR
        vals = df['pdr_exp'].values.astype(float)
        lo, hi = bootstrap_ci(vals, agg='mean')
        mote_mean = float(np.mean(vals))
        pdr_err_low.append(max(0.0, mote_mean - lo))
        pdr_err_high.append(max(0.0, hi - mote_mean))
        # Delay
        vals = df['delay_s'].values.astype(float)
        lo, hi = bootstrap_ci(vals, agg='mean')
        mote_mean = float(np.mean(vals))
        delay_err_low.append(max(0.0, mote_mean - lo))
        delay_err_high.append(max(0.0, hi - mote_mean))
        # Data-plane energy sum per node and bootstrap sum
        vals = (df['tx_j_data'].values.astype(float) + df['rx_j_data'].values.astype(float))
        lo, hi = bootstrap_ci(vals, agg='sum')
        mote_sum = float(np.sum(vals))
        energy_err_low.append(max(0.0, mote_sum - lo))
        energy_err_high.append(max(0.0, hi - mote_sum))

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.4))

    # PDR
    ax = axes[0,0]
    ax.bar(range(len(xs)), pdr_vals, color=cs, edgecolor='black', linewidth=0.6,
           yerr=np.vstack([pdr_err_low, pdr_err_high]), capsize=3, ecolor='black', error_kw=dict(linewidth=0.6))
    ax.set_xticks(range(len(xs)), xs, rotation=15, ha='right')
    ax.set_ylabel('Expected PDR')
    ax.set_title('Packet Delivery Ratio (Real Dataset)')
    ax.grid(True, axis='y', alpha=0.3)
    annotate(ax, 'pdr_exp', sig)
    ax.text(-0.18, 1.05, 'A', transform=ax.transAxes, fontsize=12, weight='bold')

    # Delay
    ax = axes[0,1]
    ax.bar(range(len(xs)), delay_vals, color=cs, edgecolor='black', linewidth=0.6,
           yerr=np.vstack([delay_err_low, delay_err_high]), capsize=3, ecolor='black', error_kw=dict(linewidth=0.6))
    ax.set_xticks(range(len(xs)), xs, rotation=15, ha='right')
    ax.set_ylabel('Avg E2E Delay (s)')
    ax.set_title('Latency')
    ax.grid(True, axis='y', alpha=0.3)
    annotate(ax, 'delay_s', sig)
    ax.text(-0.18, 1.05, 'B', transform=ax.transAxes, fontsize=12, weight='bold')

    # Data-plane energy (TX+RX)
    ax = axes[1,0]
    ax.bar(range(len(xs)), data_energy_vals, color=cs, edgecolor='black', linewidth=0.6,
           yerr=np.vstack([energy_err_low, energy_err_high]), capsize=3, ecolor='black', error_kw=dict(linewidth=0.6))
    ax.set_xticks(range(len(xs)), xs, rotation=15, ha='right')
    ax.set_ylabel('Data-plane Energy (J)')
    ax.set_title('Energy (TX+RX, Data-plane)')
    ax.grid(True, axis='y', alpha=0.3)
    annotate(ax, 'txrx_data', sig)
    ax.text(-0.18, 1.05, 'C', transform=ax.transAxes, fontsize=12, weight='bold')

    # Stacked energy (Control vs Data)
    ax = axes[1,1]
    x = list(range(len(xs)))
    ax.bar(x, ctrl_energy_vals, color='#D3D3D3', edgecolor='black', linewidth=0.6, label='Control-plane')
    ax.bar(x, data_energy_vals, bottom=ctrl_energy_vals, color=cs, edgecolor='black', linewidth=0.6, label='Data-plane')
    for i in x:
        tot = ctrl_energy_vals[i] + data_energy_vals[i]
        ax.text(i, tot*1.01, f"{tot:,.0f} J", ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x, xs, rotation=15, ha='right')
    ax.set_ylabel('Total Energy (J)')
    ax.set_title('Total Energy Breakdown')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(loc='upper right', framealpha=0.85)
    ax.text(-0.18, 1.05, 'D', transform=ax.transAxes, fontsize=12, weight='bold')

    # footnote
    fig.text(0.01, 0.01,
             'Note: Control-plane overheads are computed deterministically and are identical across protocols;\n'
             'data-plane improvements refer to TX+RX energy on real topology using connectivity-derived link success.',
             fontsize=8)

    fig.tight_layout(rect=(0,0.06,1,1))
    fig.savefig(args.out_png, dpi=600, bbox_inches='tight')
    fig.savefig(args.out_pdf, bbox_inches='tight')
    fig.savefig(args.out_svg, bbox_inches='tight')
    print(f'Saved {args.out_png}, {args.out_pdf} and {args.out_svg}')


if __name__ == '__main__':
    main()

