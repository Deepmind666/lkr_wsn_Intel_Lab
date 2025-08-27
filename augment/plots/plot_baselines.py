import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import json


PROTS = ['greedy','leach','heed','trusthr']
COLORS = {'greedy':'#4C78A8','leach':'#F58518','heed':'#E45756','trusthr':'#72B7B2'}
LABELS = {'greedy':'Greedy','leach':'LEACH','heed':'HEED','trusthr':'TrustHR'}
STAR_THRESH = [(0.001,'***'), (0.01,'**'), (0.05,'*')]


def star_from_p(p):
    for t, s in STAR_THRESH:
        if p < t:
            return s
    return 'n.s.'


def get_ci_for_proto(ci_df: pd.DataFrame, n: int, z: float, metric: str, p: str):
    """
    From ci.csv rows, obtain mean and (lo, hi) CI for given protocol under (n,z,metric).
    For baseline proto p in ['greedy','leach','heed'], use base_* fields of the row with vs==p.
    For trusthr, use any matching row (take first) and use trust_* fields.
    Returns (mean, lo, hi) or (None, None, None) if not found.
    """
    sub = ci_df[(ci_df['nodes']==n) & (ci_df['noise']==z) & (ci_df['metric']==metric)]
    if sub.empty:
        return None, None, None
    if p == 'trusthr':
        r = sub.iloc[0]
        return float(r['trust_mean']), float(r['trust_ci_lo']), float(r['trust_ci_hi'])
    else:
        row = sub[sub['vs']==p]
        if row.empty:
            return None, None, None
        r = row.iloc[0]
        return float(r['base_mean']), float(r['base_ci_lo']), float(r['base_ci_hi'])


def annotate_sig(ax, sig_df: pd.DataFrame, n: int, z: float, metric: str):
    if sig_df is None:
        return
    col_map = {'pdr':'p_pdr', 'delay':'p_delay', 'txrx':'p_txrx'}
    col = col_map.get(metric)
    if col is None or col not in sig_df.columns:
        return
    rows = sig_df[(sig_df['nodes']==n) & (sig_df['noise']==z)]
    if rows.empty:
        return
    lines = []
    for vs in ['greedy','leach','heed']:
        r = rows[rows['vs']==vs]
        if r.empty:
            continue
        pval = float(r[col].iloc[0])
        lines.append(f"TrustHR vs {LABELS[vs]}: {star_from_p(pval)} (p={pval:.2g})")
    if lines:
        ax.text(0.02, 0.98, '\n'.join(lines), transform=ax.transAxes, va='top', ha='left', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='none'))


def plot_group(df, metric, ylabel, out, ci_df: pd.DataFrame=None, sig_df: pd.DataFrame=None):
    # df columns (fallback): nodes, noise, proto, <metric>_mean, <metric>_std
    fig, axes = plt.subplots(1, 3, figsize=(12,3.5), sharey=True)
    for ax, (n,z) in zip(axes, [(50,0.2),(100,0.2),(200,0.2)]):
        xs, ys, cs = [], [], []
        if ci_df is not None:
            # Use bootstrap CI from ci.csv
            err_los, err_his = [], []
            for p in PROTS:
                mean, lo, hi = get_ci_for_proto(ci_df, n, z, metric, p)
                if mean is None:
                    continue
                xs.append(LABELS[p])
                ys.append(mean)
                # asymmetric error from CI around mean
                err_lo = max(0.0, mean - lo)
                err_hi = max(0.0, hi - mean)
                err_los.append(err_lo)
                err_his.append(err_hi)
                cs.append(COLORS[p])
            ax.bar(range(len(xs)), ys, yerr=[err_los, err_his], color=cs, capsize=4)
        else:
            # Fallback to std from summary.csv
            es = []
            sub = df[(df['nodes']==n) & (df['noise']==z)]
            for p in PROTS:
                row = sub[sub['proto']==p]
                if row.empty:
                    continue
                xs.append(LABELS[p])
                ys.append(float(row[f'{metric}_mean'].iloc[0]))
                es.append(float(row[f'{metric}_std'].iloc[0]))
                cs.append(COLORS[p])
            ax.bar(range(len(xs)), ys, yerr=es, color=cs, capsize=4)
        ax.set_title(f'n={n}, noise={z}')
        ax.set_xticks(range(len(xs)), xs)
        ax.grid(True, axis='y', alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel(ylabel)
        # significance annotation per subplot/metric
        annotate_sig(ax, sig_df, n, z, metric)
    fig.tight_layout()
    # Save PNG and vector formats for publication quality
    plt.savefig(out, dpi=300)
    try:
        plt.savefig(out.with_suffix('.pdf'))
        plt.savefig(out.with_suffix('.svg'))
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description='Plot baselines summary as bar charts with error bars (std or bootstrap CI)')
    ap.add_argument('--summary', type=str, default='WSN-Intel-Lab-Project/augment/results/baselines/summary.csv')
    ap.add_argument('--ci_csv', type=str, default='WSN-Intel-Lab-Project/augment/results/baselines/ci.csv')
    ap.add_argument('--sig_csv', type=str, default='WSN-Intel-Lab-Project/augment/results/baselines/significance.csv')
    ap.add_argument('--use_ci', action='store_true', help='Use bootstrap CI from ci.csv for error bars')
    ap.add_argument('--outdir', type=str, default='WSN-Intel-Lab-Project/augment/results/figures')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.summary)

    ci_df = None
    if args.use_ci and Path(args.ci_csv).exists():
        ci_df = pd.read_csv(args.ci_csv)
    sig_df = None
    if Path(args.sig_csv).exists():
        sig_df = pd.read_csv(args.sig_csv)

    # PDR
    plot_group(df, 'pdr', 'PDR', outdir/'baselines_pdr.png', ci_df=ci_df, sig_df=sig_df)
    # Delay
    plot_group(df, 'delay', 'Avg E2E Delay (s)', outdir/'baselines_delay.png', ci_df=ci_df, sig_df=sig_df)
    # Energy: TX+RX combined -> metric key 'txrx' for CI; when using summary fallback, compute std accordingly
    if ci_df is not None:
        plot_group(df, 'txrx', 'Mean Energy (J)', outdir/'baselines_energy.png', ci_df=ci_df, sig_df=sig_df)
    else:
        df = df.copy()
        df['txrx_mean'] = df['tx_j_mean'] + df['rx_j_mean']
        df['txrx_std'] = (df['tx_j_std']**2 + df['rx_j_std']**2)**0.5
        df_txrx = df.rename(columns={'txrx_mean':'txrx_mean','txrx_std':'txrx_std'})
        plot_group(df_txrx, 'txrx', 'Mean Energy (J)', outdir/'baselines_energy.png', ci_df=None, sig_df=sig_df)

    print(f'Saved figures to {outdir}')


if __name__ == '__main__':
    main()

