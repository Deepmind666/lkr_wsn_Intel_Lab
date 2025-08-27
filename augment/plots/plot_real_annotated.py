import argparse
from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd

PROTOS = ['greedy','leach','heed','pegasis','etx','trusthr']
LABELS = {'greedy':'Greedy','leach':'LEACH','heed':'HEED','pegasis':'PEGASIS','etx':'ETX','trusthr':'TrustHR'}
COLORS = {'greedy':'#4C78A8','leach':'#F58518','heed':'#E45756','pegasis':'#54A24B','etx':'#B279A2','trusthr':'#72B7B2'}


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
    # metric names mapping
    return df


def annotate_pvals(ax, metric_key: str, sig_df: pd.DataFrame, xs, ys):
    if sig_df is None:
        return
    # build dict vs->p for given metric
    sdf = sig_df[sig_df['metric'] == metric_key]
    pairs = [('greedy','Greedy'), ('leach','LEACH'), ('heed','HEED')]
    trust_idx = xs.index('TrustHR') if 'TrustHR' in xs else None
    if trust_idx is None:
        return
    ymax = max(ys)
    y = ymax * 1.05 if ymax > 0 else 0.1
    lines = []
    for vs_key, vs_label in pairs:
        row = sdf[sdf['vs']==vs_key]
        if row.empty:
            continue
        p = float(row['p_value'].iloc[0])
        lines.append(f"TrustHR vs {vs_label}: {star_from_p(p)} (p={p:.2g})")
    if lines:
        ax.text(0.02, 0.98, '\n'.join(lines), transform=ax.transAxes, va='top', ha='left', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))


def bar_plot(ax, xs, ys, colors, ylabel):
    ax.bar(range(len(xs)), ys, color=colors)
    ax.set_xticks(range(len(xs)), xs)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(15)
        lbl.set_ha('right')
    ax.set_ylabel(ylabel)
    ax.grid(True, axis='y', alpha=0.3)


def main():
    ap = argparse.ArgumentParser(description='Plot real-data evaluation with significance annotations')
    ap.add_argument('--root', type=str, default='WSN-Intel-Lab-Project/augment/results/real')
    ap.add_argument('--outdir', type=str, default='WSN-Intel-Lab-Project/augment/results/real/figures')
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = load_agg(root)
    xs = [LABELS[r['proto']] for r in rows]
    cs = [COLORS[r['proto']] for r in rows]
    sig = load_sig(root)

    # PDR
    ys = [r['pdr_mean'] for r in rows]
    fig, ax = plt.subplots(figsize=(6,3.2))
    bar_plot(ax, xs, ys, cs, 'Expected PDR')
    annotate_pvals(ax, 'pdr_exp', sig, xs, ys)
    fig.tight_layout(); fig.savefig(outdir/'real_pdr_annotated.png', dpi=200); plt.close(fig)

    # Delay
    ys = [r['delay_mean'] for r in rows]
    fig, ax = plt.subplots(figsize=(6,3.2))
    bar_plot(ax, xs, ys, cs, 'Avg E2E Delay (s)')
    annotate_pvals(ax, 'delay_s', sig, xs, ys)
    fig.tight_layout(); fig.savefig(outdir/'real_delay_annotated.png', dpi=200); plt.close(fig)

    # Data-plane energy (TX+RX)
    ys = [r['tx_j_sum_data'] + r['rx_j_sum_data'] for r in rows]
    fig, ax = plt.subplots(figsize=(6,3.2))
    bar_plot(ax, xs, ys, cs, 'Data-plane Energy (J)')
    annotate_pvals(ax, 'txrx_data', sig, xs, ys)
    fig.tight_layout(); fig.savefig(outdir/'real_energy_data_annotated.png', dpi=200); plt.close(fig)

    print(f'Saved annotated figures to {outdir}')


if __name__ == '__main__':
    main()

