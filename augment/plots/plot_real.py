import argparse
from pathlib import Path
import json
import matplotlib.pyplot as plt

PROTOS = ['greedy','leach','heed','pegasis','etx','trusthr','ilmr']
LABELS = {'greedy':'Greedy','leach':'LEACH','heed':'HEED','pegasis':'PEGASIS','etx':'ETX','trusthr':'TrustHR','ilmr':'ILMR'}
COLORS = {'greedy':'#4C78A8','leach':'#F58518','heed':'#E45756','pegasis':'#54A24B','etx':'#B279A2','trusthr':'#72B7B2','ilmr':'#E377C2'}


def load_agg(root: Path):
    rows = []
    for p in PROTOS:
        with open(root / f'{p}_agg.json','r',encoding='utf-8') as f:
            d = json.load(f)
        d['proto'] = p
        rows.append(d)
    return rows


def bar_plot(ax, xs, ys, colors, ylabel):
    ax.bar(range(len(xs)), ys, color=colors)
    ax.set_xticks(range(len(xs)), xs)
    # rotate to avoid overlap
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(15)
        lbl.set_ha('right')
    ax.set_ylabel(ylabel)
    ax.grid(True, axis='y', alpha=0.3)


def main():
    ap = argparse.ArgumentParser(description='Plot real-data evaluation summary across protocols')
    ap.add_argument('--root', type=str, default='WSN-Intel-Lab-Project/augment/results/real')
    ap.add_argument('--outdir', type=str, default='WSN-Intel-Lab-Project/augment/results/real/figures')
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = load_agg(root)
    xs = [LABELS[r['proto']] for r in rows]
    cs = [COLORS[r['proto']] for r in rows]

    # PDR
    ys = [r['pdr_mean'] for r in rows]
    fig, ax = plt.subplots(figsize=(5,3))
    bar_plot(ax, xs, ys, cs, 'Expected PDR')
    fig.tight_layout(); fig.savefig(outdir/'real_pdr.png', dpi=200); plt.close(fig)

    # Delay
    ys = [r['delay_mean'] for r in rows]
    fig, ax = plt.subplots(figsize=(5,3))
    bar_plot(ax, xs, ys, cs, 'Avg E2E Delay (s)')
    fig.tight_layout(); fig.savefig(outdir/'real_delay.png', dpi=200); plt.close(fig)

    # Data-plane energy (TX+RX)
    ys = [r['tx_j_sum_data'] + r['rx_j_sum_data'] for r in rows]
    fig, ax = plt.subplots(figsize=(5,3))
    bar_plot(ax, xs, ys, cs, 'Data-plane Energy (J)')
    fig.tight_layout(); fig.savefig(outdir/'real_energy_data.png', dpi=200); plt.close(fig)

    # Total energy (TX+RX+CPU including control)
    ys = [r['tx_j_sum_total'] + r['rx_j_sum_total'] + r['cpu_j_sum_total'] for r in rows]
    fig, ax = plt.subplots(figsize=(5,3))
    bar_plot(ax, xs, ys, cs, 'Total Energy (J)')
    fig.tight_layout(); fig.savefig(outdir/'real_energy_total.png', dpi=200); plt.close(fig)

    print(f'Saved figures to {outdir}')


if __name__ == '__main__':
    main()

