import argparse
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib as mpl

PROTOS = ['greedy','leach','heed','etx','trusthr']
LABELS = {'greedy':'Greedy','leach':'LEACH','heed':'HEED','etx':'ETX','trusthr':'TrustHR'}
COLORS = {'greedy':'#4C78A8','leach':'#F58518','heed':'#E45756','etx':'#B279A2','trusthr':'#72B7B2'}


def load_agg(root: Path):
    rows = []
    for p in PROTOS:
        with open(root / f'{p}_elec_agg.json','r',encoding='utf-8') as f:
            d = json.load(f)
        d['proto'] = p
        rows.append(d)
    return rows


def main():
    ap = argparse.ArgumentParser(description='Paper-ready figure (electronics-only energy)')
    ap.add_argument('--root', type=str, default='WSN-Intel-Lab-Project/augment/results/real')
    ap.add_argument('--out_png', type=str, default='WSN-Intel-Lab-Project/augment/results/real/figures/real_overview_paper_elec.png')
    ap.add_argument('--out_pdf', type=str, default='WSN-Intel-Lab-Project/augment/results/real/figures/real_overview_paper_elec.pdf')
    args = ap.parse_args()

    mpl.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10})

    rows = load_agg(Path(args.root))
    xs = [LABELS[r['proto']] for r in rows]
    cs = [COLORS[r['proto']] for r in rows]

    pdr = [r['pdr_mean'] for r in rows]
    delay = [r['delay_mean'] for r in rows]
    energy = [r['tx_j_sum_data'] + r['rx_j_sum_data'] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4))

    # PDR
    ax = axes[0]
    ax.bar(range(len(xs)), pdr, color=cs)
    ax.set_xticks(range(len(xs)), xs, rotation=15, ha='right')
    ax.set_ylabel('Expected PDR')
    ax.set_title('PDR (Real Dataset)')
    ax.grid(True, axis='y', alpha=0.3)

    # Delay
    ax = axes[1]
    ax.bar(range(len(xs)), delay, color=cs)
    ax.set_xticks(range(len(xs)), xs, rotation=15, ha='right')
    ax.set_ylabel('Avg E2E Delay (s)')
    ax.set_title('Latency')
    ax.grid(True, axis='y', alpha=0.3)

    # Electronics-only energy (TX+RX)
    ax = axes[2]
    ax.bar(range(len(xs)), energy, color=cs)
    ax.set_xticks(range(len(xs)), xs, rotation=15, ha='right')
    ax.set_ylabel('Data-plane Electronics Energy (J)')
    ax.set_title('Energy (TX+RX, Electronics)')
    ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out_png, dpi=300)
    fig.savefig(args.out_pdf)
    print(f'Saved {args.out_png} and {args.out_pdf}')


if __name__ == '__main__':
    main()

