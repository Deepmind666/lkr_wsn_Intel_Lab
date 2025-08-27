import argparse
from pathlib import Path
import json
import matplotlib.pyplot as plt

PROTOS = ['greedy','leach','heed','trusthr']
LABELS = {'greedy':'Greedy','leach':'LEACH','heed':'HEED','trusthr':'TrustHR'}
COLORS = {'greedy':'#4C78A8','leach':'#F58518','heed':'#E45756','trusthr':'#72B7B2'}


def load_agg(root: Path):
    rows = []
    for p in PROTOS:
        with open(root / f'{p}_agg.json','r',encoding='utf-8') as f:
            d = json.load(f)
        d['proto'] = p
        rows.append(d)
    return rows


def main():
    ap = argparse.ArgumentParser(description='Plot stacked energy: control vs data plane')
    ap.add_argument('--root', type=str, default='WSN-Intel-Lab-Project/augment/results/real')
    ap.add_argument('--out', type=str, default='WSN-Intel-Lab-Project/augment/results/real/figures/real_energy_stacked.png')
    args = ap.parse_args()

    root = Path(args.root)
    rows = load_agg(root)

    xs = [LABELS[r['proto']] for r in rows]
    cs = [COLORS[r['proto']] for r in rows]

    data_plane = [r['tx_j_sum_data'] + r['rx_j_sum_data'] + r.get('cpu_j_sum_data',0.0) for r in rows]
    ctrl_plane = [r['ctrl_tx_j_sum'] + r['ctrl_rx_j_sum'] + r['ctrl_cpu_j_sum'] for r in rows]
    totals = [dp+cp for dp,cp in zip(data_plane, ctrl_plane)]

    fig, ax = plt.subplots(figsize=(7.5,4.2))
    x = list(range(len(xs)))
    ax.bar(x, ctrl_plane, color='#D3D3D3', label='Control-plane')
    ax.bar(x, data_plane, bottom=ctrl_plane, color=cs, label='Data-plane')

    # numeric labels on top
    for i, tot in enumerate(totals):
        ax.text(i, ctrl_plane[i]+data_plane[i]+max(tot*0.01, 50), f"{tot:,.0f} J", ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x, xs)
    ax.set_ylabel('Energy (J)')
    ax.set_title('Total Energy Breakdown (Real Dataset)')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    fig.savefig(args.out, dpi=200)
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()

