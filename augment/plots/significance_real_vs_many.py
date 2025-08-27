import argparse
import csv
from pathlib import Path
import pandas as pd
from significance_real import mannwhitney_u

PROTOS = ['greedy','leach','heed','pegasis','etx']


def main():
    ap = argparse.ArgumentParser(description='Compute MWU p-values: TrustHR vs many baselines (real data)')
    ap.add_argument('--root', type=str, default='WSN-Intel-Lab-Project/augment/results/real')
    ap.add_argument('--out_csv', type=str, default='WSN-Intel-Lab-Project/augment/results/real/significance_many.csv')
    args = ap.parse_args()

    root = Path(args.root)
    dfs = {}
    for p in PROTOS + ['trusthr']:
        df = pd.read_csv(root / f'{p}_per_mote.csv')
        df['txrx_data'] = df['tx_j_data'] + df['rx_j_data']
        df['total_energy'] = df['tx_j_total'] + df['rx_j_total'] + df['cpu_j_total']
        dfs[p] = df

    metrics = ['pdr_exp','delay_s','txrx_data','total_energy']

    rows = []
    for base in PROTOS:
        for met in metrics:
            x = dfs['trusthr'][met].values
            y = dfs[base][met].values
            U, p = mannwhitney_u(x, y)
            rows.append({'vs': base, 'metric': met, 'U': U, 'p_value': p})

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['vs','metric','U','p_value'])
        for r in rows:
            w.writerow([r['vs'], r['metric'], f"{r['U']:.2f}", f"{r['p_value']:.4g}"])
    print(f'Saved {out}')


if __name__ == '__main__':
    main()

