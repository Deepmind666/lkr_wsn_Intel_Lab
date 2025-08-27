import argparse
import csv
from pathlib import Path
import pandas as pd
import math

PROTOS = ['greedy','leach','heed','pegasis','etx','trusthr']


def mannwhitney_u(x, y):
    # Compute U statistic with tie correction and normal approx p-value (two-sided)
    import numpy as np
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n1 = len(x); n2 = len(y)
    all_vals = np.concatenate([x, y])
    # ranks with average for ties
    order = all_vals.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, n1+n2+1)
    # handle ties: average ranks in ties
    i = 0
    while i < n1+n2:
        j = i
        while j+1 < n1+n2 and all_vals[order[j+1]] == all_vals[order[i]]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0
            for k in range(i, j+1):
                ranks[order[k]] = avg
        i = j + 1
    R1 = ranks[:n1].sum()
    U1 = R1 - n1*(n1+1)/2.0
    U2 = n1*n2 - U1
    U = min(U1, U2)
    # tie correction
    # count ties in combined data
    _, counts = np.unique(all_vals, return_counts=True)
    tie_term = (counts**3 - counts).sum() / 12.0
    mu = n1*n2/2.0
    sigma_sq = n1*n2*(n1+n2+1)/12.0 - tie_term * (n1*n2)/((n1+n2)*(n1+n2-1)) if (n1+n2)>1 else 0.0
    sigma = math.sqrt(max(1e-12, sigma_sq))
    z = (U - mu) / sigma
    # two-sided p from normal
    p = math.erfc(abs(z)/math.sqrt(2.0))
    return U, p


def main():
    ap = argparse.ArgumentParser(description='Compute Mann-Whitney significance on real-data per-mote metrics (TrustHR vs others)')
    ap.add_argument('--root', type=str, default='WSN-Intel-Lab-Project/augment/results/real')
    ap.add_argument('--out_csv', type=str, default='WSN-Intel-Lab-Project/augment/results/real/significance.csv')
    args = ap.parse_args()

    root = Path(args.root)
    # load per-mote metrics
    dfs = {}
    for p in PROTOS:
        fp = root / f'{p}_per_mote.csv'
        df = pd.read_csv(fp)
        df['txrx_data'] = df['tx_j_data'] + df['rx_j_data']
        df['total_energy'] = df['tx_j_total'] + df['rx_j_total'] + df['cpu_j_total']
        dfs[p] = df

    metrics = ['pdr_exp','delay_s','txrx_data','total_energy']

    rows = []
    for base in ['greedy','leach','heed']:
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

