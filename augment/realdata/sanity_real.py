import argparse
import os
from pathlib import Path
import json
import pandas as pd

# Inject project root to sys.path to allow `import augment.*` when running by file path
import sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from augment.realdata.eval_real import eval_real

PROTOS = ['greedy','leach','heed','etx','trusthr']  # keep pegasis optional


def recompute_agg_from_per_mote(root: Path, out_csv: Path):
    rows = []
    for p in PROTOS + ['pegasis']:
        pm = root / f'{p}_per_mote.csv'
        agg = root / f'{p}_agg.json'
        if not pm.exists() or not agg.exists():
            continue
        df = pd.read_csv(pm)
        # recompute
        total_gen = max(1, df['gen_count'].sum())
        rec = {
            'proto': p,
            'pdr_mean_re': float((df['pdr_exp'] * df['gen_count']).sum() / total_gen),
            'delay_mean_re': float((df['delay_s'] * df['gen_count']).sum() / total_gen),
            'tx_j_sum_data_re': float(df['tx_j_data'].sum()),
            'rx_j_sum_data_re': float(df['rx_j_data'].sum()),
        }
        with open(agg, 'r', encoding='utf-8') as f:
            a = json.load(f)
        rec.update({
            'pdr_mean': a.get('pdr_mean'),
            'delay_mean': a.get('delay_mean'),
            'tx_j_sum_data': a.get('tx_j_sum_data'),
            'rx_j_sum_data': a.get('rx_j_sum_data'),
        })
        rows.append(rec)
    out = pd.DataFrame(rows)

    # If no data rows were found, write an empty CSV with headers to avoid KeyError downstream
    if out.empty:
        cols = ['proto','pdr_mean_re','delay_mean_re','tx_j_sum_data_re','rx_j_sum_data_re',
                'pdr_mean','delay_mean','tx_j_sum_data','rx_j_sum_data',
                'pdr_diff_pp','delay_diff_abs','tx_diff_pct','rx_diff_pct']
        out = pd.DataFrame(columns=cols)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv, index=False)
        print(f'No per_mote/agg found under {root}, wrote empty {out_csv}')
        return

    # Safe computations with zero-denominator protection
    out['pdr_diff_pp'] = (out['pdr_mean_re'] - out['pdr_mean']) * 100
    out['delay_diff_abs'] = (out['delay_mean_re'] - out['delay_mean'])
    tx_den = out['tx_j_sum_data'].replace(0, 1e-12)
    rx_den = out['rx_j_sum_data'].replace(0, 1e-12)
    out['tx_diff_pct'] = (out['tx_j_sum_data_re'] - out['tx_j_sum_data']) / tx_den * 100
    out['rx_diff_pct'] = (out['rx_j_sum_data_re'] - out['rx_j_sum_data']) / rx_den * 100

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f'Saved {out_csv}')


def scale_connectivity(in_path: Path, out_path: Path, factor: float):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(in_path, 'r', encoding='utf-8', errors='ignore') as fin, open(out_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    w = float(parts[2]) * factor
                    w = max(0.0, min(1.0, w))
                    parts[2] = f"{w}"
                    fout.write(' '.join(parts) + '\n')
                    continue
                except Exception:
                    pass
            fout.write(line)
    print(f'Scaled connectivity -> {out_path} (factor={factor})')


def sensitivity(root_data: Path, cleaned_csv: Path, mote_locs: Path, out_root: Path, factors=(0.95, 1.00, 1.05)):
    base_conn = root_data / 'connectivity.txt'
    for f in factors:
        conn_scaled = out_root / f'conn_scaled_{f:.2f}.txt'
        scale_connectivity(base_conn, conn_scaled, f)
        for p in PROTOS:
            out_prefix = out_root / f'f{f:.2f}_{p}_elec'
            eval_real(str(conn_scaled), str(cleaned_csv), str(mote_locs), bs_id=0, payload_bits=1024, slot_s=0.02,
                      out_prefix=str(out_prefix), proto=p, energy_mode='elec')


def main():
    ap = argparse.ArgumentParser(description='Sanity checks: aggregate recompute and sensitivity runs')
    ap.add_argument('--real_root', type=str, default='WSN-Intel-Lab-Project/augment/results/real')
    ap.add_argument('--data_root', type=str, default='WSN-Intel-Lab-Project/data')
    ap.add_argument('--out_root', type=str, default='WSN-Intel-Lab-Project/augment/results/real/sanity')
    args = ap.parse_args()

    real_root = Path(args.real_root)
    out_root = Path(args.out_root)

    # 1) recompute aggregates from per_mote and diff against agg.json
    recompute_agg_from_per_mote(real_root, out_root / 'agg_check.csv')

    # 2) sensitivity: scale connectivity by ±5%
    sensitivity(Path(args.data_root), Path(args.data_root) / 'processed/cleaned_data.csv', Path(args.data_root) / 'mote_locs.txt', out_root)


if __name__ == '__main__':
    main()

