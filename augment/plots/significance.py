import argparse
import csv
from pathlib import Path
import pandas as pd
import itertools
import numpy as np
import glob
import re


def bootstrap_ci_mean(vals, iters: int = 10000, alpha: float = 0.05, seed: int = 123):
    """Bootstrap CI for the mean of a 1D sample.
    Returns (ci_lo, ci_hi, sample_mean)."""
    arr = np.array(vals, dtype=float)
    n = len(arr)
    mu = float(arr.mean()) if n > 0 else 0.0
    if n <= 1:
        return mu, mu, mu
    rng = np.random.default_rng(seed)
    # Vectorized bootstrap: (iters, n)
    idx = rng.integers(0, n, size=(iters, n))
    means = arr[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi, mu


def bootstrap_ci_diff_mean(x_vals, y_vals, iters: int = 10000, alpha: float = 0.05, seed: int = 123):
    """Bootstrap CI for the difference in means E[X]-E[Y] with independent samples.
    Returns (ci_lo, ci_hi, diff_mean)."""
    x = np.array(x_vals, dtype=float)
    y = np.array(y_vals, dtype=float)
    nx, ny = len(x), len(y)
    diff_mu = float(x.mean() - y.mean()) if (nx > 0 and ny > 0) else 0.0
    if nx <= 1 or ny <= 1:
        return diff_mu, diff_mu, diff_mu
    rng = np.random.default_rng(seed)
    idx_x = rng.integers(0, nx, size=(iters, nx))
    idx_y = rng.integers(0, ny, size=(iters, ny))
    means_x = x[idx_x].mean(axis=1)
    means_y = y[idx_y].mean(axis=1)
    diffs = means_x - means_y
    lo = float(np.quantile(diffs, alpha / 2))
    hi = float(np.quantile(diffs, 1 - alpha / 2))
    return lo, hi, diff_mu


def permutation_pvalue(a, b, metric_key: str, two_sided: bool = True) -> float:
    """
    Exact permutation test for difference in means between two small samples.
    a, b are dataframes with per-seed metrics; metric_key is column name.
    Returns p-value.
    """
    x = a[metric_key].astype(float).tolist()
    y = b[metric_key].astype(float).tolist()
    n, m = len(x), len(y)
    assert n == m, "Require equal sample sizes for exact enumeration to keep combinations manageable"
    all_vals = x + y
    k = n
    obs = abs(sum(x)/n - sum(y)/m)
    count = 0
    total = 0
    # exact enumeration of all combinations C(n+m, n)
    for idx in itertools.combinations(range(n+m), k):
        grp = [all_vals[i] for i in idx]
        rest = [all_vals[i] for i in range(n+m) if i not in idx]
        diff = abs(sum(grp)/k - sum(rest)/m)
        if diff >= obs - 1e-12:
            count += 1
        total += 1
    p = count / total
    return p


# ---- New: Attacks significance helpers ----

def _load_attack_per_seed(root: Path, scenario_label: str) -> pd.DataFrame:
    """Load per-seed CSVs for a scenario label like 'attack_off_smoke', returning a DataFrame with columns
    ['pdr','delay','hops','tx_j','rx_j','cpu_j','txrx']. Missing columns are filled with zeros.
    """
    rows = []
    for p in sorted(root.glob(f"{scenario_label}_seed_*.csv")):
        try:
            df = pd.read_csv(p)
            # Expect single-row per file; if multiple, aggregate by mean
            if len(df) > 1:
                df = df.mean(numeric_only=True).to_frame().T
            r = {
                'pdr': float(df.get('pdr', pd.Series([0])).iloc[0]),
                'delay': float(df.get('delay', pd.Series([0])).iloc[0]),
                'hops': float(df.get('hops', pd.Series([0])).iloc[0]),
                'tx_j': float(df.get('tx_j', pd.Series([0])).iloc[0]),
                'rx_j': float(df.get('rx_j', pd.Series([0])).iloc[0]),
                'cpu_j': float(df.get('cpu_j', pd.Series([0])).iloc[0]),
            }
            r['txrx'] = r['tx_j'] + r['rx_j']
            rows.append(r)
        except Exception:
            continue
    return pd.DataFrame(rows)


def compute_attacks_significance(attacks_root: Path, baseline: str, compare: list[str], metrics: list[str], out_csv: Path, ci_out_csv: Path, ci_iters: int = 10000):
    attacks_root = Path(attacks_root)
    out_rows = []
    ci_rows = []
    base_df = _load_attack_per_seed(attacks_root, baseline)
    if base_df.empty:
        print(f"[ATTACKS][WARN] baseline per-seed not found: {baseline} under {attacks_root}")
        return
    for scenario in compare:
        df = _load_attack_per_seed(attacks_root, scenario)
        if df.empty:
            print(f"[ATTACKS][INFO] skip (no per-seed): {scenario}")
            continue
        # align sizes for exact permutation
        k = min(len(base_df), len(df))
        a = base_df.head(k).reset_index(drop=True)
        b = df.head(k).reset_index(drop=True)
        row = {'baseline': baseline, 'scenario': scenario}
        for met in metrics:
            try:
                pval = permutation_pvalue(a, b, met)
            except AssertionError:
                pval = np.nan
            row[f'p_{met}'] = pval
            # CIs
            t_vals = a[met].astype(float).tolist()
            s_vals = b[met].astype(float).tolist()
            t_lo, t_hi, t_mu = bootstrap_ci_mean(t_vals, iters=ci_iters)
            s_lo, s_hi, s_mu = bootstrap_ci_mean(s_vals, iters=ci_iters)
            d_lo, d_hi, d_mu = bootstrap_ci_diff_mean(t_vals, s_vals, iters=ci_iters)
            ci_rows.append({
                'baseline': baseline,
                'scenario': scenario,
                'metric': met,
                'base_mean': t_mu,
                'base_ci_lo': t_lo,
                'base_ci_hi': t_hi,
                'scen_mean': s_mu,
                'scen_ci_lo': s_lo,
                'scen_ci_hi': s_hi,
                'diff_mean': d_mu,
                'diff_ci_lo': d_lo,
                'diff_ci_hi': d_hi,
                'n': k,
            })
        out_rows.append(row)

    # save p-value csv
    if out_rows:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            headers = ['baseline','scenario'] + [f'p_{m}' for m in metrics]
            w.writerow(headers)
            for r in out_rows:
                w.writerow([r.get(h,'') for h in headers])
        print(f'[ATTACKS] Saved {out_csv}')
    else:
        print('[ATTACKS][INFO] No scenarios compared; skip p-value CSV')

    # save CI csv
    if ci_rows:
        ci_out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(ci_out_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            headers = ['baseline','scenario','metric','base_mean','base_ci_lo','base_ci_hi','scen_mean','scen_ci_lo','scen_ci_hi','diff_mean','diff_ci_lo','diff_ci_hi','n']
            w.writerow(headers)
            for r in ci_rows:
                w.writerow([r.get(h,'') for h in headers])
        print(f'[ATTACKS] Saved {ci_out_csv}')
    else:
        print('[ATTACKS][INFO] No CI rows; skip CI CSV')


def _attacks_scenario_family(name: str) -> str:
    s = str(name)
    if 'blackhole' in s:
        return 'Blackhole'
    if 'grayhole' in s:
        return 'Grayhole'
    if 'sinkhole' in s:
        return 'Sinkhole'
    return 'Baseline'


def _attacks_scenario_pretty(name: str) -> str:
    s = str(name)
    if s == 'attack_off_smoke':
        return 'Baseline (no attack)'
    if 'blackhole' in s:
        # e.g., attack_blackhole_20pct
        m = re.search(r'blackhole_(\d+)pct', s)
        if m:
            return f"Blackhole {m.group(1)}%"
        return 'Blackhole'
    if 'grayhole' in s:
        # e.g., attack_grayhole_50pct_p070
        m1 = re.search(r'(\d+)pct', s)
        m2 = re.search(r'_p(\d+)', s)
        pct = f"{m1.group(1)}%" if m1 else ''
        p = f", p={m2.group(1)[0]}.{m2.group(1)[1:]}" if m2 and len(m2.group(1)) >= 2 else ''
        if pct:
            return f"Grayhole {pct}{p}"
        return f"Grayhole{p}"
    if 'sinkhole' in s:
        # e.g., attack_sinkhole_bias080_fixed_ids
        m = re.search(r'bias(\d+)', s)
        bias = f"0.{m.group(1)}" if m and len(m.group(1)) >= 2 else ''
        return f"Sinkhole{f' bias={bias}' if bias else ''}"
    return s


def _fmt_mu_ci(mu: float, lo: float, hi: float, decimals: int = 3) -> str:
    return f"{mu:.{decimals}f} [{lo:.{decimals}f},{hi:.{decimals}f}]"


def _p_stars(p: float) -> str:
    try:
        if pd.isna(p):
            return ''
        if p < 0.01:
            return '***'
        if p < 0.05:
            return '**'
        if p < 0.1:
            return '*'
        return ''
    except Exception:
        return ''


def format_attacks_tables(attacks_p_csv: Path, attacks_ci_csv: Path, out_csv: Path, out_tex: Path, decimals: int = 3):
    """Combine p-values and CIs into a paper-ready table for attacks.
    Writes a CSV and a LaTeX table (tabular) with pretty names and star annotations.
    """
    attacks_p_csv = Path(attacks_p_csv)
    attacks_ci_csv = Path(attacks_ci_csv)
    if not attacks_p_csv.exists() or not attacks_ci_csv.exists():
        print(f"[ATTACKS][FORMAT][WARN] Missing inputs: {attacks_p_csv} or {attacks_ci_csv}")
        return
    p_df = pd.read_csv(attacks_p_csv)
    # wide to long for metrics
    p_long = p_df.melt(id_vars=['baseline','scenario'], var_name='metric', value_name='p_value')
    p_long['metric'] = p_long['metric'].str.replace('p_', '', regex=False)

    ci_df = pd.read_csv(attacks_ci_csv)

    df = pd.merge(ci_df, p_long, on=['baseline','scenario','metric'], how='left')
    # computed strings
    df['family'] = df['scenario'].apply(_attacks_scenario_family)
    df['scenario_name'] = df['scenario'].apply(_attacks_scenario_pretty)
    df['base_str'] = df.apply(lambda r: _fmt_mu_ci(r['base_mean'], r['base_ci_lo'], r['base_ci_hi'], decimals), axis=1)
    df['scen_str'] = df.apply(lambda r: _fmt_mu_ci(r['scen_mean'], r['scen_ci_lo'], r['scen_ci_hi'], decimals), axis=1)
    df['diff_str'] = df.apply(lambda r: _fmt_mu_ci(r['diff_mean'], r['diff_ci_lo'], r['diff_ci_hi'], decimals), axis=1)
    df['stars'] = df['p_value'].apply(_p_stars)
    # significance from CI of diff excluding 0
    df['sig95'] = ((df['diff_ci_lo'] > 0) | (df['diff_ci_hi'] < 0)).map({True:'yes', False:'no'})

    # order metrics
    met_order = pd.CategoricalDtype(['pdr','delay','txrx'], ordered=True)
    df['metric'] = df['metric'].astype(met_order)
    fam_order = pd.CategoricalDtype(['Baseline','Blackhole','Grayhole','Sinkhole'], ordered=True)
    df['family'] = df['family'].astype(fam_order)

    df = df.sort_values(['family','scenario_name','metric']).reset_index(drop=True)
    # final columns
    out_cols = ['family','scenario_name','metric','n','base_str','scen_str','diff_str','p_value','stars','sig95']
    out_df = df[out_cols]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"[ATTACKS][FORMAT] Saved {out_csv}")

    # LaTeX export
    try:
        tex_df = out_df.copy()
        # use math mode for stars inline next to p-value, e.g., 0.033**
        tex_df['p_value'] = tex_df.apply(lambda r: f"{r['p_value']:.3g}{r['stars']}", axis=1)
        tex_df = tex_df.drop(columns=['stars'])
        out_tex.parent.mkdir(parents=True, exist_ok=True)
        with open(out_tex, 'w', encoding='utf-8') as f:
            f.write(tex_df.to_latex(index=False, escape=False))
        print(f"[ATTACKS][FORMAT] Saved {out_tex}")
    except Exception as e:
        print(f"[ATTACKS][FORMAT][WARN] LaTeX export failed: {e}")


def main():
    ap = argparse.ArgumentParser(description="Compute exact permutation-test p-values vs TrustHR and optional Attacks analysis")
    ap.add_argument('--root', type=str, default='WSN-Intel-Lab-Project/augment/results/baselines')
    ap.add_argument('--nodes', nargs='*', type=int, default=[50,100,200])
    ap.add_argument('--noises', nargs='*', type=float, default=[0.0, 0.2])
    ap.add_argument('--protos', nargs='*', type=str, default=['leach','heed','greedy'])
    ap.add_argument('--metrics', nargs='*', type=str, default=['pdr','delay','txrx'])
    ap.add_argument('--out_csv', type=str, default='WSN-Intel-Lab-Project/augment/results/baselines/significance.csv')
    # New: CI export options
    ap.add_argument('--ci_out_csv', type=str, default='WSN-Intel-Lab-Project/augment/results/baselines/ci.csv')
    ap.add_argument('--ci_iters', type=int, default=10000)
    # New: Attacks mode
    ap.add_argument('--attacks', action='store_true', help='Enable attacks significance mode')
    ap.add_argument('--attacks_root', type=str, default='augment/results', help='Directory containing attack_*_seed_*.csv')
    ap.add_argument('--attacks_baseline', type=str, default='attack_off_smoke', help='Baseline attack scenario label')
    ap.add_argument('--attacks_compare', nargs='*', type=str, default=['attack_blackhole_20pct','attack_grayhole_50pct_p070','attack_sinkhole_bias080_fixed_ids'], help='Attack scenario labels to compare to baseline')
    ap.add_argument('--attacks_out_csv', type=str, default='augment/results/attacks_significance.csv')
    ap.add_argument('--attacks_ci_out_csv', type=str, default='augment/results/attacks_ci.csv')
    # New: Attacks formatted table export
    ap.add_argument('--attacks_format', action='store_true', help='After computing attacks stats, export combined CSV and LaTeX table')
    ap.add_argument('--attacks_table_csv', type=str, default='augment/results/attacks_table.csv')
    ap.add_argument('--attacks_table_tex', type=str, default='augment/results/attacks_table.tex')
    args = ap.parse_args()

    rows = []
    ci_rows = []  # will hold per-scenario/proto CIs for each metric
    for n in args.nodes:
        for z in args.noises:
            # load trusthr per-seed
            trust_p = Path(args.root) / f'trusthr_n{n}_z{z}' / 'per_seed.csv'
            if not trust_p.exists():
                continue
            df_trust = pd.read_csv(trust_p)
            df_trust['txrx'] = df_trust['tx_j'] + df_trust['rx_j']
            for proto in args.protos:
                base_p = Path(args.root) / f'{proto}_n{n}_z{z}' / 'per_seed.csv'
                if not base_p.exists():
                    continue
                df_base = pd.read_csv(base_p)
                df_base['txrx'] = df_base['tx_j'] + df_base['rx_j']
                out = {'nodes': n, 'noise': z, 'vs': proto}
                # p-values
                for met in args.metrics:
                    try:
                        pval = permutation_pvalue(df_trust, df_base, met)
                    except AssertionError:
                        # fallback: if unequal sample size, subsample to min
                        k = min(len(df_trust), len(df_base))
                        pval = permutation_pvalue(df_trust.head(k), df_base.head(k), met)
                    out[f'p_{met}'] = pval
                rows.append(out)

                # CIs for means and mean differences
                for met in args.metrics:
                    t_vals = df_trust[met].astype(float).tolist()
                    b_vals = df_base[met].astype(float).tolist()
                    t_lo, t_hi, t_mu = bootstrap_ci_mean(t_vals, iters=args.ci_iters)
                    b_lo, b_hi, b_mu = bootstrap_ci_mean(b_vals, iters=args.ci_iters)
                    d_lo, d_hi, d_mu = bootstrap_ci_diff_mean(t_vals, b_vals, iters=args.ci_iters)
                    ci_rows.append({
                        'nodes': n,
                        'noise': z,
                        'vs': proto,
                        'metric': met,
                        'trust_mean': t_mu,
                        'trust_ci_lo': t_lo,
                        'trust_ci_hi': t_hi,
                        'base_mean': b_mu,
                        'base_ci_lo': b_lo,
                        'base_ci_hi': b_hi,
                        'diff_mean': d_mu,
                        'diff_ci_lo': d_lo,
                        'diff_ci_hi': d_hi,
                    })

    # save p-value csv
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        headers = ['nodes','noise','vs','p_pdr','p_delay','p_txrx']
        w.writerow(headers)
        for r in rows:
            w.writerow([r.get(h,'') for h in headers])
    print(f'Saved {out_csv}')

    # save CI csv
    ci_out = Path(args.ci_out_csv)
    ci_out.parent.mkdir(parents=True, exist_ok=True)
    with open(ci_out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        headers = ['nodes','noise','vs','metric','trust_mean','trust_ci_lo','trust_ci_hi',
                   'base_mean','base_ci_lo','base_ci_hi','diff_mean','diff_ci_lo','diff_ci_hi']
        w.writerow(headers)
        for r in ci_rows:
            w.writerow([r.get(h,'') for h in headers])
    print(f'Saved {ci_out}')

    # Optional: Attacks mode
    if args.attacks:
        compute_attacks_significance(
            attacks_root=Path(args.attacks_root),
            baseline=args.attacks_baseline,
            compare=list(args.attacks_compare),
            metrics=args.metrics,
            out_csv=Path(args.attacks_out_csv),
            ci_out_csv=Path(args.attacks_ci_out_csv),
            ci_iters=args.ci_iters,
        )
        if args.attacks_format:
            format_attacks_tables(
                attacks_p_csv=Path(args.attacks_out_csv),
                attacks_ci_csv=Path(args.attacks_ci_out_csv),
                out_csv=Path(args.attacks_table_csv),
                out_tex=Path(args.attacks_table_tex),
                decimals=3,
            )


if __name__ == '__main__':
    main()

