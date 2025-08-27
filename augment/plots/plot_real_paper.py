import argparse
from pathlib import Path
import json
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Prefer relative style import if available
try:
    from . import style  # type: ignore
except Exception:  # pragma: no cover
    import style  # type: ignore

# For supplemental data (heatmaps/sensitivity)
try:
    from .heatmaps_real_sweep import load_summary as _load_summary_supp, ROOT_DEFAULT as _ROOT_DEFAULT  # type: ignore
except Exception:  # pragma: no cover
    from heatmaps_real_sweep import load_summary as _load_summary_supp, ROOT_DEFAULT as _ROOT_DEFAULT  # type: ignore

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
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='none'), zorder=6)


# ---------- Supplemental helpers (Plan A) ----------

def _compute_global_vrange(df: pd.DataFrame, value_col: str):
    vmin = float(df[value_col].min())
    vmax = float(df[value_col].max())
    if np.isfinite(vmin) and np.isfinite(vmax) and abs(vmax - vmin) < 1e-9:
        eps = 1e-3 if value_col == 'pdr' else max(0.01, vmax * 0.01)
        vmin -= eps
        vmax += eps
    return vmin, vmax


def _draw_heatmap_panel(ax: plt.Axes, df_mode: pd.DataFrame, proto: str, value_col: str, title: str):
    sub = df_mode[df_mode['proto'] == proto]
    if sub.empty:
        ax.set_visible(False)
        return
    mat = sub.pivot(index='h', columns='r', values=value_col).sort_index().sort_index(axis=1)
    vmin, vmax = _compute_global_vrange(df_mode, value_col)
    annot = False  # keep compact panels
    sns.heatmap(
        mat, ax=ax, annot=annot, fmt='', cmap=('YlGnBu' if value_col=='pdr' else 'YlOrRd'),
        vmin=vmin, vmax=vmax, cbar=False, linewidths=0.4, linecolor='white'
    )
    ax.set_title(title, loc='left')
    ax.set_xlabel('r')
    ax.set_ylabel('h')
    # readable ticks
    ax.set_xticklabels(sorted(mat.columns.tolist()))
    ax.set_yticklabels(sorted(mat.index.tolist()))
    try:
        style.beautify_axes(ax)
    except Exception:
        pass


def _prepare_series(df: pd.DataFrame, vary: str, fixed_value: int, metric: str, proto: str):
    other = 'r' if vary == 'h' else 'h'
    sub = df[(df['proto'] == proto) & (df[other] == int(fixed_value))].sort_values(by=vary)
    if sub.empty:
        return [], []
    return sub[vary].astype(int).tolist(), sub[metric].astype(float).tolist()


def _draw_sensitivity_panels(ax_left: plt.Axes, ax_right: plt.Axes, df_mode: pd.DataFrame, metric: str,
                              fixed_h: int, fixed_r: int):
    # Left: h-sweep at fixed r
    for proto in ['etx', 'trusthr']:
        xs, ys = _prepare_series(df_mode, 'h', fixed_r, metric, proto)
        if not xs: continue
        ax_left.plot(xs, ys, marker=('o' if proto=='etx' else 's'), color=COLORS[proto], label=LABELS[proto], linewidth=2)
    ax_left.set_xlabel('h')
    ax_left.set_title(f'h-sweep (r={fixed_r})', loc='left')
    ax_left.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

    # Right: r-sweep at fixed h
    for proto in ['etx', 'trusthr']:
        xs, ys = _prepare_series(df_mode, 'r', fixed_h, metric, proto)
        if not xs: continue
        ax_right.plot(xs, ys, marker=('o' if proto=='etx' else 's'), color=COLORS[proto], label=LABELS[proto], linewidth=2)
    ax_right.set_xlabel('r')
    ax_right.set_title(f'r-sweep (h={fixed_h})', loc='left')
    ax_right.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

    ylabel = {'pdr': 'PDR', 'ctrl_percent': 'Control Energy (%)', 'delay_s': 'Avg E2E Delay (s)'}
    ax_left.set_ylabel(ylabel.get(metric, metric))

    # y-lims heuristics
    if metric == 'pdr':
        ax_left.set_ylim(0.0, 1.05)
    elif metric == 'ctrl_percent':
        vmax = float(df_mode['ctrl_percent'].max()) if 'ctrl_percent' in df_mode.columns else 1.0
        if vmax <= 1.5:
            ax_left.set_ylim(0, 1.05)
        else:
            ax_left.set_ylim(0, max(100.0, vmax * 1.05))

    ax_right.legend(loc='lower right', frameon=True)
    for ax in (ax_left, ax_right):
        try:
            style.beautify_axes(ax)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description='Paper-ready multi-panel plot for real-data evaluation')
    ap.add_argument('--root', type=str, default='WSN-Intel-Lab-Project/augment/results/real')
    ap.add_argument('--out_png', type=str, default='WSN-Intel-Lab-Project/augment/results/real/figures/real_overview_paper.png')
    ap.add_argument('--out_pdf', type=str, default='WSN-Intel-Lab-Project/augment/results/real/figures/real_overview_paper.pdf')
    ap.add_argument('--out_svg', type=str, default='WSN-Intel-Lab-Project/augment/results/real/figures/real_overview_paper.svg')
    # Supplemental integration (Plan A)
    ap.add_argument('--include-supp', action='store_true', help='Include supplemental panels (heatmaps + sensitivity lines)')
    ap.add_argument('--project-root', type=str, default=str(_ROOT_DEFAULT), help='Project root for supplemental data (summary CSVs)')
    ap.add_argument('--supp-mode', type=str, choices=['elec','amp'], default='elec', help='Mode for supplemental panels')
    ap.add_argument('--supp-heatmap-metric', choices=['pdr','ctrl_percent'], default='pdr', help='Heatmap metric')
    ap.add_argument('--supp-sens-metric', choices=['pdr','ctrl_percent','delay_s'], default='pdr', help='Sensitivity metric')
    ap.add_argument('--sens-fixed-h', type=int, default=500, help='Fixed h for r-sweep panel')
    ap.add_argument('--sens-fixed-r', type=int, default=500, help='Fixed r for h-sweep panel')
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

    # Dynamic layout: 2x2 (original) or 4x2 (with supplemental panels)
    nrows = 4 if args.include_supp else 2
    fig_height = 5.4 if nrows == 2 else 10.8
    fig, axes = plt.subplots(nrows, 2, figsize=(8.4, fig_height))

    # PDR
    ax = axes[0,0]
    ax.bar(range(len(xs)), pdr_vals, color=cs, edgecolor='black', linewidth=0.6,
           yerr=np.vstack([pdr_err_low, pdr_err_high]), capsize=3, ecolor='black', error_kw=dict(linewidth=0.6))
    ax.set_xticks(range(len(xs)), xs)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(15)
        lbl.set_ha('right')
    ax.set_ylabel('Expected PDR')
    ax.set_title('Packet Delivery Ratio (Real Dataset)')
    ax.grid(True, axis='y', alpha=0.3)
    ax.margins(y=0.08)
    annotate(ax, 'pdr_exp', sig)
    ax.text(0.0, 1.02, 'A', transform=ax.transAxes, fontsize=12, weight='bold', va='bottom', ha='left')

    # Delay
    ax = axes[0,1]
    ax.bar(range(len(xs)), delay_vals, color=cs, edgecolor='black', linewidth=0.6,
           yerr=np.vstack([delay_err_low, delay_err_high]), capsize=3, ecolor='black', error_kw=dict(linewidth=0.6))
    ax.set_xticks(range(len(xs)), xs)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(15)
        lbl.set_ha('right')
    ax.set_ylabel('Avg E2E Delay (s)')
    ax.set_title('Latency')
    ax.grid(True, axis='y', alpha=0.3)
    ax.margins(y=0.08)
    annotate(ax, 'delay_s', sig)
    ax.text(0.0, 1.02, 'B', transform=ax.transAxes, fontsize=12, weight='bold', va='bottom', ha='left')

    # Data-plane energy (TX+RX)
    ax = axes[1,0]
    ax.bar(range(len(xs)), data_energy_vals, color=cs, edgecolor='black', linewidth=0.6,
           yerr=np.vstack([energy_err_low, energy_err_high]), capsize=3, ecolor='black', error_kw=dict(linewidth=0.6))
    ax.set_xticks(range(len(xs)), xs)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(15)
        lbl.set_ha('right')
    ax.set_ylabel('Data-plane Energy (J)')
    ax.set_title('Energy (TX+RX, Data-plane)')
    ax.grid(True, axis='y', alpha=0.3)
    ax.margins(y=0.08)
    annotate(ax, 'txrx_data', sig)
    ax.text(0.0, 1.02, 'C', transform=ax.transAxes, fontsize=12, weight='bold', va='bottom', ha='left')

    # Stacked energy (Control vs Data)
    ax = axes[1,1]
    x = list(range(len(xs)))
    ax.bar(x, ctrl_energy_vals, color='#D3D3D3', edgecolor='black', linewidth=0.6, label='Control-plane')
    ax.bar(x, data_energy_vals, bottom=ctrl_energy_vals, color=cs, edgecolor='black', linewidth=0.6, label='Data-plane')
    for i in x:
        tot = ctrl_energy_vals[i] + data_energy_vals[i]
        ax.text(i, tot*1.01, f"{tot:,.0f} J", ha='center', va='bottom', fontsize=8, clip_on=False, zorder=5)
    ax.set_xticks(x, xs)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(15)
        lbl.set_ha('right')
    ax.set_ylabel('Total Energy (J)')
    ax.set_title('Total Energy Breakdown')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(loc='upper right', framealpha=0.85)
    ax.margins(y=0.15)
    ax.text(0.0, 1.02, 'D', transform=ax.transAxes, fontsize=12, weight='bold', va='bottom', ha='left')

    # ---- Supplemental panels ----
    if args.include_supp:
        proj_root = Path(args.project_root)
        try:
            df_mode = _load_summary_supp(proj_root, args.supp_mode)
            # Ensure integer types
            df_mode = df_mode.copy()
            if 'h' in df_mode.columns: df_mode['h'] = df_mode['h'].astype(int)
            if 'r' in df_mode.columns: df_mode['r'] = df_mode['r'].astype(int)
        except Exception as e:
            df_mode = None
            print(f'[WARN] Failed to load supplemental summary for {args.supp_mode}: {e}')

        if df_mode is not None:
            # Row 3: heatmaps (ETX vs TRUSTHR) for selected metric
            hm_metric = args.supp_heatmap_metric
            ax_hm_left = axes[2,0]
            ax_hm_right = axes[2,1]
            _draw_heatmap_panel(ax_hm_left, df_mode, 'etx', hm_metric, title=f'ETX {args.supp_mode.upper()} {("PDR" if hm_metric=="pdr" else "CTRL %")}')
            _draw_heatmap_panel(ax_hm_right, df_mode, 'trusthr', hm_metric, title=f'TrustHR {args.supp_mode.upper()} {("PDR" if hm_metric=="pdr" else "CTRL %")}')
            ax_hm_left.text(0.0, 1.02, 'E', transform=ax_hm_left.transAxes, fontsize=12, weight='bold', va='bottom', ha='left')
            ax_hm_right.text(0.0, 1.02, 'F', transform=ax_hm_right.transAxes, fontsize=12, weight='bold', va='bottom', ha='left')

            # Row 4: sensitivity lines (dual panel: h-sweep, r-sweep)
            ax_s_left = axes[3,0]
            ax_s_right = axes[3,1]
            _draw_sensitivity_panels(ax_s_left, ax_s_right, df_mode, args.supp_sens_metric, args.sens_fixed_h, args.sens_fixed_r)
            ax_s_left.text(0.0, 1.02, 'G', transform=ax_s_left.transAxes, fontsize=12, weight='bold', va='bottom', ha='left')
            ax_s_right.text(0.0, 1.02, 'H', transform=ax_s_right.transAxes, fontsize=12, weight='bold', va='bottom', ha='left')

    # footnote
    fig.text(0.01, 0.01,
             'Note: Control-plane overheads are computed deterministically and are identical across protocols;\n'
             'data-plane improvements refer to TX+RX energy on real topology using connectivity-derived link success.',
             fontsize=8)

    fig.tight_layout(rect=(0,0.08,1,1))
    fig.savefig(args.out_png, dpi=600, bbox_inches='tight')
    fig.savefig(args.out_pdf, bbox_inches='tight')
    fig.savefig(args.out_svg, bbox_inches='tight')
    print(f'Saved {args.out_png}, {args.out_pdf} and {args.out_svg}')


if __name__ == '__main__':
    main()

