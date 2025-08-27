#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sensitivity line plots over h (hello_per_gen) and r (routing_per_gen) for ETX vs TRUSTHR.
- Dual-panel: left = h-sweep at fixed r; right = r-sweep at fixed h
- Unified publication style via plots/style.py
- Multi-format export (svg/png/pdf)
- Optional target threshold horizontal line and crossing annotations
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd
import numpy as np
import os
import matplotlib
if os.environ.get('MPLBACKEND') is None:
    try:
        matplotlib.use('Agg', force=True)
    except Exception:
        pass
import matplotlib.pyplot as plt
import json

# Prefer relative imports if run inside package; fallback to same folder import
try:
    from . import style  # type: ignore
    from .heatmaps_real_sweep import load_summary, ROOT_DEFAULT  # type: ignore
except Exception:  # pragma: no cover
    import style  # type: ignore
    from heatmaps_real_sweep import load_summary, ROOT_DEFAULT  # type: no cover


COLORS = {
    'etx': '#B279A2',       # purple (consistent with other figs)
    'trusthr': '#72B7B2',   # teal   (consistent with other figs)
}
MARKERS = {
    'etx': 'o',
    'trusthr': 's',
}
LABELS = {
    'etx': 'ETX',
    'trusthr': 'TrustHR',
}


def prepare_sweep_series(df: pd.DataFrame, vary: str, fixed_value: int, metric: str, proto: str) -> Tuple[List[int], List[float]]:
    """
    Extract x (varying parameter values) and y (metric) for a given proto.
    - vary: 'h' or 'r'
    - fixed_value: value for the other parameter
    - metric: one of ['pdr','ctrl_percent','delay_s']
    - proto: 'etx' or 'trusthr'
    Returns sorted x list and corresponding y values (NaN filtered out)
    """
    other = 'r' if vary == 'h' else 'h'
    sub = df[(df['proto'] == proto) & (df[other] == int(fixed_value))]
    if sub.empty:
        return [], []
    # sort by vary
    sub = sub.sort_values(by=vary)
    xs = sub[vary].astype(int).tolist()
    ys = sub[metric].astype(float).tolist()
    return xs, ys


def _find_crossing_index(ys: List[float], target: float) -> Optional[int]:
    """Return an index near where the series crosses target (sign change between segments),
    or None if there is no crossing. If any point equals the target, return that index.
    """
    if target is None or len(ys) == 0:
        return None
    # Exact hit
    for i, y in enumerate(ys):
        if np.isfinite(y) and abs(y - target) < 1e-12:
            return i
    # Segment crossing
    for i in range(len(ys) - 1):
        y0, y1 = ys[i], ys[i + 1]
        if not (np.isfinite(y0) and np.isfinite(y1)):
            continue
        if (y0 - target) * (y1 - target) < 0:
            # choose closer endpoint to the crossing
            return i if abs(y0 - target) <= abs(y1 - target) else i + 1
    return None


def annotate_threshold(ax: plt.Axes, xs: List[int], ys: List[float], target: float, label: str, color: str,
                       offset: Tuple[int, int] = (10, 14), ha: str = 'left') -> None:
    """Annotate the crossing point if the series crosses the target; otherwise annotate the closest point as 'closest'.
    Added offset and alignment to avoid text overlaps between protocols.
    """
    if not xs or not ys or target is None:
        return
    # Try to find a true crossing first
    cross_idx = _find_crossing_index(ys, target)
    if cross_idx is not None:
        x0, y0 = xs[cross_idx], ys[cross_idx]
        ax.plot([x0], [y0], marker='D', color=color, markersize=6, zorder=5)
        text = f'hit {label}\n@{x0}: {y0:.3g}'
    else:
        # No crossing -> annotate the closest point, but DO NOT claim a hit
        arr = np.array(ys, dtype=float)
        idx = int(np.nanargmin(np.abs(arr - target))) if np.isfinite(arr).any() else None
        if idx is None:
            return
        x0, y0 = xs[idx], ys[idx]
        ax.plot([x0], [y0], marker='D', color=color, markersize=6, zorder=5)
        text = f'closest to {label}\n@{x0}: {y0:.3g}'
    ax.annotate(text,
                xy=(x0, y0), xytext=offset, textcoords='offset points',
                fontsize=style.scaled_size(0.85), color=color, ha=ha,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='none'),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.0))


def _qc_svg(svg_path: Path, metric: str, target: float, annotate_crossing: bool) -> Dict[str, Any]:
    """Lightweight QC by scanning SVG text for expected labels/annotations.
    Returns a dict with individual checks and overall pass flag.
    """
    checks: Dict[str, Any] = {
        'exists': False,
        'has_target_legend': None,
        'has_crossing_annotation': None,
        'has_ylabel_text': None,
        'strict_pass': False,
    }
    if not svg_path.exists():
        return checks
    checks['exists'] = True
    try:
        text = svg_path.read_text(encoding='utf-8', errors='ignore')
    except Exception as e:
        checks['read_error'] = str(e)
        return checks

    # Check target legend text if target is requested
    if target is not None:
        checks['has_target_legend'] = (f'target={target}' in text) or ('target=' in text)
    else:
        checks['has_target_legend'] = True  # not required

    # Check annotations text when requested
    if annotate_crossing and target is not None:
        checks['has_crossing_annotation'] = ('hit target' in text) or ('closest to target' in text)
    else:
        checks['has_crossing_annotation'] = True  # not required

    # Y label presence
    ylabel_map = {
        'pdr': 'PDR',
        'ctrl_percent': 'Control Energy (%)',
        'delay_s': 'Avg E2E Delay (s)'
    }
    ylabel = ylabel_map.get(metric, metric)
    checks['has_ylabel_text'] = (ylabel in text)

    # Overall pass if all required checks are True
    required_keys = ['exists', 'has_target_legend', 'has_crossing_annotation', 'has_ylabel_text']
    checks['strict_pass'] = all(bool(checks.get(k)) for k in required_keys)
    return checks


def plot_sens_lines(df_mode: pd.DataFrame, mode: str, metric: str, fixed_h: int, fixed_r: int,
                    out_dir: Path, formats: List[str], target_y: Optional[float], annotate_crossing: bool,
                    qc: bool = False, qc_strict: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Figure configuration
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    # Left: h-sweep at fixed r
    ax = axes[0]
    for proto in ['etx', 'trusthr']:
        xs, ys = prepare_sweep_series(df_mode, 'h', fixed_r, metric, proto)
        if not xs:
            continue
        ax.plot(xs, ys, marker=MARKERS[proto], color=COLORS[proto], label=LABELS[proto], linewidth=2)
        if annotate_crossing and target_y is not None:
            # Different offsets to reduce overlap between protocols
            off = (12, 18) if proto == 'etx' else (-86, -20)  # expanded to avoid overlaps
            halign = 'left' if proto == 'etx' else 'right'
            annotate_threshold(ax, xs, ys, target_y, 'target', COLORS[proto], offset=off, ha=halign)
    ax.set_xlabel('hello_per_gen (h)')
    ax.set_title(f'h-sweep (r={fixed_r})', loc='left')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

    # Right: r-sweep at fixed h
    ax = axes[1]
    for proto in ['etx', 'trusthr']:
        xs, ys = prepare_sweep_series(df_mode, 'r', fixed_h, metric, proto)
        if not xs:
            continue
        ax.plot(xs, ys, marker=MARKERS[proto], color=COLORS[proto], label=LABELS[proto], linewidth=2)
        if annotate_crossing and target_y is not None:
            off = (12, 18) if proto == 'etx' else (-86, -20)
            halign = 'left' if proto == 'etx' else 'right'
            annotate_threshold(ax, xs, ys, target_y, 'target', COLORS[proto], offset=off, ha=halign)
    ax.set_xlabel('routing_per_gen (r)')
    ax.set_title(f'r-sweep (h={fixed_h})', loc='left')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

    # Shared Y label and legend
    ylabel = {
        'pdr': 'PDR',
        'ctrl_percent': 'Control Energy (%)',
        'delay_s': 'Avg E2E Delay (s)'
    }.get(metric, metric)
    axes[0].set_ylabel(ylabel)

    # Reasonable y-limits
    if metric == 'pdr':
        axes[0].set_ylim(0.0, 1.05)
    elif metric == 'ctrl_percent':
        # infer if values already percent-like (>1.5) from df
        if float(df_mode['ctrl_percent'].max()) <= 1.5:
            axes[0].set_ylim(0, 1.05)
        else:
            axes[0].set_ylim(0, max(100.0, float(df_mode['ctrl_percent'].max()) * 1.05))

    # Target threshold line (show on the right axis only to avoid duplicate legend entries)
    if target_y is not None:
        axes[1].axhline(target_y, color='#888888', linestyle=':', linewidth=1.2, label=f'target={target_y}')

    # Place legend to avoid overlapping data; use the right axis
    axes[1].legend(loc='lower right', frameon=True)
    for ax in axes:
        style.beautify_axes(ax)

    # Title at figure level
    fig.suptitle(f'{mode.upper()} sensitivity: {ylabel}', x=0.01, y=0.98, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    base = out_dir / f'sens_{mode}_{metric}'
    style.savefig_multi(fig, base, formats=formats)
    plt.close(fig)

    # QC step (SVG text scan + JSON log)
    if qc:
        svg_path = base.with_suffix('.svg')
        svg_checks = _qc_svg(svg_path, metric, target_y, annotate_crossing)
        # Data sanity checks based on df values (more robust than parsing SVG ticks)
        series = pd.to_numeric(df_mode[metric], errors='coerce')
        series = series[np.isfinite(series)]
        data_checks = {
            'data_points': int(series.size),
            'data_min': None,
            'data_max': None,
            'unit_class': None,  # for ctrl_percent: 'fraction' or 'percent'
            'unit_mismatch': False,
            'range_ok': True,
            'pass': False,
        }
        if series.size > 0:
            vmin = float(series.min())
            vmax = float(series.max())
            data_checks['data_min'] = vmin
            data_checks['data_max'] = vmax
            if metric == 'pdr':
                data_checks['range_ok'] = (vmin >= -1e-9) and (vmax <= 1.05)
            elif metric == 'ctrl_percent':
                percent_like = vmax <= 1.5
                data_checks['unit_class'] = 'fraction' if percent_like else 'percent'
                if percent_like:
                    data_checks['range_ok'] = (vmin >= -1e-9) and (vmax <= 1.05)
                else:
                    data_checks['range_ok'] = (vmin >= -1e-6) and (vmax <= 105.0)
                if target_y is not None:
                    # If data looks like fraction but target is >1.5 (looks percent), or vice versa -> mismatch
                    if percent_like and target_y > 1.5:
                        data_checks['unit_mismatch'] = True
                    if (not percent_like) and target_y <= 1.5:
                        data_checks['unit_mismatch'] = True
            elif metric == 'delay_s':
                data_checks['range_ok'] = (vmin >= -1e-9) and bool(np.isfinite(vmax))
            else:
                # Unknown metric: do not enforce extra constraints
                data_checks['range_ok'] = True
        else:
            data_checks['range_ok'] = False
        # Overall pass requires both SVG checks and data checks to be good, and no unit mismatch
        combined = {**svg_checks, **data_checks}
        combined['strict_pass'] = bool(svg_checks.get('strict_pass', False) and data_checks['range_ok'] and not data_checks['unit_mismatch'])
        combined['pass'] = combined['strict_pass']
        qc_path = base.with_suffix('.qc.json')
        try:
            qc_path.write_text(
    json.dumps(
        combined,
        indent=2,
        default=lambda o: bool(o) if isinstance(o, (np.bool_, np.bool8)) else (
            float(o) if isinstance(o, (np.floating,)) else (
                int(o) if isinstance(o, (np.integer,)) else (
                    o.tolist() if isinstance(o, np.ndarray) else str(o)
                )
            )
        )
    ),
    encoding='utf-8'
)

        except Exception as e:
            print(f'[WARN] Failed to write QC log: {e}')
        if qc_strict and not combined.get('strict_pass', False):
            print(f'[ERROR] QC failed for {svg_path}: {combined}')
            raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description='Generate dual-panel sensitivity line plots (h/r sweep) for ETX vs TRUSTHR.')
    ap.add_argument('--root', default=str(ROOT_DEFAULT), help='Project root path')
    ap.add_argument('--mode', default='both', choices=['elec', 'amp', 'both'], help='Which mode(s) to plot')
    ap.add_argument('--metric', default='pdr', choices=['pdr', 'ctrl_percent', 'delay_s', 'all'], help='Metric to plot, or "all" to generate all metrics')
    ap.add_argument('--fixed-h', type=int, default=500, help='Fixed h when sweeping r')
    ap.add_argument('--fixed-r', type=int, default=500, help='Fixed r when sweeping h')
    ap.add_argument('--outdir-elec', default='augment/results/real/sens/figs_pub/elec', help='Output dir for ELEC figures')
    ap.add_argument('--outdir-amp', default='augment/results/real/sens/figs_pub/amp', help='Output dir for AMP figures')
    ap.add_argument('--formats', default='svg,png', help='Comma-separated formats, e.g. svg,png,pdf')
    ap.add_argument('--target-y', type=float, default=None, help='Draw horizontal threshold line at this Y value')
    ap.add_argument('--annotate-crossing', action='store_true', help='Annotate closest points to the target line')
    # Style
    ap.add_argument('--theme', default='light', choices=['light', 'dark'])
    ap.add_argument('--context', default='talk', choices=['paper', 'talk', 'notebook', 'poster'])
    ap.add_argument('--font', default='DejaVu Sans')
    ap.add_argument('--base-font-size', type=int, default=12)
    ap.add_argument('--palette', default='default')
    # QC
    ap.add_argument('--qc', action='store_true', help='Enable SVG auto quality checks and emit qc.json next to figures')
    ap.add_argument('--qc-strict', action='store_true', help='Fail with non-zero exit if QC fails')

    args = ap.parse_args()

    root = Path(args.root)
    outdir_elec = Path(args.outdir_elec)
    outdir_amp = Path(args.outdir_amp)
    if not outdir_elec.is_absolute():
        outdir_elec = root / outdir_elec
    if not outdir_amp.is_absolute():
        outdir_amp = root / outdir_amp

    # Apply unified style
    style.apply_style(theme=args.theme, context=args.context, font=args.font, base_font_size=args.base_font_size)
    _ = style.get_palette(args.palette)
    plt.rcParams['svg.fonttype'] = 'none'

    # Load data
    modes = []
    if args.mode in ('elec', 'both'):
        try:
            df_elec = load_summary(root, 'elec')
            modes.append(('elec', df_elec, outdir_elec))
        except FileNotFoundError as e:
            print(f'[WARN] ELEC summary not found: {e}')
    if args.mode in ('amp', 'both'):
        try:
            df_amp = load_summary(root, 'amp')
            modes.append(('amp', df_amp, outdir_amp))
        except FileNotFoundError as e:
            print(f'[WARN] AMP summary not found: {e}')

    fmts = [s.strip() for s in args.formats.split(',') if s.strip()]
    metrics_list = ['pdr', 'ctrl_percent', 'delay_s'] if args.metric == 'all' else [args.metric]

    for mode, dfm, outd in modes:
        for metric in metrics_list:
            plot_sens_lines(
                dfm, mode, metric, args.fixed_h, args.fixed_r, outd, fmts,
                args.target_y, args.annotate_crossing, qc=args.qc, qc_strict=args.qc_strict
            )
            print(f'Saved sensitivity plots to {outd} (mode={mode}, metric={metric})')


if __name__ == '__main__':
    main()