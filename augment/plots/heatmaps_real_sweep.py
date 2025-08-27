#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 summary CSV 生成 ELEC/AMP 模式下 ETX 与 TRUSTHR 的 3×3 热力图（PDR 与控制能耗占比）。
输出为 SVG/PNG（等）多格式矢量/栅格图；统一使用项目的出版级样式。
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import os
import matplotlib
# 使用非交互式后端，避免无 Tcl/Tk 环境报错
if os.environ.get('MPLBACKEND') is None:
    try:
        matplotlib.use('Agg', force=True)
    except Exception:
        pass
import matplotlib.pyplot as plt
import seaborn as sns

# 统一样式模块（相对导入优先）
try:
    from . import style  # type: ignore
except Exception:  # pragma: no cover
    import style  # type: ignore

ROOT_DEFAULT = Path('c:/WSN-Intel-Lab-Project')


def load_summary(root: Path, mode: str) -> pd.DataFrame:
    if mode == 'elec':
        csv_path = root / 'augment/results/real/sweep/elec/summary_etx_trusthr_elec.csv'
    elif mode == 'amp':
        csv_path = root / 'augment/results/real/sweep/amp/summary_etx_trusthr_amp.csv'
    else:
        raise ValueError('mode must be elec or amp')

    if not csv_path.exists():
        raise FileNotFoundError(f'CSV not found: {csv_path}')

    df = pd.read_csv(csv_path)
    # 统一列名
    if mode == 'elec':
        df = df.rename(columns={
            'hello_per_gen': 'h',
            'routing_per_gen': 'r',
            'pdr_mean': 'pdr',
            'ctrl_pct': 'ctrl_percent',
            'delay_mean': 'delay_s',
        })
    else:  # amp 基本已统一
        df = df.rename(columns={
            'delay': 'delay_s',  # 以防万一
        })

    # 过滤两种协议并规范类型
    df = df[df['proto'].isin(['etx', 'trusthr'])].copy()
    df['h'] = df['h'].astype(int)
    df['r'] = df['r'].astype(int)
    return df


def _compute_global_vrange(df: pd.DataFrame, value_col: str):
    vmin = float(df[value_col].min())
    vmax = float(df[value_col].max())
    if np.isfinite(vmin) and np.isfinite(vmax) and abs(vmax - vmin) < 1e-9:
        # 扩一丢丢边界，避免纯色热力图难以辨识
        eps = 1e-3 if value_col == 'pdr' else max(0.01, vmax * 0.01)
        vmin -= eps
        vmax += eps
    return vmin, vmax


def _format_annot(value: float, is_percent_like: bool, fmt: str) -> str:
    if is_percent_like:
        return f"{value:.1f}%"
    try:
        return format(value, fmt)
    except Exception:
        return f"{value:.3g}"


def plot_heatmaps_for_mode(df_mode: pd.DataFrame, mode: str, out_dir: Path,
                           formats: list[str], cmap_pdr: str, cmap_ctrl: str,
                           annotate: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric, value_col, fmt, cmap in [
        ('PDR', 'pdr', '.3f', cmap_pdr),
        ('CTRL %', 'ctrl_percent', '.2f', cmap_ctrl),
    ]:
        # 识别 ctrl 是否以 0-100 计；若是则标注加 %，色尺直接按 0-100 处理
        if value_col == 'ctrl_percent':
            ctrl_max = float(df_mode[value_col].max()) if not df_mode.empty else 0.0
            ctrl_is_percent_like = ctrl_max > 1.5  # 认为>1.5 即已经是百分比数值
        else:
            ctrl_is_percent_like = False

        vmin, vmax = _compute_global_vrange(df_mode, value_col)

        saved = []
        for proto in ['etx', 'trusthr']:
            dfp = df_mode[df_mode['proto'] == proto]
            if dfp.empty:
                continue
            mat = dfp.pivot(index='h', columns='r', values=value_col).sort_index().sort_index(axis=1)

            fig, ax = plt.subplots(figsize=(6.0, 4.8))
            # 将注解转换为字符串以便百分号标注
            annot = None
            if annotate:
                annot = mat.copy()
                try:
                    annot = annot.map(lambda v: _format_annot(v, ctrl_is_percent_like and value_col == 'ctrl_percent', fmt))
                except Exception:
                    annot = annot.applymap(lambda v: _format_annot(v, ctrl_is_percent_like and value_col == 'ctrl_percent', fmt))

            sns.heatmap(
                mat, ax=ax, annot=(annot if annotate else False), fmt='', cmap=cmap,
                vmin=vmin, vmax=vmax, cbar=True, linewidths=0.6, linecolor='white',
                annot_kws={"fontsize": style.scaled_size(0.85)} if annotate else None
            )

            ax.set_title(f'{proto.upper()} {mode.upper()} {metric}', loc='left', pad=8)
            ax.set_xlabel('routing_per_gen (r)')
            ax.set_ylabel('hello_per_gen (h)')
            # 确保刻度标签按数值排序显示
            ax.set_xticklabels(sorted(mat.columns.tolist()))
            ax.set_yticklabels(sorted(mat.index.tolist()))
            style.beautify_axes(ax)

            style.savefig_multi(fig, out_dir / f'{proto}_{mode}_{"pdr" if value_col=="pdr" else "ctrlpct"}', formats=formats)
            plt.close(fig)
            saved.append(str(out_dir / f'{proto}_{mode}_{"pdr" if value_col=="pdr" else "ctrlpct"}'))
        print(f'[{mode}] {metric} saved files (base path per format):')
        for p in saved:
            print(' -', p)


def main():
    ap = argparse.ArgumentParser(description='Generate publication-grade heatmaps for ETX/TRUSTHR (ELEC & AMP).')
    ap.add_argument('--root', default=str(ROOT_DEFAULT), help='Project root path')
    ap.add_argument('--outdir-elec', default='augment/results/real/sweep/elec/figs_pub', help='Output dir for ELEC figures (relative to root or absolute)')
    ap.add_argument('--outdir-amp', default='augment/results/real/sweep/amp/figs_pub', help='Output dir for AMP figures (relative to root or absolute)')
    # 样式相关
    ap.add_argument('--theme', default='light', choices=['light', 'dark'])
    ap.add_argument('--context', default='talk', choices=['paper', 'talk', 'notebook', 'poster'])
    ap.add_argument('--font', default='DejaVu Sans')
    ap.add_argument('--base-font-size', type=int, default=12)
    ap.add_argument('--palette', default='default')
    ap.add_argument('--formats', default='svg,png', help='Comma-separated formats, e.g. svg,png,pdf')
    ap.add_argument('--cmap-pdr', default='YlGnBu')
    ap.add_argument('--cmap-ctrl', default='YlOrRd')
    ap.add_argument('--no-annotate', action='store_true')

    args = ap.parse_args()

    root = Path(args.root)
    outdir_elec = Path(args.outdir_elec)
    outdir_amp = Path(args.outdir_amp)
    if not outdir_elec.is_absolute():
        outdir_elec = root / outdir_elec
    if not outdir_amp.is_absolute():
        outdir_amp = root / outdir_amp

    # 应用统一风格
    style.apply_style(theme=args.theme, context=args.context, font=args.font, base_font_size=args.base_font_size)
    _ = style.get_palette(args.palette)  # palette 保留以兼容接口，热力图使用 cmap

    # 字体嵌入策略（SVG 保留文本）
    plt.rcParams['svg.fonttype'] = 'none'

    # 载入数据
    df_elec = load_summary(root, 'elec')
    df_amp = load_summary(root, 'amp')

    # 绘制与保存
    fmts = [s.strip() for s in args.formats.split(',') if s.strip()]
    plot_heatmaps_for_mode(df_elec, 'elec', outdir_elec, fmts, args.cmap_pdr, args.cmap_ctrl, annotate=(not args.no_annotate))
    plot_heatmaps_for_mode(df_amp, 'amp', outdir_amp, fmts, args.cmap_pdr, args.cmap_ctrl, annotate=(not args.no_annotate))

    print('All heatmaps generated.')


if __name__ == '__main__':
    main()