import argparse
import json
from pathlib import Path
import os
import csv
import matplotlib
# Headless-safe backend: if MPLBACKEND is not preset, force non-GUI backend before importing pyplot
if not os.environ.get("MPLBACKEND"):
    try:
        matplotlib.use("Agg", force=True)
    except Exception:
        pass
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# 兼容作为模块或脚本运行：优先相对导入，否则回退同目录导入
try:
    from . import style  # type: ignore
except Exception:  # pragma: no cover
    import style  # type: ignore


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _bar_with_error(ax, labels, means, stds, colors, ylabel, title,
                    annotate=True, y_is_percent=False):
    xs = range(len(labels))
    bars = ax.bar(xs, means, yerr=stds, color=colors[: len(labels)],
                  edgecolor="#333333", linewidth=1.2, capsize=6)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=8)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.8)

    if y_is_percent:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))
        ax.set_ylim(0, min(1.0, max(0.05, max(means) * 1.2)))
    else:
        # 留一点顶部空间便于标签显示
        ymax = max(means) + (max(stds) if stds else 0)
        ax.set_ylim(0, ymax * 1.25 if ymax > 0 else 1)

    if annotate:
        if y_is_percent:
            fmt = "{:.1%}"
        else:
            # 自适应小数位
            max_val = max(means) if means else 0
            fmt = "{:.3g}" if max_val < 10 else "{:.2f}"
        style.annotate_bars(ax, bars, fmt=fmt)

    style.beautify_axes(ax)
    return bars


def main():
    ap = argparse.ArgumentParser(description="Compare trust ablation/weights with publication-grade figures")
    ap.add_argument("--off", required=True, help="Path to trust_off_agg.json")
    # 将原来的强制 a05/a08 改为可选，同时增加 a02/a035/a065 以支持细粒度 sweep
    ap.add_argument("--a02", required=False, help="Path to trust_on_alpha02_agg.json")
    ap.add_argument("--a035", required=False, help="Path to trust_on_alpha035_agg.json")
    ap.add_argument("--a05", required=False, help="Path to trust_on_alpha05_agg.json")
    ap.add_argument("--a065", required=False, help="Path to trust_on_alpha065_agg.json")
    ap.add_argument("--a08", required=False, help="Path to trust_on_alpha08_agg.json")
    ap.add_argument("--outdir", default="augment/results/figures", help="Output directory for figures")

    # 高级外观配置
    ap.add_argument("--theme", default="light", choices=["light", "dark"], help="Figure theme")
    ap.add_argument("--context", default="talk", choices=["paper", "talk", "notebook", "poster"], help="Seaborn context for scaling")
    ap.add_argument("--font", default="DejaVu Sans", help="Font family")
    ap.add_argument("--base-font-size", type=int, default=12, help="Base font size")
    ap.add_argument("--palette", default="default", help="Named palette in style module")
    ap.add_argument("--formats", default="svg,png", help="Comma-separated formats to save, e.g. svg,png")
    ap.add_argument("--dpi", type=int, default=300, help="DPI for raster outputs like PNG")
    ap.add_argument("--no-annotate", action="store_true", help="Disable value annotations above bars")
    ap.add_argument("--pdr-as-percent", action="store_true", help="Format PDR axis and labels as percent (0-1 -> 0%-100%)")
    ap.add_argument("--export-csv", default=None, help="Optional CSV path to export summary metrics")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 应用统一风格
    style.apply_style(theme=args.theme, context=args.context, font=args.font, base_font_size=args.base_font_size)
    colors = style.get_palette(args.palette)

    # 动态收集不同 alpha 的数据，保持向后兼容：off 总是存在，其他按提供顺序附加
    series = [("off", load(args.off))]
    if args.a02:
        series.append(("alpha=0.2", load(args.a02)))
    if args.a035:
        series.append(("alpha=0.35", load(args.a035)))
    if args.a05:
        series.append(("alpha=0.5", load(args.a05)))
    if args.a065:
        series.append(("alpha=0.65", load(args.a065)))
    if args.a08:
        series.append(("alpha=0.8", load(args.a08)))

    labels = [lab for lab, _ in series]
    data_list = [d for _, d in series]

    # 确保颜色数量足够
    if len(colors) < len(labels):
        repeat = (len(labels) + len(colors) - 1) // len(colors)
        colors = colors * repeat

    # 单图保存帮助
    def save_current(basename: str):
        style.savefig_multi(plt.gcf(), outdir / basename, formats=args.formats.split(","), dpi=args.dpi)

    # 辅助函数获取指标列表
    def mlist(key: str):
        return [d.get(key, 0) for d in data_list]

    # PDR
    plt.figure(figsize=(7.5, 5.0))
    _bar_with_error(
        plt.gca(),
        labels,
        mlist("pdr_mean"),
        mlist("pdr_std"),
        colors,
        ylabel="PDR" + (" (%)" if args.pdr_as_percent else ""),
        title="Trust Ablation: Packet Delivery Ratio",
        annotate=(not args.no_annotate),
        y_is_percent=args.pdr_as_percent,
    )
    save_current("trust_pdr")
    plt.close()

    # Delay
    plt.figure(figsize=(7.5, 5.0))
    _bar_with_error(
        plt.gca(),
        labels,
        mlist("delay_mean"),
        mlist("delay_std"),
        colors,
        ylabel="Avg E2E Delay (s)",
        title="Trust Ablation: End-to-End Delay",
        annotate=(not args.no_annotate),
        y_is_percent=False,
    )
    save_current("trust_delay")
    plt.close()

    # Hops
    plt.figure(figsize=(7.5, 5.0))
    _bar_with_error(
        plt.gca(),
        labels,
        mlist("hops_mean"),
        mlist("hops_std"),
        colors,
        ylabel="Avg Hops",
        title="Trust Ablation: Hops",
        annotate=(not args.no_annotate),
        y_is_percent=False,
    )
    save_current("trust_hops")
    plt.close()

    # Energy (TX+RX)
    plt.figure(figsize=(7.5, 5.0))
    energy_means = [d.get("tx_j_mean", 0) + d.get("rx_j_mean", 0) for d in data_list]
    # 若能量标准差不可得，则以 tx/rx 方差简单合成（近似），否则用 0
    energy_stds = [d.get("tx_j_std", 0) + d.get("rx_j_std", 0) for d in data_list]
    _bar_with_error(
        plt.gca(),
        labels,
        energy_means,
        energy_stds,
        colors,
        ylabel="Mean Energy (J)",
        title="Trust Ablation: Energy (TX + RX)",
        annotate=(not args.no_annotate),
        y_is_percent=False,
    )
    save_current("trust_energy")
    plt.close()

    # 2x2 总览图
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    _bar_with_error(
        axes[0, 0],
        labels,
        mlist("pdr_mean"),
        mlist("pdr_std"),
        colors,
        ylabel="PDR" + (" (%)" if args.pdr_as_percent else ""),
        title="PDR",
        annotate=(not args.no_annotate),
        y_is_percent=args.pdr_as_percent,
    )
    _bar_with_error(
        axes[0, 1],
        labels,
        mlist("delay_mean"),
        mlist("delay_std"),
        colors,
        ylabel="Avg E2E Delay (s)",
        title="Delay",
        annotate=(not args.no_annotate),
        y_is_percent=False,
    )
    _bar_with_error(
        axes[1, 0],
        labels,
        mlist("hops_mean"),
        mlist("hops_std"),
        colors,
        ylabel="Avg Hops",
        title="Hops",
        annotate=(not args.no_annotate),
        y_is_percent=False,
    )
    _bar_with_error(
        axes[1, 1],
        labels,
        energy_means,
        energy_stds,
        colors,
        ylabel="Mean Energy (J)",
        title="Energy (TX + RX)",
        annotate=(not args.no_annotate),
        y_is_percent=False,
    )
    fig.suptitle("Trust Ablation Summary", x=0.02, ha="left", fontsize=style.scaled_size(1.4))
    style.savefig_multi(fig, outdir / "trust_summary", formats=args.formats.split(","), dpi=args.dpi)
    plt.close(fig)

    # 可选：导出 CSV 汇总
    if args.export_csv:
        csv_path = Path(args.export_csv)
        if not csv_path.is_absolute():
            csv_path = outdir / csv_path
        header = [
            "label",
            "pdr_mean", "pdr_std",
            "delay_mean", "delay_std",
            "hops_mean", "hops_std",
            "tx_j_mean", "tx_j_std",
            "rx_j_mean", "rx_j_std",
            "energy_mean", "energy_std",
        ]
        rows = []
        for lab, d in zip(labels, data_list):
            rows.append([
                lab,
                d.get("pdr_mean", 0), d.get("pdr_std", 0),
                d.get("delay_mean", 0), d.get("delay_std", 0),
                d.get("hops_mean", 0), d.get("hops_std", 0),
                d.get("tx_j_mean", 0), d.get("tx_j_std", 0),
                d.get("rx_j_mean", 0), d.get("rx_j_std", 0),
                (d.get("tx_j_mean", 0) + d.get("rx_j_mean", 0)),
                (d.get("tx_j_std", 0) + d.get("rx_j_std", 0)),
            ])
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"Exported CSV summary to {csv_path.resolve()}")

    print(f"Saved trust comparison figures to {outdir.resolve()}")


if __name__ == "__main__":
    main()

