"""
Publication-grade plotting style helpers for the WSN Intel Lab project.
Centralizes Seaborn/Matplotlib styling, palettes, annotations, and multi-format saving.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import os
import matplotlib as mpl
# Headless-safe backend: ensure non-GUI backend before importing pyplot
if not os.environ.get("MPLBACKEND"):
    try:
        mpl.use("Agg", force=True)
    except Exception:
        pass
import matplotlib.pyplot as plt
import seaborn as sns


# ---------- Palettes ----------
_DEFAULT_PALETTE = [
    "#4C78A8",  # blue
    "#72B7B2",  # teal
    "#F58518",  # orange
    "#54A24B",  # green
    "#EECA3B",  # yellow
    "#B279A2",  # purple
    "#FF9DA6",  # pink
]

_WSN_PALETTE = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
]


def get_palette(name: str = "default") -> List[str]:
    name = (name or "").lower()
    if name in ("default", "vega"):
        return _DEFAULT_PALETTE
    if name in ("wsn", "classic"):
        return _WSN_PALETTE
    if name in ("colorblind", "cb"):
        return list(sns.color_palette("colorblind"))
    # fall back to seaborn current palette
    return list(sns.color_palette())


# ---------- Theme / Style ----------
_DEF_CONTEXT = {
    "paper": ("paper", 1.0),
    "talk": ("talk", 1.0),
    "notebook": ("notebook", 1.0),
    "poster": ("poster", 1.0),
}


def apply_style(theme: str = "light", context: str = "talk", font: str = "DejaVu Sans", base_font_size: int = 12) -> None:
    theme = (theme or "light").lower()
    context_key = (context or "talk").lower()
    ctx, scale = _DEF_CONTEXT.get(context_key, ("talk", 1.0))

    # Seaborn context controls label/tick scaling; font_scale relative to base
    sns.set_context(ctx, font_scale=max(base_font_size / 12.0, 0.8))

    # Base theme
    if theme == "dark":
        sns.set_theme(style="darkgrid")
        facecolor = "#111111"
        gridcolor = "#333333"
        textcolor = "#EEEEEE"
        spinecolor = "#AAAAAA"
    else:
        sns.set_theme(style="whitegrid")
        facecolor = "#FFFFFF"
        gridcolor = "#DDDDDD"
        textcolor = "#222222"
        spinecolor = "#444444"

    mpl.rcParams.update(
        {
            "figure.facecolor": facecolor,
            "axes.facecolor": facecolor,
            "axes.labelcolor": textcolor,
            "axes.edgecolor": spinecolor,
            "axes.linewidth": 1.0,
            "text.color": textcolor,
            "xtick.color": textcolor,
            "ytick.color": textcolor,
            "grid.color": gridcolor,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "font.family": font,
            "savefig.bbox": "tight",
            "savefig.facecolor": facecolor,
            "savefig.edgecolor": facecolor,
        }
    )


def scaled_size(factor: float = 1.0, base: float | None = None) -> float:
    base_size = base if base is not None else mpl.rcParams.get("font.size", 12.0)
    return float(base_size) * float(factor)


def beautify_axes(ax: plt.Axes) -> None:
    # Clean up spines and enable minor ticks
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="both", which="both", labelsize=scaled_size(0.9))
    try:
        ax.minorticks_on()
    except Exception:
        pass


def annotate_bars(ax: plt.Axes, bars: Iterable[plt.Artist], fmt: str = "{:.2f}", dy: float = 0.01) -> None:
    # Add value labels above bars, handling positive/zero values
    for bar in bars:
        try:
            h = bar.get_height()
            if h is None:
                continue
            x = bar.get_x() + bar.get_width() / 2.0
            y = h
            offset = dy * (ax.get_ylim()[1] - ax.get_ylim()[0])
            ax.text(x, y + offset, fmt.format(h), ha="center", va="bottom", fontsize=scaled_size(0.9))
        except Exception:
            continue


def savefig_multi(fig: plt.Figure, basepath: Path | str, formats: Iterable[str] = ("svg",), dpi: int = 300) -> None:
    base = Path(basepath)
    base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fmt_l = (fmt or "").lower().strip()
        path = base.with_suffix("." + fmt_l)
        if fmt_l in ("svg", "pdf"):
            fig.savefig(path, format=fmt_l)
        else:
            fig.savefig(path, format=fmt_l, dpi=dpi)