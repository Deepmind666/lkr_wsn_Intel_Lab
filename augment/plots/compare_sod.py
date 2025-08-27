import argparse
import json
import glob
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_agg(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sod_ratio(per_seed_glob: str) -> float:
    files = glob.glob(per_seed_glob)
    if not files:
        return 1.0
    total_cand, total_sent = 0, 0
    for p in files:
        df = pd.read_csv(p)
        total_cand += int(df.get("sod_candidates", pd.Series([0])).sum())
        total_sent += int(df.get("sod_sent", pd.Series([0])).sum())
    return 1.0 if total_cand == 0 else total_sent / total_cand


def main():
    ap = argparse.ArgumentParser(description="Compare SoD vs No-SoD aggregates and per-seed CSVs")
    ap.add_argument("--no_sod_agg", required=True)
    ap.add_argument("--sod_agg", required=True)
    ap.add_argument("--no_sod_glob", required=True)
    ap.add_argument("--sod_glob", required=True)
    ap.add_argument("--outdir", default="augment/results/figures")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    agg0 = load_agg(args.no_sod_agg)
    agg1 = load_agg(args.sod_agg)

    # Figure A: PDR/Delay/Hops with error bars
    metrics = [
        ("pdr_mean", "pdr_std", "PDR"),
        ("delay_mean", "delay_std", "Avg E2E Delay (s)"),
        ("hops_mean", "hops_std", "Avg Hops"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    labels = ["No-SoD", "SoD"]
    for ax, (m_mean, m_std, title) in zip(axes, metrics):
        means = [agg0.get(m_mean, 0), agg1.get(m_mean, 0)]
        stds = [agg0.get(m_std, 0), agg1.get(m_std, 0)]
        ax.bar(range(2), means, yerr=stds, color=["#4C78A8", "#72B7B2"], capsize=4)
        ax.set_xticks(range(2), labels)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "sod_vs_nosod_perf.png", dpi=200)

    # Figure B: Energy breakdown (means only)
    e_labels = ["TX (J)", "RX (J)", "CPU (J)"]
    e0 = [agg0.get("tx_j_mean", 0), agg0.get("rx_j_mean", 0), agg0.get("cpu_j_mean", 0)]
    e1 = [agg1.get("tx_j_mean", 0), agg1.get("rx_j_mean", 0), agg1.get("cpu_j_mean", 0)]
    x = range(len(e_labels))
    plt.figure(figsize=(6, 4))
    w = 0.35
    plt.bar([i - w/2 for i in x], e0, width=w, label="No-SoD", color="#4C78A8")
    plt.bar([i + w/2 for i in x], e1, width=w, label="SoD", color="#72B7B2")
    plt.xticks(x, e_labels)
    plt.ylabel("Mean Energy (J per run)")
    plt.title("Energy Breakdown (mean)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "sod_vs_nosod_energy.png", dpi=200)

    # Figure C: SoD trigger ratio
    r0 = 1.0  # baseline implicitly "send all" when generated
    r1 = sod_ratio(args.sod_glob)
    plt.figure(figsize=(4.5, 4))
    plt.bar([0, 1], [r0, r1], color=["#4C78A8", "#72B7B2"])
    plt.xticks([0, 1], labels)
    plt.ylim(0, 1.05)
    plt.ylabel("SoD Trigger Ratio (sent/candidates)")
    plt.title("SoD Trigger Efficiency")
    for i, v in enumerate([r0, r1]):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(outdir / "sod_trigger_ratio.png", dpi=200)

    print(f"Saved comparison figures to {outdir}")


if __name__ == "__main__":
    main()

