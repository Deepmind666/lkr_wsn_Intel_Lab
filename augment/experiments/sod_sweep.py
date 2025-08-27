import argparse
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt
import pandas as pd

import sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from augment.simulation.packet_sim import PacketLevelSimulator, SimConfig, LinkModel


def run_multi(cfg_base: dict, seeds: list[int]):
    metrics = []
    for seed in seeds:
        sc = SimConfig(
            area=tuple(cfg_base.get("area", (100, 100))),
            n_nodes=int(cfg_base.get("nodes", 50)),
            bs_pos=tuple(cfg_base.get("bs", (50.0, 50.0))),
            init_energy_j=float(cfg_base.get("init_energy_j", 2.0)),
            radio_payload_bits=int(cfg_base.get("payload_bits", 1024)),
            slot_s=float(cfg_base.get("slot_s", 0.02)),
            max_retries=int(cfg_base.get("retries", 2)),
            rng_seed=int(seed),
            sod_enabled=bool(cfg_base.get("sod_enabled", False)),
            sod_mode=str(cfg_base.get("sod_mode", "adaptive")),
            sod_k=float(cfg_base.get("sod_k", 1.5)),
            sod_window=int(cfg_base.get("sod_window", 24)),
            sod_delta_day=float(cfg_base.get("sod_delta_day", 0.5)),
            sod_delta_night=float(cfg_base.get("sod_delta_night", 0.2)),
        )
        link = LinkModel(noise_factor=float(cfg_base.get("noise", 0.0)))
        sim = PacketLevelSimulator(sc, link)
        sim.init_random_topology(comm_range=float(cfg_base.get("comm_range", 30.0)))
        m = sim.run(rounds=int(cfg_base.get("rounds", 200)), gen_rate=float(cfg_base.get("gen_rate", 0.2)))
        # collect
        metrics.append({
            "pdr": m.pdr(),
            "delay": m.avg_delay(),
            "hops": m.avg_hops(),
            "tx_j": m.total_tx_j,
            "rx_j": m.total_rx_j,
            "cpu_j": m.total_cpu_j,
            "sod_ratio": (1.0 if m.sod_candidates == 0 else m.sod_sent / m.sod_candidates),
        })
    return metrics


def summarize(rows: list[dict]):
    def agg(key):
        vals = [r[key] for r in rows]
        return mean(vals), (0.0 if len(vals) <= 1 else pstdev(vals))
    keys = ["pdr", "delay", "hops", "tx_j", "rx_j", "cpu_j", "sod_ratio"]
    out = {}
    for k in keys:
        m, s = agg(k)
        out[f"{k}_mean"] = m
        out[f"{k}_std"] = s
    return out


def plot_adaptive(df: pd.DataFrame, outdir: Path):
    # PDR vs k (group by window)
    plt.figure(figsize=(6,4))
    for window, g in df.groupby("sod_window"):
        g_sorted = g.sort_values("sod_k")
        plt.errorbar(g_sorted["sod_k"], g_sorted["pdr_mean"], yerr=g_sorted["pdr_std"], marker="o", capsize=3, label=f"window={window}")
    plt.xlabel("k")
    plt.ylabel("PDR")
    plt.title("Adaptive SoD: PDR vs k")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir/"sod_adaptive_pdr_vs_k.png", dpi=200)

    # Energy vs k (TX+RX)
    plt.figure(figsize=(6,4))
    for window, g in df.groupby("sod_window"):
        g_sorted = g.sort_values("sod_k")
        energy = g_sorted["tx_j_mean"] + g_sorted["rx_j_mean"]
        plt.plot(g_sorted["sod_k"], energy, marker="s", label=f"window={window}")
    plt.xlabel("k")
    plt.ylabel("Mean Energy (J)")
    plt.title("Adaptive SoD: Energy vs k")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir/"sod_adaptive_energy_vs_k.png", dpi=200)


def plot_fixed(df: pd.DataFrame, outdir: Path):
    # PDR vs delta_day (each curve for delta_night)
    plt.figure(figsize=(6,4))
    for dn, g in df.groupby("sod_delta_night"):
        g_sorted = g.sort_values("sod_delta_day")
        plt.errorbar(g_sorted["sod_delta_day"], g_sorted["pdr_mean"], yerr=g_sorted["pdr_std"], marker="o", capsize=3, label=f"night={dn}")
    plt.xlabel("delta_day")
    plt.ylabel("PDR")
    plt.title("Fixed SoD: PDR vs delta_day")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir/"sod_fixed_pdr_vs_delta_day.png", dpi=200)


def main():
    ap = argparse.ArgumentParser(description="Run SoD parameter sweep and plot sensitivity")
    ap.add_argument("--outdir", default="WSN-Intel-Lab-Project/augment/results/sod_sweep")
    ap.add_argument("--seeds", nargs="*", type=int, default=[1,2,3,4,5])
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base = dict(area=(100,100), nodes=50, bs=(50.0,50.0), comm_range=30.0, payload_bits=1024, gen_rate=0.2, retries=2, noise=0.0, rounds=200)

    # Adaptive sweep
    adaptive_rows = []
    for window in [12, 24, 36]:
        for k in [1.0, 1.5, 2.0, 2.5]:
            cfg = dict(base, sod_enabled=True, sod_mode="adaptive", sod_k=k, sod_window=window, sod_delta_day=0.5, sod_delta_night=0.2)
            rows = run_multi(cfg, args.seeds)
            summ = summarize(rows)
            summ.update(dict(sod_mode="adaptive", sod_k=k, sod_window=window))
            adaptive_rows.append(summ)
    df_ad = pd.DataFrame(adaptive_rows)
    df_ad.to_csv(outdir/"adaptive_summary.csv", index=False)
    plot_adaptive(df_ad, outdir)

    # Fixed sweep
    fixed_rows = []
    for dn in [0.1, 0.2, 0.3]:
        for dd in [0.3, 0.5, 0.7]:
            cfg = dict(base, sod_enabled=True, sod_mode="fixed", sod_k=1.5, sod_window=24, sod_delta_day=dd, sod_delta_night=dn)
            rows = run_multi(cfg, args.seeds)
            summ = summarize(rows)
            summ.update(dict(sod_mode="fixed", sod_delta_day=dd, sod_delta_night=dn))
            fixed_rows.append(summ)
    df_fx = pd.DataFrame(fixed_rows)
    df_fx.to_csv(outdir/"fixed_summary.csv", index=False)
    plot_fixed(df_fx, outdir)

    print(f"Saved summaries and figures to {outdir}")


if __name__ == "__main__":
    main()

