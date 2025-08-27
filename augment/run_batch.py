import sys
from pathlib import Path as _Path
# Ensure project root (WSN-Intel-Lab-Project) is importable
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import csv
import json
from pathlib import Path

from augment.simulation.packet_sim import PacketLevelSimulator, SimConfig, LinkModel


def run_one(cfg, seed, out_path):
    # Flatten attack config (support both nested and flat)
    attack = cfg.get("attack", {}) if isinstance(cfg.get("attack", {}), dict) else {}
    attack_enabled = attack.get("enabled", cfg.get("attack_enabled", False))
    attack_types = tuple(attack.get("types", cfg.get("attack_types", [])))
    attack_ratio = float(attack.get("ratio", cfg.get("attack_ratio", 0.0)))
    attack_grayhole_p = float(attack.get("grayhole_p", cfg.get("attack_grayhole_p", 0.5)))
    attack_sinkhole_bias = float(attack.get("sinkhole_bias", cfg.get("attack_sinkhole_bias", 0.5)))
    attack_start_round = int(attack.get("start_round", cfg.get("attack_start_round", 0)))
    attack_end_round = int(attack.get("end_round", cfg.get("attack_end_round", 10**9)))
    compromised_ids = attack.get("compromised_ids", cfg.get("attack_compromised_ids"))
    attack_compromised_ids = tuple(compromised_ids) if compromised_ids else None

    sc = SimConfig(
        area=(cfg["area"][0], cfg["area"][1]),
        n_nodes=cfg["nodes"],
        bs_pos=(cfg["bs"][0], cfg["bs"][1]),
        init_energy_j=2.0,
        radio_payload_bits=cfg["payload_bits"],
        slot_s=0.02,
        max_retries=cfg["retries"],
        rng_seed=seed,
        sod_enabled=cfg.get("sod_enabled", False),
        sod_mode=cfg.get("sod_mode", "adaptive"),
        sod_k=cfg.get("sod_k", 1.5),
        sod_window=cfg.get("sod_window", 24),
        sod_delta_day=cfg.get("sod_delta_day", 0.5),
        sod_delta_night=cfg.get("sod_delta_night", 0.2),
        trust_enabled=cfg.get("trust_enabled", False),
        trust_alpha=cfg.get("trust_alpha", 0.5),
        trust_prior_succ=cfg.get("trust_prior_succ", 0.8),
        # attacks
        attack_enabled=attack_enabled,
        attack_types=attack_types,
        attack_ratio=attack_ratio,
        attack_grayhole_p=attack_grayhole_p,
        attack_sinkhole_bias=attack_sinkhole_bias,
        attack_start_round=attack_start_round,
        attack_end_round=attack_end_round,
        attack_compromised_ids=attack_compromised_ids,
    )
    link = LinkModel(noise_factor=cfg["noise"])
    sim = PacketLevelSimulator(sc, link)
    sim.init_random_topology(comm_range=cfg["comm_range"])
    m = sim.run(rounds=cfg["rounds"], gen_rate=cfg["gen_rate"])

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "seed", "generated", "delivered", "pdr", "avg_delay_s", "avg_hops",
            "ctrl_hello", "ctrl_cluster", "ctrl_routing", "ctrl_trust", "data_pkts",
            "tx_j", "rx_j", "cpu_j", "sod_candidates", "sod_sent",
            "attack_active_rounds", "attack_compromised", "drop_blackhole", "drop_grayhole", "sinkhole_choices",
            "trust_updates", "malicious_detected"
        ])
        w.writerow([
            seed,
            m.generated_packets, m.delivered_packets, f"{m.pdr():.6f}", f"{m.avg_delay():.6f}", f"{m.avg_hops():.6f}",
            m.ctrl_hello, m.ctrl_cluster, m.ctrl_routing, m.ctrl_trust, m.data_packets,
            f"{m.total_tx_j:.6e}", f"{m.total_rx_j:.6e}", f"{m.total_cpu_j:.6e}",
            m.sod_candidates, m.sod_sent,
            m.attack_active_rounds, m.attack_compromised_count, m.drop_blackhole, m.drop_grayhole, m.sinkhole_choices,
            m.trust_updates, m.malicious_detected,
        ])


def aggregate(per_seed_paths, out_json):
    import pandas as pd
    dfs = [pd.read_csv(p) for p in per_seed_paths]
    df = pd.concat(dfs, ignore_index=True)
    agg = {
        "n_runs": int(len(df)),
        "generated_sum": int(df["generated"].sum()),
        "delivered_sum": int(df["delivered"].sum()),
        "pdr_mean": float(df["pdr"].mean()),
        "pdr_std": float(df["pdr"].std(ddof=1)),
        "delay_mean": float(df["avg_delay_s"].mean()),
        "delay_std": float(df["avg_delay_s"].std(ddof=1)),
        "hops_mean": float(df["avg_hops"].mean()),
        "hops_std": float(df["avg_hops"].std(ddof=1)),
        "tx_j_mean": float(df["tx_j"].mean()),
        "rx_j_mean": float(df["rx_j"].mean()),
        "cpu_j_mean": float(df["cpu_j"].mean()),
    }
    # Optional attack stats if present
    for col in ["attack_active_rounds", "attack_compromised", "drop_blackhole", "drop_grayhole", "sinkhole_choices"]:
        if col in df.columns:
            agg[f"{col}_mean"] = float(df[col].mean())
    # Optional trust stats if present
    for col in ["trust_updates", "malicious_detected"]:
        if col in df.columns:
            agg[f"{col}_mean"] = float(df[col].mean())
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)


def append_overview(cfg, agg):
    # derive overview row fields
    attack = cfg.get("attack", {}) if isinstance(cfg.get("attack", {}), dict) else {}
    row = {
        "config": cfg.get("_config_name", "unknown"),
        "nodes": cfg.get("nodes"),
        "rounds": cfg.get("rounds"),
        "comm_range": cfg.get("comm_range"),
        "noise": cfg.get("noise"),
        "trust_enabled": cfg.get("trust_enabled", False),
        "sod_enabled": cfg.get("sod_enabled", False),
        "attack_enabled": attack.get("enabled", cfg.get("attack_enabled", False)),
        "attack_types": ",".join(attack.get("types", cfg.get("attack_types", [])) or []),
        "attack_ratio": attack.get("ratio", cfg.get("attack_ratio", "")),
        "grayhole_p": attack.get("grayhole_p", cfg.get("attack_grayhole_p", "")),
        "sinkhole_bias": attack.get("sinkhole_bias", cfg.get("attack_sinkhole_bias", "")),
        "attack_start": attack.get("start_round", cfg.get("attack_start_round", "")),
        "attack_end": attack.get("end_round", cfg.get("attack_end_round", "")),
        # key aggregates
        "pdr_mean": agg.get("pdr_mean"),
        "delay_mean": agg.get("delay_mean"),
        "hops_mean": agg.get("hops_mean"),
        "tx_j_mean": agg.get("tx_j_mean"),
        "rx_j_mean": agg.get("rx_j_mean"),
        "cpu_j_mean": agg.get("cpu_j_mean"),
        # optional attack/trust
        "attack_active_rounds_mean": agg.get("attack_active_rounds_mean", ""),
        "attack_compromised_mean": agg.get("attack_compromised_mean", ""),
        "drop_blackhole_mean": agg.get("drop_blackhole_mean", ""),
        "drop_grayhole_mean": agg.get("drop_grayhole_mean", ""),
        "sinkhole_choices_mean": agg.get("sinkhole_choices_mean", ""),
        "trust_updates_mean": agg.get("trust_updates_mean", ""),
        "malicious_detected_mean": agg.get("malicious_detected_mean", ""),
    }
    overview_path = cfg.get("overview_csv", "augment/results/overview_master.csv")
    Path(overview_path).parent.mkdir(parents=True, exist_ok=True)
    header = list(row.keys())
    # append with header if new
    if not Path(overview_path).exists():
        with open(overview_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerow([row[k] for k in header])
    else:
        with open(overview_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([row[k] for k in header])


def main():
    parser = argparse.ArgumentParser(description="Run batch simulations from JSON config")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    # add config name for overview
    cfg["_config_name"] = Path(args.config).name

    per_seed_paths = []
    for seed in cfg["seeds"]:
        p = cfg["per_seed_csv"].format(seed=seed)
        run_one(cfg, seed, p)
        per_seed_paths.append(p)

    aggregate(per_seed_paths, cfg["out_json"])
    # read back agg for overview
    with open(cfg["out_json"], "r", encoding="utf-8") as f:
        agg = json.load(f)
    append_overview(cfg, agg)
    print(f"Aggregated results saved to {cfg['out_json']}")


if __name__ == "__main__":
    main()

