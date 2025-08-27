import sys
from pathlib import Path
# Ensure project root (WSN-Intel-Lab-Project) is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import csv
from pathlib import Path

from augment.simulation.packet_sim import PacketLevelSimulator, SimConfig, LinkModel


def main():
    parser = argparse.ArgumentParser(description="Run packet-level WSN simulation (augment workspace)")
    parser.add_argument("--nodes", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--area", type=int, nargs=2, default=[100, 100])
    parser.add_argument("--bs", type=float, nargs=2, default=[50.0, 50.0])
    parser.add_argument("--comm_range", type=float, default=30.0)
    parser.add_argument("--payload", type=int, default=1024)
    parser.add_argument("--gen_rate", type=float, default=0.2)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    # Attack configuration
    parser.add_argument("--attack-enabled", action="store_true")
    parser.add_argument("--attack-types", nargs="+", default=[], help="List of attack types, e.g., blackhole grayhole sinkhole")
    parser.add_argument("--attack-ratio", type=float, default=0.0)
    parser.add_argument("--attack-grayhole-p", type=float, default=0.5)
    parser.add_argument("--attack-sinkhole-bias", type=float, default=0.5)
    parser.add_argument("--attack-start", type=int, default=0)
    parser.add_argument("--attack-end", type=int, default=10**9)
    parser.add_argument("--attack-ids", type=int, nargs="*", default=None, help="Optional fixed compromised node IDs")

    parser.add_argument("--out", type=str, default="augment/results/packet_metrics.csv")
    args = parser.parse_args()

    cfg = SimConfig(
        area=(args.area[0], args.area[1]),
        n_nodes=args.nodes,
        bs_pos=(args.bs[0], args.bs[1]),
        init_energy_j=2.0,
        radio_payload_bits=args.payload,
        slot_s=0.02,
        max_retries=args.retries,
        rng_seed=args.seed,
        attack_enabled=args.attack_enabled,
        attack_types=tuple(args.attack_types),
        attack_ratio=args.attack_ratio,
        attack_grayhole_p=args.attack_grayhole_p,
        attack_sinkhole_bias=args.attack_sinkhole_bias,
        attack_start_round=args.attack_start,
        attack_end_round=args.attack_end,
        attack_compromised_ids=tuple(args.attack_ids) if args.attack_ids else None,
    )
    link = LinkModel(noise_factor=args.noise)

    sim = PacketLevelSimulator(cfg, link)
    sim.init_random_topology(comm_range=args.comm_range)
    m = sim.run(rounds=args.rounds, gen_rate=args.gen_rate)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["generated", "delivered", "pdr", "avg_delay_s", "avg_hops", "ctrl_hello", "ctrl_cluster", "ctrl_routing", "ctrl_trust", "data_pkts", "tx_j", "rx_j", "cpu_j", "sod_candidates", "sod_sent", "attack_active_rounds", "attack_compromised", "drop_blackhole", "drop_grayhole", "sinkhole_choices"]) 
        w.writerow([
            m.generated_packets,
            m.delivered_packets,
            f"{m.pdr():.6f}",
            f"{m.avg_delay():.6f}",
            f"{m.avg_hops():.6f}",
            m.ctrl_hello,
            m.ctrl_cluster,
            m.ctrl_routing,
            m.ctrl_trust,
            m.data_packets,
            f"{m.total_tx_j:.6e}",
            f"{m.total_rx_j:.6e}",
            f"{m.total_cpu_j:.6e}",
            m.sod_candidates,
            m.sod_sent,
            m.attack_active_rounds,
            m.attack_compromised_count,
            m.drop_blackhole,
            m.drop_grayhole,
            m.sinkhole_choices,
        ])
    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()

