import argparse
from pathlib import Path

import sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from augment.simulation.packet_sim import PacketLevelSimulator, SimConfig, LinkModel


def main():
    ap = argparse.ArgumentParser(description='Run HEED-like baseline (energy-ranked heads) with multi-seed aggregate')
    ap.add_argument('--nodes', type=int, default=100)
    ap.add_argument('--noise', type=float, default=0.2)
    ap.add_argument('--seeds', nargs='*', type=int, default=[1,2,3,4,5])
    ap.add_argument('--rounds', type=int, default=200)
    ap.add_argument('--out', type=str, default='WSN-Intel-Lab-Project/augment/results/heed_n{nodes}_z{noise}_agg.json')
    args = ap.parse_args()

    import json, statistics as stats
    rows = []
    for s in args.seeds:
        cfg = SimConfig(area=(100,100), n_nodes=args.nodes, bs_pos=(50.0,50.0), rng_seed=s,
                        heed_enabled=True, heed_period=20, heed_topk_ratio=0.2)
        sim = PacketLevelSimulator(cfg, LinkModel(noise_factor=args.noise))
        sim.init_random_topology(comm_range=30.0)
        m = sim.run(rounds=args.rounds, gen_rate=0.2)
        rows.append(dict(pdr=m.pdr(), delay=m.avg_delay(), hops=m.avg_hops(), tx_j=m.total_tx_j, rx_j=m.total_rx_j, cpu_j=m.total_cpu_j))
    def ms(key):
        arr=[r[key] for r in rows]
        return (stats.mean(arr), (0.0 if len(arr)<=1 else stats.pstdev(arr)))
    agg={}
    for k in ['pdr','delay','hops','tx_j','rx_j','cpu_j']:
        mu,sd = ms(k)
        agg[f'{k}_mean']=mu
        agg[f'{k}_std']=sd
    out_path = Path(args.out.format(nodes=args.nodes, noise=args.noise))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(agg, f, indent=2)
    print(f'Saved {out_path}')

if __name__ == '__main__':
    main()

