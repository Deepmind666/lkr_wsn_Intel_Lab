import argparse
import csv
from pathlib import Path

import sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from augment.simulation.packet_sim import PacketLevelSimulator, SimConfig, LinkModel


def run_one(n_nodes:int, noise:float, proto:str, seed:int, rounds:int=200, gen_rate:float=0.2):
    cfg = SimConfig(
        area=(100,100), n_nodes=n_nodes, bs_pos=(50.0,50.0), init_energy_j=2.0,
        radio_payload_bits=1024, slot_s=0.02, max_retries=2, rng_seed=seed,
    )
    # Protocol toggles
    if proto == 'leach':
        cfg.leach_enabled = True
        cfg.leach_p_ch = 0.1
        cfg.leach_period = 20
    elif proto == 'heed':
        cfg.heed_enabled = True
        cfg.heed_period = 20
        cfg.heed_topk_ratio = 0.2
    elif proto == 'trusthr':
        cfg.trust_enabled = True
        cfg.trust_alpha = 0.5
        cfg.trust_prior_succ = 0.8
    else:
        # greedy (baseline inside simulator)
        pass
    link = LinkModel(noise_factor=noise)
    sim = PacketLevelSimulator(cfg, link)
    sim.init_random_topology(comm_range=30.0)
    m = sim.run(rounds=rounds, gen_rate=gen_rate)
    return m


def summarize(rows):
    import statistics as stats
    def mean_std(vals):
        if not vals: return 0.0, 0.0
        if len(vals) == 1: return vals[0], 0.0
        return stats.mean(vals), stats.pstdev(vals)
    out = {}
    keys = [
        'pdr','delay','hops','tx_j','rx_j','cpu_j',
        'ctrl_hello','ctrl_cluster','ctrl_routing','ctrl_trust'
    ]
    for k in keys:
        vals = [r[k] for r in rows]
        mu, sd = mean_std(vals)
        out[f'{k}_mean'] = mu
        out[f'{k}_std'] = sd
    return out


def collect_metrics(m):
    return {
        'pdr': m.pdr(),
        'delay': m.avg_delay(),
        'hops': m.avg_hops(),
        'tx_j': m.total_tx_j,
        'rx_j': m.total_rx_j,
        'cpu_j': m.total_cpu_j,
        'ctrl_hello': m.ctrl_hello,
        'ctrl_cluster': m.ctrl_cluster,
        'ctrl_routing': m.ctrl_routing,
        'ctrl_trust': m.ctrl_trust,
    }


def main():
    ap = argparse.ArgumentParser(description='Run baselines vs TrustHR across scales/noise with multi-seed')
    ap.add_argument('--seeds', nargs='*', type=int, default=[1,2,3,4,5])
    ap.add_argument('--nodes', nargs='*', type=int, default=[50,100,200])
    ap.add_argument('--noises', nargs='*', type=float, default=[0.0, 0.2])
    ap.add_argument('--rounds', type=int, default=200)
    ap.add_argument('--outdir', type=str, default='augment/results/baselines')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    protos = ['greedy','leach','heed','trusthr']
    summary_rows = []

    for n in args.nodes:
        for z in args.noises:
            for proto in protos:
                rows = []
                for s in args.seeds:
                    m = run_one(n_nodes=n, noise=z, proto=proto, seed=s, rounds=args.rounds)
                    rows.append(collect_metrics(m))
                summ = summarize(rows)
                summ.update({'nodes': n, 'noise': z, 'proto': proto})
                summary_rows.append(summ)
                # write per-scenario CSV
                scenedir = outdir / f'{proto}_n{n}_z{z}'
                scenedir.mkdir(parents=True, exist_ok=True)
                # write per-seed csv
                import csv as _csv
                with open(scenedir/'per_seed.csv', 'w', newline='', encoding='utf-8') as pf:
                    pw = _csv.writer(pf)
                    pw.writerow(['seed','pdr','delay','hops','tx_j','rx_j','cpu_j','ctrl_hello','ctrl_cluster','ctrl_routing','ctrl_trust'])
                    for s, r in zip(args.seeds, rows):
                        pw.writerow([s, r['pdr'], r['delay'], r['hops'], r['tx_j'], r['rx_j'], r['cpu_j'], r['ctrl_hello'], r['ctrl_cluster'], r['ctrl_routing'], r['ctrl_trust']])
                # write summary json
                with open(scenedir/'summary.json', 'w', encoding='utf-8') as f:
                    import json
                    json.dump(summ, f, indent=2)

    # write combined CSV
    summary_csv = outdir / 'summary.csv'
    with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        headers = ['nodes','noise','proto',
                   'pdr_mean','pdr_std','delay_mean','delay_std','hops_mean','hops_std',
                   'tx_j_mean','tx_j_std','rx_j_mean','rx_j_std','cpu_j_mean','cpu_j_std',
                   'ctrl_hello_mean','ctrl_hello_std','ctrl_cluster_mean','ctrl_cluster_std',
                   'ctrl_routing_mean','ctrl_routing_std','ctrl_trust_mean','ctrl_trust_std']
        w.writerow(headers)
        for r in summary_rows:
            w.writerow([r.get(h,'') for h in headers])

    print(f'Saved baseline summaries to {summary_csv}')


if __name__ == '__main__':
    main()

