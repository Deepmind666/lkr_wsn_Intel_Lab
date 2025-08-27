import argparse
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import math
import numpy as np

import sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from augment.realdata.loader import load_connectivity, load_cleaned_data, load_mote_locs
from src.metrics.energy_model import EnergyModelConfig
from typing import Set
from augment.runlog.logger import log_run


def path_success(p_ij: Dict[Tuple[int,int], float], path: list[int]) -> float:
    s = 1.0
    for u, v in zip(path, path[1:]):
        s *= p_ij.get((u, v), 0.0)
    return s


def greedy_next_hop(p_ij: Dict[Tuple[int,int], float], src: int, neighbors: Dict[int, list[int]]) -> int | None:
    # choose neighbor with highest p_ij
    nbrs = neighbors.get(src, [])
    if not nbrs:
        return None
    best = None
    bestp = -1.0
    for v in nbrs:
        w = p_ij.get((src, v), 0.0)
        if w > bestp:
            bestp = w
            best = v
    return best


def build_path_greedy(p_ij: Dict[Tuple[int,int], float], src: int, bs: int, neighbors: Dict[int, list[int]]) -> list[int]:
    path = [src]
    curr = src
    visited = {src}
    for _ in range(500):
        if curr == bs:
            break
        nx = greedy_next_hop(p_ij, curr, neighbors)
        if nx is None or nx in visited:
            break
        visited.add(nx)
        path.append(nx)
        curr = nx
    return path


def build_path_maxprod(p_ij: Dict[Tuple[int,int], float], src: int, bs: int, neighbors: Dict[int, list[int]]) -> list[int]:
    # Dijkstra on -log(p) weights
    import heapq
    nodes = set(neighbors.keys()) | {bs} | {v for vs in neighbors.values() for v in vs}
    dist = {n: math.inf for n in nodes}
    prev = {}
    dist[src] = 0.0
    h = [(0.0, src)]
    while h:
        d, u = heapq.heappop(h)
        if d > dist[u]:
            continue
        if u == bs:
            break
        for v in neighbors.get(u, []):
            pij = p_ij.get((u, v), 0.0)
            if pij <= 0:
                continue
            w = -math.log(max(1e-12, pij))
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(h, (nd, v))
    # reconstruct
    if bs not in prev and src != bs:
        # no path
        return [src]
    path = [bs]
    cur = bs
    while cur != src:
        cur = prev.get(cur)
        if cur is None:
            return [src]
        path.append(cur)
    path.reverse()
    return path


def select_heads_by_degree(neighbors: Dict[int, list[int]], k: int) -> list[int]:
    if k <= 0:
        return []
    deg = [(n, len(vs)) for n, vs in neighbors.items()]
    deg.sort(key=lambda x: x[1], reverse=True)
    return [n for n, _ in deg[:k]]


def select_heads_by_stride(mote_ids: list[int], ratio: float) -> list[int]:
    ratio = max(1e-6, min(1.0, ratio))
    stride = max(1, int(round(1.0 / ratio)))
    ids = sorted(mote_ids)
    heads = ids[::stride]
    if not heads and ids:
        heads = [ids[0]]
    return heads


def nearest_head(node: int, heads: list[int], locs: Dict[int, Tuple[float,float]]) -> int | None:
    if not heads:
        return None
    def dist_to(h):
        (x1,y1) = locs.get(node, (0.0,0.0)); (x2,y2) = locs.get(h, (0.0,0.0))
        return ((x1-x2)**2 + (y1-y2)**2) ** 0.5
    return min(heads, key=dist_to)


def build_path_leach(p_ij: Dict[Tuple[int,int], float], src: int, bs: int, neighbors: Dict[int, list[int]], locs: Dict[int, Tuple[float,float]], heads: list[int]) -> list[int]:
    if src in heads:
        return build_path_maxprod(p_ij, src, bs, neighbors)
    h = nearest_head(src, heads, locs)
    if h is None:
        return build_path_maxprod(p_ij, src, bs, neighbors)
    p1 = build_path_maxprod(p_ij, src, h, neighbors)
    p2 = build_path_maxprod(p_ij, h, bs, neighbors)
    if len(p1) <= 1:
        return [src]
    if len(p2) <= 1:
        return p1
    return p1 + p2[1:]


def build_path_heed(p_ij: Dict[Tuple[int,int], float], src: int, bs: int, neighbors: Dict[int, list[int]], locs: Dict[int, Tuple[float,float]], heads: list[int]) -> list[int]:
    # Same assignment as LEACH but heads由度数选出
    return build_path_leach(p_ij, src, bs, neighbors, locs, heads)


def build_path_trusthr(p_ij: Dict[Tuple[int,int], float], src: int, bs: int, neighbors: Dict[int, list[int]]) -> list[int]:
    # Max-product path as deterministic trust-aware routing proxy
    return build_path_maxprod(p_ij, src, bs, neighbors)


def build_path_pegasis(p_ij: Dict[Tuple[int,int], float], src: int, bs: int, neighbors: Dict[int, list[int]], locs: Dict[int, Tuple[float,float]], mote_ids: list[int]) -> list[int]:
    # Build a simple nearest-neighbor chain and head as node closest to BS
    # Then route src -> chain_head -> BS via max-product subpaths
    if not mote_ids:
        return [src]
    # choose head: nearest to BS among mote_ids
    def dist_to_bs(n:int):
        (x,y) = locs.get(n, (0.0,0.0)); (bx,by) = locs.get(bs, (0.0,0.0))
        return ((x-bx)**2 + (y-by)**2) ** 0.5
    head = min(mote_ids, key=dist_to_bs)
    # connect src to head via max-product; then head to BS via max-product
    p1 = build_path_maxprod(p_ij, src, head, neighbors)
    p2 = build_path_maxprod(p_ij, head, bs, neighbors)
    if len(p1) <= 1:
        return [src]
    if len(p2) <= 1:
        return p1
    return p1 + p2[1:]


def build_path_etx(p_ij: Dict[Tuple[int,int], float], src: int, bs: int, neighbors: Dict[int, list[int]]) -> list[int]:
    # Shortest path on ETX metric: weight = 1/max(p_ij,eps)
    import heapq
    eps = 1e-6
    nodes = set(neighbors.keys()) | {bs} | {v for vs in neighbors.values() for v in vs}
    dist = {n: math.inf for n in nodes}
    prev = {}
    dist[src] = 0.0
    h = [(0.0, src)]
    while h:
        d, u = heapq.heappop(h)
        if d > dist[u]:
            continue
        if u == bs:
            break
        for v in neighbors.get(u, []):
            pij = p_ij.get((u, v), 0.0)
            if pij <= 0:
                continue
            w = 1.0 / max(eps, pij)
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(h, (nd, v))
    if bs not in prev and src != bs:
        return [src]
    path = [bs]
    cur = bs
    while cur != src:
        cur = prev.get(cur)
        if cur is None:
            return [src]
        path.append(cur)
    path.reverse()
    return path


def build_neighbors_from_connectivity(p_ij: Dict[Tuple[int,int], float]) -> Dict[int, list[int]]:
    nbrs: Dict[int, list[int]] = {}
    for (u, v), w in p_ij.items():
        if w <= 0: continue
        nbrs.setdefault(u, []).append(v)
    return nbrs


def build_reverse_neighbors(nbrs: Dict[int, list[int]]) -> Dict[int, list[int]]:
    r: Dict[int, list[int]] = {}
    for u, vs in nbrs.items():
        for v in vs:
            r.setdefault(v, []).append(u)
    return r


def build_greedy_path(p_ij: Dict[Tuple[int,int], float], src: int, bs: int, neighbors: Dict[int, list[int]]) -> list[int]:
    path = [src]
    curr = src
    visited = {src}
    for _ in range(200):  # guard
        if curr == bs:
            break
        nx = greedy_next_hop(p_ij, curr, neighbors)
        if nx is None or nx in visited:
            break
        visited.add(nx)
        path.append(nx)
        curr = nx
        if curr == bs:
            break
    return path


def _edge_distance(u:int, v:int, locs:Dict[int,Tuple[float,float]])->float:
    if (u not in locs) or (v not in locs):
        return 1.0
    (x1,y1) = locs[u]; (x2,y2) = locs[v]
    return ((x1-x2)**2 + (y1-y2)**2) ** 0.5


def _infer_total_seconds(series):
    try:
        t = pd.to_datetime(series, errors='coerce', utc=True)
        if t.notna().any():
            dt = (t.max() - t.min()).total_seconds()
            if isinstance(dt, (int, float)) and dt > 0:
                return float(dt)
    except Exception:
        pass
    # numeric fallback
    try:
        x = pd.to_numeric(series, errors='coerce')
        rng = float(x.max() - x.min())
        if not pd.isna(rng) and rng > 0:
            # attempt unit normalization via median diff heuristic
            diffs = x.dropna().diff().dropna()
            q = diffs.quantile(0.5) if not diffs.empty else None
            scale = 1.0
            if q is not None:
                if q > 1e6:  # likely microseconds
                    scale = 1e-6
                elif q > 1e3:  # likely milliseconds
                    scale = 1e-3
            return max(1.0, rng * scale)
    except Exception:
        pass
    return float(max(1, int(series.shape[0])))


def _eff_distance(d: float, energy_mode: str) -> float:
    # if energy_mode == 'elec', ignore amplifier by zeroing distance
    return 0.0 if energy_mode == 'elec' else d


def eval_real(connectivity_path: str, cleaned_csv: str, mote_locs_path: str = 'WSN-Intel-Lab-Project/data/mote_locs.txt', bs_id: int = 0, payload_bits:int = 1024, slot_s: float = 0.02, out_prefix: str = 'WSN-Intel-Lab-Project/augment/results/real/real',
              ctrl_bits:int=128, ctrl_cluster_bits:int=192, ctrl_routing_bits:int=160, ctrl_trust_bits:int=192,
              ctrl_cluster_period:int=600, ctrl_routing_period:int=300, ctrl_trust_period:int=900,
              ctrl_hello_per_gen:int=5000, ctrl_cluster_per_gen:int=5000, ctrl_routing_per_gen:int=1000, ctrl_trust_per_gen:int=2000,
              proto:str='greedy', leach_ratio:float=0.1, heed_topk_ratio:float=0.2, energy_mode:str='amp'):
    p_ij = load_connectivity(connectivity_path)
    df = load_cleaned_data(cleaned_csv)
    locs = load_mote_locs(mote_locs_path)
    neighbors = build_neighbors_from_connectivity(p_ij)

    # motes present in data and connectivity
    mote_ids = sorted(set(df['mote_id'].unique().tolist()) & set([i for i,_ in p_ij.keys()]))

    # per-mote generated count from real data
    gen_counts = df.groupby('mote_id').size().to_dict()

    # time steps (deterministic control schedule derives from timeline length, not raw row count)
    # NOTE: 控制面事件不再直接由时间步驱动，转而采用按每个节点的生成数进行比例计数，避免超长时间轴导致控制能耗不成比例地放大
    total_seconds = _infer_total_seconds(df['time'])
    n_steps = max(1, int(math.ceil(total_seconds / slot_s)))

    # 旧的时间步驱动逻辑保留变量但不直接使用，防止外部调用依赖；默认 period 参数被视为“秒”，如需转换可在后续扩展
    # ctrl_hello_period_steps = max(1, int(round(60.0 / slot_s)))
    # cluster_period_steps = max(1, int(round(ctrl_cluster_period / slot_s)))
    # routing_period_steps = max(1, int(round(ctrl_routing_period / slot_s)))
    # trust_period_steps   = max(1, int(round(ctrl_trust_period / slot_s)))

    # 基于每个节点生成数的控制事件计数：events = ceil(gen_count / K)
    def events_by_gen(gen:int, per_gen:int) -> int:
        per_gen = max(1, int(per_gen))
        gen = max(0, int(gen))
        return int(math.ceil(gen / per_gen)) if gen > 0 else 0

    # cluster heads if needed
    heads: list[int] = []
    if proto == 'leach':
        heads = select_heads_by_stride(mote_ids, leach_ratio)
    elif proto == 'heed':
        # approximate HEED: pick top-k degrees
        k = max(1, int(round(len(mote_ids) * heed_topk_ratio)))
        heads = select_heads_by_degree(neighbors, k)

    # ILMR preparation（仅在选择 ILMR 协议时构建一次图与特征）
    if proto == 'ilmr':
        from src.advanced_algorithms.ilmr_algorithm import ILMRAlgorithm
        all_ids = sorted(set(mote_ids) | {bs_id})
        id2idx = {nid: i for i, nid in enumerate(all_ids)}
        N = len(all_ids)
        adj = np.zeros((N, N), dtype=float)
        for (u, v), pij in p_ij.items():
            if pij > 0 and (u in id2idx) and (v in id2idx):
                # 使用几何距离作为边权，零表示不可达
                adj[id2idx[u], id2idx[v]] = _edge_distance(u, v, locs)
        feats = np.zeros((N, 4), dtype=float)  # [x, y, energy, alive]
        for nid, idx in id2idx.items():
            x, y = locs.get(nid, (0.0, 0.0))
            feats[idx, 0] = x
            feats[idx, 1] = y

    energy = EnergyModelConfig()
    rows: list[dict] = []
    for m in mote_ids:
        # routing path selection by protocol
        if proto == 'leach':
            path = build_path_leach(p_ij, m, bs_id, neighbors, locs, heads)
        elif proto == 'heed':
            path = build_path_heed(p_ij, m, bs_id, neighbors, locs, heads)
        elif proto == 'pegasis':
            path = build_path_pegasis(p_ij, m, bs_id, neighbors, locs, mote_ids)
        elif proto == 'etx':
            path = build_path_etx(p_ij, m, bs_id, neighbors)
        elif proto == 'trusthr':
            path = build_path_trusthr(p_ij, m, bs_id, neighbors)
        else:
            path = build_greedy_path(p_ij, m, bs_id, neighbors)
        hops = max(0, len(path) - 1)
        pdr = path_success(p_ij, path) if hops >= 1 else 0.0
        # delay: hops * slot_s
        delay = hops * slot_s
        # data energy per message: sum over edges of TX(u->v, d_uv) + RX at v; CPU small per hop
        tx_per_msg = 0.0; rx_per_msg = 0.0; cpu_per_msg = 0.0
        for u, v in zip(path, path[1:]):
            d = _eff_distance(_edge_distance(u, v, locs), energy_mode)
            tx_per_msg += energy.radio_tx_energy(payload_bits, d)
            rx_per_msg += energy.radio_rx_energy(payload_bits)
            cpu_per_msg += energy.cpu_energy(slot_s * 0.05)
        gen = int(gen_counts.get(m, 0))
        data_tx = tx_per_msg * gen
        data_rx = rx_per_msg * gen
        data_cpu = cpu_per_msg * gen

        # Control overheads (broadcast to neighbors), now derived from per-mote traffic volume
        nbrs = neighbors.get(m, [])
        deg = len(nbrs)
        if deg > 0:
            avg_d = sum(_edge_distance(m, v, locs) for v in nbrs) / deg
        else:
            avg_d = 1.0
        def ctrl_energy(num_events:int, bits:int):
            tx_e = energy.radio_tx_energy(bits, avg_d)
            rx_e = energy.radio_rx_energy(bits)
            cpu_e = energy.cpu_energy(slot_s * 0.05)
            tx_sum = tx_e * num_events
            rx_sum = rx_e * deg * num_events
            cpu_sum = cpu_e * num_events
            rx_count = deg * num_events
            return tx_sum, rx_sum, cpu_sum, rx_count

        # 基于 gen_count 的控制事件计数
        hello_events   = events_by_gen(gen, ctrl_hello_per_gen)
        cluster_events = events_by_gen(gen, ctrl_cluster_per_gen)
        routing_events = events_by_gen(gen, ctrl_routing_per_gen)
        trust_events   = events_by_gen(gen, ctrl_trust_per_gen)

        h_tx,h_rx,h_cpu,h_cnt = ctrl_energy(hello_events,   ctrl_bits)
        c_tx,c_rx,c_cpu,c_cnt = ctrl_energy(cluster_events, ctrl_cluster_bits)
        r_tx,r_rx,r_cpu,r_cnt = ctrl_energy(routing_events, ctrl_routing_bits)
        t_tx,t_rx,t_cpu,t_cnt = ctrl_energy(trust_events,   ctrl_trust_bits)

        rows.append({
            'mote_id': m,
            'path_len': hops,
            'pdr_exp': pdr,
            'delay_s': delay,
            'gen_count': gen,
            'tx_j_data': data_tx,
            'rx_j_data': data_rx,
            'cpu_j_data': data_cpu,
            'ctrl_hello': h_cnt,
            'ctrl_cluster': c_cnt,
            'ctrl_routing': r_cnt,
            'ctrl_trust': t_cnt,
            'ctrl_tx_j': h_tx + c_tx + r_tx + t_tx,
            'ctrl_rx_j': h_rx + c_rx + r_rx + t_rx,
            'ctrl_cpu_j': h_cpu + c_cpu + r_cpu + t_cpu,
            'tx_j_total': data_tx + h_tx + c_tx + r_tx + t_tx,
            'rx_j_total': data_rx + h_rx + c_rx + r_rx + t_rx,
            'cpu_j_total': data_cpu + h_cpu + c_cpu + r_cpu + t_cpu,
        })

    per_mote = pd.DataFrame(rows)
    outdir = Path(out_prefix).parent
    outdir.mkdir(parents=True, exist_ok=True)
    per_mote.to_csv(f"{out_prefix}_per_mote.csv", index=False)

    # aggregate (weighted by gen_count where appropriate)
    total_gen = max(1, per_mote['gen_count'].sum())
    agg = {
        'n_motes': int(len(per_mote)),
        'total_gen': int(total_gen),
        'pdr_mean': float((per_mote['pdr_exp'] * per_mote['gen_count']).sum() / total_gen),
        'delay_mean': float((per_mote['delay_s'] * per_mote['gen_count']).sum() / total_gen),
        'tx_j_sum_data': float(per_mote['tx_j_data'].sum()),
        'rx_j_sum_data': float(per_mote['rx_j_data'].sum()),
        'cpu_j_sum_data': float(per_mote['cpu_j_data'].sum()),
        'ctrl_hello_sum': int(per_mote['ctrl_hello'].sum()),
        'ctrl_cluster_sum': int(per_mote['ctrl_cluster'].sum()),
        'ctrl_routing_sum': int(per_mote['ctrl_routing'].sum()),
        'ctrl_trust_sum': int(per_mote['ctrl_trust'].sum()),
        'ctrl_tx_j_sum': float(per_mote['ctrl_tx_j'].sum()),
        'ctrl_rx_j_sum': float(per_mote['ctrl_rx_j'].sum()),
        'ctrl_cpu_j_sum': float(per_mote['ctrl_cpu_j'].sum()),
        'tx_j_sum_total': float(per_mote['tx_j_total'].sum()),
        'rx_j_sum_total': float(per_mote['rx_j_total'].sum()),
        'cpu_j_sum_total': float(per_mote['cpu_j_total'].sum()),
    }
    import json
    with open(f"{out_prefix}_agg.json", 'w', encoding='utf-8') as f:
        json.dump(agg, f, indent=2)
    # log run
    meta = {
        'proto': proto,
        'energy_mode': energy_mode,
        'pij_scale': '',
        'connectivity': connectivity_path,
        'cleaned': cleaned_csv,
        'mote_locs': mote_locs_path,
        'out_prefix': out_prefix,
        'n_motes': int(len(per_mote)),
        'total_gen': int(total_gen),
        'pdr_mean': agg['pdr_mean'],
        'delay_mean': agg['delay_mean'],
        'tx_j_sum_data': agg['tx_j_sum_data'],
        'rx_j_sum_data': agg['rx_j_sum_data'],
        # Added control parameters for auditability
        'ctrl_bits': int(ctrl_bits),
        'ctrl_cluster_bits': int(ctrl_cluster_bits),
        'ctrl_routing_bits': int(ctrl_routing_bits),
        'ctrl_trust_bits': int(ctrl_trust_bits),
        'ctrl_cluster_period': int(ctrl_cluster_period),
        'ctrl_routing_period': int(ctrl_routing_period),
        'ctrl_trust_period': int(ctrl_trust_period),
        'ctrl_hello_per_gen': int(ctrl_hello_per_gen),
        'ctrl_cluster_per_gen': int(ctrl_cluster_per_gen),
        'ctrl_routing_per_gen': int(ctrl_routing_per_gen),
        'ctrl_trust_per_gen': int(ctrl_trust_per_gen),
    }
    log_run('real_eval', meta)
    print(f"Saved {out_prefix}_per_mote.csv and {out_prefix}_agg.json")


def main():
    ap = argparse.ArgumentParser(description='Evaluate real dataset (connectivity + cleaned_data) with deterministic metrics')
    ap.add_argument('--connectivity', type=str, default='WSN-Intel-Lab-Project/data/connectivity.txt')
    ap.add_argument('--cleaned', type=str, default='WSN-Intel-Lab-Project/data/processed/cleaned_data.csv')
    ap.add_argument('--mote_locs', type=str, default='WSN-Intel-Lab-Project/data/mote_locs.txt')
    ap.add_argument('--bs', type=int, default=0)
    ap.add_argument('--payload', type=int, default=1024)
    ap.add_argument('--slot', type=float, default=0.02)
    ap.add_argument('--proto', type=str, default='greedy', choices=['greedy','leach','heed','trusthr','pegasis','etx','ilmr'])
    ap.add_argument('--leach_ratio', type=float, default=0.1)
    ap.add_argument('--heed_topk_ratio', type=float, default=0.2)
    ap.add_argument('--out_prefix', type=str, default='WSN-Intel-Lab-Project/augment/results/real/real')
    ap.add_argument('--energy_mode', type=str, default='amp', choices=['amp','elec'])
    # Control overhead parameters (CLI)
    ap.add_argument('--ctrl_bits', type=int, default=128)
    ap.add_argument('--ctrl_cluster_bits', type=int, default=192)
    ap.add_argument('--ctrl_routing_bits', type=int, default=160)
    ap.add_argument('--ctrl_trust_bits', type=int, default=192)
    ap.add_argument('--ctrl_cluster_period', type=int, default=600)
    ap.add_argument('--ctrl_routing_period', type=int, default=300)
    ap.add_argument('--ctrl_trust_period', type=int, default=900)
    ap.add_argument('--ctrl_hello_per_gen', type=int, default=5000)
    ap.add_argument('--ctrl_cluster_per_gen', type=int, default=5000)
    ap.add_argument('--ctrl_routing_per_gen', type=int, default=1000)
    ap.add_argument('--ctrl_trust_per_gen', type=int, default=2000)
    args = ap.parse_args()
    eval_real(args.connectivity, args.cleaned, args.mote_locs, args.bs, args.payload, args.slot, args.out_prefix,
              ctrl_bits=args.ctrl_bits, ctrl_cluster_bits=args.ctrl_cluster_bits, ctrl_routing_bits=args.ctrl_routing_bits, ctrl_trust_bits=args.ctrl_trust_bits,
              ctrl_cluster_period=args.ctrl_cluster_period, ctrl_routing_period=args.ctrl_routing_period, ctrl_trust_period=args.ctrl_trust_period,
              ctrl_hello_per_gen=args.ctrl_hello_per_gen, ctrl_cluster_per_gen=args.ctrl_cluster_per_gen, ctrl_routing_per_gen=args.ctrl_routing_per_gen, ctrl_trust_per_gen=args.ctrl_trust_per_gen,
              proto=args.proto, leach_ratio=args.leach_ratio, heed_topk_ratio=args.heed_topk_ratio, energy_mode=args.energy_mode)


if __name__ == '__main__':
    main()

