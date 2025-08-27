from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

# Reuse the project's energy model for consistency
from src.metrics.energy_model import EnergyModelConfig
# SoD controller from project utils
from src.utils.sod import SoDController, SoDConfig
# Trust evaluator from project src (optional integration)
from src.trust_evaluator import TrustEvaluator, TrustMetrics, NodeStatus


@dataclass
class Node:
    nid: int
    pos: Tuple[float, float]
    energy: float
    is_alive: bool = True


@dataclass
class LinkModel:
    # Simple log-distance path loss turned into PRR via SNR mapping (heuristic)
    prr_at_1m: float = 0.99
    path_loss_exp: float = 2.0
    noise_factor: float = 0.0  # 0 (clean) .. 1 (noisy)

    def prr(self, d: float) -> float:
        if d <= 0.0:
            return self.prr_at_1m
        base = self.prr_at_1m / (1.0 + (d ** self.path_loss_exp) * 0.01)
        prr = max(0.0, min(1.0, base - self.noise_factor * 0.2))
        return prr


@dataclass
class SimConfig:
    area: Tuple[int, int] = (100, 100)
    n_nodes: int = 50
    bs_pos: Tuple[float, float] = (50.0, 50.0)
    init_energy_j: float = 2.0
    radio_payload_bits: int = 1024
    control_bits: int = 128
    slot_s: float = 0.02  # per-hop service time (transmit + process) in seconds
    max_retries: int = 2
    rng_seed: int = 42
    # SoD configuration
    sod_enabled: bool = False
    sod_mode: str = "adaptive"
    sod_k: float = 1.5
    sod_window: int = 24
    sod_delta_day: float = 0.5
    sod_delta_night: float = 0.2
    # Control overhead scheduling (round-based)
    ctrl_cluster_period: int = 20
    ctrl_routing_period: int = 10
    ctrl_trust_period: int = 30
    ctrl_cluster_bits: int = 192
    ctrl_routing_bits: int = 160
    ctrl_trust_bits: int = 192
    # Trust-aware routing
    trust_enabled: bool = False
    trust_alpha: float = 0.5   # weight for trust vs distance-improvement
    trust_prior_succ: float = 0.8  # prior success rate for unseen links
    # LEACH baseline (optional)
    leach_enabled: bool = False
    leach_p_ch: float = 0.1
    leach_period: int = 20
    # HEED baseline (optional, simplified)
    heed_enabled: bool = False
    heed_period: int = 20
    heed_topk_ratio: float = 0.2  # top-k by residual energy as heads
    # Adversarial attacks (configurable)
    attack_enabled: bool = False
    attack_types: Tuple[str, ...] = tuple()  # e.g., ("blackhole", "grayhole", "sinkhole")
    attack_ratio: float = 0.0  # fraction of nodes compromised
    attack_grayhole_p: float = 0.5  # drop probability for grayhole
    attack_sinkhole_bias: float = 0.5  # additional score bias to attract traffic
    attack_start_round: int = 0
    attack_end_round: int = 10**9
    attack_compromised_ids: Optional[Tuple[int, ...]] = None  # optional fixed set


@dataclass
class Measurements:
    delivered_packets: int = 0
    generated_packets: int = 0
    total_delay_s: float = 0.0
    total_hops: int = 0
    # Control overhead breakdown
    ctrl_hello: int = 0
    ctrl_cluster: int = 0
    ctrl_routing: int = 0
    ctrl_trust: int = 0
    # Data/energy
    data_packets: int = 0
    total_tx_j: float = 0.0
    total_rx_j: float = 0.0
    total_cpu_j: float = 0.0
    # SoD stats
    sod_candidates: int = 0
    sod_sent: int = 0
    # Attack stats
    attack_active_rounds: int = 0
    attack_compromised_count: int = 0
    drop_blackhole: int = 0
    drop_grayhole: int = 0
    sinkhole_choices: int = 0
    # Trust stats
    trust_updates: int = 0
    malicious_detected: int = 0

    def ctrl_total(self) -> int:
        return self.ctrl_hello + self.ctrl_cluster + self.ctrl_routing + self.ctrl_trust

    def pdr(self) -> float:
        return 0.0 if self.generated_packets == 0 else self.delivered_packets / self.generated_packets

    def avg_delay(self) -> float:
        return 0.0 if self.delivered_packets == 0 else self.total_delay_s / self.delivered_packets

    def avg_hops(self) -> float:
        return 0.0 if self.delivered_packets == 0 else self.total_hops / self.delivered_packets


class PacketLevelSimulator:
    def __init__(self, cfg: SimConfig, link: LinkModel | None = None):
        self.cfg = cfg
        self.link = link or LinkModel()
        self.energy = EnergyModelConfig()
        random.seed(cfg.rng_seed)
        self.nodes: Dict[int, Node] = {}
        self.topology: Dict[int, List[int]] = {}
        self.m = Measurements()
        # SoD controllers by node id
        self.sod: Dict[int, SoDController] = {}
        # Link trust stats: (u,v) -> [successes, attempts]
        self.link_stats: Dict[Tuple[int, int], List[int]] = {}
        # LEACH heads per period
        self.leach_heads: List[int] = []
        # HEED heads per period
        self.heed_heads: List[int] = []
        # Adversary state
        self._compromised: Set[int] = set()
        self._blackhole: Set[int] = set()
        self._grayhole: Set[int] = set()
        self._sinkhole: Set[int] = set()
        self._attack_initialized: bool = False
        # Trust evaluator (optional)
        self.trust_evaluator: Optional[TrustEvaluator] = None
        # Per-round communication counters for trust metrics
        self._round_attempts: Dict[int, int] = {}
        self._round_successes: Dict[int, int] = {}

    def _attack_active(self, round_idx: int) -> bool:
        if not self.cfg.attack_enabled:
            return False
        return self.cfg.attack_start_round <= round_idx <= self.cfg.attack_end_round

    def _attack_init_if_needed(self) -> None:
        if self._attack_initialized:
            return
        self._attack_initialized = True
        if not self.cfg.attack_enabled:
            return
        # choose compromised nodes
        cand_ids = list(range(self.cfg.n_nodes))
        if self.cfg.attack_compromised_ids is not None and len(self.cfg.attack_compromised_ids) > 0:
            chosen = list(self.cfg.attack_compromised_ids)
        else:
            k = max(0, min(len(cand_ids), int(round(self.cfg.attack_ratio * self.cfg.n_nodes))))
            random.shuffle(cand_ids)
            chosen = cand_ids[:k]
        self._compromised = set(chosen)
        # split types
        tset = set(self.cfg.attack_types)
        for nid in self._compromised:
            # distribute types uniformly across compromised set
            # if multiple types, assign by position modulo
            if not tset:
                break
        if self._compromised:
            types = list(self.cfg.attack_types)
            if not types:
                types = []
            for i, nid in enumerate(sorted(self._compromised)):
                if not types:
                    break
                t = types[i % len(types)]
                if t == "blackhole":
                    self._blackhole.add(nid)
                elif t == "grayhole":
                    self._grayhole.add(nid)
                elif t == "sinkhole":
                    self._sinkhole.add(nid)
        self.m.attack_compromised_count = len(self._compromised)

    def _dist(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def init_random_topology(self, comm_range: float = 30.0) -> None:
        # Place nodes uniformly at random
        self.nodes = {
            i: Node(i, (random.uniform(0, self.cfg.area[0]), random.uniform(0, self.cfg.area[1])), self.cfg.init_energy_j)
            for i in range(self.cfg.n_nodes)
        }
        # Store communication range for routing and BS delivery checks
        self.comm_range = comm_range
        # Build symmetric neighbor lists
        self.topology = {i: [] for i in self.nodes}
        for i in self.nodes:
            for j in self.nodes:
                if i == j:
                    continue
                if self._dist(self.nodes[i].pos, self.nodes[j].pos) <= comm_range:
                    self.topology[i].append(j)
        # Initialize trust evaluator if enabled
        if self.cfg.trust_enabled:
            self.trust_evaluator = TrustEvaluator(alpha=self.cfg.trust_alpha)
            self.trust_evaluator.initialize_trust(list(self.nodes.keys()), initial_trust=0.5)

    def _consume_tx_rx(self, src: int, dst: Optional[int], bits: int, d: float) -> None:
        tx = self.energy.radio_tx_energy(bits, d)
        self.nodes[src].energy -= tx
        self.m.total_tx_j += tx
        if dst is not None:
            rx = self.energy.radio_rx_energy(bits)
            self.nodes[dst].energy -= rx
            self.m.total_rx_j += rx
        cpu = self.energy.cpu_energy(self.cfg.slot_s * 0.2)  # small CPU cost
        self.nodes[src].energy -= cpu
        self.m.total_cpu_j += cpu

    def _link_key(self, u: int, v: int) -> Tuple[int, int]:
        return (u, v)

    def _trust_value(self, u: int, v: int) -> float:
        stats = self.link_stats.get(self._link_key(u, v))
        if not stats:
            return float(self.cfg.trust_prior_succ)
        succ, att = stats
        return (succ + self.cfg.trust_prior_succ) / (att + 1.0)

    def single_path_to_bs(self, src: int) -> List[int]:
        # Greedy: forward to best neighbor by distance improvement and optional trust
        path = [src]
        current = src
        bs_pos = self.cfg.bs_pos
        visited = set([src])
        while True:
            nbrs = self.topology.get(current, [])
            if not nbrs:
                return path
            cur_d = self._dist(self.nodes[current].pos, bs_pos)
            best = None
            best_score = -1e18
            for n in nbrs:
                if n in visited:
                    continue
                # Skip malicious nodes if trust evaluator marks them
                if self.cfg.trust_enabled and self.trust_evaluator is not None:
                    status = self.trust_evaluator.node_status.get(n)
                    if status == NodeStatus.MALICIOUS:
                        continue
                d_n = self._dist(self.nodes[n].pos, bs_pos)
                improvement = max(0.0, cur_d - d_n)
                if improvement <= 0.0:
                    continue
                # normalize improvement by comm_range
                impr_norm = improvement / max(1e-9, getattr(self, 'comm_range', cur_d))
                if self.cfg.trust_enabled:
                    t = self._trust_value(current, n)
                    score = (1.0 - self.cfg.trust_alpha) * impr_norm + self.cfg.trust_alpha * t
                    # If node is suspicious, downweight
                    if self.trust_evaluator is not None:
                        status = self.trust_evaluator.node_status.get(n)
                        if status == NodeStatus.SUSPICIOUS:
                            score *= 0.5
                else:
                    score = impr_norm
                # Adversarial sinkhole: bias selection towards sinkhole nodes when attack active
                if self._sinkhole and self._attack_active_round_cached and (n in self._sinkhole):
                    score += max(0.0, self.cfg.attack_sinkhole_bias)
                if score > best_score:
                    best_score = score
                    best = n
            if best is None:
                return path
            # record sinkhole attraction
            if self._sinkhole and self._attack_active_round_cached and (best in self._sinkhole):
                self.m.sinkhole_choices += 1
            path.append(best)
            visited.add(best)
            current = best
            # if next hop is within comm range to BS, we consider next hop to BS
            if self._dist(self.nodes[current].pos, bs_pos) <= getattr(self, 'comm_range', 1.0):
                return path

    def send_one_packet(self, src: int) -> None:
        self.m.generated_packets += 1
        path = self.single_path_to_bs(src)
        if len(path) <= 1:
            # cannot progress; drop as routing failure (counts as control overhead only if we tried)
            return
        delay = 0.0
        # traverse path; final hop to BS is abstracted as delivery if last hop within 1m to BS
        for i in range(len(path)):
            node = path[i]
            # if last node is already within 1m to BS, deliver
            if i == len(path) - 1:
                d_bs = self._dist(self.nodes[node].pos, self.cfg.bs_pos)
                if d_bs <= self.comm_range:
                    # TX to BS
                    self._consume_tx_rx(node, None, self.cfg.radio_payload_bits, d_bs)
                    delay += self.cfg.slot_s
                    self.m.delivered_packets += 1
                    self.m.data_packets += 1
                    # finalize metrics for a delivered packet
                    self.m.total_delay_s += delay
                    # hops counted as path edges
                    self.m.total_hops += max(0, len(path) - 1)
                return
            src = path[i]
            dst = path[i+1]
            d = self._dist(self.nodes[src].pos, self.nodes[dst].pos)
            # retry up to max_retries
            success = False
            for _ in range(self.cfg.max_retries + 1):
                self._consume_tx_rx(src, dst, self.cfg.radio_payload_bits, d)
                # update attempts (per-link and per-node for trust metrics)
                key = self._link_key(src, dst)
                stats = self.link_stats.get(key)
                if not stats:
                    stats = [0, 0]
                    self.link_stats[key] = stats
                stats[1] += 1
                # per-node attempts
                self._round_attempts[src] = self._round_attempts.get(src, 0) + 1

                if random.random() <= self.link.prr(d):
                    success = True
                    stats[0] += 1  # success
                    # per-node successes
                    self._round_successes[src] = self._round_successes.get(src, 0) + 1
                    delay += self.cfg.slot_s
                    self.m.total_hops += 1
                    break
                else:
                    delay += self.cfg.slot_s  # retry costs time
            if not success:
                return  # packet dropped
            # Adversarial drop at compromised next-hop
            if self._attack_active_round_cached and (dst in self._blackhole or dst in self._grayhole):
                # grayhole drops with probability, blackhole always drops
                if (dst in self._blackhole) or (dst in self._grayhole and random.random() < max(0.0, min(1.0, self.cfg.attack_grayhole_p))):
                    if dst in self._blackhole:
                        self.m.drop_blackhole += 1
                    else:
                        self.m.drop_grayhole += 1
                    return
        # should not reach here

    def _control_broadcast(self, bits: int, count_field: str) -> None:
        # Generic local broadcast to all neighbors
        for i in self.nodes:
            if not self.nodes[i].is_alive or self.nodes[i].energy <= 0:
                continue
            nbrs = self.topology.get(i, [])
            if nbrs:
                avg_d = sum(self._dist(self.nodes[i].pos, self.nodes[j].pos) for j in nbrs) / len(nbrs)
            else:
                avg_d = 1.0
            tx = self.energy.radio_tx_energy(bits, avg_d)
            self.nodes[i].energy -= tx
            self.m.total_tx_j += tx
            cpu = self.energy.cpu_energy(self.cfg.slot_s * 0.05)
            self.nodes[i].energy -= cpu
            self.m.total_cpu_j += cpu
            for j in nbrs:
                if not self.nodes[j].is_alive or self.nodes[j].energy <= 0:
                    continue
                rx = self.energy.radio_rx_energy(bits)
                self.nodes[j].energy -= rx
                self.m.total_rx_j += rx
                # increment designated counter
                setattr(self.m, count_field, getattr(self.m, count_field) + 1)

    def _control_hello_overhead(self) -> None:
        self._control_broadcast(self.cfg.control_bits, "ctrl_hello")

    def _control_cluster_overhead(self) -> None:
        self._control_broadcast(self.cfg.ctrl_cluster_bits, "ctrl_cluster")

    def _control_routing_overhead(self) -> None:
        self._control_broadcast(self.cfg.ctrl_routing_bits, "ctrl_routing")

    def _control_trust_overhead(self) -> None:
        self._control_broadcast(self.cfg.ctrl_trust_bits, "ctrl_trust")

    def _sod_should_send(self, nid: int, round_idx: int) -> bool:
        # simple synthetic temperature signal to test SoD triggering
        x, y = self.nodes[nid].pos
        temp = 20 + 5*math.sin(0.1*round_idx) + 0.05*x + 0.03*y
        # pseudo hour mapping (every 10 rounds ~ 1 hour)
        hour = (round_idx // 10) % 24
        ctrl = self.sod.get(nid)
        if ctrl is None:
            cfg = SoDConfig(
                mode=self.cfg.sod_mode,
                delta_day=self.cfg.sod_delta_day,
                delta_night=self.cfg.sod_delta_night,
                k=self.cfg.sod_k,
                window=self.cfg.sod_window,
            )
            ctrl = SoDController(cfg)
            self.sod[nid] = ctrl
        should_send, _used_delta = ctrl.update_and_should_send(temp, hour)
        self.m.sod_candidates += 1
        if should_send:
            self.m.sod_sent += 1
        return should_send

    def _update_trust_end_of_round(self, round_idx: int) -> None:
        if not self.cfg.trust_enabled or self.trust_evaluator is None:
            return
        # Update each node's trust based on per-round comm performance and residual energy
        for nid in self.nodes:
            attempts = self._round_attempts.get(nid, 0)
            successes = self._round_successes.get(nid, 0)
            comm_rel = 0.0 if attempts == 0 else successes / max(1, attempts)
            pdr = comm_rel
            energy_eff = max(0.0, min(1.0, self.nodes[nid].energy / max(1e-9, self.cfg.init_energy_j)))
            # Approximate neighbor recommendation as average current link trust to neighbors
            nbrs = self.topology.get(nid, [])
            if nbrs:
                rec = sum(self._trust_value(nid, j) for j in nbrs) / len(nbrs)
            else:
                rec = 0.7
            # Use comm_rel as proxy for data consistency; slot time as response time
            metrics = TrustMetrics(
                data_consistency=float(comm_rel),
                communication_reliability=float(comm_rel),
                packet_delivery_ratio=float(pdr),
                response_time=float(self.cfg.slot_s),
                energy_efficiency=float(energy_eff),
                neighbor_recommendations=float(rec),
            )
            # We do not maintain real neighbor_data time series here
            self.trust_evaluator.update_trust(nid, metrics, neighbor_data={}, timestamp=float(round_idx))
        # Expose summary stats to measurements
        self.m.trust_updates = self.trust_evaluator.evaluation_stats.get('trust_updates', 0)
        self.m.malicious_detected = self.trust_evaluator.evaluation_stats.get('malicious_detected', 0)

    def run(self, rounds: int = 100, gen_rate: float = 0.2) -> Measurements:
        # gen_rate: probability each node generates one packet per round
        for r in range(rounds):
            # init attack once and cache active flag for this round
            self._attack_init_if_needed()
            self._attack_active_round_cached = self._attack_active(r)
            if self._attack_active_round_cached:
                self.m.attack_active_rounds += 1
            # scheduled control overheads
            self._control_hello_overhead()
            if r % max(1, self.cfg.ctrl_cluster_period) == 0:
                self._control_cluster_overhead()
            if r % max(1, self.cfg.ctrl_routing_period) == 0:
                self._control_routing_overhead()
            if r % max(1, self.cfg.ctrl_trust_period) == 0:
                self._control_trust_overhead()

            # LEACH baseline: elect heads periodically
            if self.cfg.leach_enabled and (r % max(1, self.cfg.leach_period) == 0):
                self.leach_heads = [nid for nid in self.nodes if self.nodes[nid].is_alive and random.random() < self.cfg.leach_p_ch]
                if not self.leach_heads:
                    alive = [nid for nid in self.nodes if self.nodes[nid].is_alive]
                    if alive:
                        self.leach_heads = [random.choice(alive)]
            # HEED baseline: energy-based head selection periodically
            if self.cfg.heed_enabled and (r % max(1, self.cfg.heed_period) == 0):
                alive = [nid for nid in self.nodes if self.nodes[nid].is_alive]
                if alive:
                    k = max(1, int(len(alive) * max(0.0, min(1.0, self.cfg.heed_topk_ratio))))
                    ranked = sorted(alive, key=lambda nid: self.nodes[nid].energy, reverse=True)
                    self.heed_heads = ranked[:k]
                else:
                    self.heed_heads = []

            # reset per-round comm counters for trust update
            self._round_attempts = {nid: 0 for nid in self.nodes}
            self._round_successes = {nid: 0 for nid in self.nodes}

            for nid in self.nodes:
                if not self.nodes[nid].is_alive or self.nodes[nid].energy <= 0.0:
                    self.nodes[nid].is_alive = False
                    continue

                # If LEACH/HEED baseline enabled: non-CH send to nearest CH; CH then to BS for this packet
                if (self.cfg.leach_enabled and self.leach_heads) or (self.cfg.heed_enabled and self.heed_heads):
                    heads = self.leach_heads if self.cfg.leach_enabled else self.heed_heads
                    if nid not in heads:
                        # Non-CH: with Bernoulli gen_rate, send to nearest CH, then CH→BS for this packet
                        if random.random() < gen_rate:
                            ch = min(heads, key=lambda h: self._dist(self.nodes[nid].pos, self.nodes[h].pos))
                            self.m.generated_packets += 1
                            delay = 0.0
                            # hop1: node -> CH
                            src, dst = nid, ch
                            d1 = self._dist(self.nodes[src].pos, self.nodes[dst].pos)
                            hop1_ok = False
                            for _ in range(self.cfg.max_retries + 1):
                                self._consume_tx_rx(src, dst, self.cfg.radio_payload_bits, d1)
                                if random.random() <= self.link.prr(d1):
                                    hop1_ok = True
                                    delay += self.cfg.slot_s
                                    break
                                else:
                                    delay += self.cfg.slot_s
                            if not hop1_ok:
                                continue
                            # hop2: CH -> BS
                            d2 = self._dist(self.nodes[ch].pos, self.cfg.bs_pos)
                            hop2_ok = False
                            for _ in range(self.cfg.max_retries + 1):
                                self._consume_tx_rx(ch, None, self.cfg.radio_payload_bits, d2)
                                if random.random() <= self.link.prr(d2):
                                    hop2_ok = True
                                    delay += self.cfg.slot_s
                                    break
                                else:
                                    delay += self.cfg.slot_s
                            if hop2_ok:
                                self.m.delivered_packets += 1
                                self.m.data_packets += 1
                                self.m.total_delay_s += delay
                                self.m.total_hops += 2
                    else:
                        # CH: do not originate extra data in baseline; forwarding handled per packet above
                        pass
                    # continue to next node
                    continue

                # SoD gating if enabled, otherwise Bernoulli by gen_rate
                if self.cfg.sod_enabled:
                    if self._sod_should_send(nid, r):
                        self.send_one_packet(nid)
                else:
                    if random.random() < gen_rate:
                        self.send_one_packet(nid)

            # After all transmissions in the round, update trust
            self._update_trust_end_of_round(r)
        return self.m

