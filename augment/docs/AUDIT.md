# Algorithm & Experiment Audit (Conservative, Evidence-based)

Purpose: identify issues risking overclaiming or non-reproducibility, and define immediate fixes. We will only use measurements produced by augment/* tools in the paper.

Findings (initial)
- Random metrics in enhanced_eehfr_system.py
  - Lines ~591–597: packet_delivery_ratio, average_delay, throughput, average_hop_count, routing_overhead are drawn from random distributions and placeholders. Not suitable for publication.
  - Action: For paper results, DO NOT use these fields. Use augment/simulation/packet_sim.py outputs instead.
- Mixed responsibilities
  - enhanced_eehfr_system.py mixes system logic, metrics, visualization. Risk of hidden couplings and silent changes.
  - Action: keep system for qualitative demos only; quantitative results from augment pipeline.
- Energy model usage
  - src/metrics/energy_model.py is sound and simple; we will reuse it. But parameters must be documented with sources.
  - Action: add a parameter table in the paper and in augment configs.
- Trust & SoD contributions
  - Present but not quantitatively isolated; need ablations for +SoD and +Trust.
  - Action: implement ablation toggles and report deltas using augment runner.

Immediate Decisions
- Publication metrics source: augment simulations only (packet-level PDR, E2E delay, hops, control overhead, energy breakdown). No random placeholders.
- Claims language: conservative; emphasize measurement rigor and reproducibility.

Next Fixes
1) Extend augment simulator to export per-seed CSV and aggregate stats with confidence intervals.
2) Provide YAML configs for scenarios (node counts, densities, noise, retries, seeds, repetitions).
3) Implement ablations: disable SoD / Trust / Multi-hop; vary thresholds.
4) Produce baseline stubs (LEACH/HEED/PEGASIS) in augment for fair comparison (or wrappers if provided).

Notes
- We will not modify core files outside augment/; any integration will be through adapters.

