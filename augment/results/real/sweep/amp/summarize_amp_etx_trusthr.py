import json
import math
import os
from pathlib import Path
import pandas as pd

# Summarize AMP-mode sweeps for ETX and TRUSTHR into a single CSV
# Input files expected:
#   c:/WSN-Intel-Lab-Project/augment/results/real/sweep/amp/etx/etx_amp_h{H}_r{R}_agg.json
#   c:/WSN-Intel-Lab-Project/augment/results/real/sweep/amp/trusthr/trusthr_amp_h{H}_r{R}_agg.json
# Output CSV:
#   c:/WSN-Intel-Lab-Project/augment/results/real/sweep/amp/summary_etx_trusthr_amp.csv

ROOT = Path(r"c:/WSN-Intel-Lab-Project/augment/results/real/sweep/amp")
H_LIST = [200, 500, 1000]
R_LIST = [200, 500, 1000]
PROTOS = ["etx", "trusthr"]

# Reference horizon (seconds) when cleaned CSV duration is unavailable
T_REF_SECONDS = 3600

# Helpers
RUNLOG_DIR_CANDIDATES = [
    # project_root / 'augment' / 'runlog' / 'runs'
    ROOT.parents[4] / 'augment' / 'runlog' / 'runs',
    # some environments mirror path inside realdata
    ROOT.parents[4] / 'augment' / 'realdata' / 'WSN-Intel-Lab-Project' / 'augment' / 'runlog' / 'runs',
    ROOT.parents[4] / 'augment' / 'realdata' / 'runlog' / 'runs',
]

def _norm(p: str) -> str:
    return str(p).replace('\\', '/').lower()

def _find_runlog_for_out_prefix(out_prefix: Path) -> dict | None:
    target = _norm(str(out_prefix))
    for d in RUNLOG_DIR_CANDIDATES:
        if not d.exists():
            continue
        for jf in d.glob('*.json'):
            try:
                with open(jf, 'r', encoding='utf-8') as fh:
                    meta = json.load(fh)
                if _norm(meta.get('out_prefix', '')) == target:
                    return meta
            except Exception:
                continue
    return None

def _infer_total_seconds_from_csv(cleaned_csv: str) -> float:
    # mimic eval_real._infer_total_seconds
    try:
        df = pd.read_csv(cleaned_csv)
        if 'time' in df.columns:
            t = pd.to_datetime(df['time'], errors='coerce', utc=True)
            if t.notna().any():
                dt = (t.max() - t.min()).total_seconds()
                if isinstance(dt, (int, float)) and dt > 0:
                    return float(dt)
        # numeric fallback
        x = pd.to_numeric(df.get('time', pd.Series([])), errors='coerce')
        if x.notna().any():
            rng = float(x.max() - x.min())
            if not math.isnan(rng) and rng > 0:
                diffs = x.dropna().diff().dropna()
                q = diffs.quantile(0.5) if not diffs.empty else None
                scale = 1.0
                if q is not None:
                    if q > 1e6:
                        scale = 1e-6
                    elif q > 1e3:
                        scale = 1e-3
                return max(1.0, rng * scale)
        # last resort: duration ~ number of rows (in slots)
        return float(max(1, int(df.shape[0])))
    except Exception:
        return 0.0

def _ceil_div(a: int, b: int) -> int:
    if b <= 0:
        return 0
    return int(math.ceil(a / float(b))) if a > 0 else 0

rows = []
for proto in PROTOS:
    for h in H_LIST:
        for r in R_LIST:
            f = ROOT / proto / f"{proto}_amp_h{h}_r{r}_agg.json"
            if not f.exists():
                # Skip missing combo
                continue
            with open(f, "r", encoding="utf-8") as fh:
                agg = json.load(fh)

            # Original totals
            tx = float(agg.get("tx_j_sum_total", 0.0))
            rx = float(agg.get("rx_j_sum_total", 0.0))
            cpu = float(agg.get("cpu_j_sum_total", 0.0))
            ctrl_tx = float(agg.get("ctrl_tx_j_sum", 0.0))
            ctrl_rx = float(agg.get("ctrl_rx_j_sum", 0.0))
            ctrl_cpu = float(agg.get("ctrl_cpu_j_sum", 0.0))
            ctrl_sum_j = ctrl_tx + ctrl_rx + ctrl_cpu
            data_sum_j = max(0.0, (tx + rx + cpu) - ctrl_sum_j)

            # Baseline gen-based total RX counts (sum over all control types)
            base_rx_counts = float(agg.get('ctrl_hello_sum', 0)) + float(agg.get('ctrl_cluster_sum', 0)) + float(agg.get('ctrl_routing_sum', 0)) + float(agg.get('ctrl_trust_sum', 0))
            base_rx_counts = max(1.0, base_rx_counts)

            # Discover run metadata and per-mote stats
            out_prefix = f.with_suffix('').with_name(f.stem.replace('_agg', ''))  # path without suffix
            runlog = _find_runlog_for_out_prefix(out_prefix)
            # Defaults if runlog missing
            cleaned_csv = runlog.get('cleaned') if runlog else None
            cluster_period_s = int(runlog.get('ctrl_cluster_period', 600 if runlog is None else runlog.get('ctrl_cluster_period'))) if runlog else 600
            routing_period_s = int(runlog.get('ctrl_routing_period', 300 if runlog is None else runlog.get('ctrl_routing_period'))) if runlog else 300
            trust_period_s = int(runlog.get('ctrl_trust_period', 900 if runlog is None else runlog.get('ctrl_trust_period'))) if runlog else 900

            # Compute total seconds from cleaned CSV when available; otherwise use reference horizon
            csv_seconds = _infer_total_seconds_from_csv(cleaned_csv) if cleaned_csv and os.path.exists(cleaned_csv) else 0.0
            total_seconds = csv_seconds if csv_seconds > 0 else float(T_REF_SECONDS)

            # Derive sum of degrees from per_mote file using hello column and h as per_gen used during generation
            per_mote_path = out_prefix.parent / (out_prefix.name + '_per_mote.csv')
            sum_deg = 0.0
            if per_mote_path.exists():
                pm = pd.read_csv(per_mote_path)
                # events per mote under gen-based model used in this run
                per_gen_hello = int(h)
                ev_gen = pm['gen_count'].apply(lambda g: max(1, _ceil_div(int(g), per_gen_hello)))
                # avoid div by zero
                deg_est = pm['ctrl_hello'] / ev_gen.replace(0, 1)
                sum_deg = float(deg_est.sum())
            else:
                # Fallback: approximate from baseline RX counts assuming 1 event per mote
                sum_deg = float(base_rx_counts)

            # Time-based events per mote (interpret h/r as seconds now)
            hello_events_time = max(0, _ceil_div(int(round(total_seconds)), int(h)))
            routing_events_time = max(0, _ceil_div(int(round(total_seconds)), int(r)))
            cluster_events_time = max(0, _ceil_div(int(round(total_seconds)), int(cluster_period_s)))
            trust_events_time = max(0, _ceil_div(int(round(total_seconds)), int(trust_period_s)))

            # Total RX counts under time-based model
            time_rx_counts = sum_deg * float(hello_events_time + routing_events_time + cluster_events_time + trust_events_time)

            # Scale control energy by RX-count ratio, and recompute total accordingly
            scale = time_rx_counts / base_rx_counts if base_rx_counts > 0 else 1.0
            ctrl_sum_time_j = ctrl_sum_j * scale
            total_time_j = data_sum_j + ctrl_sum_time_j
            ctrl_pct_time = (100.0 * ctrl_sum_time_j / total_time_j) if total_time_j > 0 else 0.0

            rows.append({
                "proto": proto,
                "h": h,
                "r": r,
                "pdr": float(agg.get("pdr_mean", 0.0)),
                "delay_s": float(agg.get("delay_mean", 0.0)),
                "ctrl_percent": float(ctrl_pct_time),
            })

df = pd.DataFrame(rows)
out_csv = ROOT / "summary_etx_trusthr_amp.csv"
ROOT.mkdir(parents=True, exist_ok=True)
df.to_csv(out_csv, index=False)

# Print quick insights
for proto in PROTOS:
    sub = df[df["proto"] == proto]
    if sub.empty:
        print(f"{proto}: no data.")
    else:
        print(f"{proto}: PDR [{sub['pdr'].min():.3f}, {sub['pdr'].max():.3f}], CTRL% [{sub['ctrl_percent'].min():.2f}, {sub['ctrl_percent'].max():.2f}]")
print(f"Saved {out_csv}")