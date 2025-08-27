import os, re, json, csv
import math
from pathlib import Path
import pandas as pd

base = r"c:\\WSN-Intel-Lab-Project\\augment\\results\\real\\sweep\\elec"
protos = ['etx','trusthr']
rows = []
pat = re.compile(r"^(?P<proto>[a-z]+)_elec_h(?P<h>\d+)_r(?P<r>\d+)_agg\.json$")

ROOT = Path(base)
RUNLOG_DIR_CANDIDATES = [
    ROOT.parents[4] / 'augment' / 'runlog' / 'runs',
    ROOT.parents[4] / 'augment' / 'realdata' / 'WSN-Intel-Lab-Project' / 'augment' / 'runlog' / 'runs',
    ROOT.parents[4] / 'augment' / 'realdata' / 'runlog' / 'runs',
]
# Introduce a sensible reference horizon (seconds) when cleaned CSV duration is unavailable
T_REF_SECONDS = 3600

def _norm(p: str) -> str:
    return str(p).replace('\\\\', '/').replace('\\', '/').lower()

def _find_runlog_for_out_prefix(out_prefix: Path):
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
    try:
        df = pd.read_csv(cleaned_csv)
        if 'time' in df.columns:
            t = pd.to_datetime(df['time'], errors='coerce', utc=True)
            if t.notna().any():
                dt = (t.max() - t.min()).total_seconds()
                if isinstance(dt, (int, float)) and dt > 0:
                    return float(dt)
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
                return max(0.0, rng * scale)
        # if cannot infer, return 0.0 so caller can apply reference fallback
        return 0.0
    except Exception:
        # On any failure, signal caller to use the reference horizon
        return 0.0

def _ceil_div(a: int, b: int) -> int:
    if b <= 0:
        return 0
    return int(math.ceil(a / float(b))) if a > 0 else 0

for proto in protos:
    d = os.path.join(base, proto)
    if not os.path.isdir(d):
        continue
    for fn in os.listdir(d):
        m = pat.match(fn)
        if not m: continue
        path = os.path.join(d, fn)
        with open(path, 'r', encoding='utf-8') as f:
            agg = json.load(f)

        # Original energy partition
        tx = float(agg.get('tx_j_sum_total', 0.0))
        rx = float(agg.get('rx_j_sum_total', 0.0))
        cpu = float(agg.get('cpu_j_sum_total', 0.0))
        ctrl_tx = float(agg.get('ctrl_tx_j_sum', 0.0))
        ctrl_rx = float(agg.get('ctrl_rx_j_sum', 0.0))
        ctrl_cpu = float(agg.get('ctrl_cpu_j_sum', 0.0))
        ctrl_sum_j = ctrl_tx + ctrl_rx + ctrl_cpu
        data_sum_j = max(0.0, (tx + rx + cpu) - ctrl_sum_j)

        # Baseline RX-counts (gen-based)
        base_rx_counts = float(agg.get('ctrl_hello_sum', 0)) + float(agg.get('ctrl_cluster_sum', 0)) + float(agg.get('ctrl_routing_sum', 0)) + float(agg.get('ctrl_trust_sum', 0))
        base_rx_counts = max(1.0, base_rx_counts)

        # Locate runlog and per-mote CSV
        out_prefix = Path(path).with_suffix('').with_name(Path(path).stem.replace('_agg', ''))
        runlog = _find_runlog_for_out_prefix(out_prefix)
        cleaned_csv = runlog.get('cleaned') if runlog else None
        cluster_period_s = int(runlog.get('ctrl_cluster_period', 600)) if runlog else 600
        routing_period_s = int(runlog.get('ctrl_routing_period', 300)) if runlog else 300
        trust_period_s = int(runlog.get('ctrl_trust_period', 900)) if runlog else 900

        # Infer duration from cleaned CSV; if unavailable, fall back to reference horizon
        csv_seconds = _infer_total_seconds_from_csv(cleaned_csv) if cleaned_csv and os.path.exists(cleaned_csv) else 0.0
        total_seconds = csv_seconds if csv_seconds > 0 else float(T_REF_SECONDS)

        per_mote_path = out_prefix.parent / (out_prefix.name + '_per_mote.csv')
        sum_deg = 0.0
        h = int(m.group('h'))
        if per_mote_path.exists():
            pm = pd.read_csv(per_mote_path)
            ev_gen = pm['gen_count'].apply(lambda g: max(1, _ceil_div(int(g), h)))
            deg_est = pm['ctrl_hello'] / ev_gen.replace(0, 1)
            sum_deg = float(deg_est.sum())
        else:
            sum_deg = float(base_rx_counts)

        # Time-based events (interpret h/r as seconds)
        r = int(m.group('r'))
        hello_events_time = max(0, _ceil_div(int(round(total_seconds)), h))
        routing_events_time = max(0, _ceil_div(int(round(total_seconds)), r))
        cluster_events_time = max(0, _ceil_div(int(round(total_seconds)), cluster_period_s))
        trust_events_time = max(0, _ceil_div(int(round(total_seconds)), trust_period_s))

        time_rx_counts = sum_deg * float(hello_events_time + routing_events_time + cluster_events_time + trust_events_time)
        scale = time_rx_counts / base_rx_counts if base_rx_counts > 0 else 1.0
        ctrl_sum_time_j = ctrl_sum_j * scale
        total_time_j = data_sum_j + ctrl_sum_time_j
        ctrl_pct_time = (ctrl_sum_time_j / total_time_j * 100.0) if total_time_j > 0 else 0.0

        rows.append({
            'proto': m.group('proto'),
            'hello_per_gen': h,
            'routing_per_gen': r,
            'pdr_mean': float(agg.get('pdr_mean', 0.0)),
            'delay_mean': float(agg.get('delay_mean', 0.0)),
            'ctrl_j_sum': float(ctrl_sum_time_j),
            'total_j_sum': float(total_time_j),
            'ctrl_pct': float(ctrl_pct_time),
        })

# sort rows
rows.sort(key=lambda x: (x['proto'], x['hello_per_gen'], x['routing_per_gen']))
# write CSV
out_csv = os.path.join(base, 'summary_etx_trusthr_elec.csv')
with open(out_csv, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['proto','hello_per_gen','routing_per_gen','pdr_mean','delay_mean','ctrl_j_sum','total_j_sum','ctrl_pct'])
    w.writeheader(); w.writerows(rows)
# insights
from collections import defaultdict
by_proto = defaultdict(list)
for rrow in rows:
    by_proto[rrow['proto']].append(rrow)
print('SUMMARY INSIGHTS:')
for proto, lst in by_proto.items():
    max_pdr = max(r['pdr_mean'] for r in lst) if lst else 0.0
    min_ctrl = min(lst, key=lambda r: r['ctrl_pct']) if lst else None
    max_ctrl = max(lst, key=lambda r: r['ctrl_pct']) if lst else None
    print(f"- {proto}: PDR range [{min(r['pdr_mean'] for r in lst):.3f}, {max_pdr:.3f}], ctrl% range [{min_ctrl['ctrl_pct']:.2f}%, {max_ctrl['ctrl_pct']:.2f}%]")
    print(f"  min ctrl% at h={min_ctrl['hello_per_gen']}, r={min_ctrl['routing_per_gen']}; max ctrl% at h={max_ctrl['hello_per_gen']}, r={max_ctrl['routing_per_gen']}")
print(f"CSV saved: {out_csv}")
