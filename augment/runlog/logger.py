import csv
import json
import os
from pathlib import Path
from datetime import datetime
import platform
import hashlib

RUNLOG_DIR = Path('WSN-Intel-Lab-Project/augment/runlog')
RUNLOG_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_CSV = RUNLOG_DIR / 'experiment_registry.csv'
RUNS_DIR = RUNLOG_DIR / 'runs'
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def _sha1(path: str) -> str:
    try:
        h = hashlib.sha1()
        with open(path, 'rb') as f:
            while True:
                b = f.read(8192)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except Exception:
        return ''


def log_run(task_tag: str, meta: dict) -> str:
    """Append a row to experiment_registry.csv and dump a JSON snapshot under runs/.
    Returns run_id.
    """
    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    run_id = f"{task_tag}_{ts}"
    meta = dict(meta) if meta else {}
    meta.update({
        'run_id': run_id,
        'task_tag': task_tag,
        'datetime_utc': ts,
        'python': platform.python_version(),
        'platform': platform.platform(),
    })
    # add file hashes if present
    for k in ['connectivity', 'cleaned', 'mote_locs']:
        p = meta.get(k)
        if p:
            meta[f'{k}_sha1'] = _sha1(p)
    # write JSON snapshot
    snapshot = RUNS_DIR / f'{run_id}.json'
    with open(snapshot, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    # append CSV registry (create header if new)
    headers = ['run_id','task_tag','datetime_utc','proto','energy_mode','pij_scale','connectivity','cleaned','mote_locs','out_prefix','n_motes','total_gen','pdr_mean','delay_mean','tx_j_sum_data','rx_j_sum_data']
    new_file = not REGISTRY_CSV.exists()
    with open(REGISTRY_CSV, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=headers)
        if new_file:
            w.writeheader()
        row = {h: meta.get(h, '') for h in headers}
        w.writerow(row)
    return run_id

