import pandas as pd
from pathlib import Path
from typing import Dict, Tuple


def load_mote_locs(path: str) -> Dict[int, Tuple[float,float]]:
    """Load mote_locs.txt as {mote_id: (x,y)}. Space-separated: id x y per line."""
    locs: Dict[int, Tuple[float,float]] = {}
    p = Path(path)
    with p.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            parts=line.split()
            if len(parts)<3: continue
            try:
                i = int(parts[0]); x = float(parts[1]); y = float(parts[2])
            except Exception:
                continue
            locs[i] = (x,y)
    return locs


def load_connectivity(path: str) -> Dict[Tuple[int,int], float]:
    """Load connectivity.txt as directed link success probability p_ij.
    Expected format: src dst weight per line (space separated)."""
    adj = {}
    p = Path(path)
    with p.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            parts=line.split()
            if len(parts)<3: continue
            try:
                i = int(parts[0]); j = int(parts[1]); w = float(parts[2])
            except Exception:
                continue
            adj[(i,j)] = w
    return adj


def load_cleaned_data(path: str) -> pd.DataFrame:
    """Load processed/cleaned_data.csv (or similarly formatted CSV).
    Normalize to columns: time, mote_id, value.
    Tries multiple common header names and auto-detects delimiter.
    """
    df = pd.read_csv(path, sep=None, engine='python')
    # normalize column names
    cols_map = {c.lower().strip(): c for c in df.columns}
    # time
    time_col = None
    for key in ['timestamp','time','datetime','date','ts']:
        if key in cols_map:
            time_col = cols_map[key]; break
    if time_col is None:
        # fabricate monotonic index as time index (still real order)
        df['time_idx'] = range(len(df))
        time_col = 'time_idx'
    # mote id
    id_col = None
    for key in ['mote_id','mote','moteid','node','node_id','nodeid','id','sensor_id','sensorid','src','source']:
        if key in cols_map:
            id_col = cols_map[key]; break
    if id_col is None:
        raise ValueError('Cannot find mote_id-like column in cleaned_data.csv')
    # value column (prefer temperature-like)
    val_col = None
    for key in ['temperature','temp','value','reading','val','measure']:
        if key in cols_map:
            val_col = cols_map[key]; break
    if val_col is None:
        # fallback to first numeric column excluding id/time
        num_cols = [c for c in df.columns if c not in [time_col, id_col] and pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            raise ValueError('Cannot identify a numeric sensor column in cleaned_data.csv')
        val_col = num_cols[0]
    df = df[[time_col, id_col, val_col]].rename(columns={time_col:'time', id_col:'mote_id', val_col:'value'})
    return df

