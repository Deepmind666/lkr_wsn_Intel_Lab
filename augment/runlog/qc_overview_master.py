import re
from pathlib import Path
import pandas as pd

root = Path(r"C:/WSN-Intel-Lab-Project/augment/results")
fig_dir = root/"figures"
ov_path = fig_dir/"overview_master.csv"

print(f"[QC] Loading: {ov_path}")
df = pd.read_csv(ov_path)
print(f"[QC] Rows={len(df)}, Cols={len(df.columns)}")
print("[QC] Columns:", list(df.columns))

# Heuristics to find method column (holding 'Trust'/'SoD'/baseline names)
method_col = None
for col in df.columns:
    if df[col].dtype == 'object':
        vals = df[col].astype(str).head(50).tolist()
        joined = ' '.join(vals)
        if any(k in joined for k in ["Trust","SoD","No-SoD","Greedy","LEACH","HEED","TrustHR"]):
            method_col = col
            break
print(f"[QC] method_col={method_col}")

# Heuristics to find label column which may include 'Trust:alpha=...'
label_col = None
for col in df.columns:
    if df[col].dtype == 'object':
        s = df[col].astype(str)
        if s.str.contains(r"Trust:alpha=", regex=True, na=False).any():
            label_col = col
            break
print(f"[QC] label_col={label_col}")

# Extract Trust alphas
trust_alphas = []
if label_col is not None:
    for v in df[label_col].astype(str):
        m = re.search(r"Trust:alpha=([0-9.]+)", v)
        if m:
            trust_alphas.append(float(m.group(1)))
trust_alphas = sorted(set(trust_alphas))
print(f"[QC] Trust alphas detected: {trust_alphas}")

# Group counts by method
if method_col is not None:
    counts = df[method_col].value_counts()
    print("[QC] Counts by method:")
    for k,v in counts.items():
        print(f"  {k}: {v}")

# Try cross-check with sanity CSV
sanity_csv = root/"sanity_trust_alpha_sweep.csv"
if sanity_csv.exists() and label_col is not None:
    s = pd.read_csv(sanity_csv)
    # Map alpha to pdr_mean from sanity
    map_alpha_to_pdr = {}
    for _,row in s.iterrows():
        label = str(row.get('label',''))
        if label.startswith('a'):
            try:
                if label == 'off':
                    continue
                a = label[1:]
                # Normalize numeric part
                if '.' in a:
                    alpha = float(a)
                else:
                    # length-based rule: '02'->0.2, '05'->0.5, '035'->0.35, '08'->0.8
                    if len(a) == 2:
                        alpha = int(a)/10
                    elif len(a) == 3:
                        alpha = int(a)/100
                    else:
                        # fallback
                        alpha = float(a)
                map_alpha_to_pdr[alpha] = row.get('pdr_mean', None)
            except Exception:
                pass
    # Find candidate pdr column in overview_master
    pdr_col = None
    for cand in ['pdr_mean','pdr','PDR','avg_pdr','pdr_avg']:
        if cand in df.columns:
            pdr_col = cand
            break
    if pdr_col is None:
        # Fallback: choose first float-like column in range [0,1]
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                series = df[col].dropna()
                if len(series) and (series.between(0,1).mean() > 0.8):
                    pdr_col = col
                    break
    print(f"[QC] pdr_col={pdr_col}")
    if pdr_col is not None:
        diffs = []
        for alpha, pdr_ref in sorted(map_alpha_to_pdr.items()):
            # rows where label contains this alpha text
            mask = df[label_col].astype(str).str.contains(fr"Trust:alpha={alpha}", regex=True, na=False)
            if mask.any():
                pdr_vals = df.loc[mask, pdr_col].astype(float)
                pdr_ov = pdr_vals.mean()
                diffs.append((alpha, float(pdr_ref), float(pdr_ov), float(pdr_ov)-float(pdr_ref)))
        print("[QC] Cross-check PDR (overview vs sanity):")
        for alpha, ref, ov, delta in diffs:
            print(f"  alpha={alpha}: sanity={ref:.6f} overview={ov:.6f} delta={delta:.6e}")
else:
    print("[QC] Skipped cross-check: sanity CSV or label column missing.")

print("[QC] Done.")