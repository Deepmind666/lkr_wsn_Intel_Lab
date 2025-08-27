import json, os, csv
from pathlib import Path

root = Path(r"C:/WSN-Intel-Lab-Project/augment/results")
nested = Path(r"C:/WSN-Intel-Lab-Project/WSN-Intel-Lab-Project/augment/results")
files = [
    ("off", root/"trust_off_agg.json"),
    ("a02", nested/"trust_on_alpha02_agg.json"),
    ("a035", nested/"trust_on_alpha035_agg.json"),
    ("a05", root/"trust_on_alpha05_agg.json"),
    ("a065", nested/"trust_on_alpha065_agg.json"),
    ("a08", root/"trust_on_alpha08_agg.json"),
]
keys = ["pdr_mean","delay_mean","hops_mean","tx_j_mean","rx_j_mean","cpu_j_mean"]

records = {}
print("[Sanity] Trust alpha sweep aggregates:")
for label, p in files:
    if p.exists():
        with open(p, 'r', encoding='utf-8') as f:
            d = json.load(f)
        rec = {k: d.get(k) for k in keys}
        records[label] = rec
        energy = sum((rec.get('tx_j_mean') or 0, rec.get('rx_j_mean') or 0, rec.get('cpu_j_mean') or 0))
        out = ' '.join([f"{k}={rec.get(k)}" for k in keys]) + f" tot_energy_mean={energy}"
        print(f"  {label}: {out}")
    else:
        print(f"  {label}: MISSING {p}")

# Deltas vs off
if 'off' in records:
    print("\n[Sanity] Deltas vs off:")
    off = records['off']
    def energy(x):
        return (x.get('tx_j_mean') or 0) + (x.get('rx_j_mean') or 0) + (x.get('cpu_j_mean') or 0)
    for label in ["a02","a035","a05","a065","a08"]:
        if label in records:
            r = records[label]
            dpdr = (r.get('pdr_mean') or 0) - (off.get('pdr_mean') or 0)
            ddelay = (r.get('delay_mean') or 0) - (off.get('delay_mean') or 0)
            dhops = (r.get('hops_mean') or 0) - (off.get('hops_mean') or 0)
            dE = energy(r) - energy(off)
            print(f"  {label}: dpdr={dpdr:.6f} ddelay={ddelay:.6f} dhops={dhops:.6f} dEnergy={dE:.6f}")

# Export CSV
out_csv = root/"sanity_trust_alpha_sweep.csv"
with open(out_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["label", *keys, "tot_energy_mean", "dpdr_vs_off", "ddelay_vs_off", "dhops_vs_off", "dEnergy_vs_off"])
    off = records.get('off')
    def energy(x):
        return (x.get('tx_j_mean') or 0) + (x.get('rx_j_mean') or 0) + (x.get('cpu_j_mean') or 0)
    for label, p in files:
        rec = records.get(label)
        if rec is None:
            continue
        totE = energy(rec)
        if off is not None:
            dpdr = (rec.get('pdr_mean') or 0) - (off.get('pdr_mean') or 0)
            ddelay = (rec.get('delay_mean') or 0) - (off.get('delay_mean') or 0)
            dhops = (rec.get('hops_mean') or 0) - (off.get('hops_mean') or 0)
            dE = totE - energy(off)
        else:
            dpdr = ddelay = dhops = dE = ''
        writer.writerow([
            label,
            *(rec.get(k) for k in keys),
            totE, dpdr, ddelay, dhops, dE
        ])
print(f"\n[Sanity] CSV exported: {out_csv}")