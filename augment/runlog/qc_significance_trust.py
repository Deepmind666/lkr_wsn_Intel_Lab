import csv
from pathlib import Path
from math import sqrt

nested = Path(r"C:/WSN-Intel-Lab-Project/WSN-Intel-Lab-Project/augment/results")
labels = [
    ("off", nested/"trust_off_seed_{}.csv"),
    ("a02", nested/"trust_on_alpha02_seed_{}.csv"),
    ("a035", nested/"trust_on_alpha035_seed_{}.csv"),
    ("a05", nested/"trust_on_alpha05_seed_{}.csv"),
    ("a065", nested/"trust_on_alpha065_seed_{}.csv"),
    ("a08", nested/"trust_on_alpha08_seed_{}.csv"),
]
metrics = {
    'pdr': 'pdr',
    'delay': 'avg_delay_s',
    'hops': 'avg_hops',
    'tx_j': 'tx_j',
    'rx_j': 'rx_j',
    'cpu_j': 'cpu_j',
}

def read_metric_for_label(label_fmt, seed, col):
    p = Path(str(label_fmt).format(seed))
    with open(p, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        row = next(r)
    return float(row[col])

def paired_stats(off_vals, on_vals):
    # compute mean diff, std diff, 95% CI using t_crit for df=4 (n=5): 2.776
    d = [on_vals[i] - off_vals[i] for i in range(len(off_vals))]
    n = len(d)
    mean = sum(d)/n
    if n > 1:
        var = sum((x-mean)**2 for x in d)/(n-1)
        sd = var**0.5
    else:
        sd = 0.0
    se = sd/sqrt(n) if n>0 else 0.0
    t_crit = 2.776  # two-tailed 95% for df=4
    lo = mean - t_crit*se
    hi = mean + t_crit*se
    significant = (lo > 0) or (hi < 0)
    return mean, sd, (lo, hi), significant

rows = []
for name, fmt in labels:
    if name == 'off':
        continue
    for mkey, col in metrics.items():
        off_vals = [read_metric_for_label(labels[0][1], s, col) for s in range(1,6)]
        on_vals = [read_metric_for_label(fmt, s, col) for s in range(1,6)]
        # energy aggregate
        if mkey == 'cpu_j':
            # also compute energy totals
            off_E = [read_metric_for_label(labels[0][1], s, 'tx_j') + read_metric_for_label(labels[0][1], s, 'rx_j') + read_metric_for_label(labels[0][1], s, 'cpu_j') for s in range(1,6)]
            on_E = [read_metric_for_label(fmt, s, 'tx_j') + read_metric_for_label(fmt, s, 'rx_j') + read_metric_for_label(fmt, s, 'cpu_j') for s in range(1,6)]
            mean, sd, ci, sig = paired_stats(off_E, on_E)
            rows.append([name, 'energy_total', mean, sd, ci[0], ci[1], 'yes' if sig else 'no'])
        mean, sd, ci, sig = paired_stats(off_vals, on_vals)
        rows.append([name, mkey, mean, sd, ci[0], ci[1], 'yes' if sig else 'no'])

out_csv = Path(r"C:/WSN-Intel-Lab-Project/augment/results/figures/trust_significance.csv")
out_csv.parent.mkdir(parents=True, exist_ok=True)
with open(out_csv, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['vs', 'metric', 'mean_diff', 'std_diff', 'ci_lo', 'ci_hi', 'significant_95ci'])
    w.writerows(rows)
print(f"[Significance] Exported: {out_csv}")