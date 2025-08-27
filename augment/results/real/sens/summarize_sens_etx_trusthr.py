import json
import re
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent

MODES = ["elec", "amp"]
PROTOS = ["etx", "trusthr"]

fname_re = re.compile(r"_(h\d+)_r(\d+)_agg\.json$")


def parse_hr_from_name(p: Path) -> tuple[int, int]:
    s = p.stem  # e.g., etx_elec_h500_r2000_agg
    m = re.search(r"_h(\d+)_r(\d+)_agg$", s)
    if not m:
        raise ValueError(f"Unexpected filename for hr parse: {p.name}")
    return int(m.group(1)), int(m.group(2))


def per_file_row(mode: str, proto: str, f: Path) -> dict:
    with open(f, "r", encoding="utf-8") as fh:
        agg = json.load(fh)
    h, r = parse_hr_from_name(f)
    ctrl_total_j = float(agg.get("ctrl_tx_j_sum", 0.0) + agg.get("ctrl_rx_j_sum", 0.0) + agg.get("ctrl_cpu_j_sum", 0.0))
    total_j = float(agg.get("tx_j_sum_total", 0.0) + agg.get("rx_j_sum_total", 0.0) + agg.get("cpu_j_sum_total", 0.0))
    ctrl_percent = (100.0 * ctrl_total_j / total_j) if total_j > 0 else 0.0
    return {
        "mode": mode,
        "proto": proto,
        "h": h,
        "r": r,
        "pdr": float(agg.get("pdr_mean", 0.0)),
        "delay_s": float(agg.get("delay_mean", 0.0)),
        "ctrl_hello_sum": int(agg.get("ctrl_hello_sum", 0)),
        "ctrl_routing_sum": int(agg.get("ctrl_routing_sum", 0)),
        "ctrl_total_j": ctrl_total_j,
        "total_j": total_j,
        "ctrl_percent": ctrl_percent,
    }


def summarize_mode(mode: str) -> pd.DataFrame:
    rows: list[dict] = []
    for proto in PROTOS:
        d = BASE / mode / proto
        for f in sorted(d.glob("*_agg.json")):
            try:
                rows.append(per_file_row(mode, proto, f))
            except Exception as e:
                print(f"Skip {f.name}: {e}")
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"No records for mode={mode}")
        return df
    df = df.sort_values(by=["proto", "h", "r"]).reset_index(drop=True)
    return df


def quick_insights(df: pd.DataFrame, mode: str):
    print(f"=== Quick insights for {mode} ===")
    for proto, g in df.groupby("proto"):
        pdr_min, pdr_max = g["pdr"].min(), g["pdr"].max()
        ctrl_min, ctrl_max = g["ctrl_percent"].min(), g["ctrl_percent"].max()
        print(f"{proto.upper()}: PDR [{pdr_min:.3f}, {pdr_max:.3f}] | ctrl% [{ctrl_min:.2f}, {ctrl_max:.2f}]")
        # identify min/max ctrl% settings
        idx_min = g["ctrl_percent"].idxmin()
        idx_max = g["ctrl_percent"].idxmax()
        if pd.notna(idx_min):
            row_min = g.loc[idx_min]
            print(f"  min ctrl% at h={int(row_min['h'])}, r={int(row_min['r'])}: {row_min['ctrl_percent']:.2f}%")
        if pd.notna(idx_max):
            row_max = g.loc[idx_max]
            print(f"  max ctrl% at h={int(row_max['h'])}, r={int(row_max['r'])}: {row_max['ctrl_percent']:.2f}%")


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    for mode in MODES:
        df = summarize_mode(mode)
        if df.empty:
            continue
        out_csv = BASE / f"summary_sens_etx_trusthr_{mode}.csv"
        df.to_csv(out_csv, index=False)
        print(f"Saved {out_csv}")
        quick_insights(df, mode)


if __name__ == "__main__":
    main()