import json
from pathlib import Path
import argparse
import sys
import pandas as pd
import matplotlib.pyplot as plt
import importlib


def main():
    parser = argparse.ArgumentParser(
        description="Generate figures and orchestrate publication pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=("""Examples:
      Basic:
        python -m augment.plots.make_figures --glob augment/results/example_seed_*.csv --agg augment/results/example_agg.json --outdir augment/results/figures
    
      Run full pipeline (default project root):
        python -m augment.plots.make_figures --pipeline
    
      Run pipeline and skip heatmaps and sens:
        python -m augment.plots.make_figures --pipeline --no-heatmaps --no-sens
    
      Run pipeline and include compare_trust with explicit inputs:
        python -m augment.plots.make_figures --pipeline --compare-trust ^
          --trust-off C:/WSN-Intel-Lab-Project/augment/results/trust_off_agg.json ^
          --trust-a05 C:/WSN-Intel-Lab-Project/augment/results/trust_on_alpha05_agg.json ^
          --trust-a08 C:/WSN-Intel-Lab-Project/augment/results/trust_on_alpha08_agg.json
    """)
    )
    parser.add_argument("--glob", type=str, default="augment/results/example_seed_*.csv")
    parser.add_argument("--agg", type=str, default="augment/results/example_agg.json")
    parser.add_argument("--outdir", type=str, default="augment/results/figures")
    # Pipeline entry: run publication pipeline across modules
    parser.add_argument("--pipeline", action="store_true", help="Run the publication pipeline (multi-step orchestrator)")
    parser.add_argument("--project-root", type=str, default=r"C:/WSN-Intel-Lab-Project", help="Project root for modules that require it")
    parser.add_argument("--no-significance", action="store_true", help="Skip significance_real stage")
    parser.add_argument("--no-paper", action="store_true", help="Skip plot_real_paper stage")
    parser.add_argument("--no-paper-elec", action="store_true", help="Skip plot_real_paper_elec stage")
    parser.add_argument("--no-heatmaps", action="store_true", help="Skip heatmaps_real_sweep stage")
    parser.add_argument("--no-sens", action="store_true", help="Skip sens_lines stage")
    parser.add_argument("--no-plot-real", action="store_true", help="Skip plot_real stage")
    parser.add_argument("--compare-trust", action="store_true", help="Include compare_trust stage if inputs are available")
    # Delegate to sens_lines when requested; unknown args will be forwarded
    parser.add_argument("--sens-lines", action="store_true", help="Delegate to augment.plots.sens_lines CLI (forward extra args)")
    # New: one-click overview export (Baselines + Trust + SoD)
    parser.add_argument("--overview", action="store_true", help="Run one-click overview export: Baselines + Trust (multi-alpha autodetect) + SoD, and export overview_master.csv")
    # Optional: restrict which attack scenarios appear in overview attack charts
    parser.add_argument("--attacks-whitelist", type=str, default=None, help="Comma-separated raw attack labels to include (e.g., 'attack_off_smoke,attack_blackhole_10pct,attack_grayhole_50pct_p050')")
    parser.add_argument("--attacks-whitelist-file", type=str, default=None, help="Path to a text/CSV file containing attack labels, one per line or comma-separated")
    # Paper mode: enforce fixed order, set default whitelist if none provided, and export PDF copies
    parser.add_argument("--paper-mode", action="store_true", help="Enable paper-ready settings: fixed bar order, default whitelist (if none provided), and export PDF copies of attack charts")
    args, unknown = parser.parse_known_args()

    def _load_mod(name: str):
        # Prefer relative import within package; fallback to absolute
        try:
            base = __package__ if __package__ else "augment.plots"
            return importlib.import_module(f"{base}.{name}")
        except Exception:
            return importlib.import_module(f"augment.plots.{name}")

    def _run_main(mod, argv_list: list[str]):
        orig_argv = sys.argv[:]
        try:
            sys.argv = [sys.argv[0]] + argv_list
            return mod.main()
        finally:
            sys.argv = orig_argv

    # Helper: safe path with fallback to nested results
    def _first_exist(paths: list[Path]) -> Path | None:
        for p in paths:
            if p.exists():
                return p
        return None

    # New: One-click overview orchestrator
    if args.overview:
        project_root = Path(args.project_root)
        results_root = project_root / "augment/results"
        nested_results_root = project_root / "WSN-Intel-Lab-Project/augment/results"
        figures_dir = results_root / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        print("[OVERVIEW] Start with project_root=", project_root)

        # A) Baselines summary figures
        try:
            mod = _load_mod("plot_baselines")
            summary = results_root / "baselines/summary.csv"
            ci_csv = results_root / "baselines/ci.csv"
            sig_csv = results_root / "baselines/significance.csv"
            argv = ["--summary", str(summary), "--outdir", str(figures_dir)]
            if ci_csv.exists():
                argv += ["--use_ci", "--ci_csv", str(ci_csv)]
            if sig_csv.exists():
                argv += ["--sig_csv", str(sig_csv)]
            print(f"[OVERVIEW] plot_baselines -> {figures_dir}")
            _run_main(mod, argv)
        except Exception as e:
            print("[OVERVIEW][WARN] plot_baselines failed:", e)

        # B) Trust compare (auto-detect multi-alpha series)
        try:
            mod = _load_mod("compare_trust")
            # mandatory off
            off = _first_exist([
                results_root / "trust_off_agg.json",
                nested_results_root / "trust_off_agg.json",
            ])
            if not off:
                raise FileNotFoundError("trust_off_agg.json not found in results roots")
            # optional alphas
            alpha_map = {
                "--a02": [results_root / "trust_on_alpha02_agg.json", nested_results_root / "trust_on_alpha02_agg.json"],
                "--a035": [results_root / "trust_on_alpha035_agg.json", nested_results_root / "trust_on_alpha035_agg.json"],
                "--a05": [results_root / "trust_on_alpha05_agg.json", nested_results_root / "trust_on_alpha05_agg.json"],
                "--a065": [results_root / "trust_on_alpha065_agg.json", nested_results_root / "trust_on_alpha065_agg.json"],
                "--a08": [results_root / "trust_on_alpha08_agg.json", nested_results_root / "trust_on_alpha08_agg.json"],
            }
            argv = ["--off", str(off), "--outdir", str(figures_dir)]
            detected = 0
            for flag, cands in alpha_map.items():
                p = _first_exist(cands)
                if p:
                    argv += [flag, str(p)]
                    detected += 1
            # export CSV summary alongside
            csv_name = f"trust_compare_{detected+1}levels.csv"  # include 'off'
            argv += ["--export-csv", csv_name]
            print(f"[OVERVIEW] compare_trust ({detected+1} series) -> {figures_dir}")
            _run_main(mod, argv)
        except Exception as e:
            print("[OVERVIEW][WARN] compare_trust failed:", e)

        # C) SoD compare (aggregation + per-seed ratio)
        try:
            mod = _load_mod("compare_sod")
            no_sod_agg = _first_exist([
                results_root / "sod_off_agg.json",
                nested_results_root / "sod_off_agg.json",
            ])
            sod_agg = _first_exist([
                results_root / "sod_adaptive_agg.json",
                nested_results_root / "sod_adaptive_agg.json",
            ])
            no_sod_glob = str(results_root / "sod_off_seed_*.csv")
            sod_glob = str(results_root / "sod_adaptive_seed_*.csv")
            if not no_sod_agg or not sod_agg:
                print("[OVERVIEW][INFO] SoD inputs not found; skipping SoD stage.")
            else:
                print(f"[OVERVIEW] compare_sod -> {figures_dir}")
                _run_main(mod, [
                    "--no_sod_agg", str(no_sod_agg),
                    "--sod_agg", str(sod_agg),
                    "--no_sod_glob", no_sod_glob,
                    "--sod_glob", sod_glob,
                    "--outdir", str(figures_dir),
                ])
        except Exception as e:
            print("[OVERVIEW][WARN] compare_sod failed:", e)

        # D) Export overview_master.csv by consolidating Baselines(n=50,z=0.2), Trust series, and SoD
        try:
            rows = []
            # Baselines snapshot
            bsum = results_root / "baselines/summary.csv"
            if bsum.exists():
                bdf = pd.read_csv(bsum)
                bsub = bdf[(bdf.get("nodes") == 50) & (bdf.get("noise") == 0.2)]
                if not bsub.empty:
                    for proto, label in [("greedy", "Baseline:Greedy"), ("leach", "Baseline:LEACH"), ("heed", "Baseline:HEED"), ("trusthr", "Baseline:TrustHR")]:
                        r = bsub[bsub["proto"] == proto]
                        if not r.empty:
                            r0 = r.iloc[0]
                            rows.append({
                                "group": "Baselines",
                                "label": label,
                                "pdr_mean": float(r0.get("pdr_mean", 0)),
                                "pdr_std": float(r0.get("pdr_std", 0)),
                                "delay_mean": float(r0.get("delay_mean", 0)),
                                "delay_std": float(r0.get("delay_std", 0)),
                                "hops_mean": float(r0.get("hops_mean", 0)),
                                "hops_std": float(r0.get("hops_std", 0)),
                                "tx_j_mean": float(r0.get("tx_j_mean", 0)),
                                "tx_j_std": float(r0.get("tx_j_std", 0)),
                                "rx_j_mean": float(r0.get("rx_j_mean", 0)),
                                "rx_j_std": float(r0.get("rx_j_std", 0)),
                                "cpu_j_mean": float(r0.get("cpu_j_mean", 0)),
                                "cpu_j_std": float(r0.get("cpu_j_std", 0)),
                            })
            # Trust series: reuse detection above
            trust_candidates = [
                ("Trust:off", _first_exist([results_root/"trust_off_agg.json", nested_results_root/"trust_off_agg.json"])),
                ("Trust:alpha=0.2", _first_exist([results_root/"trust_on_alpha02_agg.json", nested_results_root/"trust_on_alpha02_agg.json"])),
                ("Trust:alpha=0.35", _first_exist([results_root/"trust_on_alpha035_agg.json", nested_results_root/"trust_on_alpha035_agg.json"])),
                ("Trust:alpha=0.5", _first_exist([results_root/"trust_on_alpha05_agg.json", nested_results_root/"trust_on_alpha05_agg.json"])),
                ("Trust:alpha=0.65", _first_exist([results_root/"trust_on_alpha065_agg.json", nested_results_root/"trust_on_alpha065_agg.json"])),
                ("Trust:alpha=0.8", _first_exist([results_root/"trust_on_alpha08_agg.json", nested_results_root/"trust_on_alpha08_agg.json"])),
            ]
            for label, path in trust_candidates:
                if path and path.exists():
                    with open(path, "r", encoding="utf-8") as f:
                        d = json.load(f)
                    rows.append({
                        "group": "Trust",
                        "label": label,
                        "pdr_mean": d.get("pdr_mean", 0),
                        "pdr_std": d.get("pdr_std", 0),
                        "delay_mean": d.get("delay_mean", 0),
                        "delay_std": d.get("delay_std", 0),
                        "hops_mean": d.get("hops_mean", 0),
                        "hops_std": d.get("hops_std", 0),
                        "tx_j_mean": d.get("tx_j_mean", 0),
                        "tx_j_std": d.get("tx_j_std", 0),
                        "rx_j_mean": d.get("rx_j_mean", 0),
                        "rx_j_std": d.get("rx_j_std", 0),
                        "cpu_j_mean": d.get("cpu_j_mean", 0),
                        "cpu_j_std": d.get("cpu_j_std", 0),
                    })
            # SoD
            nosod = _first_exist([results_root/"sod_off_agg.json", nested_results_root/"sod_off_agg.json"]) 
            sod = _first_exist([results_root/"sod_adaptive_agg.json", nested_results_root/"sod_adaptive_agg.json"]) 
            if nosod and sod:
                # compute trigger ratio from per-seed in standard root (if exists)
                import glob
                import numpy as np
                def _sod_ratio(glob_pat: str) -> float:
                    files = glob.glob(glob_pat)
                    if not files:
                        return 1.0
                    total_cand, total_sent = 0, 0
                    for p in files:
                        df = pd.read_csv(p)
                        total_cand += int(df.get("sod_candidates", pd.Series([0])).sum())
                        total_sent += int(df.get("sod_sent", pd.Series([0])).sum())
                    return 1.0 if total_cand == 0 else float(total_sent) / float(total_cand)
                with open(nosod, "r", encoding="utf-8") as f0, open(sod, "r", encoding="utf-8") as f1:
                    d0, d1 = json.load(f0), json.load(f1)
                rows.append({
                    "group": "SoD",
                    "label": "No-SoD",
                    "pdr_mean": d0.get("pdr_mean", 0),
                    "pdr_std": d0.get("pdr_std", 0),
                    "delay_mean": d0.get("delay_mean", 0),
                    "delay_std": d0.get("delay_std", 0),
                    "hops_mean": d0.get("hops_mean", 0),
                    "hops_std": d0.get("hops_std", 0),
                    "tx_j_mean": d0.get("tx_j_mean", 0),
                    "tx_j_std": d0.get("tx_j_std", 0),
                    "rx_j_mean": d0.get("rx_j_mean", 0),
                    "rx_j_std": d0.get("rx_j_std", 0),
                    "cpu_j_mean": d0.get("cpu_j_mean", 0),
                    "cpu_j_std": d0.get("cpu_j_std", 0),
                    "sod_trigger_ratio": 1.0,
                })
                rows.append({
                    "group": "SoD",
                    "label": "SoD",
                    "pdr_mean": d1.get("pdr_mean", 0),
                    "pdr_std": d1.get("pdr_std", 0),
                    "delay_mean": d1.get("delay_mean", 0),
                    "delay_std": d1.get("delay_std", 0),
                    "hops_mean": d1.get("hops_mean", 0),
                    "hops_std": d1.get("hops_std", 0),
                    "tx_j_mean": d1.get("tx_j_mean", 0),
                    "tx_j_std": d1.get("tx_j_std", 0),
                    "rx_j_mean": d1.get("rx_j_mean", 0),
                    "rx_j_std": d1.get("rx_j_std", 0),
                    "cpu_j_mean": d1.get("cpu_j_mean", 0),
                    "cpu_j_std": d1.get("cpu_j_std", 0),
                    "sod_trigger_ratio": _sod_ratio(str(results_root/"sod_adaptive_seed_*.csv")),
                })
            # Attacks: auto-detect all aggregated JSONs matching attack_*_agg.json
            try:
                import glob
                attack_jsons = []
                # prefer standard results root
                attack_jsons += glob.glob(str(results_root / "attack_*_agg.json"))
                # include nested results root if any
                attack_jsons += glob.glob(str(nested_results_root / "attack_*_agg.json"))
                # de-duplicate while preserving order
                seen = set()
                attack_jsons = [p for p in attack_jsons if not (p in seen or seen.add(p))]
                for p in attack_jsons:
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            d = json.load(f)
                        label = Path(p).stem.replace("_agg", "")
                        rows.append({
                            "group": "Attacks",
                            "label": label,
                            "pdr_mean": d.get("pdr_mean", 0),
                            "pdr_std": d.get("pdr_std", 0),
                            "delay_mean": d.get("delay_mean", 0),
                            "delay_std": d.get("delay_std", 0),
                            "hops_mean": d.get("hops_mean", 0),
                            "hops_std": d.get("hops_std", 0),
                            "tx_j_mean": d.get("tx_j_mean", 0),
                            "tx_j_std": d.get("tx_j_std", 0),
                            "rx_j_mean": d.get("rx_j_mean", 0),
                            "rx_j_std": d.get("rx_j_std", 0),
                            "cpu_j_mean": d.get("cpu_j_mean", 0),
                            "cpu_j_std": d.get("cpu_j_std", 0),
                            # attack-specific metrics if available
                            "attack_active_rounds_mean": d.get("attack_active_rounds_mean", 0),
                            "attack_compromised_mean": d.get("attack_compromised_mean", 0),
                            "drop_blackhole_mean": d.get("drop_blackhole_mean", 0),
                            "drop_grayhole_mean": d.get("drop_grayhole_mean", 0),
                            "sinkhole_choices_mean": d.get("sinkhole_choices_mean", 0),
                            "trust_updates_mean": d.get("trust_updates_mean", 0),
                            "malicious_detected_mean": d.get("malicious_detected_mean", 0),
                        })
                    except Exception as _:
                        continue
            except Exception as e:
                print("[OVERVIEW][WARN] Detect attacks failed:", e)
            # Write master CSV
            if rows:
                import csv
                master_csv = figures_dir / "overview_master.csv"
                cols = sorted({k for r in rows for k in r.keys()})
                with open(master_csv, "w", newline='', encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=cols)
                    w.writeheader()
                    for r in rows:
                        w.writerow(r)
                print(f"[OVERVIEW] overview_master.csv -> {master_csv}")
                # Optional: quick PDR bar for Attacks
                try:
                    adf = pd.DataFrame([r for r in rows if r.get("group") == "Attacks"]).copy()
                    if not adf.empty:
                        # Paper mode desired raw-label order
                        paper_order_raw = [
                            "attack_off_smoke",
                            "attack_blackhole_10pct",
                            "attack_blackhole_30pct",
                            "attack_grayhole_50pct_p050",
                            "attack_grayhole_50pct_p090",
                            "attack_sinkhole_bias050",
                            "attack_sinkhole_bias080",
                        ]
                        
                        # Optional whitelist filter for attack charts
                        wl = set()
                        if getattr(args, "attacks_whitelist", None):
                            wl.update([s.strip() for s in str(args.attacks_whitelist).split(",") if s.strip()])
                        if getattr(args, "attacks_whitelist_file", None):
                            try:
                                p = Path(args.attacks_whitelist_file)
                                if p.exists():
                                    for line in p.read_text(encoding="utf-8").splitlines():
                                        for tok in line.replace(",", " ").split():
                                            t = tok.strip()
                                            if t:
                                                wl.add(t)
                            except Exception as _e:
                                print(f"[OVERVIEW][WARN] Failed to read whitelist file: {_e}")
                        # If paper-mode and no explicit whitelist provided, apply default paper whitelist
                        if args.paper_mode and not wl:
                            wl.update(paper_order_raw)
                        if wl:
                            before = len(adf)
                            adf = adf[adf["label"].isin(wl)].copy()
                            after = len(adf)
                            print(f"[OVERVIEW] Attacks whitelist filter applied: kept {after}/{before}")
                        
                        # Fixed order in paper-mode (based on raw labels)
                        desired_order = [x for x in paper_order_raw if x in set(adf["label"]) ] if args.paper_mode else None
                        if desired_order:
                            adf["label"] = pd.Categorical(adf["label"], categories=desired_order, ordered=True)
                            adf = adf.sort_values(by="label")
                        else:
                            # sort labels for readability (fallback)
                            adf = adf.sort_values(by="pdr_mean", ascending=False)
                        
                        # Create paper-friendly labels with explicit mapping and regex fallback
                        import re as _re
                        _label_map = {
                            # Baseline
                            "attack_off_smoke": "Baseline",
                            "attack_off_trust_off_smoke": "Baseline (trust-off)",
                            # Blackhole
                            "attack_blackhole_10pct": "BH-10%",
                            "attack_blackhole_20pct": "BH-20%",
                            "attack_blackhole_30pct": "BH-30%",
                            # Grayhole
                            "attack_grayhole_50pct_p050": "GH-50% (p=0.50)",
                            "attack_grayhole_50pct_p070": "GH-50% (p=0.70)",
                            "attack_grayhole_50pct_p090": "GH-50% (p=0.90)",
                            # Sinkhole
                            "attack_sinkhole_bias050": "SH-b0.050",
                            "attack_sinkhole_bias080": "SH-b0.080",
                            "attack_sinkhole_bias080_fixed_ids": "SH-b0.080 (fixed-ids)",
                        }
                        def clean_label(label: str) -> str:
                            s = str(label)
                            if s in _label_map:
                                return _label_map[s]
                            # Fallback patterns
                            # BH_kpct
                            m = _re.match(r"^attack_blackhole_(\d{1,3})pct$", s)
                            if m:
                                return f"BH-{int(m.group(1))}%"
                            # GH_kpct_pXYZ -> p=0.XYZ
                            m = _re.match(r"^attack_grayhole_(\d{1,3})pct_p(\d{3})$", s)
                            if m:
                                pct = int(m.group(1))
                                p = m.group(2)
                                return f"GH-{pct}% (p=0.{p})"
                            # SH-biasXYZ -> b0.XYZ
                            m = _re.match(r"^attack_sinkhole_bias(\d{3})(?:_.*)?$", s)
                            if m:
                                b = m.group(1)
                                return f"SH-b0.{b}"
                            # Generic compact: remove 'attack_' prefix
                            s2 = s.replace("attack_", "")
                            return s2
                        
                        adf["clean_label"] = adf["label"].map(clean_label)
                        
                        # Dynamic figure size based on number of bars
                        n_bars = len(adf)
                        fig_width = max(10, n_bars * 0.8)
                        
                        plt.figure(figsize=(fig_width, 5))
                        bars = plt.bar(adf["clean_label"], adf["pdr_mean"], 
                                     color="#4C78A8", edgecolor="white", linewidth=0.8, alpha=0.85)
                        
                        # Add value annotations on bars
                        for bar, val in zip(bars, adf["pdr_mean"]):
                            height = bar.get_height()
                            plt.annotate(f'{val:.3f}',
                                       xy=(bar.get_x() + bar.get_width()/2, height),
                                       xytext=(0, 3),  
                                       textcoords="offset points",
                                       ha='center', va='bottom', fontsize=8)
                        
                        plt.ylabel("PDR (Packet Delivery Ratio)", fontsize=11)
                        plt.title("PDR vs Attack Scenarios", fontsize=13, pad=20)
                        plt.xticks(rotation=60, ha="right", fontsize=9)
                        plt.yticks(fontsize=10)
                        plt.grid(axis='y', alpha=0.3, linestyle='--')
                        plt.ylim(0, max(adf["pdr_mean"]) * 1.15)  # Extra space for annotations
                        plt.tight_layout()
                        plt.subplots_adjust(bottom=0.25)  # More space for rotated labels
                        plt.savefig(figures_dir / "attacks_pdr.svg", dpi=150, bbox_inches='tight')
                        if args.paper_mode:
                            plt.savefig(figures_dir / "paper_attacks_pdr.pdf", bbox_inches='tight')
                        print(f"[OVERVIEW] attacks_pdr.svg -> {figures_dir}")

                        # Additional: Delay and TX+RX with family-based coloring
                        import re
                        from matplotlib.patches import Patch

                        def _attack_family(lbl: str) -> str:
                            # Extract family after 'attack_' prefix; take first token
                            name = re.sub(r"^attack_", "", str(lbl))
                            # e.g., blackhole_20pct -> blackhole; grayhole_50pct_p070 -> grayhole; off_smoke -> off
                            return (name.split("_")[0] if name else "other").lower()

                        fam_colors = {
                            "off": "#4C78A8",       # blue
                            "blackhole": "#D62728", # red
                            "grayhole": "#9467BD",  # purple
                            "sinkhole": "#2CA02C",  # green
                        }
                        fam_order = ["off", "blackhole", "grayhole", "sinkhole"]

                        adf["family"] = adf["label"].map(_attack_family)
                        adf["color"] = adf["family"].map(lambda f: fam_colors.get(f, "#72B7B2"))

                        # Delay: fixed order in paper-mode, otherwise ascending for readability
                        ddf = adf.copy()
                        if desired_order:
                            ddf["label"] = pd.Categorical(ddf["label"], categories=desired_order, ordered=True)
                            ddf = ddf.sort_values(by="label")
                        else:
                            ddf = ddf.sort_values(by="delay_mean", ascending=True)
                        
                        # use same cleaned labels
                        ddf["clean_label"] = ddf["label"].map(clean_label)
                        n_bars_d = len(ddf)
                        fig_width_d = max(10, n_bars_d * 0.8)
                        
                        plt.figure(figsize=(fig_width_d, 5))
                        bars = plt.bar(ddf["clean_label"], ddf["delay_mean"], color=ddf["color"], edgecolor="white", linewidth=0.8, alpha=0.9)
                        plt.ylabel("E2E Delay (s, mean)", fontsize=11)
                        plt.title("Delay vs Attack Scenarios", fontsize=13, pad=20)
                        plt.xticks(rotation=60, ha="right", fontsize=9)
                        plt.yticks(fontsize=10)
                        plt.grid(axis='y', alpha=0.3, linestyle='--')
                        
                        # Add value annotations
                        for bar, val in zip(bars, ddf["delay_mean"]):
                            height = bar.get_height()
                            plt.annotate(f'{val:.3f}',
                                         xy=(bar.get_x() + bar.get_width()/2, height),
                                         xytext=(0, 3), textcoords="offset points",
                                         ha='center', va='bottom', fontsize=8)
                        
                        # Legend by family
                        handles = [Patch(facecolor=fam_colors[k], label=k.title()) for k in fam_order if k in set(ddf["family"]) ]
                        if handles:
                            plt.legend(handles=handles, title="Attack Family", frameon=False)
                        
                        plt.ylim(0, max(ddf["delay_mean"]) * 1.15)
                        plt.tight_layout()
                        plt.subplots_adjust(bottom=0.25)
                        plt.savefig(figures_dir / "attacks_delay.svg", dpi=150, bbox_inches='tight')
                        if args.paper_mode:
                            plt.savefig(figures_dir / "paper_attacks_delay.pdf", bbox_inches='tight')
                        print(f"[OVERVIEW] attacks_delay.svg -> {figures_dir}")

                        # TX+RX energy: fixed order in paper-mode, otherwise ascending
                        adf["txrx_mean"] = adf.get("tx_j_mean", 0) + adf.get("rx_j_mean", 0)
                        edf = adf.sort_values(by="txrx_mean", ascending=True).copy()
                        
                        edf["clean_label"] = edf["label"].map(clean_label)
                        n_bars_e = len(edf)
                        fig_width_e = max(10, n_bars_e * 0.8)
                        
                        plt.figure(figsize=(fig_width_e, 5))
                        bars = plt.bar(edf["clean_label"], edf["txrx_mean"], color=edf["color"], edgecolor="white", linewidth=0.8, alpha=0.9)
                        plt.ylabel("TX+RX Energy (J, mean)", fontsize=11)
                        plt.title("TX+RX Energy vs Attack Scenarios", fontsize=13, pad=20)
                        plt.xticks(rotation=60, ha="right", fontsize=9)
                        plt.yticks(fontsize=10)
                        plt.grid(axis='y', alpha=0.3, linestyle='--')
                        
                        # Add value annotations
                        for bar, val in zip(bars, edf["txrx_mean"]):
                            height = bar.get_height()
                            plt.annotate(f'{val:.2f}',
                                         xy=(bar.get_x() + bar.get_width()/2, height),
                                         xytext=(0, 3), textcoords="offset points",
                                         ha='center', va='bottom', fontsize=8)
                        
                        handles = [Patch(facecolor=fam_colors[k], label=k.title()) for k in fam_order if k in set(edf["family"]) ]
                        if handles:
                            plt.legend(handles=handles, title="Attack Family", frameon=False)
                        
                        plt.ylim(0, max(edf["txrx_mean"]) * 1.15)
                        plt.tight_layout()
                        plt.subplots_adjust(bottom=0.25)
                        plt.savefig(figures_dir / "attacks_txrx.svg", dpi=150, bbox_inches='tight')
                        if args.paper_mode:
                            plt.savefig(figures_dir / "paper_attacks_txrx.pdf", bbox_inches='tight')
                        print(f"[OVERVIEW] attacks_txrx.svg -> {figures_dir}")
                except Exception as e:
                    print(f"[OVERVIEW][WARN] Attacks quick figures failed: {e}")
            else:
                print("[OVERVIEW][INFO] No rows assembled for overview_master.csv")
        except Exception as e:
            print("[OVERVIEW][WARN] export overview_master.csv failed:", e)

        print("[OVERVIEW] Finished.")
        return

    # Pipeline orchestrator
    if args.pipeline:
        project_root = Path(args.project_root)
        real_root = project_root / "augment/results/real"
        figures_dir = real_root / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        print("[PIPE] Start pipeline with project_root=", project_root)

        # Stage 1: significance (TrustHR vs others)
        if not args.no_significance:
            try:
                mod = _load_mod("significance_real")
                out_csv = real_root / "significance.csv"
                print(f"[PIPE] significance_real -> {out_csv}")
                _run_main(mod, ["--root", str(real_root), "--out_csv", str(out_csv)])
            except Exception as e:
                print("[PIPE][WARN] significance_real failed:", e)
        else:
            print("[PIPE] Skip significance_real")

        # Stage 2: paper multi-panel
        if not args.no_paper:
            try:
                mod = _load_mod("plot_real_paper")
                print(f"[PIPE] plot_real_paper -> {figures_dir}")
                _run_main(mod, ["--root", str(real_root), "--out_png", str(figures_dir/"real_overview_paper.png"),
                                  "--out_pdf", str(figures_dir/"real_overview_paper.pdf"),
                                  "--out_svg", str(figures_dir/"real_overview_paper.svg")])
            except Exception as e:
                print("[PIPE][WARN] plot_real_paper failed:", e)
        else:
            print("[PIPE] Skip plot_real_paper")

        # Stage 3: electronics-only multi-panel
        if not args.no_paper_elec:
            try:
                mod = _load_mod("plot_real_paper_elec")
                print(f"[PIPE] plot_real_paper_elec -> {figures_dir}")
                _run_main(mod, ["--root", str(real_root), "--out_png", str(figures_dir/"real_overview_paper_elec.png"),
                                  "--out_pdf", str(figures_dir/"real_overview_paper_elec.pdf")])
            except Exception as e:
                print("[PIPE][WARN] plot_real_paper_elec failed:", e)
        else:
            print("[PIPE] Skip plot_real_paper_elec")

        # Stage 4: ETX/TRUSTHR heatmaps for ELEC & AMP
        if not args.no_heatmaps:
            try:
                mod = _load_mod("heatmaps_real_sweep")
                print(f"[PIPE] heatmaps_real_sweep -> ELEC/AMP figs_pub")
                _run_main(mod, ["--root", str(project_root)])
            except Exception as e:
                print("[PIPE][WARN] heatmaps_real_sweep failed:", e)
        else:
            print("[PIPE] Skip heatmaps_real_sweep")

        # Stage 5: Sensitivity lines (ELEC + AMP), strict QC
        if not args.no_sens:
            try:
                mod = _load_mod("sens_lines")
                print(f"[PIPE] sens_lines (both, metric all, qc strict) -> sens/figs_pub")
                _run_main(mod, ["--mode", "both", "--metric", "all", "--qc", "--qc-strict"])  # use defaults for outdir
            except Exception as e:
                print("[PIPE][WARN] sens_lines failed:", e)
        else:
            print("[PIPE] Skip sens_lines")

        # Stage 6: Simple real summary bars
        if not args.no_plot_real:
            try:
                mod = _load_mod("plot_real")
                print(f"[PIPE] plot_real -> {figures_dir}")
                _run_main(mod, ["--root", str(real_root), "--outdir", str(figures_dir)])
            except Exception as e:
                print("[PIPE][WARN] plot_real failed:", e)
        else:
            print("[PIPE] Skip plot_real")

        # Optional Stage 7: Trust ablation/weights comparison (only if enabled + inputs exist)
        if args.compare_trust:
            try:
                mod = _load_mod("compare_trust")
                # Try to locate inputs in augment/results (can be overridden by CLI)
                results_root = project_root / "augment/results"
                off = Path(args.trust_off) if getattr(args, "trust_off", None) else results_root / "trust_off_agg.json"
                a05 = Path(args.trust_a05) if getattr(args, "trust_a05", None) else results_root / "trust_on_alpha05_agg.json"
                a08 = Path(args.trust_a08) if getattr(args, "trust_a08", None) else results_root / "trust_on_alpha08_agg.json"
                outdir = Path(args.trust_outdir) if getattr(args, "trust_outdir", None) else (project_root / "augment/results/figures")
                if off.exists() and a05.exists() and a08.exists():
                    print(f"[PIPE] compare_trust -> {outdir}")
                    _run_main(mod, ["--off", str(off), "--a05", str(a05), "--a08", str(a08),
                                      "--outdir", str(outdir)])
                else:
                    print("[PIPE][INFO] compare_trust inputs not found; skipping. Expected any missing:")
                    print("  ", off)
                    print("  ", a05)
                    print("  ", a08)
            except Exception as e:
                print("[PIPE][WARN] compare_trust failed:", e)
        else:
            print("[PIPE] Skip compare_trust (disabled)")

        print("[PIPE] All requested stages finished.")
        return

    # If --sens-lines is specified (and not running pipeline), forward remaining args to sens_lines and exit
    if args.sens_lines:
        try:
            from . import sens_lines as sens_core  # type: ignore
        except Exception:
            from augment.plots import sens_lines as sens_core  # type: ignore
        orig_argv = sys.argv[:]
        try:
            sys.argv = [sys.argv[0]] + unknown
            sens_core.main()
        finally:
            sys.argv = orig_argv
        return

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load per-seed
    import glob
    files = glob.glob(args.glob)
    dfs = [pd.read_csv(f) for f in files]
    if not dfs:
        print("No per-seed CSV files found.")
        return
    df = pd.concat(dfs, ignore_index=True)

    # Load aggregate
    with open(args.agg, "r", encoding="utf-8") as f:
        agg = json.load(f)

    # Figure 1: PDR distribution (boxplot) + mean line
    plt.figure(figsize=(6,4))
    df.boxplot(column=["pdr"])
    plt.axhline(agg.get("pdr_mean", 0), color="red", linestyle="--", label=f"mean={agg.get('pdr_mean',0):.3f}")
    plt.ylabel("PDR")
    plt.title("Packet Delivery Ratio (per-seed)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir/"pdr_box.png", dpi=200)

    # Figure 2: Delay distribution
    plt.figure(figsize=(6,4))
    df.boxplot(column=["avg_delay_s"])
    plt.axhline(agg.get("delay_mean", 0), color="red", linestyle="--", label=f"mean={agg.get('delay_mean',0):.3f}s")
    plt.ylabel("Average E2E Delay (s)")
    plt.title("End-to-End Delay (per-seed)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir/"delay_box.png", dpi=200)

    # Figure 3: Hops distribution
    plt.figure(figsize=(6,4))
    df.boxplot(column=["avg_hops"])
    plt.axhline(agg.get("hops_mean", 0), color="red", linestyle="--", label=f"mean={agg.get('hops_mean',0):.3f}")
    plt.ylabel("Average Hops")
    plt.title("Average Hops (per-seed)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir/"hops_box.png", dpi=200)

    print(f"Saved figures to {outdir}")


if __name__ == "__main__":
    main()

