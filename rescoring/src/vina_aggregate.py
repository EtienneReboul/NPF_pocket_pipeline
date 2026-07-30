#!/usr/bin/env python3
"""
src/vina_aggregate.py
========================
Pools every results/vina/per_complex/*.csv and reports:
  1. Which poses needed correction beyond the universal ligand bond-order/H
     fix (see vina_prep.py / ligand_fix.py), and the importer vs.
     non_importer ratio — as requested.
  2. Basic Vina score summary by class, for completeness.

Writes results/vina/all_scores.csv and results/vina/correction_summary.csv.
"""
import sys

import pandas as pd

import config


def load_all() -> pd.DataFrame:
    frames = []
    for csv_path in sorted(config.VINA_PER_COMPLEX_DIR.glob("*.csv")):
        df = pd.read_csv(csv_path)
        if not df.empty:
            frames.append(df)
    if not frames:
        sys.exit(f"No per-complex results in {config.VINA_PER_COMPLEX_DIR} — run vina_run_batch.py first.")
    return pd.concat(frames, ignore_index=True)


def correction_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["any_raw_anomaly"] = df["hetatm_count_anomaly"] | df["bond_count_anomaly"] | df["receptor_prep_warning"]

    rows = []
    for cls, g in df.groupby("class"):
        n = len(g)
        rows.append({
            "class": cls,
            "n_complexes": n,
            "n_ligand_bond_order_fix_needed": n,  # universal — every ligand needs it, see docstring
            "pct_ligand_bond_order_fix_needed": 100.0,
            "n_hetatm_count_anomaly": int(g["hetatm_count_anomaly"].sum()),
            "pct_hetatm_count_anomaly": 100 * g["hetatm_count_anomaly"].mean(),
            "n_bond_count_anomaly": int(g["bond_count_anomaly"].sum()),
            "pct_bond_count_anomaly": 100 * g["bond_count_anomaly"].mean(),
            "n_receptor_prep_warning": int(g["receptor_prep_warning"].sum()),
            "pct_receptor_prep_warning": 100 * g["receptor_prep_warning"].mean(),
            "n_any_raw_anomaly": int(g["any_raw_anomaly"].sum()),
            "pct_any_raw_anomaly": 100 * g["any_raw_anomaly"].mean(),
        })
    summary = pd.DataFrame(rows)

    print("=== Correction / anomaly summary ===")
    print("Every single ligand (100%, both classes) needed the bond-order/hydrogen "
          "correction from ligand_fix.py — that part is universal, driven by the "
          "same upstream sanitize_cif.py issue in every pose, not a class-specific signal.")
    print()
    print("Beyond that universal fix, raw-data anomalies (extra/missing ligand atoms "
          "or heavy-heavy bonds vs. the clean reference; receptor-prep warnings):")
    print(summary.to_string(index=False))

    if "importer" in summary["class"].values and "non_importer" in summary["class"].values:
        imp = summary.set_index("class").loc["importer", "pct_any_raw_anomaly"]
        non = summary.set_index("class").loc["non_importer", "pct_any_raw_anomaly"]
        ratio = (non / imp) if imp > 0 else float("inf")
        print(f"\nnon_importer : importer raw-anomaly rate ratio = {ratio:.2f}x "
              f"({non:.2f}% vs {imp:.2f}%)")

    return summary


def score_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["total", "lig_inter", "total_optimized", "lig_inter_optimized"]
    summary = df.groupby("class")[cols].agg(["mean", "median", "std", "count"])
    print("\n=== Vina score summary by class (kcal/mol-scale) ===")
    print(summary)
    return summary


def main():
    df = load_all()
    print(f"[vina_aggregate] loaded {len(df)} complexes")

    df.to_csv(config.VINA_RESULTS_DIR / "all_scores.csv", index=False)

    corr = correction_summary(df)
    corr.to_csv(config.VINA_RESULTS_DIR / "correction_summary.csv", index=False)

    score_summary(df)


if __name__ == "__main__":
    main()
