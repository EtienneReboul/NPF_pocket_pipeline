#!/usr/bin/env python3
"""
src/aggregate.py
===================
HANDOFF_rescoring.md §6: pool every per-complex tidy table, map each
protein's own residue numbering onto the common LDA "position" (1..35,
data/position_resnr_map.csv — see build_position_mapping.py for why this
mapping is needed at all), and produce:

  results/all_contacts.csv              pooled long table (every row from
                                         every results/per_complex/*.csv)
  results/residue_class_summary.csv     per (position, scoretype, class):
                                         mean/std/n weighted_energy
  results/residue_rank.csv              per (position, class): mean/std/n
                                         twobody_total, ranked
  results/lda_overlay.csv               residue_rank pivoted importer vs.
                                         non_importer, joined with the LDA
                                         loadings at that position

Run after run_batch.py has produced at least some results/per_complex/*.csv.
"""
import sys

import pandas as pd

import config


def load_all_contacts() -> pd.DataFrame:
    frames = []
    for csv_path in sorted(config.PER_COMPLEX_DIR.glob("*.csv")):
        df = pd.read_csv(csv_path)
        if not df.empty:
            frames.append(df)
    if not frames:
        sys.exit(f"No per-complex results found in {config.PER_COMPLEX_DIR} — run run_batch.py first.")
    return pd.concat(frames, ignore_index=True)


def add_position(contacts: pd.DataFrame) -> pd.DataFrame:
    position_map = pd.read_csv(config.POSITION_RESNR_MAP_CSV)
    merged = contacts.merge(
        position_map, left_on=["protein", "prot_resi"], right_on=["protein", "resnr"], how="left",
    )
    n_unmapped = merged["position"].isna().sum()
    if n_unmapped:
        print(f"[aggregate] NOTE: {n_unmapped}/{len(merged)} contact rows are outside the "
              "35 LDA pocket positions (residues Rosetta found in contact with the ligand "
              "but that CDD/the hc LDA didn't flag) — kept, with position=NaN, for the "
              "full picture in all_contacts.csv, but they drop out of the position-indexed "
              "aggregates below.")
    return merged


def residue_class_summary(contacts: pd.DataFrame) -> pd.DataFrame:
    mapped = contacts.dropna(subset=["position"])
    summary = (
        mapped.groupby(["position", "scoretype", "class"])["weighted_energy"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    return summary.sort_values(["position", "scoretype", "class"])


def residue_rank(contacts: pd.DataFrame) -> pd.DataFrame:
    mapped = contacts.dropna(subset=["position"])
    # one twobody_total per (complex_id, replica, position) — dedupe across scoretype rows
    per_edge = mapped.drop_duplicates(["complex_id", "replica", "position"])
    rank = (
        per_edge.groupby(["position", "class"])["twobody_total"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    return rank.sort_values("mean")


def lda_overlay(rank: pd.DataFrame) -> pd.DataFrame:
    pivot = rank.pivot(index="position", columns="class", values=["mean", "std", "n"])
    pivot.columns = [f"{stat}_{cls}" for stat, cls in pivot.columns]
    pivot = pivot.reset_index()
    if "mean_importer" in pivot and "mean_non_importer" in pivot:
        pivot["class_diff_importer_minus_nonimporter"] = pivot["mean_importer"] - pivot["mean_non_importer"]

    lda = pd.read_csv(config.LDA_RESIDUES_CSV)
    lda_wide = lda.pivot(index="position", columns="z_name", values="lda_coef").reset_index()
    return pivot.merge(lda_wide, on="position", how="left").sort_values(
        "class_diff_importer_minus_nonimporter"
        if "class_diff_importer_minus_nonimporter" in pivot.columns else "position"
    )


def main():
    contacts = load_all_contacts()
    print(f"[aggregate] loaded {len(contacts)} rows from "
          f"{contacts['complex_id'].nunique()} complexes")
    contacts = add_position(contacts)
    contacts.to_csv(config.RESULTS_DIR / "all_contacts.csv", index=False)

    summary = residue_class_summary(contacts)
    summary.to_csv(config.RESULTS_DIR / "residue_class_summary.csv", index=False)
    print(f"[aggregate] wrote residue_class_summary.csv ({len(summary)} rows)")

    rank = residue_rank(contacts)
    rank.to_csv(config.RESULTS_DIR / "residue_rank.csv", index=False)
    print(f"[aggregate] wrote residue_rank.csv ({len(rank)} rows)")
    print(rank.head(10))
    print(rank.tail(10))

    overlay = lda_overlay(rank)
    overlay.to_csv(config.RESULTS_DIR / "lda_overlay.csv", index=False)
    print(f"[aggregate] wrote lda_overlay.csv ({len(overlay)} rows)")


if __name__ == "__main__":
    main()
