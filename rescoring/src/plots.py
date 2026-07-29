#!/usr/bin/env python3
"""
src/plots.py
==============
HANDOFF_rescoring.md §6 plots, from aggregate.py's outputs:
  (a) per-residue stacked bar of score terms for one representative complex
  (b) residue x protein heatmap of mean two-body totals (see note below)
  (c) class-difference plot restricted to the LDA (position) residues

Note on (b): the doc asks for a "residue x complex" heatmap. With ~4950
complexes that's unreadable and not that useful visually, so this
aggregates to residue x PROTEIN (mean two-body total over that protein's
complexes/replicas) — the per-complex numbers are all still in
results/all_contacts.csv if finer granularity is ever needed for a specific
protein.
"""
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import config

sns.set_theme(style="whitegrid")

GROUP_COLOR = {"importer": "#2ca02c", "non_importer": "#d62728"}


def plot_representative_stacked_bar(contacts: pd.DataFrame, complex_id: str | None = None):
    if complex_id is None:
        # highest total unfavorable contribution -> visually interesting
        totals = (
            contacts.dropna(subset=["position"])
            .drop_duplicates(["complex_id", "replica", "position"])
            .groupby("complex_id")["twobody_total"].sum()
        )
        complex_id = totals.idxmax()

    sub = contacts[contacts["complex_id"] == complex_id].dropna(subset=["position"])
    pivot = sub.pivot_table(index="position", columns="scoretype", values="weighted_energy", aggfunc="mean").fillna(0)
    pivot = pivot.loc[pivot.abs().sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(pivot)), 5))
    pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("LDA position")
    ax.set_ylabel("weighted energy (REU)")
    ax.set_title(f"Score-term decomposition — {complex_id}")
    ax.legend(title="scoretype", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    out = config.FIGURES_DIR / "representative_complex_stacked_bar.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[plots] wrote {out} (complex_id={complex_id})")


def plot_residue_by_protein_heatmap(contacts: pd.DataFrame):
    mapped = contacts.dropna(subset=["position"]).drop_duplicates(["complex_id", "replica", "position"])
    per_protein = mapped.groupby(["protein", "position"])["twobody_total"].mean().reset_index()
    pivot = per_protein.pivot(index="protein", columns="position", values="twobody_total")

    fig, ax = plt.subplots(figsize=(max(10, 0.3 * pivot.shape[1]), max(6, 0.3 * pivot.shape[0])))
    sns.heatmap(pivot, cmap="RdBu_r", center=0, ax=ax, cbar_kws={"label": "mean two-body total (REU)"})
    ax.set_xlabel("LDA position")
    ax.set_ylabel("protein")
    ax.set_title("Ligand<->residue two-body total, mean per protein")
    fig.tight_layout()
    out = config.FIGURES_DIR / "residue_by_protein_heatmap.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[plots] wrote {out}")


def plot_lda_class_difference(overlay: pd.DataFrame):
    if "class_diff_importer_minus_nonimporter" not in overlay.columns:
        print("[plots] skipping LDA class-difference plot: missing class_diff column "
              "(need both importer and non_importer data).")
        return
    df = overlay.dropna(subset=["class_diff_importer_minus_nonimporter"]).sort_values(
        "class_diff_importer_minus_nonimporter"
    )
    colors = ["#2ca02c" if v < 0 else "#d62728" for v in df["class_diff_importer_minus_nonimporter"]]

    fig, ax = plt.subplots(figsize=(max(8, 0.3 * len(df)), 5))
    ax.bar(df["position"].astype(str), df["class_diff_importer_minus_nonimporter"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("LDA position")
    ax.set_ylabel("mean two-body total: importer - non_importer (REU)")
    ax.set_title("Class difference in ligand<->residue energy at LDA-flagged positions\n"
                  "(green = more stabilizing in importers, red = more stabilizing in non-importers)")
    plt.xticks(rotation=90)
    fig.tight_layout()
    out = config.FIGURES_DIR / "lda_class_difference_bar.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[plots] wrote {out}")


def main():
    all_contacts_path = config.RESULTS_DIR / "all_contacts.csv"
    overlay_path = config.RESULTS_DIR / "lda_overlay.csv"
    if not all_contacts_path.exists() or not overlay_path.exists():
        sys.exit("Missing aggregate.py outputs — run aggregate.py first.")

    contacts = pd.read_csv(all_contacts_path)
    overlay = pd.read_csv(overlay_path)

    plot_representative_stacked_bar(contacts)
    plot_residue_by_protein_heatmap(contacts)
    plot_lda_class_difference(overlay)


if __name__ == "__main__":
    main()
