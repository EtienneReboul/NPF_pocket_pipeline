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
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import config

sns.set_theme(style="whitegrid")

GROUP_COLOR = {"importer": "#2ca02c", "non_importer": "#d62728"}


def _complex_totals(contacts: pd.DataFrame) -> pd.Series:
    """complex_id -> summed twobody_total, one edge per (complex_id, replica, position)."""
    return (
        contacts.dropna(subset=["position"])
        .drop_duplicates(["complex_id", "replica", "position"])
        .groupby("complex_id")["twobody_total"].sum()
    )


def _draw_stacked_bar(sub: pd.DataFrame, title: str, out: Path) -> None:
    pivot = sub.pivot_table(index="position", columns="scoretype", values="weighted_energy", aggfunc="mean").fillna(0)
    pivot = pivot.loc[pivot.abs().sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(pivot)), 5))
    pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("LDA position")
    ax.set_ylabel("weighted energy (REU)")
    ax.set_title(title)
    ax.legend(title="scoretype", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_representative_stacked_bar(contacts: pd.DataFrame, complex_id: str | None = None):
    if complex_id is None:
        # highest total unfavorable contribution -> visually interesting
        complex_id = _complex_totals(contacts).idxmax()

    sub = contacts[contacts["complex_id"] == complex_id].dropna(subset=["position"])
    out = config.FIGURES_DIR / "representative_complex_stacked_bar.png"
    _draw_stacked_bar(sub, f"Score-term decomposition — {complex_id}", out)
    print(f"[plots] wrote {out} (complex_id={complex_id})")


def plot_representative_stacked_bar_per_protein(contacts: pd.DataFrame):
    """
    Same stacked-bar plot as plot_representative_stacked_bar, but one per
    protein (its own highest-total-unfavorable-contribution complex), filed
    under results/figures/representative_stacked_bars/{importer,non_importer}/.
    """
    mapped = contacts.dropna(subset=["position"])
    out_dir = config.FIGURES_DIR / "representative_stacked_bars"

    n_written = 0
    for protein, group in mapped.groupby("protein"):
        cls = group["class"].iloc[0]
        complex_id = _complex_totals(group).idxmax()
        sub = group[group["complex_id"] == complex_id]
        out = out_dir / cls / f"{protein}.png"
        _draw_stacked_bar(sub, f"Score-term decomposition — {protein} ({cls})\n{complex_id}", out)
        n_written += 1

    print(f"[plots] wrote {n_written} per-protein stacked bars under {out_dir}/"
          "{importer,non_importer}/")


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


def plot_residue_by_protein_heatmap_by_class(contacts: pd.DataFrame):
    """
    Same data as plot_residue_by_protein_heatmap, but rows are grouped by
    class (all importers, then all non_importers, alphabetical within each)
    with a colored sidebar + a divider line between the two blocks, so the
    importer/non_importer split is visually obvious at a glance.
    """
    mapped = contacts.dropna(subset=["position"]).drop_duplicates(["complex_id", "replica", "position"])
    per_protein = mapped.groupby(["protein", "position"])["twobody_total"].mean().reset_index()
    pivot = per_protein.pivot(index="protein", columns="position", values="twobody_total")

    protein_class = contacts.drop_duplicates("protein").set_index("protein")["class"]
    row_order = protein_class.reindex(pivot.index).sort_values().index  # "importer" < "non_importer"
    pivot = pivot.loc[row_order]
    classes = protein_class.loc[row_order]
    boundary = (classes == "importer").sum()  # row index where non_importer block starts

    fig, (ax_side, ax) = plt.subplots(
        1, 2, figsize=(max(10, 0.3 * pivot.shape[1]) + 1, max(6, 0.3 * pivot.shape[0])),
        gridspec_kw={"width_ratios": [0.03, 1]}, sharey=True,
    )
    sidebar = pd.DataFrame({"": [1 if c == "importer" else 0 for c in classes]}, index=row_order)
    sidebar_cmap = mcolors.ListedColormap([GROUP_COLOR["non_importer"], GROUP_COLOR["importer"]])
    sns.heatmap(
        sidebar, ax=ax_side, cmap=sidebar_cmap, vmin=0, vmax=1,
        cbar=False, xticklabels=False, yticklabels=True,
    )
    ax_side.set_ylabel("protein")
    ax_side.tick_params(left=False)

    sns.heatmap(pivot, cmap="RdBu_r", center=0, ax=ax, cbar_kws={"label": "mean two-body total (REU)"},
                yticklabels=False)
    ax.axhline(boundary, color="black", linewidth=2.5)
    ax.set_xlabel("LDA position")
    ax.set_ylabel("")
    ax.set_title("Ligand<->residue two-body total, mean per protein\n"
                  "(grouped by class — importer block above the line, non_importer below)")

    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR[c]) for c in ("importer", "non_importer")]
    ax.legend(handles, ("importer", "non_importer"), loc="upper left", bbox_to_anchor=(1.15, 1.0),
              title="class", frameon=False)

    fig.tight_layout()
    out = config.FIGURES_DIR / "residue_by_protein_heatmap_by_class.png"
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
    plot_representative_stacked_bar_per_protein(contacts)
    plot_residue_by_protein_heatmap(contacts)
    plot_residue_by_protein_heatmap_by_class(contacts)
    plot_lda_class_difference(overlay)


if __name__ == "__main__":
    main()
