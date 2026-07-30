#!/usr/bin/env python3
"""
src/vina_plots.py
====================
Violin + box plot of AutoDock Vina scores, importer vs. non_importer, from
vina_aggregate.py's results/vina/all_scores.csv. One plot per score column
(raw total, optimized total, raw ligand-receptor intermolecular term) —
violin shows the full distribution shape, inner box shows median/IQR, and
the title reports the Mann-Whitney U test between classes.
"""
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

import config

sns.set_theme(style="whitegrid")

GROUP_COLOR = {"importer": "#2ca02c", "non_importer": "#d62728"}
CLASS_ORDER = ["importer", "non_importer"]


def plot_violin_box(df: pd.DataFrame, column: str, ylabel: str, out_name: str) -> None:
    sub = df.dropna(subset=[column])

    fig, ax = plt.subplots(figsize=(6, 5.5))
    sns.violinplot(
        data=sub, x="class", y=column, order=CLASS_ORDER, hue="class", legend=False,
        palette=GROUP_COLOR, inner=None, cut=0, ax=ax, linewidth=1,
    )
    for patch in ax.collections:
        patch.set_alpha(0.6)
    sns.boxplot(
        data=sub, x="class", y=column, order=CLASS_ORDER,
        width=0.15, showcaps=True,
        boxprops={"zorder": 3, "facecolor": "white"},
        whiskerprops={"zorder": 3}, medianprops={"zorder": 3, "color": "black"},
        showfliers=False, ax=ax,
    )

    a = sub.loc[sub["class"] == "importer", column]
    b = sub.loc[sub["class"] == "non_importer", column]
    u, p = mannwhitneyu(a, b, alternative="two-sided")

    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{ylabel}\n"
        f"importer median={a.median():.2f} (n={len(a)})  "
        f"non_importer median={b.median():.2f} (n={len(b)})  "
        f"Mann-Whitney p={p:.3g}"
    )
    fig.tight_layout()
    out = config.VINA_FIGURES_DIR / out_name
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[vina_plots] wrote {out}")


def main():
    path = config.VINA_RESULTS_DIR / "all_scores.csv"
    if not path.exists():
        sys.exit(f"{path} not found — run vina_run_batch.py + vina_aggregate.py first.")
    df = pd.read_csv(path)

    plot_violin_box(df, "total", "Vina total score (kcal/mol)", "total_score_violin_box.png")
    plot_violin_box(df, "total_optimized", "Vina total score, optimized (kcal/mol)",
                     "total_score_optimized_violin_box.png")
    plot_violin_box(df, "lig_inter", "Vina ligand<->receptor intermolecular energy (kcal/mol)",
                     "lig_inter_violin_box.png")


if __name__ == "__main__":
    main()
