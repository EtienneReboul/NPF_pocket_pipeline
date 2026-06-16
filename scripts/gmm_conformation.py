#!/usr/bin/env python3
"""
scripts/gmm_conformation.py
============================
Fits Gaussian Mixture Models (GMMs) to the distribution of MFS gate distances
across all Boltz-2 predictions and compares model complexity using BIC.

Feature space (2D):
  • ext_gate_A — minimum Cα tip distance at the extracellular gate
                 (TM1/TM2 N-bundle tips  vs  TM7/TM8 C-bundle tips)
  • int_gate_A — minimum Cα tip distance at the intracellular gate
                 (TM4/TM5 N-bundle tips  vs  TM10/TM11 C-bundle tips)

These two distances directly encode the alternating-access state:
  outward_open  → high ext_gate, low int_gate
  inward_open   → low  ext_gate, high int_gate
  occluded      → low  ext_gate, low  int_gate

TM2/TM8 angle (Qureshi 2020) is retained as a diagnostic column.

Models evaluated:
  • GMM sweep k=1–10: BIC curve to identify optimal number of components
  • GMM-3: one Gaussian per canonical MFS conformation
  • GMM-6: one per sub-state (apo/holo within each main conformation)

Reference: Qureshi et al. (2020) Nature https://doi.org/10.1038/s41586-020-1963-z

Usage (called by Snakemake rule `gmm_analysis`):
    python scripts/gmm_conformation.py \\
        --input   results/tm_angles/*/*/angles.csv \\
        --output-dir results/gmm \\
        --sentinel   results/gmm/gmm.done
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",       required=True, nargs="+",
                   help="Per-conformation angle CSV files (results/tm_angles/**/angles.csv)")
    p.add_argument("--output-dir",  required=True)
    p.add_argument("--sentinel",    required=True)
    p.add_argument("--n-init",      type=int, default=20,
                   help="GMM restarts for robustness (default: 20)")
    return p.parse_args()


# ── GMM helpers ────────────────────────────────────────────────────────────────

def fit_gmm(X: np.ndarray, n_components: int, n_init: int) -> GaussianMixture:
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        n_init=n_init,
        random_state=42,
        max_iter=500,
    )
    gmm.fit(X)
    if not gmm.converged_:
        print(f"[gmm] WARNING: GMM-{n_components} did not converge")
    return gmm


def component_order(gmm: GaussianMixture) -> np.ndarray:
    """Sort components by ext_gate_mean − int_gate_mean (descending).
    Result: index 0 = outward (high ext, low int), last = inward (low ext, high int)."""
    scores = gmm.means_[:, 0] - gmm.means_[:, 1]
    return np.argsort(-scores)


# ── Colours ────────────────────────────────────────────────────────────────────

CONFORMATION_COLORS = {
    "outward_open_apo":      "#1976D2",
    "outward_occluded_holo": "#42A5F5",
    "occluded_apo":          "#43A047",
    "occluded_holo":         "#A5D6A7",
    "inward_open_apo":       "#E53935",
    "inward_occluded_holo":  "#EF9A9A",
}

GMM3_LABELS = ["Outward-open", "Occluded", "Inward-open"]
GMM6_LABELS = [
    "Outward-open (apo)", "Outward-open (holo)",
    "Occluded (apo)",     "Occluded (holo)",
    "Inward-open (apo)",  "Inward-open (holo)",
]


# ── Plots ──────────────────────────────────────────────────────────────────────

def _draw_ellipse(ax, mean, cov, n_std: float = 1.5, **kwargs):
    vals, vecs = np.linalg.eigh(cov)
    idx = vals.argsort()[::-1]
    vals, vecs = vals[idx], vecs[:, idx]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(np.abs(vals))
    ax.add_patch(Ellipse(xy=mean, width=w, height=h, angle=theta, **kwargs))


def plot_gate_scatter(X: np.ndarray, gmm: GaussianMixture, df: pd.DataFrame,
                      title: str, component_labels: list[str], bic: float,
                      out_path: Path) -> None:
    """2-D scatter of (ext_gate, int_gate) coloured by input conformation, with GMM ellipses."""
    fig, ax = plt.subplots(figsize=(10, 8))

    conformations = sorted(df["conformation"].unique())
    for conf in conformations:
        mask = df["conformation"] == conf
        ax.scatter(
            X[mask.values, 0], X[mask.values, 1],
            s=14, alpha=0.4, linewidths=0,
            color=CONFORMATION_COLORS.get(conf, "#9E9E9E"),
            label=conf,
        )

    order = component_order(gmm)
    cmap  = plt.cm.tab10(np.linspace(0, 0.8, gmm.n_components))
    for rank, i in enumerate(order):
        mean   = gmm.means_[i]
        cov    = gmm.covariances_[i]
        weight = gmm.weights_[i]
        label  = component_labels[rank] if rank < len(component_labels) else f"C{rank+1}"
        color  = cmap[rank]
        _draw_ellipse(ax, mean, cov, n_std=1.5,
                      edgecolor=color, facecolor=color, alpha=0.12, lw=1.5)
        ax.scatter(*mean, marker="*", s=220, color=color, zorder=5,
                   label=f"{label}  w={weight:.2f}  "
                         f"ext={mean[0]:.1f}Å  int={mean[1]:.1f}Å")

    ax.set_xlabel("Extracellular gate distance  (Å)", fontsize=12)
    ax.set_ylabel("Intracellular gate distance  (Å)", fontsize=12)
    ax.set_title(f"{title}   BIC = {bic:.1f}", fontsize=13)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[gmm] Plot saved: {out_path.name}")


def plot_bic_curve(bic_by_k: dict[int, float], best_k: int, out_path: Path) -> None:
    ks   = sorted(bic_by_k)
    bics = [bic_by_k[k] for k in ks]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, bics, "o-", color="#1565C0", lw=2, ms=7, zorder=3)
    ax.axvline(best_k, color="#E53935", ls="--", lw=1.5,
               label=f"Best k = {best_k}  (BIC = {bic_by_k[best_k]:.1f})")
    for k_ref in (3, 6):
        if k_ref in bic_by_k:
            ax.scatter([k_ref], [bic_by_k[k_ref]], color="#FF6F00", zorder=4, s=60)
            ax.annotate(f"k={k_ref}", xy=(k_ref, bic_by_k[k_ref]),
                        xytext=(k_ref + 0.15, bic_by_k[k_ref]),
                        fontsize=8, color="#FF6F00", va="center")
    ax.set_xlabel("Number of GMM components (k)", fontsize=12)
    ax.set_ylabel("BIC  (lower = better)", fontsize=12)
    ax.set_title("GMM model selection  —  BIC sweep k = 1 … 10", fontsize=13)
    ax.set_xticks(ks)
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[gmm] BIC curve saved: {out_path.name}")


def plot_angle_histogram(df: pd.DataFrame, out_path: Path) -> None:
    """Ungrouped TM2/TM8 angle distribution — diagnostic."""
    angles = df["angle_deg"].dropna().values
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(angles, bins=40, color="#455A64", alpha=0.75, edgecolor="white", lw=0.4)
    ax.set_xlabel("TM2 / TM8 angle  (°)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"TM2/TM8 angle distribution  (n = {len(angles)})  [diagnostic]", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[gmm] Angle histogram saved: {out_path.name}")


def plot_gate_by_conformation(df: pd.DataFrame, out_path: Path) -> None:
    """Strip-plot of ext_gate and int_gate per template conformation."""
    conformations = sorted(df["conformation"].unique())
    n = len(conformations)
    fig, axes = plt.subplots(1, 2, figsize=(max(8, n * 1.4), 5), sharey=False)
    rng = np.random.default_rng(0)

    for ax, col, ylabel in zip(
        axes,
        ["ext_gate_A", "int_gate_A"],
        ["Extracellular gate (Å)", "Intracellular gate (Å)"],
    ):
        for xi, conf in enumerate(conformations):
            sub = df[df["conformation"] == conf][col].dropna().values
            jitter = rng.uniform(-0.15, 0.15, len(sub))
            ax.scatter(
                np.full(len(sub), xi) + jitter, sub,
                color=CONFORMATION_COLORS.get(conf, "#9E9E9E"),
                s=14, alpha=0.5, linewidths=0,
            )
        ax.set_xticks(range(n))
        ax.set_xticklabels(conformations, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=11)

    plt.suptitle("Gate distances per template conformation", fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[gmm] Gate-by-conformation plot saved: {out_path.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load all angle CSVs ───────────────────────────────────────────────────
    frames = []
    for path in args.input:
        try:
            frames.append(pd.read_csv(path))
        except Exception as e:
            print(f"[gmm] WARNING: could not read {path}: {e}")

    if not frames:
        print("[gmm] ERROR: no valid input files — aborting")
        sys.exit(1)

    all_data = pd.concat(frames, ignore_index=True)

    # Rows with status "ok*" AND both gate distances present
    df_valid = all_data[
        all_data["status"].str.startswith("ok") &
        all_data["ext_gate_A"].notna() &
        all_data["int_gate_A"].notna()
    ].copy()

    n_total   = len(all_data)
    n_no_gate = (all_data["status"].str.startswith("ok") &
                 (all_data["ext_gate_A"].isna() | all_data["int_gate_A"].isna())).sum()

    print(
        f"[gmm] Loaded {n_total} rows total; "
        f"{len(df_valid)} with valid gate distances"
        + (f" ({n_no_gate} ok rows lack gate data — rerun compute_tm_angle)" if n_no_gate else "")
    )

    if len(df_valid) < 12:
        print(f"[gmm] ERROR: only {len(df_valid)} valid rows — need ≥12 for GMM fitting")
        sys.exit(1)

    X = df_valid[["ext_gate_A", "int_gate_A"]].values   # (n, 2)

    # ── BIC sweep k=1–10 ─────────────────────────────────────────────────────
    MAX_K = 10
    print(f"[gmm] BIC sweep k=1..{MAX_K} on 2-D gate distances (n_init={args.n_init}) ...")
    bic_by_k: dict[int, float] = {}
    gmm_by_k: dict[int, GaussianMixture] = {}
    for k in range(1, MAX_K + 1):
        gmm_k = fit_gmm(X, n_components=k, n_init=args.n_init)
        bic_by_k[k] = round(gmm_k.bic(X), 3)
        gmm_by_k[k] = gmm_k
        print(f"[gmm]   k={k:2d}  BIC = {bic_by_k[k]:.1f}")
    best_k = min(bic_by_k, key=bic_by_k.__getitem__)
    print(f"[gmm] Best k by BIC = {best_k}")

    gmm3 = gmm_by_k[3]
    gmm6 = gmm_by_k[6]
    bic3 = bic_by_k[3]
    bic6 = bic_by_k[6]
    delta_bic = bic6 - bic3

    if delta_bic < -10:
        interpretation = (f"GMM-6 is strongly preferred (ΔBIC = {delta_bic:+.1f}): "
                          "6 sub-states better explain the gate-distance distribution")
    elif delta_bic < 0:
        interpretation = (f"Weak evidence for GMM-6 (ΔBIC = {delta_bic:+.1f}): "
                          "6 sub-states marginally better")
    else:
        interpretation = (f"GMM-3 is sufficient (ΔBIC = {delta_bic:+.1f}): "
                          "additional sub-states not justified")

    print(f"[gmm] BIC  GMM-3 = {bic3:.2f}")
    print(f"[gmm] BIC  GMM-6 = {bic6:.2f}")
    print(f"[gmm] {interpretation}")

    # ── Build report ─────────────────────────────────────────────────────────
    def _sorted_components(gmm):
        idx = component_order(gmm)
        return {
            "means_ext_gate": gmm.means_[idx, 0].round(2).tolist(),
            "means_int_gate": gmm.means_[idx, 1].round(2).tolist(),
            "weights":        gmm.weights_[idx].round(4).tolist(),
            "converged":      bool(gmm.converged_),
        }

    report = {
        "n_valid_samples":       int(len(df_valid)),
        "n_invalid_samples":     int(n_total - len(df_valid)),
        "ext_gate_mean_A":       round(float(df_valid["ext_gate_A"].mean()), 2),
        "ext_gate_std_A":        round(float(df_valid["ext_gate_A"].std()),  2),
        "int_gate_mean_A":       round(float(df_valid["int_gate_A"].mean()), 2),
        "int_gate_std_A":        round(float(df_valid["int_gate_A"].std()),  2),
        "angle_mean_deg":        round(float(df_valid["angle_deg"].mean()),  2) if "angle_deg" in df_valid else None,
        "bic_by_k":              bic_by_k,
        "best_k_by_bic":         best_k,
        "gmm3":                  {"bic": round(bic3, 3), **_sorted_components(gmm3)},
        "gmm6":                  {"bic": round(bic6, 3), **_sorted_components(gmm6)},
        "delta_bic_6_minus_3":   round(delta_bic, 3),
        "preferred_model":       "GMM-6" if delta_bic < 0 else "GMM-3",
        "interpretation":        interpretation,
    }
    (out_dir / "gmm_report.json").write_text(json.dumps(report, indent=2))
    print(f"[gmm] Report: {out_dir / 'gmm_report.json'}")

    # ── Per-sample assignments ────────────────────────────────────────────────
    # Components are sorted: 0 = outward (high ext − int), last = inward
    def _rank_map(gmm):
        order = component_order(gmm)
        return {old: new for new, old in enumerate(order)}

    df_valid["gmm3_component"]    = gmm3.predict(X)
    df_valid["gmm6_component"]    = gmm6.predict(X)
    df_valid["gmm_best_component"] = gmm_by_k[best_k].predict(X)

    for col, gmm in [
        ("gmm3_component",     gmm3),
        ("gmm6_component",     gmm6),
        ("gmm_best_component", gmm_by_k[best_k]),
    ]:
        rm = _rank_map(gmm)
        df_valid[col] = df_valid[col].map(rm)

    df_valid.to_csv(out_dir / "angles_with_assignments.csv", index=False)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_bic_curve(bic_by_k, best_k, out_dir / "bic_curve.png")
    plot_gate_scatter(X, gmm3, df_valid,
                      "GMM-3  (outward / occluded / inward)",
                      GMM3_LABELS, bic3, out_dir / "gmm3.png")
    plot_gate_scatter(X, gmm6, df_valid,
                      "GMM-6  (6 sub-states)",
                      GMM6_LABELS, bic6, out_dir / "gmm6.png")
    plot_gate_scatter(X, gmm_by_k[best_k], df_valid,
                      f"GMM-{best_k}  (BIC-optimal)",
                      GMM3_LABELS if best_k == 3 else GMM6_LABELS, bic_by_k[best_k],
                      out_dir / "gmm_best.png")
    plot_angle_histogram(df_valid, out_dir / "angle_histogram.png")
    plot_gate_by_conformation(df_valid, out_dir / "gate_by_conformation.png")

    # ── Sentinel ──────────────────────────────────────────────────────────────
    Path(args.sentinel).write_text(
        f"GMM done. Best k={best_k} (BIC sweep on 2-D gate distances). "
        f"BIC(3)={bic3:.1f}, BIC(6)={bic6:.1f}, ΔBIC={delta_bic:+.1f}. "
        f"Preferred biological model: {report['preferred_model']}.\n"
    )
    print(f"[gmm] Done. Sentinel: {args.sentinel}")


if __name__ == "__main__":
    main()
