#!/usr/bin/env python3
"""
scripts/gmm_conformation.py
============================
Fits Gaussian Mixture Models (GMMs) to the distribution of TM2/TM8 inter-helix
angles across all Boltz-2 predictions and compares model complexity using the
Bayesian Information Criterion (BIC).

Models evaluated:
  • GMM sweep k=1–10: full BIC curve to identify optimal number of components
  • GMM-3: one Gaussian per canonical MFS conformation
           (inward-open / occluded / outward-open)
  • GMM-6: one Gaussian per sub-state, allowing apo/holo discrimination
           within each main conformation

BIC interpretation: lower is better.  ΔBIC = BIC(6) − BIC(3).
  •  ΔBIC < −10  → strong evidence the 6-component model is more informative
  • −10 < ΔBIC < 0 → weak evidence for GMM-6
  •  ΔBIC > 0   → GMM-3 is sufficient; extra complexity is not justified

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
matplotlib.use("Agg")          # non-interactive backend for headless runs
import matplotlib.pyplot as plt
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


def component_pdf(x_range: np.ndarray, mean: float, var: float, weight: float) -> np.ndarray:
    return weight * np.exp(-0.5 * (x_range - mean) ** 2 / var) / np.sqrt(2 * np.pi * var)


# ── Plots ──────────────────────────────────────────────────────────────────────

CONFORMATION_COLORS = {
    "outward_open_apo":     "#1976D2",
    "outward_occluded_holo": "#42A5F5",
    "occluded_apo":         "#43A047",
    "occluded_holo":        "#A5D6A7",
    "inward_open_apo":      "#E53935",
    "inward_occluded_holo": "#EF9A9A",
}

GMM3_LABELS  = ["Outward-open", "Occluded", "Inward-open"]
GMM6_LABELS  = [
    "Outward-open (apo)", "Outward-open (holo)",
    "Occluded (apo)",     "Occluded (holo)",
    "Inward-open (apo)",  "Inward-open (holo)",
]


def plot_gmm(X: np.ndarray, gmm: GaussianMixture, df_valid: pd.DataFrame,
             title: str, component_labels: list[str], bic: float,
             out_path: Path) -> None:
    x_range = np.linspace(X.min() - 5, X.max() + 5, 600)
    fig, ax = plt.subplots(figsize=(11, 5))

    # Histogram coloured by known conformation
    conformations = sorted(df_valid["conformation"].unique())
    for conf in conformations:
        angles = df_valid.loc[df_valid["conformation"] == conf, "angle_deg"].values
        color  = CONFORMATION_COLORS.get(conf, "#9E9E9E")
        ax.hist(angles, bins=25, density=True, alpha=0.35, color=color, label=conf)

    # Total GMM density
    log_prob = gmm.score_samples(x_range.reshape(-1, 1))
    ax.plot(x_range, np.exp(log_prob), "k--", lw=2, label="Total GMM", zorder=5)

    # Individual components sorted by mean
    order   = np.argsort(gmm.means_.flatten())
    colors  = plt.cm.tab10(np.linspace(0, 0.8, gmm.n_components))
    for rank, i in enumerate(order):
        mean   = float(gmm.means_[i, 0])
        var    = float(gmm.covariances_[i, 0, 0])
        weight = float(gmm.weights_[i])
        label  = component_labels[rank] if rank < len(component_labels) else f"C{rank+1}"
        pdf    = component_pdf(x_range, mean, var, weight)
        ax.fill_between(x_range, pdf, alpha=0.20, color=colors[rank])
        ax.plot(x_range, pdf, color=colors[rank], lw=1.8,
                label=f"{label}  μ={mean:.1f}°  w={weight:.2f}")

    ax.set_xlabel("TM2 / TM8 angle  (°)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"{title}   BIC = {bic:.1f}", fontsize=13)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[gmm] Plot saved: {out_path.name}")


def plot_bic_curve(bic_by_k: dict[int, float], best_k: int, out_path: Path) -> None:
    """Line plot of BIC for k=1..max_k with the elbow/minimum highlighted."""
    ks   = sorted(bic_by_k)
    bics = [bic_by_k[k] for k in ks]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, bics, "o-", color="#1565C0", lw=2, ms=7, zorder=3)
    ax.axvline(best_k, color="#E53935", ls="--", lw=1.5,
               label=f"Best k = {best_k}  (BIC = {bic_by_k[best_k]:.1f})")
    # annotate k=3 and k=6 for biological reference
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


def plot_angle_histogram(X: np.ndarray, out_path: Path) -> None:
    """Simple ungrouped histogram of all angles — shows overall density without labelling."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(X.flatten(), bins=40, color="#455A64", alpha=0.75, edgecolor="white", lw=0.4)
    ax.set_xlabel("TM2 / TM8 angle  (°)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Overall TM2/TM8 angle distribution  (n = {len(X)})", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[gmm] Overall histogram saved: {out_path.name}")


def plot_angle_by_conformation(df_valid: pd.DataFrame, gmm3: GaussianMixture,
                               out_path: Path) -> None:
    """Strip-plot of angles per conformation coloured by GMM-3 assignment."""
    conformations = sorted(df_valid["conformation"].unique())
    n = len(conformations)
    fig, ax = plt.subplots(figsize=(max(6, n * 1.4), 5))
    cmap = plt.cm.RdYlBu
    for xi, conf in enumerate(conformations):
        sub     = df_valid[df_valid["conformation"] == conf]
        angles  = sub["angle_deg"].values
        comps   = sub["gmm3_component"].values
        jitter  = np.random.default_rng(0).uniform(-0.15, 0.15, len(angles))
        scatter = ax.scatter(
            np.full(len(angles), xi) + jitter,
            angles,
            c=comps, cmap=cmap, vmin=0, vmax=2,
            s=20, alpha=0.6, linewidths=0,
        )
    plt.colorbar(scatter, ax=ax, label="GMM-3 component (0=outward, 1=occluded, 2=inward)")
    ax.set_xticks(range(n))
    ax.set_xticklabels(conformations, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("TM2 / TM8 angle  (°)", fontsize=11)
    ax.set_title("Angle distribution per template conformation", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[gmm] Per-conformation plot saved: {out_path.name}")


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

    all_data  = pd.concat(frames, ignore_index=True)
    df_valid  = all_data[
        (all_data["status"].str.startswith("ok")) &
        all_data["angle_deg"].notna()
    ].copy()

    print(
        f"[gmm] Loaded {len(all_data)} rows total; "
        f"{len(df_valid)} valid angle measurements"
    )

    if len(df_valid) < 12:
        print(
            f"[gmm] ERROR: only {len(df_valid)} valid angles — need ≥12 for reliable GMM fitting"
        )
        sys.exit(1)

    X = df_valid["angle_deg"].values.reshape(-1, 1)

    # ── BIC sweep k=1–10 ─────────────────────────────────────────────────────
    MAX_K = 10
    print(f"[gmm] BIC sweep k=1..{MAX_K} (n_init={args.n_init}) ...")
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
        interpretation = f"GMM-6 is strongly preferred (ΔBIC = {delta_bic:+.1f}): 6 sub-states better explain the data"
    elif delta_bic < 0:
        interpretation = f"Weak evidence for GMM-6 (ΔBIC = {delta_bic:+.1f}): 6 sub-states marginally better"
    else:
        interpretation = f"GMM-3 is sufficient (ΔBIC = {delta_bic:+.1f}): additional sub-states not justified"

    print(f"[gmm] BIC  GMM-3 = {bic3:.2f}")
    print(f"[gmm] BIC  GMM-6 = {bic6:.2f}")
    print(f"[gmm] {interpretation}")

    # ── Sort GMM components by mean for interpretable labelling ──────────────
    def _sorted_components(gmm):
        idx = np.argsort(gmm.means_.flatten())
        return {
            "means":       gmm.means_.flatten()[idx].tolist(),
            "variances":   gmm.covariances_.flatten()[idx].tolist(),
            "weights":     gmm.weights_[idx].tolist(),
            "converged":   bool(gmm.converged_),
        }

    report = {
        "n_valid_samples":      int(len(df_valid)),
        "n_invalid_samples":    int(len(all_data) - len(df_valid)),
        "angle_mean_deg":       float(df_valid["angle_deg"].mean()),
        "angle_std_deg":        float(df_valid["angle_deg"].std()),
        "angle_min_deg":        float(df_valid["angle_deg"].min()),
        "angle_max_deg":        float(df_valid["angle_deg"].max()),
        "bic_by_k":             bic_by_k,
        "best_k_by_bic":        best_k,
        "gmm3": {"bic": round(bic3, 3), **_sorted_components(gmm3)},
        "gmm6": {"bic": round(bic6, 3), **_sorted_components(gmm6)},
        "delta_bic_6_minus_3":  round(delta_bic, 3),
        "preferred_model":      "GMM-6" if delta_bic < 0 else "GMM-3",
        "interpretation":       interpretation,
    }

    (out_dir / "gmm_report.json").write_text(json.dumps(report, indent=2))
    print(f"[gmm] Report: {out_dir / 'gmm_report.json'}")

    # ── Per-sample assignments ────────────────────────────────────────────────
    df_valid["gmm3_component"] = gmm3.predict(X)
    df_valid["gmm6_component"] = gmm6.predict(X)
    # Re-label components by ascending mean so 0 = lowest angle
    for col, gmm in [("gmm3_component", gmm3), ("gmm6_component", gmm6)]:
        rank_map = {old: new for new, old in enumerate(np.argsort(gmm.means_.flatten()))}
        df_valid[col] = df_valid[col].map(rank_map)
    df_valid.to_csv(out_dir / "angles_with_assignments.csv", index=False)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_angle_histogram(X, out_dir / "angle_histogram.png")
    plot_bic_curve(bic_by_k, best_k, out_dir / "bic_curve.png")
    plot_gmm(X, gmm3, df_valid, "GMM-3  (inward / occluded / outward)",
             GMM3_LABELS, bic3, out_dir / "gmm3.png")
    plot_gmm(X, gmm6, df_valid, "GMM-6  (6 sub-states)",
             GMM6_LABELS, bic6, out_dir / "gmm6.png")
    plot_angle_by_conformation(df_valid, gmm3, out_dir / "angle_by_conformation.png")

    # ── Sentinel ──────────────────────────────────────────────────────────────
    Path(args.sentinel).write_text(
        f"GMM done. Best k={best_k} (BIC sweep). "
        f"BIC(3)={bic3:.1f}, BIC(6)={bic6:.1f}, ΔBIC={delta_bic:+.1f}. "
        f"Preferred biological model: {report['preferred_model']}.\n"
    )
    print(f"[gmm] Done. Sentinel: {args.sentinel}")


if __name__ == "__main__":
    main()
