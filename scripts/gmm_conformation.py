#!/usr/bin/env python3
"""
scripts/gmm_conformation.py
============================
Fits Gaussian Mixture Models (GMMs) to the distribution of MFS gate distances
per protein and assigns each sample to a conformational state.

Strategy
--------
Clustering is done independently per protein so the model is not constrained
by the assumption that all 6 canonical conformations were sampled.

Feature space (2D):
  • ext_gate_A — minimum Cα tip distance at the extracellular gate
                 (TM1/TM2 N-bundle tips  vs  TM7/TM8 C-bundle tips)
  • int_gate_A — minimum Cα tip distance at the intracellular gate
                 (TM4/TM5 N-bundle tips  vs  TM10/TM11 C-bundle tips)

Per-protein workflow:
  1. BIC sweep k=1..min(n//5, 6) — BIC naturally prevents overfitting.
  2. For the BIC-optimal k, sort clusters by (ext − int) score (descending).
  3. Assign states by rank:
       rank 0        → outward_open  (high ext, low int)
       rank k−1      → inward_open   (low ext, high int)
       middle ranks  → occluded      (both gates relatively closed)
     For k=1 (single dominant state), the sign of centroid (ext−int) is used.

A global reference GMM (k sweep to 20, all proteins pooled) is saved in
results/gmm/global/ for diagnostic purposes but is NOT used for reannotation.

Output columns added to angles_with_assignments.csv:
  • clade      — NPFx extracted from protein name
  • gmm_state  — outward_open | occluded | inward_open | unknown
  • gmm_k      — BIC-optimal k for this protein

Reference: Qureshi et al. (2020) Nature https://doi.org/10.1038/s41586-020-1963-z

Usage (called by Snakemake rule `gmm_analysis`):
    python scripts/gmm_conformation.py \\
        --input      results/tm_angles/*/*/angles.csv \\
        --output-dir results/gmm \\
        --sentinel   results/gmm/gmm.done
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from kneed import KneeLocator


# ── Constants ──────────────────────────────────────────────────────────────────

MIN_SAMPLES_PER_PROTEIN = 6   # below this we skip GMM and label as unknown
MAX_K_PER_PROTEIN       = 6   # biological cap: at most 3 states × 2 sub-states
MAX_K_GLOBAL            = 20  # full sweep for the reference global fit

STATE_COLORS = {
    "outward_open": "#1976D2",  # blue
    "occluded":     "#43A047",  # green
    "inward_open":  "#E53935",  # red
    "unknown":      "#9E9E9E",  # grey
}

CONFORMATION_COLORS = {
    "outward_open_apo":      "#1976D2",
    "outward_occluded_holo": "#42A5F5",
    "occluded_apo":          "#43A047",
    "occluded_holo":         "#A5D6A7",
    "inward_open_apo":       "#E53935",
    "inward_occluded_holo":  "#EF9A9A",
}


def extract_clade(protein: str) -> str:
    """NPF4.7_Q9FM20 → 'NPF4'."""
    m = re.match(r'^(NPF\d+)', protein)
    return m.group(1) if m else "unknown"


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",      required=True, nargs="+",
                   help="Per-conformation angle CSV files")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--sentinel",   required=True)
    p.add_argument("--n-init",     type=int, default=20,
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
    """Sort components by (ext_gate − int_gate) score, descending.
    Index 0 = outward (high ext, low int), last = inward (low ext, high int).
    """
    scores = gmm.means_[:, 0] - gmm.means_[:, 1]
    return np.argsort(-scores)


def find_best_k(bic_by_k: dict[int, float]) -> int:
    """Return the knee of the BIC curve (convex, decreasing).
    Falls back to the global minimum if no knee is detected.
    Requires ≥3 k values for KneeLocator to be meaningful.
    """
    ks   = sorted(bic_by_k)
    bics = [bic_by_k[k] for k in ks]

    if len(ks) >= 3:
        try:
            kl = KneeLocator(ks, bics, curve="convex", direction="decreasing")
            if kl.knee is not None:
                return int(kl.knee)
        except Exception as e:
            print(f"[gmm] WARNING: KneeLocator failed ({e}), falling back to BIC minimum")

    return ks[int(np.argmin(bics))]


def assign_states(gmm: GaussianMixture) -> dict[int, str]:
    """Map GMM component index → conformational state by rank of (ext − int).

    Rank 0 (highest ext−int) → outward_open
    Rank k−1                 → inward_open
    Middle ranks             → occluded
    k=1: use sign of centroid (ext−int) directly.
    """
    k     = gmm.n_components
    order = component_order(gmm)
    state_map: dict[int, str] = {}
    for rank, comp_idx in enumerate(order):
        if k == 1:
            diff = float(gmm.means_[comp_idx, 0] - gmm.means_[comp_idx, 1])
            state_map[comp_idx] = (
                "outward_open" if diff > 0 else
                "inward_open"  if diff < 0 else
                "occluded"
            )
        elif rank == 0:
            state_map[comp_idx] = "outward_open"
        elif rank == k - 1:
            state_map[comp_idx] = "inward_open"
        else:
            state_map[comp_idx] = "occluded"
    return state_map


# ── Plots ──────────────────────────────────────────────────────────────────────

def _draw_ellipse(ax, mean, cov, n_std: float = 1.5, **kwargs):
    vals, vecs = np.linalg.eigh(cov)
    idx = vals.argsort()[::-1]
    vals, vecs = vals[idx], vecs[:, idx]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h  = 2 * n_std * np.sqrt(np.abs(vals))
    ax.add_patch(Ellipse(xy=mean, width=w, height=h, angle=theta, **kwargs))


def plot_gate_scatter(X: np.ndarray, gmm: GaussianMixture, df: pd.DataFrame,
                      title: str, state_labels_by_rank: list[str],
                      bic: float, out_path: Path) -> None:
    """Scatter of (ext_gate, int_gate) coloured by template conformation.
    GMM ellipses + centroids coloured by assigned state.
    X and df must be row-aligned (same order, reset index).
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    for conf in sorted(df["conformation"].unique()):
        mask = (df["conformation"] == conf).values
        ax.scatter(X[mask, 0], X[mask, 1],
                   s=14, alpha=0.4, linewidths=0,
                   color=CONFORMATION_COLORS.get(conf, "#9E9E9E"),
                   label=conf)

    order = component_order(gmm)
    for rank, comp_idx in enumerate(order):
        mean  = gmm.means_[comp_idx]
        cov   = gmm.covariances_[comp_idx]
        w     = gmm.weights_[comp_idx]
        state = (state_labels_by_rank[rank]
                 if rank < len(state_labels_by_rank) else f"comp{rank}")
        color = STATE_COLORS.get(state, "#9E9E9E")
        _draw_ellipse(ax, mean, cov, n_std=1.5,
                      edgecolor=color, facecolor=color, alpha=0.15, lw=2.0)
        ax.scatter(*mean, marker="*", s=240, color=color, zorder=5,
                   label=f"{state}  w={w:.2f}  ext={mean[0]:.1f}Å  int={mean[1]:.1f}Å")

    ax.set_xlabel("Extracellular gate  (Å)", fontsize=12)
    ax.set_ylabel("Intracellular gate  (Å)", fontsize=12)
    ax.set_title(f"{title}   BIC = {bic:.1f}", fontsize=13)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[gmm] Plot: {out_path.name}")


def plot_state_scatter(X: np.ndarray, df: pd.DataFrame, out_path: Path) -> None:
    """Global scatter of all samples coloured by their assigned gmm_state."""
    fig, ax = plt.subplots(figsize=(10, 8))
    for state in ["outward_open", "occluded", "inward_open", "unknown"]:
        mask = (df["gmm_state"] == state).values
        if mask.any():
            ax.scatter(X[mask, 0], X[mask, 1],
                       s=14, alpha=0.5, linewidths=0,
                       color=STATE_COLORS[state], label=f"{state} (n={mask.sum()})")
    ax.set_xlabel("Extracellular gate  (Å)", fontsize=12)
    ax.set_ylabel("Intracellular gate  (Å)", fontsize=12)
    ax.set_title("All samples — per-protein GMM state assignment", fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[gmm] State scatter: {out_path.name}")


def plot_bic_curve(bic_by_k: dict[int, float], best_k: int, out_path: Path) -> None:
    ks   = sorted(bic_by_k)
    bics = [bic_by_k[k] for k in ks]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, bics, "o-", color="#1565C0", lw=2, ms=7, zorder=3)
    ax.axvline(best_k, color="#E53935", ls="--", lw=1.5,
               label=f"Best k = {best_k}  (BIC = {bic_by_k[best_k]:.1f})")
    ax.set_xlabel("Number of GMM components (k)", fontsize=12)
    ax.set_ylabel("BIC  (lower = better)", fontsize=12)
    ax.set_title(f"GMM model selection — BIC sweep k = 1 … {max(ks)}", fontsize=13)
    ax.set_xticks(ks)
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[gmm] BIC curve: {out_path.name}")


def plot_angle_histogram(df: pd.DataFrame, out_path: Path) -> None:
    angles = df["angle_deg"].dropna().values
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(angles, bins=40, color="#455A64", alpha=0.75, edgecolor="white", lw=0.4)
    ax.set_xlabel("TM2 / TM8 angle  (°)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"TM2/TM8 angle distribution  (n = {len(angles)})  [diagnostic]", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[gmm] Angle histogram: {out_path.name}")


def plot_gate_by_conformation(df: pd.DataFrame, out_path: Path) -> None:
    conformations = sorted(df["conformation"].unique())
    n = len(conformations)
    fig, axes = plt.subplots(1, 2, figsize=(max(8, n * 1.4), 5))
    rng = np.random.default_rng(0)
    for ax, col, ylabel in zip(
        axes,
        ["ext_gate_A", "int_gate_A"],
        ["Extracellular gate (Å)", "Intracellular gate (Å)"],
    ):
        for xi, conf in enumerate(conformations):
            sub    = df[df["conformation"] == conf][col].dropna().values
            jitter = rng.uniform(-0.15, 0.15, len(sub))
            ax.scatter(np.full(len(sub), xi) + jitter, sub,
                       color=CONFORMATION_COLORS.get(conf, "#9E9E9E"),
                       s=14, alpha=0.5, linewidths=0)
        ax.set_xticks(range(n))
        ax.set_xticklabels(conformations, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=11)
    plt.suptitle("Gate distances per template conformation", fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[gmm] Gate-by-conformation: {out_path.name}")


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

    df_valid = all_data[
        all_data["status"].str.startswith("ok") &
        all_data["ext_gate_A"].notna() &
        all_data["int_gate_A"].notna()
    ].copy().reset_index(drop=True)

    n_total   = len(all_data)
    n_no_gate = (all_data["status"].str.startswith("ok") &
                 (all_data["ext_gate_A"].isna() | all_data["int_gate_A"].isna())).sum()
    print(
        f"[gmm] Loaded {n_total} rows total; {len(df_valid)} with valid gate distances"
        + (f" ({n_no_gate} ok rows lack gate data — rerun compute_tm_angle)" if n_no_gate else "")
    )

    if len(df_valid) < MIN_SAMPLES_PER_PROTEIN:
        print(f"[gmm] ERROR: only {len(df_valid)} valid rows — need ≥{MIN_SAMPLES_PER_PROTEIN}")
        sys.exit(1)

    df_valid["clade"]     = df_valid["protein"].apply(extract_clade)
    df_valid["gmm_state"] = "unknown"
    df_valid["gmm_k"]     = 0

    # ── Per-protein GMM ───────────────────────────────────────────────────────
    proteins    = sorted(df_valid["protein"].unique())
    per_protein: dict = {}

    print(f"[gmm] {len(proteins)} proteins to process (per-protein BIC sweep k=1..{MAX_K_PER_PROTEIN})")

    for protein in proteins:
        mask    = df_valid["protein"] == protein
        df_prot = df_valid[mask].reset_index(drop=True)
        X_prot  = df_prot[["ext_gate_A", "int_gate_A"]].values
        n       = len(df_prot)

        if n < MIN_SAMPLES_PER_PROTEIN:
            print(f"[gmm]   {protein}: {n} samples — too few, skipping (marked unknown)")
            per_protein[protein] = {"n_samples": n, "status": "skipped", "best_k": 0}
            continue

        k_max = max(1, min(n // 5, MAX_K_PER_PROTEIN))

        bic_by_k: dict[int, float] = {}
        gmm_by_k: dict[int, GaussianMixture] = {}
        for k in range(1, k_max + 1):
            gm = fit_gmm(X_prot, n_components=k, n_init=args.n_init)
            bic_by_k[k] = round(gm.bic(X_prot), 3)
            gmm_by_k[k] = gm

        best_k    = find_best_k(bic_by_k)
        gmm_best  = gmm_by_k[best_k]
        state_map = assign_states(gmm_best)

        raw_preds = gmm_best.predict(X_prot)
        states    = [state_map[p] for p in raw_preds]

        df_valid.loc[mask, "gmm_state"] = states
        df_valid.loc[mask, "gmm_k"]     = best_k

        state_labels_by_rank = [state_map[i] for i in component_order(gmm_best)]
        state_counts = pd.Series(states).value_counts().to_dict()

        per_protein[protein] = {
            "n_samples":    n,
            "status":       "ok",
            "best_k":       best_k,
            "bic_by_k":     bic_by_k,
            "state_counts": state_counts,
            "clusters": [
                {
                    "state":             state_map[i],
                    "ext_gate_centroid": round(float(gmm_best.means_[i, 0]), 2),
                    "int_gate_centroid": round(float(gmm_best.means_[i, 1]), 2),
                    "weight":            round(float(gmm_best.weights_[i]),    4),
                }
                for i in component_order(gmm_best)
            ],
        }

        prot_dir = out_dir / "per_protein" / protein
        prot_dir.mkdir(parents=True, exist_ok=True)
        plot_bic_curve(bic_by_k, best_k, prot_dir / "bic_curve.png")
        plot_gate_scatter(X_prot, gmm_best, df_prot,
                          f"{protein}  GMM-{best_k}  (BIC-optimal)",
                          state_labels_by_rank, bic_by_k[best_k],
                          prot_dir / "gmm_best.png")

        sc_str = "  ".join(f"{s}:{c}" for s, c in sorted(state_counts.items()))
        print(f"[gmm]   {protein}: k={best_k}  {sc_str}")

    # ── Global reference GMM (diagnostic only) ────────────────────────────────
    X_all = df_valid[["ext_gate_A", "int_gate_A"]].values
    print(f"[gmm] Fitting global reference GMM (n={len(df_valid)}, k_max={MAX_K_GLOBAL}) ...")
    global_bic: dict[int, float] = {}
    global_gmm_by_k: dict[int, GaussianMixture] = {}
    for k in range(1, MAX_K_GLOBAL + 1):
        gm = fit_gmm(X_all, n_components=k, n_init=args.n_init)
        global_bic[k] = round(gm.bic(X_all), 3)
        global_gmm_by_k[k] = gm
    global_best_k = find_best_k(global_bic)
    gmm_global    = global_gmm_by_k[global_best_k]
    global_states = assign_states(gmm_global)
    global_labels = [global_states[i] for i in component_order(gmm_global)]

    global_dir = out_dir / "global"
    global_dir.mkdir(exist_ok=True)
    plot_bic_curve(global_bic, global_best_k, global_dir / "bic_curve.png")
    plot_gate_scatter(X_all, gmm_global, df_valid,
                      f"Global GMM-{global_best_k}  (reference only — not used for reannotation)",
                      global_labels, global_bic[global_best_k],
                      global_dir / "gmm_best.png")
    print(f"[gmm] Global reference: best_k={global_best_k}")

    # ── Summary plots ─────────────────────────────────────────────────────────
    plot_state_scatter(X_all, df_valid, out_dir / "state_scatter.png")
    plot_angle_histogram(df_valid, out_dir / "angle_histogram.png")
    plot_gate_by_conformation(df_valid, out_dir / "gate_by_conformation.png")

    # ── Save report + assignments ─────────────────────────────────────────────
    n_ok      = sum(1 for v in per_protein.values() if v["status"] == "ok")
    n_skipped = len(per_protein) - n_ok
    report    = {
        "n_valid_samples":   int(len(df_valid)),
        "n_invalid_samples": int(n_total - len(df_valid)),
        "n_proteins_fitted": n_ok,
        "n_proteins_skipped": n_skipped,
        "global_reference":  {
            "best_k":   global_best_k,
            "bic_by_k": global_bic,
        },
        "per_protein": per_protein,
    }
    (out_dir / "gmm_report.json").write_text(json.dumps(report, indent=2))
    df_valid.to_csv(out_dir / "angles_with_assignments.csv", index=False)
    print(f"[gmm] Report: {out_dir / 'gmm_report.json'}")

    # ── Overall state distribution ────────────────────────────────────────────
    state_dist = df_valid["gmm_state"].value_counts().to_dict()
    for state in ["outward_open", "occluded", "inward_open", "unknown"]:
        print(f"[gmm]   {state:15s}: {state_dist.get(state, 0):5d} samples")

    # ── Sentinel ──────────────────────────────────────────────────────────────
    Path(args.sentinel).write_text(
        f"GMM done. {n_ok}/{len(proteins)} proteins fitted. "
        f"Global ref best_k={global_best_k}. "
        f"States: {state_dist}.\n"
    )
    print(f"[gmm] Done. Sentinel: {args.sentinel}")


if __name__ == "__main__":
    main()
