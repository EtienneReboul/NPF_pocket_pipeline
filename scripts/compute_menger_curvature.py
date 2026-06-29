#!/usr/bin/env python3
"""
scripts/compute_menger_curvature.py
=====================================
Computes per-residue Menger curvature for one protein, pooling all Boltz-2
predicted structures across every conformation into a single pseudo-ensemble.

Background
----------
Menger curvature at position i (with spacing s) is the reciprocal radius of
the circumscribed circle through the Cα triplet (i, i+s, i+2s):

    κ(i) = 4·Area / (a·b·c)

where a, b, c are the three edge lengths (Heron's formula for the area).
Reference: Léger, Ann. Math. 1999, DOI 10.2307/121074

This is the same computation as menger.analysis.mengercurvature.menger_curvature
but implemented in pure NumPy so no numba/MDAnalysis dependency is needed.

Usage
-----
    python scripts/compute_menger_curvature.py \\
        --protein  NPF1.1_Q8LPL2 \\
        --boltz-dir results/boltz \\
        --output    results/menger_curvature/NPF1.1_Q8LPL2/curvature.csv \\
        --spacing   2

Outputs
-------
  <output>           — CSV with per-residue mean curvature and flexibility
  <output stem>.npy  — float32 array of shape (n_frames, n_positions)
                       stored alongside the CSV for downstream analysis
"""

import argparse
import csv
from pathlib import Path

import gemmi
import numpy as np


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--protein",   required=True, help="Protein identifier, e.g. NPF1.1_Q8LPL2")
    p.add_argument("--boltz-dir", required=True, help="Root boltz output dir (results/boltz)")
    p.add_argument("--output",    required=True, help="Output CSV path")
    p.add_argument("--spacing",   type=int, default=2,
                   help="Triplet spacing (recommended 2 for protein backbones)")
    return p.parse_args()


# ── Geometry ───────────────────────────────────────────────────────────────────

def menger_curvature_np(ca_coords: np.ndarray, spacing: int) -> np.ndarray:
    """Pure-NumPy Menger curvature for one structure.

    Returns an array of length n_ca - 2*spacing.  Position i corresponds to
    the triplet of Cα atoms at indices (i, i+spacing, i+2*spacing).
    """
    n = len(ca_coords)
    n_out = n - 2 * spacing
    if n_out <= 0:
        raise ValueError(f"Too few Cα atoms ({n}) for spacing={spacing}")

    v0 = ca_coords[:n_out]
    v1 = ca_coords[spacing: n_out + spacing]
    v2 = ca_coords[2 * spacing:]

    a = np.linalg.norm(v0 - v1, axis=1)
    b = np.linalg.norm(v1 - v2, axis=1)
    c = np.linalg.norm(v2 - v0, axis=1)

    # Heron's formula
    s = (a + b + c) / 2
    arg = s * (s - a) * (s - b) * (s - c)
    # clamp numerical noise below zero before sqrt
    area = np.sqrt(np.maximum(arg, 0.0))

    product = a * b * c
    # positions where the triangle is degenerate (collinear atoms) → curvature = 0
    safe = product > 0
    kappa = np.where(safe, 4 * area / product, 0.0)
    return kappa.astype(np.float32)


# ── Cα extraction ──────────────────────────────────────────────────────────────

def longest_chain_name(model: gemmi.Model) -> str:
    return max(model, key=lambda c: sum(1 for _ in c)).name


def extract_ca(cif_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (ca_coords [N,3], resids [N]) for the longest chain."""
    structure = gemmi.read_structure(cif_path)
    model = structure[0]
    chain_name = longest_chain_name(model)
    coords, resids = [], []
    for residue in model[chain_name]:
        for atom in residue:
            if atom.name == "CA":
                coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
                resids.append(residue.seqid.num)
    return np.array(coords, dtype=np.float32), np.array(resids, dtype=np.int32)


# ── CIF discovery ──────────────────────────────────────────────────────────────

def find_cifs(boltz_dir: str, protein: str) -> list[Path]:
    pattern = "*/boltz_out/*/predictions/*/target_model_*.cif"
    root = Path(boltz_dir) / protein
    cifs = sorted(root.glob(pattern))
    if not cifs:
        # fallback: older Boltz output layout (model_0.cif, model_1.cif …)
        cifs = sorted(root.glob("*/boltz_out/predictions/*/model_*.cif"))
    return cifs


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    cif_paths = find_cifs(args.boltz_dir, args.protein)
    if not cif_paths:
        raise FileNotFoundError(
            f"No CIF files found for protein {args.protein} under {args.boltz_dir}"
        )
    print(f"[menger] {args.protein}: found {len(cif_paths)} CIF files")

    # Determine reference residue count from the first CIF
    ref_coords, ref_resids = extract_ca(str(cif_paths[0]))
    n_ca_ref = len(ref_coords)
    n_out = n_ca_ref - 2 * args.spacing
    print(f"[menger] {args.protein}: {n_ca_ref} Cα → {n_out} curvature positions (spacing={args.spacing})")

    rows = []
    for cif_path in cif_paths:
        try:
            coords, _ = extract_ca(str(cif_path))
        except Exception as e:
            print(f"[menger] WARNING: could not parse {cif_path}: {e} — skipping")
            continue

        if len(coords) != n_ca_ref:
            print(
                f"[menger] WARNING: {cif_path.name} has {len(coords)} Cα "
                f"(expected {n_ca_ref}) — skipping"
            )
            continue

        rows.append(menger_curvature_np(coords, args.spacing))

    n_used = len(rows)
    if n_used == 0:
        raise RuntimeError(f"All CIF files were skipped for {args.protein}")

    valid_array = np.stack(rows)  # shape (n_used, n_out)
    local_curvatures   = np.mean(valid_array, axis=0)
    local_flexibilities = np.std(valid_array,  axis=0)

    print(
        f"[menger] {args.protein}: used {n_used}/{len(cif_paths)} frames, "
        f"mean curvature range [{local_curvatures.min():.4f}, {local_curvatures.max():.4f}]"
    )

    # ── Write outputs ──────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # CSV: per-position summary
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "protein", "position",
            "resid_left", "resid_center", "resid_right",
            "mean_curvature", "std_curvature",
            "n_frames",
        ])
        writer.writeheader()
        for i in range(n_out):
            writer.writerow({
                "protein":        args.protein,
                "position":       i,
                "resid_left":     int(ref_resids[i]),
                "resid_center":   int(ref_resids[i + args.spacing]),
                "resid_right":    int(ref_resids[i + 2 * args.spacing]),
                "mean_curvature": round(float(local_curvatures[i]), 6),
                "std_curvature":  round(float(local_flexibilities[i]), 6),
                "n_frames":       n_used,
            })
    print(f"[menger] CSV written → {out_path}")

    # NPY: full curvature_array (valid frames only)
    npy_path = out_path.with_suffix(".npy")
    np.save(npy_path, valid_array)
    print(f"[menger] curvature_array {valid_array.shape} saved → {npy_path}")


if __name__ == "__main__":
    main()
