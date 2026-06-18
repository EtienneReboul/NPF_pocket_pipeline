#!/usr/bin/env python3
"""
scripts/reannotate_conformations.py
=====================================
Reorganise Boltz-2 CIF structures into the 6 canonical NPF conformation folders
using GMM-assigned structural state + original apo/holo label from the template.

Folder assignment rules
-----------------------
best_k == 3  (or other k as fallback):
    GMM-3 component (sorted ascending by angle) → structural state:
        0 → outward   1 → occluded   2 → inward
    Original conformation name → ligand state:
        contains "apo"  → apo forms  (outward_open_apo, occluded_apo, inward_open_apo)
        contains "holo" → holo forms (outward_occluded_holo, occluded_holo, inward_occluded_holo)

best_k == 6:
    GMM-6 components directly, sorted ascending by mean angle:
        0 → outward_open_apo      1 → outward_occluded_holo
        2 → occluded_apo          3 → occluded_holo
        4 → inward_open_apo       5 → inward_occluded_holo

Output
------
results/reannotated/
    <protein>/
        outward_open_apo/       }
        outward_occluded_holo/  }  symlinks: <orig_conformation>__<sample_id>.cif -> raw CIF
        occluded_apo/           }
        occluded_holo/          }
        inward_open_apo/        }
        inward_occluded_holo/   }
    reannotation_summary.csv
    reannotation.done

Usage (called by Snakemake rule reannotate_conformations):
    python scripts/reannotate_conformations.py \\
        --assignments  results/gmm/angles_with_assignments.csv \\
        --report       results/gmm/gmm_report.json \\
        --boltz-out    results/boltz \\
        --output-dir   results/reannotated \\
        --sentinel     results/reannotated/reannotation.done
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd


def extract_clade(protein: str) -> str:
    """NPF4.7_Q9FM20 → 'NPF4'."""
    m = re.match(r'^(NPF\d+)', protein)
    return m.group(1) if m else "unknown"


# ── Canonical conformation mapping ────────────────────────────────────────────

CANONICAL_FOLDERS = [
    "outward_open_apo",
    "outward_occluded_holo",
    "occluded_apo",
    "occluded_holo",
    "inward_open_apo",
    "inward_occluded_holo",
]

# GMM-3 component (0=outward, 1=occluded, 2=inward) × ligand state → folder
GMM3_TO_FOLDER = {
    (0, "apo"):  "outward_open_apo",
    (0, "holo"): "outward_occluded_holo",
    (1, "apo"):  "occluded_apo",
    (1, "holo"): "occluded_holo",
    (2, "apo"):  "inward_open_apo",
    (2, "holo"): "inward_occluded_holo",
}

# GMM-6 component (0–5 sorted ascending by mean angle) → folder
GMM6_TO_FOLDER = {
    0: "outward_open_apo",
    1: "outward_occluded_holo",
    2: "occluded_apo",
    3: "occluded_holo",
    4: "inward_open_apo",
    5: "inward_occluded_holo",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--assignments", required=True)
    p.add_argument("--report",      required=True)
    p.add_argument("--boltz-out",   required=True)
    p.add_argument("--output-dir",  required=True)
    p.add_argument("--sentinel",    required=True)
    return p.parse_args()


def find_cif(boltz_root: Path, protein: str, conformation: str, sample_id: str) -> Path | None:
    run_root = boltz_root / protein / conformation / "boltz_out"
    if not run_root.exists():
        return None
    for cif in run_root.glob("**/predictions/**/*.cif"):
        if cif.stem == sample_id:
            return cif
    return None


def ligand_state(conformation: str) -> str:
    if "apo" in conformation:
        return "apo"
    if "holo" in conformation:
        return "holo"
    return "unknown"


def assign_folder(row: pd.Series, best_k: int) -> str:
    comp3 = int(row["gmm3_component"])
    lig   = ligand_state(row["conformation"])

    if best_k == 6:
        comp_best = int(row["gmm_best_component"])
        folder = GMM6_TO_FOLDER.get(comp_best)
        if folder is not None:
            return folder
        # GMM-6 component out of range → fall back to GMM-3
        print(f"[reannotate] WARNING: gmm6 component {comp_best} out of range, "
              f"falling back to gmm3 for {row['protein']}/{row['sample_id']}")

    # GMM-3 (or fallback from GMM-6 out-of-range)
    key    = (comp3, lig)
    folder = GMM3_TO_FOLDER.get(key)
    if folder is None:
        return f"component_{comp3}_{lig}"
    return folder


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args     = parse_args()
    out_dir  = Path(args.output_dir)
    boltz    = Path(args.boltz_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load GMM report ───────────────────────────────────────────────────────
    report      = json.loads(Path(args.report).read_text())
    per_clade   = report.get("per_clade", {})
    global_best_k = report.get("global_best_k", 3)
    print(f"[reannotate] Per-clade GMM report loaded. "
          f"{len(per_clade)} clades, global fallback best_k={global_best_k}")

    # ── Canonical folders are created per-protein below ──────────────────────

    # ── Load assignments ──────────────────────────────────────────────────────
    df = pd.read_csv(args.assignments)
    required = {"protein", "conformation", "sample_id", "angle_deg",
                "gmm3_component", "gmm_best_component"}
    missing = required - set(df.columns)
    if missing:
        print(f"[reannotate] ERROR: missing columns in assignments CSV: {missing}")
        sys.exit(1)
    print(f"[reannotate] {len(df)} samples with valid angle assignments")

    # ── Create symlinks ───────────────────────────────────────────────────────
    rows      = []
    n_ok      = 0
    n_missing = 0

    for _, row in df.iterrows():
        protein      = row["protein"]
        conformation = row["conformation"]
        sample_id    = row["sample_id"]

        cif_abs = find_cif(boltz, protein, conformation, sample_id)
        if cif_abs is None:
            print(f"[reannotate] WARNING: CIF not found for "
                  f"{protein}/{conformation}/{sample_id} — skipping")
            n_missing += 1
            continue

        clade   = extract_clade(protein)
        best_k  = per_clade.get(clade, {}).get("best_k_by_bic", global_best_k)
        folder  = assign_folder(row, best_k)
        dest_dir  = out_dir / protein / folder
        dest_dir.mkdir(parents=True, exist_ok=True)
        link_name = f"{conformation}__{sample_id}.cif"
        link_path = dest_dir / link_name
        rel_cif   = os.path.relpath(cif_abs, start=dest_dir)

        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(rel_cif)

        rows.append({
            "protein":             protein,
            "original_conformation": conformation,
            "sample_id":           sample_id,
            "angle_deg":           float(row["angle_deg"]),
            "gmm3_component":      int(row["gmm3_component"]),
            "gmm_best_component":  int(row["gmm_best_component"]),
            "assigned_conformation": folder,
        })
        n_ok += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "reannotation_summary.csv", index=False)

    counts = summary["assigned_conformation"].value_counts().to_dict()
    print(f"[reannotate] {n_ok} symlinks created across "
          f"{summary['protein'].nunique()} proteins, {n_missing} CIFs not found")
    for folder in CANONICAL_FOLDERS:
        print(f"  {folder:30s}  {counts.get(folder, 0):4d} structures")

    if per_clade:
        print("[reannotate] Per-clade best_k:")
        for clade_name in sorted(per_clade):
            ck = per_clade[clade_name].get("best_k_by_bic", global_best_k)
            st = per_clade[clade_name].get("status", "?")
            print(f"  {clade_name:12s}  best_k={ck}  ({st})")

    if n_missing > 0 and n_ok == 0:
        print("[reannotate] ERROR: no CIFs found — check --boltz-out path")
        sys.exit(1)

    # ── Sentinel ──────────────────────────────────────────────────────────────
    clade_summary = ", ".join(
        f"{c}:k={v.get('best_k_by_bic', global_best_k)}"
        for c, v in sorted(per_clade.items())
    )
    Path(args.sentinel).write_text(
        f"Reannotation done (per-clade GMM). "
        f"{n_ok} symlinks, {n_missing} missing. "
        f"Clades: {clade_summary}\n"
    )
    print(f"[reannotate] Done. Sentinel: {args.sentinel}")


if __name__ == "__main__":
    main()
