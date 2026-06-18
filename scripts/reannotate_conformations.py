#!/usr/bin/env python3
"""
scripts/reannotate_conformations.py
=====================================
Reorganise Boltz-2 CIF structures into the 6 canonical NPF conformation folders
using the per-protein GMM state assignment + the apo/holo label from the template.

Folder assignment
-----------------
GMM state (from angles_with_assignments.csv)  ×  ligand state (from template name):

    outward_open  × apo  → outward_open_apo
    outward_open  × holo → outward_occluded_holo
    occluded      × apo  → occluded_apo
    occluded      × holo → occluded_holo
    inward_open   × apo  → inward_open_apo
    inward_open   × holo → inward_occluded_holo

Samples with gmm_state == "unknown" are placed in an "unclassified/" folder.

Output
------
results/reannotated/
    <protein>/
        outward_open_apo/       }
        outward_occluded_holo/  }  symlinks: <orig_conformation>__<sample_id>.cif
        occluded_apo/           }
        occluded_holo/          }
        inward_open_apo/        }
        inward_occluded_holo/   }
        unclassified/           }  (gmm_state == unknown)
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
import sys
from pathlib import Path

import pandas as pd


# ── Folder mapping ────────────────────────────────────────────────────────────

CANONICAL_FOLDERS = [
    "outward_open_apo",
    "outward_occluded_holo",
    "occluded_apo",
    "occluded_holo",
    "inward_open_apo",
    "inward_occluded_holo",
]

STATE_LIGAND_TO_FOLDER = {
    ("outward_open", "apo"):  "outward_open_apo",
    ("outward_open", "holo"): "outward_occluded_holo",
    ("occluded",     "apo"):  "occluded_apo",
    ("occluded",     "holo"): "occluded_holo",
    ("inward_open",  "apo"):  "inward_open_apo",
    ("inward_open",  "holo"): "inward_occluded_holo",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--assignments", required=True)
    p.add_argument("--report",      required=True,
                   help="GMM report JSON (used for logging only)")
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
    if "apo"  in conformation:
        return "apo"
    if "holo" in conformation:
        return "holo"
    return "unknown"


def assign_folder(gmm_state: str, conformation: str) -> str:
    lig = ligand_state(conformation)
    return STATE_LIGAND_TO_FOLDER.get((gmm_state, lig), "unclassified")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    out_dir = Path(args.output_dir)
    boltz   = Path(args.boltz_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load GMM report (for logging) ─────────────────────────────────────────
    report = json.loads(Path(args.report).read_text())
    n_fitted  = report.get("n_proteins_fitted",  "?")
    n_skipped = report.get("n_proteins_skipped", "?")
    print(f"[reannotate] GMM report: {n_fitted} proteins fitted, {n_skipped} skipped")

    # ── Load assignments ──────────────────────────────────────────────────────
    df = pd.read_csv(args.assignments)
    required = {"protein", "conformation", "sample_id", "gmm_state"}
    missing  = required - set(df.columns)
    if missing:
        print(f"[reannotate] ERROR: missing columns in assignments CSV: {missing}")
        sys.exit(1)
    print(f"[reannotate] {len(df)} samples loaded")

    # ── Create symlinks ───────────────────────────────────────────────────────
    rows      = []
    n_ok      = 0
    n_missing = 0

    for _, row in df.iterrows():
        protein      = row["protein"]
        conformation = row["conformation"]
        sample_id    = row["sample_id"]
        gmm_state    = row["gmm_state"]

        cif_abs = find_cif(boltz, protein, conformation, sample_id)
        if cif_abs is None:
            print(f"[reannotate] WARNING: CIF not found for "
                  f"{protein}/{conformation}/{sample_id} — skipping")
            n_missing += 1
            continue

        folder   = assign_folder(gmm_state, conformation)
        dest_dir = out_dir / protein / folder
        dest_dir.mkdir(parents=True, exist_ok=True)

        link_name = f"{conformation}__{sample_id}.cif"
        link_path = dest_dir / link_name
        rel_cif   = os.path.relpath(cif_abs, start=dest_dir)

        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(rel_cif)

        rows.append({
            "protein":               protein,
            "original_conformation": conformation,
            "sample_id":             sample_id,
            "gmm_state":             gmm_state,
            "assigned_conformation": folder,
        })
        n_ok += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "reannotation_summary.csv", index=False)

    counts = summary["assigned_conformation"].value_counts().to_dict()
    print(f"[reannotate] {n_ok} symlinks across "
          f"{summary['protein'].nunique()} proteins, {n_missing} CIFs missing")
    for folder in CANONICAL_FOLDERS + ["unclassified"]:
        c = counts.get(folder, 0)
        if c or folder in CANONICAL_FOLDERS:
            print(f"  {folder:30s}  {c:4d}")

    if n_missing > 0 and n_ok == 0:
        print("[reannotate] ERROR: no CIFs found — check --boltz-out path")
        sys.exit(1)

    # ── Sentinel ──────────────────────────────────────────────────────────────
    state_summary = summary["gmm_state"].value_counts().to_dict()
    Path(args.sentinel).write_text(
        f"Reannotation done (per-protein GMM). "
        f"{n_ok} symlinks, {n_missing} missing. "
        f"States: {state_summary}\n"
    )
    print(f"[reannotate] Done. Sentinel: {args.sentinel}")


if __name__ == "__main__":
    main()
