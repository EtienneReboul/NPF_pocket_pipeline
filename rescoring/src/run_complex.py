#!/usr/bin/env python3
"""
src/run_complex.py
=====================
Run the full Tier-3 pipeline (HANDOFF_rescoring.md §5) on ONE complex:
stage the corrected pose (pose_prep.py) -> clash relief (relief.py) ->
per-residue decomposition (decompose.py) -> tidy CSV.

This is acceptance criterion 1 ("Runs end-to-end on one example complex").
run_batch.py calls `run_one()` directly (in-process, not via subprocess) to
drive the full manifest.

Usage:
    python src/run_complex.py --complex-id NPF3.1_Q9SX20__inward_occluded_holo__target_model_0 \\
        [--n-replicas 1] [--relax-cycles 1]
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import pyrosetta

import config
import decompose as dc
import pose_prep as pp
import relief as rl

STAGED_DIR = config.RESULTS_DIR / "staged_poses"


def run_one(complex_id: str, pdb_path: Path, protein: str, cls: str,
            naming: dict, template, n_replicas: int = 1, relax_cycles: int = 1,
            seed: int | None = None, log_file=None) -> pd.DataFrame:
    def log(msg):
        print(msg, file=log_file or sys.stdout, flush=True)

    t0 = time.time()
    staged_path = STAGED_DIR / f"{complex_id}.pdb"
    pp.prepare_complex_pdb(pdb_path, template, naming, staged_path)
    log(f"[{complex_id}] staged pose -> {staged_path}")

    replicas = rl.relieve_clashes(
        staged_path, config.PARAMS_DIR / "LIG.params",
        n_replicas=n_replicas, relax_cycles=relax_cycles, seed=seed,
    )

    frames = []
    for rep in replicas:
        log(f"[{complex_id}] replica {rep['replica']}: "
            f"fa_rep {rep['fa_rep_raw']:.1f} -> {rep['fa_rep_relaxed']:.1f}  "
            f"total {rep['total_raw']:.1f} -> {rep['total_relaxed']:.1f}")
        sfxn = pyrosetta.get_score_function()
        df = dc.decompose_ligand_contacts(rep["pose"], sfxn)
        df.insert(0, "replica", rep["replica"])
        df.insert(0, "class", cls)
        df.insert(0, "protein", protein)
        df.insert(0, "complex_id", complex_id)
        df["fa_rep_raw"] = rep["fa_rep_raw"]
        df["fa_rep_relaxed"] = rep["fa_rep_relaxed"]
        df["total_raw"] = rep["total_raw"]
        df["total_relaxed"] = rep["total_relaxed"]
        frames.append(df)

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    log(f"[{complex_id}] done in {time.time() - t0:.1f}s, {len(result)} rows")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--complex-id", required=True)
    ap.add_argument("--n-replicas", type=int, default=1)
    ap.add_argument("--relax-cycles", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None,
                    help="fixed Rosetta RNG seed for a fully reproducible run (omit for genuine ensemble variation)")
    args = ap.parse_args()

    manifest = pd.read_csv(config.MANIFEST_CSV)
    rows = manifest[manifest["complex_id"] == args.complex_id]
    if rows.empty:
        sys.exit(f"complex_id {args.complex_id!r} not found in {config.MANIFEST_CSV}")
    row = rows.iloc[0]

    naming = pp.load_atom_naming()
    template = pp.load_template(naming)

    df = run_one(
        row["complex_id"], config.PIPELINE_ROOT / row["pdb_path"], row["protein"], row["class"],
        naming, template, n_replicas=args.n_replicas, relax_cycles=args.relax_cycles, seed=args.seed,
    )
    out_path = config.PER_COMPLEX_DIR / f"{row['complex_id']}.csv"
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
