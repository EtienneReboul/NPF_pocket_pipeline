#!/usr/bin/env python3
"""
src/run_batch.py
===================
HANDOFF_rescoring.md acceptance criterion 2: batch the whole manifest with a
single command, per-complex logs, resumable (skips complexes that already
have a results/per_complex/<complex_id>.csv — safe to Ctrl-C and re-run).

Uses one OS process per worker (multiprocessing, not threads) — PyRosetta is
not thread-safe within a process, but each worker process calls
pyrosetta.init() exactly once (relief.py guards it) and is otherwise
independent.

Usage:
    python src/run_batch.py [--workers N] [--n-replicas 1] [--relax-cycles 1]
                             [--protein NPF3.1_Q9SX20] [--class importer] [--limit 50]
"""
import argparse
import multiprocessing as mp
import os
import sys
import time
import traceback
import zlib
from pathlib import Path

import pandas as pd

import config
import pose_prep as pp
import run_complex as rc


def _complex_seed(base_seed: int, complex_id: str) -> int:
    """
    Per-complex seed derived from (base_seed, complex_id) — NOT the same
    base_seed reused for every complex. Work is distributed across worker
    processes by imap_unordered in whatever order the OS scheduler picks, so
    a shared global seed would make each complex's result depend on which
    worker happened to process it first (and in what sequence relative to
    other complexes on that worker) — not reproducible batch-to-batch. Hashing
    the complex_id in makes every complex's seed independent of scheduling.
    """
    return (base_seed + zlib.crc32(complex_id.encode())) % (2**31 - 1)


def _worker(args):
    complex_id, pdb_rel, protein, cls, n_replicas, relax_cycles, base_seed = args
    seed = _complex_seed(base_seed, complex_id) if base_seed is not None else None
    log_path = config.LOGS_DIR / f"{complex_id}.log"
    out_path = config.PER_COMPLEX_DIR / f"{complex_id}.csv"
    try:
        naming = pp.load_atom_naming()
        template = pp.load_template(naming)
        with open(log_path, "w") as log_file:
            df = rc.run_one(
                complex_id, config.PIPELINE_ROOT / pdb_rel, protein, cls,
                naming, template, n_replicas=n_replicas, relax_cycles=relax_cycles,
                seed=seed, log_file=log_file,
            )
        df.to_csv(out_path, index=False)
        return complex_id, True, None
    except Exception as e:
        with open(log_path, "a") as log_file:
            log_file.write(f"FAILED: {e}\n{traceback.format_exc()}")
        return complex_id, False, str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--n-replicas", type=int, default=1)
    ap.add_argument("--relax-cycles", type=int, default=1)
    ap.add_argument("--protein", action="append", help="restrict to this protein (repeatable)")
    ap.add_argument("--class", dest="cls", choices=["importer", "non_importer"])
    ap.add_argument("--limit", type=int, help="only process the first N (post-filter, post-resume) complexes")
    ap.add_argument("--seed", type=int, default=None,
                    help="fixed base seed for a fully reproducible batch (per-complex seeds are derived "
                         "from this + complex_id, independent of worker scheduling); omit for genuine "
                         "ensemble variation")
    args = ap.parse_args()

    if not (config.PARAMS_DIR / "atom_naming.json").exists():
        sys.exit("params/atom_naming.json not found — run prep_ligand.py first (see README.md).")

    manifest = pd.read_csv(config.MANIFEST_CSV)
    if args.protein:
        manifest = manifest[manifest["protein"].isin(args.protein)]
    if args.cls:
        manifest = manifest[manifest["class"] == args.cls]

    done = {p.stem for p in config.PER_COMPLEX_DIR.glob("*.csv")}
    todo = manifest[~manifest["complex_id"].isin(done)]
    if args.limit:
        todo = todo.head(args.limit)

    print(f"[run_batch] {len(manifest)} in manifest, {len(done)} already done, "
          f"{len(todo)} to run now, {args.workers} workers")
    if todo.empty:
        return

    # NOTE: iterrows(), not itertuples() — "class" is a reserved keyword and
    # itertuples() silently renames such columns to positional fields.
    tasks = [
        (row["complex_id"], row["pdb_path"], row["protein"], row["class"],
         args.n_replicas, args.relax_cycles, args.seed)
        for _, row in todo.iterrows()
    ]

    t0 = time.time()
    n_ok, n_fail = 0, 0
    with mp.Pool(processes=args.workers) as pool:
        for i, (complex_id, ok, err) in enumerate(pool.imap_unordered(_worker, tasks), 1):
            n_ok += ok
            n_fail += not ok
            if ok:
                status = "OK"
            else:
                status = f"FAILED ({err})"
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta_min = (len(tasks) - i) / rate / 60 if rate > 0 else float("nan")
            print(f"[run_batch] {i}/{len(tasks)} {complex_id}: {status}  "
                  f"({rate:.2f}/s, ETA {eta_min:.0f} min)")

    print(f"[run_batch] done: {n_ok} ok, {n_fail} failed, {time.time() - t0:.0f}s total")


if __name__ == "__main__":
    main()
