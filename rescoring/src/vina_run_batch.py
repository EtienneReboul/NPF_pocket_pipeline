#!/usr/bin/env python3
"""
src/vina_run_batch.py
========================
Batch AutoDock Vina scoring over the whole manifest (same manifest/poses as
the PyRosetta pipeline). Resumable — skips complexes that already have a
results/vina/per_complex/<complex_id>.csv — and uses one OS process per
worker (multiprocessing), mirroring run_batch.py's design.

Usage:
    python src/vina_run_batch.py [--workers N] [--protein X] [--class importer] [--limit 50]
"""
import argparse
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

import config
import pose_prep as pp
import vina_score as vs


def _worker(args):
    complex_id, pdb_rel, protein, cls = args
    log_path = config.VINA_LOGS_DIR / f"{complex_id}.log"
    out_path = config.VINA_PER_COMPLEX_DIR / f"{complex_id}.csv"
    try:
        naming = pp.load_atom_naming()
        template = pp.load_template(naming)
        with open(log_path, "w") as log_file:
            df = vs.run_one(complex_id, config.PIPELINE_ROOT / pdb_rel, protein, cls,
                             naming, template, log_file=log_file)
        df.to_csv(out_path, index=False)
        return complex_id, True, None
    except Exception as e:
        with open(log_path, "a") as log_file:
            log_file.write(f"FAILED: {e}\n{traceback.format_exc()}")
        return complex_id, False, str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--protein", action="append", help="restrict to this protein (repeatable)")
    ap.add_argument("--class", dest="cls", choices=["importer", "non_importer"])
    ap.add_argument("--limit", type=int, help="only process the first N (post-filter, post-resume) complexes")
    args = ap.parse_args()

    manifest = pd.read_csv(config.MANIFEST_CSV)
    if args.protein:
        manifest = manifest[manifest["protein"].isin(args.protein)]
    if args.cls:
        manifest = manifest[manifest["class"] == args.cls]

    done = {p.stem for p in config.VINA_PER_COMPLEX_DIR.glob("*.csv")}
    todo = manifest[~manifest["complex_id"].isin(done)]
    if args.limit:
        todo = todo.head(args.limit)

    print(f"[vina_run_batch] {len(manifest)} in manifest, {len(done)} already done, "
          f"{len(todo)} to run now, {args.workers} workers")
    if todo.empty:
        return

    tasks = [
        (row["complex_id"], row["pdb_path"], row["protein"], row["class"])
        for _, row in todo.iterrows()
    ]

    t0 = time.time()
    n_ok, n_fail = 0, 0
    with mp.Pool(processes=args.workers) as pool:
        for i, (complex_id, ok, err) in enumerate(pool.imap_unordered(_worker, tasks), 1):
            n_ok += ok
            n_fail += not ok
            status = "OK" if ok else f"FAILED ({err})"
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta_min = (len(tasks) - i) / rate / 60 if rate > 0 else float("nan")
            print(f"[vina_run_batch] {i}/{len(tasks)} {complex_id}: {status}  "
                  f"({rate:.2f}/s, ETA {eta_min:.0f} min)")

    print(f"[vina_run_batch] done: {n_ok} ok, {n_fail} failed, {time.time() - t0:.0f}s total")


if __name__ == "__main__":
    main()
