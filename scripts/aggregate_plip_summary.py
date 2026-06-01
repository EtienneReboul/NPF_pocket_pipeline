#!/usr/bin/env python3
"""
scripts/aggregate_plip_summary.py
===================================
Stage 7b of the NPF pipeline:
Merge all per-diffusion-sample PLIP summary CSVs for one protein × conformation
pair into a single CSV, tagging each row with the sample index.

Expected input path structure:
    {plip_dir}/{protein}/{conformation}/sample_{N}/{model_id}_report/csv/summary.csv

Usage (called by Snakemake rule `aggregate_plip`):
    python scripts/aggregate_plip_summary.py \\
        --output results/plip/NPF6.3_Q05085/occluded_holo/summary.csv \\
        results/plip/NPF6.3_Q05085/occluded_holo/sample_0/.../summary.csv \\
        results/plip/NPF6.3_Q05085/occluded_holo/sample_1/.../summary.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", help="Paths to individual summary.csv files")
    p.add_argument("--output", required=True)
    return p.parse_args()


def extract_sample(path: Path) -> str:
    """
    Derive sample identifier from path.
    Expected: .../protein/conformation/sample_{N}/{model_id}_report/csv/summary.csv
    parts[-5] → sample_N
    """
    parts = path.parts
    return parts[-5] if len(parts) >= 5 else "unknown"


def main():
    args   = parse_args()
    frames = []

    for p in (Path(x) for x in args.inputs):
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[WARNING] Could not read {p}: {e}", file=sys.stderr)
            continue
        if df.empty:
            continue
        sample = extract_sample(p)
        df.insert(0, "sample", sample)
        frames.append(df)

    if not frames:
        print("[ERROR] No valid summary CSVs loaded.", file=sys.stderr)
        sys.exit(1)

    agg = pd.concat(frames, ignore_index=True).sort_values("sample").reset_index(drop=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out, index=False)
    print(f"[OK] {len(frames)} files → {out} ({len(agg)} rows)")


if __name__ == "__main__":
    main()
