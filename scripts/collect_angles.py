#!/usr/bin/env python3
"""
scripts/collect_angles.py
==========================
Concatenates per-sample angle.csv files for one protein × conformation
into a single angles.csv file.  Called by Snakemake rule `collect_tm_angles`.

Usage:
    python scripts/collect_angles.py --output angles.csv sample1/angle.csv sample2/angle.csv ...
"""

import argparse
import pandas as pd
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("inputs", nargs="+")
    return p.parse_args()


def main():
    args = parse_args()
    frames = [pd.read_csv(f) for f in args.inputs]
    df = pd.concat(frames, ignore_index=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"[collect_angles] {len(df)} rows written to {args.output}")


if __name__ == "__main__":
    main()
