#!/usr/bin/env python3
"""
scripts/prepare_synthetic_templates.py
========================================
Copy per-clade synthetic template CIFs from config.yaml into a staging
directory, then strip all non-protein residues using ChimeraX.

Boltz-2 template validation rejects residue names absent from the CCD
(e.g. LIG1 in holo prediction outputs), so every synthetic template must be
protein-only before it can be used in a new run.

Output layout:
    {out_dir}/{clade}_{state}.cif    (e.g. cd17413_outward_open.cif)

Usage:
    python scripts/prepare_synthetic_templates.py \\
        --config config.yaml \\
        --out-dir data/synthetic_templates \\
        --chimerax-bin /Applications/ChimeraX.app/Contents/MacOS/ChimeraX \\
        --sentinel data/synthetic_templates/stripped.done
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml  # pyright: ignore[reportMissingModuleSource]


def strip_one(cif_path: Path, chimerax_bin: str) -> bool:
    """Strip all non-protein atoms from cif_path in-place using ChimeraX."""
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        script_path = Path(f.name)
        f.write(f"""\
from chimerax.core.commands import run
cif_path = {str(cif_path)!r}
run(session, 'open ' + cif_path)
run(session, 'delete ~protein')
run(session, 'save ' + cif_path + ' format mmcif')
run(session, 'exit')
""")
    try:
        result = subprocess.run(
            [chimerax_bin, "--nogui", "--script", str(script_path)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            print(f"  ERROR: ChimeraX exited {result.returncode}")
            print(result.stderr[-500:] if result.stderr else "")
            return False
        return True
    finally:
        script_path.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",        default="config.yaml")
    p.add_argument("--out-dir",       required=True)
    p.add_argument("--chimerax-bin",  required=True)
    p.add_argument("--sentinel",      required=True)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    per_clade = (
        cfg.get("templates", {})
           .get("synthetic", {})
           .get("per_clade", {})
    )
    if not per_clade:
        print("No per_clade synthetic templates defined in config — nothing to do.")
        Path(args.sentinel).parent.mkdir(parents=True, exist_ok=True)
        Path(args.sentinel).write_text("done\n")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = fail = 0
    for clade, states in per_clade.items():
        for state, src_path in states.items():
            src = Path(src_path)
            if not src.exists():
                print(f"WARNING: {src} not found — skipping {clade}/{state}")
                continue
            dst = out_dir / f"{clade}_{state}.cif"
            shutil.copy2(src, dst)
            print(f"  strip {clade}/{state} ... ", end="", flush=True)
            if strip_one(dst, args.chimerax_bin):
                print("done")
                ok += 1
            else:
                print("FAILED")
                fail += 1

    print(f"\n{ok} stripped, {fail} failed.")
    if fail:
        sys.exit(1)

    Path(args.sentinel).write_text("done\n")
    print(f"Sentinel written: {args.sentinel}")


if __name__ == "__main__":
    main()