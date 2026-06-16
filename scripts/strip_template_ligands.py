#!/usr/bin/env python3
"""
scripts/strip_template_ligands.py
===================================
Strip all non-protein atoms from template CIF files using ChimeraX.

Boltz-2 only uses the polymer backbone for structural templating.  Ligands in
the original PDB entry are irrelevant and crash boltz with CCD-not-found errors
for novel compounds (e.g. A1EP2 in 9UIF).

Usage (run once after downloading templates, or re-run after adding new ones):
    python scripts/strip_template_ligands.py
    python scripts/strip_template_ligands.py --templates-root data/templates
    python scripts/strip_template_ligands.py --chimerax-bin /path/to/ChimeraX
    python scripts/strip_template_ligands.py --dry-run
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def find_chimerax(hint: str | None = None) -> str | None:
    candidates = [hint] if hint else []
    for name in ("ChimeraX", "chimerax"):
        found = shutil.which(name)
        if found:
            candidates.append(found)
    candidates += [
        "/Applications/ChimeraX_Daily.app/Contents/MacOS/ChimeraX",
        "/Applications/ChimeraX.app/Contents/MacOS/ChimeraX",
        "/Applications/UCSF ChimeraX.app/Contents/MacOS/ChimeraX",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    return None


def strip_one(cif_path: Path, chimerax_bin: str) -> bool:
    """Run ChimeraX to keep only protein, overwrite CIF in-place. Returns True on success."""
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
            print(f"    ERROR: ChimeraX exited {result.returncode}")
            print(result.stderr[-500:] if result.stderr else "")
            return False
        return True
    finally:
        script_path.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--templates-root", default="data/templates")
    p.add_argument("--chimerax-bin", default=None)
    p.add_argument("--dry-run", "-n", action="store_true")
    args = p.parse_args()

    chimerax = find_chimerax(args.chimerax_bin)
    if chimerax is None:
        print("ERROR: ChimeraX not found. Use --chimerax-bin to specify the path.")
        sys.exit(1)
    print(f"ChimeraX: {chimerax}")

    cifs = sorted(Path(args.templates_root).rglob("*.cif"))
    if not cifs:
        print(f"No .cif files found under {args.templates_root}")
        sys.exit(0)

    print(f"Found {len(cifs)} template CIF(s)\n")
    ok = fail = 0
    for cif in cifs:
        print(f"  {'[dry-run] ' if args.dry_run else ''}strip: {cif}", end=" ... ", flush=True)
        if args.dry_run:
            print()
            ok += 1
            continue
        if strip_one(cif, chimerax):
            print("done")
            ok += 1
        else:
            print("FAILED")
            fail += 1

    print(f"\nDone: {ok} stripped, {fail} failed.")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
