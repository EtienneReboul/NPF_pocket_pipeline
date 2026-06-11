"""
ChimeraX Script: DSSP Secondary Structure Assignment
=====================================================
Usage (called by Snakemake rule `run_dssp`):

    chimerax --nogui --script "chimerax_dssp.py /path/to/model.cif /path/to/dssp.json"

What it does:
    1. Opens the Boltz-2 CIF file.
    2. Runs DSSP (secondary structure assignment): `dssp #1 report true`.
    3. Exports per-residue secondary structure to a JSON file.

The JSON output is consumed by compute_tm_angle.py to validate that TM2 and
TM8 (identified from Phobius/TMHMM sequence annotations) are indeed alpha-helices
in the predicted structure — serving as a sanity check on Boltz-2 output quality.

Output schema:
    [
      {"chain": "A", "resnum": 1, "resname": "MET", "is_helix": false, "is_strand": false},
      ...
    ]
"""

import sys
import json
from pathlib import Path

# ChimeraX runtime imports — not available to static analysis
from chimerax.core.commands import run  # pyright: ignore[reportMissingImports]


def main(session):
    if len(sys.argv) < 3:
        session.logger.error(
            "Usage: chimerax --nogui --script chimerax_dssp.py <input.cif> <output.json>"
        )
        raise SystemExit(1)

    cif_path = Path(sys.argv[1]).expanduser().resolve()
    out_path = Path(sys.argv[2]).expanduser().resolve()

    if not cif_path.exists():
        session.logger.error(f"Input CIF not found: {cif_path}")
        raise SystemExit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    session.logger.info(f"[dssp] Opening: {cif_path.name}")
    run(session, f"open {str(cif_path)!r}")

    session.logger.info("[dssp] Running DSSP secondary structure assignment ...")
    run(session, "dssp #1 report true")

    model = session.models[0]
    residues_data = []
    for residue in model.residues:
        residues_data.append({
            "chain":     residue.chain_id,
            "resnum":    residue.number,
            "resname":   residue.name,
            "is_helix":  bool(residue.is_helix),
            "is_strand": bool(residue.is_strand),
        })

    out_path.write_text(json.dumps(residues_data, indent=2))

    n_helix  = sum(1 for r in residues_data if r["is_helix"])
    n_strand = sum(1 for r in residues_data if r["is_strand"])
    session.logger.info(
        f"[dssp] {len(residues_data)} residues: "
        f"{n_helix} helix, {n_strand} strand. Output: {out_path.name}"
    )

    run(session, "close #1")
    run(session, "quit")


# ChimeraX injects `session` at module scope at runtime
main(session)  # pyright: ignore[reportUndefinedVariable] # noqa: F821
