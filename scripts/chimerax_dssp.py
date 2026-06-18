"""
ChimeraX Script: DSSP Secondary Structure Assignment
=====================================================
Usage:

  Single mode (one CIF → one JSON):
    chimerax --nogui --script "chimerax_dssp.py /path/to/model.cif /path/to/dssp.json"

  Batch mode (one session processes all samples for a protein × conformation):
    chimerax --nogui --script "chimerax_dssp.py /path/to/cif_manifest.json /path/to/dssp_out_dir"

  In batch mode sys.argv[1] is the Boltz cif_manifest.json (a JSON list of CIF paths)
  and sys.argv[2] is the output directory.  Each <stem>.cif writes to
  <out_dir>/<stem>/dssp.json, matching the path expected by compute_tm_angle.py.

Output schema (per JSON file):
    [
      {"chain": "A", "resnum": 1, "resname": "MET", "is_helix": false, "is_strand": false},
      ...
    ]
"""

import sys
import json
from pathlib import Path

from chimerax.core.commands import run  # pyright: ignore[reportMissingImports]


def _run_dssp_on_cif(session, cif_path: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    session.logger.info(f"[dssp] Opening: {cif_path.name}")
    run(session, f"open {str(cif_path)!r}")
    run(session, "dssp #1 report true")

    model = session.models[0]
    residues_data = [
        {
            "chain":     r.chain_id,
            "resnum":    r.number,
            "resname":   r.name,
            "is_helix":  bool(r.is_helix),
            "is_strand": bool(r.is_strand),
        }
        for r in model.residues
    ]
    out_path.write_text(json.dumps(residues_data, indent=2))

    n_helix  = sum(1 for r in residues_data if r["is_helix"])
    n_strand = sum(1 for r in residues_data if r["is_strand"])
    session.logger.info(
        f"[dssp] {len(residues_data)} residues: "
        f"{n_helix} helix, {n_strand} strand → {out_path}"
    )
    run(session, "close #1")


def main(session):
    if len(sys.argv) < 3:
        session.logger.error(
            "Usage: chimerax_dssp.py <model.cif> <output.json>  "
            "OR  chimerax_dssp.py <cif_manifest.json> <dssp_out_dir>"
        )
        raise SystemExit(1)

    arg1 = Path(sys.argv[1]).expanduser().resolve()
    arg2 = Path(sys.argv[2]).expanduser().resolve()

    if arg1.suffix == ".json":
        # Batch mode: arg1 is the Boltz cif_manifest.json, arg2 is the output directory
        cif_paths = [Path(p) for p in json.loads(arg1.read_text())]
        out_dir = arg2
        for cif in cif_paths:
            out_json = out_dir / cif.stem / "dssp.json"
            _run_dssp_on_cif(session, cif, out_json)
        session.logger.info(f"[dssp] Batch complete: {len(cif_paths)} CIFs processed")
    else:
        # Single mode: arg1 = CIF path, arg2 = JSON output path
        if not arg1.exists():
            session.logger.error(f"Input CIF not found: {arg1}")
            raise SystemExit(1)
        _run_dssp_on_cif(session, arg1, arg2)

    run(session, "quit")


main(session)  # pyright: ignore[reportUndefinedVariable] # noqa: F821
