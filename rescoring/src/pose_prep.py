"""
src/pose_prep.py
===================
Combines the protein chain (kept as-is) with the bond-order-corrected ligand
(see ligand_fix.py) into one staged PDB file ready for PyRosetta
(`-extra_res_fa params/LIG.params`).
"""
from __future__ import annotations

import json
from pathlib import Path

from rdkit import Chem

import config
import ligand_fix as lf


def load_atom_naming() -> dict:
    path = config.PARAMS_DIR / "atom_naming.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run prep_ligand.py first (see ../README.md)."
        )
    naming = json.loads(path.read_text())
    naming["heavy_bonds_by_name"] = {tuple(p) for p in naming["heavy_bonds_by_name"]}
    return naming


def load_template(naming: dict) -> Chem.Mol:
    return lf.build_template(naming["smiles"])


def _protein_lines(pdb_text: str) -> list[str]:
    """All non-ligand lines (protein ATOM/TER/etc.), CONECT for the ligand dropped."""
    out = []
    for l in pdb_text.splitlines():
        if l.startswith("HETATM") and l[17:20].strip() == config.LIGAND_RESNAME:
            continue
        if l.startswith("CONECT"):
            continue  # ligand CONECT is regenerated; protein has none in these poses
        if l.startswith("END"):
            continue
        out.append(l)
    return out


def _renumber_ligand_block(pdb_block: str, start_serial: int) -> list[str]:
    """Shift HETATM/CONECT serials in an RDKit-written ligand PDB block past start_serial."""
    lines = [l for l in pdb_block.splitlines() if l.strip() and not l.startswith(("END", "MASTER"))]
    old_to_new: dict[int, int] = {}
    next_serial = start_serial
    out = []
    for l in lines:
        if l.startswith(("ATOM", "HETATM")):
            old = int(l[6:11])
            old_to_new[old] = next_serial
            # Force HETATM regardless of what RDKit wrote (record type, not just the
            # serial, must say HETATM for the ligand for downstream tools/eyeballing).
            new_line = f"HETATM{next_serial:>5d}{l[11:]}"
            out.append(new_line)
            next_serial += 1
    for l in lines:
        if l.startswith("CONECT"):
            fields = [int(x) for x in l[6:].split()]
            new_fields = [old_to_new[fields[0]]] + [old_to_new[f] for f in fields[1:]]
            out.append("CONECT" + "".join(f"{f:>5d}" for f in new_fields))
    return out


def prepare_complex_pdb(pdb_path: Path, template: Chem.Mol, naming: dict, out_path: Path) -> Path:
    """Write a staged PDB (protein as-is + bond-order-corrected, Rosetta-named ligand) to out_path."""
    text = pdb_path.read_text()
    protein_lines = _protein_lines(text)

    max_serial = 0
    for l in protein_lines:
        if l.startswith(("ATOM", "HETATM")):
            max_serial = max(max_serial, int(l[6:11]))

    ligand_mol = lf.build_corrected_ligand_mol(
        text, template, naming["heavy_bonds_by_name"],
        naming["heavy_name_to_rosetta"], naming["h_rosetta_by_position"],
    )
    ligand_block = Chem.MolToPDBBlock(ligand_mol)
    ligand_lines = _renumber_ligand_block(ligand_block, max_serial + 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(protein_lines + ligand_lines + ["END", ""]))
    return out_path
