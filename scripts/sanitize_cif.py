#!/usr/bin/env python3
"""
sanitize_cif.py — Pre-process Boltz-2 CIF for ChimeraX minimization.

Boltz-2 predicts ligand geometry (LIG1) that ChimeraX's IDATM atom-type
inference may misassign, causing AM1-BCC charge failures during dock prep.

This script:
  1. Reads the Boltz-2 CIF with gemmi
  2. Splits into protein (standard amino acids) and ligand (non-standard)
  3. Normalizes the ligand with RDKit:
     - Bond perception from 3D coordinates
     - Valence cleanup and sanitization
     - Proper hybridization assignment
  4. Writes a combined PDB with CONECT records for the ligand,
     so ChimeraX can determine correct atom types

Usage:
    python scripts/sanitize_cif.py \\
        --input  results/boltz/.../target_model_0.cif \\
        --output results/sanitized/.../model_sanitized.pdb
"""

import argparse
import tempfile
from pathlib import Path

import gemmi # pyright: ignore[reportMissingImports]
from rdkit import Chem # pyright: ignore[reportMissingImports]


# ── Helpers ───────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Sanitize Boltz-2 CIF: normalize ligand with RDKit "
                    "for ChimeraX minimization."
    )
    p.add_argument("--input",  required=True, help="Input CIF from Boltz-2")
    p.add_argument("--output", required=True, help="Output sanitized PDB")
    return p.parse_args()


# ── Step 1: CIF → PDB, then split by record type ─────────────────────────────

def cif_to_pdb(cif_path: str, pdb_path: str):
    """Convert a Boltz-2 CIF to PDB using gemmi (format conversion only)."""
    st = gemmi.read_structure(str(cif_path))
    st.setup_entities()
    st.assign_serial_numbers()
    st.write_pdb(str(pdb_path))


def split_pdb(full_pdb: str, protein_pdb: str, ligand_pdb: str) -> bool:
    """
    Split a PDB file into protein (ATOM) and ligand (HETATM) by record type.

    Returns True if any HETATM records (ligand) were found.
    """
    has_ligand = False
    with open(full_pdb) as f, \
         open(protein_pdb, "w") as pf, \
         open(ligand_pdb, "w") as lf:
        for line in f:
            if line.startswith("ATOM  "):
                pf.write(line)
            elif line.startswith("TER"):
                pf.write(line)
            elif line.startswith("HETATM"):
                lf.write(line)
                has_ligand = True
    return has_ligand


# ── Step 2: Normalize ligand with RDKit ───────────────────────────────────────

def sanitize_ligand(ligand_pdb_path: str) -> str:
    """
    Load ligand PDB with RDKit, sanitize, return PDB block with CONECT
    records.

    Following the TeachOpenCADD T009 normalisation approach:
      - Bond perception from 3D coordinates (proximityBonding)
      - Full sanitization (valence, aromaticity, hybridisation)
      - Stereochemistry assignment from 3D

    The CONECT records in the output tell ChimeraX the correct bond
    topology, preventing IDATM atom-type misassignment.
    """
    mol = Chem.MolFromPDBFile(
        str(ligand_pdb_path),
        removeHs=False,
        sanitize=False,
        proximityBonding=True,
    )
    if mol is None:
        raise ValueError(f"RDKit could not parse {ligand_pdb_path}")

    # ── Sanitize ──────────────────────────────────────────────────────────
    try:
        Chem.SanitizeMol(mol)
        print("[sanitize] Full RDKit sanitization OK")
    except Chem.AtomValenceException as e:
        print(f"[sanitize] WARNING: Valence issue ({e}), trying partial sanitization")
        Chem.SanitizeMol(
            mol,
            (
                Chem.SanitizeFlags.SANITIZE_CLEANUP
                | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
                | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
            ),
        )
    except Exception as e:
        print(f"[sanitize] WARNING: Sanitization failed ({e}), writing as-is with CONECT")

    # ── Stereochemistry from 3D ───────────────────────────────────────────
    try:
        Chem.AssignStereochemistryFrom3D(mol)
    except Exception:
        pass  # non-critical

    return Chem.MolToPDBBlock(mol)


# ── Step 3: Combine protein + sanitized ligand ────────────────────────────────

def max_atom_serial(pdb_path: str) -> int:
    """Find the highest atom serial number in a PDB file."""
    max_s = 0
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM  ", "HETATM")):
                try:
                    max_s = max(max_s, int(line[6:11]))
                except ValueError:
                    pass
    return max_s


def renumber_pdb_block(pdb_block: str, offset: int) -> list:
    """
    Renumber atom serials and CONECT references in a PDB block by *offset*.

    Returns a list of lines (HETATM + CONECT only).
    """
    old_to_new = {}
    lines_out = []

    for line in pdb_block.splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            old_serial = int(line[6:11])
            new_serial = old_serial + offset
            old_to_new[old_serial] = new_serial
            line = line[:6] + f"{new_serial:5d}" + line[11:]
            lines_out.append(line)
        elif line.startswith("CONECT"):
            # CONECT fields are 5-char wide after the keyword
            raw = line[6:]
            serials = []
            for i in range(0, len(raw), 5):
                chunk = raw[i : i + 5].strip()
                if chunk:
                    old_s = int(chunk)
                    serials.append(old_to_new.get(old_s, old_s + offset))
            conect = "CONECT"
            for s in serials:
                conect += f"{s:5d}"
            lines_out.append(conect)

    return lines_out


def combine_pdb(protein_pdb: str, ligand_pdb_block: str, output: str):
    """
    Combine protein PDB file and sanitized ligand PDB block into one file.
    Ligand atom serials are renumbered to avoid collisions with the protein.
    """
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)

    offset = max_atom_serial(protein_pdb)
    ligand_lines = renumber_pdb_block(ligand_pdb_block, offset)

    with open(out, "w") as fh:
        # ── Protein ATOM + TER records ────────────────────────────────────
        with open(protein_pdb) as pf:
            for line in pf:
                if line.startswith(("ATOM  ", "TER")):
                    fh.write(line)

        # ── Ligand HETATM records ─────────────────────────────────────────
        for line in ligand_lines:
            if line.startswith("HETATM"):
                fh.write(line + "\n")

        # ── CONECT records (critical for ChimeraX IDATM) ─────────────────
        for line in ligand_lines:
            if line.startswith("CONECT"):
                fh.write(line + "\n")

        fh.write("END\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    print(f"[sanitize] Input:  {args.input}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Step 1: CIF → PDB (gemmi handles format conversion)
        full_pdb    = str(tmp / "full.pdb")
        protein_pdb = str(tmp / "protein.pdb")
        ligand_pdb  = str(tmp / "ligand.pdb")

        cif_to_pdb(args.input, full_pdb)

        # Step 2: Split by ATOM / HETATM records
        has_ligand = split_pdb(full_pdb, protein_pdb, ligand_pdb)

        if not has_ligand:
            # Apo model — no ligand, just use the converted PDB as-is
            print("[sanitize] No ligand detected (apo model), converting CIF → PDB")
            import shutil
            shutil.copy(full_pdb, str(out))
        else:
            # Holo model — sanitize ligand with RDKit, then recombine
            print("[sanitize] Ligand found, sanitizing with RDKit")
            ligand_block = sanitize_ligand(ligand_pdb)
            combine_pdb(protein_pdb, ligand_block, str(out))

    print(f"[sanitize] Output: {args.output}")


if __name__ == "__main__":
    main()