"""
src/vina_prep.py
===================
Ligand + receptor PDBQT preparation for AutoDock Vina
(https://autodock-vina.readthedocs.io/en/latest/docking_python.html), reusing
the same bond-order/hydrogen correction built for the PyRosetta pipeline
(ligand_fix.py) — every stored pose has the same chemically wrong ligand (see
README.md), and that fix is exactly as necessary here as it was for Rosetta.

Both receptor and ligand PDBQTs are prepared with `meeko`
(mk_prepare_receptor.py for the protein, MoleculePreparation for the ligand)
— the tool the official Vina docs themselves use. meeko's ligand prep
consumes an RDKit mol directly, so the corrected ligand from ligand_fix.py
plugs straight in, no intermediate file format needed.

A receptor is prepared PER COMPLEX, not once per protein: each of a
protein's ~150 poses is a distinct Boltz-2 conformation, not just a distinct
ligand placement, so the receptor atoms genuinely differ pose to pose.

Correction/anomaly tracking (per the request to report which structures
needed correction, importer vs. non_importer): every ligand needs the
bond-order/H fix — that part is 100% universal, driven by the same upstream
sanitize_cif.py issue documented in ligand_fix.py, and isn't a
class-differentiated signal. What DOES vary per complex, and is tracked
here, is whether the RAW pose data itself was anomalous beyond that
universal fix — e.g. a handful of poses found during the PyRosetta work had
an extra spurious heavy-heavy CONECT bond and/or a missing hydrogen or two
in their raw HETATM records (an artifact of whatever produced that specific
pose, not something ligand_fix.py's reference-bond-graph approach needs to
care about for correctness, but worth flagging as "this source pose's raw
data was messier than the clean reference").
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import meeko
import numpy as np

import config
import ligand_fix as lf

REFERENCE_RAW_HETATM_COUNT = lf.N_HEAVY + 30  # 25 heavy + 30 H, in every pose's *wrong* hydrogenation
REFERENCE_RAW_BOND_COUNT = 29                  # heavy-heavy CONECT bonds in the clean reference pose

BOX_PADDING_A = 10.0  # Angstrom padding around the ligand's own bounding box


def raw_ligand_anomaly(pdb_text: str) -> dict:
    """
    Cheap, correction-independent check: does this complex's raw ligand data
    match the reference pose's raw atom/bond counts? Purely informational —
    ligand_fix.py never trusts a pose's own CONECT/H records anyway (see its
    module docstring) — this just flags how anomalous the SOURCE data was.
    """
    n_hetatm = sum(
        1 for l in pdb_text.splitlines()
        if l.startswith("HETATM") and l[17:20].strip() == config.LIGAND_RESNAME
    )
    bonds = lf.heavy_heavy_bonds_from_pdb(pdb_text)
    return {
        "n_hetatm": n_hetatm,
        "n_heavy_heavy_bonds": len(bonds),
        "hetatm_count_anomaly": n_hetatm != REFERENCE_RAW_HETATM_COUNT,
        "bond_count_anomaly": len(bonds) != REFERENCE_RAW_BOND_COUNT,
    }


def prepare_ligand_pdbqt(pdb_text: str, template, naming: dict) -> tuple[str, np.ndarray, dict]:
    """
    Returns (pdbqt_string, heavy_atom_coords, anomaly_info).
    Raises ValueError if the heavy-atom NAME SET itself doesn't match the
    reference (a harder failure than the raw-count anomalies above — see
    ligand_fix.build_corrected_ligand_mol).
    """
    anomaly = raw_ligand_anomaly(pdb_text)
    mol = lf.build_corrected_ligand_mol(pdb_text, template, naming["heavy_bonds_by_name"], naming["heavy_names"])

    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])

    mk_prep = meeko.MoleculePreparation()
    setups = mk_prep.prepare(mol)
    pdbqt_string, is_ok, err = meeko.PDBQTWriterLegacy.write_string(setups[0])
    if not is_ok:
        raise RuntimeError(f"meeko ligand PDBQT preparation failed: {err}")
    return pdbqt_string, coords, anomaly


def prepare_receptor_pdbqt(pdb_text: str, out_prefix: Path) -> tuple[Path, str]:
    """
    Writes a protein-only PDB alongside out_prefix, runs mk_prepare_receptor.py,
    returns (pdbqt_path, combined stdout+stderr). The output text is inspected
    by callers for anything beyond the routine "Files written" success
    message — empirically rare for these complete, already-protonated,
    Boltz-2-predicted structures (spot-checked clean across 10 random
    proteins), but logged for every complex regardless.
    """
    protein_lines = [l for l in pdb_text.splitlines() if l.startswith("ATOM") or l.startswith("TER")]
    receptor_input_pdb = Path(f"{out_prefix}_input.pdb")
    receptor_input_pdb.parent.mkdir(parents=True, exist_ok=True)
    receptor_input_pdb.write_text("\n".join(protein_lines + ["END", ""]))

    result = subprocess.run(
        ["mk_prepare_receptor.py", "--read_pdb", str(receptor_input_pdb), "-o", str(out_prefix), "-p"],
        capture_output=True, text=True,
    )
    output = (result.stdout or "") + (result.stderr or "")
    pdbqt_path = Path(f"{out_prefix}.pdbqt")
    receptor_input_pdb.unlink(missing_ok=True)
    if result.returncode != 0 or not pdbqt_path.exists():
        raise RuntimeError(f"mk_prepare_receptor.py failed:\n{output}")
    return pdbqt_path, output


def ligand_box(coords: np.ndarray, padding: float = BOX_PADDING_A) -> tuple[list[float], list[float]]:
    """(center, box_size) enclosing the ligand's own bounding box + padding — for
    scoring/local-optimization of the CURRENT pose, not a blind docking search."""
    center = coords.mean(axis=0)
    box_size = (coords.max(axis=0) - coords.min(axis=0)) + padding
    return center.tolist(), box_size.tolist()
