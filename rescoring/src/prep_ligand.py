#!/usr/bin/env python3
"""
src/prep_ligand.py
=====================
One-time (per machine/PyRosetta-install) ligand parameterization.

Requires PyRosetta (see envs/pyrosetta_rescoring.yaml + README.md — this is
the one script in this project that MUST be run interactively, since it's
worth eyeballing the params file once).

Uses `rdkit_to_params` (pip install rdkit_to_params) to generate the .params
file directly from an RDKit mol, in-process. The classic
`molfile_to_params.py` (what HANDOFF_rescoring.md §5 step 1 originally names)
ships with the full Rosetta source/binary distribution but NOT with the
PyPI/pyrosetta-installer wheel this project's env is built on — its
`rosetta_py` helper package isn't on PyPI either. `rdkit_to_params` is a
maintained, pure RDKit+PyRosetta reimplementation built for exactly this
wheel-install scenario, and — usefully — lets us assign atom names directly
rather than reverse-engineering whatever an external tool picks.

What it does
------------
1. Builds the canonical, correctly-bonded GA1 mol from the SMILES in
   config.yaml (see ligand_fix.py's docstring for *why* this correction is
   needed at all — every pose's stored ligand has wrong bond orders/H count),
   using ONE reference complex (the first row of manifest.csv) for the
   heavy-atom 3D coordinates and Boltz's own heavy-atom names.
2. Computes Gasteiger partial charges (rdkit_to_params' default) and
   generates params/LIG.params via `rdkit_to_params.Params.from_mol()`,
   explicitly naming atoms: Boltz's own names for the 25 heavy atoms, H01..H24
   for the added hydrogens (ligand_fix.name_new_hydrogens — deterministic, so
   every complex reproduces the same names without any lookup).
3. Caches the reference-derived heavy-heavy bond list + heavy atom name set
   to params/atom_naming.json (used by every complex at run time instead of
   trusting that complex's own, sometimes-anomalous, CONECT records — see
   ligand_fix.py).
4. Validates the whole correction pipeline against N additional complexes
   (different proteins, sampled from manifest.csv): same atom count (49),
   same heavy-atom name SET, correct formula, AND — since PyRosetta is
   available here — actually loads each corrected ligand into a PyRosetta
   pose with `-extra_res_fa params/LIG.params` and scores it, to catch any
   params/naming mismatch before the full batch run. Fails loudly on any
   mismatch (see the doc's "fail loudly" instruction for PyRosetta-adjacent
   tooling).

Usage:
    python src/prep_ligand.py [--n-validate 20]
"""
import argparse
import sys

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

import config
import ligand_fix as lf
import pose_prep as pp


def build_and_embed_reference(reference_pdb) -> tuple[Chem.Mol, Chem.Mol, set[tuple[str, str]]]:
    """Returns (template_noH, corrected_reference_mol, heavy_bonds_by_name)."""
    smiles = config.load_ligand_smiles()
    template = lf.build_template(smiles)

    text = reference_pdb.read_text()
    heavy_bonds_by_name = lf.heavy_heavy_bonds_from_pdb(text)
    heavy_atoms = lf.parse_heavy_atoms(text)
    if len(heavy_atoms) != lf.N_HEAVY:
        sys.exit(f"{reference_pdb}: expected {lf.N_HEAVY} heavy ligand atoms, found {len(heavy_atoms)}")

    heavy_mol = lf.build_heavy_mol(heavy_atoms, heavy_bonds_by_name)
    fixed = lf.fix_and_rehydrogenate(heavy_mol, template)
    if fixed.GetNumAtoms() != lf.N_TOTAL:
        sys.exit(f"{reference_pdb}: corrected ligand has {fixed.GetNumAtoms()} atoms, expected {lf.N_TOTAL}")
    return template, fixed, heavy_bonds_by_name


def generate_params(mol: Chem.Mol, out_path) -> None:
    import pyrosetta
    pyrosetta.init("-mute all")
    from rdkit_to_params import Params

    mol = Chem.Mol(mol)
    AllChem.ComputeGasteigerCharges(mol)
    atomnames = {i: a.GetPDBResidueInfo().GetName() for i, a in enumerate(mol.GetAtoms())}
    p = Params.from_mol(mol, name=config.LIGAND_RESNAME, atomnames=atomnames)
    p.dump(str(out_path))
    print(f"[prep_ligand] wrote {out_path}")


def validate(template_noH: Chem.Mol, heavy_bonds_by_name: set[tuple[str, str]],
             expected_heavy_names: set[str], expected_formula: str, pdb_paths: list) -> None:
    import pyrosetta
    pyrosetta.init(f"-extra_res_fa {config.PARAMS_DIR / 'LIG.params'} -mute all")

    for pdb_path in pdb_paths:
        text = pdb_path.read_text()
        try:
            mol = lf.build_corrected_ligand_mol(text, template_noH, heavy_bonds_by_name, expected_heavy_names)
        except (ValueError, Chem.AtomValenceException) as e:
            sys.exit(f"[prep_ligand] VALIDATION FAILED {pdb_path}: {e}")
        if mol.GetNumAtoms() != lf.N_TOTAL:
            sys.exit(f"[prep_ligand] VALIDATION FAILED {pdb_path}: {mol.GetNumAtoms()} atoms, expected {lf.N_TOTAL}")
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
        if formula != expected_formula:
            sys.exit(f"[prep_ligand] VALIDATION FAILED {pdb_path}: formula {formula} != {expected_formula}")

        # Real end-to-end check: stage the full complex and actually load+score it in PyRosetta.
        naming = {"heavy_bonds_by_name": heavy_bonds_by_name, "heavy_names": expected_heavy_names}
        staged = config.RESULTS_DIR / "staged_poses" / f"_prep_validate_{pdb_path.parent.name}.pdb"
        pp.prepare_complex_pdb(pdb_path, template_noH, naming, staged)
        try:
            pose = pyrosetta.pose_from_pdb(str(staged))
            sfxn = pyrosetta.get_score_function()
            score = sfxn(pose)
        except Exception as e:
            sys.exit(f"[prep_ligand] VALIDATION FAILED {pdb_path}: PyRosetta could not load/score it: {e}")
        finally:
            staged.unlink(missing_ok=True)
        print(f"[prep_ligand]   validated {pdb_path.parent.name}: OK "
              f"({formula}, total_score={score:.1f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-validate", type=int, default=20,
                    help="number of additional (non-reference) complexes to spot-check")
    args = ap.parse_args()

    if not config.MANIFEST_CSV.exists():
        sys.exit("data/manifest.csv not found — run make_labels.py + make_manifest.py first.")
    manifest = pd.read_csv(config.MANIFEST_CSV)

    reference_row = manifest.iloc[0]
    reference_pdb = config.PIPELINE_ROOT / reference_row["pdb_path"]
    print(f"[prep_ligand] reference complex: {reference_row['complex_id']} ({reference_pdb})")

    template_noH, reference_mol, heavy_bonds_by_name = build_and_embed_reference(reference_pdb)
    expected_formula = Chem.rdMolDescriptors.CalcMolFormula(reference_mol)
    expected_heavy_names = {a.GetPDBResidueInfo().GetName().strip() for a in list(reference_mol.GetAtoms())[:lf.N_HEAVY]}
    print(f"[prep_ligand] expected formula: {expected_formula}")
    print(f"[prep_ligand] heavy_bonds_by_name ({len(heavy_bonds_by_name)} bonds, fixed for all complexes)")

    generate_params(reference_mol, config.PARAMS_DIR / "LIG.params")

    naming = {
        "smiles": config.load_ligand_smiles(),
        "reference_complex_id": reference_row["complex_id"],
        "heavy_bonds_by_name": sorted(heavy_bonds_by_name),
        "heavy_names": sorted(expected_heavy_names),
    }
    out_path = config.PARAMS_DIR / "atom_naming.json"
    import json
    out_path.write_text(json.dumps(naming, indent=2))
    print(f"[prep_ligand] wrote {out_path}")

    other_rows = manifest[manifest["complex_id"] != reference_row["complex_id"]]
    sample_rows = other_rows.sample(n=min(args.n_validate, len(other_rows)), random_state=0)
    sample_paths = [config.PIPELINE_ROOT / p for p in sample_rows["pdb_path"]]
    print(f"[prep_ligand] validating against {len(sample_paths)} random complexes "
          f"({list(sample_rows['complex_id'])})...")
    validate(template_noH, heavy_bonds_by_name, expected_heavy_names, expected_formula, sample_paths)

    print("[prep_ligand] done — params/LIG.params is ready for -extra_res_fa.")


if __name__ == "__main__":
    main()
