#!/usr/bin/env python3
"""
src/prep_ligand.py
=====================
One-time (per machine/PyRosetta-install) ligand parameterization.

Requires PyRosetta (see envs/pyrosetta_rescoring.yaml + README.md — this is
the one script in this project that MUST be run interactively, since
molfile_to_params.py's exact atom-naming output should be eyeballed once).

What it does
------------
1. Builds the canonical, correctly-bonded GA1 mol from the SMILES in
   config.yaml (see ligand_fix.py's docstring for *why* this correction is
   needed at all — every pose's stored ligand has wrong bond orders/H count).
2. Embeds + MMFF-optimizes a 3D conformer, writes params/ligand_template.sdf.
3. Runs Rosetta's molfile_to_params.py on it -> params/LIG.params (+ a
   generated conformer PDB). Rosetta assigns its own atom names in SDF atom
   order — we don't need to control or predict that scheme, we just read
   back whatever it picked (parsed from the ATOM lines of LIG.params).
4. Using ONE reference complex (the first row of manifest.csv), builds:
     - heavy_name_to_rosetta: {boltz PDB atom name -> rosetta atom name}
       for the 25 heavy atoms (by substructure match against the template).
     - h_rosetta_by_position: the 24 rosetta names for the newly-added
       hydrogens, in the deterministic order Chem.AddHs() produces them
       (reproducible for every complex — see ligand_fix.py).
   Both are cached to params/atom_naming.json for relief.py/run_complex.py
   to reuse without re-deriving per complex.
5. Validates the whole correction+renaming pipeline against N additional
   complexes (different proteins, sampled from manifest.csv) — same atom
   count (49), same heavy-atom name SET, and that the corrected ligand
   round-trips through Chem.SanitizeMol cleanly. Fails loudly on any
   mismatch (see the doc's "fail loudly" instruction for PyRosetta-adjacent
   tooling).

Usage:
    python src/prep_ligand.py [--n-validate 5]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

import config
import ligand_fix as lf

N_HEAVY = 25
N_H = 24
N_TOTAL = N_HEAVY + N_H


def locate_molfile_to_params() -> Path:
    """Find molfile_to_params.py shipped inside the installed pyrosetta package."""
    try:
        import pyrosetta  # noqa: F401
    except ImportError:
        sys.exit(
            "PyRosetta is not installed in this environment. "
            "See ../README.md for install instructions (academic license required)."
        )
    import pyrosetta as pr

    pkg_dir = Path(pr.__file__).resolve().parent
    hits = list(pkg_dir.rglob("molfile_to_params.py"))
    if not hits:
        # Fall back: search the whole site-packages tree (some distributions
        # ship the database as a sibling package, not under pyrosetta/).
        site_packages = pkg_dir.parent
        hits = list(site_packages.rglob("molfile_to_params.py"))
    if not hits:
        sys.exit(
            "Could not find molfile_to_params.py under the installed pyrosetta "
            f"package ({pkg_dir}). It ships in Rosetta's database/scripts — "
            "check your PyRosetta install includes the database."
        )
    return hits[0]


def build_and_embed_template(smiles: str) -> Chem.Mol:
    template = lf.build_template(smiles)
    template_h = Chem.AddHs(template)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    if AllChem.EmbedMolecule(template_h, params) != 0:
        sys.exit("RDKit conformer embedding failed for the GA1 template.")
    AllChem.MMFFOptimizeMolecule(template_h, maxIters=2000)
    return template_h


def run_molfile_to_params(script: Path, sdf_path: Path, out_prefix: Path) -> list[str]:
    """Run molfile_to_params.py, return the ATOM names in input-atom order."""
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(script),
        "-n", "LIG",
        "-p", str(out_prefix),
        "--clobber",
        str(sdf_path),
    ]
    print(f"[prep_ligand] running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=out_prefix.parent, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(f"molfile_to_params.py failed (exit {result.returncode})")

    params_path = out_prefix.with_suffix(".params")
    if not params_path.exists():
        sys.exit(f"Expected {params_path} was not created.")

    names = []
    for line in params_path.read_text().splitlines():
        if line.startswith("ATOM "):
            names.append(line.split()[1])
    if len(names) != N_TOTAL:
        sys.exit(f"Expected {N_TOTAL} ATOM lines in {params_path}, found {len(names)}.")
    return names


def derive_naming(template_h: Chem.Mol, template_noH: Chem.Mol, rosetta_names: list[str],
                   reference_pdb: Path, bonds: set[tuple[str, str]]) -> tuple[dict[str, str], list[str]]:
    text = reference_pdb.read_text()
    heavy_atoms = lf.parse_heavy_atoms(text)
    if len(heavy_atoms) != N_HEAVY:
        sys.exit(f"{reference_pdb}: expected {N_HEAVY} heavy ligand atoms, found {len(heavy_atoms)}")

    heavy_mol = lf.build_heavy_mol(heavy_atoms, bonds)
    fixed = lf.fix_and_rehydrogenate(heavy_mol, template_noH)
    if fixed.GetNumAtoms() != N_TOTAL:
        sys.exit(f"{reference_pdb}: corrected ligand has {fixed.GetNumAtoms()} atoms, expected {N_TOTAL}")

    match = fixed.GetSubstructMatch(template_h)
    if len(match) != N_TOTAL:
        sys.exit(
            f"{reference_pdb}: substructure match against the full (with-H) template "
            f"found only {len(match)}/{N_TOTAL} atoms."
        )

    pos_to_rosetta = {match[t]: rosetta_names[t] for t in range(N_TOTAL)}

    heavy_name_to_rosetta = {}
    for i in range(N_HEAVY):
        info = fixed.GetAtomWithIdx(i).GetPDBResidueInfo()
        heavy_name_to_rosetta[info.GetName().strip()] = pos_to_rosetta[i]
    h_rosetta_by_position = [pos_to_rosetta[i] for i in range(N_HEAVY, N_TOTAL)]

    return heavy_name_to_rosetta, h_rosetta_by_position


EXPECTED_FORMULA = None  # set in main() from the template, so a real formula change is caught


def validate(template_noH: Chem.Mol, heavy_bonds_by_name: set[tuple[str, str]],
             heavy_name_to_rosetta: dict[str, str],
             h_rosetta_by_position: list[str], pdb_paths: list[Path]) -> None:
    for pdb_path in pdb_paths:
        text = pdb_path.read_text()
        try:
            mol = lf.build_corrected_ligand_mol(
                text, template_noH, heavy_bonds_by_name, heavy_name_to_rosetta, h_rosetta_by_position
            )
        except (ValueError, Chem.AtomValenceException) as e:
            sys.exit(f"[prep_ligand] VALIDATION FAILED {pdb_path}: {e}")
        if mol.GetNumAtoms() != N_TOTAL:
            sys.exit(f"[prep_ligand] VALIDATION FAILED {pdb_path}: {mol.GetNumAtoms()} atoms, expected {N_TOTAL}")
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
        if EXPECTED_FORMULA is not None and formula != EXPECTED_FORMULA:
            sys.exit(f"[prep_ligand] VALIDATION FAILED {pdb_path}: formula {formula} != {EXPECTED_FORMULA}")
        print(f"[prep_ligand]   validated {pdb_path.parent.name}: OK ({formula})")


def main():
    global EXPECTED_FORMULA
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-validate", type=int, default=20,
                    help="number of additional (non-reference) complexes to spot-check")
    args = ap.parse_args()

    if not config.MANIFEST_CSV.exists():
        sys.exit("data/manifest.csv not found — run make_labels.py + make_manifest.py first.")
    manifest = pd.read_csv(config.MANIFEST_CSV)

    smiles = config.load_ligand_smiles()
    print(f"[prep_ligand] GA1 SMILES: {smiles}")
    template_h = build_and_embed_template(smiles)
    template_noH = Chem.RemoveHs(template_h)
    EXPECTED_FORMULA = Chem.rdMolDescriptors.CalcMolFormula(template_h)
    print(f"[prep_ligand] expected formula: {EXPECTED_FORMULA}")

    sdf_path = config.PARAMS_DIR / "ligand_template.sdf"
    Chem.MolToMolFile(template_h, str(sdf_path))
    print(f"[prep_ligand] wrote {sdf_path}")

    script = locate_molfile_to_params()
    print(f"[prep_ligand] using {script}")
    rosetta_names = run_molfile_to_params(script, sdf_path, config.PARAMS_DIR / "LIG")
    print(f"[prep_ligand] rosetta atom names ({len(rosetta_names)}): {rosetta_names}")

    reference_row = manifest.iloc[0]
    reference_pdb = config.PIPELINE_ROOT / reference_row["pdb_path"]
    print(f"[prep_ligand] reference complex: {reference_row['complex_id']} ({reference_pdb})")

    heavy_bonds_by_name = lf.heavy_heavy_bonds_from_pdb(reference_pdb.read_text())
    print(f"[prep_ligand] heavy_bonds_by_name ({len(heavy_bonds_by_name)} bonds, fixed for all complexes): "
          f"{sorted(heavy_bonds_by_name)}")

    heavy_name_to_rosetta, h_rosetta_by_position = derive_naming(
        template_h, template_noH, rosetta_names, reference_pdb, heavy_bonds_by_name
    )
    print(f"[prep_ligand] heavy_name_to_rosetta: {heavy_name_to_rosetta}")
    print(f"[prep_ligand] h_rosetta_by_position: {h_rosetta_by_position}")

    other_rows = manifest[manifest["complex_id"] != reference_row["complex_id"]]
    sample_rows = other_rows.sample(n=min(args.n_validate, len(other_rows)), random_state=0)
    sample_paths = [config.PIPELINE_ROOT / p for p in sample_rows["pdb_path"]]
    print(f"[prep_ligand] validating against {len(sample_paths)} random complexes "
          f"({list(sample_rows['complex_id'])})...")
    validate(template_noH, heavy_bonds_by_name, heavy_name_to_rosetta, h_rosetta_by_position, sample_paths)

    naming = {
        "smiles": smiles,
        "reference_complex_id": reference_row["complex_id"],
        "heavy_bonds_by_name": sorted(heavy_bonds_by_name),
        "heavy_name_to_rosetta": heavy_name_to_rosetta,
        "h_rosetta_by_position": h_rosetta_by_position,
    }
    out_path = config.PARAMS_DIR / "atom_naming.json"
    out_path.write_text(json.dumps(naming, indent=2))
    print(f"[prep_ligand] wrote {out_path}")
    print("[prep_ligand] done — params/LIG.params is ready for -extra_res_fa.")


if __name__ == "__main__":
    main()
