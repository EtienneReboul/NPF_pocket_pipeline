"""
src/ligand_fix.py
====================
Corrects the gibberellin A1 (GA1) ligand's bond orders and hydrogens.

Why this exists: every `model_minimized.pdb` pose in `results/minimized_synth/`
has a chemically wrong ligand. Boltz-2's raw CIF output carries no bond-order
records for the ligand, and `scripts/sanitize_cif.py` fills that gap with
`Chem.MolFromPDBFile(..., proximityBonding=True)` (pure-distance bond
guessing) followed by `Chem.AddHs()`, which blindly saturates every atom to
its default valence. The result: all 19 ligand carbons end up degree-4 and
all 6 oxygens degree-2 in every pose — i.e. *zero* double bonds anywhere,
when real GA1 needs three sp2 centers (the lactone C=O, the carboxylic-acid
C=O, and the exocyclic C=CH2). The carboxylic acid becomes a geminal diol,
the lactone carbonyl becomes a hydroxyl, and 6 extra explicit H atoms appear
(30 vs. the correct 24) to pad valence.

This is scoped as a **local, rescoring-only fix** (see README.md) — it does
not touch results/minimized_synth or any other analysis (PLIP, ligand_iptm,
...) that already consumes those poses as-is.

Method (validated in prep_ligand.py against the reference complex + a
handful of others — see its docstring):
  1. Parse only the ligand's *heavy* atoms (name, element, xyz) from the PDB.
     Boltz's atom naming for this ligand is the same fixed 25-heavy-atom name
     set in every complex.
  2. Connect them using a **fixed, reference-derived** heavy-heavy bond list
     (by atom name, cached in params/atom_naming.json — see prep_ligand.py),
     NOT each pose's own CONECT records. This matters: spot-checking found
     that a handful of poses carry an extra spurious heavy-heavy CONECT bond
     (and/or are missing a hydrogen or two) — presumably an artifact of
     whatever minimization/close-contact produced that particular pose. Since
     it's the same molecule everywhere, heavy-atom connectivity is invariant;
     re-deriving it per pose is both unnecessary and fragile. Only the 3D
     coordinates are taken from each pose.
  3. `AllChem.AssignBondOrdersFromTemplate(template, heavy_mol)` against the
     canonical GA1 template built from the SMILES in config.yaml — this
     fixes bond orders/aromaticity while keeping this complex's own 3D
     coordinates and atom names.
  4. `Chem.AddHs(fixed, addCoords=True)` adds the correct 24 hydrogens fresh,
     placed from the now-correct heavy-atom hybridisation.
  5. Rename every atom to the name Rosetta's `molfile_to_params.py` assigned
     to it (see prep_ligand.py) so the PDB matches `params/LIG.params`.
"""
from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

import config

LIG_RESNAME = config.LIGAND_RESNAME
LIG_CHAIN = config.LIGAND_CHAIN


@dataclass
class HeavyAtom:
    name: str
    element: str
    x: float
    y: float
    z: float


def parse_heavy_atoms(pdb_text: str) -> list[HeavyAtom]:
    """Parse one PDB's LIG residue heavy atoms (name, element, xyz), canonical name order."""
    heavy_atoms: list[HeavyAtom] = []
    for l in pdb_text.splitlines():
        if not (l.startswith("HETATM") and l[17:20].strip() == LIG_RESNAME):
            continue
        name = l[12:16].strip()
        elem = l[76:78].strip() or name[0]
        if elem == "H":
            continue
        x, y, z = float(l[30:38]), float(l[38:46]), float(l[46:54])
        heavy_atoms.append(HeavyAtom(name, elem, x, y, z))
    return sorted(heavy_atoms, key=lambda a: a.name)


def heavy_heavy_bonds_from_pdb(pdb_text: str) -> set[tuple[str, str]]:
    """Heavy-heavy bonds (by atom name) from one pose's own CONECT records.

    Used ONLY by prep_ligand.py to derive the one, fixed, reusable bond list
    from the reference complex — see the module docstring for why per-pose
    CONECT records aren't trusted for every complex.
    """
    lines = pdb_text.splitlines()
    serial_to_name, serial_to_elem = {}, {}
    for l in lines:
        if not (l.startswith("HETATM") and l[17:20].strip() == LIG_RESNAME):
            continue
        serial = int(l[6:11])
        name = l[12:16].strip()
        serial_to_name[serial] = name
        serial_to_elem[serial] = l[76:78].strip() or name[0]

    lig_serials = set(serial_to_name)
    bonds: set[tuple[str, str]] = set()
    for l in lines:
        if not l.startswith("CONECT"):
            continue
        fields = [int(x) for x in l[6:].split()]
        a0 = fields[0]
        if a0 not in lig_serials or serial_to_elem[a0] == "H":
            continue
        for a1 in fields[1:]:
            if a1 in lig_serials and serial_to_elem.get(a1) != "H":
                bonds.add(tuple(sorted((serial_to_name[a0], serial_to_name[a1]))))
    return bonds


def build_heavy_mol(heavy_atoms: list[HeavyAtom], bonds: set[tuple[str, str]]) -> Chem.Mol:
    """RWMol from heavy atoms (in the given, already-canonical, order) + single bonds."""
    name_to_idx = {a.name: i for i, a in enumerate(heavy_atoms)}
    rw = Chem.RWMol()
    for a in heavy_atoms:
        rw.AddAtom(Chem.Atom(a.element))
    for n0, n1 in bonds:
        if n0 in name_to_idx and n1 in name_to_idx:
            rw.AddBond(name_to_idx[n0], name_to_idx[n1], Chem.BondType.SINGLE)

    conf = Chem.Conformer(rw.GetNumAtoms())
    for i, a in enumerate(heavy_atoms):
        conf.SetAtomPosition(i, Point3D(a.x, a.y, a.z))
    mol = rw.GetMol()
    mol.AddConformer(conf)

    for atom, ha in zip(mol.GetAtoms(), heavy_atoms):
        info = Chem.AtomPDBResidueInfo()
        info.SetName(f"{ha.name:>4s}")
        info.SetResidueName(LIG_RESNAME)
        info.SetChainId(LIG_CHAIN)
        info.SetIsHeteroAtom(True)
        atom.SetMonomerInfo(info)

    Chem.SanitizeMol(
        mol,
        sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE ^ Chem.SANITIZE_SETAROMATICITY,
    )
    return mol


def build_template(smiles: str) -> Chem.Mol:
    """Heavy-atom-only GA1 mol from the canonical SMILES (correct bond orders, no Hs)."""
    template = Chem.MolFromSmiles(smiles)
    if template is None:
        raise ValueError(f"RDKit could not parse ligand SMILES: {smiles}")
    return template


def fix_and_rehydrogenate(heavy_mol: Chem.Mol, template: Chem.Mol) -> Chem.Mol:
    """AssignBondOrdersFromTemplate (fix bond orders) then AddHs (correct H count/placement)."""
    fixed = AllChem.AssignBondOrdersFromTemplate(template, heavy_mol)
    return Chem.AddHs(fixed, addCoords=True)


def rename_atoms(mol: Chem.Mol, heavy_name_to_rosetta: dict[str, str], h_rosetta_by_position: list[str]) -> Chem.Mol:
    """Rename every atom to its Rosetta .params atom name (see prep_ligand.py)."""
    n_heavy = len(heavy_name_to_rosetta)
    if mol.GetNumAtoms() != n_heavy + len(h_rosetta_by_position):
        raise ValueError(
            f"Atom count mismatch after AddHs: got {mol.GetNumAtoms()}, "
            f"expected {n_heavy + len(h_rosetta_by_position)}"
        )
    for i, atom in enumerate(mol.GetAtoms()):
        info = atom.GetPDBResidueInfo()
        if i < n_heavy:
            original_name = info.GetName().strip()
            rosetta_name = heavy_name_to_rosetta[original_name]
        else:
            rosetta_name = h_rosetta_by_position[i - n_heavy]
        if info is None:
            info = Chem.AtomPDBResidueInfo()
        info.SetName(f"{rosetta_name:>4s}")
        info.SetResidueName(LIG_RESNAME)
        info.SetChainId(LIG_CHAIN)
        info.SetIsHeteroAtom(True)
        atom.SetMonomerInfo(info)
    return mol


def build_corrected_ligand_mol(pdb_text: str, template: Chem.Mol,
                                heavy_bonds_by_name: set[tuple[str, str]],
                                heavy_name_to_rosetta: dict[str, str],
                                h_rosetta_by_position: list[str]) -> Chem.Mol:
    """Full per-complex correction: raw pose PDB text -> corrected, Rosetta-named RDKit mol."""
    heavy_atoms = parse_heavy_atoms(pdb_text)
    found_names = {a.name for a in heavy_atoms}
    expected_names = set(heavy_name_to_rosetta)
    if found_names != expected_names:
        raise ValueError(
            f"Ligand heavy-atom name set differs from reference. "
            f"Missing={expected_names - found_names} Extra={found_names - expected_names}"
        )
    heavy_mol = build_heavy_mol(heavy_atoms, heavy_bonds_by_name)
    fixed = fix_and_rehydrogenate(heavy_mol, template)
    return rename_atoms(fixed, heavy_name_to_rosetta, h_rosetta_by_position)


def corrected_ligand_pdb_block(pdb_text: str, template: Chem.Mol,
                                heavy_bonds_by_name: set[tuple[str, str]],
                                heavy_name_to_rosetta: dict[str, str],
                                h_rosetta_by_position: list[str]) -> str:
    """Same as build_corrected_ligand_mol, serialized to a PDB block (HETATM + CONECT)."""
    mol = build_corrected_ligand_mol(
        pdb_text, template, heavy_bonds_by_name, heavy_name_to_rosetta, h_rosetta_by_position
    )
    return Chem.MolToPDBBlock(mol)
