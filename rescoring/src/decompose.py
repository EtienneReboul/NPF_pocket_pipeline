"""
src/decompose.py
===================
HANDOFF_rescoring.md §5 step 4: per-residue ligand<->protein energy
decomposition from a scored PyRosetta pose's energy graph.

Sign convention preserved throughout: negative REU = stabilizing, positive =
unfavorable. `fa_rep` isolates clashes; `fa_sol`/`lk_ball_wtd` carry
desolvation penalties; `fa_elec` + `hbond_*` carry polar/directional
contributions (REF2015; see README.md for the "not kcal/mol" caveat).
"""
from __future__ import annotations

import pandas as pd
import pyrosetta
from pyrosetta import rosetta

# The two-body score terms HANDOFF_rescoring.md §5 asks for, at minimum.
SCORE_TERMS = [
    "fa_atr", "fa_rep", "fa_sol", "lk_ball_wtd", "fa_elec",
    "hbond_sc", "hbond_bb_sc", "hbond_lr_bb", "hbond_sr_bb",
]


def ligand_residue_index(pose: pyrosetta.Pose, ligand_resname: str = "LIG") -> int:
    for i in range(1, pose.total_residue() + 1):
        if pose.residue(i).name3().strip() == ligand_resname:
            return i
    raise ValueError(f"No {ligand_resname} residue found in pose.")


def decompose_ligand_contacts(pose: pyrosetta.Pose, sfxn, ligand_resname: str = "LIG") -> pd.DataFrame:
    """One row per (protein residue, score term) with a non-zero ligand<->residue edge."""
    sfxn(pose)
    L = ligand_residue_index(pose, ligand_resname)
    eg = pose.energies().energy_graph()
    weights = sfxn.weights()

    term_enums = {}
    for term in SCORE_TERMS:
        try:
            term_enums[term] = getattr(rosetta.core.scoring, term)
        except AttributeError:
            continue  # score term not defined in this Rosetta build — skip, don't fail loudly here

    rows = []
    for r in range(1, pose.total_residue() + 1):
        if r == L:
            continue
        edge = eg.find_energy_edge(L, r)
        if edge is None:
            continue
        emap = edge.fill_energy_map()

        weighted = {}
        for term, term_enum in term_enums.items():
            raw = emap[term_enum]
            if raw == 0.0:
                continue
            weighted[term] = raw * weights[term_enum]
        if not weighted:
            continue

        twobody_total = sum(weighted.values())
        residue = pose.residue(r)
        for term, value in weighted.items():
            rows.append({
                "prot_chain": pose.pdb_info().chain(r) if pose.pdb_info() else "A",
                "prot_resi": pose.pdb_info().number(r) if pose.pdb_info() else r,
                "prot_resn": residue.name3().strip(),
                "scoretype": term,
                "weighted_energy": value,
                "twobody_total": twobody_total,
            })

    return pd.DataFrame(rows, columns=[
        "prot_chain", "prot_resi", "prot_resn", "scoretype", "weighted_energy", "twobody_total",
    ])
