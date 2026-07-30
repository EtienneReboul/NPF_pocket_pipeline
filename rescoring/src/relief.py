"""
src/relief.py
================
Tier-3 clash relief (HANDOFF_rescoring.md §5 step 3): score the raw pose,
then a light coordinate-constrained FastRelax to relieve clashes without
drifting from the co-folded/minimized geometry.

**Restricted to a neighborhood around the ligand, not the whole protein.**
This is a deliberate deviation from a literal "FastRelax the whole pose", and
was found necessary, not just faster: an unrestricted FastRelax on these
~590-residue Boltz-2 structures (never crystallographically refined) blew the
total score up catastrophically (-665 REU -> +36,677 REU on the first real
complex tested) — isolating the FastRelax stages showed it was full-protein
*repacking* specifically (all residues repacked, no restriction) that was
unstable (-675 -> +1,548 from packing alone; plain minimization alone, by
contrast, was fine and improved the score to -1,645). Restricting packing +
minimization to residues within RELIEF_RADIUS_A of the ligand keeps the score
change modest (no explosion) and is exactly what the doc itself calls for —
"light", local clash relief without drifting/over-minimizing far-away parts
of the structure — while also being far cheaper across ~4,950 complexes.
"""
from __future__ import annotations

import pyrosetta
from pyrosetta import rosetta

_INITIALIZED = False

RELIEF_RADIUS_A = 10.0  # neighborhood radius around the ligand allowed to repack/minimize


def init_pyrosetta(params_path, seed: int | None = None) -> None:
    """
    seed: HANDOFF_rescoring.md acceptance criterion 4 ("deterministic given a
    fixed seed"). FastRelax is stochastic, so without a fixed seed successive
    replicas (and reruns) genuinely differ — that's what n_replicas is for.
    Pass the same seed to get a fully reproducible run (same replica-to-
    replica sequence every time); omit it for normal ensemble variation.
    """
    global _INITIALIZED
    if _INITIALIZED:
        return
    seed_flags = f"-run:constant_seed -run:jran {seed} " if seed is not None else ""
    pyrosetta.init(
        f"-extra_res_fa {params_path} "
        f"{seed_flags}"
        "-mute all "
        "-relax:constrain_relax_to_start_coords "
        "-relax:coord_constrain_sidechains "
        "-relax:ramp_constraints true "
        "-no_optH false"
    )
    _INITIALIZED = True


def load_pose(pdb_path) -> pyrosetta.Pose:
    return pyrosetta.pose_from_pdb(str(pdb_path))


def ligand_residue_index(pose: pyrosetta.Pose, ligand_resname: str = "LIG") -> int:
    for i in range(1, pose.total_residue() + 1):
        if pose.residue(i).name3().strip() == ligand_resname:
            return i
    raise ValueError(f"No {ligand_resname} residue found in pose.")


def neighborhood_movemap_and_task(pose: pyrosetta.Pose, ligand_resi: int, radius: float = RELIEF_RADIUS_A):
    """MoveMap (bb+chi) and TaskFactory (repack only), both restricted to
    residues within `radius` of the ligand — see module docstring for why."""
    lig_sel = rosetta.core.select.residue_selector.ResidueIndexSelector(str(ligand_resi))
    nbr_sel = rosetta.core.select.residue_selector.NeighborhoodResidueSelector(lig_sel, radius, True)
    in_neighborhood = nbr_sel.apply(pose)

    movemap = rosetta.core.kinematics.MoveMap()
    for i in range(1, pose.total_residue() + 1):
        if in_neighborhood[i]:
            movemap.set_bb(i, True)
            movemap.set_chi(i, True)

    task_factory = rosetta.core.pack.task.TaskFactory()
    task_factory.push_back(rosetta.core.pack.task.operation.RestrictToRepacking())
    prevent = rosetta.core.pack.task.operation.PreventRepacking()
    for i in range(1, pose.total_residue() + 1):
        if not in_neighborhood[i]:
            prevent.include_residue(i)
    task_factory.push_back(prevent)

    return movemap, task_factory


def light_relax(pose: pyrosetta.Pose, sfxn, ligand_resi: int, cycles: int = 1) -> None:
    """
    Coordinate-constrained FastRelax restricted to the ligand's neighborhood
    (see module docstring). The coordinate-constraint behavior (constrain to
    start coords, ramp down) is controlled by the `-relax:*` flags passed
    once to pyrosetta.init() in init_pyrosetta() above — that's the standard,
    documented way to configure this (rather than mover setter methods,
    whose exact names vary across Rosetta versions). fa_rep ramping is
    FastRelax's normal default staged repulsive-ramping schedule, no extra
    flag needed.
    """
    movemap, task_factory = neighborhood_movemap_and_task(pose, ligand_resi)
    relax = rosetta.protocols.relax.FastRelax(sfxn, cycles)
    relax.set_movemap(movemap)
    relax.set_task_factory(task_factory)
    relax.apply(pose)


def relieve_clashes(pdb_path, params_path, n_replicas: int = 1, relax_cycles: int = 1,
                     seed: int | None = None):
    """
    Score raw -> light relax -> score relaxed, for n_replicas independent
    trajectories (stochastic FastRelax — replicas give an ensemble estimate,
    per HANDOFF_rescoring.md §3/§7).

    Returns: list of dicts, one per replica:
        {replica, fa_rep_raw, total_raw, fa_rep_relaxed, total_relaxed, pose}
    """
    init_pyrosetta(params_path, seed=seed)
    sfxn = pyrosetta.get_score_function()

    results = []
    for replica in range(n_replicas):
        pose = load_pose(pdb_path)
        ligand_resi = ligand_residue_index(pose)
        sfxn(pose)
        fa_rep_raw = pose.energies().total_energies()[rosetta.core.scoring.fa_rep]
        total_raw = pose.energies().total_energies()[rosetta.core.scoring.total_score]

        light_relax(pose, sfxn, ligand_resi, cycles=relax_cycles)

        sfxn(pose)
        fa_rep_relaxed = pose.energies().total_energies()[rosetta.core.scoring.fa_rep]
        total_relaxed = pose.energies().total_energies()[rosetta.core.scoring.total_score]

        results.append({
            "replica": replica,
            "fa_rep_raw": fa_rep_raw,
            "total_raw": total_raw,
            "fa_rep_relaxed": fa_rep_relaxed,
            "total_relaxed": total_relaxed,
            "pose": pose,
        })
    return results
