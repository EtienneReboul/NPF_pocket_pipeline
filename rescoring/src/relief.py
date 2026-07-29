"""
src/relief.py
================
Tier-3 clash relief (HANDOFF_rescoring.md §5 step 3): score the raw pose,
then a light coordinate-constrained FastRelax to relieve clashes without
drifting from the co-folded/minimized geometry.

Deliberately NOT a full relax — the point is to fix `fa_rep` blowups from
co-folding/energy-minimization artifacts (see the README's ligand-hydrogen
finding for one such artifact) while preserving the pose PyRosetta is meant
to be scoring.
"""
from __future__ import annotations

import pyrosetta
from pyrosetta import rosetta

_INITIALIZED = False


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


def score_fa_rep(pose: pyrosetta.Pose, sfxn) -> float:
    sfxn(pose)
    return pose.energies().total_energies()[rosetta.core.scoring.fa_rep]


def light_relax(pose: pyrosetta.Pose, sfxn, cycles: int = 1) -> None:
    """
    Coordinate-constrained FastRelax, ramping fa_rep — relieves clashes in
    place. The coordinate-constraint behavior (constrain to start coords,
    ramp down) is controlled by the `-relax:*` flags passed once to
    pyrosetta.init() in init_pyrosetta() above — that's the standard,
    documented way to configure this (rather than mover setter methods,
    whose exact names vary across Rosetta versions). fa_rep ramping is
    FastRelax's normal default staged repulsive-ramping schedule, no extra
    flag needed.
    """
    relax = rosetta.protocols.relax.FastRelax(sfxn, cycles)
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
        sfxn(pose)
        fa_rep_raw = pose.energies().total_energies()[rosetta.core.scoring.fa_rep]
        total_raw = pose.energies().total_energies()[rosetta.core.scoring.total_score]

        light_relax(pose, sfxn, cycles=relax_cycles)

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
