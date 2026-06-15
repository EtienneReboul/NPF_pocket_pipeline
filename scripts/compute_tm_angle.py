#!/usr/bin/env python3
"""
scripts/compute_tm_angle.py
============================
Computes the angle between the TM2 and TM8 helix principal axes for one
Boltz-2 predicted structure.

Background
----------
Qureshi et al. (2020) Nature https://doi.org/10.1038/s41586-020-1963-z
identified the inter-helix angle between TM2 (N-terminal 6-TM bundle) and
TM8 (C-terminal 6-TM bundle) as a reliable scalar metric to discriminate
inward-open, occluded, and outward-open MFS transporter conformations.

Method
------
1. Read TM helix positions (1-based, from Phobius/TMHMM) for this protein.
2. Sort all TM helices by start position; index 1 = TM2, index 7 = TM8.
3. Read per-residue secondary structure from the DSSP JSON (sanity check):
   flag if TM2 or TM8 helix content < `--min-helix-frac` in the prediction.
4. Extract Cα coordinates for TM2 and TM8 from the raw Boltz-2 CIF.
5. Compute each helix principal axis via truncated SVD on the mean-centred
   Cα coordinates (first right-singular vector).
6. Return the unsigned angle (0–90°) between the two axis vectors.

Usage (called by Snakemake rule `compute_tm_angle`):
    python scripts/compute_tm_angle.py \\
        --cif           results/boltz/NPF1.1_Q9LYD5/outward_open_apo/boltz_out/.../model_0.cif \\
        --dssp          results/dssp/NPF1.1_Q9LYD5/outward_open_apo/model_0/dssp.json \\
        --topology      data/interpro/tm_topology_summary.json \\
        --protein       NPF1.1_Q9LYD5 \\
        --conformation  outward_open_apo \\
        --sample-id     model_0 \\
        --output        results/tm_angles/NPF1.1_Q9LYD5/outward_open_apo/model_0/angle.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import gemmi
import numpy as np


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cif",            required=True)
    p.add_argument("--dssp",           required=True)
    p.add_argument("--topology",       required=True)
    p.add_argument("--protein",        required=True)
    p.add_argument("--conformation",   required=True)
    p.add_argument("--sample-id",      required=True)
    p.add_argument("--output",         required=True)
    p.add_argument("--min-helix-frac", type=float, default=0.5,
                   help="Minimum fraction of TM residues that must be helical in DSSP")
    return p.parse_args()


# ── Geometry helpers ───────────────────────────────────────────────────────────

def helix_axis(ca_coords: np.ndarray) -> np.ndarray:
    """
    Compute the principal axis of a helix via SVD of mean-centred Cα coordinates.
    Returns a unit vector.
    """
    centred = ca_coords - ca_coords.mean(axis=0)
    _, _, Vt = np.linalg.svd(centred, full_matrices=False)
    return Vt[0]   # first right-singular vector is the principal axis


def unsigned_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Unsigned angle (0–90°) between two helix axis vectors.
    Helix direction is arbitrary, so we fold the full 0–180° range onto 0–90°.
    """
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(np.degrees(np.arccos(np.clip(abs(cos_theta), 0.0, 1.0))))


# ── Cα extraction from CIF ────────────────────────────────────────────────────

def get_ca_coords(structure_model: gemmi.Model, chain_name: str,
                  start_res: int, end_res: int) -> np.ndarray:
    """Return Cα coordinates for residues in [start_res, end_res] on chain_name."""
    coords = []
    try:
        chain = structure_model[chain_name]
    except (KeyError, RuntimeError):
        return np.empty((0, 3))
    for residue in chain:
        if start_res <= residue.seqid.num <= end_res:
            for atom in residue:
                if atom.name == "CA":
                    coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
    return np.array(coords) if coords else np.empty((0, 3))


def longest_chain_name(model: gemmi.Model) -> str:
    """Return the name of the chain with the most residues (the protein chain)."""
    best = max(model, key=lambda c: len(list(c)))
    return best.name


# ── Main ───────────────────────────────────────────────────────────────────────

RESULT_FIELDS = [
    "protein", "conformation", "sample_id",
    "chain",
    "tm2_start", "tm2_end", "n_ca_tm2",
    "tm8_start", "tm8_end", "n_ca_tm8",
    "tm2_helix_frac", "tm8_helix_frac",
    "angle_deg",
    "n_tm_helices", "tool_used",
    "status",
]


def write_result(path: str, row: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerow({f: row.get(f) for f in RESULT_FIELDS})


def main():
    args = parse_args()

    base = dict(
        protein=args.protein,
        conformation=args.conformation,
        sample_id=args.sample_id,
        angle_deg=None,
        n_tm_helices=None,
        tool_used=None,
        status="unknown",
    )

    # ── 1. Load TM topology ───────────────────────────────────────────────────
    topology = json.loads(Path(args.topology).read_text())
    if args.protein not in topology:
        print(f"[tm_angle] ERROR: protein '{args.protein}' not in topology summary")
        write_result(args.output, {**base, "status": "protein_not_in_topology"})
        sys.exit(0)   # exit 0 so Snakemake marks the output as created

    prot_topo = topology[args.protein]
    # Prefer Phobius (more accurate for eukaryotes), fall back to TMHMM
    tool_used = None
    raw_helices = None
    for tool in ("DEEPTMHMM", "UNIPROT", "PHOBIUS", "TMHMM"):
        if tool in prot_topo and prot_topo[tool]:
            tool_used  = tool
            raw_helices = prot_topo[tool]
            break

    if raw_helices is None:
        print(f"[tm_angle] ERROR: no TM helices in topology for {args.protein}")
        write_result(args.output, {**base, "status": "no_tm_helices"})
        sys.exit(0)

    sorted_helices = sorted(raw_helices, key=lambda h: h["start"])
    n_tm = len(sorted_helices)
    base["n_tm_helices"] = n_tm
    base["tool_used"]    = tool_used

    if n_tm < 8:
        print(
            f"[tm_angle] WARNING: only {n_tm} TM helices for {args.protein} "
            f"(need ≥8 for TM2/TM8 identification)"
        )
        write_result(args.output, {**base, "status": f"only_{n_tm}_tm_helices"})
        sys.exit(0)

    tm2 = sorted_helices[1]   # 0-indexed: index 1 → TM2
    tm8 = sorted_helices[7]   # 0-indexed: index 7 → TM8
    base.update(
        tm2_start=tm2["start"], tm2_end=tm2["end"],
        tm8_start=tm8["start"], tm8_end=tm8["end"],
    )

    # ── 2. Load DSSP and compute helix fraction for TM2/TM8 ──────────────────
    dssp_data   = json.loads(Path(args.dssp).read_text())
    helix_res   = {(r["chain"], r["resnum"]) for r in dssp_data if r["is_helix"]}

    # Determine which chain carries the protein (longest chain in DSSP output)
    from collections import Counter
    chain_counts = Counter(r["chain"] for r in dssp_data)
    protein_chain = chain_counts.most_common(1)[0][0] if chain_counts else "A"

    def _helix_frac(start, end):
        total = end - start + 1
        helical = sum(
            1 for rn in range(start, end + 1)
            if (protein_chain, rn) in helix_res
        )
        return helical / total if total > 0 else 0.0

    tm2_hfrac = _helix_frac(tm2["start"], tm2["end"])
    tm8_hfrac = _helix_frac(tm8["start"], tm8["end"])
    base.update(tm2_helix_frac=round(tm2_hfrac, 3), tm8_helix_frac=round(tm8_hfrac, 3))

    dssp_flag = ""
    if tm2_hfrac < args.min_helix_frac:
        dssp_flag += f"low_tm2_helix({tm2_hfrac:.2f})"
    if tm8_hfrac < args.min_helix_frac:
        dssp_flag += f"low_tm8_helix({tm8_hfrac:.2f})"

    # ── 3. Extract Cα coordinates from Boltz-2 CIF ───────────────────────────
    structure = gemmi.read_structure(args.cif)
    model     = structure[0]
    chain_name = longest_chain_name(model)
    base["chain"] = chain_name

    ca_tm2 = get_ca_coords(model, chain_name, tm2["start"], tm2["end"])
    ca_tm8 = get_ca_coords(model, chain_name, tm8["start"], tm8["end"])
    base.update(n_ca_tm2=len(ca_tm2), n_ca_tm8=len(ca_tm8))

    if len(ca_tm2) < 4:
        write_result(args.output, {**base, "status": "insufficient_tm2_ca"})
        print(f"[tm_angle] {args.protein}/{args.conformation}/{args.sample_id}: "
              f"only {len(ca_tm2)} Cα in TM2 — skipping angle")
        sys.exit(0)
    if len(ca_tm8) < 4:
        write_result(args.output, {**base, "status": "insufficient_tm8_ca"})
        print(f"[tm_angle] {args.protein}/{args.conformation}/{args.sample_id}: "
              f"only {len(ca_tm8)} Cα in TM8 — skipping angle")
        sys.exit(0)

    # ── 4. Compute helix axes and angle ───────────────────────────────────────
    axis2  = helix_axis(ca_tm2)
    axis8  = helix_axis(ca_tm8)
    angle  = unsigned_angle(axis2, axis8)

    status = "ok" if not dssp_flag else f"ok_with_flags:{dssp_flag}"
    write_result(args.output, {**base, "angle_deg": round(angle, 3), "status": status})

    print(
        f"[tm_angle] {args.protein}/{args.conformation}/{args.sample_id}: "
        f"chain={chain_name}, TM2={tm2['start']}-{tm2['end']} "
        f"({len(ca_tm2)} Cα, helix={tm2_hfrac:.0%}), "
        f"TM8={tm8['start']}-{tm8['end']} "
        f"({len(ca_tm8)} Cα, helix={tm8_hfrac:.0%}), "
        f"angle={angle:.1f}°, status={status}"
    )


if __name__ == "__main__":
    main()
