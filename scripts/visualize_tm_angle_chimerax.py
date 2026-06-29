#!/usr/bin/env python3
"""
scripts/visualize_tm_angle_chimerax.py
========================================
Generates a ChimeraX .cxc script to visualise the TM2/TM8 helix angle,
mirroring the computation in compute_tm_angle.py exactly.

Method (same as compute_tm_angle.py):
  1. Load TM topology → pick TM2 (sorted index 1) and TM8 (sorted index 7).
  2. Extract Cα coordinates from the CIF.
  3. Compute each helix principal axis via truncated SVD on the mean-centred
     Cα positions (first right-singular vector = first principal component).
  4. Return the unsigned angle (0–90°) between the two axis vectors.

The CXC script:
  • Colours TM2 orange and TM8 purple against a grey cartoon background.
  • Shows Cα atoms as spheres for both helices.
  • Draws visual axis cylinders via ChimeraX `define axis … atoms` (which runs
    the same PCA internally, so the displayed axes match the computed angle).
  • Overlays the pre-computed angle as a 2D text label.

Usage:
    python scripts/visualize_tm_angle_chimerax.py \\
        --cif        results/boltz/PROTEIN/conformation/model_0.cif \\
        --topology   data/interpro/tm_topology_summary.json \\
        --protein    NPF1.1_Q9LYD5 \\
        --output     results/chimerax/NPF1.1_Q9LYD5/conformation/tm_angle.cxc
"""

import argparse
import json
import sys
from pathlib import Path

import gemmi
import numpy as np


# ── Geometry (mirrors compute_tm_angle.py) ─────────────────────────────────────

def helix_axis(ca: np.ndarray) -> np.ndarray:
    centred = ca - ca.mean(axis=0)
    _, _, Vt = np.linalg.svd(centred, full_matrices=False)
    return Vt[0]


def unsigned_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(np.degrees(np.arccos(np.clip(abs(cos_theta), 0.0, 1.0))))


def axis_display_length(ca: np.ndarray, axis_vec: np.ndarray, padding: float = 4.0) -> float:
    """Return display length for the axis cylinder: span of Cα projections + padding."""
    proj = (ca - ca.mean(axis=0)) @ axis_vec
    return round(float(proj.max() - proj.min()) + padding, 1)


# ── Structure helpers ──────────────────────────────────────────────────────────

def get_ca(model: gemmi.Model, chain: str, start: int, end: int) -> np.ndarray:
    coords = []
    try:
        ch = model[chain]
    except (KeyError, RuntimeError):
        return np.empty((0, 3))
    for residue in ch:
        if start <= residue.seqid.num <= end:
            for atom in residue:
                if atom.name == "CA":
                    coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
    return np.array(coords) if coords else np.empty((0, 3))


def longest_chain(model: gemmi.Model) -> str:
    return max(model, key=lambda c: len(list(c))).name


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate a ChimeraX .cxc script to visualise the TM2/TM8 helix angle."
    )
    p.add_argument("--cif",      required=True, help="Path to Boltz-2 CIF file.")
    p.add_argument("--topology", required=True, help="Path to tm_topology_summary.json.")
    p.add_argument("--protein",  required=True, help="Protein key in the topology JSON.")
    p.add_argument("--output",   required=True, help="Output .cxc script path.")
    return p.parse_args()


# ── CXC generation ─────────────────────────────────────────────────────────────

def build_cxc(
    cif_abs: str,
    protein: str,
    chain: str,
    tm2: dict,
    tm8: dict,
    ca_tm2: np.ndarray,
    ca_tm8: np.ndarray,
) -> list[str]:
    axis2  = helix_axis(ca_tm2)
    axis8  = helix_axis(ca_tm8)
    angle  = unsigned_angle(axis2, axis8)
    len2   = axis_display_length(ca_tm2, axis2)
    len8   = axis_display_length(ca_tm8, axis8)

    lines = [
        f"# ChimeraX TM2/TM8 angle visualisation",
        f"# protein   : {protein}",
        f"# TM2 (idx 1): residues {tm2['start']}-{tm2['end']}  ({len(ca_tm2)} Cα)",
        f"# TM8 (idx 7): residues {tm8['start']}-{tm8['end']}  ({len(ca_tm8)} Cα)",
        f"# TM2-TM8 angle (unsigned): {angle:.2f}°",
        f"# Axis method: SVD on mean-centred Cα (= PCA, mirrors compute_tm_angle.py)",
        f"",
        f"open {cif_abs}",
        f"",
        f"# Base view",
        f"set bgColor white",
        f"cartoon",
        f"color /{chain} gray",
        f"",
        f"# ── TM2 (N-bundle, sorted helix index 1) ─────────────────────────────────",
        f"color /{chain}:{tm2['start']}-{tm2['end']} darkorange",
        f"show  /{chain}:{tm2['start']}-{tm2['end']}@CA atoms",
        f"style /{chain}:{tm2['start']}-{tm2['end']}@CA sphere",
        f"",
        f"# ── TM8 (C-bundle, sorted helix index 7) ─────────────────────────────────",
        f"color /{chain}:{tm8['start']}-{tm8['end']} mediumpurple",
        f"show  /{chain}:{tm8['start']}-{tm8['end']}@CA atoms",
        f"style /{chain}:{tm8['start']}-{tm8['end']}@CA sphere",
        f"",
        f"# ── Helix axes ────────────────────────────────────────────────────────────",
        f"# ChimeraX 'define axis atoms' uses PCA on the selected Cα atoms,",
        f"# which is equivalent to the SVD in compute_tm_angle.py — so the",
        f"# displayed cylinders correspond exactly to the measured angle.",
        f"define axis /{chain}:{tm2['start']}-{tm2['end']}@CA length {len2} name tm2 radius 0.8 color darkorange",
        f"define axis /{chain}:{tm8['start']}-{tm8['end']}@CA length {len8} name tm8 radius 0.8 color mediumpurple",
        f"",
        f"# ── Angle label ───────────────────────────────────────────────────────────",
        f'2dlabel text "TM2–TM8 angle: {angle:.1f}°" xpos 0.02 ypos 0.96 size 18 color black',
        f"",
        f"lighting soft",
        f"view",
    ]
    return lines, angle


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    topology = json.loads(Path(args.topology).read_text())
    if args.protein not in topology:
        print(f"[tm_angle_cxc] ERROR: protein '{args.protein}' not in topology", file=sys.stderr)
        sys.exit(1)

    raw_helices = None
    for tool in ("DEEPTMHMM", "UNIPROT", "PHOBIUS", "TMHMM"):
        entry = topology[args.protein].get(tool)
        if entry:
            raw_helices = entry
            break

    if raw_helices is None:
        print(f"[tm_angle_cxc] ERROR: no TM helices for {args.protein}", file=sys.stderr)
        sys.exit(1)

    sorted_helices = sorted(raw_helices, key=lambda h: h["start"])
    n_tm = len(sorted_helices)

    if n_tm < 8:
        print(f"[tm_angle_cxc] ERROR: only {n_tm} TM helices — need ≥8 for TM2/TM8", file=sys.stderr)
        sys.exit(1)

    tm2 = sorted_helices[1]
    tm8 = sorted_helices[7]

    structure = gemmi.read_structure(args.cif)
    model     = structure[0]
    chain     = longest_chain(model)

    ca_tm2 = get_ca(model, chain, tm2["start"], tm2["end"])
    ca_tm8 = get_ca(model, chain, tm8["start"], tm8["end"])

    if len(ca_tm2) < 4:
        print(f"[tm_angle_cxc] ERROR: only {len(ca_tm2)} Cα in TM2 — cannot compute axis", file=sys.stderr)
        sys.exit(1)
    if len(ca_tm8) < 4:
        print(f"[tm_angle_cxc] ERROR: only {len(ca_tm8)} Cα in TM8 — cannot compute axis", file=sys.stderr)
        sys.exit(1)

    cif_abs = str(Path(args.cif).resolve())
    lines, angle = build_cxc(cif_abs, args.protein, chain, tm2, tm8, ca_tm2, ca_tm8)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(
        f"[tm_angle_cxc] Written {out}"
        f"  (chain={chain}, TM2={tm2['start']}-{tm2['end']},"
        f" TM8={tm8['start']}-{tm8['end']}, angle={angle:.1f}°)"
    )


if __name__ == "__main__":
    main()
