#!/usr/bin/env python3
"""
scripts/visualize_gate_chimerax.py
====================================
Generates a ChimeraX .cxc script to visualize the residues involved in the
extracellular and intracellular gate distance calculations, mirroring the
logic in compute_tm_angle.py.

Gate residues (0-indexed into sorted TM helices):
  ext_gate: N-bundle tips — last 4 Cα of TM1 (idx 0)  + first 4 Cα of TM2 (idx 1)
            C-bundle tips — last 4 Cα of TM7 (idx 6)  + first 4 Cα of TM8 (idx 7)
  int_gate: N-bundle tips — last 4 Cα of TM4 (idx 3)  + first 4 Cα of TM5 (idx 4)
            C-bundle tips — last 4 Cα of TM10 (idx 9) + first 4 Cα of TM11 (idx 10)

A ChimeraX `distance` pseudobond is drawn between the closest Cα pair found
for each gate (reproducing the value stored in ext_gate_A / int_gate_A).

Usage:
    python scripts/visualize_gate_chimerax.py \\
        --cif        results/boltz/PROTEIN/conformation/model_0.cif \\
        --topology   data/interpro/tm_topology_summary.json \\
        --protein    NPF1.1_Q9LYD5 \\
        --output     results/chimerax/NPF1.1_Q9LYD5_gates.cxc
"""

import argparse
import json
import sys
from pathlib import Path

import gemmi
import numpy as np

TIP_WINDOW = 4

# Colours used in the CXC script
COLOR_SCHEME = {
    "background":   "white",
    "protein":      "gray",
    "ext_nb":       "cornflower blue",   # ext-gate N-bundle helices (TM1/TM2)
    "ext_cb":       "dodger blue",        # ext-gate C-bundle helices (TM7/TM8)
    "int_nb":       "tomato",             # int-gate N-bundle helices (TM4/TM5)
    "int_cb":       "orange red",         # int-gate C-bundle helices (TM10/TM11)
    "ext_tip":      "yellow",             # Cα tip atoms (extracellular)
    "int_tip":      "lime green",         # Cα tip atoms (intracellular)
}


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate a ChimeraX .cxc script to visualise MFS gate residues."
    )
    p.add_argument("--cif",      required=True, help="Path to Boltz-2 CIF file.")
    p.add_argument("--topology", required=True, help="Path to tm_topology_summary.json.")
    p.add_argument("--protein",  required=True, help="Protein key in the topology JSON.")
    p.add_argument("--output",   required=True, help="Output .cxc script path.")
    return p.parse_args()


# ── Geometry helpers (mirrors compute_tm_angle.py) ────────────────────────────

def helix_tip_resnums(ca_coords: np.ndarray, resnums: list, n_terminal: bool):
    """Return the tip slice of (resnums, coords) matching helix_tip() logic."""
    n = min(TIP_WINDOW, len(resnums))
    if n == 0:
        return [], np.empty((0, 3))
    if n_terminal:
        return resnums[:n], ca_coords[:n]
    else:
        return resnums[-n:], ca_coords[-n:]


def get_ca_with_resnums(model: gemmi.Model, chain: str, start: int, end: int):
    coords, resnums = [], []
    try:
        ch = model[chain]
    except (KeyError, RuntimeError):
        return np.empty((0, 3)), []
    for residue in ch:
        if start <= residue.seqid.num <= end:
            for atom in residue:
                if atom.name == "CA":
                    coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
                    resnums.append(residue.seqid.num)
    return (np.array(coords) if coords else np.empty((0, 3))), resnums


def longest_chain(model: gemmi.Model) -> str:
    return max(model, key=lambda c: len(list(c))).name


def closest_ca_pair(ca_a, res_a, ca_b, res_b):
    """Return (resnum_a, resnum_b, distance_Å) for the closest Cα pair."""
    if len(ca_a) == 0 or len(ca_b) == 0:
        return None, None, None
    d = np.linalg.norm(ca_a[:, None, :] - ca_b[None, :, :], axis=-1)
    i, j = np.unravel_index(d.argmin(), d.shape)
    return res_a[i], res_b[j], round(float(d[i, j]), 2)


# ── CXC generation ─────────────────────────────────────────────────────────────

def residue_spec(chain: str, resnums: list) -> str:
    return f"/{chain}:{','.join(str(r) for r in resnums)}"


def build_cxc(cif_path: str, chain: str, sorted_helices: list, model: gemmi.Model) -> list[str]:
    # Collect tip residues for all 8 helix tips
    def tip(helix_idx: int, n_terminal: bool):
        h = sorted_helices[helix_idx]
        ca, res = get_ca_with_resnums(model, chain, h["start"], h["end"])
        return helix_tip_resnums(ca, res, n_terminal)

    # ext_gate — _ext_tip uses n_terminal=(i % 2 == 1)
    tm1_res,  tm1_ca  = tip(0, n_terminal=False)   # last 4 of TM1
    tm2_res,  tm2_ca  = tip(1, n_terminal=True)    # first 4 of TM2
    tm7_res,  tm7_ca  = tip(6, n_terminal=False)   # last 4 of TM7
    tm8_res,  tm8_ca  = tip(7, n_terminal=True)    # first 4 of TM8

    nb_ext_res = tm1_res + tm2_res
    cb_ext_res = tm7_res + tm8_res
    nb_ext_ca  = np.vstack([tm1_ca, tm2_ca]) if len(tm1_ca) and len(tm2_ca) else (tm1_ca if len(tm1_ca) else tm2_ca)
    cb_ext_ca  = np.vstack([tm7_ca, tm8_ca]) if len(tm7_ca) and len(tm8_ca) else (tm7_ca if len(tm7_ca) else tm8_ca)

    # int_gate — _int_tip uses n_terminal=(i % 2 == 0)
    tm4_res,  tm4_ca  = tip(3, n_terminal=False)   # last 4 of TM4  (3%2==0 → False)
    tm5_res,  tm5_ca  = tip(4, n_terminal=True)    # first 4 of TM5 (4%2==0 → True)
    tm10_res, tm10_ca = tip(9, n_terminal=False)   # last 4 of TM10
    tm11_res, tm11_ca = tip(10, n_terminal=True)   # first 4 of TM11

    nb_int_res = tm4_res + tm5_res
    cb_int_res = tm10_res + tm11_res
    nb_int_ca  = np.vstack([tm4_ca, tm5_ca]) if len(tm4_ca) and len(tm5_ca) else (tm4_ca if len(tm4_ca) else tm5_ca)
    cb_int_ca  = np.vstack([tm10_ca, tm11_ca]) if len(tm10_ca) and len(tm11_ca) else (tm10_ca if len(tm10_ca) else tm11_ca)

    # Closest Cα pairs (the actual gate distance atoms)
    ext_r1, ext_r2, ext_d = closest_ca_pair(nb_ext_ca, nb_ext_res, cb_ext_ca, cb_ext_res)
    int_r1, int_r2, int_d = closest_ca_pair(nb_int_ca, nb_int_res, cb_int_ca, cb_int_res)

    sc = COLOR_SCHEME
    h = sorted_helices
    lines = [
        f"# ChimeraX gate-distance visualisation",
        f"# protein: {args_protein}",
        f"# ext_gate_A = {ext_d if ext_d else 'N/A'} Å  (Cα/{chain}:{ext_r1} — /{chain}:{ext_r2})",
        f"# int_gate_A = {int_d if int_d else 'N/A'} Å  (Cα/{chain}:{int_r1} — /{chain}:{int_r2})",
        f"",
        f"open {cif_path}",
        f"",
        f"# Base view",
        f"set bgColor {sc['background']}",
        f"cartoon",
        f"color /{chain} {sc['protein']}",
        f"",
        f"# ── Extracellular-gate helices ────────────────────────────────────────────",
        f"# N-bundle (TM1 idx0: {h[0]['start']}-{h[0]['end']}, TM2 idx1: {h[1]['start']}-{h[1]['end']})",
        f"color /{chain}:{h[0]['start']}-{h[0]['end']} {sc['ext_nb']}",
        f"color /{chain}:{h[1]['start']}-{h[1]['end']} {sc['ext_nb']}",
        f"# C-bundle (TM7 idx6: {h[6]['start']}-{h[6]['end']}, TM8 idx7: {h[7]['start']}-{h[7]['end']})",
        f"color /{chain}:{h[6]['start']}-{h[6]['end']} {sc['ext_cb']}",
        f"color /{chain}:{h[7]['start']}-{h[7]['end']} {sc['ext_cb']}",
        f"",
        f"# ── Intracellular-gate helices ────────────────────────────────────────────",
        f"# N-bundle (TM4 idx3: {h[3]['start']}-{h[3]['end']}, TM5 idx4: {h[4]['start']}-{h[4]['end']})",
        f"color /{chain}:{h[3]['start']}-{h[3]['end']} {sc['int_nb']}",
        f"color /{chain}:{h[4]['start']}-{h[4]['end']} {sc['int_nb']}",
        f"# C-bundle (TM10 idx9: {h[9]['start']}-{h[9]['end']}, TM11 idx10: {h[10]['start']}-{h[10]['end']})",
        f"color /{chain}:{h[9]['start']}-{h[9]['end']} {sc['int_cb']}",
        f"color /{chain}:{h[10]['start']}-{h[10]['end']} {sc['int_cb']}",
        f"",
        f"# ── Tip residues (shown as spheres) ──────────────────────────────────────",
    ]

    for label, resnums, color in [
        ("ext-gate N-bundle tips (TM1 C-term + TM2 N-term)", nb_ext_res, sc["ext_tip"]),
        ("ext-gate C-bundle tips (TM7 C-term + TM8 N-term)", cb_ext_res, sc["ext_tip"]),
        ("int-gate N-bundle tips (TM4 C-term + TM5 N-term)", nb_int_res, sc["int_tip"]),
        ("int-gate C-bundle tips (TM10 C-term + TM11 N-term)", cb_int_res, sc["int_tip"]),
    ]:
        if not resnums:
            continue
        spec = residue_spec(chain, resnums)
        lines += [
            f"# {label}",
            f"show {spec} atoms",
            f"style {spec} sphere",
            f"color {spec} {color} atoms",
        ]

    lines += [
        f"",
        f"# ── Gate distance measurements ────────────────────────────────────────────",
    ]

    if ext_r1 is not None:
        lines += [
            f"# Extracellular gate: {ext_d} Å",
            f"distance /{chain}:{ext_r1}@CA /{chain}:{ext_r2}@CA",
        ]
    if int_r1 is not None:
        lines += [
            f"# Intracellular gate: {int_d} Å",
            f"distance /{chain}:{int_r1}@CA /{chain}:{int_r2}@CA",
        ]

    lines += [
        f"",
        f"# ── Residue labels on gate tips ──────────────────────────────────────────",
    ]

    all_tip_res = list(dict.fromkeys(nb_ext_res + cb_ext_res + nb_int_res + cb_int_res))
    if all_tip_res:
        spec = residue_spec(chain, all_tip_res)
        lines += [
            f"label {spec} residues text {{0.label_one_letter_code}}{{0.number}} height 2",
        ]

    lines += [
        f"",
        f"lighting soft",
        f"view",
    ]

    return lines


# ── Main ───────────────────────────────────────────────────────────────────────

args_protein = ""   # module-level for comment generation inside build_cxc


def main():
    global args_protein
    args = parse_args()
    args_protein = args.protein

    topology = json.loads(Path(args.topology).read_text())
    if args.protein not in topology:
        print(f"[gate_cxc] ERROR: protein '{args.protein}' not in topology", file=sys.stderr)
        sys.exit(1)

    raw_helices = None
    for tool in ("DEEPTMHMM", "UNIPROT", "PHOBIUS", "TMHMM"):
        entry = topology[args.protein].get(tool)
        if entry:
            raw_helices = entry
            break

    if raw_helices is None:
        print(f"[gate_cxc] ERROR: no TM helices found for {args.protein}", file=sys.stderr)
        sys.exit(1)

    sorted_helices = sorted(raw_helices, key=lambda h: h["start"])
    n_tm = len(sorted_helices)

    if n_tm < 12:
        print(f"[gate_cxc] ERROR: only {n_tm} TM helices — gate distances need ≥12", file=sys.stderr)
        sys.exit(1)

    structure = gemmi.read_structure(args.cif)
    model     = structure[0]
    chain     = longest_chain(model)

    cif_abs = str(Path(args.cif).resolve())
    lines   = build_cxc(cif_abs, chain, sorted_helices, model)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"[gate_cxc] Written {out}  (chain={chain}, {n_tm} TM helices)")


if __name__ == "__main__":
    main()
