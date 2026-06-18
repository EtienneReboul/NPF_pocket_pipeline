#!/usr/bin/env python3
"""
scripts/compute_tm_angle.py
============================
Computes the angle between the TM2 and TM8 helix principal axes for one or all
Boltz-2 predicted structures belonging to one protein × conformation.

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
   flag if TM2 or TM8 helix content < --min-helix-frac in the prediction.
4. Extract Cα coordinates for TM2 and TM8 from the raw Boltz-2 CIF.
5. Compute each helix principal axis via truncated SVD on the mean-centred
   Cα coordinates (first right-singular vector).
6. Return the unsigned angle (0–90°) between the two axis vectors.

Usage — single-sample mode (original):
    python scripts/compute_tm_angle.py \\
        --cif           results/boltz/.../model_0.cif \\
        --dssp          results/dssp/.../model_0/dssp.json \\
        --topology      data/interpro/tm_topology_summary.json \\
        --protein       NPF1.1_Q9LYD5 \\
        --conformation  outward_open_apo \\
        --sample-id     model_0 \\
        --output        results/tm_angles/.../model_0/angle.csv

Usage — batch mode (packaging per protein × conformation):
    python scripts/compute_tm_angle.py \\
        --batch-manifest results/boltz/{protein}/{conformation}/cif_manifest.json \\
        --dssp-dir       results/dssp/{protein}/{conformation} \\
        --topology       data/interpro/tm_topology_summary.json \\
        --protein        NPF1.1_Q9LYD5 \\
        --conformation   outward_open_apo \\
        --output         results/tm_angles/{protein}/{conformation}/angles.csv \\
        --min-helix-frac 0.5

In batch mode --batch-manifest is the Boltz cif_manifest.json (JSON list of CIF paths).
The dssp.json for each sample is read from <dssp-dir>/<sample_id>/dssp.json.
All rows are written directly to --output (the aggregated angles.csv), replacing the
separate per-sample angle.csv + collect step.
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import gemmi
import numpy as np

TIP_WINDOW = 4


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    # single-sample mode
    p.add_argument("--cif")
    p.add_argument("--dssp")
    p.add_argument("--sample-id")
    # batch mode
    p.add_argument("--batch-manifest", help="Path to Boltz cif_manifest.json")
    p.add_argument("--dssp-dir",       help="Directory containing <sample_id>/dssp.json files")
    # shared
    p.add_argument("--topology",       required=True)
    p.add_argument("--protein",        required=True)
    p.add_argument("--conformation",   required=True)
    p.add_argument("--output",         required=True)
    p.add_argument("--min-helix-frac", type=float, default=0.5)
    return p.parse_args()


# ── Geometry helpers ───────────────────────────────────────────────────────────

def helix_axis(ca_coords: np.ndarray) -> np.ndarray:
    centred = ca_coords - ca_coords.mean(axis=0)
    _, _, Vt = np.linalg.svd(centred, full_matrices=False)
    return Vt[0]


def helix_tip(ca: np.ndarray, n_terminal: bool) -> np.ndarray:
    n = min(TIP_WINDOW, len(ca))
    if n == 0:
        return np.empty((0, 3))
    return ca[:n] if n_terminal else ca[-n:]


def unsigned_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(np.degrees(np.arccos(np.clip(abs(cos_theta), 0.0, 1.0))))


# ── Cα extraction from CIF ────────────────────────────────────────────────────

def get_ca_coords(structure_model: gemmi.Model, chain_name: str,
                  start_res: int, end_res: int) -> np.ndarray:
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
    best = max(model, key=lambda c: len(list(c)))
    return best.name


# ── Result schema ─────────────────────────────────────────────────────────────

RESULT_FIELDS = [
    "protein", "conformation", "sample_id",
    "chain",
    "tm2_start", "tm2_end", "n_ca_tm2",
    "tm8_start", "tm8_end", "n_ca_tm8",
    "tm2_helix_frac", "tm8_helix_frac",
    "angle_deg",
    "ext_gate_A", "int_gate_A",
    "n_tm_helices", "tool_used",
    "status",
]


def write_rows(path: str, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({f: row.get(f) for f in RESULT_FIELDS})


# ── Per-sample computation ────────────────────────────────────────────────────

def compute_one(
    cif_path: str,
    dssp_path: str,
    protein: str,
    conformation: str,
    sample_id: str,
    sorted_helices: list,
    n_tm: int,
    tool_used: str,
    min_helix_frac: float,
) -> dict:
    """Compute TM2/TM8 angle for a single CIF+DSSP pair. Returns a RESULT_FIELDS dict."""
    base = dict(
        protein=protein,
        conformation=conformation,
        sample_id=sample_id,
        n_tm_helices=n_tm,
        tool_used=tool_used,
        angle_deg=None,
        status="unknown",
    )

    tm2 = sorted_helices[1]
    tm8 = sorted_helices[7]
    base.update(
        tm2_start=tm2["start"], tm2_end=tm2["end"],
        tm8_start=tm8["start"], tm8_end=tm8["end"],
    )

    # DSSP sanity check
    dssp_data   = json.loads(Path(dssp_path).read_text())
    helix_res   = {(r["chain"], r["resnum"]) for r in dssp_data if r["is_helix"]}
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
    if tm2_hfrac < min_helix_frac:
        dssp_flag += f"low_tm2_helix({tm2_hfrac:.2f})"
    if tm8_hfrac < min_helix_frac:
        dssp_flag += f"low_tm8_helix({tm8_hfrac:.2f})"

    # Cα coordinates
    structure  = gemmi.read_structure(cif_path)
    model      = structure[0]
    chain_name = longest_chain_name(model)
    base["chain"] = chain_name

    ca_tm2 = get_ca_coords(model, chain_name, tm2["start"], tm2["end"])
    ca_tm8 = get_ca_coords(model, chain_name, tm8["start"], tm8["end"])
    base.update(n_ca_tm2=len(ca_tm2), n_ca_tm8=len(ca_tm8))

    if len(ca_tm2) < 4:
        print(f"[tm_angle] {protein}/{conformation}/{sample_id}: "
              f"only {len(ca_tm2)} Cα in TM2 — skipping angle")
        return {**base, "status": "insufficient_tm2_ca"}
    if len(ca_tm8) < 4:
        print(f"[tm_angle] {protein}/{conformation}/{sample_id}: "
              f"only {len(ca_tm8)} Cα in TM8 — skipping angle")
        return {**base, "status": "insufficient_tm8_ca"}

    axis2 = helix_axis(ca_tm2)
    axis8 = helix_axis(ca_tm8)
    angle = unsigned_angle(axis2, axis8)

    # Gate distances
    ext_gate = float("nan")
    int_gate = float("nan")
    if n_tm >= 12:
        def _ext_tip(i):
            h  = sorted_helices[i]
            ca = get_ca_coords(model, chain_name, h["start"], h["end"])
            return helix_tip(ca, n_terminal=(i % 2 == 1))

        def _int_tip(i):
            h  = sorted_helices[i]
            ca = get_ca_coords(model, chain_name, h["start"], h["end"])
            return helix_tip(ca, n_terminal=(i % 2 == 0))

        def _tip_set(*idxs, fn):
            parts = [t for i in idxs for t in (fn(i),) if len(t) > 0]
            return np.vstack(parts) if parts else np.empty((0, 3))

        nb_ext = _tip_set(0, 1, fn=_ext_tip)
        cb_ext = _tip_set(6, 7, fn=_ext_tip)
        nb_int = _tip_set(3, 4, fn=_int_tip)
        cb_int = _tip_set(9, 10, fn=_int_tip)

        if len(nb_ext) and len(cb_ext):
            d = np.linalg.norm(nb_ext[:, None, :] - cb_ext[None, :, :], axis=-1)
            ext_gate = round(float(d.min()), 3)
        if len(nb_int) and len(cb_int):
            d = np.linalg.norm(nb_int[:, None, :] - cb_int[None, :, :], axis=-1)
            int_gate = round(float(d.min()), 3)

    status = "ok" if not dssp_flag else f"ok_with_flags:{dssp_flag}"
    print(
        f"[tm_angle] {protein}/{conformation}/{sample_id}: "
        f"chain={chain_name}, TM2={tm2['start']}-{tm2['end']} "
        f"({len(ca_tm2)} Cα, helix={tm2_hfrac:.0%}), "
        f"TM8={tm8['start']}-{tm8['end']} "
        f"({len(ca_tm8)} Cα, helix={tm8_hfrac:.0%}), "
        f"angle={angle:.1f}°  ext_gate={ext_gate:.1f}Å  int_gate={int_gate:.1f}Å  "
        f"status={status}"
    )
    return {
        **base,
        "angle_deg":  round(angle, 3),
        "ext_gate_A": ext_gate,
        "int_gate_A": int_gate,
        "status":     status,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Load topology (shared across all samples for this protein)
    topology = json.loads(Path(args.topology).read_text())
    base_err = dict(
        protein=args.protein,
        conformation=args.conformation,
        angle_deg=None,
    )

    if args.protein not in topology:
        print(f"[tm_angle] ERROR: protein '{args.protein}' not in topology summary")
        write_rows(args.output, [{**base_err, "sample_id": "N/A", "status": "protein_not_in_topology"}])
        sys.exit(0)

    prot_topo = topology[args.protein]
    tool_used = None
    raw_helices = None
    for tool in ("DEEPTMHMM", "UNIPROT", "PHOBIUS", "TMHMM"):
        if tool in prot_topo and prot_topo[tool]:
            tool_used   = tool
            raw_helices = prot_topo[tool]
            break

    if raw_helices is None:
        print(f"[tm_angle] ERROR: no TM helices in topology for {args.protein}")
        write_rows(args.output, [{**base_err, "sample_id": "N/A", "status": "no_tm_helices"}])
        sys.exit(0)

    sorted_helices = sorted(raw_helices, key=lambda h: h["start"])
    n_tm = len(sorted_helices)

    if n_tm < 8:
        print(f"[tm_angle] WARNING: only {n_tm} TM helices for {args.protein}")
        write_rows(args.output, [{**base_err, "sample_id": "N/A",
                                  "n_tm_helices": n_tm, "tool_used": tool_used,
                                  "status": f"only_{n_tm}_tm_helices"}])
        sys.exit(0)

    if args.batch_manifest:
        # Batch mode: process all samples for one protein × conformation
        cif_paths = [Path(p) for p in json.loads(Path(args.batch_manifest).read_text())]
        dssp_dir  = Path(args.dssp_dir)
        rows = []
        for cif in cif_paths:
            sample_id = cif.stem
            dssp_path = str(dssp_dir / sample_id / "dssp.json")
            row = compute_one(
                cif_path=str(cif),
                dssp_path=dssp_path,
                protein=args.protein,
                conformation=args.conformation,
                sample_id=sample_id,
                sorted_helices=sorted_helices,
                n_tm=n_tm,
                tool_used=tool_used,
                min_helix_frac=args.min_helix_frac,
            )
            rows.append(row)
        write_rows(args.output, rows)
        print(f"[tm_angle] Batch complete: {len(rows)} samples → {args.output}")
    else:
        # Single-sample mode (backward compatible)
        if not args.cif or not args.dssp or not args.sample_id:
            print("[tm_angle] ERROR: single mode requires --cif, --dssp, and --sample-id")
            sys.exit(1)
        row = compute_one(
            cif_path=args.cif,
            dssp_path=args.dssp,
            protein=args.protein,
            conformation=args.conformation,
            sample_id=args.sample_id,
            sorted_helices=sorted_helices,
            n_tm=n_tm,
            tool_used=tool_used,
            min_helix_frac=args.min_helix_frac,
        )
        write_rows(args.output, [row])


if __name__ == "__main__":
    main()
