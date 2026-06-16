#!/usr/bin/env python3
"""
select_templates.py  --  NPF/POT-aware template selection and editing.

Implements the strategy discussed for the NPF_pocket_pipeline (Stage 3):

  1. EDIT   each candidate structure:
              - keep a single protomer (NRT1.1 etc. are dimers)
              - strip waters; remove ligand for *_apo bins, keep it for *_holo
              - MASK the TM6-TM7 insertion (plant ICD / bacterial HA-HB) so a
                mixed-family cocktail only constrains the shared 12-TM MFS core
  2. VERIFY each candidate's conformational state GEOMETRICALLY, not by label:
              - extracellular gate-tip distance  (TM1,TM2  vs TM7,TM8)
              - intracellular gate-tip distance  (TM4,TM5  vs TM10,TM11)
              - TR angle                          (TM2 axis vs TM8 axis)
              classified against the GTR1 outward (9UI1) / inward (9UI6) anchors.
  3. SCORE  each candidate against each target NPF sequence over the gate
              helices (sequence identity/coverage in the gate columns).
  4. EMIT   an enriched per-bin (optionally per-target) cocktail in your
              config.yaml schema, plus the edited template files.

Gate topology is the MFS-universal "first two helices of each inverted-topology
repeat" rule, confirmed for a native NPF by Yan et al. (GTR1, Cell Discovery
2026): intracellular gate = TM4-5 / TM10-11, extracellular gate = TM1-2 / TM7-8.
The GTR1 ICD to mask is Pro283-Thr358 (Yan et al.).

Dependencies: gemmi, numpy, biopython.  DSSP refinement (optional) needs `mkdssp`.

The only thing you must curate once: the TM1-12 residue ranges of the reference
structures in REFERENCE_ANNOTATIONS. Run  `--annotate REF.cif`  to get a DSSP
draft, then confirm TM numbering (excluding HA/HB and the ICD).
"""
from __future__ import annotations
import argparse, json, os, sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import gemmi

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
# Topological gate definition (MFS-universal; TM numbers, not residues).
EXTRACELLULAR_GATE = (("TM1", "TM2"), ("TM7", "TM8"))     # (N-bundle, C-bundle)
INTRACELLULAR_GATE = (("TM4", "TM5"), ("TM10", "TM11"))
TR_HELICES = ("TM2", "TM8")                               # Qureshi-style TR angle

TIP_WINDOW = 4          # n terminal CA used to define each helix tip
TRIM_ENDS = 2           # CA trimmed from each helix end before axis fitting
OCCLUDED_FACTOR = 0.65  # a gate counts as "closed" if < this * open-anchor value

# Candidate registry. tier: A=plant NPF, B=other POT, C=adjacent nitrate-MFS,
# D=distant MFS. nominal_state is only a hint; the script re-derives the state.
# insert = (start, end) residues of the TM6-TM7 insertion to mask (or None).
DEFAULT_CANDIDATES = {
    # ---- plant NPF (tier A) ----
    "9UI1": dict(family="NPF",  tier="A", nominal="outward_open",     ligand=None,   ref="GTR1"),
    "9UI6": dict(family="NPF",  tier="A", nominal="inward_open",      ligand=None,   ref="GTR1"),
    "9UIF": dict(family="NPF",  tier="A", nominal="inward_occluded",  ligand="4MTB", ref="GTR1"),
    "9UIT": dict(family="NPF",  tier="A", nominal="inward_occluded",  ligand="3IMG", ref="GTR1"),
    "4OH3": dict(family="NPF",  tier="A", nominal="inward_open",      ligand=None,   ref="NRT1.1"),
    # ---- other POT (tier B) ----
    "7S8U": dict(family="POT",  tier="B", nominal="inward_open",      ligand=None,   ref="GTR1"),
    "7PN1": dict(family="POT",  tier="B", nominal="outward_open",     ligand=None,   ref="GTR1"),
    "2XUT": dict(family="POT",  tier="B", nominal="occluded",         ligand="ALF",  ref="GTR1"),
    "4IKZ": dict(family="POT",  tier="B", nominal="inward_occluded",  ligand="ALF",  ref="GTR1"),
    "4UVM": dict(family="POT",  tier="B", nominal="inward_open",      ligand=None,   ref="GTR1"),
    # ---- distant MFS fallbacks (tier D) ----
    "6RW3": dict(family="SP",   tier="D", nominal="occluded",         ligand="GLC",  ref="GTR1"),
    "4YBQ": dict(family="SP",   tier="D", nominal="outward_open",     ligand=None,   ref="GTR1"),
    "4GBY": dict(family="SP",   tier="D", nominal="outward_occluded", ligand="XYL",  ref="GTR1"),
    "4ZW9": dict(family="SP",   tier="D", nominal="outward_occluded", ligand="GLC",  ref="GTR1"),
}

# Reference TM1-12 ranges (inclusive author residue numbers) + the ICD/HA-HB
# insertion to mask + a residue known to be CYTOPLASMIC (orients the membrane
# normal; for NPF both termini are cytoplasmic so the first residue works).
# GTR1 ranges are anchored to residues each paper assigns per helix; REFINE with
# --annotate before production use. NRT1.1 is a placeholder skeleton.
REFERENCE_ANNOTATIONS = {
    "GTR1": dict(
        # approximate, paper-anchored; verify with DSSP (--annotate 9UI6.cif)
        tms={"TM1": (62, 90),   "TM2": (108, 136), "TM3": (140, 168),
             "TM4": (176, 208), "TM5": (214, 250), "TM6": (256, 282),
             "TM7": (359, 392), "TM8": (396, 426), "TM9": (430, 458),
             "TM10": (480, 515), "TM11": (520, 545), "TM12": (558, 588)},
        insert=(283, 358),       # ICD Pro283-Thr358 (Yan et al.) -- masked
        cyto_residue=62,         # N-terminal resolved residue (cytoplasmic)
    ),
    "NRT1.1": dict(
        tms={"TM1": (40, 68),   "TM2": (90, 118),  "TM3": (124, 152),
             "TM4": (160, 192), "TM5": (198, 232), "TM6": (236, 264),
             "TM7": (300, 332), "TM8": (336, 366), "TM9": (372, 398),
             "TM10": (420, 455), "TM11": (462, 488), "TM12": (500, 530)},
        insert=(265, 299),       # interdomain loop -- refine
        cyto_residue=40,
    ),
}

AA3to1 = gemmi.ResidueInfo


# --------------------------------------------------------------------------- #
# Structure I/O + editing
# --------------------------------------------------------------------------- #
def load(path: str) -> gemmi.Structure:
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    return st


def largest_protein_chain(model: gemmi.Model) -> gemmi.Chain:
    best, best_len = None, -1
    for ch in model:
        poly = ch.get_polymer()
        n = sum(1 for r in poly if r.find_atom("CA", "*"))
        if n > best_len:
            best, best_len = ch, n
    if best is None:
        raise ValueError("no protein chain found")
    return best


def edit_template(st: gemmi.Structure, ann: dict, keep_ligand: str | None,
                  mask_insert: bool = True) -> gemmi.Structure:
    """Single protomer, drop waters, mask the TM6-TM7 insert, set apo/holo."""
    model = st[0]
    keep_chain = largest_protein_chain(model).name
    out = gemmi.Structure()
    out.spacegroup_hm = st.spacegroup_hm
    out.cell = st.cell
    om = gemmi.Model("1")
    new_chain = gemmi.Chain(keep_chain)
    ins = ann.get("insert") if mask_insert else None
    for ch in model:
        if ch.name != keep_chain:
            # keep only a requested ligand if it sits on the protomer's chain
            continue
        for res in ch:
            info = gemmi.find_tabulated_residue(res.name)
            is_water = res.is_water()
            is_aa = info and info.is_amino_acid()
            if is_water:
                continue
            if is_aa:
                num = res.seqid.num
                if ins and ins[0] <= num <= ins[1]:
                    continue                      # mask ICD / HA-HB
                new_chain.add_residue(res)
            else:                                  # heteroatom / ligand
                if keep_ligand and res.name == keep_ligand:
                    new_chain.add_residue(res)
    om.add_chain(new_chain)
    out.add_model(om)
    out.setup_entities()
    return out


def write(st: gemmi.Structure, path: str):
    p = Path(path)
    if p.suffix.lower() in (".cif", ".mmcif"):
        st.make_mmcif_document().write_file(str(p))
    else:
        st.write_pdb(str(p))


# --------------------------------------------------------------------------- #
# Sequence handling + reference->candidate residue mapping
# --------------------------------------------------------------------------- #
def chain_sequence(chain: gemmi.Chain):
    """Return (one_letter_seq, [seqid.num ...]) for CA-bearing residues."""
    seq, nums = [], []
    for res in chain:
        if res.find_atom("CA", "*"):
            info = gemmi.find_tabulated_residue(res.name)
            if info and info.is_amino_acid():
                seq.append(gemmi.find_tabulated_residue(res.name).one_letter_code.upper())
                nums.append(res.seqid.num)
    return "".join(seq), nums


def align_map(ref_seq: str, ref_nums, qry_seq: str, qry_nums):
    """Global align; return dict ref_resnum -> qry_resnum for aligned columns."""
    from Bio import Align
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    aligner.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")
    aln = aligner.align(ref_seq, qry_seq)[0]
    mapping = {}
    for (r0, r1), (q0, q1) in zip(aln.aligned[0], aln.aligned[1]):
        for k in range(r1 - r0):
            mapping[ref_nums[r0 + k]] = qry_nums[q0 + k]
    return mapping


def transfer_tms(ref_ann: dict, ref_chain, qry_chain) -> dict:
    """Map reference TM1-12 ranges onto query residue numbers by seq alignment."""
    rseq, rnum = chain_sequence(ref_chain)
    qseq, qnum = chain_sequence(qry_chain)
    m = align_map(rseq, rnum, qseq, qnum)
    qtms = {}
    for tm, (lo, hi) in ref_ann["tms"].items():
        mapped = [m[r] for r in range(lo, hi + 1) if r in m]
        if len(mapped) >= 4:
            qtms[tm] = (min(mapped), max(mapped))
    return qtms


# --------------------------------------------------------------------------- #
# Geometry: helix axes, tips, gate distances, TR angle
# --------------------------------------------------------------------------- #
def ca_coords(chain: gemmi.Chain, lo: int, hi: int) -> np.ndarray:
    pts = []
    for res in chain:
        if lo <= res.seqid.num <= hi:
            a = res.find_atom("CA", "*")
            if a:
                pts.append([a.pos.x, a.pos.y, a.pos.z])
    return np.asarray(pts, float)


def helix_axis(coords: np.ndarray):
    """Return (centroid, unit_dir, tipA, tipB) from CA coords via SVD."""
    c = coords[TRIM_ENDS:len(coords) - TRIM_ENDS] if len(coords) > 2 * TRIM_ENDS + 2 else coords
    centroid = c.mean(0)
    _, _, vt = np.linalg.svd(c - centroid)
    direction = vt[0] / np.linalg.norm(vt[0])
    proj = (c - centroid) @ direction
    order = np.argsort(proj)
    tipA = c[order[:TIP_WINDOW]].mean(0)
    tipB = c[order[-TIP_WINDOW:]].mean(0)
    return centroid, direction, tipA, tipB


def membrane_normal(axes: list[np.ndarray], chain, cyto_resnum: int):
    """Average TM direction, signed so +normal points toward extracellular side."""
    ref = axes[0]
    summed = sum(a if a @ ref >= 0 else -a for a in axes)
    n = summed / np.linalg.norm(summed)
    cyto = ca_coords(chain, cyto_resnum, cyto_resnum)
    if len(cyto):                       # flip so cytoplasmic residue is on -side
        if (cyto[0] - np.mean([a for a in [0]])) is not None:
            pass
    return n


def helix_geometry(chain, tms: dict, cyto_resnum: int):
    """Compute per-helix axis/tips and an oriented membrane normal."""
    geom, axes = {}, []
    for tm, (lo, hi) in tms.items():
        coords = ca_coords(chain, lo, hi)
        if len(coords) < TIP_WINDOW + 2:
            continue
        cen, dirn, tA, tB = helix_axis(coords)
        geom[tm] = dict(centroid=cen, dir=dirn, tipA=tA, tipB=tB)
        axes.append(dirn)
    # orient normal so cytoplasmic reference residue is on the negative side
    ref = axes[0]
    n = sum(a if a @ ref >= 0 else -a for a in axes)
    n = n / np.linalg.norm(n)
    cyto = ca_coords(chain, cyto_resnum, cyto_resnum)
    allca = np.vstack([np.array([g["centroid"] for g in geom.values()])])
    centre = allca.mean(0)
    if len(cyto) and (cyto[0] - centre) @ n > 0:
        n = -n                          # ensure +n = extracellular
    return geom, n


def side_tip(g, normal, extracellular=True):
    """Pick the helix tip on the requested membrane side."""
    pa = g["tipA"] @ normal
    pb = g["tipB"] @ normal
    hi, lo = (g["tipA"], g["tipB"]) if pa >= pb else (g["tipB"], g["tipA"])
    return hi if extracellular else lo


def gate_distance(geom, normal, gate, extracellular: bool):
    (n_helices, c_helices) = gate
    n_tips = [side_tip(geom[h], normal, extracellular) for h in n_helices if h in geom]
    c_tips = [side_tip(geom[h], normal, extracellular) for h in c_helices if h in geom]
    if not n_tips or not c_tips:
        return float("nan")
    return min(np.linalg.norm(a - b) for a in n_tips for b in c_tips)


def tr_angle(geom):
    a, b = TR_HELICES
    if a not in geom or b not in geom:
        return float("nan")
    u, v = geom[a]["dir"], geom[b]["dir"]
    cos = abs(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def metrics_for(chain, tms, cyto_resnum):
    geom, n = helix_geometry(chain, tms, cyto_resnum)
    return dict(
        ext_gate=gate_distance(geom, n, EXTRACELLULAR_GATE, extracellular=True),
        int_gate=gate_distance(geom, n, INTRACELLULAR_GATE, extracellular=False),
        tr_angle=tr_angle(geom),
    )


# --------------------------------------------------------------------------- #
# State classification (calibrated against outward/inward anchors)
# --------------------------------------------------------------------------- #
def classify(m: dict, anchors: dict) -> str:
    """anchors = {'outward': metrics, 'inward': metrics}."""
    ext, intr = m["ext_gate"], m["int_gate"]
    out_ext = anchors["outward"]["ext_gate"]   # open extracellular
    in_int = anchors["inward"]["int_gate"]     # open intracellular
    ext_open = ext > OCCLUDED_FACTOR * out_ext
    int_open = intr > OCCLUDED_FACTOR * in_int
    if ext_open and not int_open:
        return "outward_open"
    if int_open and not ext_open:
        return "inward_open"
    if not ext_open and not int_open:
        return "occluded"
    return "ambiguous_both_open"   # likely an escaped / distorted model


# --------------------------------------------------------------------------- #
# Target scoring over gate helices
# --------------------------------------------------------------------------- #
def gate_alignment_score(ref_ann, ref_chain, target_seq: str, target_id: str):
    """Identity of a target NPF sequence to the reference over gate-helix columns."""
    rseq, rnum = chain_sequence(ref_chain)
    # synthetic numbering for a bare target sequence
    tnum = list(range(1, len(target_seq) + 1))
    m = align_map(rseq, rnum, target_seq, tnum)
    gate_tms = [h for pair in (EXTRACELLULAR_GATE + INTRACELLULAR_GATE) for h in pair]
    hit = tot = 0
    rseq_by_num = dict(zip(rnum, rseq))
    for tm in gate_tms:
        lo, hi = ref_ann["tms"][tm]
        for r in range(lo, hi + 1):
            if r in m and r in rseq_by_num:
                tot += 1
                if rseq_by_num[r] == target_seq[m[r] - 1]:
                    hit += 1
    return hit / tot if tot else 0.0


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    pdb: str
    tier: str
    family: str
    nominal: str
    derived: str = "NA"
    metrics: dict = field(default_factory=dict)
    edited_apo: str = ""
    edited_holo: str = ""


def run(args):
    cand = DEFAULT_CANDIDATES
    if args.candidates:
        cand = json.loads(Path(args.candidates).read_text())
    struct_dir = Path(args.structures)
    out_dir = Path(args.out)
    (out_dir / "edited").mkdir(parents=True, exist_ok=True)

    # load reference chains
    refs = {}
    for name, ann in REFERENCE_ANNOTATIONS.items():
        rp = struct_dir / f"{args.ref_files.get(name, name)}.cif"
        if rp.exists():
            refs[name] = (ann, largest_protein_chain(load(str(rp))[0]))

    # anchors for state calibration (GTR1 outward 9UI1, inward 9UI6)
    anchors = {}
    for key, pdb in (("outward", args.outward_anchor), ("inward", args.inward_anchor)):
        sp = struct_dir / f"{pdb}.cif"
        if sp.exists() and "GTR1" in refs:
            ann = REFERENCE_ANNOTATIONS["GTR1"]
            ch = largest_protein_chain(load(str(sp))[0])
            tms = ann["tms"] if pdb in (args.ref_files.get("GTR1"), "9UI6") else \
                transfer_tms(*refs["GTR1"], ch)
            anchors[key] = metrics_for(ch, ann["tms"], ann["cyto_residue"])
    if len(anchors) < 2:
        print("WARN: missing outward/inward anchors; state verification disabled",
              file=sys.stderr)

    results = []
    for pdb, meta in cand.items():
        sp = struct_dir / f"{pdb}.cif"
        if not sp.exists():
            print(f"skip {pdb}: not in {struct_dir}", file=sys.stderr)
            continue
        st = load(str(sp))
        ref_name = meta.get("ref", "GTR1")
        if ref_name not in refs:
            print(f"skip {pdb}: reference {ref_name} not loaded", file=sys.stderr)
            continue
        ref_ann, ref_chain = refs[ref_name]
        qchain = largest_protein_chain(st[0])
        tms = ref_ann["tms"] if pdb == args.ref_files.get(ref_name) \
            else transfer_tms(ref_ann, ref_chain, qchain)

        res = Result(pdb, meta["tier"], meta["family"], meta["nominal"])
        if anchors and tms:
            res.metrics = metrics_for(qchain, tms, _cyto(ref_ann, ref_chain, qchain))
            res.derived = classify(res.metrics, anchors)

        # edit: apo + (if applicable) holo variants
        apo = edit_template(st, ref_ann, keep_ligand=None)
        ap = out_dir / "edited" / f"{pdb}_apo_core.cif"
        write(apo, str(ap)); res.edited_apo = str(ap)
        if meta.get("ligand"):
            holo = edit_template(st, ref_ann, keep_ligand=meta["ligand"])
            hp = out_dir / "edited" / f"{pdb}_holo_core.cif"
            write(holo, str(hp)); res.edited_holo = str(hp)
        results.append(res)

    # target scoring (optional)
    target_scores = {}
    if args.targets and refs:
        ref_ann, ref_chain = refs.get("GTR1", next(iter(refs.values())))
        for rec in _read_fasta(args.targets):
            tid, tseq = rec
            target_scores[tid] = gate_alignment_score(ref_ann, ref_chain, tseq, tid)

    _emit(results, anchors, target_scores, out_dir, args)


def _cyto(ref_ann, ref_chain, qchain):
    """Map reference cytoplasmic residue onto the query for normal orientation."""
    rseq, rnum = chain_sequence(ref_chain)
    qseq, qnum = chain_sequence(qchain)
    m = align_map(rseq, rnum, qseq, qnum)
    return m.get(ref_ann["cyto_residue"], qnum[0] if qnum else 1)


def _read_fasta(path):
    name, buf = None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            if name:
                yield name, "".join(buf)
            name, buf = line[1:].split()[0], []
        else:
            buf.append(line.strip())
    if name:
        yield name, "".join(buf)


BIN_MAP = {  # derived state -> your config bin(s)
    "outward_open": ["outward_open_apo"],
    "occluded": ["occluded_apo", "occluded_holo"],
    "inward_open": ["inward_open_apo", "inward_occluded_holo"],
}


def _emit(results, anchors, target_scores, out_dir, args):
    # group verified templates by config bin, ordered by tier then nominal match
    bins = {b: [] for b in
            ["outward_open_apo", "outward_occluded_holo", "occluded_apo",
             "occluded_holo", "inward_occluded_holo", "inward_open_apo"]}
    rows = []
    for r in results:
        state = r.derived if r.derived not in ("NA", "ambiguous_both_open") else r.nominal
        target_bins = BIN_MAP.get(state, [])
        for b in target_bins:
            bins[b].append((r.tier, r.pdb))
        flag = ""
        if r.derived not in ("NA",) and r.derived != r.nominal.split("_occluded")[0] \
           and r.nominal not in r.derived:
            flag = f"  <-- label '{r.nominal}' != derived '{r.derived}'"
        m = r.metrics
        rows.append(f"  {r.pdb} [{r.tier}/{r.family}] nominal={r.nominal} "
                    f"derived={r.derived} "
                    f"ext={m.get('ext_gate', float('nan')):.1f} "
                    f"int={m.get('int_gate', float('nan')):.1f} "
                    f"tr={m.get('tr_angle', float('nan')):.1f}{flag}")

    yaml_lines = ["templates:", "  conformations:"]
    for b, items in bins.items():
        items = sorted(set(items))                 # tier A first
        codes = [pdb for _, pdb in items]
        yaml_lines.append(f"    {b}:")
        yaml_lines.append(f"      pdb_codes: {codes}")
    (out_dir / "templates_selected.yaml").write_text("\n".join(yaml_lines) + "\n")

    report = ["# Template verification report", "", "## per-candidate metrics"]
    report += rows
    if target_scores:
        report += ["", "## gate-region alignment to targets (identity)"]
        for tid, sc in sorted(target_scores.items(), key=lambda x: -x[1]):
            report.append(f"  {tid}: {sc:.2f}")
    (out_dir / "report.txt").write_text("\n".join(report) + "\n")
    print("\n".join(rows))
    print(f"\nwrote {out_dir/'templates_selected.yaml'} and {out_dir/'report.txt'}")


def annotate_reference(cif_path):
    """Print a DSSP draft of helical segments to seed REFERENCE_ANNOTATIONS."""
    import subprocess, tempfile
    st = load(cif_path)
    pdb_tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False).name
    st.write_pdb(pdb_tmp)
    try:
        out = subprocess.run(["mkdssp", pdb_tmp], capture_output=True, text=True).stdout
    except FileNotFoundError:
        print("mkdssp not found; install dssp or annotate manually", file=sys.stderr)
        return
    segs, cur = [], None
    started = False
    for line in out.splitlines():
        if line.startswith("  #  RESIDUE"):
            started = True; continue
        if not started or len(line) < 17:
            continue
        ss = line[16]
        try:
            num = int(line[5:10])
        except ValueError:
            continue
        if ss in "HGI":
            cur = (cur[0], num) if cur else (num, num)
        else:
            if cur:
                segs.append(cur); cur = None
    if cur:
        segs.append(cur)
    print("# helical segments (label TM1..TM12; drop HA/HB and the ICD):")
    for i, (a, b) in enumerate(segs, 1):
        if b - a >= 12:
            print(f"  seg{i}: ({a}, {b})")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--structures", default="data/templates",
                   help="dir with <PDBID>.cif candidate + reference structures")
    p.add_argument("--out", default="results/template_selection")
    p.add_argument("--candidates", help="optional JSON overriding DEFAULT_CANDIDATES")
    p.add_argument("--targets", help="FASTA of the 53 NPF target sequences")
    p.add_argument("--outward-anchor", default="9UI1")
    p.add_argument("--inward-anchor", default="9UI6")
    p.add_argument("--annotate", help="run DSSP on a reference cif and exit")
    args = p.parse_args()
    args.ref_files = {"GTR1": "9UI6", "NRT1.1": "4OH3"}
    if args.annotate:
        annotate_reference(args.annotate); return
    run(args)


if __name__ == "__main__":
    main()
