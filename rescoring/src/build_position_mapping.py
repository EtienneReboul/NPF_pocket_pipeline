#!/usr/bin/env python3
"""
src/build_position_mapping.py
==============================
Reconstruct, for every high-confidence (hc) protein, the per-protein residue
number (`resnr`, in the same full-sequence numbering used by PLIP / the
Boltz-2 poses) at each of the 35 LDA "position" indices used throughout
NPF_LDA_kernel's `hc_lda_loadings.tsv` / `position_residues.tsv`.

Why this is needed: those "position" indices are 0-based-column-turned-1-based
indices into a fixed HMMER/Stockholm alignment (cd17351 profile HMM), NOT raw
ascending residue numbers — two different proteins' residue 158 is not
necessarily "the same" pocket position. See
data/lda_kernel_inputs/PROVENANCE.md for the full derivation and how this
script was validated (exact letter-for-letter reconstruction of every hc
protein's `pocket_sites_cdd_msa.tsv` row).

Method
------
1. Load the cached Stockholm alignment (`npf_aligned.sto`).
2. Map the anchor protein's (NPF6.1_Q9LYR6, the corpus's one cd17351
   root-entry match) own binding-site residues to alignment columns —
   this reproduces the exact 35 columns ("positions") used by
   `extract_cdd_msa.py` in NPF_LDA_kernel.
3. For every hc protein, invert the alignment at those 35 columns to get its
   own `resnr` (or NaN if that protein has a gap at that column).
4. Validate: re-derive each protein's 35-letter pocket string from its own
   FASTA + the reconstructed resnr and assert it matches
   `pocket_sites_cdd_msa.tsv` exactly. Fails loudly on any mismatch.

Output: data/position_resnr_map.csv (protein, position, resnr)
"""
import re
import sys
from pathlib import Path

from Bio import AlignIO

import config

_UNIPROT_RE = re.compile(
    r"\b([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})\b"
)


def extract_uid(header: str) -> str | None:
    m = _UNIPROT_RE.search(header)
    return m.group(0) if m else None


def seq_pos_to_cols(gapped_seq: str, target_positions: set[int]) -> list[int]:
    """1-based ungapped seq positions -> sorted 0-based alignment column indices."""
    cols, seq_pos = [], 0
    for col, char in enumerate(gapped_seq):
        if char not in ("-", "."):
            seq_pos += 1
            if seq_pos in target_positions:
                cols.append(col)
    return sorted(cols)


def cols_to_seq_pos(gapped_seq: str, cols: list[int]) -> dict[int, int | None]:
    """0-based alignment column -> 1-based ungapped seq position (None if gap there)."""
    col_set = set(cols)
    out: dict[int, int | None] = {}
    seq_pos = 0
    for col, char in enumerate(gapped_seq):
        is_gap = char in ("-", ".")
        if not is_gap:
            seq_pos += 1
        if col in col_set:
            out[col] = None if is_gap else seq_pos
    return out


def load_alignment() -> tuple[dict[str, str], dict[str, str]]:
    """Returns (uid -> gapped_seq, uid -> pipeline_name)."""
    aln = AlignIO.read(str(config.LDA_INPUTS_DIR / "npf_aligned.sto"), "stockholm")
    uid_to_seq, uid_to_name = {}, {}
    for rec in aln:
        header = f"{rec.id} {rec.description}"
        uid = extract_uid(header)
        if uid is None:
            continue
        uid_to_seq[uid] = str(rec.seq)
        m = re.search(r"GN=(\S+)", header)
        if m:
            uid_to_name[uid] = f"{m.group(1)}_{uid}"
    return uid_to_seq, uid_to_name


def hc_protein_names() -> list[str]:
    lines = (config.LDA_INPUTS_DIR / "hc_labels.tsv").read_text().splitlines()
    return [line.split()[0] for line in lines if line.strip()]


def load_fasta(protein: str) -> str:
    text = (config.SEQUENCES_DIR / f"{protein}.fasta").read_text()
    return "".join(l.strip() for l in text.splitlines() if not l.startswith(">"))


def load_expected_pockets() -> dict[str, str]:
    expected = {}
    for line in (config.LDA_INPUTS_DIR / "pocket_sites_cdd_msa.tsv").read_text().splitlines()[1:]:
        if not line.strip():
            continue
        name, pocket = line.split("\t")
        expected[name] = pocket
    return expected


def main():
    uid_to_seq, uid_to_name = load_alignment()

    if config.ANCHOR_UNIPROT_ID not in uid_to_seq:
        sys.exit(f"Anchor {config.ANCHOR_UNIPROT_ID} not found in alignment.")

    anchor_resnr = sorted(
        int(x)
        for x in (config.INTERPRO_DIR / f"{config.ANCHOR_PROTEIN}_binding_site_residues.txt")
        .read_text()
        .strip()
        .split(",")
    )
    best_cols = seq_pos_to_cols(uid_to_seq[config.ANCHOR_UNIPROT_ID], set(anchor_resnr))
    if len(best_cols) != len(anchor_resnr):
        sys.exit(
            f"Anchor mapping incomplete: {len(best_cols)}/{len(anchor_resnr)} "
            "binding-site residues mapped to alignment columns."
        )
    print(f"[build_position_mapping] {len(best_cols)} positions anchored on {config.ANCHOR_PROTEIN}")

    names = hc_protein_names()
    expected_pockets = load_expected_pockets()

    rows = []
    n_ok, n_bad, n_missing = 0, 0, 0
    for name in names:
        uid = name.split("_")[-1]
        if uid not in uid_to_seq:
            print(f"  [!] {name}: not in Stockholm alignment — skipped", file=sys.stderr)
            n_missing += 1
            continue

        mapping = cols_to_seq_pos(uid_to_seq[uid], best_cols)
        resnrs = [mapping[c] for c in best_cols]

        seq = load_fasta(name)
        derived_pocket = "".join(seq[r - 1] if r is not None else "X" for r in resnrs)
        expected = expected_pockets.get(name)
        if expected is None:
            sys.exit(f"No expected pocket string for {name} in pocket_sites_cdd_msa.tsv")
        if derived_pocket != expected:
            print(f"  [!] MISMATCH {name}: derived={derived_pocket} expected={expected}", file=sys.stderr)
            n_bad += 1
            continue
        n_ok += 1

        for position, resnr in enumerate(resnrs, start=1):
            rows.append({"protein": name, "position": position, "resnr": resnr})

    print(f"[build_position_mapping] validated {n_ok}/{len(names)} proteins "
          f"({n_bad} mismatched, {n_missing} missing from alignment)")
    if n_bad or n_missing:
        sys.exit("Validation failed for one or more proteins — see warnings above.")

    import pandas as pd

    df = pd.DataFrame(rows, columns=["protein", "position", "resnr"])
    df.to_csv(config.POSITION_RESNR_MAP_CSV, index=False)
    print(f"[build_position_mapping] wrote {config.POSITION_RESNR_MAP_CSV} "
          f"({len(df)} rows, {df.protein.nunique()} proteins)")


if __name__ == "__main__":
    main()
