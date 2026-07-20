#!/usr/bin/env python3
"""
scripts/make_default_boltz_input.py
====================================
Standalone stage (outside the protein × conformation matrix):
Generate a "standard" Boltz-2 input YAML per protein, using the top hits
from that protein's own ColabFold PDB70 template search (data/msa/pdb/{protein}.m8)
as templates — no curated (natural/RCSB per-conformation) or synthetic template,
and no pocket constraint.

Produces two YAMLs per protein: apo (protein only) and holo (protein + GA1 ligand).

Usage (called by Snakemake rule `prepare_default_boltz_input`):
    python scripts/make_default_boltz_input.py \\
        --fasta            data/sequences/NPF6.3_Q05085.fasta \\
        --a3m              data/msa/a3m/NPF6.3_Q05085.a3m \\
        --m8               data/msa/pdb/NPF6.3_Q05085.m8 \\
        --protein-name     NPF6.3_Q05085 \\
        --templates-dir    data/templates/default/NPF6.3_Q05085 \\
        --top-n-templates  5 \\
        --output-apo       data/boltz_inputs_default/NPF6.3_Q05085/apo/target.yaml \\
        --output-holo      data/boltz_inputs_default/NPF6.3_Q05085/holo/target.yaml \\
        --ligand-smiles    "OC(...)=O" \\
        --ligand-entity-id L \\
        --protein-entity-id A
"""

import argparse
import time
from pathlib import Path

import requests # pyright: ignore[reportMissingModuleSource]
import yaml # pyright: ignore[reportMissingModuleSource]
from Bio import SeqIO # pyright: ignore[reportMissingImports]

RCSB_URL = "https://files.rcsb.org/download/{code}.cif"


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fasta",              required=True)
    p.add_argument("--a3m",                required=True)
    p.add_argument("--m8",                 required=True,
                   help="ColabFold PDB70 template-hits file, e.g. data/msa/pdb/{protein}.m8")
    p.add_argument("--protein-name",       required=True)
    p.add_argument("--templates-dir",      required=True,
                   help="Where to cache the downloaded default template CIFs for this protein")
    p.add_argument("--top-n-templates",    type=int, default=5,
                   help="Number of top unique PDB hits from the .m8 file to use as templates")
    p.add_argument("--output-apo",         required=True)
    p.add_argument("--output-holo",        required=True)
    p.add_argument("--ligand-smiles",      required=True)
    p.add_argument("--ligand-entity-id",   default="L")
    p.add_argument("--protein-entity-id",  default="A")
    return p.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_sequence(fasta_path: Path) -> str:
    records = list(SeqIO.parse(fasta_path, "fasta"))
    if not records:
        raise RuntimeError(f"No sequence found in {fasta_path}")
    return str(records[0].seq)


def top_pdb_codes(m8_path: Path, top_n: int) -> list[str]:
    """
    Return up to `top_n` unique PDB codes from a ColabFold PDB70 .m8 file,
    ordered by ascending e-value (best hit first).

    Each row's 2nd column looks like "4oh3_B" (PDB code + template chain);
    only the PDB code is kept, since Boltz aligns templates to the query
    chain automatically regardless of which chain the hit came from.
    """
    rows = []
    for line in m8_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        fields = line.split("\t")
        target = fields[1]
        evalue = float(fields[10])
        code = target.split("_")[0].upper()
        rows.append((evalue, code))
    rows.sort(key=lambda r: r[0])

    codes: list[str] = []
    seen: set[str] = set()
    for _, code in rows:
        if code not in seen:
            seen.add(code)
            codes.append(code)
        if len(codes) >= top_n:
            break
    return codes


def download_cif(code: str, dest: Path, retries: int = 3) -> bool:
    if dest.exists():
        return True
    url = RCSB_URL.format(code=code)
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                dest.write_bytes(r.content)
                return True
            if r.status_code == 404:
                print(f"    x {code}: not found (404) — skipping")
                return False
            print(f"    ! {code}: HTTP {r.status_code}, attempt {attempt}/{retries}")
        except requests.RequestException as e:
            print(f"    ! {code}: network error ({e}), attempt {attempt}/{retries}")
        time.sleep(2 ** attempt)
    print(f"    x {code}: failed after {retries} attempts")
    return False


def fetch_default_templates(m8_path: Path, templates_dir: Path, top_n: int) -> list[str]:
    """Download up to `top_n` template CIFs and return cwd-relative paths."""
    templates_dir.mkdir(parents=True, exist_ok=True)
    cwd = Path.cwd()
    paths: list[str] = []
    for code in top_pdb_codes(m8_path, top_n):
        dest = templates_dir / f"{code}.cif"
        if download_cif(code, dest):
            paths.append(str(dest.resolve().relative_to(cwd)))
    if not paths:
        raise RuntimeError(f"No default templates could be downloaded from hits in {m8_path}")
    return paths


# ── YAML construction ──────────────────────────────────────────────────────────

def build_yaml(
    sequence:           str,
    a3m_path:           Path,
    template_paths:     list[str],
    protein_entity_id:  str,
    holo:               bool,
    ligand_smiles:      str,
    ligand_entity_id:   str,
) -> dict:
    """
    Standard Boltz-2 run: sequence + MSA + default (.m8 top-hit) templates.
    No pocket constraint, no curated/synthetic template override.
    """
    doc: dict = {
        "sequences": [
            {
                "protein": {
                    "id":       protein_entity_id,
                    "sequence": sequence,
                    "msa":      str(a3m_path.resolve().relative_to(Path.cwd())),
                }
            }
        ],
        "templates": [
            {"cif": cif_path, "chain_id": protein_entity_id}
            for cif_path in template_paths
        ],
    }
    if holo:
        doc["sequences"].append({
            "ligand": {
                "id":     ligand_entity_id,
                "smiles": ligand_smiles,
            }
        })
    return doc


def write_yaml(doc: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        yaml.dump(doc, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    fasta_path    = Path(args.fasta)
    a3m_path      = Path(args.a3m)
    m8_path       = Path(args.m8)
    templates_dir = Path(args.templates_dir)

    sequence       = load_sequence(fasta_path)
    template_paths = fetch_default_templates(m8_path, templates_dir, args.top_n_templates)

    for holo, output in ((False, args.output_apo), (True, args.output_holo)):
        doc = build_yaml(
            sequence          = sequence,
            a3m_path          = a3m_path,
            template_paths    = template_paths,
            protein_entity_id = args.protein_entity_id,
            holo              = holo,
            ligand_smiles     = args.ligand_smiles,
            ligand_entity_id  = args.ligand_entity_id,
        )
        write_yaml(doc, Path(output))
        print(
            f"[default_boltz_input] {args.protein_name} × {'holo' if holo else 'apo'}: "
            f"{len(template_paths)} default template(s) from {m8_path.name} -> {output}"
        )


if __name__ == "__main__":
    main()
