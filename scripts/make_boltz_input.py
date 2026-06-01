#!/usr/bin/env python3
"""
scripts/make_boltz_input.py
============================
Stage 4a of the NPF pipeline:
Generate a Boltz-2 input YAML for one protein × one conformation.

The pocket constraint block is populated from the CDD binding-site residues
stored in cdd_summary.json (produced by run_interproscan.py in Stage 2).

Usage (called by Snakemake rule `prepare_boltz_input`):
    python scripts/make_boltz_input.py \\
        --fasta             data/sequences/NPF6.3_Q05085.fasta \\
        --a3m               data/msa/a3m/NPF6.3_Q05085.a3m \\
        --cdd-summary       data/interpro/cdd_summary.json \\
        --protein-name      NPF6.3_Q05085 \\
        --templates-dir     data/templates/occluded_holo \\
        --conformation      occluded_holo \\
        --output            data/boltz_inputs/NPF6.3_Q05085/occluded_holo/target.yaml \\
        --ligand-smiles     "[O-][N+](=O)[O-]" \\
        --ligand-entity-id  L \\
        --protein-entity-id A \\
        --pocket-max-distance 6.0 \\
        --pocket-force      true
"""

import argparse
import json
from pathlib import Path

import yaml
from Bio import SeqIO


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fasta",               required=True)
    p.add_argument("--a3m",                 required=True)
    p.add_argument("--cdd-summary",         required=True,
                   help="Path to cdd_summary.json from run_interproscan.py")
    p.add_argument("--protein-name",        required=True,
                   help="Protein key in cdd_summary.json, e.g. NPF6.3_Q05085")
    p.add_argument("--templates-dir",       required=True,
                   help="Folder of template .cif files for this conformation")
    p.add_argument("--conformation",        required=True,
                   help="Conformation name, e.g. occluded_holo")
    p.add_argument("--output",              required=True)
    p.add_argument("--ligand-smiles",       required=True)
    p.add_argument("--ligand-entity-id",    default="L")
    p.add_argument("--protein-entity-id",   default="A")
    p.add_argument("--pocket-max-distance", type=float, default=6.0)
    p.add_argument("--pocket-force",        type=lambda x: x.lower() == "true",
                   default=True)
    return p.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────────

def is_holo(conformation: str) -> bool:
    return "holo" in conformation.lower()


def load_sequence(fasta_path: Path) -> str:
    records = list(SeqIO.parse(fasta_path, "fasta"))
    if not records:
        raise RuntimeError(f"No sequence found in {fasta_path}")
    return str(records[0].seq)


def load_residues_from_summary(summary_path: Path, protein_name: str) -> list[int]:
    """
    Extract binding-site residues for one protein from cdd_summary.json.

    Expected JSON structure:
    {
      "NPF6.3_Q05085": {
        "accession": "cd17416",
        "subclade":  "MFS_NPF1_2",
        "residues":  [45, 87, 123, ...]
      }, ...
    }
    """
    data = json.loads(summary_path.read_text())
    if protein_name not in data:
        print(
            f"[boltz_input] WARNING: {protein_name} not found in {summary_path}. "
            f"No pocket constraint will be applied."
        )
        return []
    return data[protein_name].get("residues", [])


def collect_template_paths(templates_dir: Path) -> list[str]:
    """Return sorted list of absolute CIF paths in the template directory."""
    cifs = sorted(templates_dir.glob("*.cif"))
    if not cifs:
        raise RuntimeError(f"No .cif files found in templates directory: {templates_dir}")
    return [str(p.resolve()) for p in cifs]


# ── YAML construction ──────────────────────────────────────────────────────────

def build_yaml(
    sequence:           str,
    a3m_path:           Path,
    template_paths:     list[str],
    conformation:       str,
    ligand_smiles:      str,
    ligand_entity_id:   str,
    protein_entity_id:  str,
    binding_residues:   list[int],
    pocket_max_dist:    float,
    pocket_force:       bool,
) -> dict:
    """
    Build the Boltz-2 input dict.

    YAML structure (Boltz-2 format):
      sequences:
        - protein:
            id: A
            sequence: MAST...
            msa: path/to/file.a3m
      templates:
        - cif: path/to/template.cif
          chain_id: A
      constraints:                    ← only when binding residues available
        - pocket:
            binder: L
            contacts: [[A, 45], [A, 87], ...]
            max_distance: 6.0
            force: true
      ligand_smiles:                  ← only for holo conformations
        - id: L
          smiles: "[O-][N+](=O)[O-]"
    """
    doc = {}

    # ── Sequences ──────────────────────────────────────────────────────────────
    protein_entry: dict = {
        "id":       protein_entity_id,
        "sequence": sequence,
        "msa":      str(a3m_path.resolve()),
    }
    doc["sequences"] = [{"protein": protein_entry}]

    # Append ligand for holo conformations
    if is_holo(conformation):
        doc["sequences"].append({
            "ligand": {
                "id":     ligand_entity_id,
                "smiles": ligand_smiles,
            }
        })

    # ── Templates ──────────────────────────────────────────────────────────────
    doc["templates"] = [
        {"cif": cif_path, "chain_id": protein_entity_id}
        for cif_path in template_paths
    ]

    # ── Pocket constraint ──────────────────────────────────────────────────────
    # Only meaningful for holo conformations (ligand must be present).
    if binding_residues and is_holo(conformation):
        contacts = [[protein_entity_id, r] for r in binding_residues]
        doc["constraints"] = [
            {
                "pocket": {
                    "binder":       ligand_entity_id,
                    "contacts":     contacts,
                    "max_distance": pocket_max_dist,
                    "force":        pocket_force,
                }
            }
        ]

    return doc


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    fasta_path    = Path(args.fasta)
    a3m_path      = Path(args.a3m)
    summary_path  = Path(args.cdd_summary)
    templates_dir = Path(args.templates_dir)
    output_path   = Path(args.output)

    sequence         = load_sequence(fasta_path)
    binding_residues = load_residues_from_summary(summary_path, args.protein_name)
    template_paths   = collect_template_paths(templates_dir)

    doc = build_yaml(
        sequence          = sequence,
        a3m_path          = a3m_path,
        template_paths    = template_paths,
        conformation      = args.conformation,
        ligand_smiles     = args.ligand_smiles,
        ligand_entity_id  = args.ligand_entity_id,
        protein_entity_id = args.protein_entity_id,
        binding_residues  = binding_residues,
        pocket_max_dist   = args.pocket_max_distance,
        pocket_force      = args.pocket_force,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        yaml.dump(doc, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)

    holo = is_holo(args.conformation)
    print(
        f"[boltz_input] {args.protein_name} × {args.conformation}: "
        f"{'holo' if holo else 'apo'}, "
        f"{len(binding_residues)} pocket residues, "
        f"{len(template_paths)} templates."
    )


if __name__ == "__main__":
    main()