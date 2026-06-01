#!/usr/bin/env python3
"""
scripts/run_interproscan.py
===========================
Stage 2 of the NPF pipeline:
  1. Submit the full NPF family FASTA to the EMBL-EBI InterProScan 5 REST API
     with only the CDD application enabled.
  2. Poll until the job completes and download the JSON result.
  3. For each protein in the result, find which cd174xx accession matches
     (each protein belongs to exactly one NPF subclade).
  4. Extract binding-site residues (siteLocations) for that accession.
  5. Write one TXT file per protein: {protein_name}_binding_site_residues.txt
     containing comma-separated residue positions.
  6. Write a summary JSON mapping protein → {accession, subclade, residues}.

Called by Snakemake rule `run_interproscan`.

Usage (standalone):
    python scripts/run_interproscan.py \\
        --fasta       data/sequences/npf_arabidopsis.fasta \\
        --out-dir     data/interpro \\
        --email       your@email.com \\
        --accessions  cd17351 cd17413 cd17414 cd17415 cd17416 cd17417 cd17418 cd17419 \\
        --poll-interval 30 \\
        --poll-timeout  3600
"""

import argparse
import json
import re
import time
from pathlib import Path

import requests
from Bio import SeqIO

EBI_BASE = "https://www.ebi.ac.uk/Tools/services/rest/iprscan5"

# Known cd174xx → subclade name mapping (for annotation only)
CDD_NAMES = {
    "cd17351": "MFS_NPF",
    "cd17413": "MFS_NPF6",
    "cd17414": "MFS_NPF4",
    "cd17415": "MFS_NPF3",
    "cd17416": "MFS_NPF1_2",
    "cd17417": "MFS_NPF5",
    "cd17418": "MFS_NPF8",
    "cd17419": "MFS_NPF7",
}


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fasta",          required=True)
    p.add_argument("--out-dir",        required=True)
    p.add_argument("--email",          required=True)
    p.add_argument("--accessions",     nargs="+", required=True,
                   help="List of cd174xx accessions to search for")
    p.add_argument("--sentinel",       required=True,
                   help="Sentinel file to touch on success")
    p.add_argument("--poll-interval",  type=int, default=30)
    p.add_argument("--poll-timeout",   type=int, default=3600)
    return p.parse_args()


# ── EBI REST helpers ───────────────────────────────────────────────────────────

def submit_job(fasta_text: str, email: str) -> str:
    """Submit a batch FASTA to InterProScan 5 REST API. Returns job ID."""
    r = requests.post(
        f"{EBI_BASE}/run",
        data={
            "email":        email,
            "sequence":     fasta_text,
            "appl":         "CDD",         # CDD only
            "goterms":      "false",
            "pathways":     "false",
            "stype":        "p",
        },
        timeout=120,
    )
    if r.status_code not in (200, 202):
        raise RuntimeError(f"InterProScan submission failed ({r.status_code}): {r.text}")
    return r.text.strip()


def poll_job(job_id: str, poll_interval: int, timeout: int) -> None:
    """Block until the job reaches FINISHED state."""
    elapsed = 0
    while elapsed < timeout:
        r = requests.get(f"{EBI_BASE}/status/{job_id}", timeout=30)
        status = r.text.strip()
        print(f"[interpro] [{elapsed}s] {job_id}: {status}", flush=True)
        if status == "FINISHED":
            return
        if status in ("FAILURE", "ERROR", "NOT_FOUND"):
            raise RuntimeError(f"InterProScan job {job_id} failed: {status}")
        time.sleep(poll_interval)
        elapsed += poll_interval
    raise RuntimeError(f"InterProScan timeout after {timeout}s for {job_id}")


def download_result(job_id: str, out_path: Path) -> None:
    """Download JSON result and save to out_path."""
    r = requests.get(f"{EBI_BASE}/result/{job_id}/json", timeout=300)
    if r.status_code != 200:
        raise RuntimeError(f"Result download failed ({r.status_code}): {r.text}")
    out_path.write_bytes(r.content)
    print(f"[interpro] Result saved: {out_path}")


# ── JSON parsing (adapted from extract_cdd_pattern.py) ────────────────────────

def recursive_search(obj, target_accessions: set, results=None):
    """Recursively find all match blocks whose signature.accession is in target_accessions."""
    if results is None:
        results = []
    if isinstance(obj, dict):
        sig = obj.get("signature")
        if isinstance(sig, dict):
            acc = sig.get("accession", "")
            if acc in target_accessions:
                results.append(obj)
        for v in obj.values():
            recursive_search(v, target_accessions, results)
    elif isinstance(obj, list):
        for item in obj:
            recursive_search(item, target_accessions, results)
    return results


def extract_binding_site_residues(match_obj: dict) -> list[int]:
    """Extract unique binding-site residue positions from siteLocations blocks."""
    residues = []
    seen = set()
    for loc in match_obj.get("locations", []):
        for site in loc.get("sites", []):
            for site_loc in site.get("siteLocations", []):
                pos = site_loc.get("start")
                if pos is not None and pos not in seen:
                    seen.add(pos)
                    residues.append(pos)
    return sorted(residues)


def parse_uniprot_id_from_header(sequence_header: str) -> str:
    """Extract UniProt accession from an InterProScan sequence ID field."""
    # InterProScan echoes the FASTA header — we need the UniProt accession
    # Header format: sp|Q05085|NPF6_ARATH or plain Q05085
    parts = sequence_header.split("|")
    return parts[1] if len(parts) >= 2 else sequence_header.split()[0]


def parse_gene_name(description: str) -> str | None:
    m = re.search(r"GN=(\S+)", description)
    return m.group(1) if m else None


# ── Per-protein result extraction ─────────────────────────────────────────────

def extract_per_protein(
    json_data: dict,
    target_accessions: set,
    fasta_records: dict,  # uniprot_id → SeqRecord
) -> dict:
    """
    Walk the InterProScan JSON and build a per-protein result dict:
    {
      "NPF6.3_Q05085": {
        "accession":  "cd17416",
        "subclade":   "MFS_NPF1_2",
        "residues":   [45, 87, 123, ...]
      }, ...
    }
    """
    results = {}

    for protein_entry in json_data.get("results", []):
        # sequence.ac holds the sequence identifier
        seq_info = protein_entry.get("sequence", {})
        raw_id   = seq_info.get("ac", "")
        uniprot_id = parse_uniprot_id_from_header(raw_id)

        # Recover gene name from the original FASTA record if available
        rec = fasta_records.get(uniprot_id)
        gene = parse_gene_name(rec.description) if rec else None
        protein_name = f"{gene}_{uniprot_id}" if gene else uniprot_id

        matches = recursive_search(protein_entry, target_accessions)
        if not matches:
            print(f"[interpro] No cd174xx match for {protein_name} — skipping")
            continue

        # Each protein belongs to exactly one subclade; take first match
        match = matches[0]
        accession = match["signature"]["accession"]
        residues  = extract_binding_site_residues(match)

        results[protein_name] = {
            "accession": accession,
            "subclade":  CDD_NAMES.get(accession, accession),
            "residues":  residues,
        }
        print(
            f"[interpro] {protein_name}: {accession} "
            f"({CDD_NAMES.get(accession, '?')}) → {len(residues)} binding-site residues"
        )

    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_accessions = set(args.accessions)

    # Load FASTA for gene-name lookup
    fasta_path = Path(args.fasta)
    fasta_records = {}
    for rec in SeqIO.parse(fasta_path, "fasta"):
        parts = rec.id.split("|")
        uid = parts[1] if len(parts) >= 2 else rec.id
        fasta_records[uid] = rec

    fasta_text = fasta_path.read_text()

    # Submit + poll + download
    json_path = out_dir / "interproscan_cdd.json"
    if not json_path.exists():
        print("[interpro] Submitting batch job to EMBL-EBI InterProScan 5 ...")
        job_id = submit_job(fasta_text, args.email)
        print(f"[interpro] Job ID: {job_id}")
        poll_job(job_id, args.poll_interval, args.poll_timeout)
        download_result(job_id, json_path)
    else:
        print(f"[interpro] JSON already exists — skipping submission: {json_path}")

    # Parse JSON
    data = json.loads(json_path.read_text())
    per_protein = extract_per_protein(data, target_accessions, fasta_records)

    if not per_protein:
        raise RuntimeError(
            "No cd174xx matches found in the InterProScan result. "
            "Check that the CDD database was used and the accession list is correct."
        )

    # Write per-protein binding-site residue files
    for protein_name, info in per_protein.items():
        residue_path = out_dir / f"{protein_name}_binding_site_residues.txt"
        residue_path.write_text(",".join(str(r) for r in info["residues"]) + "\n")

    # Write master summary JSON
    summary_path = out_dir / "cdd_summary.json"
    summary_path.write_text(json.dumps(per_protein, indent=2))
    print(f"[interpro] Summary JSON: {summary_path}")

    # Write sentinel
    sentinel = Path(args.sentinel)
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(f"InterProScan done. {len(per_protein)} proteins processed.\n")
    print(f"[interpro] Sentinel: {sentinel}")


if __name__ == "__main__":
    main()
