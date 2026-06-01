#!/usr/bin/env python3
"""
scripts/run_interproscan.py
===========================
Stage 2 of the NPF pipeline:
  1. Submit each NPF sequence individually to the EMBL-EBI InterProScan 5
     REST API (one sequence per request — API limit).
  2. Respect the fair-use policy: max 25 concurrent jobs; wait for a full
     batch to complete before submitting the next one.
  3. Download JSON results per protein and cache them locally (resumable).
  4. Extract CDD binding-site residues per protein.
  5. Write per-protein binding-site residue TXT files + master summary JSON.

Fair-use policy compliance:
  - EBI Job Dispatcher docs: ≤30 jobs submitted at a time, wait for completion
                             before submitting more.
  - InterProScan FAQ:        ≤25 parallel requests (more restrictive → we use 25)
  Sources:
    https://www.ebi.ac.uk/jdispatcher/docs/webservices/
    https://interpro-documentation.readthedocs.io/en/latest/faq.html

Usage (called by Snakemake rule `run_interproscan`):
    python scripts/run_interproscan.py \\
        --fasta         data/sequences/npf_arabidopsis.fasta \\
        --out-dir       data/interpro \\
        --email         your@email.com \\
        --accessions    cd17351 cd17413 cd17414 cd17415 cd17416 cd17417 cd17418 cd17419 \\
        --sentinel      data/interpro/interproscan.done \\
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

EBI_BASE     = "https://www.ebi.ac.uk/Tools/services/rest/iprscan5"
MAX_PARALLEL = 25   # InterProScan FAQ hard limit (stricter than the 30-job general policy)

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
    p.add_argument("--accessions",     nargs="+", required=True)
    p.add_argument("--sentinel",       required=True)
    p.add_argument("--poll-interval",  type=int, default=30)
    p.add_argument("--poll-timeout",   type=int, default=3600)
    return p.parse_args()


# ── EBI REST helpers ───────────────────────────────────────────────────────────

def submit_one(sequence: str, email: str) -> str:
    """
    Submit a single-sequence FASTA to InterProScan 5 REST API.
    Returns job ID.
    The API accepts exactly one sequence per request.
    """
    r = requests.post(
        f"{EBI_BASE}/run",
        data={
            "email":    email,
            "sequence": sequence,
            "appl":     "CDD",      # CDD database only
            "goterms":  "false",
            "pathways": "false",
            "stype":    "p",        # protein sequence
        },
        timeout=60,
    )
    if r.status_code not in (200, 202):
        raise RuntimeError(f"Submission failed ({r.status_code}): {r.text}")
    return r.text.strip()


def check_status(job_id: str) -> str:
    """Return current job status from EBI (RUNNING, QUEUED, FINISHED, ERROR, ...)."""
    r = requests.get(f"{EBI_BASE}/status/{job_id}", timeout=30)
    return r.text.strip()


def download_result(job_id: str) -> dict:
    """Download and parse the JSON result for a finished job."""
    r = requests.get(f"{EBI_BASE}/result/{job_id}/json", timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"Result download failed ({r.status_code}) for {job_id}")
    return r.json()


def wait_for_batch(jobs: dict, poll_interval: int, timeout: int) -> dict:
    """
    Block until every job in the batch reaches a terminal state.

    Implements the EBI fair-use policy: we never submit the next batch
    until this one is fully complete.

    jobs:    {job_id: protein_name}
    returns: {protein_name: result_dict}
    """
    pending = dict(jobs)   # job_id → protein_name
    results = {}
    elapsed = 0

    while pending and elapsed < timeout:
        for job_id in list(pending.keys()):
            status = check_status(job_id)
            if status == "FINISHED":
                name = pending.pop(job_id)
                print(f"  [✓] {name} ({job_id}): FINISHED", flush=True)
                results[name] = download_result(job_id)
            elif status in ("FAILURE", "ERROR", "NOT_FOUND"):
                name = pending.pop(job_id)
                print(f"  [✗] {name} ({job_id}): {status} — will be skipped")

        if pending:
            print(
                f"  [{elapsed}s] {len(pending)} job(s) still running — "
                f"waiting {poll_interval}s ...",
                flush=True,
            )
            time.sleep(poll_interval)
            elapsed += poll_interval

    if pending:
        raise RuntimeError(
            f"Timeout after {timeout}s. Unfinished jobs: {list(pending.values())}"
        )
    return results


# ── Sequence helpers ───────────────────────────────────────────────────────────

def parse_uniprot_id(record_id: str) -> str:
    parts = record_id.split("|")
    return parts[1] if len(parts) >= 2 else record_id


def parse_gene_name(description: str) -> str | None:
    m = re.search(r"GN=(\S+)", description)
    return m.group(1) if m else None


def protein_base_name(record) -> str:
    uid  = parse_uniprot_id(record.id)
    gene = parse_gene_name(record.description)
    return f"{gene}_{uid}" if gene else uid


# ── JSON parsing (adapted from extract_cdd_pattern.py) ────────────────────────

def recursive_search(obj, target_accessions: set, results=None):
    """Find all match blocks whose signature.accession is in target_accessions."""
    if results is None:
        results = []
    if isinstance(obj, dict):
        sig = obj.get("signature")
        if isinstance(sig, dict) and sig.get("accession", "") in target_accessions:
            results.append(obj)
        for v in obj.values():
            recursive_search(v, target_accessions, results)
    elif isinstance(obj, list):
        for item in obj:
            recursive_search(item, target_accessions, results)
    return results


def extract_binding_site_residues(match_obj: dict) -> list[int]:
    """Extract unique binding-site residue positions from siteLocations blocks."""
    residues, seen = [], set()
    for loc in match_obj.get("locations", []):
        for site in loc.get("sites", []):
            for sl in site.get("siteLocations", []):
                pos = sl.get("start")
                if pos is not None and pos not in seen:
                    seen.add(pos)
                    residues.append(pos)
    return sorted(residues)


def parse_result(result: dict, target_accessions: set) -> dict | None:
    """
    Extract accession + binding-site residues from one protein's result JSON.
    Returns None if no cd174xx match found.
    """
    matches = recursive_search(result, target_accessions)
    if not matches:
        return None
    match     = matches[0]   # one protein → exactly one subclade
    accession = match["signature"]["accession"]
    residues  = extract_binding_site_residues(match)
    return {
        "accession": accession,
        "subclade":  CDD_NAMES.get(accession, accession),
        "residues":  residues,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_accessions = set(args.accessions)

    # Load sequences
    records = list(SeqIO.parse(args.fasta, "fasta"))
    if not records:
        raise RuntimeError(f"No sequences found in {args.fasta}")
    print(f"[interpro] {len(records)} sequences loaded.")

    # Build submission queue — skip proteins already cached on disk (resumable)
    todo = []
    for rec in records:
        name      = protein_base_name(rec)
        json_path = out_dir / f"{name}.json"
        if json_path.exists():
            print(f"[interpro] {name}: JSON cached — skipping submission")
            continue
        uid  = parse_uniprot_id(rec.id)
        gene = parse_gene_name(rec.description)
        header    = f">{uid} GN={gene}" if gene else f">{uid}"
        fasta_str = f"{header}\n{rec.seq}"
        todo.append((name, fasta_str))

    print(f"[interpro] {len(todo)} sequence(s) to submit.")

    # ── Submit in batches of MAX_PARALLEL ─────────────────────────────────────
    # Fair-use policy: submit ≤25 jobs, wait for ALL to finish, then next batch.
    n_batches = (len(todo) + MAX_PARALLEL - 1) // MAX_PARALLEL
    for batch_num, batch_start in enumerate(range(0, len(todo), MAX_PARALLEL), 1):
        batch = todo[batch_start : batch_start + MAX_PARALLEL]
        print(
            f"\n[interpro] Batch {batch_num}/{n_batches}: "
            f"submitting {len(batch)} sequence(s) ..."
        )

        job_ids = {}   # job_id → protein_name
        for name, fasta_str in batch:
            job_id = submit_one(fasta_str, args.email)
            job_ids[job_id] = name
            print(f"  → {name}: job ID {job_id}", flush=True)
            time.sleep(1)   # small courtesy pause between individual submissions

        # ── Wait for the ENTIRE batch before submitting the next one ──────────
        # This is the core of the fair-use compliance.
        print(f"[interpro] Waiting for batch {batch_num}/{n_batches} to complete ...")
        batch_results = wait_for_batch(job_ids, args.poll_interval, args.poll_timeout)

        # Cache each result individually so the run is resumable
        for name, result in batch_results.items():
            json_path = out_dir / f"{name}.json"
            json_path.write_text(json.dumps(result, indent=2))
        print(f"[interpro] Batch {batch_num}/{n_batches} complete.")

    # ── Parse all cached results ───────────────────────────────────────────────
    print(f"\n[interpro] Parsing results for all {len(records)} proteins ...")
    per_protein = {}

    for rec in records:
        name      = protein_base_name(rec)
        json_path = out_dir / f"{name}.json"
        if not json_path.exists():
            print(f"[interpro] WARNING: no cached result for {name} — skipping")
            continue

        result = json.loads(json_path.read_text())
        info   = parse_result(result, target_accessions)
        if info is None:
            print(f"[interpro] No cd174xx match for {name} — skipping")
            continue

        per_protein[name] = info

        # Write per-protein binding-site residues file (input to Boltz-2 pocket constraint)
        residue_path = out_dir / f"{name}_binding_site_residues.txt"
        residue_path.write_text(",".join(str(r) for r in info["residues"]) + "\n")

        print(
            f"[interpro] {name}: {info['accession']} "
            f"({info['subclade']}) → {len(info['residues'])} binding-site residues"
        )

    if not per_protein:
        raise RuntimeError(
            "No cd174xx matches found in any InterProScan result. "
            "Check that the CDD application ran and the accession list is correct."
        )

    # Write master summary JSON
    summary_path = out_dir / "cdd_summary.json"
    summary_path.write_text(json.dumps(per_protein, indent=2))
    print(f"[interpro] Summary JSON: {summary_path}")

    # Write sentinel to signal Snakemake the rule is complete
    sentinel = Path(args.sentinel)
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(
        f"InterProScan done. {len(per_protein)} proteins with CDD matches.\n"
    )
    print(f"[interpro] Done. Sentinel: {sentinel}")


if __name__ == "__main__":
    main()