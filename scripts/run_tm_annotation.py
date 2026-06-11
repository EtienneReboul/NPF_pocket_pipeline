#!/usr/bin/env python3
"""
scripts/run_tm_annotation.py
=============================
Stage 2b of the NPF pipeline — Transmembrane helix annotation.
Submits each NPF sequence to EMBL-EBI InterProScan 5 REST API with the
Phobius and TMHMM applications to predict TM helix positions.

Results feed into compute_tm_angle.py, which uses TM2 and TM8 positions
as the conformational metric from Qureshi et al. (2020) Nature
https://doi.org/10.1038/s41586-020-1963-z

Fair-use policy compliance: same ≤25-job batched strategy as run_interproscan.py.
Cached results are stored as {name}_tm.json (separate from CDD {name}.json cache).

Usage (called by Snakemake rule `run_tm_annotation`):
    python scripts/run_tm_annotation.py \\
        --fasta         data/sequences/npf_arabidopsis.fasta \\
        --out-dir       data/interpro \\
        --email         your@email.com \\
        --sentinel      data/interpro/tm_annotation.done \\
        --poll-interval 30 \\
        --poll-timeout  3600
"""

import argparse
import json
import re
import time
from pathlib import Path

import requests # pyright: ignore[reportMissingModuleSource]
from Bio import SeqIO # pyright: ignore[reportMissingImports]

EBI_BASE     = "https://www.ebi.ac.uk/Tools/services/rest/iprscan5"
MAX_PARALLEL = 25
TM_APPS      = ["Phobius", "TMHMM"]
TM_LIBRARIES = {"PHOBIUS", "TMHMM"}


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fasta",          required=True)
    p.add_argument("--out-dir",        required=True)
    p.add_argument("--email",          required=True)
    p.add_argument("--sentinel",       required=True)
    p.add_argument("--poll-interval",  type=int, default=30)
    p.add_argument("--poll-timeout",   type=int, default=3600)
    return p.parse_args()


# ── EBI REST helpers ───────────────────────────────────────────────────────────

def submit_one(sequence: str, email: str) -> str:
    r = requests.post(
        f"{EBI_BASE}/run",
        data={
            "email":    email,
            "sequence": sequence,
            "appl":     ",".join(TM_APPS),
            "goterms":  "false",
            "pathways": "false",
            "stype":    "p",
        },
        timeout=60,
    )
    if r.status_code not in (200, 202):
        raise RuntimeError(f"Submission failed ({r.status_code}): {r.text}")
    return r.text.strip()


def check_status(job_id: str) -> str:
    r = requests.get(f"{EBI_BASE}/status/{job_id}", timeout=30)
    return r.text.strip()


def download_result(job_id: str) -> dict:
    r = requests.get(f"{EBI_BASE}/result/{job_id}/json", timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"Result download failed ({r.status_code}) for {job_id}")
    return r.json()


def wait_for_batch(jobs: dict, poll_interval: int, timeout: int) -> dict:
    """Block until every job reaches a terminal state. Fair-use compliant."""
    pending = dict(jobs)
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


# ── TM helix extraction ────────────────────────────────────────────────────────

def extract_tm_helices(result: dict) -> dict[str, list[dict]]:
    """
    Recursively search the InterProScan JSON result for Phobius/TMHMM matches
    with locationType == "TRANSMEMBRANE".

    Returns {'PHOBIUS': [{'start': .., 'end': ..}, ...], 'TMHMM': [...]}
    Helices are sorted by start position and deduplicated within each tool.
    """
    tm_by_tool: dict[str, list[dict]] = {}

    def _recurse(obj):
        if isinstance(obj, dict):
            sig      = obj.get("signature", {})
            lib_info = sig.get("signatureLibraryRelease", {})
            library  = lib_info.get("library", "").upper()
            if library in TM_LIBRARIES:
                locs = [
                    {"start": loc["start"], "end": loc["end"]}
                    for loc in obj.get("locations", [])
                    if loc.get("locationType") == "TRANSMEMBRANE"
                ]
                if locs:
                    tm_by_tool.setdefault(library, []).extend(locs)
            for v in obj.values():
                _recurse(v)
        elif isinstance(obj, list):
            for item in obj:
                _recurse(item)

    _recurse(result)

    # Deduplicate and sort each tool's list by start position
    for tool in tm_by_tool:
        seen, dedup = set(), []
        for h in sorted(tm_by_tool[tool], key=lambda x: x["start"]):
            key = (h["start"], h["end"])
            if key not in seen:
                seen.add(key)
                dedup.append(h)
        tm_by_tool[tool] = dedup

    return tm_by_tool


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = list(SeqIO.parse(args.fasta, "fasta"))
    if not records:
        raise RuntimeError(f"No sequences found in {args.fasta}")
    print(f"[tm_annot] {len(records)} sequences loaded.")

    # Build queue — skip proteins whose TM cache already exists (resumable)
    todo = []
    for rec in records:
        name    = protein_base_name(rec)
        tm_path = out_dir / f"{name}_tm.json"
        if tm_path.exists():
            print(f"[tm_annot] {name}: TM JSON cached — skipping submission")
            continue
        uid       = parse_uniprot_id(rec.id)
        gene      = parse_gene_name(rec.description)
        header    = f">{uid} GN={gene}" if gene else f">{uid}"
        fasta_str = f"{header}\n{rec.seq}"
        todo.append((name, fasta_str))

    print(f"[tm_annot] {len(todo)} sequence(s) to submit (apps: {', '.join(TM_APPS)}).")

    # ── Submit in batches of MAX_PARALLEL ─────────────────────────────────────
    n_batches = (len(todo) + MAX_PARALLEL - 1) // MAX_PARALLEL
    for batch_num, batch_start in enumerate(range(0, len(todo), MAX_PARALLEL), 1):
        batch = todo[batch_start : batch_start + MAX_PARALLEL]
        print(
            f"\n[tm_annot] Batch {batch_num}/{n_batches}: "
            f"submitting {len(batch)} sequence(s) ..."
        )
        job_ids = {}
        for name, fasta_str in batch:
            job_id = submit_one(fasta_str, args.email)
            job_ids[job_id] = name
            print(f"  → {name}: job ID {job_id}", flush=True)
            time.sleep(1)
        print(f"[tm_annot] Waiting for batch {batch_num}/{n_batches} to complete ...")
        batch_results = wait_for_batch(job_ids, args.poll_interval, args.poll_timeout)
        for name, result in batch_results.items():
            (out_dir / f"{name}_tm.json").write_text(json.dumps(result, indent=2))
        print(f"[tm_annot] Batch {batch_num}/{n_batches} complete.")

    # ── Parse all cached TM results ───────────────────────────────────────────
    print(f"\n[tm_annot] Parsing TM topology for all {len(records)} proteins ...")
    tm_summary: dict[str, dict] = {}
    for rec in records:
        name    = protein_base_name(rec)
        tm_path = out_dir / f"{name}_tm.json"
        if not tm_path.exists():
            print(f"[tm_annot] WARNING: no TM cache for {name} — skipping")
            continue
        result     = json.loads(tm_path.read_text())
        tm_by_tool = extract_tm_helices(result)
        if not tm_by_tool:
            print(f"[tm_annot] WARNING: no TM helices found for {name}")
        n_tm = max((len(v) for v in tm_by_tool.values()), default=0)
        tools_str = ", ".join(f"{k}: {len(v)}" for k, v in tm_by_tool.items())
        print(f"[tm_annot] {name}: {n_tm} TM helices ({tools_str})")
        tm_summary[name] = tm_by_tool

    if not tm_summary:
        raise RuntimeError(
            "No TM helix predictions found in any result. "
            "Check that Phobius and TMHMM applications are available."
        )

    summary_path = out_dir / "tm_topology_summary.json"
    summary_path.write_text(json.dumps(tm_summary, indent=2))
    print(f"[tm_annot] TM topology summary: {summary_path}")

    sentinel = Path(args.sentinel)
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(
        f"TM annotation done. {len(tm_summary)} proteins with TM predictions.\n"
    )
    print(f"[tm_annot] Done. Sentinel: {sentinel}")


if __name__ == "__main__":
    main()
