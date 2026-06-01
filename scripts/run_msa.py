#!/usr/bin/env python3
"""
scripts/run_msa.py
==================
Stage 1 of the NPF pipeline:
  1. Download all Arabidopsis NPF sequences from UniProt (Swiss-Prot reviewed).
  2. Split into per-protein FASTA files.
  3. Submit each sequence to the ColabFold MMseqs2 API and retrieve:
       - {gene}_{uniprot_id}.a3m   (MSA for structure prediction)
       - {gene}_{uniprot_id}.m8    (PDB70 template hits)
       - {gene}_{uniprot_id}.sh    (MMseqs2 provenance script)

Called by Snakemake rule `run_msa`. Outputs a sentinel file listing all
discovered proteins once every a3m has been retrieved.

Usage (Snakemake calls this, but also works standalone):
    python scripts/run_msa.py \\
        --fasta-dir   data/sequences \\
        --a3m-dir     data/msa/a3m \\
        --pdb-dir     data/msa/pdb \\
        --sh-dir      data/msa/sh \\
        --sentinel    data/msa/msa.done \\
        --query       'reviewed:true AND organism_id:3702 AND protein_name:"NRT1/ PTR FAMILY"' \\
        --size        500 \\
        --delay       8 \\
        --retries     3 \\
        --poll-interval 15 \\
        --poll-timeout  900
"""

import argparse
import gzip
import io
import re
import tarfile
import time
import zipfile
from pathlib import Path

import requests
from Bio import SeqIO
from tqdm import tqdm


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fasta-dir",     required=True)
    p.add_argument("--a3m-dir",       required=True)
    p.add_argument("--pdb-dir",       required=True)
    p.add_argument("--sh-dir",        required=True)
    p.add_argument("--sentinel",      required=True,
                   help="File to write listing all protein base-names on success")
    p.add_argument("--query",         required=True)
    p.add_argument("--size",          type=int, default=500)
    p.add_argument("--delay",         type=float, default=8)
    p.add_argument("--retries",       type=int, default=3)
    p.add_argument("--poll-interval", type=int, default=15)
    p.add_argument("--poll-timeout",  type=int, default=900)
    return p.parse_args()


# ── UniProt ────────────────────────────────────────────────────────────────────

def download_uniprot_fasta(query: str, size: int, out_fasta: Path) -> None:
    if out_fasta.exists():
        print(f"[msa] UniProt FASTA exists — skipping download: {out_fasta}")
        return

    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {"query": query, "format": "fasta", "size": size}
    chunks = []

    print("[msa] Downloading NPF sequences from UniProt ...")
    while url:
        r = requests.get(url, params=params, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"UniProt request failed ({r.status_code}): {r.text}")
        chunk = r.text.strip()
        if chunk:
            chunks.append(chunk)
        link = r.headers.get("Link", "")
        if 'rel="next"' in link:
            url = link.split("<")[1].split(">")[0]
            params = {}
        else:
            url = None

    if not chunks:
        raise RuntimeError("No sequences returned — check UniProt query syntax.")

    out_fasta.parent.mkdir(parents=True, exist_ok=True)
    out_fasta.write_text("\n".join(chunks) + "\n")
    n = sum(1 for line in "\n".join(chunks).splitlines() if line.startswith(">"))
    print(f"[msa] Downloaded {n} sequences → {out_fasta}")


# ── Sequence helpers ───────────────────────────────────────────────────────────

def parse_uniprot_id(record_id: str) -> str:
    parts = record_id.split("|")
    return parts[1] if len(parts) >= 2 else record_id


def parse_gene_name(description: str) -> str | None:
    m = re.search(r"GN=(\S+)", description)
    return m.group(1) if m else None


def base_name(record) -> str:
    uid = parse_uniprot_id(record.id)
    gene = parse_gene_name(record.description)
    return f"{gene}_{uid}" if gene else uid


# ── ColabFold MSA API ──────────────────────────────────────────────────────────

def submit_ticket(fasta_str: str) -> str:
    r = requests.post(
        "https://api.colabfold.com/ticket/msa",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={"q": fasta_str, "mode": "all"},
        timeout=60,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Ticket submission failed ({r.status_code}): {r.text}")
    result = r.json()
    if "id" not in result:
        raise RuntimeError(f"No ticket ID in response: {result}")
    return result["id"]


def poll_ticket(ticket_id: str, poll_interval: int, timeout: int) -> bytes:
    url = f"https://api.colabfold.com/ticket/{ticket_id}"
    elapsed = 0
    while elapsed < timeout:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"Poll failed ({r.status_code}): {r.text}")
        data = r.json()
        status = data.get("status", "")
        if status == "COMPLETE":
            dl = requests.get(
                f"https://api.colabfold.com/result/download/{ticket_id}",
                timeout=120,
            )
            if dl.status_code != 200:
                raise RuntimeError(f"Download failed ({dl.status_code})")
            return dl.content
        if status in ("ERROR", "FAILED"):
            raise RuntimeError(f"ColabFold job failed: {data}")
        print(f"    [{elapsed}s] {status} — waiting ...", flush=True)
        time.sleep(poll_interval)
        elapsed += poll_interval
    raise RuntimeError(f"Timeout after {timeout}s for ticket {ticket_id}")


# ── Output saving ──────────────────────────────────────────────────────────────

def save_outputs(raw: bytes, name: str, a3m_dir: Path, pdb_dir: Path, sh_dir: Path) -> None:
    """Unpack ColabFold result (tar/zip/gz/plain) into the three output dirs."""

    # TAR (standard ColabFold response: uniref.a3m + pdb70.m8 + msa.sh)
    if tarfile.is_tarfile(io.BytesIO(raw)):
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as tf:
            for member in tf.getmembers():
                f = tf.extractfile(member)
                if f is None:
                    continue
                content = f.read().decode("utf-8")
                if member.name.endswith(".a3m"):
                    (a3m_dir / f"{name}.a3m").write_text(content)
                elif member.name.endswith(".m8"):
                    (pdb_dir / f"{name}.m8").write_text(content)
                elif member.name.endswith(".sh"):
                    (sh_dir / f"{name}.sh").write_text(content)
        return

    # ZIP
    if raw[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            a3m_names = [n for n in zf.namelist() if n.endswith(".a3m")]
            target = a3m_names[0] if a3m_names else zf.namelist()[0]
            (a3m_dir / f"{name}.a3m").write_text(zf.open(target).read().decode("utf-8"))
        return

    # GZIP
    if raw[:2] == b"\x1f\x8b":
        (a3m_dir / f"{name}.a3m").write_text(gzip.decompress(raw).decode("utf-8"))
        return

    # Plain text fallback
    (a3m_dir / f"{name}.a3m").write_text(raw.decode("utf-8"))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    a3m_dir  = Path(args.a3m_dir)
    pdb_dir  = Path(args.pdb_dir)
    sh_dir   = Path(args.sh_dir)
    fasta_dir = Path(args.fasta_dir)
    for d in (a3m_dir, pdb_dir, sh_dir, fasta_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Step 1 — download FASTA
    all_fasta = fasta_dir / "npf_arabidopsis.fasta"
    download_uniprot_fasta(args.query, args.size, all_fasta)

    # Step 2 — parse sequences
    records = list(SeqIO.parse(all_fasta, "fasta"))
    if not records:
        raise RuntimeError(f"No sequences in {all_fasta}")
    print(f"[msa] Loaded {len(records)} sequences.")

    # Step 3 — save per-protein FASTA files (used by downstream rules as inputs)
    names = []
    for record in records:
        name = base_name(record)
        names.append(name)
        per_fasta = fasta_dir / f"{name}.fasta"
        if not per_fasta.exists():
            uid  = parse_uniprot_id(record.id)
            gene = parse_gene_name(record.description)
            header = f">{uid} GN={gene}" if gene else f">{uid}"
            per_fasta.write_text(f"{header}\n{record.seq}\n")

    # Step 4 — ColabFold MSA (resumable)
    for record in tqdm(records, desc="ColabFold MSA"):
        name = base_name(record)
        a3m_path = a3m_dir / f"{name}.a3m"
        if a3m_path.exists():
            continue

        uid  = parse_uniprot_id(record.id)
        gene = parse_gene_name(record.description)
        header    = f">{uid} GN={gene}" if gene else f">{uid}"
        fasta_str = f"{header}\n{record.seq}"

        for attempt in range(args.retries):
            try:
                ticket_id = submit_ticket(fasta_str)
                print(f"\n[msa] {name}: ticket {ticket_id}", flush=True)
                raw = poll_ticket(ticket_id, args.poll_interval, args.poll_timeout)
                save_outputs(raw, name, a3m_dir, pdb_dir, sh_dir)
                break
            except Exception as e:
                print(f"\n[msa] {name} attempt {attempt + 1}/{args.retries}: {e}")
                if attempt < args.retries - 1:
                    time.sleep(15)

        time.sleep(args.delay)

    # Step 5 — write sentinel listing all protein names
    sentinel = Path(args.sentinel)
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("\n".join(names) + "\n")
    print(f"\n[msa] Done. Sentinel written: {sentinel}")


if __name__ == "__main__":
    main()
