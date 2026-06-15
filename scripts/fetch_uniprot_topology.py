#!/usr/bin/env python3
"""
scripts/fetch_uniprot_topology.py
====================================
Fetch TM helix topology from UniProt REST API for all NPF proteins.

Primary source: UniProt curated "Transmembrane" features (Swiss-Prot
manually reviewed entries are most reliable).

Patching: when UniProt annotates fewer than 12 TM helices, the missing
helix is almost always TM1 at the N-terminus (Swiss-Prot curators leave
it out when it is ambiguous). We detect this from a large N-terminal gap
before the first annotated helix, then fill it with Phobius helices
(preferred) or TMHMM helices that fall entirely within that gap.

Fallback: when UniProt has no TM annotation at all, cached Phobius/TMHMM
predictions (from run_tm_annotation.py, stored as _tm.json) are used.

Output format (consumed by compute_tm_angle.py):
  {
    "NPF1.1_Q8LPL2": {
      "UNIPROT":  [{"start": 68, "end": 88}, ...],   # curated + patched
      "PHOBIUS":  [...],                               # fallback only
      "TMHMM":    [...]                               # fallback only
    },
    ...
  }

Usage (called by Snakemake rule fetch_uniprot_topology):
    python scripts/fetch_uniprot_topology.py \\
        --fasta     data/sequences/npf_arabidopsis.fasta \\
        --out-dir   data/interpro \\
        --sentinel  data/interpro/uniprot_topology.done \\
        --target-tm 12
"""

import argparse
import json
import re
import time
from pathlib import Path

import requests
from Bio import SeqIO

UNIPROT_BASE  = "https://rest.uniprot.org/uniprotkb"
REQUEST_DELAY = 0.2          # seconds between UniProt calls (≤5 req/s)
TM_LIBRARIES  = {"PHOBIUS", "TMHMM"}
TM_ACCESSIONS = {"TRANSMEMBRANE", "TMhelix"}


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fasta",     required=True)
    p.add_argument("--out-dir",   required=True)
    p.add_argument("--sentinel",  required=True)
    p.add_argument("--target-tm", type=int, default=12,
                   help="Expected number of TM helices (default: 12)")
    return p.parse_args()


# ── FASTA helpers ──────────────────────────────────────────────────────────────

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


# ── UniProt fetch ──────────────────────────────────────────────────────────────

def fetch_uniprot_tm(uniprot_id: str, cache_path: Path) -> list[dict] | None:
    """
    Return sorted list of {"start": int, "end": int} TM helix positions
    from UniProt, or None if unavailable.  Results are cached to cache_path
    so re-runs skip the network call.
    """
    if cache_path.exists():
        data = json.loads(cache_path.read_text())
    else:
        try:
            r = requests.get(f"{UNIPROT_BASE}/{uniprot_id}.json", timeout=30)
            if r.status_code == 404:
                cache_path.write_text(json.dumps({}))
                return None
            r.raise_for_status()
        except requests.RequestException as exc:
            print(f"  [!] UniProt request failed for {uniprot_id}: {exc}")
            return None
        data = r.json()
        cache_path.write_text(json.dumps(data, indent=2))
        time.sleep(REQUEST_DELAY)

    helices = []
    for feat in data.get("features", []):
        if feat.get("type") != "Transmembrane":
            continue
        loc   = feat.get("location", {})
        start = loc.get("start", {}).get("value")
        end   = loc.get("end",   {}).get("value")
        if start is not None and end is not None:
            helices.append({"start": start, "end": end})

    return sorted(helices, key=lambda h: h["start"]) if helices else None


# ── Phobius/TMHMM extraction ───────────────────────────────────────────────────

def extract_tm_helices_from_cache(result: dict) -> dict[str, list[dict]]:
    """Parse Phobius/TMHMM helices from a cached _tm.json file."""
    tm_by_tool: dict[str, list[dict]] = {}

    def _recurse(obj):
        if isinstance(obj, dict):
            sig       = obj.get("signature", {})
            lib_info  = sig.get("signatureLibraryRelease", {})
            library   = lib_info.get("library", "").upper()
            accession = sig.get("accession", "")
            if library in TM_LIBRARIES and accession in TM_ACCESSIONS:
                locs = [
                    {"start": loc["start"], "end": loc["end"]}
                    for loc in obj.get("locations", [])
                ]
                if locs:
                    tm_by_tool.setdefault(library, []).extend(locs)
            for v in obj.values():
                _recurse(v)
        elif isinstance(obj, list):
            for item in obj:
                _recurse(item)

    _recurse(result)

    for tool in tm_by_tool:
        seen, dedup = set(), []
        for h in sorted(tm_by_tool[tool], key=lambda x: x["start"]):
            key = (h["start"], h["end"])
            if key not in seen:
                seen.add(key)
                dedup.append(h)
        tm_by_tool[tool] = dedup

    return tm_by_tool


# ── N-terminal gap patching ────────────────────────────────────────────────────

def patch_nterm_gap(uni_helices: list[dict],
                    pred_tools: dict[str, list[dict]],
                    target: int) -> tuple[list[dict], str]:
    """
    If uni_helices has fewer than target TM helices, look for predicted
    helices (Phobius preferred, then TMHMM) that fall entirely within the
    N-terminal gap before the first UniProt helix and prepend them.

    Returns (patched_helices, patch_note).
    patch_note is empty when no patching was needed or possible.
    """
    n_missing = target - len(uni_helices)
    if n_missing <= 0 or not uni_helices:
        return uni_helices, ""

    first_start = uni_helices[0]["start"]

    for tool in ("PHOBIUS", "TMHMM"):
        pred = pred_tools.get(tool, [])
        if not pred:
            continue
        # helices that end before the first UniProt helix (no overlap)
        gap_helices = [h for h in pred if h["end"] < first_start]
        if not gap_helices:
            continue
        # take the last n_missing from the gap (those closest to first TM)
        to_add = gap_helices[-n_missing:]
        patched = sorted(to_add + uni_helices, key=lambda h: h["start"])
        note = (f"patched {len(to_add)} N-term helix/helices from {tool} "
                f"(positions {[h['start'] for h in to_add]}) → {len(patched)} total")
        return patched, note

    return uni_helices, f"patch failed: no predicted helices in N-term gap (residues 1–{first_start})"


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target  = args.target_tm

    records = list(SeqIO.parse(args.fasta, "fasta"))
    if not records:
        raise RuntimeError(f"No sequences found in {args.fasta}")
    print(f"[uniprot_topo] {len(records)} sequences loaded. Target TM count: {target}")

    n_full, n_patched, n_fallback, n_missing = 0, 0, 0, 0
    tm_summary: dict[str, dict] = {}

    for rec in records:
        name       = protein_base_name(rec)
        uniprot_id = parse_uniprot_id(rec.id)
        topo: dict = {}

        # ── 1. UniProt curated annotation ─────────────────────────────────────
        cache_path = out_dir / f"{name}_uniprot.json"
        cached_str = " (cached)" if cache_path.exists() else ""
        print(f"[uniprot_topo] {name}: UniProt {uniprot_id}{cached_str} ...",
              end=" ", flush=True)

        uni_helices = fetch_uniprot_tm(uniprot_id, cache_path)

        # ── 2. Phobius/TMHMM (always load — used for patching + fallback) ─────
        pred_tools: dict[str, list[dict]] = {}
        tm_cache = out_dir / f"{name}_tm.json"
        if tm_cache.exists():
            pred_tools = extract_tm_helices_from_cache(json.loads(tm_cache.read_text()))

        # ── 3. Patch N-terminal gap if UniProt is incomplete ──────────────────
        if uni_helices is not None:
            if len(uni_helices) < target:
                patched, note = patch_nterm_gap(uni_helices, pred_tools, target)
                if note:
                    print(f"{len(uni_helices)} TM (UniProt) → {note}")
                    n_patched += 1
                else:
                    print(f"{len(uni_helices)} TM")
                    if len(uni_helices) == target:
                        n_full += 1
                topo["UNIPROT"] = patched
            else:
                print(f"{len(uni_helices)} TM")
                topo["UNIPROT"] = uni_helices
                n_full += 1

            # also store raw predictions (compute_tm_angle uses them as fallback)
            topo.update(pred_tools)

        else:
            # No UniProt annotation → pure Phobius/TMHMM fallback
            print("no TM annotation in UniProt")
            if pred_tools:
                best = max(pred_tools, key=lambda k: len(pred_tools[k]))
                print(f"  → fallback: {best} {len(pred_tools[best])} helices")
                topo.update(pred_tools)
                n_fallback += 1
            else:
                print(f"  [!] {name}: WARNING — no TM topology from any source")
                n_missing += 1

        tm_summary[name] = topo

    print(
        f"\n[uniprot_topo] Summary: {n_full} complete UniProt, "
        f"{n_patched} patched, {n_fallback} prediction-only fallback, "
        f"{n_missing} missing"
    )

    if not any(topo for topo in tm_summary.values()):
        raise RuntimeError("No TM topology found for any protein.")

    summary_path = out_dir / "tm_topology_summary.json"
    summary_path.write_text(json.dumps(tm_summary, indent=2))
    print(f"[uniprot_topo] Written: {summary_path}")

    sentinel = Path(args.sentinel)
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(
        f"UniProt TM topology done. {len(tm_summary)} proteins "
        f"({n_full} complete, {n_patched} patched, "
        f"{n_fallback} fallback, {n_missing} missing).\n"
    )
    print(f"[uniprot_topo] Done. Sentinel: {sentinel}")


if __name__ == "__main__":
    main()
