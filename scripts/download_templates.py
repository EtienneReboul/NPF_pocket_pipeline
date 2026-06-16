#!/usr/bin/env python3
"""
scripts/download_templates.py
==============================
Stage 3 of the NPF pipeline:
Download mmCIF template files from RCSB for each conformation set defined in
config.yaml. Called by Snakemake rule `download_templates`.

Usage (standalone):
    python scripts/download_templates.py \\
        --config config.yaml \\
        --templates-root data/templates \\
        [--efflux-templates] \\
        [--dry-run]
"""

import argparse
import time
from pathlib import Path

import requests
import yaml

RCSB_URL = "https://files.rcsb.org/download/{code}.cif"

# Efflux pump supplement (merged when --efflux-templates is set)
EFFLUX_CONFORMATIONS = {
    "outward_open_apo":     {"pdb_codes": ["3WDO", "6GV1"]},
    "outward_open_holo":    {"pdb_codes": ["6T1Z", "7D5P", "7D5Q"]},
    "occluded_apo":         {"pdb_codes": ["2GFP"]},
    "occluded_holo":        {"pdb_codes": ["6VS0", "6VS1", "6VS2", "6VRZ"]},
    "inward_open_apo":      {"pdb_codes": ["6KKJ", "6KKK", "6KKL"]},
    "inward_open_holo":     {"pdb_codes": ["4ZP0", "4ZP2", "4ZOW", "6KKI"]},
    "inward_occluded_holo": {"pdb_codes": ["6EUQ", "6OOM", "6OOP", "6OOQ"]},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",          default="config.yaml")
    p.add_argument("--templates-root",  required=True)
    p.add_argument("--sentinel",        required=True)
    p.add_argument("--efflux-templates", action="store_true")
    p.add_argument("--prune",           action="store_true",
                   help="Remove .cif files not in the current config (keeps folders in sync)")
    p.add_argument("--dry-run", "-n",   action="store_true")
    return p.parse_args()


def download_cif(code: str, dest: Path, retries: int = 3) -> bool:
    url = RCSB_URL.format(code=code.upper())
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


def main():
    args = parse_args()
    cfg  = yaml.safe_load(Path(args.config).read_text())
    root = Path(args.templates_root)
    root.mkdir(parents=True, exist_ok=True)

    conformations = cfg["templates"]["conformations"]

    # Optionally merge efflux templates
    if args.efflux_templates or cfg["templates"].get("efflux", False):
        print("[templates] Merging efflux pump templates.")
        for name, econf in EFFLUX_CONFORMATIONS.items():
            if name in conformations:
                existing = set(c.upper() for c in conformations[name].get("pdb_codes", []))
                for code in econf["pdb_codes"]:
                    if code.upper() not in existing:
                        conformations[name].setdefault("pdb_codes", []).append(code)
            else:
                conformations[name] = {"pdb_codes": list(econf["pdb_codes"])}

    total_ok = total_skip = total_fail = 0

    for conf_name, conf in conformations.items():
        folder = root / conf_name
        folder.mkdir(parents=True, exist_ok=True)

        # Collect (code, tier_label) pairs from all three downloadable fields.
        # `models` entries are homology/repeat-swap models — not on RCSB, skip them.
        # Placeholder codes containing '<' or spaces are also skipped.
        code_tiers: list[tuple[str, str]] = []
        for field, label in (("pdb_codes", "primary"), ("consider", "consider"), ("fallback", "fallback")):
            for raw in conf.get(field, []):
                code = raw.strip()
                if not code or "<" in code or " " in code:
                    continue   # placeholder — no RCSB entry
                code_tiers.append((code.upper(), label))
        # deduplicate while preserving order
        seen: set[str] = set()
        unique: list[tuple[str, str]] = []
        for code, label in code_tiers:
            if code not in seen:
                seen.add(code)
                unique.append((code, label))

        print(f"\n── {conf_name}  ({len(unique)} structures)")
        if args.dry_run:
            for code, label in unique:
                print(f"    [dry-run] {code}  [{label}]")
            continue

        for code, label in unique:
            dest = folder / f"{code}.cif"
            if dest.exists():
                print(f"    + {code}: already present  [{label}]")
                total_skip += 1
                continue
            print(f"    ↓ {code} [{label}] ...", end=" ", flush=True)
            if download_cif(code, dest):
                print(f"done ({dest.stat().st_size // 1024} KB)")
                total_ok += 1
            else:
                total_fail += 1

    print(f"\n[templates] {total_ok} downloaded, {total_skip} skipped, {total_fail} failed.")

    # ── Prune stale CIF files not in current config ───────────────────────────
    if args.prune and not args.dry_run:
        total_pruned = 0
        for conf_name, conf in conformations.items():
            folder = root / conf_name
            if not folder.exists():
                continue
            current_codes: set[str] = set()
            for field in ("pdb_codes", "consider", "fallback"):
                for raw in conf.get(field, []):
                    code = raw.strip().upper()
                    if code and "<" not in code and " " not in code:
                        current_codes.add(code)
            for cif in sorted(folder.glob("*.cif")):
                if cif.stem.upper() not in current_codes:
                    print(f"  [prune] {conf_name}/{cif.name}")
                    cif.unlink()
                    total_pruned += 1
        print(f"[templates] Pruned {total_pruned} stale template file(s).")

    if not args.dry_run:
        sentinel = Path(args.sentinel)
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text(
            f"Templates downloaded: {total_ok} ok, {total_skip} skipped, {total_fail} failed.\n"
        )
        print(f"[templates] Sentinel: {sentinel}")


if __name__ == "__main__":
    main()
