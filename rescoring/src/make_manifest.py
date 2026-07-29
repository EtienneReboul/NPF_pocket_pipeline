#!/usr/bin/env python3
"""
src/make_manifest.py
======================
Enumerate every holo Boltz-2 pose (already minimized — the same
`model_minimized.pdb` files used by the PLIP / ligand_iptm notebooks) for
every protein in data/labels.csv, and write data/manifest.csv:

    complex_id, protein, class, conformation, sample_id, pdb_path

`pdb_path` is stored relative to the pipeline root (NPF_pocket_pipeline/) so
it stays valid regardless of where this rescoring/ subproject is invoked
from. `conformation` here is the GMM-reannotated conformation (the
`minimized_synth` directory name), matching the `gibberellin_pocket_contacts`
/ `gibberellin_ligand_iptm` notebooks' convention.
"""
import pandas as pd

import config


def holo_conformations(protein: str) -> list[str]:
    prot_dir = config.MINIMIZED_DIR / protein
    if not prot_dir.exists():
        return []
    return sorted(d.name for d in prot_dir.iterdir() if d.is_dir() and "_holo" in d.name)


def main():
    labels = pd.read_csv(config.LABELS_CSV)

    rows = []
    for _, row in labels.iterrows():
        protein, cls = row["protein"], row["class"]
        for conformation in holo_conformations(protein):
            conf_dir = config.MINIMIZED_DIR / protein / conformation
            for sample_dir in sorted(conf_dir.iterdir()):
                pdb = sample_dir / "model_minimized.pdb"
                if not pdb.exists():
                    continue
                sample_id = sample_dir.name
                rows.append({
                    "complex_id": f"{protein}__{sample_id}",
                    "protein": protein,
                    "class": cls,
                    "conformation": conformation,
                    "sample_id": sample_id,
                    "pdb_path": str(pdb.relative_to(config.PIPELINE_ROOT)),
                })

    manifest = pd.DataFrame(rows).sort_values(["protein", "conformation", "sample_id"]).reset_index(drop=True)
    dupes = manifest["complex_id"].duplicated().sum()
    if dupes:
        raise RuntimeError(f"{dupes} duplicate complex_id values in manifest — naming collision.")

    manifest.to_csv(config.MANIFEST_CSV, index=False)
    print(f"[make_manifest] wrote {config.MANIFEST_CSV} "
          f"({len(manifest)} complexes, {manifest.protein.nunique()} proteins)")
    print(manifest.groupby("class")["complex_id"].count())


if __name__ == "__main__":
    main()
