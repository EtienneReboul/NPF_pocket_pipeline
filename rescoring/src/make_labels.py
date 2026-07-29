#!/usr/bin/env python3
"""
src/make_labels.py
====================
Derive data/labels.csv (protein,class) and data/lda_residues.csv
(position,chain,z_dim,z_name,lda_coef) from the NPF_LDA_kernel snapshot in
data/lda_kernel_inputs/ (see PROVENANCE.md there).

hc_labels.tsv uses 1 = importer, 0 = non_importer (checked against
hc_lda_scores.tsv sign conventions in NPF_LDA_kernel: e.g. NPF2.5_Q9M172 is
labelled 1 there and "importer" with a positive LDA score in
hc_lda_scores.tsv; NPF6.1_Q9LYR6 is 0 / "non-importer").
"""
import pandas as pd

import config

labels_rows = []
for line in (config.LDA_INPUTS_DIR / "hc_labels.tsv").read_text().splitlines():
    if not line.strip():
        continue
    name, val = line.split()
    cls = "importer" if int(val) == 1 else "non_importer"
    labels_rows.append({"protein": name, "class": cls})

labels = pd.DataFrame(labels_rows).sort_values("protein").reset_index(drop=True)
labels.to_csv(config.LABELS_CSV, index=False)
print(f"[make_labels] wrote {config.LABELS_CSV} "
      f"({(labels['class'] == 'importer').sum()} importer, "
      f"{(labels['class'] == 'non_importer').sum()} non_importer)")

lda = pd.read_csv(config.LDA_INPUTS_DIR / "hc_lda_loadings.tsv", sep="\t")
lda.insert(1, "chain", config.PROTEIN_CHAIN)
lda.to_csv(config.LDA_RESIDUES_CSV, index=False)
print(f"[make_labels] wrote {config.LDA_RESIDUES_CSV} ({len(lda)} rows)")
