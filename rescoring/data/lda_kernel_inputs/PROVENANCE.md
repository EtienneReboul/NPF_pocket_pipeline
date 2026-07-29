# Provenance

Snapshot copied 2026-07-29 from the sibling `NPF_LDA_kernel` project (not a git
repo, so no commit hash — paths below are relative to that project's root):

| file here                  | source path                                                  |
|-----------------------------|--------------------------------------------------------------|
| `npf_aligned.sto`           | `data/cdd_msa/npf_aligned.sto`                                |
| `hc_labels.tsv`             | `results/ga_classifier/hc/labels.tsv`                         |
| `hc_lda_loadings.tsv`       | `results/ga_classifier/hc/hc_lda_loadings.tsv`                |
| `pocket_sites_cdd_msa.tsv`  | `results/ga_classifier/pocket_sites_cdd_msa.tsv`              |

`npf_aligned.sto` is the HMMER `hmmalign` output (cd17351/MFS_NPF profile HMM)
used by `workflow/scripts/extract_cdd_msa.py` in `NPF_LDA_kernel` to build
`pocket_sites_cdd_msa.tsv`, which in turn is the **input to the "hc" (high-
confidence, 33-protein) Track-A LDA** that produced `hc_lda_loadings.tsv` /
`hc_labels.tsv`. `position` 1–35 in `hc_lda_loadings.tsv` is therefore an
**ascending Stockholm-alignment-column index**, anchored on the one root-entry
(cd17351) sequence in the corpus, `NPF6.1_Q9LYR6`.

`src/build_position_mapping.py` in this project reconstructs, for each of the
33 hc proteins, the per-protein residue number (`resnr`, matching PLIP/pose
numbering) at each of those 35 positions — by re-deriving the same 35
alignment columns from `NPF6.1_Q9LYR6`'s own
`data/interpro/NPF6.1_Q9LYR6_binding_site_residues.txt` (already in this repo)
and inverting the alignment for every other protein. This was validated by
reconstructing every one of the 33 `pocket_sites_cdd_msa.tsv` rows exactly
(letter-for-letter) from `data/sequences/*.fasta` + the reconstructed resnr —
see the assertion in `build_position_mapping.py`.

If `NPF_LDA_kernel`'s hc Track-A LDA is ever re-run (e.g. more proteins added,
alignment rebuilt), refresh these four files and re-run
`build_position_mapping.py` + `make_labels.py`.
