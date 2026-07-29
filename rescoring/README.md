# PyRosetta rescoring (Tier 3)

Implements `../HANDOFF_rescoring.md`: per-residue ligand<->protein REF2015
energy decomposition for gibberellin A1 (GA1) holo poses, comparing the
high-confidence importer/non-importer NPF proteins and overlaying the result
against the LDA-flagged pocket positions from the sibling `NPF_LDA_kernel`
project.

This is a standalone, non-Snakemake mini-project living inside
`NPF_pocket_pipeline` — it reads poses from `../results/minimized_synth/`
but doesn't touch anything else in the main pipeline.

## Setup

```bash
conda env create -f ../envs/pyrosetta_rescoring.yaml
conda activate pyrosetta_rescoring
pip install pyrosetta-installer
python -c "import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()"
```

PyRosetta requires an academic license (free) — the installer prompts you to
accept it. No GPU needed for this tier.

## Important: the ligand's hydrogens/bond orders are wrong in every stored pose — fixed locally here

Every `model_minimized.pdb` in `../results/minimized_synth/` has a
chemically incorrect GA1: Boltz-2's raw CIF output carries no bond-order
records for the ligand, and `../scripts/sanitize_cif.py` fills that gap with
pure-distance RDKit bond perception (`proximityBonding=True`) followed by
`Chem.AddHs()`, which blindly saturates every atom to its default valence.
The result, in every pose: all 19 ligand carbons end up degree-4 and all 6
oxygens degree-2 — **zero double bonds anywhere**. Real GA1 needs three sp2
centers (the lactone C=O, the carboxylic-acid C=O, the exocyclic C=CH2); the
carboxylic acid ends up a geminal diol, the lactone carbonyl a hydroxyl, and
6 spurious explicit hydrogens appear (30 vs. the correct 24 — see
`src/ligand_fix.py`'s module docstring for the full diagnosis).

This is scoped as a **local fix, inside this rescoring project only**:
`src/ligand_fix.py` reconstructs the correct bond orders and hydrogens for
each complex's ligand at load time (from the SMILES already in
`../config.yaml`, matched against each pose's own heavy-atom coordinates via
RDKit's `AssignBondOrdersFromTemplate`), before handing anything to
PyRosetta. It does **not** modify `../results/minimized_synth/` or affect any
other analysis (PLIP, `ligand_iptm`, ...) that already consumes those poses
as-is. If this bug ever gets fixed upstream in `sanitize_cif.py`, this
correction step becomes redundant (but harmless) and can be removed.

One further wrinkle this fix works around: a handful of poses have a
spurious extra heavy-heavy CONECT bond and/or are missing a hydrogen or two
in their raw HETATM records (an artifact of whatever produced that specific
pose) — so the correct heavy-atom *connectivity* is derived once from a
clean reference pose and reused (by atom name) for every complex, rather
than re-derived per-pose.

## Numbering: LDA "position" (1..35) vs. per-protein residue number

`NPF_LDA_kernel`'s hc Track-A LDA (`data/lda_kernel_inputs/hc_lda_loadings.tsv`)
indexes pocket positions 1..35 by **HMMER/Stockholm alignment column**, not
raw ascending residue number — two proteins' "position 12" is the same
conserved pocket position even if their actual residue numbers differ (due
to indels). `src/build_position_mapping.py` reconstructs, for each of the 33
hc proteins, the actual `resnr` (matching PLIP/pose numbering) at each of
those 35 positions, anchored on `NPF6.1_Q9LYR6` (the one root cd17351-entry
sequence in the corpus) and validated by exactly reproducing every hc
protein's `pocket_sites_cdd_msa.tsv` row letter-for-letter. See
`data/lda_kernel_inputs/PROVENANCE.md` for the full derivation.

## Pipeline

```bash
cd src

# 1. Derive labels, LDA residue table, position<->resnr mapping, and the pose manifest
python build_position_mapping.py
python make_labels.py
python make_manifest.py

# 2. One-time ligand parameterization (interactive — eyeball the output).
#    Requires PyRosetta.
python prep_ligand.py

# 3. Acceptance criterion 1: one complex, end to end
python run_complex.py --complex-id <some_complex_id_from_manifest.csv>

# 4. Acceptance criterion 2: the whole manifest (resumable — safe to re-run,
#    skips complexes that already have a results/per_complex/<id>.csv)
python run_batch.py --workers 6

# 5. Aggregate + plot
python aggregate.py
python plots.py
```

Pass `--seed N` to `run_complex.py`/`run_batch.py` for a fully reproducible
run (per-complex Rosetta RNG seeds are derived from `N` + `complex_id`, so
results don't depend on worker-scheduling order); omit it for genuine
replica-to-replica ensemble variation.

`data/manifest.csv` has 4,950 complexes (33 hc proteins x 150 holo samples
each — all 3 conformations x 50 diffusion samples). That's the full ensemble
per protein, run per explicit choice over the "top-N by ligand_iptm" /
"single best pose" alternatives — expect this to take a while on a laptop;
`run_batch.py --limit N` / `--protein` / `--class` let you run a subset
first.

## Sign convention & REU caveat (read before interpreting anything)

- **Negative REU = stabilizing, positive REU = unfavorable**, everywhere.
- `fa_rep` isolates steric clashes; `fa_sol`/`lk_ball_wtd` carry desolvation
  penalties; `fa_elec` + `hbond_*` carry polar/directional contributions.
- These are **Rosetta Energy Units, not kcal/mol** — never present them as
  binding free energies.
- Two-body ligand-residue energies omit solvation coupling beyond the
  implicit term and omit entropy entirely — this is a **triage/ranking**
  tool, not ΔG. Treat hotspots as candidates for later Tier 1 (MM/GBSA)
  confirmation.
- Single-pose Rosetta energies are noisy and clash-sensitive; prefer the
  full-ensemble aggregates in `residue_rank.csv`/`lda_overlay.csv` over any
  one complex's numbers. Every `results/per_complex/*.csv` row also carries
  `fa_rep_raw`/`fa_rep_relaxed` so you can see how much clash relief moved
  things for that specific pose.

## Repo layout

```text
rescoring/
  data/
    lda_kernel_inputs/         snapshot from NPF_LDA_kernel (see PROVENANCE.md)
    labels.csv                 protein, class (importer/non_importer) — 33 hc proteins
    lda_residues.csv           position, chain, z_dim, z_name, lda_coef
    position_resnr_map.csv     protein, position, resnr
    manifest.csv               complex_id, protein, class, conformation, sample_id, pdb_path
  params/
    ligand_template.sdf        embedded GA1 conformer used to generate params
    LIG.params                 Rosetta ligand params (molfile_to_params.py output)
    atom_naming.json           boltz-name <-> rosetta-name mapping + fixed heavy-heavy bonds
  src/
    config.py                  shared paths/constants
    build_position_mapping.py
    make_labels.py
    make_manifest.py
    ligand_fix.py               ligand bond-order/H correction (see above)
    prep_ligand.py               one-time params generation + naming derivation
    pose_prep.py                 stage one complex's PDB for PyRosetta
    relief.py                    raw score -> light coord-constrained FastRelax
    decompose.py                 energy-graph -> per-residue tidy table
    run_complex.py                single-complex CLI
    run_batch.py                   batch driver (resumable, multiprocessing)
    aggregate.py                   pool + rank + LDA overlay
    plots.py                        stacked bar / heatmap / LDA class-difference
  results/
    staged_poses/               per-complex corrected PDB fed to PyRosetta
    per_complex/                 one tidy CSV per complex
    logs/                         one log per complex
    figures/                       plots.py output
    all_contacts.csv, residue_class_summary.csv, residue_rank.csv,
    lda_overlay.csv               aggregate.py output
```
