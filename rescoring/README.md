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

**Note on ligand parameterization:** HANDOFF_rescoring.md's §5 step 1 names
`molfile_to_params.py` (the classic Rosetta tool) for generating the ligand
`.params` file. That script ships only with the full Rosetta source/binary
distribution, not with the PyPI/`pyrosetta-installer` wheel this env uses (it
also needs a `rosetta_py` helper package that isn't on PyPI). `prep_ligand.py`
uses `rdkit_to_params` instead (already in `envs/pyrosetta_rescoring.yaml`) —
a maintained pure RDKit+PyRosetta reimplementation that generates the same
kind of `.params` file directly from an RDKit mol, in-process.

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

Atom naming in the corrected ligand: the 25 heavy atoms keep Boltz's own PDB
atom names (e.g. `C31`, `O27` — identical across every complex already, so no
renaming needed); the 24 hydrogens RDKit adds back are named `H01`..`H24` in
a fixed, deterministic order (`ligand_fix.name_new_hydrogens`). `LIG.params`
is generated with that exact naming, so every complex's corrected ligand
matches it with no per-complex lookup required.

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
"single best pose" alternatives. With the neighborhood-restricted relax (see
below), one complex takes ~6-10s, so the full batch is tractable in a few
hours with `--workers` parallelism; `run_batch.py --limit N` / `--protein` /
`--class` let you run a subset first regardless.

## Clash relief is restricted to a neighborhood around the ligand, not the whole protein

`relief.py` deliberately does NOT run a plain, unrestricted `FastRelax` on
the whole ~590-residue pose. On the first real complex tested, that blew the
total score up catastrophically: -665 REU -> **+36,677 REU**. Isolating the
FastRelax stages found the cause: full-protein *repacking* (rotamer
reselection) with no restriction was itself unstable on these Boltz-2
structures (never crystallographically refined) — repacking alone took the
score from -675 to +1,548; plain minimization alone (no repacking) was fine
and improved it to -1,645.

The fix — and arguably the more faithful reading of the doc's own "light...
without drifting... do not over-minimize" instruction — is to restrict both
the `MoveMap` (bb+chi) and the packer's `TaskFactory` to residues within
`RELIEF_RADIUS_A` (10 Å, `relief.py`) of the ligand; everything else is held
completely fixed. Re-tested on the same complex: -668 -> -620 REU, no
explosion, and ~60x faster (6.5s vs. 392s) since the packer/minimizer only
ever touches ~20 residues instead of ~590 — which is also why the full
4,950-complex batch is actually tractable on a laptop.

`fa_rep` typically moves only modestly under this light, local relief (e.g.
713.2 -> 712.9 on that same complex) — that's expected, not a bug: Boltz-2
rotamers routinely don't match Rosetta's own rotamer-library ideal geometry
exactly, and a *light* relax isn't meant to fully erase that (see the sign
convention section below — `fa_rep_raw`/`fa_rep_relaxed` are both reported
per complex so you can see this directly rather than have it hidden).

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

## AutoDock Vina scoring (second, independent scoring function)

`src/vina_score.py` / `vina_run_batch.py` / `vina_aggregate.py` score the same
poses/manifest with AutoDock Vina
(<https://autodock-vina.readthedocs.io/en/latest/docking_python.html>) —
`v.score()` on the pose as-is, plus `v.optimize()` (quick local BFGS
minimization) for a raw-vs-optimized comparison, mirroring the PyRosetta
pipeline's raw-vs-relaxed pattern. Setup: `envs/vina_rescoring.yaml` (the
`vina` PyPI package has no prebuilt wheel for macOS arm64 — it builds from
source against Boost, installed via conda-forge first).

Ligand PDBQT prep reuses `ligand_fix.py`'s bond-order/hydrogen correction
directly (an RDKit mol in, a PDBQT out, via `meeko.MoleculePreparation` — no
intermediate file format) — every pose needs it for Vina exactly as much as
it did for Rosetta. Receptor PDBQT prep uses meeko's `mk_prepare_receptor.py`
CLI, per complex (each of a protein's ~150 poses is a distinct Boltz-2
receptor conformation, not just a different ligand placement).

**Correction/anomaly tracking, full batch (4,945/4,950 complexes):**

- Every single ligand (100%, both classes) needed the universal bond-order
  fix — not a class-differentiated signal, see above.
- Beyond that: raw-data anomalies (extra/missing ligand atoms or heavy-heavy
  bonds vs. the clean reference pose) hit **0.44% of importer complexes
  (8/1799) vs. 1.02% of non_importer complexes (32/3146) — a 2.29x higher
  rate in non-importers**, though small in absolute terms either way.
- 5 complexes (1 importer, 4 non_importer) hard-failed receptor prep: meeko's
  residue-perception flagged an unhandled apparent disulfide bond (e.g.
  Cys126-Cys134 in one case) it has no padding template for. Not retried —
  logged in `results/vina/logs/<complex_id>.log`.
- Caught mid-analysis and fixed: `ligand_fix.heavy_heavy_bonds_from_pdb`
  originally parsed CONECT records with `.split()`, which silently mangles
  lines where 5-digit atom serials (proteins with 10,000+ atoms before the
  ligand, e.g. `NPF2.10_Q944G5`) butt up against each other with no
  whitespace (PDB CONECT fields are fixed-width, not space-delimited) —
  now parsed by fixed column position. This only affected the diagnostic
  anomaly counts above, never the actual ligand correction used for scoring
  (that always used the one cached reference bond list, from a complex with
  4-digit serials) — no rescoring was needed once fixed.

**Vina score summary** (kcal/mol-scale; total = full complex binding score,
not the ligand-residue decomposition the Rosetta pipeline reports):
non-importers score *more* favorable on average than importers (`total`
median -9.45 vs -8.81 kcal/mol, Mann-Whitney p=1.4e-127) — the opposite
direction from what the label might suggest, and not directly comparable to
the Rosetta per-residue findings (different scoring function, different
quantity: whole-complex empirical score vs. local two-body decomposition).
Worth treating as a caveat, not a contradiction — Vina's function is a
general docking scorer, not tailored to this system, and pocket
volume/burial differences between classes could confound it.

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
    LIG.params                 Rosetta ligand params (rdkit_to_params output)
    atom_naming.json           fixed heavy-heavy bonds + expected heavy-atom name set
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
