"""Shared paths and constants for the PyRosetta rescoring project.

This project lives inside NPF_pocket_pipeline but is a standalone,
non-Snakemake pipeline (see ../README.md) — Tier 3 of the rescoring plan in
../../HANDOFF_rescoring.md.
"""

from pathlib import Path

import yaml

RESCORING_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = RESCORING_ROOT.parent

DATA_DIR = RESCORING_ROOT / "data"
LDA_INPUTS_DIR = DATA_DIR / "lda_kernel_inputs"
PARAMS_DIR = RESCORING_ROOT / "params"
RESULTS_DIR = RESCORING_ROOT / "results"
PER_COMPLEX_DIR = RESULTS_DIR / "per_complex"
LOGS_DIR = RESULTS_DIR / "logs"
FIGURES_DIR = RESULTS_DIR / "figures"

VINA_RESULTS_DIR = RESULTS_DIR / "vina"
VINA_PDBQT_DIR = VINA_RESULTS_DIR / "pdbqt"
VINA_PER_COMPLEX_DIR = VINA_RESULTS_DIR / "per_complex"
VINA_LOGS_DIR = VINA_RESULTS_DIR / "logs"

MINIMIZED_DIR = PIPELINE_ROOT / "results" / "minimized_synth"
SEQUENCES_DIR = PIPELINE_ROOT / "data" / "sequences"
INTERPRO_DIR = PIPELINE_ROOT / "data" / "interpro"

LABELS_CSV = DATA_DIR / "labels.csv"
LDA_RESIDUES_CSV = DATA_DIR / "lda_residues.csv"
POSITION_RESNR_MAP_CSV = DATA_DIR / "position_resnr_map.csv"
MANIFEST_CSV = DATA_DIR / "manifest.csv"

LIGAND_RESNAME = "LIG"
LIGAND_CHAIN = "L"
PROTEIN_CHAIN = "A"

# The CDD Feature-1 (cd17351) MSA anchor: the one root-entry sequence in the
# hc corpus, whose own binding-site residue list defines the 35 alignment
# columns used as "position" 1..35 everywhere in NPF_LDA_kernel's hc outputs.
ANCHOR_PROTEIN = "NPF6.1_Q9LYR6"
ANCHOR_UNIPROT_ID = "Q9LYR6"


def load_ligand_smiles() -> str:
    """Single source of truth: the GA1 SMILES already used to run Boltz-2 (config.yaml)."""
    cfg = yaml.safe_load((PIPELINE_ROOT / "config.yaml").read_text())
    return cfg["boltz"]["ligand_smiles"]


for _d in (DATA_DIR, PARAMS_DIR, PER_COMPLEX_DIR, LOGS_DIR, FIGURES_DIR,
           VINA_PDBQT_DIR, VINA_PER_COMPLEX_DIR, VINA_LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
