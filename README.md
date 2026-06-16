# NPF Family Structure Modelling Pipeline

End-to-end workflow to model the Arabidopsis Nitrate Peptide Transporter Family
(NPF) in six rocking-switch conformations, profile protein–ligand interactions,
and generate ChimeraX visualisation scripts.

The pipeline is split into three independent parts so that the GPU-intensive
Boltz-2 step can run on an HPC cluster while pre- and post-processing run
locally.

---

## Pipeline overview

```text
┌─────────────────────────────────────────────────────┐
│  PRE-PROCESSING (local)   Snakefile_preprocess       │
│                                                      │
│  1. MSA          UniProt → ColabFold MSA (.a3m)      │
│        ↓                                             │
│  2. InterProScan EMBL-EBI CDD → binding-site residues│
│        ↓                                             │
│  3. Templates    RCSB → mmCIF per conformation       │
│        ↓                                             │
│  4. Boltz-2 YAML per protein × conformation          │
└──────────────────────────┬──────────────────────────┘
                           │  rsync data/boltz_inputs/
                           ▼
┌─────────────────────────────────────────────────────┐
│  PROCESSING (HPC cluster)   submit_boltz2.sh         │
│                                                      │
│  5. Boltz-2 run → mmCIF structures (N samples)       │
│     one SLURM job per protein × conformation         │
└──────────────────────────┬──────────────────────────┘
                           │  rsync results/boltz/
                           ▼
┌─────────────────────────────────────────────────────┐
│  POST-PROCESSING (local)  Snakefile_postprocess      │
│                                                      │
│  6. Minimize    ChimeraX → PDB per sample            │
│        ↓                                             │
│  7. PLIP        Docker → interaction report          │
│        ↓                                             │
│  8. pliparser   → CSV tables + ChimeraX .cxc scripts │
│        ↓                                             │
│  9. Aggregate   → summary.csv per protein×conformation│
└─────────────────────────────────────────────────────┘
```

> A monolithic `Snakefile` covering all 9 stages is also provided for
> single-machine runs (e.g. with MPS on Apple Silicon or a local GPU).

Conformations modelled (apo = no ligand, holo = ligand present):

| Folder name             | State                     |
|-------------------------|---------------------------|
| `outward_open_apo`      | Outward-open, empty       |
| `occluded_apo`          | Occluded, empty           |
| `inward_open_apo`       | Inward-open, empty        |
| `occluded_holo`         | Occluded + ligand         |
| `outward_occluded_holo` | Outward-occluded + ligand |
| `inward_occluded_holo`  | Inward-occluded + ligand  |

---

## Directory layout

```text
npf_workflow/
├── Snakefile                   ← monolithic pipeline (all 9 stages)
├── worflows/
│   ├── preprocessing/
│   │   └── Snakefile           ← stages 6–9 (local)             
│   ├── processing/
│   │   └──submit_boltz2.sh     ← stage 5 SLURM submission (cluster)  
│   ├── postprocessing/
│       └──Snakefile            ← stages 6–9 (local)        
├── config.yaml                 ← shared defaults (tracked by git)
├── config.local.yaml           ← personal overrides — CREATE THIS FIRST
├── envs/
│   ├── pipeline.yaml           ← controller env (install once)
│   ├── boltz2.yaml             ← Boltz-2 compute env
│   ├── pliparser.yaml          ← pliparser env
│   └── aggregate.yaml          ← pandas aggregation env
├── scripts/
│   ├── run_msa.py              ← Stage 1: UniProt + ColabFold
│   ├── run_interproscan.py     ← Stage 2: EBI CDD annotation
│   ├── download_templates.py   ← Stage 3: RCSB mmCIF download
│   ├── make_boltz_input.py     ← Stage 4: Boltz-2 YAML generator
│   ├── minimize_cif.py         ← Stage 6: ChimeraX minimization
│   ├── make_cxc_config.py      ← Stage 8: pliparser CXC config
│   ├── aggregate_plip_summary.py ← Stage 9: CSV merger
│   └── fix_pdb.py              ← (optional) TER/heavy-atom repair
└── data/                       ← created automatically
    ├── sequences/              ← FASTA files
    ├── msa/                    ← ColabFold outputs
    ├── interpro/               ← InterProScan JSON + residue TXT
    ├── templates/              ← mmCIF files per conformation
    └── boltz_inputs/           ← generated Boltz-2 YAMLs
```

---

## Quick start

### 1. Create config.local.yaml (required)

`config.yaml` is tracked by git and contains shared defaults. Personal values
(email, local paths) go in `config.local.yaml`, which is gitignored.

```bash
cp config.local.yaml.example config.local.yaml
# then edit config.local.yaml
```

Minimum required values:

```yaml
interproscan:
  email: "your@email.com"     # required by EMBL-EBI API

chimerax_bin: "/Applications/ChimeraX.app/Contents/MacOS/ChimeraX"
```

### 2. Install the controller environment (once)

```bash
conda env create -f envs/pipeline.yaml
conda activate npf-pipeline
```

### 3. Pull the PLIP Docker image (once)

```bash
docker pull docker.io/pharmai/plip:latest
```

---

## Running the split pipeline (recommended)

### Part 1 — Pre-processing (local)

Runs stages 1–4. Produces one `target.yaml` per protein × conformation.

```bash
snakemake -s Snakefile_preprocess --cores 4
```

### Part 2 — Processing (HPC cluster)

Transfer inputs to the cluster, submit Boltz-2 jobs, then transfer results back.

```bash
# 1. Transfer inputs to cluster
cd ../
rsync -av \
  --exclude='.snakemake/' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  --exclude='.mypy_cache/' \
  --exclude='.ruff_cache/' \
  --exclude='.cache/' \
  --exclude='.ipynb_checkpoints/' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  NPF_pocket_pipeline \
  username@your_cluster:path/to/projects/

# 2. Connect on your cluster
ssh username@your_cluster
cd path/to/projects

# 3. On the cluster — dry-run first to check the plan
#    GPU requirement: boltz2 env uses PyTorch 2.12.0+cu130 (CUDA 13.0).
#    L40S nodes have driver 580 (CUDA 13.0 ✓). A100 nodes have driver ~520
#    (CUDA 12.2 ✗ — crashes at inference). Always use L40S or verify driver ≥575.
bash submit_boltz2.sh --dry-run                      # uses gpu:l40s:1 by default

# 4. Submit all jobs
bash submit_boltz2.sh                                # default: L40S, max-concurrent 1
bash submit_boltz2.sh --max-concurrent 2             # if QoS allows 2 GPU slots

# 5. Monitor
squeue -u $USER

# 6. Transfer results back when complete
rsync -av user@cluster:project/results/boltz/ results/boltz/
```

Edit the SLURM settings at the top of `submit_boltz2.sh` to match your cluster
(partition, GPU type, account, etc.). You can also override the default `--gres`
value at runtime using the `--gres` argument shown above.

### Part 3 — Post-processing (local)

Runs stages 6–9. Discovers Boltz-2 CIF outputs automatically by globbing
`results/boltz/`. Skips any protein × conformation without a `prediction.done`.

```bash
snakemake -s Snakefile_postprocess --cores 10 --use-conda
```

> Tip: control CPU usage with `OMP_NUM_THREADS` and `MKL_NUM_THREADS`
> if ChimeraX saturates your machine during minimization.

---

## Running the monolithic pipeline (single machine)

For single-machine runs (local GPU or Apple Silicon MPS):

```bash
# Dry-run
snakemake -n --use-conda

# Full run
snakemake --cores 10 --use-conda
```

Set `accelerator: "mps"` in `config.yaml` to use Apple Silicon GPU acceleration,
which reduces Boltz-2 runtime from ~30 min to ~2 min per job.

---

## Output

```text
results/
├── boltz/
│   └── {protein}/{conformation}/
│       ├── boltz_out/predictions/…/*.cif   ← Boltz-2 structures
│       └── prediction.done
├── minimized/
│   └── {protein}/{conformation}/{sample_id}/
│       ├── model_minimized.pdb
│       └── model_minimized_energy.csv
└── plip/
    └── {protein}/{conformation}/
        ├── {sample_id}/
        │   └── model_minimized_report/
        │       ├── model_minimized_report.txt
        │       ├── model_minimized_protonated.pdb
        │       ├── csv/summary.csv
        │       └── interaction.cxc        ← open in ChimeraX
        └── summary.csv                    ← ★ final output per protein × conformation
```

---

## Key configuration options

| Key                       | Description                                       |
|---------------------------|---------------------------------------------------|
| `boltz.accelerator`       | `"cpu"`, `"mps"` (Apple Silicon), or `"gpu"` (HPC)|
| `boltz.no_kernels`        | `true` for CPU/MPS/V100S, `false` for A100/H100   |
| `boltz.diffusion_samples` | Number of structural samples per run (default 5)  |
| `boltz.ligand_smiles`     | SMILES of the transported substrate               |
| `boltz.pocket_force`      | `true` = enforce pocket via steering potential    |
| `templates.efflux`        | `true` to include MFS-MDR efflux pump templates   |
| `plip.docker_platform`    | `"linux/amd64"` needed on Apple Silicon           |
| `chimerax_bin`            | Full path to ChimeraX executable (local override) |

---

## Resuming

- All stages are resumable: existing output files are never rewritten.
- To force-rerun one stage, delete its sentinel or output file and rerun.
- `submit_boltz2.sh` skips any job whose `prediction.done` already exists.
- To add a new conformation: add it under `templates.conformations` in
  `config.yaml` and rerun — only the new conformation will be processed.

---

## Troubleshooting

| Symptom                                   | Fix                                                                                                          |
|-------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| `No cd174xx match` for a protein          | Check InterProScan JSON; protein may be unannotated in CDD                                                   |
| `No .cif files` in templates dir          | Check `data/templates/templates.done`; rerun `download_templates`                                            |
| ChimeraX segfault                         | Verify `chimerax_bin` path; test with `chimerax --nogui --exit`                                              |
| Docker permission error                   | Ensure Docker Desktop is running                                                                             |
| Boltz-2 OOM on CPU                        | Reduce `diffusion_samples` or `recycling_steps` in `config.yaml`                                             |
| Boltz-2 crashes: `NVIDIA driver too old`  | boltz2 env requires CUDA 13.0 (driver ≥575). Use `--gres gpu:l40s:1`; A100 nodes have driver ~520 (too old)  |
| PLIP segfault (exit 139)                  | Check PDB backbone; enable `fix_pdb` step if needed                                                          |
| Missing email error on startup            | Run `cp config.local.yaml.example config.local.yaml` and set your email                                      |
| Post-processing finds no targets          | Check `data/msa/msa.done` exists and `results/boltz/` was transferred                                        |
