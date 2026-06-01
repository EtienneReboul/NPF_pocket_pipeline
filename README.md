# NPF Family Structure Modelling Pipeline

End-to-end Snakemake workflow to model the Arabidopsis Nitrate Peptide
Transporter Family (NPF) in six rocking-switch conformations, profile
protein–ligand interactions, and generate ChimeraX visualisation scripts.

---

## Pipeline overview

```
1. MSA           UniProt → per-protein FASTA + ColabFold MSA (.a3m)
       ↓
2. InterProScan  EMBL-EBI CDD → binding-site residues per protein (cd174xx)
       ↓
3. Templates     RCSB → mmCIF templates per conformation folder
       ↓
4. Boltz-2 input  per protein × conformation → target.yaml
    (pocket constraint from CDD residues, holo/apo from folder name)
       ↓
5. Boltz-2 run   → mmCIF structure predictions (N diffusion samples)
       ↓
6. Minimize      ChimeraX energy minimization → PDB per sample
       ↓
7. PLIP          Docker → protein–ligand interaction report per sample
       ↓
8. pliparser     → CSV interaction tables + ChimeraX .cxc scripts
       ↓
9. Aggregate     → summary.csv per protein × conformation  ★
```

Conformations modelled (apo = no ligand, holo = ligand present):

| Folder name             | State                    |
|-------------------------|--------------------------|
| `outward_open_apo`      | Outward-open, empty      |
| `occluded_apo`          | Occluded, empty          |
| `inward_open_apo`       | Inward-open, empty       |
| `occluded_holo`         | Occluded + ligand        |
| `outward_occluded_holo` | Outward-occluded + ligand|
| `inward_occluded_holo`  | Inward-occluded + ligand |

---

## Directory layout

```
npf_workflow/
├── Snakefile
├── config.yaml                 ← edit this first
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
│   ├── fix_pdb.py              ← (optional) TER/heavy-atom repair
│   ├── make_cxc_config.py      ← Stage 8: pliparser CXC config
│   ├── aggregate_plip_summary.py ← Stage 9: CSV merger
│   └── aggregate_summaries.py  ← (legacy, kept for reference)
└── data/                       ← created automatically
    ├── sequences/              ← FASTA files
    ├── msa/                    ← ColabFold outputs
    ├── interpro/               ← InterProScan JSON + residue TXT
    ├── templates/              ← mmCIF files per conformation
    └── boltz_inputs/           ← generated Boltz-2 YAMLs
```

---

## Quick start

### 1. Edit config.yaml

Minimum required edits:

```yaml
interproscan:
  email: "your@email.com"       # required by EMBL-EBI REST API

chimerax_bin: "/Applications/ChimeraX.app/Contents/MacOS/ChimeraX"

boltz:
  ligand_smiles: "[O-][N+](=O)[O-]"  # update if not nitrate
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

### 4. Dry-run — check the plan

```bash
snakemake -n --use-conda
```

### 5. Full run

```bash
snakemake --cores 4 --use-conda
```

> On first run, Snakemake builds per-rule conda environments automatically.
> This takes a few minutes once, then they are cached.

---

## Output

```
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
        │       └── interaction.cxc             ← open in ChimeraX
        └── summary.csv                         ← ★ per protein × conformation
```

---

## Key configuration options

| Key | Description |
|-----|-------------|
| `boltz.accelerator` | `"cpu"` (local Mac) or `"gpu"` (HPC) |
| `boltz.no_kernels` | `true` for CPU/V100S, `false` for A100/H100 |
| `boltz.diffusion_samples` | Number of structural samples per run (default 5) |
| `boltz.ligand_smiles` | SMILES of the transported substrate |
| `boltz.pocket_force` | `true` = enforce pocket via steering potential |
| `templates.efflux` | `true` to include MFS-MDR efflux pump templates |
| `plip.docker_platform` | `"linux/amd64"` needed on Apple Silicon |
| `chimerax_bin` | Full path to ChimeraX executable |

---

## HPC / SLURM

To run on a cluster, change in `config.yaml`:

```yaml
boltz:
  accelerator: "gpu"
  no_kernels: false    # if cluster has A100/H100
```

Then submit with the SLURM executor:

```bash
snakemake --executor slurm --cores 100 --use-conda \
    --default-resources slurm_partition=gpu mem_mb=64000
```

---

## Resuming

- All stages are resumable: existing output files are never rewritten.
- To force-rerun a single stage, delete its sentinel or output and rerun.
- To add a new conformation: add its entry under `templates.conformations`
  in `config.yaml` and rerun — only the new conformation will be processed.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `No cd174xx match` for a protein | Check InterProScan JSON manually; the protein may be unannotated in CDD |
| `No .cif files` in templates dir | Check `data/templates.done`; rerun `download_templates` rule |
| ChimeraX segfault | Verify `chimerax_bin` path; test with `chimerax --nogui --exit` |
| Docker permission error | Ensure Docker is running; on Mac check Docker Desktop is open |
| Boltz-2 OOM on CPU | Reduce `diffusion_samples` or `recycling_steps` in config.yaml |
| PLIP segfault (exit 139) | Check PDB for broken backbone; `fix_pdb` step can be enabled if needed |
