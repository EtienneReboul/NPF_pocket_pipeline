#!/usr/bin/env bash
# =============================================================================
# submit_boltz2.sh — Stage 5: Boltz-2 structure prediction on HPC
# =============================================================================
#
# Submits one SLURM job per protein × conformation combination.
# Reads the protein list from data/msa/msa.done (written by Snakefile_preprocess).
# Reads conformations from the data/boltz_inputs directory structure.
#
# Prerequisites:
#   - Snakefile_preprocess must have completed (all target.yaml files exist)
#   - conda env "boltz2" must exist on the cluster
#   - Run this script from the pipeline root directory
#
# Usage:
#   bash submit_boltz2.sh                 # submit all pending jobs
#   bash submit_boltz2.sh --dry-run       # show plan without submitting
#
# The script is idempotent: jobs with prediction.done are skipped.
# =============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
# Edit these to match your cluster and config.yaml

BOLTZ_INPUTS="data/boltz_inputs"
BOLTZ_RESULTS="results/boltz"
CONDA_ENV="boltz2"

# Boltz-2 parameters (match config.yaml)
RECYCLING_STEPS=3
DIFFUSION_SAMPLES=5
OUTPUT_FORMAT="mmcif"

# SLURM resources
PARTITION="gpu"
CPUS=8
MEM="64G"
GRES="gpu:tesla:1"
TIME=240   # minutes per job
ACCOUNT=""  # set if your cluster requires --account

# ── Parse flags ───────────────────────────────────────────────────────────────
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]] || [[ "${1:-}" == "-n" ]]; then
    DRY_RUN=true
fi

# ── Validate prerequisites ────────────────────────────────────────────────────
SENTINEL="data/msa/msa.done"
if [[ ! -f "$SENTINEL" ]]; then
    echo "ERROR: $SENTINEL not found."
    echo "       Run Snakefile_preprocess first:"
    echo "       snakemake -s Snakefile_preprocess --cores 4"
    exit 1
fi

if [[ ! -d "$BOLTZ_INPUTS" ]]; then
    echo "ERROR: $BOLTZ_INPUTS not found."
    echo "       Run Snakefile_preprocess first."
    exit 1
fi

# Detect conda base for activation on compute nodes
CONDA_BASE=$(conda info --base 2>/dev/null)
if [[ -z "$CONDA_BASE" ]]; then
    echo "ERROR: conda not found. Load your conda module first."
    exit 1
fi

# ── Build account flag if needed ──────────────────────────────────────────────
ACCOUNT_FLAG=""
if [[ -n "$ACCOUNT" ]]; then
    ACCOUNT_FLAG="--account=$ACCOUNT"
fi

# ── Submit jobs ───────────────────────────────────────────────────────────────
echo "============================================================"
echo " Boltz-2 SLURM batch submission"
echo " Inputs:  $BOLTZ_INPUTS"
echo " Results: $BOLTZ_RESULTS"
echo " Samples: $DIFFUSION_SAMPLES"
if $DRY_RUN; then
    echo " Mode:    DRY RUN (no jobs submitted)"
fi
echo "============================================================"
echo ""

submitted=0
skipped=0
missing=0

# Walk every protein × conformation that has a target.yaml
for YAML in "$BOLTZ_INPUTS"/*/*/target.yaml; do
    [[ -f "$YAML" ]] || continue

    # Extract protein and conformation from path
    CONF=$(basename "$(dirname "$YAML")")
    PROTEIN=$(basename "$(dirname "$(dirname "$YAML")")")

    DONE_FILE="$BOLTZ_RESULTS/$PROTEIN/$CONF/prediction.done"
    OUT_DIR="$BOLTZ_RESULTS/$PROTEIN/$CONF/boltz_out"
    LOG_DIR="$BOLTZ_RESULTS/$PROTEIN/$CONF/logs"

    # Skip completed
    if [[ -f "$DONE_FILE" ]]; then
        ((skipped++))
        continue
    fi

    echo "[PENDING] $PROTEIN / $CONF"

    if $DRY_RUN; then
        ((submitted++))
        continue
    fi

    mkdir -p "$OUT_DIR" "$LOG_DIR"

    sbatch \
        --partition="$PARTITION" \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task="$CPUS" \
        --mem="$MEM" \
        --gres="$GRES" \
        --time="$TIME" \
        $ACCOUNT_FLAG \
        --job-name="boltz2_${PROTEIN}_${CONF}" \
        --output="$LOG_DIR/boltz2.log" \
        --error="$LOG_DIR/boltz2.err" \
        << SLURM_SCRIPT
#!/usr/bin/env bash
set -euo pipefail

echo "[\$(date)] Boltz-2 — $PROTEIN × $CONF"
echo "  YAML:    $YAML"
echo "  Out dir: $OUT_DIR"
echo "  Samples: $DIFFUSION_SAMPLES"

# Activate conda env on compute node
set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

echo "[\$(date)] Python: \$(which python)"
echo "[\$(date)] CUDA:   \$(python -c 'import torch; print(torch.cuda.is_available())')"
echo "[\$(date)] GPU:    \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

boltz predict \\
    "$YAML" \\
    --out_dir "$OUT_DIR" \\
    --recycling_steps $RECYCLING_STEPS \\
    --diffusion_samples $DIFFUSION_SAMPLES \\
    --output_format $OUTPUT_FORMAT \\
    --no_kernels

echo "\$(date): prediction finished" > "$DONE_FILE"
echo "[\$(date)] Done."
SLURM_SCRIPT

    echo "  → submitted"
    ((submitted++))

done

echo ""
echo "============================================================"
if $DRY_RUN; then
    echo " Would submit: $submitted jobs"
else
    echo " Submitted:    $submitted jobs"
fi
echo " Skipped:      $skipped (already done)"
echo "============================================================"
echo ""
if ! $DRY_RUN && (( submitted > 0 )); then
    echo "Monitor with:"
    echo "  squeue -u \$USER"
    echo "  tail -f $BOLTZ_RESULTS/<protein>/<conformation>/logs/boltz2.log"
fi