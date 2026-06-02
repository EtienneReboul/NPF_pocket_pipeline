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
#   bash submit_boltz2.sh                             # submit all pending jobs
#   bash submit_boltz2.sh --dry-run                   # show plan without submitting
#   bash submit_boltz2.sh --gres gpu:3g.20gb:1        # IFB A100 20GB MIG slice
#   bash submit_boltz2.sh --gres gpu:7g.40gb:1        # IFB A100 full 40GB card
#   bash submit_boltz2.sh --gres gpu:tesla:1          # old cluster (V100S)
#   bash submit_boltz2.sh --gres gpu:tesla:1 --dry-run  # combine flags freely
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
GRES="gpu:3g.20gb:1"   # default — override with --gres <profile> on the command line
TIME=240   # minutes per job (4 hours)
             # GPU (V100S): ~8 min/job → 4h is very generous, reduce if queue is busy
             # GPU (A100):  ~3 min/job → could use TIME=30
ACCOUNT=""  # set if your cluster requires --account

# ── Parse flags ───────────────────────────────────────────────────────────────
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run|-n)
            DRY_RUN=true
            shift
            ;;
        --gres)
            GRES="$2"
            shift 2
            ;;
        --gres=*)
            GRES="${1#--gres=}"
            shift
            ;;
        *)
            echo "ERROR: unknown argument '$1'"
            echo "Usage: bash submit_boltz2.sh [--dry-run] [--gres <gpu_profile>]"
            exit 1
            ;;
    esac
done

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
        skipped=$((skipped + 1))
        continue
    fi

    echo "[PENDING] $PROTEIN / $CONF"

    if $DRY_RUN; then
        submitted=$((submitted + 1))
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
        --job-name="b2_${PROTEIN:0:6}_${CONF:0:6}" \
        --output="$LOG_DIR/boltz2.log" \
        --error="$LOG_DIR/boltz2.err" \
        << SLURM_SCRIPT
#!/usr/bin/env bash
set -euo pipefail

echo "[\$(date)] Boltz-2 — $PROTEIN × $CONF"
echo "  YAML:    $YAML"
echo "  Out dir: $OUT_DIR"
echo "  Samples: $DIFFUSION_SAMPLES"

# Activate conda env on IFB (uses module system, not conda base path)
module load conda
source activate "$CONDA_ENV"

echo "[\$(date)] Python: \$(which python)"
echo "[\$(date)] CUDA:   \$(python -c 'import torch; print(torch.cuda.is_available())')"
echo "[\$(date)] GPU:    \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

boltz predict \\
    "$YAML" \\
    --out_dir "$OUT_DIR" \\
    --recycling_steps $RECYCLING_STEPS \\
    --diffusion_samples $DIFFUSION_SAMPLES \\
    --output_format $OUTPUT_FORMAT \\
    # --no_kernels   # remove this comment for V100S/CPU; A100 does NOT need this

echo "\$(date): prediction finished" > "$DONE_FILE"
echo "[\$(date)] Done."
SLURM_SCRIPT

    echo "  → submitted"
    submitted=$((submitted + 1))

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