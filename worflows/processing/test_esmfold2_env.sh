#!/usr/bin/env bash
# =============================================================================
# test_esmfold2_env.sh — Validate the esmfold2 conda environment on a GPU node
# =============================================================================
#
# Runs scripts/test_esmfold2.py on a real H100 to confirm that:
#   - xformers C++/CUDA extensions load (ABI match with installed torch)
#   - flash-attn is importable
#   - transformer-engine is importable
#   - esm / ESMC class import correctly
#
# Usage (from pipeline root):
#   bash worflows/processing/test_esmfold2_env.sh
#
# Output: logs/test_esmfold2_env.log  (tailed automatically after submission)
# =============================================================================

set -euo pipefail

CONDA_ENV="esmfold2"
PARTITION="gpu"
GRES="gpu:h200:1"
CPUS=2
MEM="16G"
TIME=15            # minutes — the test runs in <1 min; 15 min for queue headroom
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/test_esmfold2_env.log"
ACCOUNT=""

mkdir -p "$LOG_DIR"

ACCOUNT_FLAG=""
[[ -n "$ACCOUNT" ]] && ACCOUNT_FLAG="--account=$ACCOUNT"

echo "============================================================"
echo " ESMFold2 environment validation"
echo " Conda env  : $CONDA_ENV"
echo " GRES       : $GRES"
echo " Log        : $LOG_FILE"
echo "============================================================"
echo ""

JOB_ID=$(sbatch --parsable \
    --partition="$PARTITION" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --gres="$GRES" \
    --time="$TIME" \
    $ACCOUNT_FLAG \
    --job-name="test_esmfold2_env" \
    --output="$LOG_FILE" \
    --error="$LOG_FILE" \
    << 'SLURM_SCRIPT'
#!/usr/bin/env bash
set -euo pipefail

echo "============================================================"
echo " Host      : $(hostname)"
echo " Date      : $(date)"
echo " GPU       : $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo 'nvidia-smi unavailable')"
echo " CUDA lib  : $(nvidia-smi | grep -oP 'CUDA Version: \K[^ ]+'  2>/dev/null || echo 'unknown')"
echo "============================================================"
echo ""

module load conda
source activate esmfold2

echo "Python  : $(which python)  $(python --version)"
echo "Torch   : $(python -c 'import torch; print(torch.__version__)')"
echo ""

# Run from the pipeline root (sbatch --chdir is not used, SLURM_SUBMIT_DIR is set)
cd "$SLURM_SUBMIT_DIR"
python scripts/test_esmfold2.py
SLURM_SCRIPT
)

echo "Submitted job $JOB_ID"
echo ""
echo "Follow the log (blocks until output appears):"
echo "  tail -f $LOG_FILE"
echo ""
echo "Or check job status:"
echo "  squeue -j $JOB_ID"
