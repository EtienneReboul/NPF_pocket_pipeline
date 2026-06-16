#!/usr/bin/env bash
# =============================================================================
# submit_boltz2.sh — Stage 5: Boltz-2 structure prediction on HPC
# =============================================================================
#
# Collects all pending protein × conformation predictions, packs them into
# batches, and submits a single SLURM job array — one array task per batch.
# Max concurrent tasks is controlled with --max-concurrent (uses % syntax).
#
# With ~8 min/prediction on A100 and --batch-size 20:
#   20 × 8 min = ~160 min per array task → fits in a 4-hour slot.
#
# Example: 317 pending predictions, batch-size 20, max-concurrent 10
#   → 16 array tasks (0-15), at most 10 running at once → sbatch --array=0-15%10
#
# Prerequisites:
#   - Snakefile_preprocess completed (all target.yaml files exist)
#   - conda env "boltz2" exists on the cluster
#   - Run from the pipeline root directory
#
# Usage:
#   bash submit_boltz2.sh                                   # defaults
#   bash submit_boltz2.sh --dry-run                         # show plan only
#   bash submit_boltz2.sh --test                            # submit task 0 only (QoS-safe test)
#   bash submit_boltz2.sh --batch-size 20                   # predictions per task
#   bash submit_boltz2.sh --max-concurrent 10               # max parallel tasks
#   bash submit_boltz2.sh --gres gpu:l40s:1                 # L40S (default, recommended)
#   bash submit_boltz2.sh --gres gpu:7g.40gb:1              # A100 full — ONLY if driver ≥575
#   bash submit_boltz2.sh --gres gpu:tesla:1                # V100S
#   bash submit_boltz2.sh --test --gres gpu:l40s:1          # test on L40S
#
# GPU COMPATIBILITY NOTE (boltz2 env uses PyTorch 2.12.0+cu130, requires CUDA 13.0):
#   L40S nodes  — driver 580 ✓  (verified 2026-06-16)
#   A100 nodes  — driver ~520 ✗  crashes at inference (CUDA 12.2 max, too old)
#   bash submit_boltz2.sh --batch-size 20 --max-concurrent 5 --dry-run
#
# The script is idempotent: predictions with prediction.done are skipped.
# =============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
BOLTZ_INPUTS="data/boltz_inputs"
BOLTZ_RESULTS="results/boltz"
CONDA_ENV="boltz2"

# Boltz-2 parameters (match config.yaml)
RECYCLING_STEPS=3
DIFFUSION_SAMPLES=5
OUTPUT_FORMAT="mmcif"

# Batching
BATCH_SIZE=20       # predictions per array task (~8 min each on A100 → 160 min/task)
MAX_CONCURRENT=10   # max array tasks running at once (SLURM % syntax)

# SLURM resources (per array task)
PARTITION="gpu"
CPUS=8
MEM="64G"
# IMPORTANT: must target L40S (driver 580, CUDA 13.0 compatible).
# A100 nodes on this cluster run driver ~520 (CUDA 12.2 max) and crash with
# the boltz2 env (PyTorch 2.12.0+cu130). L40S nodes have driver 580 and work.
GRES="gpu:l40s:1"
TIME=240                # minutes per task
ACCOUNT=""

# ── Parse arguments ───────────────────────────────────────────────────────────
DRY_RUN=false
TEST_MODE=false   # --test: submit only task 0 to verify config before full run

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run|-n)
            DRY_RUN=true; shift ;;
        --gres)
            GRES="$2"; shift 2 ;;
        --gres=*)
            GRES="${1#--gres=}"; shift ;;
        --batch-size)
            BATCH_SIZE="$2"; shift 2 ;;
        --batch-size=*)
            BATCH_SIZE="${1#--batch-size=}"; shift ;;
        --max-concurrent)
            MAX_CONCURRENT="$2"; shift 2 ;;
        --max-concurrent=*)
            MAX_CONCURRENT="${1#--max-concurrent=}"; shift ;;
        --test)
            TEST_MODE=true; shift ;;
        *)
            echo "ERROR: unknown argument '$1'"
            echo "Usage: bash submit_boltz2.sh [--dry-run] [--test] [--gres <profile>]"
            echo "                             [--batch-size N] [--max-concurrent N]"
            exit 1 ;;
    esac
done

# ── Validate prerequisites ────────────────────────────────────────────────────
if [[ ! -f "data/msa/msa.done" ]]; then
    echo "ERROR: data/msa/msa.done not found. Run Snakefile_preprocess first."
    exit 1
fi
if [[ ! -d "$BOLTZ_INPUTS" ]]; then
    echo "ERROR: $BOLTZ_INPUTS not found. Run Snakefile_preprocess first."
    exit 1
fi

ACCOUNT_FLAG=""
[[ -n "$ACCOUNT" ]] && ACCOUNT_FLAG="--account=$ACCOUNT"

# ── Collect pending predictions ───────────────────────────────────────────────
PENDING_YAMLS=()
PENDING_DONE=()
skipped=0

for YAML in "$BOLTZ_INPUTS"/*/*/target.yaml; do
    [[ -f "$YAML" ]] || continue
    CONF=$(basename "$(dirname "$YAML")")
    PROTEIN=$(basename "$(dirname "$(dirname "$YAML")")")
    DONE_FILE="$BOLTZ_RESULTS/$PROTEIN/$CONF/prediction.done"
    if [[ -f "$DONE_FILE" ]]; then
        skipped=$((skipped + 1))
        continue
    fi
    PENDING_YAMLS+=("$YAML")
    PENDING_DONE+=("$DONE_FILE")
done

TOTAL=${#PENDING_YAMLS[@]}
N_TASKS=$(( (TOTAL + BATCH_SIZE - 1) / BATCH_SIZE ))   # ceiling division
LAST_TASK=$(( N_TASKS - 1 ))
ARRAY_SPEC="0-${LAST_TASK}%${MAX_CONCURRENT}"

echo "============================================================"
echo " Boltz-2 SLURM job array submission"
echo " Pending predictions : $TOTAL"
echo " Already done        : $skipped"
echo " Batch size          : $BATCH_SIZE predictions/task"
echo " Array tasks         : $N_TASKS  (--array=${ARRAY_SPEC})"
echo " Max concurrent      : $MAX_CONCURRENT"
echo " GRES                : $GRES"
echo " Time limit/task     : ${TIME} min"
if $DRY_RUN; then
    echo " Mode                : DRY RUN (nothing submitted)"
fi
if $TEST_MODE; then
    echo " Mode                : TEST — task 0 only ($(( BATCH_SIZE < TOTAL ? BATCH_SIZE : TOTAL )) prediction(s))"
fi
echo "============================================================"
echo ""

if [[ $TOTAL -eq 0 ]]; then
    echo "Nothing to do — all predictions already complete."
    exit 0
fi

# ── Write the batch manifest ──────────────────────────────────────────────────
# One line per pending prediction: "yaml|out_dir|done_file"
# The array task uses $SLURM_ARRAY_TASK_ID to pick its slice of lines.

MANIFEST_DIR="$BOLTZ_RESULTS/array_manifest"
mkdir -p "$MANIFEST_DIR"
MANIFEST="$MANIFEST_DIR/manifest.txt"

: > "$MANIFEST"   # truncate
for (( i=0; i<TOTAL; i++ )); do
    yaml="${PENDING_YAMLS[$i]}"
    done_file="${PENDING_DONE[$i]}"
    conf=$(basename "$(dirname "$yaml")")
    protein=$(basename "$(dirname "$(dirname "$yaml")")")
    out_dir="$BOLTZ_RESULTS/$protein/$conf/boltz_out"
    mkdir -p "$out_dir" "$BOLTZ_RESULTS/$protein/$conf/logs"
    echo "${yaml}|${out_dir}|${done_file}" >> "$MANIFEST"
done

echo "Manifest written: $MANIFEST ($TOTAL lines)"
echo ""

if $DRY_RUN; then
    echo "Dry-run — array tasks breakdown:"
    for (( task=0; task<N_TASKS; task++ )); do
        start=$(( task * BATCH_SIZE ))
        end=$(( start + BATCH_SIZE - 1 ))
        [[ $end -ge $TOTAL ]] && end=$(( TOTAL - 1 ))
        count=$(( end - start + 1 ))
        echo "  Task $task: $count prediction(s) (lines $((start+1))–$((end+1)))"
        for (( i=start; i<=end; i++ )); do
            yaml="${PENDING_YAMLS[$i]}"
            conf=$(basename "$(dirname "$yaml")")
            protein=$(basename "$(dirname "$(dirname "$yaml")")")
            echo "    $protein / $conf"
        done
    done
    echo ""
    echo "Would submit: sbatch --array=${ARRAY_SPEC} ..."
    exit 0
fi

# ── Submit the job array ──────────────────────────────────────────────────────
LOG_DIR="$MANIFEST_DIR/logs"
mkdir -p "$LOG_DIR"

# In test mode, override the array spec to task 0 only (no % limit needed)
if $TEST_MODE; then
    ARRAY_SPEC="0"
    echo "TEST MODE: submitting task 0 only (${BATCH_SIZE} prediction(s))"
    echo "           Once it completes successfully, run without --test for the full array."
    echo ""
fi

sbatch \
    --partition="$PARTITION" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --gres="$GRES" \
    --time="$TIME" \
    --array="${ARRAY_SPEC}" \
    $ACCOUNT_FLAG \
    --job-name="boltz2_array" \
    --output="$LOG_DIR/task_%a.log" \
    --error="$LOG_DIR/task_%a.err" \
    << SLURM_SCRIPT
#!/usr/bin/env bash
set -euo pipefail

TASK_ID=\$SLURM_ARRAY_TASK_ID
BATCH_SIZE=$BATCH_SIZE
MANIFEST="$MANIFEST"

echo "[\$(date)] Array task \$TASK_ID — batch size $BATCH_SIZE"
echo "  Manifest: \$MANIFEST"

# Activate conda
module load conda
source activate "$CONDA_ENV"

echo "[\$(date)] Python : \$(which python)"
echo "[\$(date)] CUDA   : \$(python -c 'import torch; print(torch.cuda.is_available())')"
echo "[\$(date)] GPU    : \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

# Compute line range for this task (1-based for sed)
LINE_START=\$(( TASK_ID * BATCH_SIZE + 1 ))
LINE_END=\$(( LINE_START + BATCH_SIZE - 1 ))

echo "[\$(date)] Processing manifest lines \$LINE_START–\$LINE_END"
echo ""

while IFS='|' read -r yaml out_dir done_file; do
    [[ -z "\$yaml" ]] && continue

    conf=\$(basename "\$(dirname "\$yaml")")
    protein=\$(basename "\$(dirname "\$(dirname "\$yaml")")")

    if [[ -f "\$done_file" ]]; then
        echo "[\$(date)] SKIP: \$protein / \$conf (already done)"
        continue
    fi

    echo "[\$(date)] START: \$protein / \$conf"

    boltz predict \\
        "\$yaml" \\
        --out_dir "\$out_dir" \\
        --recycling_steps $RECYCLING_STEPS \\
        --diffusion_samples $DIFFUSION_SAMPLES \\
        --output_format $OUTPUT_FORMAT
    # --no_kernels  ← uncomment for V100S or CPU; not needed for A100

    echo "\$(date): prediction finished" > "\$done_file"
    echo "[\$(date)] DONE:  \$protein / \$conf"
    echo ""

done < <(sed -n "\${LINE_START},\${LINE_END}p" "\$MANIFEST")

echo "[\$(date)] Array task \$TASK_ID complete."
SLURM_SCRIPT

echo "Job array submitted: --array=${ARRAY_SPEC}"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f $LOG_DIR/task_0.log"