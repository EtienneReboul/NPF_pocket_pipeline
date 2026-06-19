#!/usr/bin/env bash
# =============================================================================
# submit_esmfold2.sh — Stage 5 (alt): ESMFold2 structure prediction on HPC
# =============================================================================
#
# Drop-in alternative to submit_boltz2.sh using ESMFold2-Fast instead of
# Boltz-2. Reads the same data/boltz_inputs/PROTEIN/CONF/target.yaml files,
# extracts the protein sequence, and writes PDB output to results/esmfold2/.
#
# ESMFold2-Fast is significantly faster than Boltz-2 (~1-2 min/prediction vs
# ~8 min on Boltz-2 at 50 diffusion steps). The model is loaded ONCE per
# array task and reused across all predictions in the batch to amortise the
# HuggingFace load time.
#
# Prerequisites:
#   - Snakefile_preprocess completed (all target.yaml files exist)
#   - conda env "esmfold2" exists on the cluster (python=3.12, torch>=2.2.0,
#     pip install esm@git+https://github.com/Biohub/esm.git@main)
#   - HuggingFace weights pre-downloaded (run once on login node):
#       module load conda && source activate esmfold2
#       hf download biohub/ESMFold2-Fast
#   - Run from the pipeline root directory
#
# Usage:
#   bash submit_esmfold2.sh                                   # defaults
#   bash submit_esmfold2.sh --dry-run                         # show plan only
#   bash submit_esmfold2.sh --test                            # task 0 only
#   bash submit_esmfold2.sh --batch-size 30                   # predictions per task
#   bash submit_esmfold2.sh --max-concurrent 10               # max parallel tasks
#   bash submit_esmfold2.sh --num-steps 20                    # faster / lower quality
#   bash submit_esmfold2.sh --gres gpu:h200:1                 # H200 80GB+ (default, recommended)
#   bash submit_esmfold2.sh --gres gpu:l40s:1                 # L40S 44GB — OOMs on long sequences
#
# GPU COMPATIBILITY NOTE (esmfold2 env uses PyTorch 2.12.1+cu130):
#   Available GRES on this cluster (sinfo -o "%P %G"):
#     gpu:h200:4   — H200, 80–141 GB VRAM ✓  recommended (model=36 GB + activations)
#     gpu:l40s:3/4 — L40S, 44 GB VRAM   ✗  model loads but OOMs on long sequences
#     gpu:3g.20gb  — MIG slice, 20 GB    ✗  far too small
#   Use: --gres gpu:h200:1
#
# The script is idempotent: predictions with prediction.done are skipped.
# =============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
ESM_INPUTS="data/boltz_inputs"      # reuse same YAML input tree as Boltz-2
ESM_RESULTS="results/esmfold2"
CONDA_ENV="esmfold2"

# HuggingFace cache — set to a project/scratch dir to avoid home quota issues.
# Pre-download weights on the login node with:
#   export HF_HOME="$HF_CACHE_DIR" && huggingface-cli download biohub/ESMFold2-Fast
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"

# ESMFold2 inference parameters
NUM_LOOPS=3      # recycling steps (≈ recycling_steps in Boltz-2)
NUM_STEPS=50     # diffusion sampling steps (≈ diffusion_samples in Boltz-2)

# Batching — ESMFold2-Fast is much faster than Boltz-2; larger batches are safe
BATCH_SIZE=30       # predictions per array task (~1-2 min/pred → ~60 min/task)
MAX_CONCURRENT=10   # max array tasks running at once (SLURM % syntax)

# SLURM resources (per array task)
PARTITION="gpu"
CPUS=4
MEM="32G"
GRES="gpu:h200:1"
TIME=120            # minutes per task (conservative; ESMFold2 is fast)
ACCOUNT=""

# ── Parse arguments ───────────────────────────────────────────────────────────
DRY_RUN=false
TEST_MODE=false

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
        --num-steps)
            NUM_STEPS="$2"; shift 2 ;;
        --num-steps=*)
            NUM_STEPS="${1#--num-steps=}"; shift ;;
        --num-loops)
            NUM_LOOPS="$2"; shift 2 ;;
        --num-loops=*)
            NUM_LOOPS="${1#--num-loops=}"; shift ;;
        --test)
            TEST_MODE=true; shift ;;
        *)
            echo "ERROR: unknown argument '$1'"
            echo "Usage: bash submit_esmfold2.sh [--dry-run] [--test] [--gres <profile>]"
            echo "                               [--batch-size N] [--max-concurrent N]"
            echo "                               [--num-steps N] [--num-loops N]"
            exit 1 ;;
    esac
done

# ── Validate prerequisites ────────────────────────────────────────────────────
if [[ ! -f "data/msa/msa.done" ]]; then
    echo "ERROR: data/msa/msa.done not found. Run Snakefile_preprocess first."
    exit 1
fi
if [[ ! -d "$ESM_INPUTS" ]]; then
    echo "ERROR: $ESM_INPUTS not found. Run Snakefile_preprocess first."
    exit 1
fi

ACCOUNT_FLAG=""
[[ -n "$ACCOUNT" ]] && ACCOUNT_FLAG="--account=$ACCOUNT"

# ── Collect pending predictions ───────────────────────────────────────────────
PENDING_YAMLS=()
PENDING_DONE=()
skipped=0

for YAML in "$ESM_INPUTS"/*/*/target.yaml; do
    [[ -f "$YAML" ]] || continue
    CONF=$(basename "$(dirname "$YAML")")
    PROTEIN=$(basename "$(dirname "$(dirname "$YAML")")")
    DONE_FILE="$ESM_RESULTS/$PROTEIN/$CONF/prediction.done"
    if [[ -f "$DONE_FILE" ]]; then
        skipped=$((skipped + 1))
        continue
    fi
    PENDING_YAMLS+=("$YAML")
    PENDING_DONE+=("$DONE_FILE")
done

TOTAL=${#PENDING_YAMLS[@]}
N_TASKS=$(( (TOTAL + BATCH_SIZE - 1) / BATCH_SIZE ))
LAST_TASK=$(( N_TASKS - 1 ))
ARRAY_SPEC="0-${LAST_TASK}%${MAX_CONCURRENT}"

echo "============================================================"
echo " ESMFold2 SLURM job array submission"
echo " Pending predictions : $TOTAL"
echo " Already done        : $skipped"
echo " Batch size          : $BATCH_SIZE predictions/task"
echo " Array tasks         : $N_TASKS  (--array=${ARRAY_SPEC})"
echo " Max concurrent      : $MAX_CONCURRENT"
echo " GRES                : $GRES"
echo " Time limit/task     : ${TIME} min"
echo " Num loops / steps   : ${NUM_LOOPS} / ${NUM_STEPS}"
echo " HF cache            : $HF_CACHE_DIR"
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
MANIFEST_DIR="$ESM_RESULTS/array_manifest"
mkdir -p "$MANIFEST_DIR"
MANIFEST="$MANIFEST_DIR/manifest.txt"

: > "$MANIFEST"
for (( i=0; i<TOTAL; i++ )); do
    yaml="${PENDING_YAMLS[$i]}"
    done_file="${PENDING_DONE[$i]}"
    conf=$(basename "$(dirname "$yaml")")
    protein=$(basename "$(dirname "$(dirname "$yaml")")")
    out_dir="$ESM_RESULTS/$protein/$conf"
    mkdir -p "$out_dir/logs"
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
    --job-name="esmfold2_array" \
    --output="$LOG_DIR/task_%a.log" \
    --error="$LOG_DIR/task_%a.err" \
    << SLURM_SCRIPT
#!/usr/bin/env bash
set -euo pipefail

TASK_ID=\$SLURM_ARRAY_TASK_ID
BATCH_SIZE=$BATCH_SIZE
MANIFEST="$MANIFEST"

export HF_HOME="$HF_CACHE_DIR"
export PYTHONWARNINGS="ignore::DeprecationWarning,ignore::FutureWarning"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "[\$(date)] Array task \$TASK_ID — batch size $BATCH_SIZE"
echo "  Manifest : \$MANIFEST"
echo "  HF cache : \$HF_HOME"

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

# Extract this task's manifest slice into a temp file
SLICE=\$(mktemp /tmp/esmfold2_slice_XXXX.txt)
sed -n "\${LINE_START},\${LINE_END}p" "\$MANIFEST" > "\$SLICE"

# Write the batch runner — loads model ONCE, iterates over all predictions
RUNNER=\$(mktemp /tmp/esmfold2_runner_XXXX.py)
cat > "\$RUNNER" << 'PYEOF'
import sys
import datetime
import torch
import yaml as pyyaml
from pathlib import Path
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

NUM_LOOPS = int(sys.argv[1])
NUM_STEPS = int(sys.argv[2])
slice_file = sys.argv[3]

entries = []
with open(slice_file) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        yaml_path, out_dir, done_file = line.split("|")
        entries.append((yaml_path, out_dir, done_file))

print(f"[{datetime.datetime.now()}] Loading ESMFold2-Fast (num_loops={NUM_LOOPS}, num_steps={NUM_STEPS})...", flush=True)
model = ESMFold2Model.from_pretrained("biohub/ESMFold2-Fast").cuda().eval()
print(f"[{datetime.datetime.now()}] Model ready. {len(entries)} prediction(s) in this batch.", flush=True)
print("", flush=True)

for yaml_path, out_dir, done_file in entries:
    if Path(done_file).exists():
        print(f"[{datetime.datetime.now()}] SKIP: {yaml_path} (already done)", flush=True)
        continue

    # Parse Boltz-2 YAML to extract protein sequence(s)
    with open(yaml_path) as f:
        data = pyyaml.safe_load(f)

    sequences = []
    for entry in data.get("sequences", []):
        if "protein" in entry:
            seq = entry["protein"].get("sequence", "")
            if seq:
                sequences.append(seq)

    if not sequences:
        print(f"[{datetime.datetime.now()}] ERROR: no protein sequence found in {yaml_path}", flush=True)
        continue

    # ESMFold2 folds one chain at a time; use the first protein sequence.
    # For multi-chain complexes, infer_complex() may be available instead.
    sequence = sequences[0]

    print(f"[{datetime.datetime.now()}] START: {yaml_path}  (len={len(sequence)})", flush=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        output = model.infer_protein(sequence, num_loops=NUM_LOOPS, num_sampling_steps=NUM_STEPS)

    # Save structure — ESMFold2 returns a ProteinChain; use .to_pdb() to serialise.
    # If infer_protein returns something else (e.g. a dict with 'positions'),
    # adjust the block below accordingly.
    out_pdb = Path(out_dir) / "structure.pdb"
    if hasattr(output, "to_pdb"):
        out_pdb.write_text(output.to_pdb())
    elif hasattr(output, "atoms"):
        # biotite AtomArray path (ESM3-style)
        import biotite.structure.io.pdb as pdb_io
        import io
        buf = io.StringIO()
        pdb_file = pdb_io.PDBFile()
        pdb_io.set_structure(pdb_file, output.atoms)
        pdb_file.write(buf)
        out_pdb.write_text(buf.getvalue())
    else:
        print(f"[{datetime.datetime.now()}] WARNING: unknown output type {type(output)}, dumping repr to {out_pdb}.txt", flush=True)
        (out_pdb.parent / "output_repr.txt").write_text(repr(output))

    Path(done_file).parent.mkdir(parents=True, exist_ok=True)
    Path(done_file).write_text(f"{datetime.datetime.now().isoformat()}: prediction finished\n")
    print(f"[{datetime.datetime.now()}] DONE:  {yaml_path}", flush=True)
    print("", flush=True)

print(f"[{datetime.datetime.now()}] Batch complete.", flush=True)
PYEOF

python "\$RUNNER" $NUM_LOOPS $NUM_STEPS "\$SLICE"

rm -f "\$RUNNER" "\$SLICE"

echo "[\$(date)] Array task \$TASK_ID complete."
SLURM_SCRIPT

echo "Job array submitted: --array=${ARRAY_SPEC}"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f $LOG_DIR/task_0.log"
