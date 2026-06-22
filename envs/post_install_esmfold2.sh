#!/usr/bin/env bash
# Post-install: flash-attn + transformer-engine for the esmfold2 env.
# These require nvcc at build time, so they can't go in the conda yaml.
#
# Run on a login node that has the CUDA toolkit module available:
#   nohup bash envs/post_install_esmfold2.sh >> logs/esmfold2_rebuild.log 2>&1 &

set -euo pipefail

echo "[$(date)] Starting post-install for esmfold2"

# ── Load modules ──────────────────────────────────────────────────────────────
module load conda

# Load the CUDA toolkit to make nvcc available.
# Adjust the module name to match your cluster (check with: module avail cuda).
module load cuda-toolkit/12.9.1

# ── Activate env ──────────────────────────────────────────────────────────────
source activate esmfold2

# ── Set CUDA_HOME from nvcc location ─────────────────────────────────────────
NVCC_PATH=$(which nvcc 2>/dev/null || true)
if [[ -z "$NVCC_PATH" ]]; then
    echo "ERROR: nvcc not found after module load. Check: module avail cuda"
    exit 1
fi
export CUDA_HOME
CUDA_HOME=$(dirname "$(dirname "$NVCC_PATH")")
echo "[$(date)] nvcc     : $NVCC_PATH"
echo "[$(date)] CUDA_HOME: $CUDA_HOME"
echo "[$(date)] torch    : $(python -c 'import torch; print(torch.__version__)')"
echo ""

# ── Install ───────────────────────────────────────────────────────────────────
echo "[$(date)] Installing flash-attn ..."
pip install flash-attn --no-build-isolation

echo "[$(date)] Installing transformer-engine ..."
pip install "transformer-engine[pytorch]"

echo ""
echo "[$(date)] Post-install complete."
