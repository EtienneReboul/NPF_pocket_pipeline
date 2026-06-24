#!/usr/bin/env bash
# =============================================================================
# tar_synth_results.sh — Archive results/boltz_synth/ for download
# =============================================================================
# Submits a short CPU job that tars the synthetic-run Boltz-2 outputs into
# a single file at the project root so you can scp/rsync it in one shot.
#
# Usage:
#   bash worflows/processing/tar_synth_results.sh
#
# Output: results/boltz_synth.tar.gz
# =============================================================================

set -euo pipefail

SRC="results/boltz_synth"
ARCHIVE="results/boltz_synth.tar.gz"
PARTITION="fast"          # CPU partition — no GPU needed
CPUS=4
MEM="16G"
TIME=120                  # minutes — adjust if the directory is very large

if [[ ! -d "$SRC" ]]; then
    echo "ERROR: $SRC not found. Has the Boltz-2 synth run completed?"
    exit 1
fi

sbatch \
    --partition="$PARTITION" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --time="$TIME" \
    --job-name="tar_synth" \
    --output="logs/tar_synth.log" \
    --error="logs/tar_synth.err" \
    << 'SLURM_SCRIPT'
#!/usr/bin/env bash
set -euo pipefail

SRC="results/boltz_synth"
ARCHIVE="results/boltz_synth.tar.gz"

echo "[$(date)] Starting tar of $SRC"
echo "[$(date)] Destination: $ARCHIVE"

tar -czf "$ARCHIVE" "$SRC"

SIZE=$(du -sh "$ARCHIVE" | cut -f1)
echo "[$(date)] Done. Archive size: $SIZE"
echo "[$(date)] Download with:"
echo "  rsync -av --progress ereboul@core.cluster.france-bioinformatique.fr:/shared/projects/npf_abinitio/NPF_pocket_pipeline/$ARCHIVE ."
SLURM_SCRIPT

echo "Job submitted. Monitor with:"
echo "  tail -f logs/tar_synth.log"
