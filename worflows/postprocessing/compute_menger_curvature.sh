#!/usr/bin/env bash
# worflows/postprocessing/compute_menger_curvature.sh
# =====================================================
# Computes per-residue Menger curvature for every protein in results/boltz_synth/,
# pooling all conformations into one pseudo-ensemble per protein.
#
# For each protein a CSV and a .npy file are written:
#   results/menger_curvature_synth/{protein}/curvature.csv
#   results/menger_curvature_synth/{protein}/curvature.npy
#
# The CSV columns are:
#   protein, position, resid_left, resid_center, resid_right,
#   mean_curvature, std_curvature, n_frames
#
# The .npy file stores the full (n_frames, n_positions) float32 curvature array.
#
# Prerequisites:
#   - Stage 1 (MSA) complete: data/msa/msa.done must list all proteins
#   - Boltz-2 synth predictions complete: results/boltz_synth/ must be populated
#   - conda env with gemmi + numpy active (npf-notebook), or pass --python
#
# Usage:
#   # Activate npf-notebook conda env first, then:
#   bash worflows/postprocessing/compute_menger_curvature.sh
#
#   # Or point to a specific Python interpreter:
#   bash worflows/postprocessing/compute_menger_curvature.sh --python /path/to/python
#
#   # Filter to a single protein:
#   bash worflows/postprocessing/compute_menger_curvature.sh --protein NPF1.1_Q8LPL2
#
#   # Dry-run (print what would be done without executing):
#   bash worflows/postprocessing/compute_menger_curvature.sh --dry-run

set -euo pipefail

# ── Locate repo root (script lives two levels below it) ───────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# ── Defaults ──────────────────────────────────────────────────────────────────
BOLTZ_OUT="results/boltz_synth"
OUT_DIR="results/menger_curvature_synth"
BOLTZ_DIR_OVERRIDE=""
MSA_SENTINEL="data/msa/msa.done"
COMPUTE_SCRIPT="scripts/compute_menger_curvature.py"
SPACING=2
PYTHON="${PYTHON:-python}"
DRY_RUN=0
FILTER_PROTEIN=""

# ── CLI ───────────────────────────────────────────────────────────────────────
usage() {
    echo "Usage: $0 [--dry-run] [--python PATH] [--protein NAME] [--spacing N] [--boltz-dir DIR] [--out-dir DIR]"
    echo ""
    echo "  --dry-run          Print what would be done without running anything."
    echo "  --python PATH      Python interpreter with gemmi + numpy installed."
    echo "                     Defaults to \$PYTHON env var or 'python'."
    echo "  --protein NAME     Process only this one protein (e.g. NPF1.1_Q8LPL2)."
    echo "  --spacing N        Triplet spacing for curvature calculation (default: 2)."
    echo "  --boltz-dir DIR    Override the default boltz output dir (results/boltz_synth)."
    echo "  --out-dir DIR      Override the default output dir (results/menger_curvature_synth)."
    echo "  -h, --help         Show this help."
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=1 ;;
        --python)    PYTHON="$2"; shift ;;
        --protein)   FILTER_PROTEIN="$2"; shift ;;
        --spacing)   SPACING="$2"; shift ;;
        -h|--help)   usage ;;
        *)           echo "[menger] Unknown argument: $1" >&2; usage ;;
    esac
    shift
done

# ── Validate prerequisites ────────────────────────────────────────────────────
if [[ ! -f "$MSA_SENTINEL" ]]; then
    echo "[menger] ERROR: MSA sentinel not found: $MSA_SENTINEL" >&2
    echo "         Run Stage 1 (MSA) first." >&2
    exit 1
fi

if [[ ! -f "$COMPUTE_SCRIPT" ]]; then
    echo "[menger] ERROR: compute script not found: $COMPUTE_SCRIPT" >&2
    exit 1
fi

if [[ ! -d "$BOLTZ_OUT" ]]; then
    echo "[menger] ERROR: boltz_synth directory not found: $BOLTZ_OUT" >&2
    echo "         Run Boltz-2 synth predictions first." >&2
    exit 1
fi

# ── Check Python has gemmi and numpy ─────────────────────────────────────────
if [[ "$DRY_RUN" -eq 0 ]]; then
    if ! "$PYTHON" -c "import gemmi, numpy" 2>/dev/null; then
        echo "[menger] ERROR: '$PYTHON' cannot import gemmi or numpy." >&2
        echo "         Activate the npf-notebook conda env or pass --python /path/to/python." >&2
        exit 1
    fi
fi

# ── Main loop ─────────────────────────────────────────────────────────────────
n_done=0
n_skip=0
n_fail=0

echo "[menger] Reading proteins from $MSA_SENTINEL"
echo "[menger] Boltz-2 input: $BOLTZ_OUT"
echo "[menger] Output root:   $OUT_DIR"
echo "[menger] Spacing:       $SPACING"
echo ""

while IFS= read -r protein || [[ -n "$protein" ]]; do
    [[ -z "$protein" ]] && continue

    # Optional single-protein filter
    if [[ -n "$FILTER_PROTEIN" && "$protein" != "$FILTER_PROTEIN" ]]; then
        continue
    fi

    protein_dir="$BOLTZ_OUT/$protein"
    if [[ ! -d "$protein_dir" ]]; then
        echo "[menger] SKIP $protein — no directory $protein_dir"
        (( n_skip++ )) || true
        continue
    fi

    # Require at least one CIF anywhere under this protein's boltz_synth dir
    n_cifs="$(find "$protein_dir" -name "*.cif" 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "$n_cifs" -eq 0 ]]; then
        echo "[menger] SKIP $protein — no CIF files found under $protein_dir"
        (( n_skip++ )) || true
        continue
    fi

    out_csv="$OUT_DIR/$protein/curvature.csv"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[dry-run] $protein"
        echo "          cifs:  $n_cifs CIF files"
        echo "          out:   $out_csv"
        continue
    fi

    mkdir -p "$OUT_DIR/$protein"

    if "$PYTHON" "$COMPUTE_SCRIPT" \
            --protein   "$protein"    \
            --boltz-dir "$BOLTZ_OUT"  \
            --output    "$out_csv"    \
            --spacing   "$SPACING"
    then
        (( n_done++ )) || true
    else
        echo "[menger] FAIL $protein" >&2
        (( n_fail++ )) || true
    fi

done < "$MSA_SENTINEL"

echo ""
echo "[menger] ── Summary ──────────────────────────────────"
echo "[menger]   Computed : $n_done"
echo "[menger]   Skipped  : $n_skip"
echo "[menger]   Failed   : $n_fail"
[[ "$n_fail" -gt 0 ]] && exit 1 || exit 0
