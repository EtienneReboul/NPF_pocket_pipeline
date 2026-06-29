#!/usr/bin/env bash
# worflows/postprocessing/generate_tm_angle_cxc.sh
# =================================================
# Generates ChimeraX .cxc TM2/TM8 angle scripts for every protein × conformation
# that has completed Boltz-2 predictions (i.e. prediction.done exists).
#
# One representative CIF (alphabetically first) is used per protein × conformation.
# TM2 and TM8 are identified from the topology file (same as compute_tm_angle.py).
#
# Output: results/chimerax/{protein}/{conformation}/tm_angle.cxc
#
# Prerequisites:
#   - Stage 1 (MSA) complete: data/msa/msa.done must list all proteins
#   - Stage 2b (topology) complete: data/interpro/tm_topology_summary.json must exist
#   - Stage 5 (Boltz-2) complete: prediction.done files must exist
#   - conda env with gemmi active, or pass --python / set PYTHON env var
#
# Usage:
#   # Activate the tm_analysis conda env first, then:
#   bash worflows/postprocessing/generate_tm_angle_cxc.sh
#
#   # Or point to a specific Python interpreter:
#   bash worflows/postprocessing/generate_tm_angle_cxc.sh --python /path/to/python
#
#   # Filter to a single protein:
#   bash worflows/postprocessing/generate_tm_angle_cxc.sh --protein NPF1.1_Q9LYD5
#
#   # Dry-run (print what would be done without executing):
#   bash worflows/postprocessing/generate_tm_angle_cxc.sh --dry-run

set -euo pipefail

# ── Locate repo root (script lives two levels below it) ───────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# ── Defaults (mirror config.yaml) ─────────────────────────────────────────────
BOLTZ_OUT="results/boltz"
TOPOLOGY="data/interpro/tm_topology_summary.json"
MSA_SENTINEL="data/msa/msa.done"
OUT_DIR="results/chimerax"
VIZ_SCRIPT="scripts/visualize_tm_angle_chimerax.py"
PYTHON="${PYTHON:-python}"
DRY_RUN=0
FILTER_PROTEIN=""

# ── CLI ───────────────────────────────────────────────────────────────────────
usage() {
    echo "Usage: $0 [--dry-run] [--python PATH] [--protein NAME]"
    echo ""
    echo "  --dry-run          Print what would be done without running anything."
    echo "  --python PATH      Python interpreter with gemmi installed."
    echo "                     Defaults to \$PYTHON env var or 'python'."
    echo "  --protein NAME     Process only this one protein (e.g. NPF1.1_Q9LYD5)."
    echo "  -h, --help         Show this help."
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)          DRY_RUN=1 ;;
        --python)           PYTHON="$2"; shift ;;
        --protein)          FILTER_PROTEIN="$2"; shift ;;
        -h|--help)          usage ;;
        *)                  echo "[tm_angle_cxc] Unknown argument: $1" >&2; usage ;;
    esac
    shift
done

# ── Validate prerequisites ────────────────────────────────────────────────────
if [[ ! -f "$MSA_SENTINEL" ]]; then
    echo "[tm_angle_cxc] ERROR: MSA sentinel not found: $MSA_SENTINEL" >&2
    echo "               Run Stage 1 (MSA) first." >&2
    exit 1
fi

if [[ ! -f "$TOPOLOGY" ]]; then
    echo "[tm_angle_cxc] ERROR: topology file not found: $TOPOLOGY" >&2
    echo "               Run Stage 2b (DeepTMHMM topology) first." >&2
    exit 1
fi

if [[ ! -f "$VIZ_SCRIPT" ]]; then
    echo "[tm_angle_cxc] ERROR: visualization script not found: $VIZ_SCRIPT" >&2
    exit 1
fi

# ── Check Python has gemmi ────────────────────────────────────────────────────
if [[ "$DRY_RUN" -eq 0 ]]; then
    if ! "$PYTHON" -c "import gemmi" 2>/dev/null; then
        echo "[tm_angle_cxc] ERROR: '$PYTHON' cannot import gemmi." >&2
        echo "               Activate the tm_analysis conda env or pass --python /path/to/python." >&2
        exit 1
    fi
fi

# ── Main loop ─────────────────────────────────────────────────────────────────
n_done=0
n_skip=0
n_fail=0

echo "[tm_angle_cxc] Reading proteins from $MSA_SENTINEL"

while IFS= read -r protein || [[ -n "$protein" ]]; do
    [[ -z "$protein" ]] && continue

    # Optional single-protein filter
    if [[ -n "$FILTER_PROTEIN" && "$protein" != "$FILTER_PROTEIN" ]]; then
        continue
    fi

    protein_dir="$BOLTZ_OUT/$protein"
    if [[ ! -d "$protein_dir" ]]; then
        echo "[tm_angle_cxc] SKIP $protein — no boltz output directory"
        (( n_skip++ )) || true
        continue
    fi

    # Iterate over every conformation subdirectory
    for conf_dir in "$protein_dir"/*/; do
        [[ -d "$conf_dir" ]] || continue

        # Strip trailing slash so paths below don't get double slashes
        conf_dir="${conf_dir%/}"
        conformation="$(basename "$conf_dir")"

        # Skip combos where Boltz-2 has not completed
        if [[ ! -f "$conf_dir/prediction.done" ]]; then
            continue
        fi

        # Find the first CIF (same glob as discover_cifs in the postprocessing Snakefile)
        cif="$(find "$conf_dir/boltz_out" -name "*.cif" 2>/dev/null | sort | head -1)"
        if [[ -z "$cif" ]]; then
            echo "[tm_angle_cxc] SKIP $protein/$conformation — no CIF found under $conf_dir/boltz_out"
            (( n_skip++ )) || true
            continue
        fi

        out="$OUT_DIR/$protein/$conformation/tm_angle.cxc"

        if [[ "$DRY_RUN" -eq 1 ]]; then
            echo "[dry-run] $protein/$conformation"
            echo "          cif: $cif"
            echo "          out: $out"
            continue
        fi

        mkdir -p "$(dirname "$out")"

        if "$PYTHON" "$VIZ_SCRIPT" \
                --cif      "$cif"      \
                --topology "$TOPOLOGY" \
                --protein  "$protein"  \
                --output   "$out"
        then
            (( n_done++ )) || true
        else
            echo "[tm_angle_cxc] FAIL $protein/$conformation" >&2
            (( n_fail++ )) || true
        fi
    done

done < "$MSA_SENTINEL"

echo ""
echo "[tm_angle_cxc] ── Summary ──────────────────────────────────"
echo "[tm_angle_cxc]   Generated : $n_done"
echo "[tm_angle_cxc]   Skipped   : $n_skip"
echo "[tm_angle_cxc]   Failed    : $n_fail"
[[ "$n_fail" -gt 0 ]] && exit 1 || exit 0
