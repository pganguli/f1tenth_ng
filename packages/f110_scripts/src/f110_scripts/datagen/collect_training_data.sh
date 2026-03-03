#!/usr/bin/env bash
# =============================================================================
# collect_training_data.sh
#
# Sweeps all F1 maps under data/maps/F1/ with multiple noise/lookahead
# configurations and collects training data via waypoint_datagen.
#
# Usage:
#   bash collect_training_data.sh [options]
#
# Options (override via env vars or edit the CONFIG section below):
#   MAX_STEPS          Steps per run  (default: 15000)
#   OUTPUT_DIR         Dataset output dir  (default: data/datasets)
#   LOG_DIR            Per-run log dir  (default: logs/datagen)
#   MAPS_DIR           Root of F1 maps  (default: data/maps/F1)
#   SKIP_MAPS          Space-separated map names to exclude
#   DRY_RUN            Set to 1 to print commands without running
# =============================================================================
set -euo pipefail

# Resolve repo root relative to this script's location
# (packages/f110_scripts/src/f110_scripts/datagen/ → 4 levels up)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MAPS_DIR="${MAPS_DIR:-$REPO_ROOT/data/maps/F1}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/data/datasets}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/datagen}"
MAX_STEPS="${MAX_STEPS:-15000}"
DRY_RUN="${DRY_RUN:-0}"

# Maps to skip (space-separated exact folder names)
SKIP_MAPS="${SKIP_MAPS:-}"

# ─── PARAMETER SWEEP ─────────────────────────────────────────────────────────
# Each entry is one configuration:
#   steering_noise  drift_prob  drift_magnitude  lookahead_min  lookahead_max
#
# Row 0 – clean reference run (minimal noise)
# Row 1 – moderate noise (default-ish)
# Row 2 – high noise / aggressive drift
# Row 3 – wide lookahead sweep, low noise
PARAM_CONFIGS=(
    "0.02  0.005 0.2  0.5 1.0"
    "0.05  0.010 0.3  0.6 1.2"
    "0.10  0.020 0.5  0.6 1.4"
    "0.03  0.008 0.25 0.8 1.6"
)

# ─── HELPERS ─────────────────────────────────────────────────────────────────
SCRIPT="python -m f110_scripts.datagen.waypoint_datagen"
PASS=0
FAIL=0
SKIP=0

log() { echo "[$(date '+%H:%M:%S')] $*"; }

should_skip() {
    local name="$1"
    for skip in $SKIP_MAPS; do
        [[ "$name" == "$skip" ]] && return 0
    done
    return 1
}

# ─── MAIN LOOP ───────────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Collect map entries: pairs of (map_path, waypoint_path)
declare -a MAP_ENTRIES
while IFS= read -r -d '' yaml_file; do
    folder_dir="$(dirname "$yaml_file")"
    folder_name="$(basename "$folder_dir")"
    stem="${yaml_file%.yaml}"   # strip .yaml → gives map path without ext

    # Derive centerline name: look for *_centerline.tsv in same folder
    centerline="$(find "$folder_dir" -maxdepth 1 -name '*_centerline.tsv' | head -1)"
    if [[ -z "$centerline" ]]; then
        log "WARN: No centerline found in $folder_dir – skipping"
        continue
    fi

    MAP_ENTRIES+=("$stem|$centerline|$folder_name")
done < <(find "$MAPS_DIR" -maxdepth 2 -name '*_map.yaml' -print0 | sort -z)

log "Found ${#MAP_ENTRIES[@]} maps in $MAPS_DIR"
log "Running ${#PARAM_CONFIGS[@]} parameter configs per map"
log "Total planned runs: $(( ${#MAP_ENTRIES[@]} * ${#PARAM_CONFIGS[@]} ))"
echo ""

cfg_idx=0
for entry in "${MAP_ENTRIES[@]}"; do
    IFS='|' read -r map_path waypoint_path map_name <<< "$entry"

    if should_skip "$map_name"; then
        log "SKIPPING $map_name (in SKIP_MAPS)"
        (( SKIP++ )) || true
        continue
    fi

    cfg_idx=0
    for cfg in "${PARAM_CONFIGS[@]}"; do
        read -r s_noise d_prob d_mag la_min la_max <<< "$cfg"
        (( cfg_idx++ )) || true

        cfg_tag="cfg${cfg_idx}"
        run_label="${map_name}_${cfg_tag}"
        log_file="${LOG_DIR}/${run_label}.log"

        cmd=(
            $SCRIPT
            --map          "$map_path"
            --map-ext      ".png"
            --waypoints    "$waypoint_path"
            --render-mode  "None"
            --max-steps    "$MAX_STEPS"
            --steering-noise  "$s_noise"
            --drift-prob      "$d_prob"
            --drift-magnitude "$d_mag"
            --lookahead-range "$la_min" "$la_max"
        )

        log "[$run_label] noise=$s_noise drift_prob=$d_prob drift_mag=$d_mag lookahead=[$la_min,$la_max]"

        if [[ "$DRY_RUN" == "1" ]]; then
            echo "  DRY-RUN: ${cmd[*]}"
            (( PASS++ )) || true
            continue
        fi

        if "${cmd[@]}" > "$log_file" 2>&1; then
            log "  ✓ done  → $(tail -1 "$log_file")"
            (( PASS++ )) || true
        else
            log "  ✗ FAILED – see $log_file"
            (( FAIL++ )) || true
        fi
    done
done

echo ""
log "═══════════════════════════════════════"
log "Sweep complete: passed=$PASS  failed=$FAIL  skipped=$SKIP"
log "Datasets saved to: $OUTPUT_DIR"
log "Logs saved to:     $LOG_DIR"
