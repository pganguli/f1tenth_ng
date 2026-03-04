#!/usr/bin/env bash
# slurm_batch.sh — interactive partition selector for train.sl
# Supports selecting multiple partitions (SLURM will use whichever has resources).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="$SCRIPT_DIR/train.sl"

# Collect available (up) partitions
mapfile -t PARTITIONS < <(sinfo -h -o "%P %a" | awk '$2=="up"{gsub(/\*$/,"",$1); print $1}' | sort -u)

if [[ ${#PARTITIONS[@]} -eq 0 ]]; then
    echo "No partitions currently available (sinfo returned nothing)." >&2
    exit 1
fi

echo "Available partitions:"
for i in "${!PARTITIONS[@]}"; do
    printf "  [%d] %s\n" "$((i+1))" "${PARTITIONS[$i]}"
done
echo ""
echo "Enter partition numbers separated by spaces to select multiple (e.g. 1 3)."
read -rp "Select partition(s) (1-${#PARTITIONS[@]}): " -a CHOICES

SELECTED=()
for CHOICE in "${CHOICES[@]}"; do
    if ! [[ "$CHOICE" =~ ^[0-9]+$ ]] || (( CHOICE < 1 || CHOICE > ${#PARTITIONS[@]} )); then
        echo "Invalid selection: $CHOICE" >&2
        exit 1
    fi
    SELECTED+=("${PARTITIONS[$((CHOICE-1))]}")
done

# Deduplicate and join with commas
PARTITION=$(printf '%s\n' "${SELECTED[@]}" | sort -u | paste -sd ',')

echo "Submitting with --partition=$PARTITION ..."
sbatch --partition="$PARTITION" "$SLURM_SCRIPT" "$@"
