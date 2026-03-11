#!/usr/bin/env bash
# sbatch_nn.sh — interactive partition selector for train_nn.sl
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="$SCRIPT_DIR/train_nn.sl"

mapfile -t PARTITIONS < <(sinfo -h -o "%P %a" | awk '$2=="up"{gsub(/\*$/,"",$1); print $1}' | sort -u)

if [[ ${#PARTITIONS[@]} -eq 0 ]]; then
    echo "No partitions available." >&2
    exit 1
fi

echo "Available partitions:"
for i in "${!PARTITIONS[@]}"; do
    printf "  [%d] %s\n" "$((i+1))" "${PARTITIONS[$i]}"
done

read -rp "Select partition (1-${#PARTITIONS[@]}): " CHOICE
if ! [[ "$CHOICE" =~ ^[0-9]+$ ]] || (( CHOICE < 1 || CHOICE > ${#PARTITIONS[@]} )); then
    echo "Invalid selection: $CHOICE" >&2
    exit 1
fi

PARTITION="${PARTITIONS[$((CHOICE-1))]}"

DOMAIN=$(hostname -d)
[[ "$DOMAIN" == *.*.* ]] && DOMAIN="${DOMAIN#*.}"
EMAIL="${USER}@${DOMAIN}"

echo "Submitting with --partition=$PARTITION --mail-user=$EMAIL ..."
sbatch --partition="$PARTITION" --mail-user="$EMAIL" "$SLURM_SCRIPT" "$@"
