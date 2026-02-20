#!/bin/bash

#SBATCH --job-name=f1tenth_dnn
#SBATCH --partition=gpu,volta-gpu,a100-gpu,l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8g
#SBATCH --time=2:00:00
#SBATCH --output=scripts/train/slurm_logs/%x_%A_%a.out
#SBATCH --error=scripts/train/slurm_logs/%x_%A_%a.err
#SBATCH --array=0-5
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pganguli@unc.edu

# ── Config array: matches SLURM_ARRAY_TASK_ID 0-5 ──
CONFIGS=(
    "scripts/train/config_heading_small.yaml"
    "scripts/train/config_heading_medium.yaml"
    "scripts/train/config_heading_large.yaml"
    "scripts/train/config_wall_small.yaml"
    "scripts/train/config_wall_medium.yaml"
    "scripts/train/config_wall_large.yaml"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "=== Task $SLURM_ARRAY_TASK_ID: $CONFIG ==="
echo "=== Node: $(hostname), GPU: $CUDA_VISIBLE_DEVICES ==="

# ── Environment setup ──
module purge
module load cuda  # adjust version to what's available: module avail cuda

# Activate venv (edit path if cloned elsewhere)
cd "$SLURM_SUBMIT_DIR" || exit 1
# shellcheck source=/dev/null
source .venv/bin/activate

# ── Run training ──
srun python scripts/train/train.py --config "$CONFIG"
