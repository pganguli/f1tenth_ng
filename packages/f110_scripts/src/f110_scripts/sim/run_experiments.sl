#!/bin/bash
# f1tenth experiment runs — Slurm job

#SBATCH --job-name=f1tenth_exp 
#SBATCH -p volta-gpu
#SBATCH --qos=hp_volta_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=8:00:00
#SBATCH --output=packages/f110_scripts/src/f110_scripts/sim/slurm_logs/%x_%j.out
#SBATCH --error=packages/f110_scripts/src/f110_scripts/sim/slurm_logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

# ── Info ──
echo "=== Job ID: $SLURM_JOB_ID ==="
echo "=== Node: $(hostname) ==="
echo "=== Working dir: $SLURM_SUBMIT_DIR ==="

# ── Environment setup ──
if command -v module &>/dev/null; then
    module purge
    module load python/3.12
fi

cd "$SLURM_SUBMIT_DIR" || exit 1
echo "=== Working directory ==="
pwd

# Activate your virtual environment
source .venv/bin/activate
# export PYTHONPATH=$SLURM_SUBMIT_DIR/packages/f110_planning/src:$SLURM_SUBMIT_DIR/packages/f110_scripts/src:$PYTHONPATH

echo "PYTHONPATH: $PYTHONPATH"

# ── Run experiment script ──
echo "=== Starting experiments ==="
srun bash packages/f110_scripts/src/f110_scripts/sim/run_experiments.sh

echo "=== Finished ==="
