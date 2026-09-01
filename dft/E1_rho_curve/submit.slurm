#!/usr/bin/env bash
#SBATCH --job-name=pris-e1
#SBATCH --array=1-20%20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
# Adjust --partition/--account for the cluster before submitting.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
mkdir -p logs

# ---- cluster-specific settings -------------------------------------------------
module load vasp/6.3.0 2>/dev/null || true
export VASP_BIN="${VASP_BIN:-srun -n $SLURM_NTASKS vasp_std}"
export VASP_PP_PATH="${VASP_PP_PATH:-$HOME/POTCAR/PBE}"
# --------------------------------------------------------------------------------

TASK_DIR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" tasklist.txt)
[ -z "$TASK_DIR" ] && exit 0
echo "task $SLURM_ARRAY_TASK_ID -> $TASK_DIR on $(hostname)"
bash "$TASK_DIR/run.sh"
