#!/bin/bash
#SBATCH -J bo_sweep
#SBATCH -o logs/bo_%j.out
#SBATCH -e logs/bo_%j.err
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -t 10:00:00
#SBATCH -A DDM25011

# Resolve the directory containing this script (works on any machine)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p results_parallel
mkdir -p logs

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate bo

# Use pylauncher to run all 273 commands across 32 cores
module load pylauncher
python $TACC_PYLAUNCHER/launch.py launcher_commands.txt
