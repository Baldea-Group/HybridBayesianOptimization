#!/bin/bash
#SBATCH -J bo_sweep
#SBATCH -o logs/bo_%j.out
#SBATCH -e logs/bo_%j.err
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -t 10:00:00
#SBATCH -A DDM25011

# cd to repo root (script lives at repo root)
cd "/work/08940/joshuaeh/ls6/projects/HybridBayesianOptimization"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate bo

# Use pylauncher to run all commands across 32 cores
module load pylauncher
python3 -c "import pylauncher; pylauncher.ClassicLauncher('tacc_launcher_commands.txt')"
