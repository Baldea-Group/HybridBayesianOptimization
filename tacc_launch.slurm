#!/bin/bash
#SBATCH -J bo_sweep
#SBATCH -o logs/bo_%j.out
#SBATCH -e logs/bo_%j.err
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -t 36:00:00
#SBATCH -A DDM25011
#SBATCH --mail-type=ALL
#SBATCH --mail-user=joshua.hammond@utexas.edu

# cd to repo root (script lives at repo root)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate bo

# Launch all commands via pylauncher
module load pylauncher
python -c "import pylauncher; pylauncher.ClassicLauncher('tacc_launcher_commands.txt', cores=4)"
