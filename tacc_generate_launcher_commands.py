"""
tacc_generate_launcher_commands.py - Generate SLURM job array for TACC Lonestar6.

Creates:
  - tacc_launcher_commands.txt  : one command per line, indexed by SLURM_ARRAY_TASK_ID
  - tacc_launch.slurm           : SLURM job array script for Lonestar6

All paths are relative to the repository root.
"""

import argparse
from pathlib import Path

PROBLEMS = [
    'Small-Feasible-Region', 'Small-Feasible-Region-2',
    'Rastrigin', 'Toy-Hydrology', 'Rosen-Suzuki',
    'CSTR', 'Heat-Exchanger', 'PSA', 'Batch-Reactor',
    'Distillation', 'Evaporator', 'Membrane', 'Williams-Otto',
]

N_INIT_VALUES = [5, 25, 50]
XI_VALUES = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]


def main():
    parser = argparse.ArgumentParser(
        description='Generate SLURM job array for Lonestar6')
    parser.add_argument('--n_iter', type=int, default=200)
    parser.add_argument('--n_reps', type=int, default=10)
    parser.add_argument('--acq', type=str, default='ei')
    parser.add_argument('--inner_solver', type=str, default='global')
    parser.add_argument('--outdir', type=str, default='results_parallel')
    parser.add_argument('--conda_env', type=str, default='bo')
    parser.add_argument('--allocation', type=str, default='MYALLOCATION',
                        help='TACC allocation name')
    parser.add_argument('--queue', type=str, default='normal')
    parser.add_argument('--runtime', type=str, default='02:00:00',
                        help='Wall time per task')
    parser.add_argument('--max_concurrent', type=int, default=128,
                        help='Max simultaneous array tasks')
    args = parser.parse_args()

    repo_root = Path(__file__).parent

    commands = []
    for problem in PROBLEMS:
        for n_init in N_INIT_VALUES:
            for xi in XI_VALUES:
                cmd = (f"python -u run_single_job.py "
                       f"--problem {problem} "
                       f"--n_init {n_init} --xi {xi} "
                       f"--n_iter {args.n_iter} --n_reps {args.n_reps} "
                       f"--acq {args.acq} --inner_solver {args.inner_solver} "
                       f"--outdir {args.outdir}")
                commands.append(cmd)

    cmd_file = repo_root / 'tacc_launcher_commands.txt'
    with open(cmd_file, 'w') as f:
        f.write('\n'.join(commands) + '\n')
    print(f"Wrote {len(commands)} commands to {cmd_file}")

    # Create output directories
    (repo_root / args.outdir).mkdir(parents=True, exist_ok=True)
    (repo_root / 'logs').mkdir(exist_ok=True)

    n_tasks = len(commands)
    max_idx = n_tasks - 1

    slurm_script = f"""#!/bin/bash
#SBATCH -J bo_sweep
#SBATCH -o logs/bo_%j.out
#SBATCH -e logs/bo_%j.err
#SBATCH -p {args.queue}
#SBATCH -N 1
#SBATCH -n {args.max_concurrent}
#SBATCH -t {args.runtime}
#SBATCH -A {args.allocation}

# cd to repo root (script lives at repo root)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate {args.conda_env}

# Launch all commands via pylauncher
module load pylauncher
python -c "import pylauncher; pylauncher.ClassicLauncher('tacc_launcher_commands.txt', cores=4)"
"""

    slurm_file = repo_root / 'tacc_launch.slurm'
    with open(slurm_file, 'w') as f:
        f.write(slurm_script)
    print(f"Wrote SLURM script to {slurm_file}")

    print(f"\nTotal tasks: {n_tasks}")
    print(f"Array range: 0-{max_idx}%{args.max_concurrent}")
    print(f"Wall time per task: {args.runtime}")

    print(f"\nUsage:")
    print(f"  Submit:               sbatch tacc_launch.slurm")
    print(f"  Check status:         sacct -j <JOBID> --format=JobID,State,Elapsed")
    print(f"  After completion:     python tacc_gather_results.py")
    print(f"\nLocal parallel run:     parallel -j 8 < tacc_launcher_commands.txt")


if __name__ == '__main__':
    main()
