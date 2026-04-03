"""
generate_launcher_commands.py - Generate SLURM job array for TACC Lonestar6.

Creates:
  - launcher_commands.txt  : one command per line, indexed by SLURM_ARRAY_TASK_ID
  - launch.slurm           : SLURM job array script for Lonestar6

All paths are relative to the repository root. The SLURM script cd's to
the repo root at runtime so that run_single_job.py and its imports resolve.
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

    tacc_dir = Path(__file__).parent
    repo_root = tacc_dir.parent

    # Generate command file — paths relative to repo root
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

    cmd_file = tacc_dir / 'launcher_commands.txt'
    with open(cmd_file, 'w') as f:
        f.write('\n'.join(commands) + '\n')
    print(f"Wrote {len(commands)} commands to {cmd_file}")

    n_tasks = len(commands)
    max_idx = n_tasks - 1

    # SLURM script uses SCRIPT_DIR to resolve paths at runtime
    slurm_script = f"""#!/bin/bash
#SBATCH -J bo_sweep
#SBATCH -o logs/bo_%A_%a.out
#SBATCH -e logs/bo_%A_%a.err
#SBATCH -p {args.queue}
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t {args.runtime}
#SBATCH -A {args.allocation}
#SBATCH --array=0-{max_idx}%{args.max_concurrent}

# Resolve the directory containing this script (works on any machine)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p {args.outdir}
mkdir -p logs

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate {args.conda_env}

# Read the command for this array task
CMD=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" launcher_commands.txt)

echo "Task $SLURM_ARRAY_TASK_ID: $CMD"
eval "$CMD"
"""

    slurm_file = script_dir / 'launch.slurm'
    with open(slurm_file, 'w') as f:
        f.write(slurm_script)
    print(f"Wrote SLURM script to {slurm_file}")

    print(f"\nTotal tasks: {n_tasks}")
    print(f"Array range: 0-{max_idx}%{args.max_concurrent}")
    print(f"Wall time per task: {args.runtime}")

    print(f"\nUsage:")
    print(f"  Submit all:           sbatch launch.slurm")
    print(f"  Submit subset:        sbatch --array=0-20 launch.slurm")
    print(f"  Rerun failed tasks:   sbatch --array=5,12,99 launch.slurm")
    print(f"  Check status:         sacct -j <JOBID> --format=JobID,State,Elapsed")
    print(f"  After completion:     python gather_results.py")
    print(f"\nLocal parallel run:     parallel -j 8 < launcher_commands.txt")


if __name__ == '__main__':
    main()
