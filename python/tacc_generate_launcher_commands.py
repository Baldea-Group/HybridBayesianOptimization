"""
tacc_generate_launcher_commands.py - Generate pylauncher command file for TACC.

Creates:
  - tacc_launcher_commands.txt  : one command per line for pylauncher

The SLURM script (tacc_launch.slurm) is maintained separately.
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
    parser.add_argument('--outdir', type=str, default='results_parallel')
    args = parser.parse_args()

    repo_root = Path(__file__).parent

    commands = []
    for problem in PROBLEMS:
        for n_init in N_INIT_VALUES:
            for xi in XI_VALUES:
                for rep in range(args.n_reps):
                    cmd = (f"python -u run_single_job.py "
                           f"--problem {problem} "
                           f"--n_init {n_init} --xi {xi} "
                           f"--rep {rep} "
                           f"--n_iter {args.n_iter} "
                           f"--acq {args.acq} "
                           f"--outdir {args.outdir}")
                    commands.append(cmd)

    cmd_file = repo_root / 'tacc_launcher_commands.txt'
    with open(cmd_file, 'w') as f:
        f.write('\n'.join(commands) + '\n')
    print(f"Wrote {len(commands)} commands to {cmd_file}")

    # Create output directory
    (repo_root / args.outdir).mkdir(parents=True, exist_ok=True)

    print(f"\nTotal tasks: {len(commands)}")
    print(f"\nUsage:")
    print(f"  Submit:               sbatch tacc_launch.slurm")
    print(f"  Local parallel run:   parallel -j 8 < tacc_launcher_commands.txt")


if __name__ == '__main__':
    main()
