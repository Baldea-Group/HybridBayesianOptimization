# Multi-Scale Bayesian Optimization

A framework for simultaneous process and material co-design using bi-level Bayesian optimization. Instead of running BO over all variables (curse of dimensionality), this approach uses BO only for black-box material variables while solving the process-level optimization exactly via NLP.

## Setup

```bash
conda activate bo
```

## Quick Start

```bash
# Run full comparison on all benchmark problems
python run_comparison.py --problems all --acq ei --n_iter 50 --n_reps 5

# Run a single problem
python run_comparison.py --problems cobalt --n_iter 100 --n_init 12

# Sweep over hyperparameters
python run_comparison.py --acq ei --n_init_sweep 1 5 20 50 --n_iter 200 --n_reps 10 \
    --xi_sweep 0.001 0.01 0.05 0.1 0.2 0.5 1.0

# Validate test problems
python functions.py
```

## Repository Structure

```
.
├── functions.py          # BiLevelProblem dataclass and 13 test problems
├── solvers.py            # Solvers: NLP, DE, black-box BO, bi-level BO
├── run_comparison.py     # Experiment runner with results caching
├── run_single_job.py     # Single-job runner for TACC parallel execution
├── plotting.py           # Plotting functions (regret, timing, comparisons)
├── utils.py              # Shared paths and utility functions
│
├── tacc_launch.slurm                 # SLURM submission script for TACC
├── tacc_generate_launcher_commands.py # Generate SLURM job array commands
├── tacc_gather_results.py            # Merge parallel job results
│
├── Report/               # LaTeX manuscript and report figures
│   ├── main.tex
│   ├── figs/             # Publication figures (EPS/PDF)
│   └── generate_sfr_figures.py
├── Notebooks/            # Jupyter notebooks for exploration
├── figures/              # Quick-reference regret plots (PNG)
├── _MATLAB/              # Legacy MATLAB implementations
├── _archive/             # Archived results
└── presentation.mplstyle # Matplotlib style for presentations
```

## Method

**Bi-level structure:**

- **Outer loop** (BO over x_bb): `min J(x_wb*, pi)` where `pi = f_bb(x_bb)`
- **Inner loop** (NLP over x_wb): `x_wb* = argmin { J(x_wb, pi) | g(x_wb) <= 0 }`

The black-box function `f_bb` maps material variables to intermediate parameters. The white-box objective `J` and constraints `g` are solved exactly by an NLP solver at each BO iteration.

## Benchmark Problems

| Problem              | Dims (x_wb + x_bb) | Constraints | Type |
|----------------------|---------------------|-------------|------|
| Rastrigin            | 2 + 1               | 0           | min  |
| Rosen-Suzuki         | 2 + 2               | 3           | min  |
| Toy-Hydrology        | 1 + 1               | 2           | min  |
| CSTR                 | 3 + 2               | 3           | max  |
| Heat-Exchanger       | 2 + 1               | 2           | min  |
| PSA                  | 2 + 1               | 1           | min  |
| Batch-Reactor        | 2 + 1               | 1           | min  |
| Distillation         | 3 + 1               | 3           | min  |
| Evaporator           | 3 + 1               | 2           | min  |
| Membrane             | 2 + 1               | 2           | min  |
| Williams-Otto        | 3 + 2               | 2           | min  |
| Small-Feasible-Region   | 1 + 1            | 2           | min  |
| Small-Feasible-Region-2 | 1 + 1            | 2           | min  |

## Acquisition Functions

| Name     | Description                                    |
|----------|------------------------------------------------|
| `ei`     | Expected Improvement (default)                 |
| `pi`     | Probability of Improvement                     |
| `lcb`    | Lower Confidence Bound                         |
| `mwb2`   | Modified Watson-Barnes 2 (from COBALT paper)   |
| `thompson` | Thompson Sampling                            |

## Running on TACC

```bash
# Generate commands and SLURM script
python tacc_generate_launcher_commands.py --allocation DDM25011

# Submit to Lonestar6
sbatch tacc_launch.slurm

# After completion, merge results
python tacc_gather_results.py --indir results_parallel --outdir results
```