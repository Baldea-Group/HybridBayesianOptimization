# CLAUDE.md

## Environment

Always use the `bo` conda environment for all Python execution in this project:
```bash
conda activate bo
```

`python run_comparison.py --acq ei --n_init_sweep 1 5 20 50 --n_iter 200 --n_reps 10 --xi_sweep 0.001 0.01 0.05 0.1 0.2 0.5 1.0`

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements a **Multi-Scale Bayesian Optimization Framework** for simultaneous process and material co-design. It compares three optimization approaches for bi-level problems where material-level decisions (x^m) couple with process-level decisions (x^p) through a black-box function.

The key insight: instead of running BO over all variables (curse of dimensionality), use BO only for material variables x^m while solving the process-level optimization exactly via NLP.

## Commands

### Run experiments comparing optimization methods
```bash
cd HybridBayesianOptimization/Examples
python run_comparison.py --problems cobalt --n-iter 50 --n-reps 5
python run_comparison.py --problems all --acq ei pi lcb  # specific acquisitions
python run_comparison.py --problems cstr --n-iter 100 --n-init 12
```

### Run single problem tests
```bash
cd HybridBayesianOptimization/Examples
python solvers.py  # runs demo on Rastrigin problem
python functions.py  # validates all test problems
```

## Architecture

### Core Module Structure (`HybridBayesianOptimization/Examples/`)

```
functions.py    - BiLevelProblem dataclass + test problems (Rastrigin, Rosen-Suzuki, Toy-Hydrology, CSTR)
solvers.py      - Three solvers: solve_blackbox_nlp, solve_blackbox_bo, solve_bilevel_bo
run_comparison.py - Experiment runner with plotting (5 figures: regret, timing, comparisons)
```

### Mathematical Formulation

**Bi-level structure:**
- Outer (BO over x^m): `min_{x^m} J(x^p*, π)` where `π = f^m(x^m)`
- Inner (NLP over x^p): `x^p* = argmin { J(x^p, π) | g(x^p) ≤ 0 }`

**Variable notation:**
- `x^p`: process variables (temperature, pressure, residence time)
- `x^m`: material/design variables (binding energy, activation shift)
- `π`: black-box outputs from molecular model `f^m(x^m)`

### BiLevelProblem Dataclass

Key attributes to understand when defining new problems:
- `n_xp`, `n_xm`, `n_pi`, `n_g`: dimensions
- `fm`: black-box function `π = f^m(x^m)`
- `J`: objective function `J(x^p, π) -> scalar`
- `g`: constraint function `g(x^p, π) -> vector` (≤ 0 feasible)
- `maximize`: True for maximization problems (CSTR), False for minimization

### Acquisition Functions (in solvers.py)

Available: `['ei', 'pi', 'lcb', 'mwb2', 'thompson']`
- EI: Expected Improvement (default)
- PI: Probability of Improvement
- LCB: Lower Confidence Bound
- mWB2: Modified Watson-Barnes 2 (from COBALT paper)
- Thompson: Thompson Sampling

### Test Problems

| Problem | Dims (x^p + x^m) | Constraints | Type |
|---------|------------------|-------------|------|
| Rastrigin | 2 + 1 | None | min |
| Rosen-Suzuki | 2 + 2 | 3 | min |
| Toy-Hydrology | 1 + 1 | 2 | min |
| CSTR | 3 + 2 | 3 | max |

## Key Implementation Details

### Constraint Handling
- Penalty method for black-box BO
- Inner NLP uses SLSQP with analytical gradients when available
- Feasibility tracked separately for regret computation

### GP Configuration
Uses sklearn's GaussianProcessRegressor with ARD RBF kernel:
```python
kernel = ConstantKernel() * RBF(length_scale_bounds=(1e-3, 1e3)) + WhiteKernel()
```

### Results Caching
`run_comparison.py` caches results to `plots/results_cache.pkl`. Delete this file to force re-run.
