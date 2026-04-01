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

## Paper Story (OCAR Framework)

### One-Sentence Story
When grey-box problems have separable black-box and white-box variables—natural in multi-scale process-material co-design—embedding an exact NLP solver inside the BO loop reduces surrogate dimensionality, satisfies constraints exactly, and achieves orders-of-magnitude better solutions.

### Opening (Big Problem)
Engineering design couples expensive black-box simulations (DFT, molecular dynamics) with well-understood analytical models (mass/energy balances). How do we integrate prior knowledge into data-driven optimization?

### Challenge (Gap)
- NLP solvers need gradients through the black box—infeasible when evaluations take hours
- Full-space BO wastes samples learning known equations and suffers curse of dimensionality
- Grey-box BO (COBALT, BOCF) surrogates intermediate outputs y and propagates uncertainty through white-box equations via moment approximations—more general but inexact
- **No method both separates variables and solves the white-box problem exactly**

### Action (Method + Evidence)
- Bilevel reformulation: BO over x^BB only, exact NLP inner solve for x^WB
- 13-problem benchmark suite (largest for separable grey-box BO), verified global optima
- 10,920 optimization runs (13 problems × 28 hyperparameter configs × 10 reps × 3 solvers)
- Vanilla BO (RBF kernel, EI) used deliberately—gains come from problem structure, not algorithmic novelty

### Resolution (Results)
- 14×–277,000× lower regret than black-box BO on all 13 problems
- Advantage grows with dimension (2D: 76–6,500×; 5D: up to 277,000×)
- Tight-constraint problems (Distillation, Heat-Exchanger): >100,000× improvement
- Comparable wall time for 12/13 problems
- Robust to hyperparameters: n_init ∈ {1, 50}, ξ ∈ {0.001, 1.0}
- Matches 20-restart NLP quality with ~200 evaluations vs. 900–3,000

### Story Tensions to Address in Revisions
1. **Simplicity tension**: Method is nearly obvious once you see separability—paper must argue why this hasn't been done systematically (position against COBALT/BOCF/bilevel BO literature)
2. **Synthetic benchmarks**: All black-box functions are closed-form surrogates, not real DFT/MD—acknowledged honestly but limits engineering motivation
3. **n_BB ≤ 2 ceiling**: All benchmarks have at most 2 black-box variables—scaling untested
4. **No head-to-head with COBALT/BOCF**: Direct empirical comparison on same problems is future work
5. **Missing Abstract**: Paper currently has no abstract section
