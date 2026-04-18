# Hybrid Bayesian Optimization for Multiscale Process-Material Co-Design

A bi-level Bayesian optimization framework for simultaneous process and material co-design. Instead of running BO over all variables (curse of dimensionality), the framework uses BO only for black-box material/catalyst variables while solving the process-level optimization exactly via NLP or local solvers.

**Key idea:** When grey-box problems have separable black-box and white-box variables -- natural in multi-scale process-material co-design -- embedding an exact solver inside the BO loop reduces surrogate dimensionality, satisfies constraints exactly, and achieves orders-of-magnitude better solutions.

## Repository Structure

```
.
|-- matlab/     # MATLAB implementation (CSTR case study)
|-- python/     # Python benchmark suite (13 problems, extended framework)
```

### `matlab/`

Original MATLAB implementation for a continuous stirred-tank reactor (CSTR) with consecutive reactions. Compares full 5D Bayesian optimization against the hybrid (bi-level) approach. See [matlab/README.md](matlab/README.md) for details.

### `python/`

Extended Python implementation with a 13-problem benchmark suite, multiple acquisition functions (EI, PI, LCB, mWB2, Thompson), and comprehensive experimental infrastructure including TACC parallel execution. See [python/README.md](python/README.md) for details.

## Citations

### Published Paper (MATLAB code)

> Baldea, M. (2026). A multiscale Bayesian optimization framework for process and material codesign. *AIChE Journal*, 81. https://doi.org/10.1002/aic.70228

```bibtex
@article{Baldea2026,
  author  = {Baldea, Michael},
  title   = {A Multiscale {B}ayesian Optimization Framework for Process and Material Codesign},
  journal = {AIChE Journal},
  year    = {2026},
  volume  = {81},
  doi     = {10.1002/aic.70228}
}
```

### Extended Benchmark Paper (Python code)

> Hammond, J. E., Soderstrom, T. A., Korgel, B. A., & Baldea, M. (2026). [Title -- to be updated upon publication]. *Preprint*.

```bibtex
@article{Hammond2026,
  author  = {Hammond, Joshua E. and Soderstrom, Tyler A. and Korgel, Brian A. and Baldea, Michael},
  title   = {[Title -- to be updated upon publication]},
  journal = {[Journal -- to be updated]},
  year    = {2026},
  note    = {Preprint}
}
```
