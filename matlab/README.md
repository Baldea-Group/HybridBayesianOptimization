# MATLAB -- Hybrid Bayesian Optimization for CSTR Codesign

MATLAB implementation of the multiscale Bayesian optimization framework for a continuous stirred-tank reactor (CSTR) case study with consecutive reactions (A -> B -> C).

## Files

| File | Description |
|------|-------------|
| `BO_full_new_LHS_only_EI_only.m` | Full 5D Bayesian optimization over all process and catalyst variables simultaneously |
| `BO_hybrid_new_LHS_only_EI_only.m` | Hybrid (bi-level) approach: BO over catalyst variables (Eb, dEa), fmincon inner solve for process variables (T, P, tau) |

## Requirements

- MATLAB R2020a or later
- Statistics and Machine Learning Toolbox (for `fitrgp`)
- Optimization Toolbox (for `fmincon`, hybrid version only)

## Design Variables

| Variable | Symbol | Description | Bounds |
|----------|--------|-------------|--------|
| Temperature | T | Reactor temperature | process |
| Pressure | P | Reactor pressure | process |
| Residence time | tau | Reactor residence time | process |
| Binding energy | Eb | Catalyst binding energy | catalyst/material |
| Activation barrier shift | dEa | Catalyst activation energy shift | catalyst/material |

## Citation

If you use this code, please cite:

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
