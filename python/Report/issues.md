# Statistical Claims Issues: main.tex vs. Notebook Data

The paper (`Report/main.tex`) was written against an older dataset with **13 problems** and `n_init in {1, 5, 20, 50}`. The notebook (`Notebooks/analyze_results.ipynb`) contains updated results with **12 problems** (Williams-Otto missing from `results_parallel/`) and `n_init in {5, 25, 50}`. All numerical claims, tables, and summary statistics in the paper must be updated to match the notebook's computed values.

## Key Discrepancies Found

| Aspect | Paper (old) | Notebook (new) |
|--------|------------|----------------|
| Problem count | 13 | 12 (no Williams-Otto) |
| n_init values | {1, 5, 20, 50} | {5, 25, 50} |
| Table n_init | n_init=50 | n_init=5 |
| Regret ratio range | 14x--277,000x | 43x--133,406,150x |
| Total BO runs | 7,280 | 5,040 |
| Total NLP runs | 130 | 120 |
| Total runs | 7,410 | 5,160 |
| Configs per problem | 28 (4x7) | 21 (3x7) |
| Total configs | 364 | 252 |
| Wall time comparable | "12 of 13" | 2 of 12 (within 2x) |
| Dimension scaling | "grows with dimension" | Spearman rho=0.079, p=0.81 (not significant) |
| Wilcoxon p-value | p < 0.01 | p_adj = 0.023 (< 0.05 after Holm correction) |

---

## Changes by Paper Section

### 1. Abstract (line 72)

**Old:** "On a suite of 13 benchmark problems bilevel BO achieves 14x--277,000x lower regret than full-space BO across 7,410 independent optimization runs, with comparable wall clock time for 12 of 13 problems. The advantage grows with dimension---up to 277,000x on 5D problems---and is robust to initialization set size and exploration parameters."

**New:** Update to 12 problems, 43x--133,000,000x, 5,160 runs. Replace "comparable wall clock time for 12 of 13" with honest statement about moderate overhead (typically 2--3x). Remove "grows with dimension" claim; replace with "substantial across all dimensions tested."

### 2. Introduction -- Contributions (line 91--95)

- Line 93: "13 separable grey-box problems" -> "12"
- Line 94: "7,410 independent optimization runs (7,280 BO + 130 NLP)" -> "5,160 independent optimization runs (5,040 BO + 120 NLP)"
- Line 94: "14x--277,000x" -> "43x--133,000,000x"
- Line 94: "13 problems" -> "12 problems"

### 3. Related Work -- Constraint Handling (line 132)

- ">100,000x lower regret" for Distillation and Heat-Exchanger: still true (97,174x and 133,406,150x). **No change needed.**

### 4. Benchmark Suite Section (lines 429--432)

- Line 431: "13 problems" -> "12 problems"
- Line 431: "10,920-run" -> "5,160-run"
- Line 431: "2--5 total variables, 0--3 constraints" -> still valid. **No change.**

### 5. Table 2 -- Benchmark Summary (lines 436--459)

- Remove Williams-Otto row (line 457)
- Update caption: "13-problem" -> "12-problem"

### 6. Experimental Setup (lines 477--493)

- Line 489: "n_init in {1, 5, 20, 50}" -> "{5, 25, 50}"
- Line 489: "7,410 independent runs across all 13 problems" -> "5,160 independent runs across all 12 problems"
- Line 489: Also update the hyperparameter details reference text.

### 7. SFR Convergence Text (line 527)

- "76x improvement" -> "113x improvement" (SFR-1 ratio)
- "42,743x improvement" -> "21,951x improvement" (SFR-2 ratio)

### 8. Table 3 -- Final Regret (lines 542--566)

Complete replacement with notebook data. **Remove Williams-Otto row.** Change caption from "n_init=50" to "n_init=5". New values (from stats_per_problem.csv at n_init=5, best xi):

| Problem | dim | NLP | BB-BO | Bi-BO | BB/Bi |
|---------|-----|-----|-------|-------|-------|
| SFR-1 | 2 | 0.2026 | 1.6043 | 0.0142 | 113x |
| SFR-2 | 2 | 0.0546 | 0.1183 | 0.0000 | 21,951x |
| Rastrigin | 3 | 3.3829 | 8.6477 | 0.0000 | 69,841,797x |
| Toy-Hydrology | 2 | 0.0000 | 0.0719 | 0.0000 | 20,000x |
| Rosen-Suzuki | 4 | 0.0000 | 7.0756 | 0.1649 | 43x |
| CSTR | 5 | 0.0000 | 3.1098 | 0.0042 | 733x |
| Heat-Exchanger | 5 | 0.0000 | 127.8772 | 0.0000 | 133,406,150x |
| PSA | 5 | 0.0000 | 0.4681 | 0.0016 | 286x |
| Batch-Reactor | 5 | 0.0000 | 1.9763 | 0.0035 | 571x |
| Distillation | 5 | 1.8786 | 3457.8397 | 0.0356 | 97,174x |
| Evaporator | 5 | 0.0000 | 1.0067 | 0.0099 | 101x |
| Membrane | 5 | 0.0000 | 9.6885 | 0.0001 | 106,740x |

### 9. Main Results Text (line 568)

- "14x--277,000x" -> "43x--133,000,000x"
- "13 problems" -> "12 problems"
- "Heat-Exchanger: 277,000x, Distillation: 173,000x" -> "Heat-Exchanger: 133,000,000x, Rastrigin: 70,000,000x"
- "Rosen-Suzuki: 14x, Evaporator: 19x" -> "Rosen-Suzuki: 43x, Evaporator: 101x"
- Update the narrative about where largest/smallest gains occur
- "13 pairwise comparisons" -> "12 pairwise comparisons"
- "p < 0.01" -> "p_adj < 0.05" (Holm-corrected Wilcoxon)

### 10. BH Baseline Text (line 570)

- "BH baseline fails on two problems: Rastrigin... and SFR-2" -> verify with new NLP data. NLP regret: Rastrigin=3.38 (fails), SFR-2=0.055 (moderate), Distillation=1.88 (moderate). Update: "fails on Rastrigin and performs poorly on Distillation"
- "remaining 11 problems" -> update count
- "~250 black-box evaluations" -> "~205 black-box evaluations" (n_init=5+200)

### 11. Convergence Speed (line 574)

- Remove Williams-Otto reference ("bilevel BO converges in 9 iterations")
- "1.5x--22x faster" -> verify or soften claim. Flag for notebook verification.

### 12. Scaling and Robustness (lines 579--601)

- Line 579: "364 configurations" -> "252 configurations"
- Line 589: "n_init in {1, 5, 20, 50}" -> "{5, 25, 50}"
- Line 591: "even n_init=1---a single initial sample---performs comparably to n_init=50" -> "even n_init=5 performs comparably to n_init=50"
- Line 597 caption: "n_init in {1, 5, 20, 50}" -> "{5, 25, 50}"

### 13. Computational Cost (lines 603--605)

**Major rewrite needed.** Old: "Both BO methods require ~20-30s wall time for 200 iterations on 12 of 13 problems... Williams-Otto (111s vs. 26s)."

New: BB-BO takes ~38-45s, Bi-BO takes ~68-183s (excluding SFR-1 outlier at 965s). Bi-BO is typically 2-5x slower due to inner BH solves. Remove Williams-Otto reference. State that the wall-time overhead is modest relative to the orders-of-magnitude regret improvement.

### 14. Discussion Section (lines 610--668)

- Line 612: "14x--277,000x" -> "43x--133,000,000x"
- Line 612: "13 problems" -> "12 problems"
- Line 615: "12 constrained problems" -> "11 constrained problems"
- Line 615: Feasibility details ("8 of 12") -> "8 of 11" or verify from data. Flag for verification.
- Line 625: "Rastrigin achieves a 57,717x improvement (at n_init=50, xi=0.2)" -> "69,841,797x improvement (at n_init=5, best xi)"
- Line 626: "Evaporator and Batch-Reactor... smallest regret ratios (6x and 41x)" -> "Rosen-Suzuki (43x) and Evaporator (101x)"
- Line 626: "Heat-Exchanger... (277,000x)" -> "(133,000,000x)"
- Line 637: "5 of 13 problems" -> update count (remove Williams-Otto)
- Line 637: "13 problems" -> "12 problems"
- Line 657: "14x improvement on Rosen-Suzuki" -> "43x"

### 15. Conclusion (lines 667--669)

- "14x--277,000x" -> "43x--133,000,000x"
- "13 benchmark problems" -> "12"
- "7,410 independent optimization runs" -> "5,160"
- Remove "grows with dimension" if present.

### 16. Appendix -- Williams-Otto (lines 1162--1230)

Keep the problem definition in the appendix for completeness (it's still a valid test problem). But it should not appear in Table 2 or any results claims.

### 17. Appendix -- Hyperparameter Details (lines ~1350)

- n_init=5 is the default (was n_init=50 in old paper)
- "205 evaluations" still correct for n_init=5 + 200 iterations

### 18. Convergence Grid Caption (line 537)

- "11 benchmark problems" -> "10 benchmark problems" (13-2 SFR = 11 old; now 12-2 SFR = 10)

### 19. Global "13" -> "12" references

Search and replace all instances of "13 problems" / "13 benchmark" / "13-problem" throughout the paper.

---

## File to Modify

- `Report/main.tex` (lines 1--1400+)

## Data Sources (read-only)

- `Notebooks/stats_per_problem.csv` -- per-problem regret ratios, p-values, effect sizes
- `Notebooks/wall_time_comparison.csv` -- wall time data
- `Notebooks/sensitivity_pivot.csv` -- hyperparameter sensitivity (geometric mean ratios)
- Notebook cell 10 output -- Friedman test results
- Notebook cell 13 output -- LaTeX-ready table
- Notebook cell 30 output -- automated claim validation

## Verification

After all edits:
1. Search for remaining "13" references that should be "12"
2. Search for "277" (old max ratio) references
3. Search for "7,410" or "7{,}410" (old run count)
4. Search for "10,920" or "10{,}920" (old total count)
5. Search for "Williams" references outside the appendix problem definition
6. Search for n_init values "\\{1," to catch old hyperparameter sets
7. Verify all ratio values in text match Table 3
8. Grep for "14$\times$" to catch old minimum ratio
