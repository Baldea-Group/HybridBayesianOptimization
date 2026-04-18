# Plan: Update `main.tex` with Current Notebook Results

## Context

The paper (`Report/main.tex`) contains results from an older experimental campaign (n_init in {1, 5, 20, 50}, 7,410 runs). The notebook (`Notebooks/analyze_results.ipynb`) now has results from a new campaign (n_init in {5, 25, 50}, 2,730 files, 4 methods: NLP, DE/BH, BB-BO, Bi-BO). Nearly every hardcoded number, figure, and table in the paper is stale. Additionally, the paper should be restructured to include 4-way comparison (NLP, BH, BB-BO, Bi-BO) as first-class baselines, new figures from the notebook should be added, and unsupported claims (dimensionality scaling, n_init=1) should be removed.

---

## Phase 1: Run the Notebook to Get Exact Numbers

Before editing the LaTeX, run the notebook to capture all current output values.

- **Action**: Execute the notebook end-to-end (`conda run -n bo jupyter nbconvert --execute`) and capture all printed statistics
- **Key outputs needed**:
  - Per-problem regret table (cell 13): NLP, BH, BB-BO, Bi-BO regret + ratios + p-values
  - Geometric mean ratio (cell 9): currently ~1034x
  - Friedman test results (cell 10): Chi-sq, p-value, Nemenyi pairwise
  - Wall time comparison (cell 11): per-problem times, ratio
  - NLP comparison (cell 15): eval counts, quality assessment
  - Sensitivity pivot (cell 14): geometric mean ratios by (n_init, xi)
  - Convergence speed (cell 33): iterations to threshold
  - Claims validation (cell 41): which claims hold

---

## Phase 2: Add Missing Figures to Notebook

The notebook currently does NOT generate 4 figures the paper references. Add cells to regenerate them with current data:

1. **`feasibility_rates.eps/.pdf`** — bar chart of feasible evaluation fraction per problem for BB-BO vs Bi-BO
   - Data source: stored search histories in results pkl files
   - Reference: old figure showed 6-72% wasted budget for BB-BO

2. **`sample_crossover.eps/.pdf`** — iterations for Bi-BO to match BB-BO final regret
   - Data source: regret curves in results pkl files
   - Reference: old figure showed crossover point per problem

3. **`dim_vs_constraint_scatter.eps/.pdf`** — scatter of dimension vs feasible fraction, colored by regret ratio
   - Data source: problem metadata + regret ratios
   - Note: the notebook has `dimensionality_scaling` which may be similar but different

4. **`sfr_convergence.eps/.pdf`** — convergence curves for SFR-1 and SFR-2
   - Data source: regret curves for these two problems
   - Note: could be extracted from the convergence grid or made standalone

---

## Phase 3: Update Experimental Setup (lines 475-493)

| Item | Old | New |
|------|-----|-----|
| n_init values | {1, 5, 20, 50} | {5, 25, 50} |
| Total runs | 7,410 (7,280 BO + 130 NLP) | Recalculate: 13 x 3 x 7 x 10 x 4 methods = ? |
| Total configs | 364 | 273 (13 x 3 x 7) |

Files: `Report/main.tex` lines ~489, 431, 94

---

## Phase 4: Update Table 2 — Final Regret (lines 542-566)

**Expand to 4-way comparison**: Add BH column alongside NLP, BB-BO, Bi-BO.

- Replace all 13 rows with notebook cell 13 values (add BH/DE column)
- Update caption: `n_init=50` -> `n_init=5` (or whichever the notebook uses)
- Add p-value column and significance stars
- Add BH regret values from notebook data
- Column order: Problem | dim | NLP | BH | BB-BO | Bi-BO | BB/Bi ratio | p_adj

---

## Phase 5: Update Results Prose (lines 568-605)

### 5a. Main results paragraph (line 568)
- Old range: 14x-277,000x → New range from notebook (6.9x-143,629,503x or similar)
- Old largest gains: Heat-Exchanger 277,000x, Distillation 173,000x → update
- Old smallest gains: Rosen-Suzuki 14x, Evaporator 19x → update with new weakest
- **Add NLP and BH comparison prose**: how Bi-BO compares to these baselines
- Update Wilcoxon sentence with Holm-Bonferroni correction detail

### 5b. BH baseline paragraph (line 570)
- Update which problems BH fails on (was Rastrigin and SFR-2)
- Add evaluation count comparison (Bi-BO uses ~205 evals vs BH's ~1817)
- Integrate Friedman test result (NLP vs Bi-BO NOT significant = Bi-BO approaches NLP quality)

### 5c. Convergence speed (lines 572-574)
- Update 1.5x-22x range with new data
- Update Williams-Otto specific claim (9 iterations)

### 5d. Scatter plot (line 579)
- Update config count: 364 → 273
- Verify "every point below diagonal" claim still holds (SFR-1 has 6.9x worst-case ratio)

### 5e. n_init robustness (line 591)
- **Remove n_init=1 narrative entirely** (no longer tested)
- Update to {5, 25, 50}; rewrite without the "single initial sample" talking point

### 5f. xi sensitivity (line 601)
- Update best xi values per problem from notebook cell 5

### 5g. Computational cost (lines 603-605)
- Old: ~20-30s for 12/13 → New: 11/13 within 2x ratio
- Old: Williams-Otto 111s vs 26s → New: 2956s vs 277s (much larger gap)
- **Add BH wall time comparison**

---

## Phase 6: Update Discussion (lines 610-660)

### 6a. Opening sentence (line 612)
- Update regret range, add NLP/BH comparison

### 6b. Feasibility paragraph (lines 614-623)
- Update with regenerated figure data (Phase 2)
- Add BH feasibility comparison

### 6c. **DELETE dimensionality decomposition paragraph** (lines 625-634)
- Spearman rho=0.224, p=0.46 — claim unsupported
- Remove Figure 9 reference (dim_vs_constraint_scatter) OR repurpose as supplementary
- Note: we may still want the regenerated figure in supplementary but without the "grows with dimension" claim

### 6d. Sample efficiency paragraph (lines 636-644)
- Update with regenerated figure data (Phase 2)
- Add NLP eval-count comparison (205 vs ~1817)

### 6e. Limitations (line 657)
- Old: "modest 14x on Rosen-Suzuki" → Rosen-Suzuki is now 193x, no longer weak
- Identify new weakest case (SFR-1 at 6.9x) and explain why
- Update Williams-Otto timing (2956s vs 277s instead of 111s vs 26s)
- Remove reference to feasibility_rates figure if paragraph was rewritten

---

## Phase 7: Update Abstract (lines 71-73) and Conclusion (lines 665-669)

### Abstract
- Regret range: 14x-277,000x → new range
- Run count: 7,410 → new count
- Wall time: 12/13 → 11/13
- **Remove "advantage grows with dimension" sentence**
- **Add NLP/BH comparison claim** (e.g., "matches multistart NLP quality with 9x fewer evaluations")

### Conclusion
- Same number updates
- Remove dimensionality scaling claim
- Update run count
- Remove "grows with dimension" from line 667

### Contribution #3 (line 94)
- Update run count and regret range
- Add mention of NLP/BH comparison

---

## Phase 8: Add New Figures from Notebook

Add these figures generated by the notebook that strengthen the paper:

1. **`wall_time_comparison.eps/.pdf`** (cell 28) — bar chart of wall times per problem
   - Add after computational cost paragraph (line 605)

2. **`regret_boxplots.eps/.pdf`** (cell 27) — distribution of final regrets
   - Add near Table 2 to show variability across repetitions

3. **`significance_matrix.eps/.pdf`** (cell 29) — pairwise statistical test matrix
   - Add in results section to support 4-way comparison

4. **`convergence_speed.eps/.pdf`** (cell 33) — convergence rate comparison
   - Add in convergence speed subsection (line 572)

5. **`fbb_evaluations.eps/.pdf`** (cell 34) — black-box evaluation counts
   - Add in sample efficiency discussion

6. **`effect_size_forest.eps/.pdf`** (cell 26) — forest plot of effect sizes
   - Add in results to quantify magnitude of differences

7. **`summary_table.eps/.pdf`** (cell 31) — publication-ready summary
   - Consider replacing Table 2 with this if format is better

---

## Phase 9: Update Appendix Tables

### Table 18: Convergence Speed (lines 1569-1592)
- Update all iteration counts and speedup ratios with new data

### Table 19: n_init Sensitivity (lines 1601-1624)
- Change from 4 columns {1, 5, 20, 50} to 3 columns {5, 25, 50}
- Update all regret values

### Table 20: Best xi Per Problem (lines 1626-1650)
- Update all 13 best-xi values from notebook cell 5

### Hyperparameter details text (line 1599)
- n_init set: {1, 5, 20, 50} → {5, 25, 50}
- Run count formula: 13 x 4 x 7 x 10 x 2 = 7,280 → 13 x 3 x 7 x 10 x 2 = 5,460

### BH Baseline Details (line 1321-1350)
- Update to reflect 4-method comparison
- Add multistart SLSQP details (number of restarts, etc.)

---

## Phase 10: Update Figure References

| Paper references | Replace with | Notes |
|---|---|---|
| `figs/sfr1_search.eps` | `figs/search_sfr1_static.eps` or `.pdf` | Rename in \includegraphics |
| `figs/sfr2_search.eps` | `figs/search_sfr2_static.eps` or `.pdf` | Rename in \includegraphics |
| `figs/sfr_convergence.eps` | Regenerated in Phase 2 | Keep same name |
| `figs/dim_vs_constraint_scatter.eps` | Either remove or regenerate | Decision: remove paragraph, may keep figure in appendix |
| `figs/feasibility_rates.eps` | Regenerated in Phase 2 | Keep same name |
| `figs/sample_crossover.eps` | Regenerated in Phase 2 | Keep same name |
| `figs/convergence_grid.pdf` | Already current | Verify panel count (11 vs 12) |

---

## Phase 11: Verify

1. **Compile LaTeX**: Run `pdflatex` + `bibtex` to check for broken references
2. **Cross-check numbers**: Grep for old values (7410, 277000, "14\\times", "n_init=1") to catch any missed updates
3. **Figure file existence**: Verify all \includegraphics paths resolve to actual files
4. **Claim consistency**: Ensure abstract, results, discussion, and conclusion all use the same numbers

---

## Summary of Scope

- **~30 number substitutions** across abstract, results, discussion, conclusion, appendix tables
- **~10 paragraphs of prose rewriting** (results interpretation, discussion, limitations)
- **2 paragraphs/figures deleted** (dimensionality scaling claim + figure)
- **4 figures regenerated** in notebook (feasibility, sample crossover, dim scatter, SFR convergence)
- **Up to 7 new figures added** to paper from notebook
- **Table 2 expanded** to 4-way comparison (NLP, BH, BB-BO, Bi-BO)
- **3 appendix tables updated** (convergence speed, n_init sensitivity, best xi)
- **Paper restructured** for 4-method comparison throughout

### Critical files
- `Report/main.tex` — primary file to edit
- `Notebooks/analyze_results.ipynb` — add missing figure cells, run to get numbers
- `Report/figs/` — verify all figure files exist after notebook run
