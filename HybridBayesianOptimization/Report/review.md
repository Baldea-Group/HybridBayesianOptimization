# Peer Review Response Plan

**Instructions**: For each issue below, choose a solution by writing `CHOSEN` next to it. Add any notes in the `AUTHOR NOTES:` field. When done, save the file and I will implement all chosen fixes.

---

## Issue 1: Objective function formulation error (MAJOR)

**Location**: Eq. (3a) line 163, line 170, and all downstream references.

**Problem**: The formulation writes $J(x^{WB})$ and line 170 states "The objective $J(x^{WB})$ depends only on the white-box variables." But at least 5 benchmarks have $J$ depending on $y$ or directly on $x^{BB}$:
- Rastrigin: $J = \ldots + y_1$
- SFR-1: $J = \sin(x^{WB}) + y_1$
- Toy-Hydrology: $J = x_1 + x_2$ where $x_1 \in x^{BB}$
- Rosen-Suzuki: $J$ contains $x_4^2$ where $x_4 \in x^{BB}$
- Batch-Reactor: $J$ depends on $C_B$ which depends on $y$

**Solution A**: Generalize the formulation to $J(x^{WB}, y)$. Change Eq. (3a) to $\min J(x^{WB}, y)$. Update line 170 to say the objective depends on white-box variables and black-box outputs. This covers most problems but not Rosen-Suzuki and Toy-Hydrology where $J$ depends directly on $x^{BB}$.

**Solution B**: Generalize fully to $J(x^{WB}, y, x^{BB})$ and then note that the bilevel decomposition still works because for fixed $x^{BB}$ and $y$, the inner NLP minimizes over $x^{WB}$ alone. This is the most honest and covers all 13 problems. Update Assumption 1(i) to allow $J$ to depend on $x^{BB}$ directly (not just through $y$), noting the separability is in the *optimization structure* (inner over $x^{WB}$, outer over $x^{BB}$), not necessarily in the function dependencies.

CHOSEN: Solution B. 

AUTHOR NOTES: $J(x^{WB}, y, x^{BB})$ is the most general form, however in practice the dependence on $x^{\BB}$ and $y$ may be implicitly captured by the feasible set of $x^{\WB}$ for each $x^{\BB}$. The Bi-level reformulation then optimizes $J(x^{WB})$ given a particular $x^{\BB}$ and $y$ (which is a function of $x^{\BB}$).

---

## Issue 2: Assumption 1(i) violated by benchmark problems (MAJOR)

**Location**: Assumption 1 (lines 176-182), Rosen-Suzuki (Appendix D.5), Toy-Hydrology (Appendix D.4).

**Problem**: Assumption 1(i) states "$J$ and $g$ depend on $x^{BB}$ only through $y = f^{BB}(x^{BB})$." Rosen-Suzuki has $x_4^2$ in $J$ and $x_3, x_4$ in $g_1$; Toy-Hydrology has $x_1$ in $J$, $g_1$, $g_2$—all direct $x^{BB}$ dependence.

**Solution A**: Redefine $f^{BB}$ for these problems to absorb the direct $x^{BB}$ terms. E.g., for Rosen-Suzuki, augment $y$ to include $x_3, x_4, x_3^2, x_4^2$ as additional black-box outputs. This makes the problems technically satisfy Assumption 1(i) without changing the optimization. Downside: the "black-box function" becomes partly trivial (identity mappings).

**Solution B**: Relax Assumption 1(i) to allow direct $x^{BB}$ dependence. Replace with: "For any fixed $x^{BB}$ and $y = f^{BB}(x^{BB})$, the resulting optimization over $x^{WB}$ alone is a well-posed NLP." This shifts the assumption from function-dependency separability to optimization-structure separability. Consistent with choosing Solution B for Issue 1.

CHOSEN: Solution B.

AUTHOR NOTES: Note that $f^{\BB}$ can be defined to include identity mappings for any direct $x^{\BB}$ dependence, but this is a technicality. The key point is that the inner optimization is over $x^{\WB}$ alone, holding $x^{\BB}$ and $y$ fixed, which is what allows the bilevel structure to work. The formulation requires that the optimization structure be *roughly* seperable, not necessarily exactly with dimminishing benefit the less seperable the problem is, as the search space of the inner NLP may be un-optimally constrained.

---

## Issue 3: Proposition 1 proof is incomplete (MAJOR)

**Location**: Appendix F, lines 1438-1440.

**Problem**: The proof is informal and has gaps:
1. Forward direction (original optimum is bilevel-feasible) is loosely stated but correct in spirit.
2. Converse ("any outer-level optimum... satisfies $J \leq J^*$ only if it also solves the original") is confusingly stated and doesn't rigorously establish $\inf(\text{bilevel}) = \inf(\text{original})$.
3. Does not address feasibility preservation: some $y \in \text{range}(f^{BB})$ may render the inner NLP infeasible.
4. Does not use Assumption 1(ii) explicitly in the argument.

**Solution A**: Rewrite the proof properly. Structure as: (i) show any feasible point of the original maps to a feasible outer-level point with equal or worse objective; (ii) show any outer-level feasible point maps back to a feasible point of the original with equal objective (using Assumption 1(ii) for inner global optimality); (iii) conclude optimal values are equal. Handle the infeasibility case separately.

**Solution B**: Downgrade Proposition 1 to a "Remark" or "Observation" with an informal argument, since the result is nearly self-evident once the structure is recognized. This avoids the expectation of formal rigor.

CHOSEN: Solution A

AUTHOR NOTES:

---

## Issue 4: Inflated run count (MODERATE)

**Location**: Line 511 (Section 5.1.2), abstract (line 70), conclusion (line 700).

**Problem**: "10,920 optimization runs" = $13 \times 4 \times 7 \times 10 \times 3$. But NLP results don't depend on $n_{init}$ or $\xi$. The NLP is either run once per (problem, seed) = 130 independent runs, or redundantly re-run 3,640 times. Actual independent runs: 7,410.

**Solution A**: Change the count to reflect independent runs: "7,410 independent optimization runs (7,280 BO + 130 NLP)." Adjust abstract and conclusion accordingly.

**Solution B**: Keep 10,920 but add a clarifying note: "...comprising 10,920 optimization runs (the NLP baseline is re-evaluated at each configuration for consistent timing comparisons)" or similar. This is defensible if the NLP was truly re-run each time.

CHOSEN: Solution A

AUTHOR NOTES:

---

## Issue 5: Inconsistent Rastrigin regret ratio (MODERATE)

**Location**: Table 3 (line 578) says 57,717x. Section 7 discussion (line 660) says 131,000x.

**Problem**: The two numbers cannot both be correct for the same setting. If 131,000x is from a different hyperparameter configuration, this must be stated.

**Solution A**: Check the data source and correct the Discussion to match Table 3 (57,717x). Add "(at $n_{init}=50$, best $\xi$)" to the Discussion reference for clarity.

**Solution B**: If 131,000x is from a different (valid) setting, state it explicitly: "Rastrigin achieves up to 131,000x improvement (at $n_{init}=X$, $\xi=Y$), compared to 57,717x at the default settings in Table 3."

CHOSEN: Solution A

AUTHOR NOTES:

---

## Issue 6: "Exact constraint satisfaction" claim overstated (MODERATE)

**Location**: Abstract (line 70), Introduction contributions (line 91), Section 4.3 (line 410), throughout.

**Problem**: The paper claims "exact constraint satisfaction—no penalty functions" but:
- Figure 8 shows <100% feasibility on 4/12 constrained problems (as low as 59%)
- Remark 2 (line 216) admits a penalty is used when the inner NLP is infeasible
- Assumption 1(ii) requires inner global optimality, which SLSQP cannot guarantee

**Solution A**: Qualify the claim everywhere it appears. E.g., "White-box constraints are satisfied exactly *whenever the inner NLP converges to a feasible point*" or "...satisfied exactly *in principle*; in practice, local NLP solvers may find infeasible local optima."

**Solution B**: Keep the theoretical claim as-is but add a sentence in the abstract/introduction acknowledging the practical gap: "In practice, local inner solvers may not achieve global optimality (Assumption 1(ii)), reducing feasibility rates on some problems (Section 7)."

CHOSEN: Solution A. 

AUTHOR NOTES: Note that the solver (SQLP) is a local NLP solver and thus may not find the global optimum of the inner problem, but using 20 random restarts is a common heuristic to improve the chances of finding a good solution. The paper should acknowledge this practical limitation while still emphasizing the theoretical advantage of bilevel BO in principle.


---

## Issue 7: Weak black-box BO baseline (MODERATE)

**Location**: Section 5.1.1 (lines 503-504), Discussion.

**Problem**: Black-box BO uses a fixed penalty of $10^6$ for constraint violations. Modern constrained BO (Gardner et al. 2014, Gelbart et al. 2014—both cited) uses probability-of-feasibility weighting, which is far more effective on tight constraints. The baseline thus understates what well-implemented black-box BO can achieve, and the largest reported gains (Heat-Exchanger 277,000x, Distillation 173,000x) occur exactly where penalty methods fail hardest.

**Solution A**: Add a Discussion paragraph acknowledging this limitation: "The black-box BO baseline uses a fixed penalty for constraint violations. More sophisticated constrained BO methods (feasibility-weighted EI, augmented Lagrangian) would likely narrow the gap on tightly constrained problems, though the dimensionality reduction advantage of bilevel BO would persist."

**Solution B**: Run a feasibility-weighted EI baseline (Gardner et al. 2014) on the constrained problems and report the comparison. This strengthens the paper substantially but requires additional computation.

CHOSEN: Solution A

AUTHOR NOTES: The point of this paper is that the *structure* of the problem (bilevel vs. black-box) is the main driver of performance, not the specific acquisition function. We intentionally use rather vanilla methods for BO and NLP to isolate the effect of the bilevel structure. Adding a more sophisticated constrained BO baseline would be interesting but is outside the scope of this paper, which focuses on the bilevel reformulation itself rather than specific BO techniques.

---

## Issue 8: Notation inconsistency in Eq. (4) (MINOR)

**Location**: Lines 197-206 (the bilevel reformulation).

**Problem**: Eq. (4) uses `x^{BB}`, `x^{WB}` (plain text superscripts) while the rest of the paper uses `x^{\BB}`, `x^{\WB}` (upright roman via `\mathrm`). Creates visual inconsistency in the rendered PDF.

**Solution**: Replace all instances in Eq. (4) with the `\BB`/`\WB` macros. No content change, just formatting.

CHOSEN: Implement the solution as described.

AUTHOR NOTES:

---

## Issue 9: Integer relaxation not acknowledged (MINOR)

**Location**: Evaporator problem, Appendix D.7 (line 1087: $n \in [2, 6]$).

**Problem**: Number of evaporator effects is physically an integer but treated as continuous.

**Solution**: Add a parenthetical: "...$n \in [2, 6]$ (treated as continuous for this study; integer rounding would apply in practice)."

CHOSEN: Implement the solution as described.

AUTHOR NOTES:

---

## Issue 10: Convergence speed table lower bounds (MINOR)

**Location**: Table 8, Appendix H (lines 1460-1479).

**Problem**: When BB-BO is ">200," the speedup is computed as $200/t_{Bi-BO}$ (e.g., 200/36 = 5.6x). This is a lower bound, not an exact speedup.

**Solution**: Change "5.6x" to "$\geq$5.6x" (and similarly for all rows where BB-BO is >200). Or add a table footnote: "Speedup values where BB-BO exceeds 200 iterations are lower bounds."

CHOSEN: Implement the solution as described.

AUTHOR NOTES:

---

## Issue 11: No convergence theory discussion (MINOR)

**Location**: Section 4 or Section 7.

**Problem**: No convergence guarantees are discussed. Standard BO convergence results (Srinivas et al. 2010, Bull 2011) apply to the outer loop since it is standard BO over a compact domain with a GP surrogate.

**Solution A**: Add 2-3 sentences in Section 4.3 (Key Properties): "Standard BO convergence results apply to the outer loop: under regularity conditions on $x^{BB} \mapsto J(x^{WB*})$, the simple regret of the outer GP-EI loop converges to zero as $N \to \infty$ (Bull, 2011). The bilevel structure does not change the convergence rate but reduces the effective dimension from $n_{WB}+n_{BB}$ to $n_{BB}$, which improves the constant in dimension-dependent regret bounds."

**Solution B**: Skip formal convergence discussion; note it as future work in Limitations.

CHOSEN: Solution B.

AUTHOR NOTES:

---

## Issue 12: Dimensionality reduction claim vs. COBALT (MINOR)

**Location**: Table 2 (line 430), Section 4.4 (line 439).

**Problem**: Table 2 shows both bilevel BO and COBALT have surrogate dimension $n_{BB}$. The dimensionality advantage over COBALT is zero. The real advantages are scalar GP (vs. multi-output) and exact NLP (vs. moment approximation). The paper sometimes conflates "dimensionality reduction over full-space BO" with "advantage over COBALT."

**Solution**: Add a clarifying sentence after Table 2: "Both bilevel BO and COBALT achieve the same input dimensionality reduction to $n_{BB}$. The distinction is in the surrogate output (scalar vs. multi-output) and constraint handling (exact vs. approximate), not the input dimension."

CHOSEN: Implement the solution as described.

AUTHOR NOTES:

---

## Issue 13: Unused black-box output $y_3$ in Batch-Reactor (MINOR)

**Location**: Appendix D.6, line 973.

**Problem**: $y_3 = f^{BB}_3(E_b, d) = \exp(-2E_b - 1)$ is defined but never appears in any white-box equation or constraint. It is a dead output.

**Solution A**: Remove $y_3$ from the Batch-Reactor formulation and change $n_y$ from 3 to 2 in Table 1.

**Solution B**: If $y_3$ is intentional (e.g., to test robustness to irrelevant outputs), add a note explaining its purpose.

CHOSEN: Solution B

AUTHOR NOTES:

---

## Issue 14: Acquisition function grid search sparsity (MINOR)

**Location**: Appendix B.6, line 1309.

**Problem**: 1,000 random candidates for acquisition maximization gives ~31 points/dimension at $n_{BB}=2$. This is sparse and may explain some of the variance in results, especially the modest 14x on Rosen-Suzuki.

**Solution A**: Acknowledge this as a limitation: "The grid resolution of 1,000 candidates is sufficient for $n_{BB} \leq 2$ but would require scaling (e.g., to $10^4$--$10^5$) for higher-dimensional black-box spaces."

**Solution B**: No change needed—1,000 is standard practice and the paper's scope is $n_{BB} \leq 2$.

CHOSEN: Solution B

AUTHOR NOTES:

---

## Summary Checklist

After editing, confirm:
- [ ] Every issue has a CHOSEN solution (A, B, or SKIP/custom)
- [ ] Any custom solutions are described in AUTHOR NOTES
- [ ] Save the file

I will then implement all chosen fixes in `main.tex`.
