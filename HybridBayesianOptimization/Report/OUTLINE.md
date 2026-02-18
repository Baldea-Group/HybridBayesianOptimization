# Article Outline: Exploiting Separability in Multi-Scale Grey-Box Bayesian Optimization

---

## Publication Venue Analysis

### Chemical Engineering Journals

**Computers & Chemical Engineering** (Elsevier)
- Paulson & Lu (2022) -- COBALT
- Kudva & Paulson (2026) -- BONSAI
- Winz, Fromme & Engell (2025) -- grey-box BO with modified UCB
- Tsay et al. -- multi-fidelity BO, CAMD with GNNs
- *Why it fits:* Home journal for grey-box BO in process systems. No page limits, accommodates 13-problem suite. Reviewers know the problem. Three of the four most direct comparators (COBALT, BONSAI, Winz) published here.

**AIChE Journal** (Wiley)
- Folie, Adjiman & Pistikopoulos (2007) -- solvent design
- Kudva, Sorourifar & Paulson (2022) -- constrained robust BO with regret bounds
- Tian & Ierapetritou (2024) -- feasibility-driven surrogate optimization
- *Why it fits:* Flagship PSE journal. Higher impact factor than CChE. Good for the process-material co-design framing.

**Industrial & Engineering Chemistry Research** (ACS)
- González & Zavala (2025) -- BOIS (AI/ML in ChE special issue)
- Boukouvala et al. (2025) -- surrogate hybridization and adaptive sampling
- Seo, Brennecke, Edgar, Stadtherr & Baldea (2023) -- multiscale ionic liquid design
- *Why it fits:* Baldea group has published here. AI/ML special issues are recurring. Application-oriented but algorithmic work is welcome.

**Chemical Engineering Science** (Elsevier)
- Zhu & Xu (2022) -- multiscale modeling and optimization
- *Why it fits:* If the framing emphasizes the multiscale modeling aspect.

**ACS Sustainable Chemistry & Engineering**
- Seo et al. (2023) -- multiscale design of ionic liquid solvents
- *Why it fits:* If a sustainability angle is added to the case studies (e.g., CO2 capture, green solvent design).

### ML Conferences

**ICML** (International Conference on Machine Learning)
- Astudillo & Frazier (2019) -- BOCF (the foundational composite BO paper)
- Gardner et al. (2014) -- BO with inequality constraints
- Tsay et al. (2025) -- EARL-BO (multi-step lookahead BO)
- *Why it fits:* Methodological ancestor (BOCF) appeared here. 13-problem benchmark suite is a strong empirical contribution. Requires tight 8-page format + appendix, possibly a regret bound.

**NeurIPS** (Neural Information Processing Systems)
- Astudillo & Frazier (2021) -- Function Networks
- Thebelt, Tsay, Lee et al. (2022) -- tree ensemble kernels for constrained BO
- Frazier group (2024) -- cost-aware BO via Gittins index
- *Why it fits:* Function Networks is the closest methodological comparator; the bilevel formulation is a distinct alternative. Strong empirical papers do get accepted.

**ICLR** (International Conference on Learning Representations)
- Fu, He, Tian & Tao (2024) -- Convergence of Bayesian Bilevel Optimization
- *Why it fits:* Direct precedent for bilevel BO. But ICLR increasingly expects theoretical analysis.

**AISTATS** (Artificial Intelligence and Statistics)
- Xie, Zhang, Paulson & Tsay (2025) -- GP acquisition function optimization
- Astudillo et al. (2023) -- preferential BO
- Shahriari et al. (2016) -- BO via regularization
- *Why it fits:* More receptive to well-motivated empirical methodology than NeurIPS/ICML. The statistical angle (scalar GP in reduced space vs. multi-output GP) is natural.

**JMLR** (Journal of Machine Learning Research)
- Ariafar et al. (2019) -- ADMMBO
- Ceccon, Tsay et al. (2022) -- OMLT toolkit
- *Why it fits:* Open-access, no page limits. Allows comprehensive treatment. Slower review cycle.

**Transactions on Machine Learning Research** (TMLR)
- Tsay group (2025) -- System-Aware Neural ODE Processes
- *Why it fits:* Newer venue, faster turnaround than JMLR, still rigorous. Good for methods papers that don't quite fit the conference format.

### Optimization / Operations Research Journals

**SIAM Journal on Optimization**
- Toscano-Palmerin & Frazier (2022) -- BO with expensive integrands
- *Why it fits:* If a theoretical analysis is added (convergence, sample complexity bounds). Prestigious optimization venue.

**Journal of Global Optimization** (Springer)
- Boukouvala group -- surrogate branch-and-bound, data-driven spatial B&B
- Merkert et al. (2022) -- bilevel mixed-integer
- *Why it fits:* Natural home for the global optimization + surrogate methodology angle. The bilevel structure and benchmark suite fit well.

**Optimization and Engineering** (Springer)
- Zhai & Boukouvala (2023) -- surrogate-based B&B for simulation optimization
- *Why it fits:* Engineering-oriented optimization. The "exploit known structure in simulation-based optimization" framing is core to this journal.

**INFORMS Journal on Computing**
- Black-box optimization with hidden constraints; bilevel combinatorial optimization
- *Why it fits:* Computational optimization methodology. The benchmark suite contribution would be valued.

**European Journal of Operational Research**
- Beck, Ljubic & Schmidt (2023) -- survey on bilevel optimization under uncertainty
- *Why it fits:* Bilevel + uncertainty is a natural framing. Broad OR audience.

### Design & Control Conferences

**ESCAPE** (European Symposium on Computer Aided Process Engineering)
- Published as Elsevier's *Computer Aided Chemical Engineering* series
- Paulson group (2024) -- BO for variational quantum algorithms
- *Why it fits:* Good venue for initial presentation before a full journal paper. PSE audience. Biennial.

**PSE** (International Symposium on Process Systems Engineering)
- *Why it fits:* Premier PSE conference. Alternates with ESCAPE. Strong fit for the multiscale co-design framing.

**IFAC ADCHEM** (Advanced Control of Chemical Processes)
- Paulson group (2024) -- performance-based MPC tuning via BO
- *Why it fits:* If framed around closed-loop co-design or real-time optimization with BO.

**ACC / CDC** (American Control Conference / IEEE Conference on Decision and Control)
- Paulson group (2023) -- local search region constrained BO
- *Why it fits:* If there's a control or real-time decision-making angle. Short paper format (6 pages).

**Winter Simulation Conference** (WSC)
- Frazier group -- simulation optimization, multi-attribute optimization
- *Why it fits:* The inner NLP + outer BO structure is essentially simulation optimization. WSC has a dedicated track for Bayesian/surrogate-based simulation optimization.

### Materials & Multidisciplinary

**Digital Discovery** (RSC)
- Gantzler et al. (2023) -- multi-fidelity BO for MOF/COF separations
- *Why it fits:* Newer RSC journal focused on AI/ML for molecular discovery. The PSA, membrane, and catalyst problems directly fit.

**Computational Materials Science** (Elsevier)
- Honarmandi et al. (2022) -- batch BO for materials design
- *Why it fits:* If the materials-side (catalyst, adsorbent, membrane polymer) is emphasized as the black-box.

**npj Computational Materials** (Nature)
- *Why it fits:* High-impact, open-access. Requires a strong materials application, ideally with a real (not synthetic) black-box.

**Structural and Multidisciplinary Optimization** (Springer)
- Engineering design DFO, MDO with surrogates
- *Why it fits:* The bilevel structure maps to MDO (multidisciplinary design optimization). Different community but the ideas transfer directly.

### Recommended Prioritization

| Priority | Venue | Framing | Risk |
|---|---|---|---|
| **1st** | **Computers & Chemical Engineering** | Grey-box BO for process-material co-design + benchmark suite | Low -- closest fit to existing literature |
| **2nd** | **AIChE Journal** | Multiscale process design via bilevel decomposition | Low-Medium -- higher prestige, need strong engineering narrative |
| **3rd** | **ICML or NeurIPS** | BO methodology: dimensionality reduction via exact inner optimization | Medium-High -- needs tight format, possibly theory, head-to-head vs BOCF |
| **4th** | **AISTATS** | Statistical efficiency of scalar vs. multi-output surrogates in structured problems | Medium -- less competitive, good fit |
| **5th** | **Journal of Global Optimization** | Surrogate-based bilevel optimization with benchmark suite | Medium -- optimization audience, values benchmarks |
| **6th** | **IECRes** | Applied BO for engineering design (target AI/ML special issue) | Low |
| **7th** | **Digital Discovery** | AI-driven materials-process co-design | Low-Medium -- newer journal |
| **8th** | **ESCAPE/PSE** | Conference paper first, then expand to journal | Low -- good stepping stone |

### Two-Paper Strategy

The approach used by Paulson (COBALT in CChE, then BONSAI in CChE) and Astudillo/Frazier (BOCF at ICML, then Function Networks at NeurIPS) suggests a possible split:
1. **CChE or AIChE J** for the full paper with 13 benchmarks and engineering framing
2. **ICML/NeurIPS/AISTATS** for a tighter methods paper with theoretical analysis and head-to-head vs. BOCF/COBALT

---

## 1. Introduction (1.5 pages)

**Opening hook:** Many real-world optimization problems couple expensive black-box simulations (DFT, molecular dynamics, CFD) with well-understood analytical models (mass/energy balances, economics). Standard BO treats everything as a black box, wasting samples learning what is already known.

**Problem statement:** When $f(x) = J(x^{WB\star}(f^{BB}(x^{BB})))$ and the white-box optimization is solvable exactly, the BO surrogate should operate over $\mathbb{R}^{n_{BB}}$ rather than $\mathbb{R}^{n_{WB}+n_{BB}}$.

**Contributions (3 bullets):**
1. A bilevel reformulation that reduces BO dimensionality by solving the white-box subproblem exactly via NLP, with exact constraint satisfaction (no chance constraints or moment approximations)
2. A benchmark suite of 13 grey-box problems (2--5 variables, 0--3 constraints, min and max) spanning chemical engineering domains -- the largest such suite for separable grey-box BO
3. Comprehensive empirical evidence: 14x--277,000x lower regret vs. black-box BO across all problems, with comparable wall time and robustness to hyperparameters (n_init, xi)

**Positioning vs. prior work (brief):** Contrast with COBALT (propagates uncertainty through white-box; requires moment approximations), Astudillo & Frazier's BOCF (composite $g(h(x))$ where all variables pass through $h$; does not exploit variable separability), and BOIS/BONSAI (structured BO but different problem class). Your method is the only one that (a) separates variables and (b) solves the white-box problem exactly rather than surrogating it.

---

## 2. Problem Formulation (1 page)

### 2.1 Notation and models
Directly from main.tex: $x^{WB}$, $x^{BB}$, $f^{WB}(x^{WB}, y) = 0$, $y = f^{BB}(x^{BB})$

### 2.2 The integrated design problem
Full problem (Eq. 1 from main.tex): joint $\min_{x^{WB}, x^{BB}}$ with process constraints, black-box coupling, and inequality constraints

### 2.3 Why standard approaches struggle
Two failure modes:
- Gradient-based NLP: $f^{BB}$ is non-differentiable
- Full-space BO: curse of dimensionality over $n_{WB} + n_{BB}$; wastes samples learning known structure

**Key remark:** The objective depends on $x^{BB}$ only implicitly through how $y$ affects the white-box solution. This is the separability being exploited.

---

## 3. Method: Bilevel Bayesian Optimization (1.5 pages)

### 3.1 Bilevel reformulation
Eq. 2 from main.tex:
- Outer: $\min_{x^{BB}} J(x^{WB\star})$ via BO
- Inner: $x^{WB\star} = \arg\min_{x^{WB}} J(x^{WB})$ s.t. $f^{WB}(x^{WB}, y) = 0$, $g(x^{WB}) \leq 0$ via NLP (SLSQP)

### 3.2 Algorithm (Algorithm 1 pseudocode)
1. Sample $n_\text{init}$ points in $\mathcal{X}^{BB}$
2. For each: evaluate $y = f^{BB}(x^{BB})$, solve inner NLP, record $J(x^{WB\star})$
3. Fit GP over $x^{BB} \mapsto J(x^{WB\star})$
4. Maximize acquisition function (EI) over $\mathcal{X}^{BB}$
5. Repeat

### 3.3 Key properties
- **Dimensionality reduction:** GP operates over $\mathbb{R}^{n_{BB}}$ not $\mathbb{R}^{n_{WB}+n_{BB}}$
- **Exact constraint satisfaction:** White-box constraints handled by NLP, not penalty/chance constraints
- **Scalar surrogate:** One GP for the optimal objective, not multi-output GP for $y \in \mathbb{R}^{n_y}$
- **Infeasibility handling:** If inner NLP is infeasible for some $y$, return penalty; acquisition naturally steers away

### 3.4 Contrast with COBALT and BOCF
Table comparing the three approaches on: what is surrogated, how constraints are handled, whether variables are separated, surrogate dimensionality. This is the material from main.tex Sec. 1.6 expanded into a crisp comparison table.

---

## 4. Benchmark Suite (2 pages)

**Motivation:** No existing benchmark suite targets the specific separable structure $J^* = \arg\min f_{WB}(x^{WB}, f^{BB}(x^{BB}))$ with constraints. Existing suites (BOCF, SMD, COBALT) either don't separate variables or lack engineering-relevant constraints.

### 4.1 Problem summary table

| Problem | $n_{WB}$ | $n_{BB}$ | $n_y$ | $n_g$ | Type | Domain |
|---|---|---|---|---|---|---|
| Small-Feasible-Region 1 | 1 | 1 | 1 | 1 | min | Synthetic |
| Small-Feasible-Region 2 | 1 | 1 | 1 | 1 | min | Synthetic |
| Rastrigin | 2 | 1 | 1 | 0 | min | Multimodal |
| Toy-Hydrology | 1 | 1 | 1 | 2 | min | Hydrology |
| Rosen-Suzuki | 2 | 2 | 2 | 3 | min | Constrained |
| CSTR | 3 | 2 | 3 | 2 | max | Catalysis |
| Heat-Exchanger | 3 | 2 | 3 | 2 | min | HEN design |
| PSA | 3 | 2 | 3 | 2 | min | Adsorption |
| Batch-Reactor | 3 | 2 | 3 | 2 | min | Kinetics |
| Distillation | 3 | 2 | 3 | 3 | min | Separation |
| Evaporator | 3 | 2 | 3 | 2 | min | Evaporation |
| Membrane | 3 | 2 | 3 | 2 | min | Membrane sep. |
| Williams-Otto | 3 | 2 | 2 | 2 | max | Process opt. |

### 4.2 Design principles
Each problem has verified global optima (via DE + bilevel DE + manual checks), physically motivated constraints, and black-box functions encoding realistic structure-property relationships (volcano curves, Langmuir isotherms, Robeson upper bound, etc.)

### 4.3 Detailed problem descriptions
Move to appendix (already written in main.tex Sec. A). Reference 2-3 representative problems in the main text (e.g., CSTR for catalysis, Membrane for Robeson trade-off, Williams-Otto as a classic benchmark).

---

## 5. Experimental Setup (0.75 pages)

**Solvers compared:**
1. **Black-box NLP** (multi-start SLSQP, 20 restarts) -- gradient-based baseline
2. **Black-box BO** (EI over full $[x^{WB}, x^{BB}]$ space) -- standard BO baseline
3. **Bilevel BO** (EI over $x^{BB}$ with inner NLP) -- proposed method

**Hyperparameter sweep:**
- $n_\text{init} \in \{1, 5, 20, 50\}$
- $\xi \in \{0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0\}$ (EI exploration parameter)
- 10 repetitions per configuration (seeds 42--51)
- 200 iterations per run

**GP setup:** ARD RBF kernel + WhiteKernel (sklearn)

**Metrics:**
- Simple regret: $|J_\text{best} - J^\star|$
- Convergence speed: iterations to reach 1% of initial regret
- Wall time
- Number of black-box evaluations

---

## 6. Results (2.5 pages)

### 6.1 Bilevel BO dominates black-box BO (main result)
- **Figure 1 (hero figure):** Log10 regret convergence curves for all 13 problems (4x4 grid of subplots, mean +/- shaded CI). Bilevel BO converges 1--4 orders of magnitude below BB-BO.
- **Table 1:** Final regret summary: BB/Bi ratio ranges from 14x (Evaporator) to 277,000x (Heat-Exchanger). Headline: bilevel BO wins on *every single problem*.

| Problem | dim | NLP | BB-BO | Bi-BO | BB/Bi ratio | best xi |
|---|---|---|---|---|---|---|
| Small-Feasible-Region | 2 | 0.0000 | 3.2002 | 0.0420 | 76x | 0.5 |
| Small-Feasible-Region-2 | 2 | 0.0546 | 0.1798 | 0.0000 | 42743x | 0.01 |
| Rastrigin | 3 | 3.3829 | 7.9551 | 0.0001 | 57717x | 0.2 |
| Toy-Hydrology | 2 | 0.0000 | 0.1278 | 0.0000 | 6459x | 0.001 |
| Rosen-Suzuki | 4 | 0.0000 | 6.8729 | 0.4763 | 14x | 0.2 |
| CSTR | 5 | 0.0000 | 2.8773 | 0.0052 | 552x | 0.001 |
| Heat-Exchanger | 5 | 0.0002 | 51.1785 | 0.0002 | 276962x | 0.01 |
| PSA | 5 | 0.0000 | 0.4743 | 0.0046 | 103x | 0.2 |
| Batch-Reactor | 5 | 0.0000 | 1.5285 | 0.0354 | 43x | 0.01 |
| Distillation | 5 | 0.0000 | 5556.2612 | 0.0321 | 173088x | 0.001 |
| Evaporator | 5 | 0.0000 | 0.7427 | 0.0393 | 19x | 0.1 |
| Membrane | 5 | 0.0000 | 16.5371 | 0.0001 | 115095x | 0.5 |
| Williams-Otto | 5 | 0.0000 | 2.2550 | 0.0000 | 115769x | 0.001 |

### 6.2 Convergence speed
- **Table 2:** Iterations to 1% target. Bilevel BO reaches 1% in 9--50 iterations where BB-BO never does within 200. Speedup 1.5x--22x.

| Problem | BB-BO | Bi-BO | Speedup |
|---|---|---|---|
| Small-Feasible-Region | >200 | >200 | -- |
| Small-Feasible-Region-2 | >200 | 36 | 5.6x |
| Rastrigin | >200 | 50 | 4.1x |
| Toy-Hydrology | >200 | 14 | 14.9x |
| Rosen-Suzuki | >200 | >200 | -- |
| CSTR | >200 | 14 | 13.9x |
| Heat-Exchanger | 151 | 102 | 1.5x |
| PSA | >200 | >200 | -- |
| Batch-Reactor | >200 | >200 | -- |
| Distillation | >200 | 10 | 20.1x |
| Evaporator | >200 | >200 | -- |
| Membrane | >200 | >200 | -- |
| Williams-Otto | >200 | 9 | 22.3x |

### 6.3 Dimensionality scaling
Plot or discussion: 2D problems show 76--6,500x improvement; 5D problems show 550--277,000x. The advantage grows because bilevel BO searches over $n_{BB}=1$--2 while BB-BO searches over $n_{WB}+n_{BB}=2$--5. This directly validates the curse-of-dimensionality argument.

### 6.4 Computational cost is comparable
- **Table 3:** Wall times. Both BO methods ~20--30s for 200 iterations. Exception: Williams-Otto (111s bilevel vs. 26s BB) due to expensive inner NLP. The GP is cheaper to fit in lower-dimensional space, roughly offsetting inner NLP cost.
- NLP uses 900--3000 $f^{BB}$ evaluations (20 multi-start runs); both BO methods use ~207. Bilevel BO is far more sample-efficient.

| Problem | NLP | BB-BO | Bi-BO |
|---|---|---|---|
| Small-Feasible-Region | 0.0 | 21.6 | 26.8 |
| Small-Feasible-Region-2 | 0.0 | 20.5 | 20.4 |
| Rastrigin | 0.0 | 19.5 | 17.4 |
| Toy-Hydrology | 0.0 | 23.4 | 21.1 |
| Rosen-Suzuki | 0.0 | 23.5 | 25.0 |
| CSTR | 0.1 | 30.0 | 28.3 |
| Heat-Exchanger | 0.0 | 27.0 | 23.3 |
| PSA | 0.1 | 28.1 | 25.4 |
| Batch-Reactor | 0.1 | 25.9 | 28.2 |
| Distillation | 0.0 | 28.2 | 24.1 |
| Evaporator | 0.1 | 27.7 | 25.7 |
| Membrane | 0.1 | 29.8 | 23.9 |
| Williams-Otto | 1.8 | 26.3 | 111.0 |

### 6.5 Robustness to hyperparameters
- **Figure 2:** $n_\text{init}$ sensitivity. Performance nearly flat across $n_\text{init} \in \{1, 5, 20, 50\}$. Even $n_\text{init}=1$ works. Practically important: fewer expensive initial evaluations needed.

| Problem | n=1 | n=5 | n=20 | n=50 |
|---|---|---|---|---|
| Small-Feasible-Region | 0.0345 | 0.0420 | 0.0449 | 0.0286 |
| Small-Feasible-Region-2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Rastrigin | 0.0001 | 0.0001 | 0.0001 | 0.0000 |
| Toy-Hydrology | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Rosen-Suzuki | 0.3909 | 0.4763 | 0.5257 | 0.3520 |
| CSTR | 0.0052 | 0.0052 | 0.0053 | 0.0050 |
| Heat-Exchanger | 0.0002 | 0.0002 | 0.0002 | 0.0002 |
| PSA | 0.0115 | 0.0046 | 0.0058 | 0.0061 |
| Batch-Reactor | 0.0282 | 0.0354 | 0.0314 | 0.0372 |
| Distillation | 0.0328 | 0.0321 | 0.0321 | 0.0321 |
| Evaporator | 0.0345 | 0.0393 | 0.0472 | 0.0781 |
| Membrane | 0.0001 | 0.0001 | 0.0001 | 0.0001 |
| Williams-Otto | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

- **Figure 3:** $\xi$ sensitivity heatmap. No single best $\xi$ (0.001--1.0 optimal depending on problem), but bilevel BO dominates BB-BO regardless of $\xi$. Suggests adaptive $\xi$ strategies may help but aren't critical.

### 6.6 Statistical significance
Wilcoxon signed-rank test on final regrets (paired across reps). Report p-values for bilevel BO vs. BB-BO on each problem.

---

## 7. Discussion (1 page)

**When does bilevel BO help most?**
- When $n_{WB} \gg n_{BB}$ (large dimensionality reduction)
- When white-box constraints are active (exact satisfaction matters)
- When the inner NLP is cheap relative to $f^{BB}$

**Limitations and when it fails:**
- Requires knowing the separable structure a priori (structure identification is not addressed)
- Inner NLP must be solvable (non-convex inner problems may find local optima -- see Rosen-Suzuki where improvement is "only" 14x)
- Williams-Otto shows that expensive inner NLPs can dominate wall time
- Black-box constraints (constraints that depend directly on $f^{BB}$ outputs without going through the white-box model) are not handled in the current formulation

**Relationship to COBALT:** COBALT surrogates the *outputs* $y$ and propagates uncertainty through $f^{WB}$. Our method surrogates the *optimal value* $J(x^{WB\star})$ directly. COBALT is more general (handles $y$-dependent constraints natively) but requires multi-output GPs and moment approximations. The approaches are complementary.

**Relationship to BOCF (Astudillo & Frazier):** BOCF assumes composite structure $g(h(x))$ where $h$ is the expensive part and $g$ is cheap. Our formulation is more structured: variables are partitioned, and $g$ (the white-box) is solved to optimality rather than simply evaluated. BOCF could in principle be applied but would not exploit the variable separation.

---

## 8. Conclusion (0.5 pages)

- Bilevel BO exploits separability in grey-box problems to achieve orders-of-magnitude improvement over black-box BO at no additional computational cost
- The 13-problem benchmark suite provides a standardized testbed for future grey-box BO methods
- Future work: adaptive acquisition functions, multi-fidelity inner models, automatic detection of separable structure, scaling to higher $n_{BB}$

---

## Appendix

- **A. Test Problem Definitions** -- Full mathematical formulations for all 13 problems (already written in main.tex)
- **B. Additional Convergence Plots** -- Per-problem regret curves for all $(n_\text{init}, \xi)$ combinations
- **C. Full Hyperparameter Sensitivity** -- Heatmaps from the notebooks
- **D. NLP Baseline Details** -- Multi-start configuration, constraint violation analysis (NLP can converge to infeasible points on constrained problems)

---

## Suggested Figures (6--7 total for main paper)

1. **Schematic diagram** -- Flow of bilevel BO framework (already sketched in MultiScaleDiagram.tex)
2. **Regret convergence of CSTR** --One representative problem to discuss while outlining the problem
3. **Regret convergence curves** -- 12-panel grid, the single most informative figure, showing the regret curves for the other problems.
4. **BB-BO vs. Bi-BO scatter** -- Each point is a (problem, config), diagonal = equal performance; all points far below diagonal
5. **Dimensionality scaling** -- BB/Bi regret ratio vs. total problem dimension; shows the curse-of-dimensionality argument
6. **$n_\text{init}$ robustness** -- Bar chart or small multiples showing flatness
7. **$\xi$ heatmap** -- Side-by-side heatmaps for Bi-BO and BB-BO
8. **Comparison table figure** -- Method comparison (bilevel BO vs. COBALT vs. BOCF vs. full-space BO) as a structured diagram

---

## Key Missing Elements to Address Before Writing

1. **Comparison with COBALT** -- Running COBALT on the same 13 problems would significantly strengthen the paper. Currently you only compare against naive baselines.
2. **Comparison with BOCF** -- Similarly, running Astudillo & Frazier's method would position the work precisely.
3. **Theoretical grounding** -- A regret bound or sample complexity argument (even informal) showing why $n_{BB}$-dimensional BO converges faster than $(n_{BB}+n_{WB})$-dimensional BO would add rigor for ML venues.
4. **A real (not synthetic) black-box** -- One problem with an actual DFT or molecular simulation call would strengthen the engineering motivation enormously. All current $f^{BB}$ are closed-form surrogates of what *would be* expensive.
