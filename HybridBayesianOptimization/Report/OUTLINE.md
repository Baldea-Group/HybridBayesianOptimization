# Article Outline: Exploiting Separability in Multi-Scale Grey-Box Bayesian Optimization

---

## Publication Venue Analysis

### Impact Factor & Prestige Comparison

#### Chemical Engineering Journals

| Journal | IF (2024) | CiteScore | SJR | Quartile | Accept Rate | Prestige |
|---------|-----------|-----------|-----|----------|-------------|----------|
| **ACS Sustain. Chem. & Eng.** | **7.3** | 12.5 | 1.623 | Q1 | — | High (but niche: sustainability) |
| **Chemical Engineering Science** | **4.3** | 7.5–7.9 | 0.84 | Q1 | ~25% | Elite (foundational ChemE) |
| **AIChE Journal** | **4.0** | 7.1–7.3 | 0.805 | Q1/Q2 | ~30% | Elite (flagship of AIChE) |
| **Computers & Chem. Eng.** | **3.9** | 7.6–8.4 | 0.872 | Q1/Q2 | ~26% | Best-in-class for PSE/optimization |
| **Ind. & Eng. Chem. Research** | **3.9** | 6.7 | 0.828 | Q2 | ~50% | Solid workhorse; less selective |

- **ACS Sustainable Chem. & Eng.** has the highest raw IF (7.3) but is a poor fit unless a sustainability angle is added — wrong audience for optimization methodology.
- **CES** and **AIChE J** are co-equal in general ChemE prestige. CES has a slightly higher IF (4.3) and leans toward fundamental physics/transport; AIChE J has stronger institutional brand and faster reviews (~2.4 months).
- **Computers & Chem. Eng.** has a slightly lower IF (3.9) but is *the* home journal for PSE and grey-box BO. COBALT, BONSAI, and Winz all published here. The IF understates its prestige within the target subcommunity.
- **I&ECR** has the highest acceptance rate (~50%) and is Q2. It's a tier below for methods-oriented work.

**Bottom line:** CChE is the best topical fit. AIChE J is the prestige stretch target requiring an engineering-forward narrative.

#### Optimization / Operations Research Journals

| Journal | IF (2024) | CiteScore | SJR | Quartile | Accept Rate | Prestige |
|---------|-----------|-----------|-----|----------|-------------|----------|
| **European J. of OR** | **6.0–7.4** | 10.5–13.6 | 2.239 | Q1 | ~13% | **Tier 1** — top OR journal |
| **SIAM J. on Optimization** | **2.3** | 4.7–4.9 | 1.388 | Q1 | ~20–25% | **Tier 1** — elite in optimization theory |
| **INFORMS J. on Computing** | **2.1–3.1** | 3.7 | 1.439 | Q1 | ~15–20% | **Tier 1** — elite in computational OR |
| **Optimization and Eng.** | **2.6** | 3.9 | 0.573 | Q2 | — | **Tier 2** — bridge journal |
| **J. of Global Optimization** | **1.7–1.9** | 4.0 | 0.807 | Q1 | ~30–40% | **Tier 2** — niche but respected |

- **EJOR** dominates on raw metrics (IF 6–7, 13% acceptance). Broadest OR readership. Ambitious but realistic — publishes BO and surrogate-based optimization papers.
- **SIOPT** has a modest IF (2.3) that is deeply misleading — it's a top-2 optimization journal worldwide (alongside *Mathematical Programming*). Would require a strong theoretical contribution (convergence/regret bounds).
- **IJOC** has very high prestige within INFORMS but leans toward discrete/combinatorial optimization. Less natural for continuous BO work.
- **JOGO** is the **best topical fit** — the primary journal for global/Bayesian/surrogate-based optimization. Reviewers would be domain experts. Q1, respectable, and a realistic target.
- **OPTE** is a good fit if leading with the engineering application angle. Lower prestige ceiling but the interdisciplinary mission directly matches the paper.

**Bottom line:** JOGO is the most natural OR venue (reviewers know BO, benchmarks are valued). EJOR is the high-impact stretch. SIOPT only with serious theory added.

#### Combined Ranking (fit × prestige for this paper)

| Priority | Venue | IF | Rationale |
|----------|-------|----|-----------|
| 1 | **Computers & Chem. Eng.** | 3.9 | Perfect scope; target community publishes here |
| 2 | **AIChE Journal** | 4.0 | Higher general prestige; needs engineering framing |
| 3 | **EJOR** | 6–7 | Highest IF of all; accepts BO/benchmark papers; very competitive |
| 4 | **J. of Global Optimization** | 1.8 | Best OR fit; BO reviewers; realistic acceptance |
| 5 | **SIAM J. on Optimization** | 2.3 | Elite prestige but requires theory not yet developed |

The prestige hierarchy is somewhat orthogonal to topical fit. CChE's IF (3.9) hides the fact that it's the #1 venue for the target subcommunity, while EJOR's IF (6–7) reflects broad OR readership but means stiffer competition and less specialized reviewers.

---

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

## Restructured Outline for Journal of Global Optimization (JOGO)

**Target audience:** Optimization researchers familiar with surrogate-based DFO, Bayesian optimization, bilevel programming, and global optimization. Not ChemE specialists.

**Key reframing:** Lead with the optimization structure (separable variables, bilevel decomposition, dimensionality reduction), not the engineering application. Engineering examples motivate but do not drive the story.

---

## 1. Introduction (~2 pages) — Absorbs Related Work

**Paragraph 1 (Opening — optimization frame):**
Surrogate-based optimization is the standard approach for expensive black-box functions, but many problems contain known, differentiable substructure that monolithic surrogates waste samples learning. Frame: the cost of ignoring structure grows with dimension.

**Paragraph 2 (Concrete motivation):**
Multi-scale engineering design as the motivating problem class: DFT + reactor balances, molecular simulations + separation models. One paragraph—examples serve the optimization story, not the other way around.

**Paragraph 3 (Separability observation):**
The key structural property: black-box and white-box variables are *separable*. The black-box depends only on $x^{BB}$; the objective depends on $x^{BB}$ only implicitly through the white-box solution. When the white-box optimization is solvable exactly, the surrogate should operate over $\mathbb{R}^{n_{BB}}$ rather than $\mathbb{R}^{n_{WB}+n_{BB}}$.

**Paragraph 4 (Related work funnel — grey-box BO):**
Absorb former Sec. 3.1. Position against COBALT, BOCF, BOFN, BONSAI in 2--3 sentences. Key contrast: they surrogate intermediate outputs $y$ and propagate uncertainty; we surrogate $J^*$ directly. One sentence on deterministic grey-box methods (ARGONAUT, trust-region filter). One sentence connecting to decomposition in global optimization (block coordinate DFO, variable partitioning).

**Paragraph 5 (Related work funnel — bilevel BO):**
Absorb former Secs. 3.2--3.4. Position against Kieffer et al., Ekmekcioglu et al., Chew et al. Bilevel BO exists but hasn't exploited grey-box separability for dimensionality reduction.

**Paragraph 6 (Challenge + Contributions):**
State the question explicitly: *When an optimization problem decomposes into a cheap differentiable subproblem and an expensive black-box subproblem with separable variables, how should we design the surrogate-based search?*

Contributions:
1. A bilevel reformulation reducing surrogate dimensionality from $n_{WB}+n_{BB}$ to $n_{BB}$ with exact constraint satisfaction
2. A 13-problem benchmark suite for separable grey-box optimization (2--5 variables, 0--3 constraints)
3. Comprehensive empirical evidence: 14x--277,000x lower regret across 10,920 runs, robust to hyperparameters

---

## 2. Problem Formulation (~1.5 pages) — Generalized language

### 2.1 Notation and models
White-box / black-box variable partition. Engineering examples in parentheses only.
$f^{WB}(x^{WB}, y) = 0$, $y = f^{BB}(x^{BB})$

### 2.2 The integrated optimization problem
Full problem (Eq. 1): joint $\min_{x^{WB}, x^{BB}}$ with white-box model constraints, black-box coupling, and inequality constraints.

### 2.3 Separability assumption
**Assumption 1 (Separable grey-box structure):** Formal definition of when the method applies — the black-box function depends only on $x^{BB}$, the objective depends on $x^{BB}$ only through $y$, and the white-box subproblem is solvable to global optimality for fixed $y$.

### 2.4 Why standard approaches struggle
Two failure modes:
- Gradient-based NLP: $f^{BB}$ is non-differentiable / expensive to finite-difference
- Full-space surrogate optimization (EGO, BO): curse of dimensionality over $n_{WB} + n_{BB}$; wastes samples learning known structure

---

## 3. Bilevel Bayesian Optimization (~2 pages) — Method

### 3.1 Bilevel reformulation
- Outer: $\min_{x^{BB}} J(x^{WB\star})$ via BO
- Inner: $x^{WB\star} = \arg\min_{x^{WB}} J(x^{WB})$ s.t. $f^{WB}(x^{WB}, y) = 0$, $g(x^{WB}) \leq 0$ via NLP

### 3.2 Algorithm (Algorithm 1 pseudocode)
1. Sample $n_\text{init}$ points in $\mathcal{X}^{BB}$
2. For each: evaluate $y = f^{BB}(x^{BB})$, solve inner NLP, record $J(x^{WB\star})$
3. Fit GP over $x^{BB} \mapsto J(x^{WB\star})$
4. Maximize acquisition function (EI) over $\mathcal{X}^{BB}$
5. Repeat

### 3.3 Surrogate model and acquisition function
ARD RBF kernel, EI with exploration parameter $\xi$.

### 3.4 Key properties
- **Proposition 1:** Under Assumption 1, the bilevel reformulation preserves the global optimum and reduces the GP surrogate dimension from $n_{WB}+n_{BB}$ to $n_{BB}$.
- **Exact constraint satisfaction:** White-box constraints handled by NLP exactly
- **Scalar surrogate:** One GP, not multi-output GP
- **Infeasibility handling:** Penalty for infeasible inner NLP

### 3.5 Relationship to existing grey-box methods
Comparison table (Bilevel BO vs. COBALT vs. BOCF vs. Full-space BO): what is surrogated, surrogate dimension, constraint handling, inner optimization. Technical positioning that complements the Introduction's high-level funnel.

---

## 4. Benchmark Suite (~1 page main text) — Condensed

**Motivation:** No existing suite targets separable grey-box structure with constraints. Position relative to CEC, COCO/BBOB, SMD bilevel suites.

### 4.1 Problem summary table
13 problems, 2--5 variables, 0--3 constraints, min/max, synthetic + engineering domains.

### 4.2 Design principles
Verified global optima, physically motivated constraints, realistic structure-property black-box functions. One paragraph.

### 4.3 Representative problems (brief)
SFR-1 (simplest instance, permits visualization) and CSTR (engineering domain with Sabatier volcano). 2--3 sentences each. All formulations in Appendix A.

---

## 5. Computational Experiments (~3.5 pages) — Merged Setup + Results

### 5.1 Experimental design (~0.75 page)
Solvers: BB-NLP (20-restart SLSQP), BB-BO (full-space EI), Bilevel BO (proposed).
Sweep: $n_\text{init} \in \{1, 5, 20, 50\}$, $\xi \in \{0.001, \ldots, 1.0\}$, 10 reps, 200 iters.
GP: ARD RBF + WhiteKernel (scikit-learn).
Metrics: simple regret, convergence speed, wall time.
Total: 10,920 runs.

### 5.2 Mechanism visualization (~0.75 page)
SFR-1 and SFR-2 search plots + convergence. Combined 2-row figure. Geometric intuition before statistical evidence.

### 5.3 Main results (~1 page)
Final regret table (all 13 problems), convergence grid figure, convergence speed as additional column.
Key result: 14x--277,000x lower regret on every problem, statistically significant (p < 0.01, Wilcoxon).

### 5.4 Scaling and robustness (~1 page)
Dimensionality scaling plot (regret ratio vs. dim), scatter plot (all configs below diagonal), wall time comparison, $n_\text{init}$ and $\xi$ sensitivity (one figure + one paragraph each).

---

## Analysis for Discussion

Additional quantitative analyses to strengthen the Discussion section and explain the results. Organized by impact level.

### Tier 1: Directly address likely reviewer objections

**1. Feasibility Rate Comparison** *(Implemented)*
- For each constrained problem, compute fraction of BB-BO samples landing in feasible regions vs. bilevel BO (always 100% by construction).
- Plot feasibility rate over iterations or report as summary table.
- Quantifies the constraint-handling advantage: BB-BO wastes 40-80% of evaluations in infeasible regions on tight-constraint problems (Distillation, Heat-Exchanger), while bilevel BO is always feasible.
- Computable from stored `G_history` and `X_history`.

**2. GP Surrogate Quality Comparison**
- After fitting GP at iteration t, measure predictive quality (LOO-CV RMSE, kernel length scales) for both methods.
- Directly tests the claim that the lower-dimensional surrogate is more accurate per sample.
- Requires storing GP objects or re-fitting from saved data. Moderate effort.

**3. Dimensionality vs. Constraint Tightness Scatter** *(Implemented)*
- Two-axis scatter: x = dimensionality reduction ratio $(n_{WB}+n_{BB})/n_{BB}$, y = feasible fraction of domain (estimated by random sampling within bounds), color = BB/Bi regret ratio.
- Separates two mechanisms: dimensionality reduction (Rastrigin) vs. dimensionality + constraint compounding (Distillation).
- Explains large spread at 5D in Figure 6. Answers "is this just about constraints?" and "is this just about dimension?" — it's both, and they compound.

### Tier 2: Deepen the narrative

**4. Sample Efficiency Crossover** *(Implemented)*
- For each problem, report the iteration at which bilevel BO achieves BB-BO's *final* regret at iteration 200.
- Reframes advantage for practitioners: "bilevel BO matches BB-BO's best-after-200 in just 12 iterations."
- Pure post-processing on stored regret curves. Trivial to compute.

**5. Inner NLP Convergence Analysis**
- For Rosen-Suzuki (weakest result, 14x), log how many distinct local optima the inner NLP converges to across 200 iterations. Compare with well-behaved problem (CSTR).
- Turns the Rosen-Suzuki speculation into a characterized limitation.
- Uses stored `X_wb_history`. Cluster solutions and count distinct clusters.

**6. Learned GP Length Scales**
- Extract fitted kernel length scales from both GP surrogates at end of optimization.
- If BB-BO GP assigns very long length scales to x^WB dimensions, it has learned those directions are "resolved"—wasting capacity. Gives mechanistic insight.
- `sklearn` GPs expose `kernel_.get_params()`. Straightforward extraction.

### Tier 3: Nice-to-have if space permits

**7. Landscape Smoothness Visualization**
- For 1D problem (SFR-1 or Toy-Hydrology), plot actual function $x^{BB} \to J(x^{WB*})$ with GP posterior overlaid at iterations 10, 50, 200.
- Pedagogical—lets readers *see* why a 1D surrogate converges faster.

**8. Robustness to Inner NLP Solver Quality**
- Vary `n_inner_starts` from 1 to 20, measure effect on final regret for 2-3 problems.
- Answers "how good does my inner solver need to be?"

---

## 6. Discussion (~1.5--2 pages) — Expanded significantly

**Paragraph 1 (Main finding in optimization terms):**
The bilevel decomposition converts a $(n_{WB}+n_{BB})$-dimensional surrogate problem into $n_{BB}$-dimensional surrogate + exact NLP. Gains are structural, not algorithmic (same GP, same EI, same optimizer).

**Paragraphs 2--3 (When and why it works):**
- Effective dimensionality reduction → lower GP sample complexity
- Exact constraint satisfaction compounds with dimensionality advantage on tight-constraint problems
- Inner-problem convexity matters: Rosen-Suzuki's 14x gain as evidence that non-convex inner problems limit the approach

**Paragraph 4 (Connection to decomposition in global optimization):**
*Critical for JOGO.* Connect to:
- Block coordinate DFO / variable decomposition (Rios & Sahinidis, Audet & Hare)
- Trust-region decomposition (Eason & Biegler)
- Bilevel optimization theory: when does exact inner solve help outer convergence?
- Separability detection in black-box optimization

**Paragraph 5 (Complementarity with COBALT and BOCF):**
COBALT: more general (y-dependent constraints, uncertainty propagation), but multi-output GP + moment approximations.
BOCF: non-separable composite functions, but no variable decomposition.
Head-to-head comparison as future work.

**Paragraph 6 (Limitations as open problems):**
1. Separability must be known a priori (automatic detection is future work)
2. Inner NLP must reach global optimum (non-convex inner → local optima, e.g. Rosen-Suzuki)
3. Expensive inner NLP can dominate wall time (Williams-Otto: 111s vs. 26s)
4. Pure black-box constraints not handled (requires constrained outer BO)
5. All black-box functions are closed-form surrogates, not real DFT/MD

**Paragraph 7 (Broader implications — closing the circle):**
Return to Opening: expensive black-box optimization with exploitable structure is ubiquitous. The bilevel decomposition is one instance of a general principle—solve what you can solve exactly, surrogate only what you must. As surrogate-based methods scale to higher dimensions, structural decomposition becomes increasingly important.

---

## 7. Conclusion (~0.5 page)

Open with main message (not method): "When grey-box optimization problems exhibit variable separability, solving the known subproblem exactly yields..."
Benchmark suite as a testbed contribution.
End on the general principle (echo Opening), not future work list. Move future work items to end of Discussion.

---

## Appendix

- **A. Test Problem Definitions** -- Full formulations for all 13 problems
- **B. Acquisition Function Details** -- EI, PI, LCB, mWB2, Thompson
- **C. GP Configuration Details** -- Kernel, hyperparameter bounds, fitting
- **D. NLP Baseline Details** -- Multi-start, gradient computation, failure modes
- **E. Global Optimum Verification** -- DE + bilevel DE + multi-start, feasibility checks
- **F. Additional Sensitivity Results** -- Full $n_\text{init}$ and $\xi$ tables/heatmaps

---

## Figures (main paper, ~8 total)

1. **Schematic diagram** -- Bilevel BO framework flow (MultiScaleDiagram.tex)
2. **SFR search behavior** -- 2x2: SFR-1 and SFR-2, BB-BO vs. Bi-BO search points
3. **SFR convergence** -- Regret curves for both SFR problems
4. **Convergence grid** -- 11-panel grid for remaining problems
5. **Dimensionality scaling** -- Regret ratio vs. total dimension
6. **BB vs. Bi scatter** -- All (problem, config) points below diagonal
7. **$n_\text{init}$ robustness** -- Bar chart showing flatness
8. **$\xi$ heatmap** -- Side-by-side for Bi-BO and BB-BO

---

## Key Changes from Previous (ChemE) Outline

| Aspect | Previous (ChemE) | JOGO version |
|---|---|---|
| Opening frame | Engineering applications (DFT, reactors) | Optimization structure (separability, surrogates) |
| Related Work | Standalone Sec. 3 | Absorbed into Introduction funnel |
| Formulation language | "process" / "material" primary | "white-box" / "black-box" primary |
| Separability | Remark | Formal Assumption |
| Dimensionality reduction | Remark | Proposition |
| Benchmark Suite | 2 pages, detailed descriptions | 1 page + appendix |
| Setup + Results | Separate sections, 7 subsections | Merged, 4 subsections |
| Discussion | 1 page, 4 paragraphs | 1.5--2 pages, connects to GO literature |
| GO literature connections | Absent | Explicit (block DFO, bilevel theory, decomposition) |

---

## Key Missing Elements

1. **Head-to-head with COBALT/BOCF** -- Direct empirical comparison would significantly strengthen positioning
2. **Theoretical grounding** -- Even an informal sample complexity argument for $n_{BB}$ vs. $n_{WB}+n_{BB}$ dimensional GP
3. **Real black-box** -- One DFT or MD call would strengthen the engineering motivation
4. **GO literature connections** -- Cite Rios & Sahinidis DFO survey, Audet & Hare structured DFO, relevant bilevel optimization theory
