"""
functions.py - Test problems for bi-level hybrid optimization

This module implements test problems for comparing black-box vs bi-level optimization:

Black-box approach:
    min     J(x^{WB}, x^{BB})                 - optimize all variables simultaneously
    x^{WB},x^{BB}
    s.t.    bounds on x^{WB}, x^{BB}

Bi-level approach:
    min     J(x^{WB}*, y)                  (outer problem - BO over x^{BB})
    x^{WB}
    s.t.    y = f^{BB}(x^{BB})                - black-box function (material/molecular model)
            x^{WB}* = argmin { J(x^{WB}, y) | g(x^{WB}) ≤ 0 }  (inner problem - constrained opt)

The key distinction:
- Black-box: treats entire problem as black-box, no structure exploited
- Bi-level: exploits known structure of J(x^{WB}, y) to solve inner optimization
           exactly, only uses BO for the truly black-box component f^{BB}(x^{BB})

Mathematical notation:
    x^{WB} : process variables (optimized in inner problem)
    x^{BB} : material/design variables (optimized by BO in outer problem)
    y   : black-box outputs, y = f^{BB}(x^{BB})
    J   : objective function J(x^{WB}, y)
    g   : constraints on process variables

Problems included:
    - 
"""

import numpy as np
from typing import Tuple, Callable, Optional
from dataclasses import dataclass, field
from scipy.optimize import minimize, basinhopping, OptimizeResult


# =============================================================================
# BiLevelProblem Dataclass
# =============================================================================

@dataclass
class BiLevelProblem:
    """
    Container for bi-level optimization problems.

    Structure:
        Outer (BO):   min_{x^{BB}}  J(x^{WB}*, y)  where y = f^{BB}(x^{BB})
        Inner (NLP):  x^{WB}* = argmin_{x^{WB}} { J(x^{WB}, y) | g(x^{WB}) ≤ 0 }

    For black-box comparison, can also optimize over x = [x^{WB}, x^{BB}] simultaneously.

    Attributes:
        name: Problem identifier

        # Dimensions
        n_x_wb: Number of white-box variables solved explicitly x^{WB}
        n_x_bb: Number of black-box variables x^{BB}
        n_y: Number of black-box outputs y
        n_g: Number of constraints

        # Bounds
        x_wb_lower: Lower bounds on x^{WB}
        x_wb_upper: Upper bounds on x^{WB}
        x_bb_lower: Lower bounds on x^{BB}
        x_bb_upper: Upper bounds on x^{BB}

        # Functions
        fbb: Black-box function y = f^{BB}(x^{BB})
        J: Objective function J(x^{WB}, y) -> scalar
        g: Optional constraint function g(x^{WB}, y) -> vector (≤ 0 feasible)

        # Derivatives (optional, for inner optimization)
        J_grad_x_wb: Gradient of J w.r.t. x^p
        J_hess_x_wb: Hessian of J w.r.t. x^p
        g_grad_x_wb: Jacobian of g w.r.t. x^p

        # Optimal solution (for benchmarking)
        x_wb_optimal: Known optimal x^p*
        x_bb_optimal: Known optimal x^m*
        J_optimal: Known optimal objective value J*

        # Problem type
        maximize: If True, the problem is maximization (default False = minimization)
    """
    name: str

    # Dimensions
    n_x_wb: int
    n_x_bb: int
    n_y: int
    n_g: int = 0

    # Bounds
    x_wb_lower: np.ndarray = field(default_factory=lambda: np.array([]))
    x_wb_upper: np.ndarray = field(default_factory=lambda: np.array([]))
    x_bb_lower: np.ndarray = field(default_factory=lambda: np.array([]))
    x_bb_upper: np.ndarray = field(default_factory=lambda: np.array([]))

    # Core functions
    fbb: Callable[[np.ndarray], np.ndarray] = None  # y = f^{BB}(x^{BB})
    J: Callable[[np.ndarray, np.ndarray], float] = None  # J(x^{WB}, y)
    g: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None  # g(x^{WB}, y) ≤ 0

    # Optional derivatives for inner optimization
    J_grad_x_wb: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
    J_hess_x_wb: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
    g_grad_x_wb: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None

    # Known optimal solution
    x_wb_optimal: Optional[np.ndarray] = None
    x_bb_optimal: Optional[np.ndarray] = None
    J_optimal: Optional[float] = None

    # Problem type
    maximize: bool = False

    # Evaluation counters
    n_fbb_evals: int = field(default=0, init=False)
    n_J_evals: int = field(default=0, init=False)

    def __post_init__(self):
        """Validate problem setup."""
        if self.x_wb_lower.size == 0:
            self.x_wb_lower = np.full(self.n_x_wb, -np.inf)
        if self.x_wb_upper.size == 0:
            self.x_wb_upper = np.full(self.n_x_wb, np.inf)
        if self.x_bb_lower.size == 0:
            self.x_bb_lower = np.full(self.n_x_bb, -np.inf)
        if self.x_bb_upper.size == 0:
            self.x_bb_upper = np.full(self.n_x_bb, np.inf)

    def reset_counters(self):
        """Reset evaluation counters."""
        self.n_fbb_evals = 0
        self.n_J_evals = 0

    def get_full_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get combined bounds for x = [x^{WB}, x^{BB}]."""
        lower = np.concatenate([self.x_wb_lower, self.x_bb_lower])
        upper = np.concatenate([self.x_wb_upper, self.x_bb_upper])
        return lower, upper

    def split_x(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split combined x = [x^{WB}, x^{BB}] into components."""
        x_wb = x[:self.n_x_wb]
        x_bb = x[self.n_x_wb:]
        return x_wb, x_bb

    def combine_x(self, x_wb: np.ndarray, x_bb: np.ndarray) -> np.ndarray:
        """Combine x^{WB} and x^{BB} into x = [x^{WB}, x^{BB}]."""
        return np.concatenate([x_wb, x_bb])

    # =========================================================================
    # Black-box evaluation (for full black-box optimization)
    # =========================================================================

    def evaluate_blackbox(self, x: np.ndarray, count_eval: bool = True) -> float:
        """
        Evaluate objective treating entire problem as black-box.

        Args:
            x: Combined decision vector [x^{WB}, x^{BB}]
            count_eval: Whether to increment evaluation counters

        Returns:
            Objective value J(x^{WB}, f^{BB}(x^{BB}))
        """
        x_wb, x_bb = self.split_x(x)
        y = self.fbb(x_bb)
        obj = self.J(x_wb, y)

        if count_eval:
            self.n_fbb_evals += 1
            self.n_J_evals += 1

        return obj

    def evaluate_blackbox_with_constraints(
        self, x: np.ndarray, count_eval: bool = True
    ) -> Tuple[float, np.ndarray]:
        """
        Evaluate objective and constraints for black-box approach.

        Args:
            x: Combined decision vector [x^{WB}, x^{BB}]
            count_eval: Whether to increment evaluation counters

        Returns:
            Tuple of (objective, constraint_values)
        """
        x_wb, x_bb = self.split_x(x)
        y = self.fbb(x_bb)
        obj = self.J(x_wb, y)

        if self.g is not None:
            cons = self.g(x_wb, y)
        else:
            cons = np.array([])

        if count_eval:
            self.n_fbb_evals += 1
            self.n_J_evals += 1

        return obj, cons

    # =========================================================================
    # Bi-level evaluation
    # =========================================================================

    def evaluate_fbb(self, x_bb: np.ndarray, count_eval: bool = True) -> np.ndarray:
        """
        Evaluate black-box function y = f^{BB}(x^{BB}).

        Args:
            x_bb: Black-box (material/design) variables
            count_eval: Whether to increment counter

        Returns:
            Black-box outputs y
        """
        y = self.fbb(x_bb)
        if count_eval:
            self.n_fbb_evals += 1
        return y

    def evaluate_J(
        self, x_wb: np.ndarray, y: np.ndarray, count_eval: bool = True
    ) -> float:
        """
        Evaluate objective J(x^{WB}, y).

        Args:
            x_wb: White-box (process) variables
            y: Black-box outputs (pre-computed)
            count_eval: Whether to increment counter

        Returns:
            Objective value
        """
        obj = self.J(x_wb, y)
        if count_eval:
            self.n_J_evals += 1
        return obj

    def solve_inner_problem(
        self,
        y: np.ndarray,
        x_wb0: Optional[np.ndarray] = None,
        n_starts: int = 10,
        method: str = 'L-BFGS-B',
        rng: Optional[np.random.Generator] = None
    ) -> OptimizeResult:
        """
        Solve inner optimization problem for given y.

        Inner problem:
            min_{x^{WB}}  J(x^{WB}, y)
            s.t.          g(x^{WB}, y) ≤ 0
                          x_wb_lower ≤ x^{WB} ≤ x_wb_upper

        Args:
            y: Black-box outputs (fixed for inner problem)
            x_wb0: Initial guess for x^{WB} (optional)
            n_starts: Number of multi-start attempts
            method: Optimization method for scipy.optimize.minimize
            rng: Random number generator

        Returns:
            scipy OptimizeResult with optimal x^{WB}* and J(x^{WB}*, y)
        """
        if rng is None:
            rng = np.random.default_rng()

        # For maximization problems, negate objective
        sign = -1.0 if self.maximize else 1.0

        def objective(x_wb):
            return sign * self.J(x_wb, y)

        # Use derivatives if available
        jac = None
        hess = None
        if self.J_grad_x_wb is not None:
            jac = lambda x_wb: sign * self.J_grad_x_wb(x_wb, y)
        if self.J_hess_x_wb is not None:
            hess = lambda x_wb: sign * self.J_hess_x_wb(x_wb, y)

        # Build bounds
        bounds = list(zip(self.x_wb_lower, self.x_wb_upper))

        # Build constraints if present
        constraints = []
        if self.g is not None:
            constraints.append({
                'type': 'ineq',
                'fun': lambda x_wb: -self.g(x_wb, y)  # scipy uses ≥ 0
            })
            if self.g_grad_x_wb is not None:
                constraints[-1]['jac'] = lambda x_wb: -self.g_grad_x_wb(x_wb, y)

        best_res = None
        best_feasible_res = None

        for i in range(n_starts):
            if i == 0 and x_wb0 is not None:
                x0 = x_wb0
            else:
                x0 = rng.uniform(self.x_wb_lower, self.x_wb_upper)

            try:
                if len(constraints) > 0:
                    res = minimize(
                        objective, x0, method='SLSQP',
                        jac=jac, bounds=bounds, constraints=constraints
                    )
                else:
                    res = minimize(
                        objective, x0, method=method,
                        jac=jac, 
                        # hess=hess, 
                        bounds=bounds
                    )

                # Check feasibility for constrained problems
                is_feasible = True
                if self.g is not None:
                    g_vals = self.g(res.x, y)
                    is_feasible = np.all(g_vals <= 1e-6)

                # Track best feasible solution separately
                if is_feasible:
                    if best_feasible_res is None or res.fun < best_feasible_res.fun:
                        best_feasible_res = res

                # Also track overall best (for fallback)
                if best_res is None or res.fun < best_res.fun:
                    best_res = res
            except Exception:
                continue

        # Prefer feasible solution if available
        if best_feasible_res is not None:
            best_res = best_feasible_res
        elif best_res is None:
            # Fallback: return a result at initial point
            x0 = rng.uniform(self.x_wb_lower, self.x_wb_upper)
            best_res = OptimizeResult(
                x=x0, fun=objective(x0), success=False,
                message="All optimization attempts failed"
            )

        # Convert back to original objective value
        best_res.fun = sign * best_res.fun

        return best_res

    def solve_inner_problem_global(
        self,
        y: np.ndarray,
        seed: Optional[int] = None,
        niter: int = 200,
    ) -> OptimizeResult:
        """
        Solve inner WB problem to global optimality using Basin-Hopping.

        Basin-Hopping with SLSQP local minimizer provides global search
        (random perturbation) with precise local convergence (gradient-based
        SLSQP polish). High temperature accepts all uphill moves, and
        full-range stepsize ensures each perturbation can reach any point
        in the domain.

        Args:
            y: Black-box outputs (fixed for inner problem)
            seed: Random seed for reproducibility
            niter: Number of Basin-Hopping iterations (default 200 for inner)

        Returns:
            scipy OptimizeResult with optimal x^{WB}* and J(x^{WB}*, y)
        """
        sign = -1.0 if self.maximize else 1.0
        rng = np.random.default_rng(seed)

        def objective(x_wb):
            return sign * self.J(x_wb, y)

        bounds = list(zip(self.x_wb_lower, self.x_wb_upper))
        has_constraints = self.g is not None and self.n_g > 0

        # SLSQP local minimizer kwargs
        jac = None
        if self.J_grad_x_wb is not None:
            jac = lambda x_wb: sign * self.J_grad_x_wb(x_wb, y)

        minimizer_kwargs = {
            'method': 'SLSQP',
            'jac': jac,
            'bounds': bounds,
            'options': {'maxiter': 500, 'ftol': 1e-14},
        }
        if has_constraints:
            con_dict = {
                'type': 'ineq',
                'fun': lambda x_wb: -self.g(x_wb, y),
            }
            if self.g_grad_x_wb is not None:
                con_dict['jac'] = lambda x_wb: -self.g_grad_x_wb(x_wb, y)
            minimizer_kwargs['constraints'] = [con_dict]

        # BoundedStep: uniform perturbation clipped to variable bounds
        ranges = self.x_wb_upper - self.x_wb_lower
        stepsize = np.max(ranges) / 2.0

        class _BoundedStep:
            def __init__(self, stepsize, lower, upper, rng):
                self.stepsize = stepsize
                self.lower = np.asarray(lower)
                self.upper = np.asarray(upper)
                self.rng = rng

            def __call__(self, x):
                x_new = x + self.rng.uniform(-self.stepsize, self.stepsize,
                                              size=x.shape)
                return np.clip(x_new, self.lower, self.upper)

        x0 = rng.uniform(self.x_wb_lower, self.x_wb_upper)

        bh_result = basinhopping(
            objective, x0,
            minimizer_kwargs=minimizer_kwargs,
            niter=niter,
            T=100.0,
            seed=int(rng.integers(0, 2**31)),
            take_step=_BoundedStep(stepsize, self.x_wb_lower,
                                   self.x_wb_upper, rng),
        )

        best = OptimizeResult(
            x=bh_result.x, fun=sign * bh_result.fun,
            success=True, message='Basin-Hopping'
        )
        return best

    def evaluate_bilevel(
        self,
        x_bb: np.ndarray,
        n_starts: int = 10,
        rng: Optional[np.random.Generator] = None,
        return_x_wb: bool = False,
        inner_solver: str = 'global'
    ) -> float | Tuple[float, np.ndarray]:
        """
        Evaluate bi-level objective: solve inner problem for given x^{BB}.

        This is the outer objective for BO:
            J*(x^{BB}) = min_{x^{WB}} J(x^{WB}, f^{BB}(x^{BB}))

        Args:
            x_bb: Black-box (material/design) variables
            n_starts: Number of multi-start attempts for inner problem
            rng: Random number generator
            return_x_wb: If True, also return optimal x^{WB}*
            inner_solver: 'multistart' for multi-start SLSQP or 'global' for
                          differential evolution

        Returns:
            If return_x_wb=False: optimal objective J*(x^{BB})
            If return_x_wb=True: tuple (J*(x^{BB}), x^{WB}*)
        """
        y = self.evaluate_fbb(x_bb)
        if inner_solver == 'global':
            seed = int(rng.integers(0, 2**31)) if rng is not None else None
            res = self.solve_inner_problem_global(y, seed=seed)
        else:
            res = self.solve_inner_problem(y, n_starts=n_starts, rng=rng)

        if return_x_wb:
            return res.fun, res.x
        return res.fun

    def verify_global_optimum(
        self,
        n_starts: int = 500,
        use_grid_search: bool = False,
        n_grid: int = 8,
        seed: int = 42,
        verbose: bool = True
    ) -> dict:
        """
        Verify global optimum via multi-start NLP optimization.

        Performs exhaustive search over the full problem domain [x_wb, x_bb]
        using Latin Hypercube Sampling for starting points, optionally
        combined with grid search refinement.

        Args:
            n_starts: Number of random starting points for multi-start
            use_grid_search: If True, also perform grid search
            n_grid: Grid density per dimension (if use_grid_search=True)
            seed: Random seed for reproducibility
            verbose: Print progress and results

        Returns:
            Dictionary with:
                - best_x: Best solution found [x_wb, x_bb]
                - best_x_wb: Best white-box variables
                - best_x_bb: Best black-box variables
                - best_y: Black-box outputs at best solution
                - best_J: Best objective value
                - best_g: Constraint values at best solution
                - feasible_solutions: List of (J, x) tuples for feasible solutions
                - comparison: Dict comparing with documented optimal (if available)
        """
        from scipy.stats import qmc

        # Combined bounds
        lower, upper = self.get_full_bounds()
        bounds = list(zip(lower, upper))
        n_dim = len(lower)

        # For maximization, we minimize the negative
        sign = -1.0 if self.maximize else 1.0

        def objective(x):
            """Combined objective: evaluate J(x_wb, f_bb(x_bb))."""
            x_wb, x_bb = self.split_x(x)
            y = self.fbb(x_bb)
            return sign * self.J(x_wb, y)

        def constraint_func(x):
            """Constraint function returning vector (<= 0 is feasible)."""
            if self.g is None:
                return np.array([])
            x_wb, x_bb = self.split_x(x)
            y = self.fbb(x_bb)
            return self.g(x_wb, y)

        # Build scipy constraints
        if self.g is not None:
            constraints = {'type': 'ineq', 'fun': lambda x: -constraint_func(x)}
        else:
            constraints = []

        if verbose:
            print("=" * 70)
            print(f"Global Optimum Verification: {self.name}")
            print("=" * 70)
            print(f"\nProblem structure:")
            print(f"  x_wb dimensions: {self.n_x_wb}, bounds: {self.x_wb_lower} to {self.x_wb_upper}")
            print(f"  x_bb dimensions: {self.n_x_bb}, bounds: {self.x_bb_lower} to {self.x_bb_upper}")
            print(f"  Constraints: {self.n_g}")
            print(f"  Type: {'maximization' if self.maximize else 'minimization'}")
            print(f"\nRunning {n_starts} multi-start optimizations...")

        # Generate starting points using Latin Hypercube Sampling
        sampler = qmc.LatinHypercube(d=n_dim, seed=seed)
        sample = sampler.random(n=n_starts)
        starting_points = qmc.scale(sample, lower, upper)

        # Add documented optimal as starting point if available
        if self.x_wb_optimal is not None and self.x_bb_optimal is not None:
            documented_opt = self.combine_x(self.x_wb_optimal, self.x_bb_optimal)
            starting_points = np.vstack([starting_points, documented_opt])

        # Track results
        best_J = np.inf
        best_x = None
        feasible_solutions = []

        for i, x0 in enumerate(starting_points):
            try:
                result = minimize(
                    objective,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 500, 'ftol': 1e-10}
                )

                # Check feasibility
                g_val = constraint_func(result.x)
                is_feasible = len(g_val) == 0 or np.all(g_val <= 1e-6)

                if is_feasible:
                    if result.fun < best_J:
                        best_J = result.fun
                        best_x = result.x.copy()
                    feasible_solutions.append((result.fun, result.x.copy()))

            except Exception:
                pass

            if verbose and (i + 1) % 100 == 0:
                current_best = sign * best_J if best_x is not None else float('inf')
                print(f"  Completed {i + 1}/{len(starting_points)} starts, best J: {current_best:.6f}")

        # Optional grid search
        if use_grid_search:
            if verbose:
                print(f"\nRunning grid search with {n_grid}^{n_dim} = {n_grid**n_dim} points...")

            # Create grid points
            grid_1d = [np.linspace(lower[i], upper[i], n_grid) for i in range(n_dim)]
            grid_points = np.array(np.meshgrid(*grid_1d)).T.reshape(-1, n_dim)

            # Find promising feasible grid points
            promising = []
            for x in grid_points:
                try:
                    J_val = objective(x)
                    g_val = constraint_func(x)
                    is_feasible = len(g_val) == 0 or np.all(g_val <= 1e-6)
                    if is_feasible:
                        promising.append((J_val, x.copy()))
                except Exception:
                    pass

            # Refine top candidates
            promising.sort(key=lambda t: t[0])
            for _, x0 in promising[:min(100, len(promising))]:
                try:
                    result = minimize(
                        objective,
                        x0,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints,
                        options={'maxiter': 500, 'ftol': 1e-12}
                    )

                    g_val = constraint_func(result.x)
                    is_feasible = len(g_val) == 0 or np.all(g_val <= 1e-6)

                    if is_feasible:
                        if result.fun < best_J:
                            best_J = result.fun
                            best_x = result.x.copy()
                        feasible_solutions.append((result.fun, result.x.copy()))

                except Exception:
                    pass

            if verbose:
                print(f"  Found {len(promising)} feasible grid points")

        # Sort feasible solutions
        feasible_solutions.sort(key=lambda t: t[0])

        # Build results
        results = {
            'best_x': best_x,
            'best_x_wb': None,
            'best_x_bb': None,
            'best_y': None,
            'best_J': None,
            'best_g': None,
            'feasible_solutions': feasible_solutions,
            'comparison': None
        }

        if best_x is not None:
            x_wb_best, x_bb_best = self.split_x(best_x)
            y_best = self.fbb(x_bb_best)
            g_best = constraint_func(best_x)

            # Convert back to original objective
            results['best_J'] = sign * best_J
            results['best_x_wb'] = x_wb_best
            results['best_x_bb'] = x_bb_best
            results['best_y'] = y_best
            results['best_g'] = g_best

            if verbose:
                print("\n" + "=" * 70)
                print("RESULTS")
                print("=" * 70)
                print(f"\nBest solution found:")
                print(f"  x_wb = {x_wb_best}")
                print(f"  x_bb = {x_bb_best}")
                print(f"  y    = {y_best}")
                print(f"  J*   = {results['best_J']:.6f}")
                if len(g_best) > 0:
                    print(f"  g    = {g_best} (should be <= 0)")

            # Compare with documented optimal if available
            if self.J_optimal is not None:
                comparison = {
                    'documented_J': self.J_optimal,
                    'found_J': results['best_J'],
                    'J_diff': abs(results['best_J'] - self.J_optimal),
                    'J_rel_diff': abs(results['best_J'] - self.J_optimal) / (abs(self.J_optimal) + 1e-10)
                }

                if self.x_wb_optimal is not None:
                    comparison['x_wb_diff'] = np.linalg.norm(x_wb_best - self.x_wb_optimal)
                if self.x_bb_optimal is not None:
                    comparison['x_bb_diff'] = np.linalg.norm(x_bb_best - self.x_bb_optimal)

                results['comparison'] = comparison

                if verbose:
                    print(f"\n  Documented optimal:")
                    if self.x_wb_optimal is not None:
                        print(f"    x_wb_optimal = {self.x_wb_optimal}")
                    if self.x_bb_optimal is not None:
                        print(f"    x_bb_optimal = {self.x_bb_optimal}")
                    print(f"    J_optimal = {self.J_optimal}")

                    print(f"\n  Difference from documented optimal:")
                    print(f"    |J - J_optimal| = {comparison['J_diff']:.6f}")
                    print(f"    Relative diff   = {comparison['J_rel_diff']*100:.4f}%")
                    if 'x_wb_diff' in comparison:
                        print(f"    ||x_wb - x_wb_optimal|| = {comparison['x_wb_diff']:.6f}")
                    if 'x_bb_diff' in comparison:
                        print(f"    ||x_bb - x_bb_optimal|| = {comparison['x_bb_diff']:.6f}")

            # Show top distinct feasible solutions
            if verbose and len(feasible_solutions) > 1:
                print(f"\n  Top 5 distinct feasible solutions (out of {len(feasible_solutions)}):")
                shown = []
                for J_val, x_val in feasible_solutions[:20]:
                    is_distinct = all(np.linalg.norm(x_val - s) > 0.1 for s in shown)
                    if is_distinct:
                        shown.append(x_val)
                        x_wb_i, x_bb_i = self.split_x(x_val)
                        J_display = sign * J_val
                        print(f"    J={J_display:.6f}: x_wb={np.round(x_wb_i, 3)}, x_bb={np.round(x_bb_i, 3)}")
                    if len(shown) >= 5:
                        break

        else:
            if verbose:
                print("\nNo feasible solution found!")

        if verbose:
            print("\n" + "=" * 70)

        return results


# =============================================================================
# Small Feasible Region Problem 1
# =============================================================================
#
# From Gardner et al. (2014) and Ariafar et al. (2019):
#
#   min   J(x) = sin(x1) + y_1
#   s.t.  g1(x) = sin(x1)*sin(y_1) + 0.95 ≤ 0
#         0 ≤ xi ≤ 6, i=1,2
#
#   Bi-level formulation:
#     x^{WB} = [x1]  - white-box (process) variable
#     x^{BB} = [x2]  - black-box (material/design) variable
#     y = [x2]       - black-box output
#
# =============================================================================

def _small_feasible_region2_fbb(x_bb: np.ndarray) -> np.ndarray:
    """Black-box function y = f^{BB}(x^{BB}) for Small Feasible Region 2."""
    x1 = x_bb[0]
    return np.array([x1])


def _small_feasible_region2_J(x_wb: np.ndarray, y: np.ndarray) -> float:
    """Objective J(x^{WB}, y) for Small Feasible Region 2."""
    x1 = x_wb[0]
    y1 = y[0]
    return np.sin(x1) + y1


def _small_feasible_region2_g(x_wb: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Constraints g(x^{WB}, y) for Small Feasible Region 2 (≤ 0 is feasible)."""
    x1 = x_wb[0]
    y1 = y[0]
    g1 = np.sin(x1) * np.sin(y1) + 0.95
    return np.array([g1])


def _small_feasible_region2_J_grad_x_wb(x_wb: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradient of J w.r.t. x^{WB} for Small Feasible Region 2."""
    return np.cos(x_wb[0:1])


def _small_feasible_region2_g_grad_x_wb(x_wb: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Jacobian of g w.r.t. x^{WB} for Small Feasible Region 2."""
    x1 = x_wb[0]
    y1 = y[0]
    dg1_dx1 = np.cos(x1) * np.sin(y1)
    return np.array([[dg1_dx1]])


def create_small_feasible_region() -> BiLevelProblem:
    """
    Create Small Feasible Region 2 bi-level optimization problem.

    This is a constrained minimization problem with a small feasible region
    from Gardner et al. (2014) and Ariafar et al. (2019).

    Bi-level formulation:
        x^{WB} = [x1]  - white-box (process) variable
        x^{BB} = [x2]  - black-box (material/design) variable
        y = [x2]       - black-box output
    """
    return BiLevelProblem(
        name="Small-Feasible-Region-1",
        n_x_wb=1,
        n_x_bb=1,
        n_y=1,
        n_g=1,
        x_wb_lower=np.array([0.0]),
        x_wb_upper=np.array([6.0]),
        x_bb_lower=np.array([0.0]),
        x_bb_upper=np.array([6.0]),
        fbb=_small_feasible_region2_fbb,
        J=_small_feasible_region2_J,
        g=_small_feasible_region2_g,
        J_grad_x_wb=_small_feasible_region2_J_grad_x_wb,
        g_grad_x_wb=_small_feasible_region2_g_grad_x_wb,
        x_wb_optimal=np.array([4.712388975429799]),
        x_bb_optimal=np.array([1.25323589564352]),
        J_optimal=0.253235895643520,
        maximize=False
    )


# =============================================================================
# Small Feasible Region Problem (Gardner et al.)
# =============================================================================
#
# From Gardner et al. (2014) and Ariafar et al. (2019):
#
#   min   J(x) = cos(2*x1)*cos(x2) + sin(x1)
#   s.t.  g1(x) = cos(x1)*cos(x2) - sin(x1)*sin(x2) - 0.5 ≤ 0
#         0 ≤ xi ≤ 6, i=1,2
#
#   Note: This is not a gray-box problem in the original formulation.
#   We reformulate it as bi-level by treating x2 as the black-box variable
#   that produces a parameter y used in the white-box optimization.
#
#   Bi-level formulation:
#     x^{WB} = [x1]  - white-box (process) variable
#     x^{BB} = [x2]  - black-box (material/design) variable
#     y = (x2-5)/2 * exp(x2/2)       - black-box output (identity for this problem)
#
# =============================================================================

def _small_feasible_region_fbb(x_bb: np.ndarray) -> np.ndarray:
    """Black-box function y = f^{BB}(x^{BB}) for Small Feasible Region."""
    x2 = (x_bb[0]-5.5) / 2 * np.exp(x_bb[0]/2)
    return np.array([x2])


def _small_feasible_region_J(x_wb: np.ndarray, y: np.ndarray) -> float:
    """Objective J(x^{WB}, y) for Small Feasible Region."""
    x1 = x_wb[0]
    y1 = y[0]
    return np.cos(2 * x1) * np.cos(y1) + np.sin(x1)


def _small_feasible_region_g(x_wb: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Constraints g(x^{WB}, y) for Small Feasible Region (≤ 0 is feasible)."""
    x1 = x_wb[0]
    y1 = y[0]
    g1 = np.cos(x1) * np.cos(y1) - np.sin(x1) * np.sin(y1) - 0.5
    return np.array([g1])


def _small_feasible_region_J_grad_x_wb(x_wb: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradient of J w.r.t. x^{WB} for Small Feasible Region."""
    x1 = x_wb[0]
    y1 = y[0]
    dJ_dx1 = -2 * np.sin(2 * x1) * np.cos(y1) + np.cos(x1)
    return np.array([dJ_dx1])


def _small_feasible_region_g_grad_x_wb(x_wb: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Jacobian of g w.r.t. x^{WB} for Small Feasible Region."""
    x1 = x_wb[0]
    y1 = y[0]
    dg1_dx1 = -np.sin(x1) * np.cos(y1) - np.cos(x1) * np.sin(y1)
    return np.array([[dg1_dx1]])


def create_small_feasible_region2() -> BiLevelProblem:
    """
    Create Small Feasible Region bi-level optimization problem.

    This is a constrained minimization problem with a small feasible region
    from Gardner et al. (2014) and Ariafar et al. (2019).

    Bi-level formulation:
        x^{WB} = [x1]  - white-box (process) variable
        x^{BB} = [x2]  - black-box (material/design) variable
        y = [x2]       - black-box output
    """
    return BiLevelProblem(
        name="Small-Feasible-Region-2",
        n_x_wb=1,
        n_x_bb=1,
        n_y=1,
        n_g=1,
        x_wb_lower=np.array([0.0]),
        x_wb_upper=np.array([6.0]),
        x_bb_lower=np.array([0.0]),
        x_bb_upper=np.array([6.0]),
        fbb=_small_feasible_region_fbb,
        J=_small_feasible_region_J,
        g=_small_feasible_region_g,
        J_grad_x_wb=_small_feasible_region_J_grad_x_wb,
        g_grad_x_wb=_small_feasible_region_g_grad_x_wb,
        x_wb_optimal=np.array([3 * np.pi / 2]),
        x_bb_optimal=np.array([5.5]),
        J_optimal=-2.0,
        maximize=False
    )



# =============================================================================
# Rastrigin Problem
# =============================================================================
#
#   min   J(x^{WB}, y) = 30 + x1^2 - 10*cos(2π*x1) + x2^2 - 10*cos(2π*x2) + y
#   s.t.  (no constraints)
#         y = f^{BB}(x^{BB}) = x3^2 - 10*cos(2π*x3)
#         -5.12 ≤ xi ≤ 5.12, i=1,2,3
#
#   x^{WB} = [x1, x2] - white-box (process) variables
#   x^{BB} = [x3]     - black-box (material/design) variable
#   y      = [y1]     - black-box output
#
#   Optimal: x* = [0, 0, 0], J* = 0
# =============================================================================

def _rastrigin_fbb(x_bb: np.ndarray) -> np.ndarray:
    """Black-box function y = f^{BB}(x^{BB}) for Rastrigin."""
    x3 = x_bb[0]
    y = x3**2 - 10 * np.cos(2 * np.pi * x3)
    return np.array([y])


def _rastrigin_J(x_wb: np.ndarray, y: np.ndarray) -> float:
    """Objective J(x^{WB}, y) for Rastrigin."""
    x1, x2 = x_wb[0], x_wb[1]
    return (30 + x1**2 - 10 * np.cos(2 * np.pi * x1)
            + x2**2 - 10 * np.cos(2 * np.pi * x2) + y[0])


def _rastrigin_J_grad_x_wb(x_wb: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradient of J w.r.t. x^{WB} for Rastrigin."""
    x1, x2 = x_wb[0], x_wb[1]
    dJ_dx1 = 2 * x1 + 20 * np.pi * np.sin(2 * np.pi * x1)
    dJ_dx2 = 2 * x2 + 20 * np.pi * np.sin(2 * np.pi * x2)
    return np.array([dJ_dx1, dJ_dx2])


def _rastrigin_J_hess_x_wb(x_wb: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Hessian of J w.r.t. x^{WB} for Rastrigin."""
    x1, x2 = x_wb[0], x_wb[1]
    d2J_dx1 = 2 + 40 * np.pi**2 * np.cos(2 * np.pi * x1)
    d2J_dx2 = 2 + 40 * np.pi**2 * np.cos(2 * np.pi * x2)
    return np.array([[d2J_dx1, 0], [0, d2J_dx2]])


def create_rastrigin() -> BiLevelProblem:
    """Create Rastrigin bi-level optimization problem."""
    bounds = np.array([-5.12, 5.12])
    return BiLevelProblem(
        name="Rastrigin",
        n_x_wb=2,
        n_x_bb=1,
        n_y=1,
        n_g=0,
        x_wb_lower=np.array([bounds[0], bounds[0]]),
        x_wb_upper=np.array([bounds[1], bounds[1]]),
        x_bb_lower=np.array([bounds[0]]),
        x_bb_upper=np.array([bounds[1]]),
        fbb=_rastrigin_fbb,
        J=_rastrigin_J,
        g=None,
        J_grad_x_wb=_rastrigin_J_grad_x_wb,
        J_hess_x_wb=_rastrigin_J_hess_x_wb,
        x_wb_optimal=np.array([0.0, 0.0]),
        x_bb_optimal=np.array([0.0]),
        J_optimal=0.0,
        maximize=False
    )


# =============================================================================
# Toy-Hydrology Problem (2D, 2 constraints)
# =============================================================================
#
# From COBALT paper Appendix A.4:
#
#   min   J(x^{WB}, y) = x1 + x2
#   s.t.  g1(x, y) = 1.5 - x1 - 2*x2 - 0.5*sin(-4π*x2 + y1) ≤ 0
#         g2(x, y) = x1^2 + x2^2 - 1.5 ≤ 0
#         y1 = d1(z) = 2π*z1^2
#         z1 = x1
#         0 ≤ xi ≤ 1, i ∈ {1, 2}
#
#   Bi-level formulation (as specified):
#     x^{BB} = [x1]    - black-box (material/design) variable (optimized by BO)
#     x^{WB} = [x2]    - white-box (process) variable (optimized in inner problem)
#     y = [y1]         - black-box output
#
#   Optimal: x* = [0.1951, 0.4047], J* = 0.5998
# =============================================================================

def _toy_hydrology_fbb(x_bb: np.ndarray) -> np.ndarray:
    """
    Black-box function y = f^{BB}(x^{BB}) for Toy-Hydrology.

    y1 = 2π * x1^2

    Args:
        x_bb: [x1] - black-box (material) variable

    Returns:
        y = [y1]
    """
    x1 = x_bb[0]
    y1 = 2 * np.pi * x1**2
    return np.array([y1, x1])  # Include x1 for use in objective and constraints


def _toy_hydrology_J(x_wb: np.ndarray, y: np.ndarray) -> float:
    """
    Objective J(x^{WB}, y) for Toy-Hydrology.

    J = x1 + x2

    Args:
        x_wb: [x2] - white-box (process) variable
        y: [y1, x1] - black-box output and x1 passed through

    Returns:
        Objective value
    """
    x2 = x_wb[0]
    x1 = y[1]  # x1 passed through from fbb
    return x1 + x2


def _toy_hydrology_g(x_wb: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Constraints g(x^{WB}, y) for Toy-Hydrology (≤ 0 is feasible).

    g1 = 1.5 - x1 - 2*x2 - 0.5*sin(-4π*x2 + y1) ≤ 0
    g2 = x1^2 + x2^2 - 1.5 ≤ 0

    Args:
        x_wb: [x2] - white-box (process) variable
        y: [y1, x1] - black-box output and x1 passed through

    Returns:
        Constraint values (≤ 0 is feasible)
    """
    x2 = x_wb[0]
    y1 = y[0]
    x1 = y[1]

    g1 = 1.5 - x1 - 2*x2 - 0.5*np.sin(-4*np.pi*x2 + y1)
    g2 = x1**2 + x2**2 - 1.5

    return np.array([g1, g2])


def _toy_hydrology_J_grad_x_wb(x_wb: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Gradient of J w.r.t. x^{WB} for Toy-Hydrology.

    dJ/dx2 = 1
    """
    return np.array([1.0])


def _toy_hydrology_g_grad_x_wb(x_wb: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Jacobian of g w.r.t. x^{WB} for Toy-Hydrology.

    dg1/dx2 = -2 - 0.5*(-4π)*cos(-4π*x2 + y1) = -2 + 2π*cos(-4π*x2 + y1)
    dg2/dx2 = 2*x2

    Returns:
        Jacobian matrix of shape (2, 1)
    """
    x2 = x_wb[0]
    y1 = y[0]

    dg1_dx2 = -2 + 2*np.pi*np.cos(-4*np.pi*x2 + y1)
    dg2_dx2 = 2*x2

    return np.array([[dg1_dx2], [dg2_dx2]])


def create_toy_hydrology() -> BiLevelProblem:
    """
    Create Toy-Hydrology bi-level optimization problem.

    This is a constrained minimization problem where:
        x^{BB} = [x1]  - black-box (material) variable (optimized by BO)
        x^{WB} = [x2]  - white-box (process) variable (optimized in inner problem)
        y = [y1, x1]   - black-box output (y1 = 2π*x1^2) plus x1 for constraints

    Constraints:
        g1: 1.5 - x1 - 2*x2 - 0.5*sin(-4π*x2 + y1) ≤ 0
        g2: x1^2 + x2^2 - 1.5 ≤ 0

    Optimal: x* = [0.1951, 0.4047], J* = 0.5998
    """
    return BiLevelProblem(
        name="Toy-Hydrology",
        n_x_wb=1,
        n_x_bb=1,
        n_y=2,  # [y1, x1]
        n_g=2,
        x_wb_lower=np.array([0.0]),
        x_wb_upper=np.array([1.0]),
        x_bb_lower=np.array([0.0]),
        x_bb_upper=np.array([1.0]),
        fbb=_toy_hydrology_fbb,
        J=_toy_hydrology_J,
        g=_toy_hydrology_g,
        J_grad_x_wb=_toy_hydrology_J_grad_x_wb,
        g_grad_x_wb=_toy_hydrology_g_grad_x_wb,
        x_wb_optimal=np.array([0.404665360529484]),
        x_bb_optimal=np.array([0.195122675064254]),
        J_optimal=0.599788035593738,
        maximize=False
    )


# =============================================================================
# Rosen-Suzuki Problem (4D, 3 constraints)
# =============================================================================
#
# From COBALT paper Appendix A.5:
#
#   min   J(x^{WB}, y) = x1^2 + x2^2 + x4^2 - 5*x1 - 5*x2 + y1
#   s.t.  g1(x,y) = -(8 - x1^2 - x2^2 - x3^2 - x4^2 - x1 + x2 - x3 + x4) ≤ 0
#         g2(x,y) = -(10 - x1^2 - 2*x2^2 - y2 + x1 + x4) ≤ 0
#         g3(x,y) = -(5 - 2*x1^2 - x2^2 - x3^2 - 2*x1 + x2 + x4) ≤ 0
#         y1 = d1(z) = 2*z1^2 - 21*z1 + 7*z2
#         y2 = d2(z) = z1^2 + 2*z2^2
#         z1 = x3, z2 = x4
#         -2 ≤ xi ≤ 2, i=1,...,4
#
#   Bi-level formulation:
#     x^{WB} = [x1, x2]  - white-box (process) variables (optimized in inner problem)
#     x^{BB} = [x3, x4]  - black-box (material/design) variables (optimized by BO)
#     y = [y1, y2]       - black-box outputs
#
#   Optimal: x* = [0, 1, 2, -1], J* = -44
# =============================================================================

def _rosen_suzuki_fbb_extended(x_bb: np.ndarray) -> np.ndarray:
    """
    Extended black-box function y = f^{BB}(x^{BB}) that also passes x3, x4 for use in constraints.

    y = [y1, y2, x3, x4] where:
        y1 = 2*x3^2 - 21*x3 + 7*x4
        y2 = x3^2 + 2*x4^2
        x3 = x_bb[0]
        x4 = x_bb[1]
    """
    x3, x4 = x_bb[0], x_bb[1]
    y1 = 2 * x3**2 - 21 * x3 + 7 * x4
    y2 = x3**2 + 2 * x4**2
    return np.array([y1, y2, x3, x4])


def _rosen_suzuki_J_extended(x_wb: np.ndarray, y: np.ndarray) -> float:
    """Objective J(x^{WB}, y) for Rosen-Suzuki with extended y."""
    x1, x2 = x_wb[0], x_wb[1]
    y1 = y[0]
    x4 = y[3]  # x4 passed through

    return x1**2 + x2**2 + x4**2 - 5*x1 - 5*x2 + y1


def _rosen_suzuki_g(x_wb: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Constraints g(x^{WB}, y) for Rosen-Suzuki (≤ 0 is feasible).

    g1 = -(8 - x1^2 - x2^2 - x3^2 - x4^2 - x1 + x2 - x3 + x4) ≤ 0
    g2 = -(10 - x1^2 - 2*x2^2 - y2 + x1 + x4) ≤ 0
    g3 = -(5 - 2*x1^2 - x2^2 - x3^2 - 2*x1 + x2 + x4) ≤ 0

    y = [y1, y2, x3, x4]
    """
    x1, x2 = x_wb[0], x_wb[1]
    y2 = y[1]
    x3 = y[2]
    x4 = y[3]

    g1 = -(8 - x1**2 - x2**2 - x3**2 - x4**2 - x1 + x2 - x3 + x4)
    g2 = -(10 - x1**2 - 2*x2**2 - y2 + x1 + x4)
    g3 = -(5 - 2*x1**2 - x2**2 - x3**2 - 2*x1 + x2 + x4)

    return np.array([g1, g2, g3])


def _rosen_suzuki_J_grad_x_wb(x_wb: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradient of J w.r.t. x^{WB} for Rosen-Suzuki."""
    x1, x2 = x_wb[0], x_wb[1]
    dJ_dx1 = 2*x1 - 5
    dJ_dx2 = 2*x2 - 5
    return np.array([dJ_dx1, dJ_dx2])


def create_rosen_suzuki() -> BiLevelProblem:
    """
    Create Rosen-Suzuki bi-level optimization problem.

    This is a constrained problem where:
        x^{WB} = [x1, x2]  - white-box (process) variables
        x^{BB} = [x3, x4]  - black-box (material/design) variables
        y = [y1, y2, x3, x4]  - extended black-box outputs (includes x3, x4 for constraints)

    Optimal: x* = [0, 1, 2, -1], J* = -44
    """
    return BiLevelProblem(
        name="Rosen-Suzuki",
        n_x_wb=2,
        n_x_bb=2,
        n_y=4,  # [y1, y2, x3, x4]
        n_g=3,
        x_wb_lower=np.array([-2.0, -2.0]),
        x_wb_upper=np.array([2.0, 2.0]),
        x_bb_lower=np.array([-2.0, -2.0]),
        x_bb_upper=np.array([2.0, 2.0]),
        fbb=_rosen_suzuki_fbb_extended,
        J=_rosen_suzuki_J_extended,
        g=_rosen_suzuki_g,
        J_grad_x_wb=_rosen_suzuki_J_grad_x_wb,
        x_wb_optimal=np.array([0.0, 1.0]),
        x_bb_optimal=np.array([2.0, -1.0]),
        J_optimal=-44.0,
        maximize=False
    )


# =============================================================================
# CSTR Problem (Continuous Stirred Tank Reactor)
# =============================================================================
#
# From main.tex and MATLAB reference (BO_full_new_LHS_only_EI_only.m):
#
# Consecutive reactions A -> B -> C in a CSTR (feed = pure A).
# Objective: MAXIMIZE Yield of intermediate product B.
#
# White-box model (process equations):
#   Y_B = (k1 * tau) / ((1 + k1*tau) * (1 + k2*tau))
#   k1 = A1 * volc * exp(-Ea1/(R*T)) * (P/Pref)^beta1 / (1 + Kads*P)^2
#   k2 = A2 * (1 - alpha*volc) * exp(-Ea2/(R*T)) * (P/Pref)^beta2 / (1 + Kads*P)^2
#
# Black-box model (catalyst/DFT outputs):
#   Ea1 = 80000 - 20000 * dEa  (J/mol)
#   Ea2 = 95000 + 5000 * dEa   (J/mol)
#   volc = exp(-0.5 * ((Eb - Eb0) / sigmaE)^2)  -- volcano activity
#
# White-box adsorption model (computed in J, g):
#   Kads = K0 * exp(kappa * (Eb - Eb0))
#
# Constraints:
#   g1: Y_C - 0.25 <= 0  (at most 25% to byproduct C)
#   g2: P/Pmax - (T - Tmin)/(Tmax - Tmin) - 0.3 <= 0  (safety)
#
# Variables:
#   x^{WB} = [T, P, tau]  - process variables (temperature, pressure, residence time)
#   x^{BB} = [Eb, dEa]    - catalyst variables (binding energy, activation shift)
#
# Optimal: T=647.25K, P=20bar, tau=5s, Eb=-1.047eV, dEa=0.2, J*=92.32%
# =============================================================================

@dataclass
class CSTRParameters:
    """Physical constants and model parameters for the CSTR system.

    Parameters match main.tex Table 1 and MATLAB reference code.
    """
    # Gas constant
    R: float = 8.314           # J/(mol·K)

    # Pre-exponential factors
    A1: float = 1e7            # s^-1 (reaction 1: A -> B)
    A2: float = 3e6            # s^-1 (reaction 2: B -> C)

    # Reference pressure
    Pref: float = 8.0          # bar

    # Pressure exponents
    beta1: float = 1.0         # reaction 1
    beta2: float = 0.35        # reaction 2

    # Volcano model parameters
    Eb0: float = -1.0          # optimal binding energy (eV)
    sigmaE: float = 0.40       # volcano width (eV)
    alpha: float = 0.6         # reduces side reaction near volcano peak

    # Adsorption parameters
    K0: float = 0.06           # adsorption pre-factor (1/bar)
    kappa: float = 3.0         # adsorption sensitivity (1/eV)

    # Base activation energies (J/mol)
    Ea1_base: float = 80000.0  # reaction 1
    Ea2_base: float = 95000.0  # reaction 2

    # Activation energy shift coefficients
    alpha1: float = 20000.0    # Ea1 = Ea1_base - alpha1 * dEa
    alpha2: float = 5000.0     # Ea2 = Ea2_base + alpha2 * dEa


# Global CSTR parameters instance
_CSTR_PARAMS = CSTRParameters()


def _cstr_fbb(x_bb: np.ndarray) -> np.ndarray:
    """
    Black-box function y = f^{BB}(x^{BB}) for CSTR.

    Maps catalyst properties (Eb, dEa) to kinetic parameters that would
    be obtained from DFT calculations or experimental measurements.

    Args:
        x_bb: [Eb, dEa] - binding energy (eV) and activation energy shift

    Returns:
        y = [Ea1, Ea2, volc, Eb] - catalyst-derived parameters
            Ea1: activation energy for reaction 1 (J/mol)
            Ea2: activation energy for reaction 2 (J/mol)
            volc: volcano activity factor (0 to 1)
            Eb: binding energy passed through for Kads computation in white-box
    """
    p = _CSTR_PARAMS
    Eb, dEa = x_bb[0], x_bb[1]

    # Activation energies (Arrhenius barrier modulated by catalyst)
    Ea1 = p.Ea1_base - p.alpha1 * dEa  # 80000 - 20000*dEa
    Ea2 = p.Ea2_base + p.alpha2 * dEa  # 95000 + 5000*dEa

    # Volcano activity model (Sabatier-type relationship)
    # Maximum activity at Eb = Eb0, decreases away from optimal binding
    volc = np.exp(-0.5 * ((Eb - p.Eb0) / p.sigmaE) ** 2)

    return np.array([Ea1, Ea2, volc, Eb])


def _cstr_J(x_wb: np.ndarray, y: np.ndarray) -> float:
    """
    Objective J(x^{WB}, y) for CSTR: yield of B as percentage.

    Computes yield using the rate expressions from MATLAB reference.
    Key: k2 has (1 - alpha*volc) term that reduces side reaction at volcano peak.

    Args:
        x_wb: [T, P, tau] - temperature (K), pressure (bar), residence time (s)
        y: [Ea1, Ea2, volc, Eb] - from black-box function

    Returns:
        Yield of B (0-100%)
    """
    p = _CSTR_PARAMS
    T, P, tau = x_wb[0], x_wb[1], x_wb[2]
    Ea1, Ea2, volc, Eb = y[0], y[1], y[2], y[3]

    # Adsorption equilibrium constant (white-box Langmuir model)
    Kads = p.K0 * np.exp(p.kappa * (Eb - p.Eb0))

    # Arrhenius temperature factors
    phi1 = np.exp(-Ea1 / (p.R * T))
    phi2 = np.exp(-Ea2 / (p.R * T))

    # Pressure factors (power law)
    Pf1 = (P / p.Pref) ** p.beta1
    Pf2 = (P / p.Pref) ** p.beta2

    # Langmuir-Hinshelwood denominator (competitive adsorption inhibition)
    dL = (1 + Kads * P) ** 2

    # Rate constants (from MATLAB reference)
    # k1: forward reaction A -> B (enhanced by volcano activity)
    # k2: side reaction B -> C (reduced near volcano peak by alpha factor)
    k1 = p.A1 * volc * phi1 * Pf1 / dL
    k2 = p.A2 * (1 - p.alpha * volc) * phi2 * Pf2 / dL

    # Yield of B for consecutive reactions A -> B -> C in CSTR
    # Derived from steady-state mass balances
    Y_B = (k1 * tau) / ((1 + k1 * tau) * (1 + k2 * tau))

    return 100.0 * Y_B


def _cstr_g(x_wb: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Constraints g(x^{WB}, y) for CSTR (≤ 0 is feasible).

    Constraints:
        g1: Y_C - 0.25 <= 0 (at most 25% to byproduct C)
        g2: P/Pmax - (T-Tmin)/(Tmax-Tmin) - 0.3 <= 0 (pressure-temperature safety)

    Args:
        x_wb: [T, P, tau] - temperature (K), pressure (bar), residence time (s)
        y: [Ea1, Ea2, volc, Eb] - from black-box function

    Returns:
        Constraint values [g1, g2] (≤ 0 is feasible)
    """
    p = _CSTR_PARAMS
    T, P, tau = x_wb[0], x_wb[1], x_wb[2]
    Ea1, Ea2, volc, Eb = y[0], y[1], y[2], y[3]

    # Adsorption equilibrium constant (white-box Langmuir model)
    Kads = p.K0 * np.exp(p.kappa * (Eb - p.Eb0))

    # Rate constants (same formulas as _cstr_J)
    phi1 = np.exp(-Ea1 / (p.R * T))
    phi2 = np.exp(-Ea2 / (p.R * T))
    Pf1 = (P / p.Pref) ** p.beta1
    Pf2 = (P / p.Pref) ** p.beta2
    dL = (1 + Kads * P) ** 2

    k1 = p.A1 * volc * phi1 * Pf1 / dL
    k2 = p.A2 * (1 - p.alpha * volc) * phi2 * Pf2 / dL

    # Conversion of A: X_A = k1*tau / (1 + k1*tau)
    X_A = (k1 * tau) / (1 + k1 * tau)

    # Yield of B: Y_B = k1*tau / ((1 + k1*tau)*(1 + k2*tau))
    Y_B = (k1 * tau) / ((1 + k1 * tau) * (1 + k2 * tau))

    # Yield of C: Y_C = X_A - Y_B (converted A that went beyond B to C)
    Y_C = X_A - Y_B

    # Bounds
    T_min, T_max = 450.0, 700.0
    P_max = 20.0

    # g1: Maximum byproduct constraint
    g1 = Y_C - 0.25

    # g2: Pressure-temperature safety constraint
    # Higher pressure requires higher temperature for safe operation
    g2 = P / P_max - (T - T_min) / (T_max - T_min) - 0.3

    return np.array([g1, g2])


def _cstr_g_grad_x_wb(x_wb: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Jacobian of g w.r.t. x^{WB} = [T, P, tau] for CSTR.

    Returns:
        Jacobian matrix of shape (2, 3) where J[i,j] = dg_i/dx_j
    """
    p = _CSTR_PARAMS
    T, P, tau = x_wb[0], x_wb[1], x_wb[2]
    Ea1, Ea2, volc, Eb = y[0], y[1], y[2], y[3]

    # Adsorption equilibrium constant (white-box Langmuir model)
    Kads = p.K0 * np.exp(p.kappa * (Eb - p.Eb0))

    # Forward pass for rate constants
    phi1 = np.exp(-Ea1 / (p.R * T))
    phi2 = np.exp(-Ea2 / (p.R * T))
    Pf1 = (P / p.Pref) ** p.beta1
    Pf2 = (P / p.Pref) ** p.beta2
    dL = (1 + Kads * P) ** 2

    k1 = p.A1 * volc * phi1 * Pf1 / dL
    k2 = p.A2 * (1 - p.alpha * volc) * phi2 * Pf2 / dL

    # Derivatives of phi1, phi2 w.r.t. T
    dphi1_dT = phi1 * Ea1 / (p.R * T**2)
    dphi2_dT = phi2 * Ea2 / (p.R * T**2)

    # Derivatives of Pf1, Pf2 w.r.t. P
    dPf1_dP = p.beta1 * Pf1 / P
    dPf2_dP = p.beta2 * Pf2 / P

    # Derivative of dL w.r.t. P
    ddL_dP = 2 * (1 + Kads * P) * Kads

    # Derivatives of k1 w.r.t. T, P
    dk1_dT = p.A1 * volc * dphi1_dT * Pf1 / dL
    dk1_dP = p.A1 * volc * phi1 * (dPf1_dP * dL - Pf1 * ddL_dP) / (dL**2)

    # Derivatives of k2 w.r.t. T, P (note the (1 - alpha*volc) factor)
    factor2 = 1 - p.alpha * volc
    dk2_dT = p.A2 * factor2 * dphi2_dT * Pf2 / dL
    dk2_dP = p.A2 * factor2 * phi2 * (dPf2_dP * dL - Pf2 * ddL_dP) / (dL**2)

    # Intermediate quantities
    d1 = 1 + k1 * tau
    d2 = 1 + k2 * tau
    denom = d1 * d2

    # Derivatives of X_A = k1*tau / d1
    dXA_dk1 = tau / (d1**2)
    dXA_dk2 = 0.0
    dXA_dtau = k1 / (d1**2)

    # Derivatives of Y_B = k1*tau / (d1 * d2)
    dYB_dk1 = tau * d2 / (denom**2)
    dYB_dk2 = -k1 * tau * d1 * tau / (denom**2)
    dYB_dtau = (k1 * denom - k1 * tau * (k1 * d2 + d1 * k2)) / (denom**2)

    # Derivatives of Y_C = X_A - Y_B
    dYC_dk1 = dXA_dk1 - dYB_dk1
    dYC_dk2 = dXA_dk2 - dYB_dk2
    dYC_dtau = dXA_dtau - dYB_dtau

    # Chain rule for g1 = Y_C - 0.25
    dg1_dT = dYC_dk1 * dk1_dT + dYC_dk2 * dk2_dT
    dg1_dP = dYC_dk1 * dk1_dP + dYC_dk2 * dk2_dP
    dg1_dtau = dYC_dtau

    # g2 = P/P_max - (T - T_min)/(T_max - T_min) - 0.3
    T_min, T_max = 450.0, 700.0
    P_max = 20.0
    dg2_dT = -1.0 / (T_max - T_min)
    dg2_dP = 1.0 / P_max
    dg2_dtau = 0.0

    return np.array([
        [dg1_dT, dg1_dP, dg1_dtau],
        [dg2_dT, dg2_dP, dg2_dtau]
    ])


def _cstr_J_grad_x_wb(x_wb: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Gradient of J w.r.t. x^{WB} = [T, P, tau] for CSTR.

    Uses analytical derivatives for efficient inner optimization.
    """
    p = _CSTR_PARAMS
    T, P, tau = x_wb[0], x_wb[1], x_wb[2]
    Ea1, Ea2, volc, Eb = y[0], y[1], y[2], y[3]

    # Adsorption equilibrium constant (white-box Langmuir model)
    Kads = p.K0 * np.exp(p.kappa * (Eb - p.Eb0))

    # Forward pass
    phi1 = np.exp(-Ea1 / (p.R * T))
    phi2 = np.exp(-Ea2 / (p.R * T))
    Pf1 = (P / p.Pref) ** p.beta1
    Pf2 = (P / p.Pref) ** p.beta2
    dL = (1 + Kads * P) ** 2

    k1 = p.A1 * volc * phi1 * Pf1 / dL
    k2 = p.A2 * (1 - p.alpha * volc) * phi2 * Pf2 / dL

    # Yield components
    d1 = 1 + k1 * tau
    d2 = 1 + k2 * tau
    denom = d1 * d2

    # Derivatives of phi, Pf, dL
    dphi1_dT = phi1 * Ea1 / (p.R * T**2)
    dphi2_dT = phi2 * Ea2 / (p.R * T**2)
    dPf1_dP = p.beta1 * Pf1 / P
    dPf2_dP = p.beta2 * Pf2 / P
    ddL_dP = 2 * (1 + Kads * P) * Kads

    # Derivatives of k1 w.r.t. T, P
    dk1_dT = p.A1 * volc * dphi1_dT * Pf1 / dL
    dk1_dP = p.A1 * volc * phi1 * (dPf1_dP * dL - Pf1 * ddL_dP) / (dL**2)

    # Derivatives of k2 w.r.t. T, P
    factor2 = 1 - p.alpha * volc
    dk2_dT = p.A2 * factor2 * dphi2_dT * Pf2 / dL
    dk2_dP = p.A2 * factor2 * phi2 * (dPf2_dP * dL - Pf2 * ddL_dP) / (dL**2)

    # Derivatives of Y_B w.r.t. k1, k2, tau
    dY_dk1 = tau * d2 / (denom**2)
    dY_dk2 = -k1 * tau * d1 * tau / (denom**2)
    dY_dtau = (k1 * denom - k1 * tau * (k1 * d2 + d1 * k2)) / (denom**2)

    # Chain rule for J = 100 * Y_B
    dJ_dT = 100.0 * (dY_dk1 * dk1_dT + dY_dk2 * dk2_dT)
    dJ_dP = 100.0 * (dY_dk1 * dk1_dP + dY_dk2 * dk2_dP)
    dJ_dtau = 100.0 * dY_dtau

    return np.array([dJ_dT, dJ_dP, dJ_dtau])


def create_cstr(constrained: bool = True) -> BiLevelProblem:
    """
    Create CSTR bi-level optimization problem.

    Consecutive reactions A -> B -> C in a CSTR. Maximize yield of B.

    Variables:
        x^{WB} = [T, P, tau]  - process: temperature (K), pressure (bar), residence time (s)
        x^{BB} = [Eb, dEa]    - catalyst: binding energy (eV), activation shift

    Output from black-box:
        y = [Ea1, Ea2, volc, Eb]  (Kads computed in white-box from Eb)

    Constraints (when constrained=True):
        g1: Y_C - 0.25 <= 0 (at most 25% to byproduct)
        g2: P/Pmax - (T-Tmin)/(Tmax-Tmin) - 0.3 <= 0 (safety)

    Args:
        constrained: If True, include process constraints (default True)

    Returns:
        BiLevelProblem instance

    Optimal solution (verified with MATLAB reference):
        x^{WB}* = [647.25, 20.0, 5.0]
        x^{BB}* = [-1.047, 0.2]
        J* = 92.324%
    """
    return BiLevelProblem(
        name="CSTR",
        n_x_wb=3,
        n_x_bb=2,
        n_y=4,  # [Ea1, Ea2, volc, Eb]
        n_g=2 if constrained else 0,
        x_wb_lower=np.array([450.0, 1.0, 0.1]),     # T_min, P_min, tau_min
        x_wb_upper=np.array([700.0, 20.0, 5.0]),    # T_max, P_max, tau_max
        x_bb_lower=np.array([-2.0, -0.2]),          # Eb_min, dEa_min
        x_bb_upper=np.array([0.0, 0.2]),            # Eb_max, dEa_max
        fbb=_cstr_fbb,
        J=_cstr_J,
        g=_cstr_g if constrained else None,
        J_grad_x_wb=_cstr_J_grad_x_wb,
        g_grad_x_wb=_cstr_g_grad_x_wb if constrained else None,
        x_wb_optimal=np.array([647.253937125, 20.0, 5.0]),
        x_bb_optimal=np.array([-1.047089294942076, 0.2]),
        J_optimal=92.324046159270395,
        maximize=True
    )


# =============================================================================
# Heat Exchanger with Novel Working Fluid
# =============================================================================
#
#   min   C_capital + C_operating
#   s.t.  C_capital = 1000 * A^0.6
#         C_operating = 500 * m_dot^3 * y2 / A
#         U = y3 / (0.01 + 0.001/y1)
#         g1: 100 - U * A * dT_lm ≤ 0  (heat duty constraint)
#         g2: 4000 - 4*m_dot / (π * 0.05 * y2) ≤ 0  (Reynolds number constraint)
#         y1 = 2.0 + 0.5*sin(π*m1/50) + 0.3*m2  (heat capacity)
#         y2 = 0.001 * exp((m1-100)/50) * (1 + 0.5*m2)  (viscosity)
#         y3 = 0.1 + 0.05*(1 - ((m1-125)/75)^2) + 0.02*m2  (thermal conductivity)
#
#   x^{WB} = [A, m_dot, dT_lm]  - white-box (process) variables
#   x^{BB} = [m1, m2]           - black-box (molecular descriptors)
#   y = [y1, y2, y3]            - thermodynamic properties
#
# =============================================================================

def _heat_exchanger_fbb(x_bb: np.ndarray) -> np.ndarray:
    """
    Black-box function y = f^{BB}(x^{BB}) for Heat Exchanger: compute fluid properties from molecular descriptors.

    Args:
        x_bb: [m1, m2] - molecular descriptors

    Returns:
        y = [y1, y2, y3] - heat capacity, viscosity, thermal conductivity
    """
    m1, m2 = x_bb[0], x_bb[1]

    # Heat capacity [kJ/kg·K]
    y1 = 2.0 + 0.5 * np.sin(np.pi * m1 / 50) + 0.3 * m2

    # Viscosity [Pa·s]
    y2 = 0.001 * np.exp((m1 - 100) / 50) * (1 + 0.5 * m2)

    # Thermal conductivity [W/m·K]
    y3 = 0.1 + 0.05 * (1 - ((m1 - 125) / 75) ** 2) + 0.02 * m2

    return np.array([y1, y2, y3])


def _heat_exchanger_J(x_wb: np.ndarray, y: np.ndarray) -> float:
    """
    Objective J(x^{WB}, y) for Heat Exchanger: total cost.

    Args:
        x_wb: [A, m_dot, dT_lm] - area, mass flow rate, log-mean temp difference
        y: [y1, y2, y3] - fluid properties

    Returns:
        Total cost (capital + operating)
    """
    A, m_dot, dT_lm = x_wb[0], x_wb[1], x_wb[2]
    y1, y2, y3 = y[0], y[1], y[2]

    C_capital = 1000 * A ** 0.6
    C_operating = 500 * m_dot ** 3 * y2 / A

    return C_capital + C_operating


def _heat_exchanger_g(x_wb: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Constraints g(x^{WB}, y) for Heat Exchanger (≤ 0 is feasible).

    g1: Heat duty constraint - Q = U*A*dT_lm >= 100 kW
    g2: Reynolds number constraint - Re >= 4000 (turbulent flow)
    """
    A, m_dot, dT_lm = x_wb[0], x_wb[1], x_wb[2]
    y1, y2, y3 = y[0], y[1], y[2]

    # Overall heat transfer coefficient
    U = y3 / (0.01 + 0.001 / y1)

    # Heat duty constraint: Q >= 100
    g1 = 100 - U * A * dT_lm

    # Reynolds number constraint: Re >= 4000
    # Re = 4*m_dot / (π * D * μ), assuming D = 0.05 m
    g2 = 4000 - 4 * m_dot / (np.pi * 0.05 * y2)

    return np.array([g1, g2])


def _heat_exchanger_J_grad_x_wb(x_wb: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradient of J w.r.t. x^{WB} for Heat Exchanger."""
    A, m_dot, dT_lm = x_wb[0], x_wb[1], x_wb[2]
    y1, y2, y3 = y[0], y[1], y[2]

    dJ_dA = 1000 * 0.6 * A ** (-0.4) - 500 * m_dot ** 3 * y2 / A ** 2
    dJ_dm_dot = 500 * 3 * m_dot ** 2 * y2 / A
    dJ_ddT_lm = 0.0

    return np.array([dJ_dA, dJ_dm_dot, dJ_ddT_lm])


def _heat_exchanger_g_grad_x_wb(x_wb: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Jacobian of g w.r.t. x^{WB} for Heat Exchanger."""
    A, m_dot, dT_lm = x_wb[0], x_wb[1], x_wb[2]
    y1, y2, y3 = y[0], y[1], y[2]

    U = y3 / (0.01 + 0.001 / y1)

    # g1 = 100 - U*A*dT_lm
    dg1_dA = -U * dT_lm
    dg1_dm_dot = 0.0
    dg1_ddT_lm = -U * A

    # g2 = 4000 - 4*m_dot / (π * 0.05 * y2)
    dg2_dA = 0.0
    dg2_dm_dot = -4 / (np.pi * 0.05 * y2)
    dg2_ddT_lm = 0.0

    return np.array([
        [dg1_dA, dg1_dm_dot, dg1_ddT_lm],
        [dg2_dA, dg2_dm_dot, dg2_ddT_lm]
    ])


def create_heat_exchanger() -> BiLevelProblem:
    """
    Create Heat Exchanger bi-level optimization problem.

    This is a constrained minimization problem where:
        x^{WB} = [A, m_dot, dT_lm]  - white-box (process) variables
        x^{BB} = [m1, m2]           - black-box (molecular descriptors)
        y = [y1, y2, y3]            - thermodynamic properties

    Constraints:
        g1: Heat duty Q >= 100 kW
        g2: Reynolds number Re >= 4000 (turbulent flow)
    """
    return BiLevelProblem(
        name="Heat-Exchanger",
        n_x_wb=3,
        n_x_bb=2,
        n_y=3,
        n_g=2,
        x_wb_lower=np.array([1.0, 0.1, 5.0]),
        x_wb_upper=np.array([50.0, 5.0, 50.0]),
        x_bb_lower=np.array([50.0, 0.0]),
        x_bb_upper=np.array([200.0, 1.0]),
        fbb=_heat_exchanger_fbb,
        J=_heat_exchanger_J,
        g=_heat_exchanger_g,
        J_grad_x_wb=_heat_exchanger_J_grad_x_wb,
        g_grad_x_wb=_heat_exchanger_g_grad_x_wb,
        x_wb_optimal=np.array([1.0, 0.1, 50.0]),
        x_bb_optimal=np.array([50.0, 0.0]),
        J_optimal=1000.000183939720614,
        maximize=False
    )


# =============================================================================
# Pressure Swing Adsorption with Novel Adsorbent
# =============================================================================
#
#   min   -Prod + 0.1 * ln(P_H / P_L)
#   s.t.  α = y2 / y3
#         Δq = y1 * (y2*P_H/(1+y2*P_H) - y2*P_L/(1+y2*P_L))
#         Prod = Δq / t_ads
#         g1: 3 - α ≤ 0  (minimum selectivity)
#         g2: 0.95 - α*P_H/(1+α*P_H) ≤ 0  (minimum purity)
#         y1 = 5.0 * exp(-((σ-6)^2)/4) * (1 + 0.1*ε)  (max loading)
#         y2 = 0.5 * exp((ε-10)/5) * (1 - 0.05*(σ-5)^2)  (K_A)
#         y3 = 0.1 * exp((ε-15)/8) * (1 + 0.03*(σ-7)^2)  (K_B)
#
#   x^{WB} = [P_H, P_L, t_ads]  - white-box (process) variables
#   x^{BB} = [σ, ε]             - black-box (adsorbent properties)
#   y = [y1, y2, y3]            - adsorption parameters
#
# =============================================================================

def _psa_fbb(x_bb: np.ndarray) -> np.ndarray:
    """
    Black-box function y = f^{BB}(x^{BB}) for PSA: compute adsorption parameters from adsorbent properties.

    Args:
        x_bb: [σ, ε] - pore size and surface energy

    Returns:
        y = [y1, y2, y3] - max loading, K_A, K_B
    """
    sigma, epsilon = x_bb[0], x_bb[1]

    # Maximum loading [mol/kg]
    y1 = 5.0 * np.exp(-((sigma - 6) ** 2) / 4) * (1 + 0.1 * epsilon)

    # Adsorption equilibrium constant for component A
    y2 = 0.5 * np.exp((epsilon - 10) / 5) * (1 - 0.05 * (sigma - 5) ** 2)

    # Adsorption equilibrium constant for component B
    y3 = 0.1 * np.exp((epsilon - 15) / 8) * (1 + 0.03 * (sigma - 7) ** 2)

    return np.array([y1, y2, y3])


def _psa_J(x_wb: np.ndarray, y: np.ndarray) -> float:
    """
    Objective J(x^{WB}, y) for PSA: negative productivity + compression cost.

    Args:
        x_wb: [P_H, P_L, t_ads] - high/low pressure, adsorption time
        y: [y1, y2, y3] - adsorption parameters

    Returns:
        Objective value (minimize)
    """
    P_H, P_L, t_ads = x_wb[0], x_wb[1], x_wb[2]
    y1, y2, y3 = y[0], y[1], y[2]

    # Working capacity
    delta_q = y1 * (y2 * P_H / (1 + y2 * P_H) - y2 * P_L / (1 + y2 * P_L))

    # Productivity
    Prod = delta_q / t_ads

    # Objective: maximize productivity, minimize compression (via pressure ratio)
    return -Prod + 0.1 * np.log(P_H / P_L)


def _psa_g(x_wb: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Constraints g(x^{WB}, y) for PSA (≤ 0 is feasible).

    g1: Minimum selectivity α >= 3
    g2: Minimum purity constraint
    """
    P_H, P_L, t_ads = x_wb[0], x_wb[1], x_wb[2]
    y1, y2, y3 = y[0], y[1], y[2]

    # Selectivity
    alpha = y2 / y3

    # g1: α >= 3 → 3 - α <= 0
    g1 = 3 - alpha

    # g2: Purity constraint: α*P_H/(1+α*P_H) >= 0.95
    g2 = 0.95 - alpha * P_H / (1 + alpha * P_H)

    return np.array([g1, g2])


def _psa_J_grad_x_wb(x_wb: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradient of J w.r.t. x^{WB} for PSA."""
    P_H, P_L, t_ads = x_wb[0], x_wb[1], x_wb[2]
    y1, y2, y3 = y[0], y[1], y[2]

    # Derivatives of Langmuir terms
    dq_H_dP_H = y1 * y2 / (1 + y2 * P_H) ** 2
    dq_L_dP_L = y1 * y2 / (1 + y2 * P_L) ** 2

    delta_q = y1 * (y2 * P_H / (1 + y2 * P_H) - y2 * P_L / (1 + y2 * P_L))

    dJ_dP_H = -dq_H_dP_H / t_ads + 0.1 / P_H
    dJ_dP_L = dq_L_dP_L / t_ads - 0.1 / P_L
    dJ_dt_ads = delta_q / t_ads ** 2

    return np.array([dJ_dP_H, dJ_dP_L, dJ_dt_ads])


def _psa_g_grad_x_wb(x_wb: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Jacobian of g w.r.t. x^{WB} for PSA."""
    P_H, P_L, t_ads = x_wb[0], x_wb[1], x_wb[2]
    y1, y2, y3 = y[0], y[1], y[2]

    alpha = y2 / y3

    # g1 = 3 - α (no dependence on x^p)
    dg1_dP_H = 0.0
    dg1_dP_L = 0.0
    dg1_dt_ads = 0.0

    # g2 = 0.95 - α*P_H/(1+α*P_H)
    dg2_dP_H = -alpha / (1 + alpha * P_H) ** 2
    dg2_dP_L = 0.0
    dg2_dt_ads = 0.0

    return np.array([
        [dg1_dP_H, dg1_dP_L, dg1_dt_ads],
        [dg2_dP_H, dg2_dP_L, dg2_dt_ads]
    ])


def create_psa() -> BiLevelProblem:
    """
    Create Pressure Swing Adsorption bi-level optimization problem.

    This is a constrained minimization problem where:
        x^{WB} = [P_H, P_L, t_ads]  - white-box (process) variables
        x^{BB} = [σ, ε]             - black-box (adsorbent properties)
        y = [y1, y2, y3]            - adsorption parameters

    Constraints:
        g1: Selectivity α >= 3
        g2: Minimum purity constraint
    """
    return BiLevelProblem(
        name="PSA",
        n_x_wb=3,
        n_x_bb=2,
        n_y=3,
        n_g=2,
        x_wb_lower=np.array([2.0, 0.1, 10.0]),
        x_wb_upper=np.array([10.0, 1.0, 120.0]),
        x_bb_lower=np.array([3.0, 5.0]),
        x_bb_upper=np.array([10.0, 25.0]),
        fbb=_psa_fbb,
        J=_psa_J,
        g=_psa_g,
        J_grad_x_wb=_psa_J_grad_x_wb,
        g_grad_x_wb=_psa_g_grad_x_wb,
        x_wb_optimal=np.array([3.9706806663424, 0.1, 10.0]),
        x_bb_optimal=np.array([6.037087608909175, 19.552432730962334]),
        J_optimal=-0.643289781399212,
        maximize=False
    )


# =============================================================================
# Batch Reactor with Catalyst Optimization
# =============================================================================
#
#   min   -C_B + 0.001 * T * t_f
#   s.t.  k1' = y1 * exp(-2000/T)
#         k2' = y2 * exp(-2400/T)
#         C_A = C_A0 * exp(-k1' * t_f)
#         C_B = C_A0 * k1'/(k2'-k1') * (exp(-k1'*t_f) - exp(-k2'*t_f))
#         g1: 0.8 - (1 - C_A/C_A0) ≤ 0  (conversion >= 80%)
#         g2: 0.7 - C_B/(C_A0 - C_A) ≤ 0  (selectivity >= 70%)
#         y1 = 10 * d * exp(-((E_b+1)^2)/0.25)  (k1 pre-factor)
#         y2 = 0.5 * d * exp(-((E_b+0.5)^2)/0.5)  (k2 pre-factor)
#         y3 = exp(-2*E_b - 1)  (equilibrium constant, unused)
#
#   x^{WB} = [T, t_f, C_A0]  - white-box (process) variables
#   x^{BB} = [E_b, d]        - black-box (catalyst properties)
#   y = [y1, y2, y3]         - kinetic parameters
#
# =============================================================================

def _batch_reactor_fbb(x_bb: np.ndarray) -> np.ndarray:
    """
    Black-box function y = f^{BB}(x^{BB}) for Batch Reactor: compute kinetic parameters from catalyst properties.

    Args:
        x_bb: [E_b, d] - binding energy and dispersion factor

    Returns:
        y = [y1, y2, y3] - rate constants and equilibrium constant
    """
    E_b, d = x_bb[0], x_bb[1]

    # Forward rate constant pre-factor
    y1 = 10 * d * np.exp(-((E_b + 1) ** 2) / 0.25)

    # Side reaction rate constant pre-factor
    y2 = 0.5 * d * np.exp(-((E_b + 0.5) ** 2) / 0.5)

    # Equilibrium constant (for completeness)
    y3 = np.exp(-2 * E_b - 1)

    return np.array([y1, y2, y3])


def _batch_reactor_J(x_wb: np.ndarray, y: np.ndarray) -> float:
    """
    Objective J(x^{WB}, y) for Batch Reactor: negative yield + operating cost.

    Args:
        x_wb: [T, t_f, C_A0] - temperature, final time, initial concentration
        y: [y1, y2, y3] - kinetic parameters

    Returns:
        Objective value (minimize)
    """
    T, t_f, C_A0 = x_wb[0], x_wb[1], x_wb[2]
    y1, y2, y3 = y[0], y[1], y[2]

    # Temperature-dependent rate constants (Ea/R values reduced for feasibility)
    k1_prime = y1 * np.exp(-2000 / T)
    k2_prime = y2 * np.exp(-2400 / T)

    # Avoid division by zero
    if abs(k2_prime - k1_prime) < 1e-10:
        k2_prime = k1_prime + 1e-10

    # Concentrations from batch reactor equations
    C_A = C_A0 * np.exp(-k1_prime * t_f)
    C_B = C_A0 * k1_prime / (k2_prime - k1_prime) * (np.exp(-k1_prime * t_f) - np.exp(-k2_prime * t_f))

    return -C_B + 0.001 * T * t_f


def _batch_reactor_g(x_wb: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Constraints g(x^{WB}, y) for Batch Reactor (≤ 0 is feasible).

    g1: Minimum conversion X_A >= 0.8
    g2: Minimum selectivity S_B >= 0.7
    """
    T, t_f, C_A0 = x_wb[0], x_wb[1], x_wb[2]
    y1, y2, y3 = y[0], y[1], y[2]

    k1_prime = y1 * np.exp(-2000 / T)
    k2_prime = y2 * np.exp(-2400 / T)

    if abs(k2_prime - k1_prime) < 1e-10:
        k2_prime = k1_prime + 1e-10

    C_A = C_A0 * np.exp(-k1_prime * t_f)
    C_B = C_A0 * k1_prime / (k2_prime - k1_prime) * (np.exp(-k1_prime * t_f) - np.exp(-k2_prime * t_f))

    # Conversion
    X_A = 1 - C_A / C_A0

    # Selectivity (avoid division by zero)
    denom = C_A0 - C_A
    if abs(denom) < 1e-10:
        S_B = 0.0
    else:
        S_B = C_B / denom

    # g1: X_A >= 0.8 → 0.8 - X_A <= 0
    g1 = 0.8 - X_A

    # g2: S_B >= 0.7 → 0.7 - S_B <= 0
    g2 = 0.7 - S_B

    return np.array([g1, g2])


def _batch_reactor_J_grad_x_wb(x_wb: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradient of J w.r.t. x^{WB} for Batch Reactor."""
    T, t_f, C_A0 = x_wb[0], x_wb[1], x_wb[2]
    y1, y2, y3 = y[0], y[1], y[2]

    k1_prime = y1 * np.exp(-2000 / T)
    k2_prime = y2 * np.exp(-2400 / T)

    dk1_dT = k1_prime * 2000 / T ** 2
    dk2_dT = k2_prime * 2400 / T ** 2

    if abs(k2_prime - k1_prime) < 1e-10:
        k2_prime = k1_prime + 1e-10

    exp1 = np.exp(-k1_prime * t_f)
    exp2 = np.exp(-k2_prime * t_f)
    diff_k = k2_prime - k1_prime

    C_B = C_A0 * k1_prime / diff_k * (exp1 - exp2)

    # Partial derivatives of C_B
    dCB_dT = C_A0 * (
        dk1_dT / diff_k * (exp1 - exp2) +
        k1_prime * (-(dk2_dT - dk1_dT) / diff_k ** 2) * (exp1 - exp2) +
        k1_prime / diff_k * (-t_f * dk1_dT * exp1 + t_f * dk2_dT * exp2)
    )

    dCB_dt_f = C_A0 * k1_prime / diff_k * (-k1_prime * exp1 + k2_prime * exp2)

    dCB_dC_A0 = k1_prime / diff_k * (exp1 - exp2)

    dJ_dT = -dCB_dT + 0.001 * t_f
    dJ_dt_f = -dCB_dt_f + 0.001 * T
    dJ_dC_A0 = -dCB_dC_A0

    return np.array([dJ_dT, dJ_dt_f, dJ_dC_A0])


def create_batch_reactor() -> BiLevelProblem:
    """
    Create Batch Reactor bi-level optimization problem.

    This is a constrained minimization problem where:
        x^{WB} = [T, t_f, C_A0]  - white-box (process) variables
        x^{BB} = [E_b, d]        - black-box (catalyst properties)
        y = [y1, y2, y3]         - kinetic parameters

    Constraints:
        g1: Conversion X_A >= 0.8
        g2: Selectivity S_B >= 0.7
    """
    return BiLevelProblem(
        name="Batch-Reactor",
        n_x_wb=3,
        n_x_bb=2,
        n_y=3,
        n_g=2,
        x_wb_lower=np.array([300.0, 0.5, 0.5]),
        x_wb_upper=np.array([500.0, 10.0, 5.0]),
        x_bb_lower=np.array([-2.0, 0.5]),
        x_bb_upper=np.array([0.0, 2.0]),
        fbb=_batch_reactor_fbb,
        J=_batch_reactor_J,
        g=_batch_reactor_g,
        J_grad_x_wb=_batch_reactor_J_grad_x_wb,
        g_grad_x_wb=None,  # Complex, numerical gradients OK
        x_wb_optimal=np.array([500.0, 4.394255183385335, 5.0]),
        x_bb_optimal=np.array([-1.006027560159181, 2.0]),
        J_optimal=-1.748817620115523,
        maximize=False
    )


# =============================================================================
# Extractive Distillation with Novel Solvent
# =============================================================================
#
#   min   1000*N + 500*R + 200*(S/F)*y3
#   s.t.  R_min = (1/(y1-1)) * (x_D/x_F - y1*(1-x_D)/(1-x_F))
#         N_min = ln((x_D*(1-x_B))/(x_B*(1-x_D))) / ln(y1)
#         g1: 1.2*R_min - R ≤ 0
#         g2: 1.5*N_min - N ≤ 0
#         g3: (S/F)*y2 - 5000 ≤ 0
#         y1 = 2.5 + 1.0*sin(π*(δ_H-10)/20)*cos(π*(δ_P-10)/15)  (modified volatility, y1 > 1 always)
#         y2 = 800 + 50*δ_H - 20*δ_P  (solvent density)
#         y3 = 0.5 + 0.1*δ_H + 0.05*δ_P^2/100  (solvent viscosity)
#
#   x^{WB} = [R, N, S/F]  - white-box (process) variables
#   x^{BB} = [δ_H, δ_P]   - black-box (solubility parameters)
#   y = [y1, y2, y3]      - solvent properties
#
# =============================================================================

# Distillation specifications (constants)
_DISTILL_X_D = 0.99  # Distillate purity
_DISTILL_X_F = 0.5   # Feed composition
_DISTILL_X_B = 0.01  # Bottoms impurity


def _distillation_fbb(x_bb: np.ndarray) -> np.ndarray:
    """
    Black-box function y = f^{BB}(x^{BB}) for Distillation: compute solvent properties from solubility parameters.

    Args:
        x_bb: [δ_H, δ_P] - Hansen solubility parameters

    Returns:
        y = [y1, y2, y3] - modified volatility, density, viscosity
    """
    delta_H, delta_P = x_bb[0], x_bb[1]

    # Modified relative volatility (shifted to ensure y1 > 1 always)
    y1 = 2.5 + 1.0 * np.sin(np.pi * (delta_H - 10) / 20) * np.cos(np.pi * (delta_P - 10) / 15)

    # Solvent density [kg/m³]
    y2 = 800 + 50 * delta_H - 20 * delta_P

    # Solvent viscosity [cP]
    y3 = 0.5 + 0.1 * delta_H + 0.05 * delta_P ** 2 / 100

    return np.array([y1, y2, y3])


def _distillation_J(x_wb: np.ndarray, y: np.ndarray) -> float:
    """
    Objective J(x^{WB}, y) for Distillation: total annual cost.

    Args:
        x_wb: [R, N, S_F] - reflux ratio, number of stages, solvent-to-feed ratio
        y: [y1, y2, y3] - solvent properties

    Returns:
        Total cost
    """
    R, N, S_F = x_wb[0], x_wb[1], x_wb[2]
    y1, y2, y3 = y[0], y[1], y[2]

    return 1000 * N + 500 * R + 200 * S_F * y3


def _distillation_g(x_wb: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Constraints g(x^{WB}, y) for Distillation (≤ 0 is feasible).

    g1: R >= 1.2 * R_min
    g2: N >= 1.5 * N_min
    g3: Solvent capacity constraint
    """
    R, N, S_F = x_wb[0], x_wb[1], x_wb[2]
    y1, y2, y3 = y[0], y[1], y[2]

    x_D = _DISTILL_X_D
    x_F = _DISTILL_X_F
    x_B = _DISTILL_X_B

    # Ensure y1 > 1 for valid distillation
    y1_safe = max(y1, 1.01)

    # Minimum reflux ratio
    R_min = (1 / (y1_safe - 1)) * (x_D / x_F - y1_safe * (1 - x_D) / (1 - x_F))

    # Minimum number of stages (Fenske equation)
    N_min = np.log((x_D * (1 - x_B)) / (x_B * (1 - x_D))) / np.log(y1_safe)

    # g1: R >= 1.2*R_min → 1.2*R_min - R <= 0
    g1 = 1.2 * R_min - R

    # g2: N >= 1.5*N_min → 1.5*N_min - N <= 0
    g2 = 1.5 * N_min - N

    # g3: Solvent capacity constraint
    g3 = S_F * y2 - 5000

    return np.array([g1, g2, g3])


def _distillation_J_grad_x_wb(x_wb: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradient of J w.r.t. x^{WB} for Distillation."""
    R, N, S_F = x_wb[0], x_wb[1], x_wb[2]
    y1, y2, y3 = y[0], y[1], y[2]

    dJ_dR = 500
    dJ_dN = 1000
    dJ_dS_F = 200 * y3

    return np.array([dJ_dR, dJ_dN, dJ_dS_F])


def _distillation_g_grad_x_wb(x_wb: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Jacobian of g w.r.t. x^{WB} for Distillation."""
    R, N, S_F = x_wb[0], x_wb[1], x_wb[2]
    y1, y2, y3 = y[0], y[1], y[2]

    # g1 = 1.2*R_min - R
    dg1_dR = -1.0
    dg1_dN = 0.0
    dg1_dS_F = 0.0

    # g2 = 1.5*N_min - N
    dg2_dR = 0.0
    dg2_dN = -1.0
    dg2_dS_F = 0.0

    # g3 = S_F*y2 - 5000
    dg3_dR = 0.0
    dg3_dN = 0.0
    dg3_dS_F = y2

    return np.array([
        [dg1_dR, dg1_dN, dg1_dS_F],
        [dg2_dR, dg2_dN, dg2_dS_F],
        [dg3_dR, dg3_dN, dg3_dS_F]
    ])


def create_distillation() -> BiLevelProblem:
    """
    Create Extractive Distillation bi-level optimization problem.

    This is a constrained minimization problem where:
        x^{WB} = [R, N, S/F]  - white-box (process) variables
        x^{BB} = [δ_H, δ_P]   - black-box (solubility parameters)
        y = [y1, y2, y3]      - solvent properties

    Constraints:
        g1: R >= 1.2 * R_min
        g2: N >= 1.5 * N_min
        g3: Solvent capacity (S/F)*ρ <= 5000
    """
    return BiLevelProblem(
        name="Distillation",
        n_x_wb=3,
        n_x_bb=2,
        n_y=3,
        n_g=3,
        x_wb_lower=np.array([1.0, 5.0, 0.5]),
        x_wb_upper=np.array([10.0, 50.0, 5.0]),
        x_bb_lower=np.array([5.0, 5.0]),
        x_bb_upper=np.array([25.0, 20.0]),
        fbb=_distillation_fbb,
        J=_distillation_J,
        g=_distillation_g,
        J_grad_x_wb=_distillation_J_grad_x_wb,
        g_grad_x_wb=_distillation_g_grad_x_wb,
        x_wb_optimal=np.array([1.0, 11.00477484891093, 0.5]),
        x_bb_optimal=np.array([19.838610136796817, 9.990868171746914]),
        J_optimal=11758.151822827872820,
        maximize=False
    )


# =============================================================================
# Multi-Effect Evaporator with Phase-Change Material
# =============================================================================
#
#   min   -m_dot_evap + 0.1*n^2 + 0.01*m_dot_HTF^2
#   s.t.  Q_HTF = m_dot_HTF * (y1 + y2*10)
#         m_dot_evap = Q_HTF * n * 0.9^(n-1) / 2260
#         g1: 100 + n*ΔT_effect - y3 ≤ 0  (temperature feasibility)
#         g2: 5 - m_dot_evap ≤ 0  (minimum evaporation rate)
#         y1 = 150*λ * (1 + 0.2*sin(π*T_m/100))  (latent heat)
#         y2 = 2.0 + 0.5*λ - 0.002*T_m  (heat capacity)
#         y3 = T_m + 10*sin(π*λ/2)  (melting temperature)
#
#   x^{WB} = [n, m_dot_HTF, ΔT_effect]  - white-box (process) variables
#   x^{BB} = [T_m, λ]                    - black-box (PCM properties)
#   y = [y1, y2, y3]                     - thermodynamic properties
#
# =============================================================================

def _evaporator_fbb(x_bb: np.ndarray) -> np.ndarray:
    """
    Black-box function y = f^{BB}(x^{BB}) for Evaporator: compute PCM properties.

    Args:
        x_bb: [T_m, λ] - melting temperature target, latent heat factor

    Returns:
        y = [y1, y2, y3] - latent heat, heat capacity, actual melting temp
    """
    T_m, lam = x_bb[0], x_bb[1]

    # Latent heat of fusion [kJ/kg]
    y1 = 150 * lam * (1 + 0.2 * np.sin(np.pi * T_m / 100))

    # Liquid heat capacity [kJ/kg·K]
    y2 = 2.0 + 0.5 * lam - 0.002 * T_m

    # Actual melting temperature [°C]
    y3 = T_m + 10 * np.sin(np.pi * lam / 2)

    return np.array([y1, y2, y3])


def _evaporator_J(x_wb: np.ndarray, y: np.ndarray) -> float:
    """
    Objective J(x^{WB}, y) for Evaporator: trade-off between evaporation and cost.

    Args:
        x_wb: [n, m_dot_HTF, dT_effect] - number of effects, HTF flow, temp drop per effect
        y: [y1, y2, y3] - PCM properties

    Returns:
        Objective value (minimize)
    """
    n, m_dot_HTF, dT_effect = x_wb[0], x_wb[1], x_wb[2]
    y1, y2, y3 = y[0], y[1], y[2]

    # Heat from HTF (latent + sensible over 10K superheat)
    Q_HTF = m_dot_HTF * (y1 + y2 * 10)

    # Evaporation rate with efficiency decay
    m_dot_evap = Q_HTF * n * 0.9 ** (n - 1) / 2260

    return -m_dot_evap + 0.1 * n ** 2 + 0.01 * m_dot_HTF ** 2


def _evaporator_g(x_wb: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Constraints g(x^{WB}, y) for Evaporator (≤ 0 is feasible).

    g1: Temperature feasibility - 100 + n*ΔT <= T_melt
    g2: Minimum evaporation rate - m_dot_evap >= 5
    """
    n, m_dot_HTF, dT_effect = x_wb[0], x_wb[1], x_wb[2]
    y1, y2, y3 = y[0], y[1], y[2]

    Q_HTF = m_dot_HTF * (y1 + y2 * 10)
    m_dot_evap = Q_HTF * n * 0.9 ** (n - 1) / 2260

    # g1: Last effect must be above 100°C: 100 + n*ΔT <= y3
    g1 = 100 + n * dT_effect - y3

    # g2: Minimum evaporation rate
    g2 = 5 - m_dot_evap

    return np.array([g1, g2])


def _evaporator_J_grad_x_wb(x_wb: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradient of J w.r.t. x^{WB} for Evaporator."""
    n, m_dot_HTF, dT_effect = x_wb[0], x_wb[1], x_wb[2]
    y1, y2, y3 = y[0], y[1], y[2]

    Q_HTF = m_dot_HTF * (y1 + y2 * 10)

    # m_dot_evap = Q_HTF * n * 0.9^(n-1) / 2260
    efficiency = 0.9 ** (n - 1)

    # d(m_dot_evap)/dn = Q_HTF/2260 * (0.9^(n-1) + n * 0.9^(n-1) * ln(0.9))
    dm_evap_dn = Q_HTF / 2260 * (efficiency + n * efficiency * np.log(0.9))

    dm_evap_dm_dot_HTF = (y1 + y2 * 10) * n * efficiency / 2260

    dJ_dn = -dm_evap_dn + 0.2 * n
    dJ_dm_dot_HTF = -dm_evap_dm_dot_HTF + 0.02 * m_dot_HTF
    dJ_ddT_effect = 0.0

    return np.array([dJ_dn, dJ_dm_dot_HTF, dJ_ddT_effect])


def _evaporator_g_grad_x_wb(x_wb: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Jacobian of g w.r.t. x^{WB} for Evaporator."""
    n, m_dot_HTF, dT_effect = x_wb[0], x_wb[1], x_wb[2]
    y1, y2, y3 = y[0], y[1], y[2]

    Q_HTF = m_dot_HTF * (y1 + y2 * 10)
    efficiency = 0.9 ** (n - 1)

    dm_evap_dn = Q_HTF / 2260 * (efficiency + n * efficiency * np.log(0.9))
    dm_evap_dm_dot_HTF = (y1 + y2 * 10) * n * efficiency / 2260

    # g1 = 100 + n*dT_effect - y3
    dg1_dn = dT_effect
    dg1_dm_dot_HTF = 0.0
    dg1_ddT_effect = n

    # g2 = 5 - m_dot_evap
    dg2_dn = -dm_evap_dn
    dg2_dm_dot_HTF = -dm_evap_dm_dot_HTF
    dg2_ddT_effect = 0.0

    return np.array([
        [dg1_dn, dg1_dm_dot_HTF, dg1_ddT_effect],
        [dg2_dn, dg2_dm_dot_HTF, dg2_ddT_effect]
    ])


def create_evaporator() -> BiLevelProblem:
    """
    Create Multi-Effect Evaporator bi-level optimization problem.

    This is a constrained minimization problem where:
        x^{WB} = [n, m_dot_HTF, ΔT_effect]  - white-box (process) variables
        x^{BB} = [T_m, λ]                    - black-box (PCM properties)
        y = [y1, y2, y3]                     - thermodynamic properties

    Constraints:
        g1: Temperature feasibility
        g2: Minimum evaporation rate >= 5
    """
    return BiLevelProblem(
        name="Evaporator",
        n_x_wb=3,
        n_x_bb=2,
        n_y=3,
        n_g=2,
        x_wb_lower=np.array([2.0, 1.0, 5.0]),
        x_wb_upper=np.array([6.0, 20.0, 20.0]),
        x_bb_lower=np.array([50.0, 0.5]),
        x_bb_upper=np.array([150.0, 2.0]),
        fbb=_evaporator_fbb,
        J=_evaporator_J,
        g=_evaporator_g,
        J_grad_x_wb=_evaporator_J_grad_x_wb,
        g_grad_x_wb=_evaporator_g_grad_x_wb,
        x_wb_optimal=np.array([4.094986344388144, 19.067216167485267, 5.0]),
        x_bb_optimal=np.array([120.47493172142639, 2.0]),
        J_optimal=-1.958696683871302,
        maximize=False
    )


# =============================================================================
# Membrane Separation with Designed Polymer
# =============================================================================
#
#   min   -F_A * y2 + 0.1*A_m + 10*Δp
#   s.t.  J_A = y1 * Δp / δ * 1e-10
#         F_A = J_A * A_m
#         g1: 1e-6 - J_A ≤ 0  (minimum flux)
#         g2: Δp*A_m/y3 - 100 ≤ 0  (mechanical stability)
#         y1 = 1000*exp((FFV-0.15)/0.05) * (1 + 0.1*d_spacing)  (permeability)
#         y2 = 50*exp(-(FFV-0.15)/0.1) * exp(-((d_spacing-5)^2)/4)  (selectivity)
#         y3 = 100*(0.4-FFV)*(10-d_spacing)  (mechanical strength)
#
#   x^{WB} = [A_m, δ, Δp]        - white-box (process) variables
#   x^{BB} = [FFV, d_spacing]    - black-box (polymer properties)
#   y = [y1, y2, y3]             - membrane properties
#
# =============================================================================

def _membrane_fbb(x_bb: np.ndarray) -> np.ndarray:
    """
    Black-box function y = f^{BB}(x^{BB}) for Membrane: compute membrane properties from polymer structure.

    Args:
        x_bb: [FFV, d_spacing] - fractional free volume, chain spacing

    Returns:
        y = [y1, y2, y3] - permeability, selectivity, mechanical strength
    """
    FFV, d_spacing = x_bb[0], x_bb[1]

    # Permeability of component A [Barrer]
    y1 = 1000 * np.exp((FFV - 0.15) / 0.05) * (1 + 0.1 * d_spacing)

    # Selectivity of A over B
    y2 = 50 * np.exp(-(FFV - 0.15) / 0.1) * np.exp(-((d_spacing - 5) ** 2) / 4)

    # Mechanical strength proxy [MPa]
    y3 = 100 * (0.4 - FFV) * (10 - d_spacing)

    return np.array([y1, y2, y3])


def _membrane_J(x_wb: np.ndarray, y: np.ndarray) -> float:
    """
    Objective J(x^{WB}, y) for Membrane: trade-off between separation and cost.

    Args:
        x_wb: [A_m, δ, Δp] - membrane area, thickness, pressure drop
        y: [y1, y2, y3] - membrane properties

    Returns:
        Objective value (minimize)
    """
    A_m, delta, dp = x_wb[0], x_wb[1], x_wb[2]
    y1, y2, y3 = y[0], y[1], y[2]

    # Flux
    J_A = y1 * dp / delta * 1e-10

    # Permeate flow
    F_A = J_A * A_m

    return -F_A * y2 + 0.1 * A_m + 10 * dp


def _membrane_g(x_wb: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Constraints g(x^{WB}, y) for Membrane (≤ 0 is feasible).

    g1: Minimum flux J_A >= 1e-6
    g2: Mechanical stability Δp*A_m/σ <= 100
    """
    A_m, delta, dp = x_wb[0], x_wb[1], x_wb[2]
    y1, y2, y3 = y[0], y[1], y[2]

    J_A = y1 * dp / delta * 1e-10

    # g1: J_A >= 1e-6 → 1e-6 - J_A <= 0
    g1 = 1e-6 - J_A

    # g2: Mechanical stability
    # Avoid division by zero
    if abs(y3) < 1e-10:
        g2 = 1e6  # Infeasible
    else:
        g2 = dp * A_m / y3 - 100

    return np.array([g1, g2])


def _membrane_J_grad_x_wb(x_wb: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradient of J w.r.t. x^{WB} for Membrane."""
    A_m, delta, dp = x_wb[0], x_wb[1], x_wb[2]
    y1, y2, y3 = y[0], y[1], y[2]

    J_A = y1 * dp / delta * 1e-10

    # F_A = J_A * A_m
    dJ_A_ddelta = -y1 * dp / delta ** 2 * 1e-10
    dJ_A_ddp = y1 / delta * 1e-10

    dF_A_dA_m = J_A
    dF_A_ddelta = A_m * dJ_A_ddelta
    dF_A_ddp = A_m * dJ_A_ddp

    dJ_dA_m = -dF_A_dA_m * y2 + 0.1
    dJ_ddelta = -dF_A_ddelta * y2
    dJ_ddp = -dF_A_ddp * y2 + 10

    return np.array([dJ_dA_m, dJ_ddelta, dJ_ddp])


def _membrane_g_grad_x_wb(x_wb: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Jacobian of g w.r.t. x^{WB} for Membrane."""
    A_m, delta, dp = x_wb[0], x_wb[1], x_wb[2]
    y1, y2, y3 = y[0], y[1], y[2]

    dJ_A_dA_m = 0.0
    dJ_A_ddelta = -y1 * dp / delta ** 2 * 1e-10
    dJ_A_ddp = y1 / delta * 1e-10

    # g1 = 1e-6 - J_A
    dg1_dA_m = 0.0
    dg1_ddelta = -dJ_A_ddelta
    dg1_ddp = -dJ_A_ddp

    # g2 = dp*A_m/y3 - 100
    if abs(y3) < 1e-10:
        dg2_dA_m = 0.0
        dg2_ddelta = 0.0
        dg2_ddp = 0.0
    else:
        dg2_dA_m = dp / y3
        dg2_ddelta = 0.0
        dg2_ddp = A_m / y3

    return np.array([
        [dg1_dA_m, dg1_ddelta, dg1_ddp],
        [dg2_dA_m, dg2_ddelta, dg2_ddp]
    ])


def create_membrane() -> BiLevelProblem:
    """
    Create Membrane Separation bi-level optimization problem.

    This is a constrained minimization problem where:
        x^{WB} = [A_m, δ, Δp]        - white-box (process) variables
        x^{BB} = [FFV, d_spacing]    - black-box (polymer properties)
        y = [y1, y2, y3]             - membrane properties

    Constraints:
        g1: Minimum flux J_A >= 1e-6
        g2: Mechanical stability
    """
    return BiLevelProblem(
        name="Membrane",
        n_x_wb=3,
        n_x_bb=2,
        n_y=3,
        n_g=2,
        x_wb_lower=np.array([10.0, 0.1, 1.0]),
        x_wb_upper=np.array([1000.0, 10.0, 20.0]),
        x_bb_lower=np.array([0.1, 3.0]),
        x_bb_upper=np.array([0.3, 8.0]),
        fbb=_membrane_fbb,
        J=_membrane_J,
        g=_membrane_g,
        J_grad_x_wb=_membrane_J_grad_x_wb,
        g_grad_x_wb=_membrane_g_grad_x_wb,
        x_wb_optimal=np.array([10.0, 0.1, 1.0]),
        x_bb_optimal=np.array([0.3, 5.132169015045111]),
        J_optimal=10.996623892426552,
        maximize=False
    )


# =============================================================================
# Williams-Otto Process
# =============================================================================
#
# From Williams and Otto (1960) and Schmid et al. (2020):
#
#   max   F_pP = μ*X_P - 0.1*μ*X_E  (net product flow)
#   s.t.  k_i(T) = a_i/ρ * exp(-b_i/T), i=1,2,3
#         r_1 = y1*k_1*X_A*X_B,  r_2 = y2*k_2*X_B*X_C,  r_3 = k_3*X_C*X_P
#         Mass balances for A, B, C, E, P, G
#         g1: F_wG - 1.0 ≤ 0  (waste limit)
#         g2: 0.5 - sum(X_i) ≤ 0  (mass balance closure)
#         y1 = exp(-((ε-15)^2)/50) * (1 + 0.1*(σ_c-5))  (activity factor 1)
#         y2 = 1.5*exp(-((ε-12)^2)/40) * exp(-((σ_c-6)^2)/8)  (activity factor 2)
#
#   x^{WB} = [T, F_fB, μ]  - white-box (process) variables
#       T in °R (Rankine), F_fB in klb/h, μ in klb/h
#   x^{BB} = [ε, σ_c]      - black-box (catalyst properties)
#       ε in kJ/mol, σ_c in Å
#   y = [y1, y2]           - activity factors
#
# =============================================================================

# Williams-Otto kinetic constants
_WO_A1 = 5.9755e9      # h^-1
_WO_A2 = 2.5962e12     # h^-1
_WO_A3 = 9.6283e15     # h^-1
_WO_B1 = 12000.0       # °R
_WO_B2 = 15000.0       # °R
_WO_B3 = 20000.0       # °R
_WO_RHO = 50.0         # lb/ft³
_WO_F_FA = 10.0        # klb/h (fixed feed of A)
_WO_M = 2000.0         # kg (reactor holdup, for dimensional consistency)


def _williams_otto_fbb(x_bb: np.ndarray) -> np.ndarray:
    """
    Black-box function y = f^{BB}(x^{BB}) for Williams-Otto: compute catalyst activity factors.

    Args:
        x_bb: [ε, σ_c] - surface energy, pore size

    Returns:
        y = [y1, y2] - activity factors for reactions 1 and 2
    """
    epsilon, sigma_c = x_bb[0], x_bb[1]

    # Activity factor for reaction 1
    y1 = np.exp(-((epsilon - 15) ** 2) / 50) * (1 + 0.1 * (sigma_c - 5))

    # Activity factor for reaction 2
    y2 = 1.5 * np.exp(-((epsilon - 12) ** 2) / 40) * np.exp(-((sigma_c - 6) ** 2) / 8)

    return np.array([y1, y2])


def _williams_otto_solve_mass_balance(T_R, F_fB, mu, y1, y2):
    """
    Solve steady-state mass balances for Williams-Otto reactor.

    Args:
        T_R: Temperature in degrees Rankine (°R)
        F_fB: Feed flow rate of B (klb/h)
        mu: Total outlet flow rate (klb/h)
        y1, y2: Catalyst activity factors

    Returns mass fractions X_A, X_B, X_C, X_E, X_P, X_G and flows F_pP, F_wG.
    """
    # Rate constants (T_R is already in Rankine)
    k1 = (_WO_A1 / _WO_RHO) * np.exp(-_WO_B1 / T_R)
    k2 = (_WO_A2 / _WO_RHO) * np.exp(-_WO_B2 / T_R)
    k3 = (_WO_A3 / _WO_RHO) * np.exp(-_WO_B3 / T_R)

    # Apply activity factors
    k1_eff = y1 * k1
    k2_eff = y2 * k2

    # Solve mass balances iteratively (simple fixed-point)
    m = _WO_M
    F_fA = _WO_F_FA

    # Initial guesses
    X_A, X_B, X_C, X_E, X_P, X_G = 0.1, 0.1, 0.1, 0.1, 0.1, 0.1

    for _ in range(50):
        r1 = k1_eff * X_A * X_B
        r2 = k2_eff * X_B * X_C
        r3 = k3 * X_C * X_P

        # Mass balances (implicit form): 0 = Feed - Outflow + Generation
        # Rearrange to explicit update
        X_A_new = F_fA / (mu + r1 * m)
        X_B_new = (F_fB + r1 * m * X_A_new) / (mu + r1 * m + r2 * m)  # approximate
        X_B_new = F_fB / (mu + r1 * m * X_A_new / X_B + r2 * m * X_C / X_B) if X_B > 1e-10 else 0.1

        # Simpler approach: direct solution using quasi-steady state
        # For numerical stability, use relaxation
        alpha = 0.5

        # A: 0 = F_fA - μ*X_A - r1*m
        if mu + k1_eff * X_B * m > 1e-10:
            X_A_new = F_fA / (mu + k1_eff * X_B * m)
        X_A = alpha * X_A_new + (1 - alpha) * X_A

        # B: 0 = F_fB - μ*X_B - r1*m - r2*m
        if mu + k1_eff * X_A * m + k2_eff * X_C * m > 1e-10:
            X_B_new = F_fB / (mu + k1_eff * X_A * m + k2_eff * X_C * m)
        X_B = alpha * X_B_new + (1 - alpha) * X_B

        # C: 0 = -μ*X_C + 2*r1*m - 2*r2*m - r3*m
        r1_curr = k1_eff * X_A * X_B
        r2_curr = k2_eff * X_B * X_C
        if mu + 2 * k2_eff * X_B * m + k3 * X_P * m > 1e-10:
            X_C_new = 2 * r1_curr * m / (mu + 2 * k2_eff * X_B * m + k3 * X_P * m)
        X_C = alpha * X_C_new + (1 - alpha) * X_C

        # E: 0 = -μ*X_E + 2*r2*m
        r2_curr = k2_eff * X_B * X_C
        if mu > 1e-10:
            X_E_new = 2 * r2_curr * m / mu
        X_E = alpha * X_E_new + (1 - alpha) * X_E

        # P: 0 = -μ*X_P + r2*m - 0.5*r3*m
        r3_curr = k3 * X_C * X_P
        if mu + 0.5 * k3 * X_C * m > 1e-10:
            X_P_new = r2_curr * m / (mu + 0.5 * k3 * X_C * m)
        X_P = alpha * X_P_new + (1 - alpha) * X_P

        # G: 0 = -μ*X_G + 1.5*r3*m
        if mu > 1e-10:
            X_G_new = 1.5 * r3_curr * m / mu
        X_G = alpha * X_G_new + (1 - alpha) * X_G

    # Product and waste flows
    F_pP = mu * X_P - 0.1 * mu * X_E  # Net product (penalize E)
    F_wG = mu * X_G

    return X_A, X_B, X_C, X_E, X_P, X_G, F_pP, F_wG


def _williams_otto_J(x_wb: np.ndarray, y: np.ndarray) -> float:
    """
    Objective J(x^{WB}, y) for Williams-Otto: net product flow.

    Args:
        x_wb: [T, F_fB, μ] - temperature, B feed, total outflow
        y: [y1, y2] - catalyst activity factors

    Returns:
        Net product flow (to maximize)
    """
    T, F_fB, mu = x_wb[0], x_wb[1], x_wb[2]
    y1, y2 = y[0], y[1]

    _, _, _, _, _, _, F_pP, _ = _williams_otto_solve_mass_balance(T, F_fB, mu, y1, y2)

    return F_pP


def _williams_otto_g(x_wb: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Constraints g(x^{WB}, y) for Williams-Otto (≤ 0 is feasible).

    g1: Waste limit F_wG <= 1.0
    g2: Mass balance closure sum(X_i) >= 0.5
    """
    T, F_fB, mu = x_wb[0], x_wb[1], x_wb[2]
    y1, y2 = y[0], y[1]

    X_A, X_B, X_C, X_E, X_P, X_G, _, F_wG = _williams_otto_solve_mass_balance(T, F_fB, mu, y1, y2)

    # g1: F_wG <= 1.0
    g1 = F_wG - 1.0

    # g2: Mass balance closure
    sum_X = X_A + X_B + X_C + X_E + X_P + X_G
    g2 = 0.5 - sum_X

    return np.array([g1, g2])


def create_williams_otto() -> BiLevelProblem:
    """
    Create Williams-Otto Process bi-level optimization problem.

    This is a constrained maximization problem where:
        x^{WB} = [T, F_fB, μ]  - white-box (process) variables
            T: Temperature in degrees Rankine (°R), range [400, 700]
            F_fB: Feed flow rate of B (klb/h), range [5, 50]
            μ: Total outlet flow rate (klb/h), range [50, 200]
        x^{BB} = [ε, σ_c]      - black-box (catalyst properties)
            ε: Surface energy (kJ/mol), range [5, 25]
            σ_c: Pore size (Å), range [3, 10]
        y = [y1, y2]           - activity factors

    Constraints:
        g1: Waste flow F_wG <= 1.0 klb/h
        g2: Mass balance closure sum(X_i) >= 0.5
    """
    return BiLevelProblem(
        name="Williams-Otto",
        n_x_wb=3,
        n_x_bb=2,
        n_y=2,
        n_g=2,
        x_wb_lower=np.array([400.0, 5.0, 50.0]),
        x_wb_upper=np.array([700.0, 50.0, 200.0]),
        x_bb_lower=np.array([5.0, 3.0]),
        x_bb_upper=np.array([25.0, 10.0]),
        fbb=_williams_otto_fbb,
        J=_williams_otto_J,
        g=_williams_otto_g,
        J_grad_x_wb=None,  # Complex due to implicit mass balances
        g_grad_x_wb=None,
        x_wb_optimal=np.array([581.8716719106122, 50.0, 50.0]),
        x_bb_optimal=np.array([12.607067688256025, 6.114130387437894]),
        J_optimal=5.469761902658782,
        maximize=True
    )


# =============================================================================
# Utility Functions
# =============================================================================

def get_all_problems() -> dict:
    """Get dictionary of all available bi-level test problems (ordered as in main.tex appendix)."""
    return {
        # Small feasible region problems (not gray-box yet)
        "Small-Feasible-Region": create_small_feasible_region(),
        "Small-Feasible-Region-2": create_small_feasible_region2(),
        # COBALT-style benchmark problems
        "Rastrigin": create_rastrigin(),
        "Toy-Hydrology": create_toy_hydrology(),
        "Rosen-Suzuki": create_rosen_suzuki(),
        # Chemical engineering problems
        "CSTR": create_cstr(),
        "Heat-Exchanger": create_heat_exchanger(),
        "PSA": create_psa(),
        "Batch-Reactor": create_batch_reactor(),
        "Distillation": create_distillation(),
        "Evaporator": create_evaporator(),
        "Membrane": create_membrane(),
        "Williams-Otto": create_williams_otto(),
    }


def get_cobalt_problems() -> dict:
    """Get dictionary of COBALT benchmark problems (minimization)."""
    return {
        "Rastrigin": create_rastrigin(),
        "Toy-Hydrology": create_toy_hydrology(),
        "Rosen-Suzuki": create_rosen_suzuki(),
    }


def get_constrained_problems() -> dict:
    """Get dictionary of problems with constraints (small feasible region)."""
    return {
        "Small-Feasible-Region": create_small_feasible_region(),
        "Small-Feasible-Region-2": create_small_feasible_region2(),
    }


def get_chemical_engineering_problems() -> dict:
    """Get dictionary of chemical engineering process design problems."""
    return {
        "CSTR": create_cstr(),
        "Heat-Exchanger": create_heat_exchanger(),
        "PSA": create_psa(),
        "Batch-Reactor": create_batch_reactor(),
        "Distillation": create_distillation(),
        "Evaporator": create_evaporator(),
        "Membrane": create_membrane(),
        "Williams-Otto": create_williams_otto(),
    }


# =============================================================================
# Main: Test problems
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Bi-Level Optimization Problems")
    print("=" * 70)

    problems = get_all_problems()

    for name, problem in problems.items():
        print(f"\n{name}:")
        print(f"  Dimensions: n_x_wb={problem.n_x_wb}, n_x_bb={problem.n_x_bb}")
        print(f"  Constraints: n_g={problem.n_g}")
        print(f"  Maximize: {problem.maximize}")
        print(f"  J* = {problem.J_optimal}")

        # Test at optimal
        if problem.x_wb_optimal is not None and problem.x_bb_optimal is not None:
            x_opt = problem.combine_x(problem.x_wb_optimal, problem.x_bb_optimal)
            J_test = problem.evaluate_blackbox(x_opt)
            print(f"  J(x*) = {J_test:.6f}")

            # Test bi-level evaluation
            J_bilevel, x_wb_opt = problem.evaluate_bilevel(
                problem.x_bb_optimal, n_starts=20, return_x_wb=True
            )
            print(f"  J_bilevel(x_bb*) = {J_bilevel:.6f}")
            print(f"  x_wb_bilevel = {x_wb_opt}")
