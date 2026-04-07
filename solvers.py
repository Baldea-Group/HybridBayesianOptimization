"""
solvers.py - Optimization solvers for bi-level problems

This module implements three optimization approaches:

1. Multi-start SLSQP (solve_multistart_slsqp):
   - Multi-start gradient-based optimization over all variables
   - Uses scipy.optimize.minimize with SLSQP

2. Black-box BO (solve_blackbox_bo):
   - Bayesian Optimization over all variables [x^{WB}, x^{BB}]
   - Treats entire problem as black-box
   - Uses GP with configurable acquisition function

3. Hybrid/Bi-level BO (solve_bilevel_bo):
   - Bayesian Optimization over black-box variables x^{BB} only
   - Inner NLP optimization over white-box variables x^{WB}
   - Exploits known structure of J(x^{WB}, y)

Acquisition Functions:
   - EI: Expected Improvement (default)
   - PI: Probability of Improvement
   - LCB/UCB: Lower/Upper Confidence Bound
   - mWB2: Modified Watson-Barnes 2 (from COBALT paper)
   - Thompson: Thompson Sampling
"""

import numpy as np
import time
from typing import Optional, Tuple, Dict, Any, Literal, List, Callable
from scipy.optimize import minimize, basinhopping
from scipy.stats import norm
from scipy.stats.qmc import LatinHypercube, Sobol
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel, RBF, Matern

from functions import BiLevelProblem


# =============================================================================
# List of Available Acquisition Functions
# =============================================================================

AVAILABLE_ACQUISITIONS = ['ei', 'pi', 'lcb', 'mwb2', 'thompson']


def get_available_acquisitions() -> List[str]:
    """Return list of available acquisition function names."""
    return AVAILABLE_ACQUISITIONS.copy()


# =============================================================================
# Acquisition Functions
# =============================================================================

def expected_improvement(
    X: np.ndarray,
    gpr: GaussianProcessRegressor,
    y_best: float,
    xi: float = 0.01,
    maximize: bool = False
) -> np.ndarray:
    """
    Expected Improvement acquisition function.

    EI(x) = E[max(0, f* - f(x))]  for minimization
    EI(x) = E[max(0, f(x) - f*)]  for maximization

    Args:
        X: Points to evaluate, shape (n_points, n_dim)
        gpr: Fitted Gaussian Process regressor
        y_best: Best observed value so far
        xi: Exploration-exploitation trade-off parameter
        maximize: If True, maximize objective; if False, minimize

    Returns:
        EI values, shape (n_points,)
    """
    X = np.atleast_2d(X)
    mu, sigma = gpr.predict(X, return_std=True)

    with np.errstate(divide='warn'):
        if maximize:
            # For maximization: EI = (mu - y_best - xi) * Phi(Z) + sigma * phi(Z)
            imp = mu - y_best - xi
        else:
            # For minimization: EI = (y_best - mu - xi) * Phi(Z) + sigma * phi(Z)
            imp = y_best - mu - xi

        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei


def probability_of_improvement(
    X: np.ndarray,
    gpr: GaussianProcessRegressor,
    y_best: float,
    xi: float = 0.01,
    maximize: bool = False
) -> np.ndarray:
    """
    Probability of Improvement acquisition function.

    PI(x) = P(f(x) < f*)  for minimization
    PI(x) = P(f(x) > f*)  for maximization

    Args:
        X: Points to evaluate, shape (n_points, n_dim)
        gpr: Fitted Gaussian Process regressor
        y_best: Best observed value so far
        xi: Exploration-exploitation trade-off parameter
        maximize: If True, maximize objective; if False, minimize

    Returns:
        PI values, shape (n_points,)
    """
    X = np.atleast_2d(X)
    mu, sigma = gpr.predict(X, return_std=True)

    with np.errstate(divide='warn'):
        if maximize:
            Z = (mu - y_best - xi) / sigma
        else:
            Z = (y_best - mu - xi) / sigma
        pi = norm.cdf(Z)
        pi[sigma == 0.0] = 0.0

    return pi


def lower_confidence_bound(
    X: np.ndarray,
    gpr: GaussianProcessRegressor,
    y_best: float = None,  # Not used but kept for interface consistency
    kappa: float = 2.0,
    maximize: bool = False
) -> np.ndarray:
    """
    Lower/Upper Confidence Bound acquisition function.

    For minimization: LCB = mu - kappa * sigma (lower is better, return negated)
    For maximization: UCB = mu + kappa * sigma (higher is better)

    Args:
        X: Points to evaluate, shape (n_points, n_dim)
        gpr: Fitted Gaussian Process regressor
        y_best: Not used (for interface consistency)
        kappa: Exploration parameter (higher = more exploration)
        maximize: If True, return UCB; if False, return -LCB

    Returns:
        Acquisition values, shape (n_points,) - higher is always better
    """
    X = np.atleast_2d(X)
    mu, sigma = gpr.predict(X, return_std=True)

    if maximize:
        return mu + kappa * sigma  # UCB - higher is better
    else:
        return -(mu - kappa * sigma)  # -LCB - higher is better (for argmax)


def modified_watson_barnes_2(
    X: np.ndarray,
    gpr: GaussianProcessRegressor,
    y_best: float,
    xi: float = 0.01,
    maximize: bool = False,
    n_iter: int = 1,
    n_total: int = 50
) -> np.ndarray:
    """
    Modified Watson-Barnes 2 (mWB2) acquisition function.

    From COBALT paper (Paulson 2022):
        mWB2(x) = s_n * EI(x) - f_hat(x)

    where:
        s_n = -3 * (1 - n/N) is a dynamic scaling factor
        n is current iteration, N is total budget
        f_hat(x) is the GP mean prediction

    This balances exploration (EI) with exploitation (mean prediction).

    Args:
        X: Points to evaluate, shape (n_points, n_dim)
        gpr: Fitted Gaussian Process regressor
        y_best: Best observed value so far
        xi: EI exploration parameter
        maximize: If True, maximize objective
        n_iter: Current iteration number
        n_total: Total iteration budget

    Returns:
        mWB2 values, shape (n_points,) - higher is always better
    """
    X = np.atleast_2d(X)
    mu, sigma = gpr.predict(X, return_std=True)

    # Compute EI component
    ei = expected_improvement(X, gpr, y_best, xi, maximize)

    # Dynamic scaling factor (decreases exploration as iterations progress)
    s_n = -3.0 * (1.0 - n_iter / n_total)

    if maximize:
        # For maximization: want high mu, so mWB2 = s_n * EI + mu
        # Higher is better
        mwb2 = s_n * ei + mu
    else:
        # For minimization: want low mu, so mWB2 = s_n * EI - mu
        # Higher is better (so we negate mu)
        mwb2 = s_n * ei - mu

    return mwb2


def thompson_sampling(
    X: np.ndarray,
    gpr: GaussianProcessRegressor,
    y_best: float = None,  # Not used
    maximize: bool = False,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Thompson Sampling acquisition function.

    Draws a sample from the GP posterior and returns it as the acquisition value.
    The point with the best sampled value is selected.

    Args:
        X: Points to evaluate, shape (n_points, n_dim)
        gpr: Fitted Gaussian Process regressor
        y_best: Not used (for interface consistency)
        maximize: If True, maximize objective
        rng: Random number generator

    Returns:
        Sampled function values, shape (n_points,) - higher is always better
    """
    X = np.atleast_2d(X)

    if rng is None:
        rng = np.random.default_rng()

    try:
        # Sample from posterior
        # sklearn's sample_y returns shape (n_samples, n_targets)
        samples = gpr.sample_y(X, n_samples=1, random_state=rng.integers(0, 2**31))
        samples = samples.flatten()
    except (np.linalg.LinAlgError, ValueError) as e:
        # SVD or other numerical issues - fall back to LCB with noise
        # This can happen when the covariance matrix is ill-conditioned
        mu, sigma = gpr.predict(X, return_std=True)
        # Add random noise scaled by sigma to simulate sampling
        noise = rng.standard_normal(len(mu))
        samples = mu + sigma * noise

    if maximize:
        return samples  # Higher is better
    else:
        return -samples  # Negate for minimization (higher is better for argmax)


def get_acquisition_function(
    name: str,
    maximize: bool = False,
    n_iter: int = 1,
    n_total: int = 50,
    rng: np.random.Generator = None,
    xi: float = 0.01,
    kappa: float = 2.0
) -> Callable:
    """
    Get acquisition function by name.

    Args:
        name: One of 'ei', 'pi', 'lcb'/'ucb', 'mwb2', 'thompson'
        maximize: Whether the problem is maximization
        n_iter: Current iteration (for mWB2)
        n_total: Total iterations (for mWB2)
        rng: Random generator (for Thompson sampling)
        xi: Exploration-exploitation trade-off for EI/PI/mWB2 (higher = more exploration)
        kappa: Exploration parameter for LCB/UCB (higher = more exploration)

    Returns:
        Acquisition function callable with signature (X, gpr, y_best) -> values
    """
    name = name.lower()

    if name == 'ei':
        return lambda X, gpr, y_best: expected_improvement(X, gpr, y_best, xi, maximize)
    elif name == 'pi':
        return lambda X, gpr, y_best: probability_of_improvement(X, gpr, y_best, xi, maximize)
    elif name in ['lcb', 'ucb']:
        return lambda X, gpr, y_best: lower_confidence_bound(X, gpr, y_best, kappa, maximize)
    elif name == 'mwb2':
        return lambda X, gpr, y_best: modified_watson_barnes_2(
            X, gpr, y_best, xi, maximize, n_iter, n_total
        )
    elif name == 'thompson':
        return lambda X, gpr, y_best: thompson_sampling(X, gpr, y_best, maximize, rng)
    else:
        raise ValueError(f"Unknown acquisition function: {name}. Available: {AVAILABLE_ACQUISITIONS}")


# =============================================================================
# Gaussian Process Setup
# =============================================================================

def create_gp(
    kernel_type: str = 'rbf',
    n_dim: int = 1,
    noise_level: float = 1e-5
) -> GaussianProcessRegressor:
    """
    Create a Gaussian Process regressor with specified kernel.

    Args:
        kernel_type: 'rbf' or 'matern'
        n_dim: Number of input dimensions
        noise_level: Noise level for WhiteKernel

    Returns:
        Configured GaussianProcessRegressor
    """
    # Lower length-scale bound of 0.01 prevents GP collapse to delta functions
    # (memorisation mode) which produces constant mean predictions away from
    # training points and kills acquisition-function guidance.
    if kernel_type.lower() == 'matern':
        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(1e-5, 1e3)) *
            Matern(length_scale=np.ones(n_dim), length_scale_bounds=(1e-2, 10.0), nu=2.5) +
            WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-10, 1e-1))
        )
    else:  # RBF
        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(1e-5, 1e3)) *
            RBF(length_scale=np.ones(n_dim), length_scale_bounds=(1e-2, 10.0)) +
            WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-10, 1e-1))
        )

    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=10
    )


def _normalize_inputs(X: np.ndarray, x_lower: np.ndarray, x_upper: np.ndarray) -> np.ndarray:
    """Normalize inputs to [0, 1]^d, matching MATLAB toUnit transform."""
    ranges = x_upper - x_lower
    ranges = np.where(ranges < 1e-12, 1.0, ranges)  # avoid division by zero
    return (X - x_lower) / ranges


def _optimize_acquisition(
    acq_func: Callable,
    gpr: GaussianProcessRegressor,
    y_best: float,
    n_dim: int,
    rng: np.random.Generator,
    n_sobol: int = 4096,
    n_local_starts: int = 5,
    refine: bool = False,
) -> np.ndarray:
    """Maximize acquisition function over [0,1]^n_dim using Sobol + L-BFGS-B.

    1. Evaluate acquisition on a Sobol quasi-random grid (better space-filling
       than uniform random, especially in low dimensions).
    2. Refine the top candidates with L-BFGS-B (bounded) using numerical
       gradients of the acquisition surface.

    Args:
        acq_func: Callable (X, gpr, y_best) -> values, shape (n_points,)
        gpr: Fitted GP regressor
        y_best: Best observed (GP-space) value
        n_dim: Dimensionality of the search space
        rng: Random number generator
        n_sobol: Number of Sobol points (rounded up to next power of 2)
        n_local_starts: Number of L-BFGS-B restarts from best Sobol points

    Returns:
        Best point in [0,1]^n_dim
    """
    # Round n_sobol up to next power of 2 (required by Sobol)
    m = int(np.ceil(np.log2(max(n_sobol, 2))))
    n_sobol_actual = 2 ** m

    # Sobol quasi-random grid — scrambled for stochastic tie-breaking
    sobol_engine = Sobol(d=n_dim, scramble=True, seed=int(rng.integers(0, 2**31)))
    X_sobol = sobol_engine.random(n_sobol_actual)  # shape (n_sobol_actual, n_dim)

    acq_values = acq_func(X_sobol, gpr, y_best)
    top_indices = np.argsort(acq_values)[-n_local_starts:]

    best_x = X_sobol[top_indices[-1]]
    best_val = acq_values[top_indices[-1]]

    if not refine:
        return best_x

    # L-BFGS-B refinement from each top candidate
    bounds_01 = [(0.0, 1.0)] * n_dim
    for idx in top_indices:
        x0 = X_sobol[idx]
        try:
            res = minimize(
                lambda x: -acq_func(x.reshape(1, -1), gpr, y_best).item(),
                x0,
                method='L-BFGS-B',
                bounds=bounds_01,
                options={'maxiter': 50, 'ftol': 1e-12},
            )
            if -res.fun > best_val:
                best_val = -res.fun
                best_x = res.x
        except (np.linalg.LinAlgError, ValueError):
            continue

    return best_x


def _adaptive_penalty_scale(Y_feasible: np.ndarray, sign: float) -> float:
    """Compute an adaptive penalty scale based on observed objective range.

    Returns a multiplier applied to max constraint violation, so that
    barely-infeasible points get a small penalty and deeply-infeasible
    points get a large one.  This keeps the GP surface smooth near the
    feasibility boundary — critical for constrained problems where the
    optimum often sits on or near that boundary.
    """
    if len(Y_feasible) == 0:
        return 10.0  # fallback before any feasible points seen
    y_range = np.ptp(Y_feasible)  # max - min of GP-space feasible values
    return max(10.0, 3.0 * y_range)


def _constraint_penalty(g_val: np.ndarray, penalty_scale: float) -> float:
    """Compute penalty proportional to constraint violation.

    Args:
        g_val: Constraint values (g <= 0 is feasible).
        penalty_scale: Multiplier from _adaptive_penalty_scale.

    Returns:
        penalty_scale * max(0, max(g))  — zero when feasible.
    """
    if g_val is None or len(g_val) == 0:
        return 0.0
    max_viol = max(0.0, float(np.max(g_val)))
    return penalty_scale * max_viol


# =============================================================================
# Solver 1: Multi-start SLSQP
# =============================================================================

def solve_multistart_slsqp(
    problem: BiLevelProblem,
    n_starts: int = 20,
    method: str = 'SLSQP',
    seed: int = 42,
    verbose: bool = True
) -> dict:
    """
    Solve problem as black-box using scipy.optimize.minimize.

    Black-box approach:
        min_{x^{WB}, x^{BB}}  J(x^{WB}, f^{BB}(x^{BB}))
        s.t.                  g(x^{WB}, f^{BB}(x^{BB})) ≤ 0

    Uses multi-start optimization to find global minimum.

    Args:
        problem: BiLevelProblem instance
        n_starts: Number of random restarts
        method: scipy.optimize method ('SLSQP', 'L-BFGS-B', etc.)
        seed: Random seed
        verbose: Whether to print progress

    Returns:
        Dictionary with optimization results
    """
    rng = np.random.default_rng(seed)
    problem.reset_counters()

    x_lower, x_upper = problem.get_full_bounds()
    bounds = list(zip(x_lower, x_upper))

    # For maximization, negate objective
    sign = -1.0 if problem.maximize else 1.0

    def objective(x):
        return sign * problem.evaluate_blackbox(x, count_eval=True)

    # Build constraints if present
    constraints = []
    if problem.g is not None:
        def constraint_fun(x):
            _, cons = problem.evaluate_blackbox_with_constraints(x, count_eval=False)
            return -cons  # scipy uses >= 0
        constraints.append({'type': 'ineq', 'fun': constraint_fun})

    best_res = None

    if verbose:
        print(f"Multi-start SLSQP optimization of {problem.name}")
        print(f"  Running {n_starts} multi-start optimizations...")

    for i in range(n_starts):
        x0 = rng.uniform(x_lower, x_upper)

        try:
            if constraints:
                res = minimize(objective, x0, method=method, bounds=bounds, constraints=constraints)
            else:
                res = minimize(objective, x0, method=method, bounds=bounds)

            if best_res is None or res.fun < best_res.fun:
                best_res = res

        except Exception as e:
            if verbose:
                print(f"    Start {i+1} failed: {e}")
            continue

    best_x_wb, best_x_bb = problem.split_x(best_res.x)
    best_J = sign * best_res.fun  # Convert back to original objective

    results = {
        'best_x': best_res.x,
        'best_x_wb': best_x_wb,
        'best_x_bb': best_x_bb,
        'best_J': best_J,
        'success': best_res.success,
        'n_fbb_evals': problem.n_fbb_evals,
        'scipy_result': best_res,
    }

    if verbose:
        print(f"  Optimization complete!")
        print(f"    Best J = {results['best_J']:.6f} (optimal: {problem.J_optimal})")
        print(f"    Best x = {results['best_x']}")
        print(f"    f^{{BB}} evaluations: {results['n_fbb_evals']}")

    return results


# =============================================================================
# Solver 2: Basin-Hopping
# =============================================================================

class _BoundedStep:
    """Uniform perturbation clipped to variable bounds."""
    def __init__(self, stepsize, lower, upper, rng):
        self.stepsize = stepsize
        self.lower = np.asarray(lower)
        self.upper = np.asarray(upper)
        self.rng = rng

    def __call__(self, x):
        x_new = x + self.rng.uniform(-self.stepsize, self.stepsize, size=x.shape)
        return np.clip(x_new, self.lower, self.upper)


class _HistoryCallback:
    """Callback for Basin-Hopping that records every local minimization."""
    def __init__(self, problem, sign):
        self.problem = problem
        self.sign = sign
        self.X_history = []
        self.Y_history = []
        self.G_history = []

    def __call__(self, x, f, accept):
        self.X_history.append(x.copy())
        self.Y_history.append(self.sign * f)  # convert back to true objective
        if self.problem.g is not None and self.problem.n_g > 0:
            _, g_val = self.problem.evaluate_blackbox_with_constraints(
                x, count_eval=False)
            self.G_history.append(g_val.copy())
        else:
            self.G_history.append(np.array([]))
        return False  # never stop early


def solve_basin_hopping(
    problem: BiLevelProblem,
    seed: int = 42,
    verbose: bool = True,
    return_history: bool = False,
    niter: int = 1000,
    **kwargs,
) -> dict:
    """
    Solve problem using Basin-Hopping with SLSQP local minimizer.

    Basin-Hopping provides global search via random perturbation + local
    SLSQP polish. High temperature (T=100) accepts all uphill moves,
    and full-range stepsize ensures each perturbation can reach any point
    in the domain.

    Args:
        problem: BiLevelProblem instance
        seed: Random seed
        verbose: Whether to print progress
        return_history: Whether to return evaluation-by-evaluation history
        niter: Number of Basin-Hopping iterations (default 1000)

    Returns:
        Dictionary with optimization results
    """
    problem.reset_counters()
    rng = np.random.default_rng(seed)

    x_lower, x_upper = problem.get_full_bounds()
    bounds = list(zip(x_lower, x_upper))
    has_constraints = problem.g is not None and problem.n_g > 0
    sign = -1.0 if problem.maximize else 1.0

    def objective(x):
        return sign * problem.evaluate_blackbox(x, count_eval=True)

    # SLSQP local minimizer with constraints
    minimizer_kwargs = {
        'method': 'SLSQP',
        'bounds': bounds,
        'options': {'maxiter': 500, 'ftol': 1e-14},
    }
    if has_constraints:
        def _con(x):
            _, g = problem.evaluate_blackbox_with_constraints(x, count_eval=False)
            return -g  # scipy uses >= 0
        minimizer_kwargs['constraints'] = [{'type': 'ineq', 'fun': _con}]

    # Stepsize = half of the maximum variable range
    ranges = x_upper - x_lower
    stepsize = np.max(ranges) / 2.0

    # History callback
    history = _HistoryCallback(problem, sign)

    if verbose:
        print(f"Global optimization of {problem.name} (Basin-Hopping)")
        if has_constraints:
            print(f"  Problem has {problem.n_g} constraints")
        print(f"  niter={niter}, T=100, stepsize={stepsize:.3f}")

    x0 = rng.uniform(x_lower, x_upper)

    bh_result = basinhopping(
        objective, x0,
        minimizer_kwargs=minimizer_kwargs,
        niter=niter,
        T=100.0,
        seed=int(rng.integers(0, 2**31)),
        take_step=_BoundedStep(stepsize, x_lower, x_upper, rng),
        callback=history if return_history else None,
    )

    best_x = bh_result.x
    best_x_wb, best_x_bb = problem.split_x(best_x)
    best_J = sign * bh_result.fun

    results = {
        'best_x': best_x,
        'best_x_wb': best_x_wb,
        'best_x_bb': best_x_bb,
        'best_J': best_J,
        'n_fbb_evals': problem.n_fbb_evals,
        'n_J_evals': problem.n_J_evals,
    }

    if has_constraints:
        _, best_g = problem.evaluate_blackbox_with_constraints(
            best_x, count_eval=False)
        results['best_g'] = best_g
        results['is_feasible'] = np.all(best_g <= 1e-6)

    if return_history and len(history.X_history) > 0:
        results['X_history'] = np.array(history.X_history)
        results['Y_history'] = np.array(history.Y_history)
        if has_constraints and len(history.G_history) > 0 and history.G_history[0].size > 0:
            results['G_history'] = np.array(history.G_history)

    if verbose:
        print(f"  Optimization complete!")
        print(f"    Best J = {best_J:.6f} (optimal: {problem.J_optimal})")
        print(f"    Best x = {best_x}")
        if has_constraints:
            print(f"    Feasible: {results.get('is_feasible', True)}, g = {results.get('best_g')}")
        print(f"    f^{{BB}} evaluations: {problem.n_fbb_evals}")

    return results


# =============================================================================
# Solver 3: Black-box Bayesian Optimization
# =============================================================================

def solve_blackbox_bo(
    problem: BiLevelProblem,
    n_iterations: int = 50,
    n_initial: int = 5,
    penalty: float = None,
    acquisition: str = 'ei',
    kernel_type: str = 'rbf',
    seed: int = 42,
    verbose: bool = True,
    return_history: bool = False,
    xi: float = 0.01,
    kappa: float = 2.0
) -> dict:
    """
    Solve problem as black-box using Bayesian Optimization.

    Black-box approach:
        min_{x^{WB}, x^{BB}}  J(x^{WB}, f^{BB}(x^{BB}))
        s.t.                  g(x^{WB}, f^{BB}(x^{BB})) ≤ 0

    Constraints are handled via penalty: if infeasible, objective is penalized.

    Args:
        problem: BiLevelProblem instance
        n_iterations: Number of BO iterations
        n_initial: Number of initial random samples
        penalty: Penalty value for infeasible points
        acquisition: Acquisition function ('ei', 'pi', 'lcb', 'mwb2', 'thompson')
        kernel_type: GP kernel type ('rbf', 'matern')
        seed: Random seed
        verbose: Whether to print progress
        return_history: Whether to return full optimization history
        xi: Exploration-exploitation trade-off for EI/PI/mWB2 (higher = more exploration)
        kappa: Exploration parameter for LCB/UCB (higher = more exploration)

    Returns:
        Dictionary with optimization results
    """
    rng = np.random.default_rng(seed)
    problem.reset_counters()

    x_lower, x_upper = problem.get_full_bounds()
    n_x = problem.n_x_wb + problem.n_x_bb
    has_constraints = problem.g is not None and problem.n_g > 0
    maximize = problem.maximize
    n_total = n_initial + n_iterations
    use_adaptive_penalty = penalty is None

    # For GP training, we always minimize (negate for maximization problems)
    sign = -1.0 if maximize else 1.0

    # Storage
    X_sample = []
    Y_sample = []  # Values for GP (sign-adjusted and penalized)
    Y_true = []    # True objective values
    G_sample = []  # Constraint values
    Y_feasible_gp = []  # GP-space values for feasible points (for adaptive penalty)

    def penalized_objective(J_val, g_val, current_penalty_scale):
        """Apply penalty proportional to constraint violation."""
        y = sign * J_val  # Convert to minimization for GP
        return y + _constraint_penalty(g_val, current_penalty_scale)

    if verbose:
        print(f"Black-box BO optimization of {problem.name}")
        print(f"  Acquisition function: {acquisition.upper()}")
        if has_constraints:
            pen_str = "adaptive" if use_adaptive_penalty else f"{penalty}"
            print(f"  Problem has {problem.n_g} constraints (penalty={pen_str})")
        print(f"  Generating {n_initial} initial samples (LHS)...")

    # Initial sampling using Latin Hypercube Sampling
    lhs_sampler = LatinHypercube(d=n_x, seed=seed)
    X_lhs = lhs_sampler.random(n=n_initial)  # Samples in [0, 1]^n_x
    X_lhs = x_lower + X_lhs * (x_upper - x_lower)  # Scale to bounds

    for x in X_lhs:
        J_val, g_val = problem.evaluate_blackbox_with_constraints(x)
        X_sample.append(x)
        Y_true.append(J_val)
        G_sample.append(g_val if has_constraints else np.array([]))
        y_gp = sign * J_val
        is_feas = (g_val is None or len(g_val) == 0 or np.max(g_val) <= 1e-6)
        if is_feas:
            Y_feasible_gp.append(y_gp)
        # Temporarily store un-penalized GP value; we'll compute penalties after
        Y_sample.append(y_gp)

    # Compute adaptive penalty scale from initial feasible points and re-penalize
    if use_adaptive_penalty:
        penalty = _adaptive_penalty_scale(np.array(Y_feasible_gp), sign)
    Y_sample_recomputed = []
    for i_init in range(len(X_sample)):
        g_val = G_sample[i_init]
        y_gp = sign * Y_true[i_init]
        y_gp += _constraint_penalty(g_val, penalty)
        Y_sample_recomputed.append(y_gp)

    X_sample = np.array(X_sample)
    Y_sample = np.array(Y_sample_recomputed)
    Y_true = np.array(Y_true)
    G_sample = np.array(G_sample) if has_constraints else None

    # GP setup — fit in normalized [0,1]^d space
    gpr = create_gp(kernel_type=kernel_type, n_dim=n_x)

    # Track per-iteration timing
    iter_times = []

    if verbose:
        print(f"  Running {n_iterations} BO iterations...")

    for i in range(n_iterations):
        iter_start = time.perf_counter()

        # Normalize inputs to [0,1]^d for GP fitting
        X_norm = _normalize_inputs(X_sample, x_lower, x_upper)

        # Fit GP on normalized inputs with error handling
        try:
            gpr.fit(X_norm, Y_sample)
        except (np.linalg.LinAlgError, ValueError) as e:
            if verbose:
                print(f"    Warning: GP fitting failed at iteration {i+1}: {e}")
                print(f"    Falling back to random sampling for this iteration.")
            x_next = rng.uniform(x_lower, x_upper)
            J_next, g_next = problem.evaluate_blackbox_with_constraints(x_next)
            y_gp_next = sign * J_next
            is_feas = not has_constraints or len(g_next) == 0 or np.max(g_next) <= 1e-6
            if is_feas:
                Y_feasible_gp.append(y_gp_next)
                if use_adaptive_penalty:
                    penalty = _adaptive_penalty_scale(np.array(Y_feasible_gp), sign)
            y_penalized = penalized_objective(J_next, g_next, penalty)
            X_sample = np.vstack([X_sample, x_next])
            Y_sample = np.append(Y_sample, y_penalized)
            Y_true = np.append(Y_true, J_next)
            if has_constraints:
                G_sample = np.vstack([G_sample, g_next])
            iter_times.append(time.perf_counter() - iter_start)
            continue

        # Get acquisition function (recreate each iteration for mWB2/Thompson)
        acq_func = get_acquisition_function(
            acquisition,
            maximize=False,  # GP is always minimization
            n_iter=n_initial + i,
            n_total=n_total,
            rng=rng,
            xi=xi,
            kappa=kappa
        )

        # Sobol quasi-random acquisition optimization in normalized [0,1]^d
        try:
            best_norm = _optimize_acquisition(
                acq_func, gpr, np.min(Y_sample), n_x, rng,
            )
            x_next = x_lower + best_norm * (x_upper - x_lower)
        except (np.linalg.LinAlgError, ValueError) as e:
            if verbose:
                print(f"    Warning: Acquisition optimization failed at iteration {i+1}: {e}")
                print(f"    Falling back to random sampling for this iteration.")
            x_next = rng.uniform(x_lower, x_upper)

        # Evaluate
        J_next, g_next = problem.evaluate_blackbox_with_constraints(x_next)
        y_gp_next = sign * J_next
        is_feas = not has_constraints or len(g_next) == 0 or np.max(g_next) <= 1e-6
        if is_feas:
            Y_feasible_gp.append(y_gp_next)
            if use_adaptive_penalty:
                penalty = _adaptive_penalty_scale(np.array(Y_feasible_gp), sign)
        y_penalized = penalized_objective(J_next, g_next, penalty)

        # Update samples
        X_sample = np.vstack([X_sample, x_next])
        Y_sample = np.append(Y_sample, y_penalized)
        Y_true = np.append(Y_true, J_next)
        if has_constraints:
            G_sample = np.vstack([G_sample, g_next])

        # Record iteration time
        iter_times.append(time.perf_counter() - iter_start)

        if verbose and (i + 1) % 10 == 0:
            if has_constraints:
                feasible_mask = np.all(G_sample <= 1e-6, axis=1)
                if np.any(feasible_mask):
                    if maximize:
                        best_feas = np.max(Y_true[feasible_mask])
                    else:
                        best_feas = np.min(Y_true[feasible_mask])
                else:
                    best_feas = np.inf if not maximize else -np.inf
                print(f"    Iteration {i + 1}: best_feasible_J = {best_feas:.6f}")
            else:
                if maximize:
                    print(f"    Iteration {i + 1}: best_J = {np.max(Y_true):.6f}")
                else:
                    print(f"    Iteration {i + 1}: best_J = {np.min(Y_sample):.6f}")

    # Find best feasible solution
    if has_constraints:
        feasible_mask = np.all(G_sample <= 1e-6, axis=1)
        if np.any(feasible_mask):
            feasible_Y = Y_true.copy()
            if maximize:
                feasible_Y[~feasible_mask] = -np.inf
                best_idx = np.argmax(feasible_Y)
            else:
                feasible_Y[~feasible_mask] = np.inf
                best_idx = np.argmin(feasible_Y)
        else:
            max_violation = np.max(G_sample, axis=1)
            best_idx = np.argmin(max_violation)
            if verbose:
                print("  Warning: No feasible solution found!")
    else:
        if maximize:
            best_idx = np.argmax(Y_true)
        else:
            best_idx = np.argmin(Y_sample)

    best_x = X_sample[best_idx]
    best_x_wb, best_x_bb = problem.split_x(best_x)

    results = {
        'best_x': best_x,
        'best_x_wb': best_x_wb,
        'best_x_bb': best_x_bb,
        'best_J': Y_true[best_idx],
        'n_fbb_evals': problem.n_fbb_evals,
    }

    if has_constraints:
        results['best_g'] = G_sample[best_idx]
        results['is_feasible'] = np.all(G_sample[best_idx] <= 1e-6)

    if return_history:
        results['X_history'] = X_sample
        results['Y_history'] = Y_true
        results['iter_times'] = np.array(iter_times)
        if has_constraints:
            results['G_history'] = G_sample

    if verbose:
        print(f"  Optimization complete!")
        print(f"    Best J = {results['best_J']:.6f} (optimal: {problem.J_optimal})")
        print(f"    Best x = {results['best_x']}")
        if has_constraints:
            print(f"    Feasible: {results['is_feasible']}, g = {results['best_g']}")
        print(f"    f^{{BB}} evaluations: {results['n_fbb_evals']}")

    return results


# =============================================================================
# Solver 4: Bi-level (Hybrid) Bayesian Optimization
# =============================================================================

def solve_bilevel_bo(
    problem: BiLevelProblem,
    n_iterations: int = 50,
    n_initial: int = 5,
    n_inner_starts: int = 20,
    penalty: float = None,
    acquisition: str = 'ei',
    kernel_type: str = 'rbf',
    seed: int = 42,
    verbose: bool = True,
    return_history: bool = False,
    xi: float = 0.01,
    kappa: float = 2.0,
    inner_solver: str = 'global'
) -> dict:
    """
    Solve bi-level problem using Bayesian Optimization for outer problem.

    Bi-level approach:
        Outer (BO):   min_{x^{BB}}  J*(x^{BB})
        Inner (NLP):  J*(x^{BB}) = min_{x^{WB}} { J(x^{WB}, f^{BB}(x^{BB})) | g(x^{WB}, y) ≤ 0 }

    Exploits known structure of J(x^{WB}, y) to solve inner problem exactly.

    Args:
        problem: BiLevelProblem instance
        n_iterations: Number of BO iterations for outer problem
        n_initial: Number of initial random samples
        n_inner_starts: Number of multi-starts for inner NLP
        penalty: Penalty value for infeasible inner solutions
        acquisition: Acquisition function ('ei', 'pi', 'lcb', 'mwb2', 'thompson')
        kernel_type: GP kernel type ('rbf', 'matern')
        seed: Random seed
        verbose: Whether to print progress
        return_history: Whether to return full optimization history
        xi: Exploration-exploitation trade-off for EI/PI/mWB2 (higher = more exploration)
        kappa: Exploration parameter for LCB/UCB (higher = more exploration)
        inner_solver: Inner optimization method ('multistart' or 'global')

    Returns:
        Dictionary with optimization results
    """
    rng = np.random.default_rng(seed)
    problem.reset_counters()

    has_constraints = problem.g is not None and problem.n_g > 0
    maximize = problem.maximize
    n_total = n_initial + n_iterations
    use_adaptive_penalty = penalty is None

    # For GP training, we always minimize
    sign = -1.0 if maximize else 1.0

    # Storage
    X_bb_sample = []
    X_wb_sample = []
    Y_sample = []      # Values for GP
    Y_true = []        # True objective
    G_sample = []      # Constraint values at optimal x_wb
    Feasible = []      # Whether inner solution is feasible
    Y_feasible_gp = [] # GP-space values for feasible points (for adaptive penalty)

    def check_feasibility(x_wb, pi):
        """Check if solution satisfies constraints."""
        if problem.g is None:
            return True, np.array([])
        g_val = problem.g(x_wb, pi)
        is_feasible = np.all(g_val <= 1e-6)
        return is_feasible, g_val

    if verbose:
        print(f"Bi-level BO optimization of {problem.name}")
        print(f"  Inner solver: {inner_solver}")
        print(f"  Acquisition function: {acquisition.upper()}")
        if has_constraints:
            pen_str = "adaptive" if use_adaptive_penalty else f"{penalty}"
            print(f"  Problem has {problem.n_g} constraints (penalty={pen_str})")
        print(f"  Generating {n_initial} initial samples (LHS)...")

    # Initial sampling using Latin Hypercube Sampling
    lhs_sampler = LatinHypercube(d=problem.n_x_bb, seed=seed)
    X_bb_lhs = lhs_sampler.random(n=n_initial)  # Samples in [0, 1]^n_x_bb
    X_bb_lhs = problem.x_bb_lower + X_bb_lhs * (problem.x_bb_upper - problem.x_bb_lower)

    for x_bb in X_bb_lhs:
        J_val, x_wb_opt = problem.evaluate_bilevel(
            x_bb, n_starts=n_inner_starts, rng=rng, return_x_wb=True,
            inner_solver=inner_solver
        )

        # Check feasibility of inner solution
        pi = problem.fbb(x_bb)
        is_feas, g_val = check_feasibility(x_wb_opt, pi)

        X_bb_sample.append(x_bb)
        X_wb_sample.append(x_wb_opt)
        Y_true.append(J_val)
        G_sample.append(g_val if has_constraints else np.array([]))
        Feasible.append(is_feas)

        y_gp = sign * J_val
        if is_feas:
            Y_feasible_gp.append(y_gp)
        # Store un-penalized for now; penalties applied after initial loop
        Y_sample.append(y_gp)

    # Compute adaptive penalty scale from initial feasible points and re-penalize
    if use_adaptive_penalty:
        penalty = _adaptive_penalty_scale(np.array(Y_feasible_gp), sign)
    Y_sample_recomputed = []
    for i_init in range(len(X_bb_sample)):
        y_gp = sign * Y_true[i_init]
        g_val = G_sample[i_init] if has_constraints else np.array([])
        y_gp += _constraint_penalty(g_val, penalty)
        Y_sample_recomputed.append(y_gp)

    X_bb_sample = np.array(X_bb_sample)
    Y_sample = np.array(Y_sample_recomputed)
    Y_true = np.array(Y_true)
    G_sample = np.array(G_sample) if has_constraints else None
    Feasible = np.array(Feasible)

    # GP setup — fit in normalized [0,1]^d space
    gpr = create_gp(kernel_type=kernel_type, n_dim=problem.n_x_bb)

    # Track per-iteration timing
    iter_times = []

    if verbose:
        print(f"  Running {n_iterations} BO iterations...")

    bb_lower = problem.x_bb_lower
    bb_upper = problem.x_bb_upper

    for i in range(n_iterations):
        iter_start = time.perf_counter()

        # Normalize inputs to [0,1]^d for GP fitting
        X_bb_norm = _normalize_inputs(X_bb_sample, bb_lower, bb_upper)

        # Fit GP on normalized inputs with error handling
        try:
            gpr.fit(X_bb_norm, Y_sample)
        except (np.linalg.LinAlgError, ValueError) as e:
            if verbose:
                print(f"    Warning: GP fitting failed at iteration {i+1}: {e}")
                print(f"    Falling back to random sampling for this iteration.")
            x_bb_next = rng.uniform(bb_lower, bb_upper)
            J_next, x_wb_next = problem.evaluate_bilevel(
                x_bb_next, n_starts=n_inner_starts, rng=rng, return_x_wb=True,
                inner_solver=inner_solver
            )
            pi_next = problem.fbb(x_bb_next)
            is_feas, g_next = check_feasibility(x_wb_next, pi_next)
            X_bb_sample = np.vstack([X_bb_sample, x_bb_next])
            X_wb_sample.append(x_wb_next)
            Y_true = np.append(Y_true, J_next)
            Feasible = np.append(Feasible, is_feas)
            if has_constraints:
                G_sample = np.vstack([G_sample, g_next])
            y_gp = sign * J_next
            if is_feas:
                Y_feasible_gp.append(y_gp)
                if use_adaptive_penalty:
                    penalty = _adaptive_penalty_scale(np.array(Y_feasible_gp), sign)
            y_gp += _constraint_penalty(g_next, penalty)
            Y_sample = np.append(Y_sample, y_gp)
            iter_times.append(time.perf_counter() - iter_start)
            continue

        # Get acquisition function (recreate each iteration for mWB2/Thompson)
        acq_func = get_acquisition_function(
            acquisition,
            maximize=False,  # GP is always minimization
            n_iter=n_initial + i,
            n_total=n_total,
            rng=rng,
            xi=xi,
            kappa=kappa
        )

        # Sobol quasi-random acquisition optimization in normalized [0,1]^d
        try:
            best_norm = _optimize_acquisition(
                acq_func, gpr, np.min(Y_sample), problem.n_x_bb, rng,
            )
            x_bb_next = bb_lower + best_norm * (bb_upper - bb_lower)
        except (np.linalg.LinAlgError, ValueError) as e:
            if verbose:
                print(f"    Warning: Acquisition optimization failed at iteration {i+1}: {e}")
                print(f"    Falling back to random sampling for this iteration.")
            x_bb_next = rng.uniform(bb_lower, bb_upper)

        J_next, x_wb_next = problem.evaluate_bilevel(
            x_bb_next, n_starts=n_inner_starts, rng=rng, return_x_wb=True,
            inner_solver=inner_solver
        )

        # Check feasibility
        pi_next = problem.fbb(x_bb_next)
        is_feas, g_next = check_feasibility(x_wb_next, pi_next)

        # Update samples
        X_bb_sample = np.vstack([X_bb_sample, x_bb_next])
        X_wb_sample.append(x_wb_next)
        Y_true = np.append(Y_true, J_next)
        Feasible = np.append(Feasible, is_feas)

        if has_constraints:
            G_sample = np.vstack([G_sample, g_next])

        y_gp = sign * J_next
        if is_feas:
            Y_feasible_gp.append(y_gp)
            if use_adaptive_penalty:
                penalty = _adaptive_penalty_scale(np.array(Y_feasible_gp), sign)
        y_gp += _constraint_penalty(g_next, penalty)
        Y_sample = np.append(Y_sample, y_gp)

        # Record iteration time
        iter_times.append(time.perf_counter() - iter_start)

        if verbose and (i + 1) % 10 == 0:
            if has_constraints:
                if np.any(Feasible):
                    if maximize:
                        best_feas = np.max(Y_true[Feasible])
                    else:
                        best_feas = np.min(Y_true[Feasible])
                else:
                    best_feas = np.inf if not maximize else -np.inf
                print(f"    Iteration {i + 1}: best_feasible_J = {best_feas:.6f}")
            else:
                if maximize:
                    print(f"    Iteration {i + 1}: best_J = {np.max(Y_true):.6f}")
                else:
                    print(f"    Iteration {i + 1}: best_J = {np.min(Y_sample):.6f}")

    # Find best feasible solution
    if has_constraints:
        if np.any(Feasible):
            feasible_Y = Y_true.copy()
            if maximize:
                feasible_Y[~Feasible] = -np.inf
                best_idx = np.argmax(feasible_Y)
            else:
                feasible_Y[~Feasible] = np.inf
                best_idx = np.argmin(feasible_Y)
        else:
            max_violation = np.max(G_sample, axis=1)
            best_idx = np.argmin(max_violation)
            if verbose:
                print("  Warning: No feasible solution found!")
    else:
        if maximize:
            best_idx = np.argmax(Y_true)
        else:
            best_idx = np.argmin(Y_sample)

    results = {
        'best_x_bb': X_bb_sample[best_idx],
        'best_x_wb': X_wb_sample[best_idx],
        'best_J': Y_true[best_idx],
        'n_fbb_evals': problem.n_fbb_evals,
    }

    if has_constraints:
        results['best_g'] = G_sample[best_idx]
        results['is_feasible'] = Feasible[best_idx]

    if return_history:
        results['X_bb_history'] = X_bb_sample
        results['X_wb_history'] = X_wb_sample
        results['Y_history'] = Y_true
        results['iter_times'] = np.array(iter_times)
        if has_constraints:
            results['G_history'] = G_sample
            results['Feasible_history'] = Feasible

    if verbose:
        print(f"  Optimization complete!")
        print(f"    Best J = {results['best_J']:.6f} (optimal: {problem.J_optimal})")
        print(f"    Best x_bb = {results['best_x_bb']}")
        print(f"    Best x_wb = {results['best_x_wb']}")
        if has_constraints:
            print(f"    Feasible: {results['is_feasible']}, g = {results['best_g']}")
        print(f"    f^{{BB}} evaluations: {results['n_fbb_evals']}")

    return results


# =============================================================================
# Utility Functions
# =============================================================================

def compute_regret(
    Y_history: np.ndarray,
    J_optimal: float,
    G_history: Optional[np.ndarray] = None,
    maximize: bool = False
) -> np.ndarray:
    """
    Compute regret (best so far vs global optimum) as a function of iteration.

    For constrained problems, infeasible points are ignored when computing
    the current best.

    Args:
        Y_history: Objective values at each iteration
        J_optimal: Global optimum value J*
        G_history: Constraint values at each iteration (g <= 0 is feasible)
        maximize: If True, problem is maximization

    Returns:
        Regret at each iteration
    """
    n = len(Y_history)
    best_so_far = np.full(n, np.inf if not maximize else -np.inf)

    for i in range(n):
        # Check feasibility
        if G_history is not None:
            is_feasible = np.all(G_history[i] <= 1e-6)
        else:
            is_feasible = True

        if is_feasible:
            if i == 0:
                best_so_far[i] = Y_history[i]
            else:
                if maximize:
                    best_so_far[i] = max(best_so_far[i-1], Y_history[i])
                else:
                    best_so_far[i] = min(best_so_far[i-1], Y_history[i])
        else:
            if i > 0:
                best_so_far[i] = best_so_far[i-1]

    if maximize:
        regret = J_optimal - best_so_far  # Positive regret for maximization
    else:
        regret = best_so_far - J_optimal  # Positive regret for minimization

    return regret


# =============================================================================
# Main: Test solvers
# =============================================================================

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')

    from functions import get_all_problems

    print("=" * 70)
    print("Testing Optimization Solvers")
    print("=" * 70)

    # Test on Rastrigin (minimization, unconstrained)
    from functions import create_rastrigin
    problem = create_rastrigin()

    print(f"\nProblem: {problem.name}")
    print(f"  Optimal: J* = {problem.J_optimal}")
    print("-" * 70)

    # Test multi-start SLSQP
    print("\n1. Multi-start SLSQP:")
    res_nlp = solve_multistart_slsqp(problem, n_starts=10, verbose=True)

    # Test BB-BO
    print("\n2. Black-box BO:")
    res_bo = solve_blackbox_bo(problem, n_iterations=20, n_initial=5, verbose=True, return_history=True)

    # Test Bi-level BO
    print("\n3. Bi-level BO:")
    res_bi = solve_bilevel_bo(problem, n_iterations=20, n_initial=5, verbose=True, return_history=True)

    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"{'Method':<20} {'Best J':<15} {'f^BB evals':<15}")
    print("-" * 50)
    print(f"{'NLP':<20} {res_nlp['best_J']:<15.6f} {res_nlp['n_fbb_evals']:<15}")
    print(f"{'Black-box BO':<20} {res_bo['best_J']:<15.6f} {res_bo['n_fbb_evals']:<15}")
    print(f"{'Bi-level BO':<20} {res_bi['best_J']:<15.6f} {res_bi['n_fbb_evals']:<15}")
    print(f"{'Optimal':<20} {problem.J_optimal:<15.6f}")
