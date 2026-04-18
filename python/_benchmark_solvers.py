"""
Benchmark each scipy global solver individually on:
  1. Inner problem (fixed y at optimal x_bb)
  2. Full-space problem (all variables)

Solvers tested:
  - differential_evolution (DE)
  - shgo (SHGO)
  - dual_annealing (DA)
  - basinhopping (BH)
  - Multi-start SLSQP (local baseline)

For each: report regret, wall time, feasibility, #evals.
"""

import numpy as np
import time
import signal as _sig
from scipy.optimize import (
    minimize, differential_evolution, shgo, dual_annealing,
    basinhopping, NonlinearConstraint, OptimizeResult,
)
from functions import get_all_problems, BiLevelProblem

TIMEOUT = 60  # seconds per solver per problem


class _Timeout(Exception):
    pass

def _handler(s, f):
    raise _Timeout()


def run_with_timeout(fn, timeout=TIMEOUT):
    """Run fn() with a SIGALRM timeout. Returns (result, elapsed) or (None, elapsed)."""
    old = _sig.signal(_sig.SIGALRM, _handler)
    _sig.alarm(timeout)
    t0 = time.time()
    try:
        result = fn()
        _sig.alarm(0)
        return result, time.time() - t0
    except _Timeout:
        _sig.alarm(0)
        return None, time.time() - t0
    except Exception as e:
        _sig.alarm(0)
        return None, time.time() - t0
    finally:
        _sig.signal(_sig.SIGALRM, old)


# ─── Inner problem solvers ───────────────────────────────────────────────────

def inner_slsqp(prob, y, n_starts=50, seed=42):
    """Multi-start SLSQP (local baseline)."""
    sign = -1.0 if prob.maximize else 1.0
    rng = np.random.default_rng(seed)
    bounds = list(zip(prob.x_wb_lower, prob.x_wb_upper))

    jac = None
    if prob.J_grad_x_wb is not None:
        jac = lambda x: sign * prob.J_grad_x_wb(x, y)

    constraints = []
    if prob.g is not None and prob.n_g > 0:
        con = {'type': 'ineq', 'fun': lambda x: -prob.g(x, y)}
        if prob.g_grad_x_wb is not None:
            con['jac'] = lambda x: -prob.g_grad_x_wb(x, y)
        constraints.append(con)

    obj = lambda x: sign * prob.J(x, y)
    best = None
    for i in range(n_starts):
        x0 = rng.uniform(prob.x_wb_lower, prob.x_wb_upper)
        try:
            r = minimize(obj, x0, method='SLSQP', jac=jac, bounds=bounds,
                         constraints=constraints,
                         options={'maxiter': 500, 'ftol': 1e-14})
            if np.isfinite(r.fun):
                feas = True
                if prob.g is not None and prob.n_g > 0:
                    feas = np.all(prob.g(r.x, y) <= 1e-6)
                if best is None or (feas and r.fun < best.fun):
                    best = r
        except:
            pass

    return best


def inner_de(prob, y, seed=42):
    """Differential evolution on inner problem."""
    sign = -1.0 if prob.maximize else 1.0
    bounds = list(zip(prob.x_wb_lower, prob.x_wb_upper))
    has_con = prob.g is not None and prob.n_g > 0

    constraints = ()
    if has_con:
        constraints = (NonlinearConstraint(
            lambda x: prob.g(x, y), -np.inf, 0.0),)

    res = differential_evolution(
        lambda x: sign * prob.J(x, y),
        bounds=bounds, seed=seed, polish=False,
        constraints=constraints, maxiter=1000, tol=1e-12,
        popsize=30,
    )
    return res


def inner_shgo(prob, y):
    """SHGO on inner problem."""
    sign = -1.0 if prob.maximize else 1.0
    bounds = list(zip(prob.x_wb_lower, prob.x_wb_upper))

    constraints = None
    if prob.g is not None and prob.n_g > 0:
        def _neg_g(x):
            return -prob.g(x, y)
        constraints = [
            {'type': 'ineq', 'fun': lambda x, i=i: _neg_g(x)[i]}
            for i in range(prob.n_g)
        ]

    res = shgo(
        lambda x: sign * prob.J(x, y),
        bounds, constraints=constraints,
        options={'maxiter': 200},
    )
    return res


def inner_da(prob, y, seed=42):
    """Dual annealing on inner problem."""
    sign = -1.0 if prob.maximize else 1.0
    bounds = list(zip(prob.x_wb_lower, prob.x_wb_upper))

    res = dual_annealing(
        lambda x: sign * prob.J(x, y),
        bounds=bounds, seed=seed, maxiter=1000,
    )
    return res


def inner_bh(prob, y, seed=42):
    """Basin-hopping on inner problem."""
    sign = -1.0 if prob.maximize else 1.0
    bounds = list(zip(prob.x_wb_lower, prob.x_wb_upper))
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(prob.x_wb_lower, prob.x_wb_upper)

    minimizer_kwargs = {
        'method': 'SLSQP',
        'bounds': bounds,
        'options': {'maxiter': 500, 'ftol': 1e-14},
    }
    if prob.g is not None and prob.n_g > 0:
        minimizer_kwargs['constraints'] = [{
            'type': 'ineq',
            'fun': lambda x: -prob.g(x, y),
        }]

    res = basinhopping(
        lambda x: sign * prob.J(x, y),
        x0, minimizer_kwargs=minimizer_kwargs,
        niter=200, seed=int(seed),
    )
    return res


# ─── Full-space solvers ──────────────────────────────────────────────────────

def full_slsqp(prob, n_starts=100, seed=42):
    """Multi-start SLSQP on full space."""
    sign = -1.0 if prob.maximize else 1.0
    x_lower, x_upper = prob.get_full_bounds()
    bounds = list(zip(x_lower, x_upper))
    rng = np.random.default_rng(seed)

    def obj(x):
        x_wb, x_bb = prob.split_x(x)
        y = prob.fbb(x_bb)
        return sign * prob.J(x_wb, y)

    constraints = []
    if prob.g is not None and prob.n_g > 0:
        def con(x):
            x_wb, x_bb = prob.split_x(x)
            y = prob.fbb(x_bb)
            return -prob.g(x_wb, y)
        constraints = [{'type': 'ineq', 'fun': con}]

    best = None
    for _ in range(n_starts):
        x0 = rng.uniform(x_lower, x_upper)
        try:
            r = minimize(obj, x0, method='SLSQP', bounds=bounds,
                         constraints=constraints,
                         options={'maxiter': 500, 'ftol': 1e-14})
            if np.isfinite(r.fun):
                feas = True
                if constraints:
                    feas = np.all(con(r.x) >= -1e-6)
                if best is None or (feas and r.fun < best.fun):
                    best = r
        except:
            pass
    return best


def full_de(prob, seed=42):
    """Differential evolution on full space."""
    sign = -1.0 if prob.maximize else 1.0
    x_lower, x_upper = prob.get_full_bounds()
    bounds = list(zip(x_lower, x_upper))

    def obj(x):
        x_wb, x_bb = prob.split_x(x)
        y = prob.fbb(x_bb)
        return sign * prob.J(x_wb, y)

    constraints = ()
    if prob.g is not None and prob.n_g > 0:
        def con(x):
            x_wb, x_bb = prob.split_x(x)
            y = prob.fbb(x_bb)
            return prob.g(x_wb, y)
        constraints = (NonlinearConstraint(con, -np.inf, 0.0),)

    res = differential_evolution(
        obj, bounds=bounds, seed=seed, polish=False,
        constraints=constraints, maxiter=1000, tol=1e-12,
        popsize=30,
    )
    return res


def full_shgo(prob):
    """SHGO on full space."""
    sign = -1.0 if prob.maximize else 1.0
    x_lower, x_upper = prob.get_full_bounds()
    bounds = list(zip(x_lower, x_upper))

    def obj(x):
        x_wb, x_bb = prob.split_x(x)
        y = prob.fbb(x_bb)
        return sign * prob.J(x_wb, y)

    constraints = None
    if prob.g is not None and prob.n_g > 0:
        def _neg_con(x):
            x_wb, x_bb = prob.split_x(x)
            y = prob.fbb(x_bb)
            return -prob.g(x_wb, y)
        constraints = [
            {'type': 'ineq', 'fun': lambda x, i=i: _neg_con(x)[i]}
            for i in range(prob.n_g)
        ]

    res = shgo(
        obj, bounds, constraints=constraints,
        options={'maxiter': 200},
    )
    return res


def full_da(prob, seed=42):
    """Dual annealing on full space."""
    sign = -1.0 if prob.maximize else 1.0
    x_lower, x_upper = prob.get_full_bounds()
    bounds = list(zip(x_lower, x_upper))

    def obj(x):
        x_wb, x_bb = prob.split_x(x)
        y = prob.fbb(x_bb)
        return sign * prob.J(x_wb, y)

    res = dual_annealing(
        obj, bounds=bounds, seed=seed, maxiter=1000,
    )
    return res


def full_bh(prob, seed=42):
    """Basin-hopping on full space."""
    sign = -1.0 if prob.maximize else 1.0
    x_lower, x_upper = prob.get_full_bounds()
    bounds = list(zip(x_lower, x_upper))
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(x_lower, x_upper)

    minimizer_kwargs = {
        'method': 'SLSQP',
        'bounds': bounds,
        'options': {'maxiter': 500, 'ftol': 1e-14},
    }
    if prob.g is not None and prob.n_g > 0:
        def con(x):
            x_wb, x_bb = prob.split_x(x)
            y = prob.fbb(x_bb)
            return -prob.g(x_wb, y)
        minimizer_kwargs['constraints'] = [{'type': 'ineq', 'fun': con}]

    res = basinhopping(
        lambda x: sign * prob.J(*((lambda xw, xb: (xw, prob.fbb(xb)))(*prob.split_x(x)))),
        x0, minimizer_kwargs=minimizer_kwargs,
        niter=200, seed=int(seed),
    )
    return res


# ─── Main benchmark ──────────────────────────────────────────────────────────

def compute_regret(prob, J_val):
    """Compute regret = |J - J*| / max(|J*|, 1e-10)."""
    if J_val is None or not np.isfinite(J_val):
        return np.inf
    return abs(J_val - prob.J_optimal) / max(abs(prob.J_optimal), 1e-10)


def check_feasibility(prob, x_wb, y):
    if prob.g is None or prob.n_g == 0:
        return True, 0.0
    g = prob.g(x_wb, y)
    return np.all(g <= 1e-6), float(np.max(g))


def extract_inner_result(prob, res, y, sign):
    """Extract J, x_wb, feasibility from an inner solver result."""
    if res is None or not hasattr(res, 'fun') or res.fun is None:
        return None, None, False, np.inf
    x_wb = np.asarray(res.x)
    J_val = sign * float(res.fun)  # convert back from sign-adjusted
    feas, max_g = check_feasibility(prob, x_wb, y)
    return J_val, x_wb, feas, max_g


def extract_full_result(prob, res, sign):
    """Extract J, feasibility from a full-space solver result."""
    if res is None or not hasattr(res, 'fun') or res.fun is None:
        return None, False, np.inf
    x_wb, x_bb = prob.split_x(np.asarray(res.x))
    y = prob.fbb(x_bb)
    J_val = sign * float(res.fun)
    feas, max_g = check_feasibility(prob, x_wb, y)
    return J_val, feas, max_g


if __name__ == '__main__':
    import pickle, json, os

    problems = list(get_all_problems().values())

    # Solver configurations with documented hyperparameters
    SOLVER_CONFIGS = {
        'inner': {
            'SLSQP-50': {
                'fn': lambda p, y: inner_slsqp(p, y, n_starts=50),
                'hyperparameters': {
                    'method': 'SLSQP',
                    'n_starts': 50,
                    'maxiter': 500,
                    'ftol': 1e-14,
                    'uses_gradients': True,
                },
            },
            'DE': {
                'fn': inner_de,
                'hyperparameters': {
                    'method': 'differential_evolution',
                    'seed': 42,
                    'polish': False,
                    'maxiter': 1000,
                    'tol': 1e-12,
                    'popsize': 30,
                    'constraint_handling': 'NonlinearConstraint',
                },
            },
            'SHGO': {
                'fn': inner_shgo,
                'hyperparameters': {
                    'method': 'shgo',
                    'maxiter': 200,
                    'constraint_handling': 'dict-style ineq',
                },
            },
            'DA': {
                'fn': inner_da,
                'hyperparameters': {
                    'method': 'dual_annealing',
                    'seed': 42,
                    'maxiter': 1000,
                    'constraint_handling': 'none',
                },
            },
            'BH': {
                'fn': inner_bh,
                'hyperparameters': {
                    'method': 'basinhopping',
                    'niter': 200,
                    'seed': 42,
                    'local_minimizer': 'SLSQP',
                    'local_maxiter': 500,
                    'local_ftol': 1e-14,
                    'constraint_handling': 'via local SLSQP',
                },
            },
        },
        'full_space': {
            'SLSQP-100': {
                'fn': lambda p: full_slsqp(p, n_starts=100),
                'hyperparameters': {
                    'method': 'SLSQP',
                    'n_starts': 100,
                    'maxiter': 500,
                    'ftol': 1e-14,
                    'uses_gradients': False,  # no analytical gradients in full space
                },
            },
            'DE': {
                'fn': full_de,
                'hyperparameters': {
                    'method': 'differential_evolution',
                    'seed': 42,
                    'polish': False,
                    'maxiter': 1000,
                    'tol': 1e-12,
                    'popsize': 30,
                    'constraint_handling': 'NonlinearConstraint',
                },
            },
            'SHGO': {
                'fn': full_shgo,
                'hyperparameters': {
                    'method': 'shgo',
                    'maxiter': 200,
                    'constraint_handling': 'dict-style ineq',
                },
            },
            'DA': {
                'fn': full_da,
                'hyperparameters': {
                    'method': 'dual_annealing',
                    'seed': 42,
                    'maxiter': 1000,
                    'constraint_handling': 'none',
                },
            },
            'BH': {
                'fn': full_bh,
                'hyperparameters': {
                    'method': 'basinhopping',
                    'niter': 200,
                    'seed': 42,
                    'local_minimizer': 'SLSQP',
                    'local_maxiter': 500,
                    'local_ftol': 1e-14,
                    'constraint_handling': 'via local SLSQP',
                },
            },
        },
    }

    # Results storage
    results_inner = {}  # {problem_name: {solver_name: {regret, J, time, feasible, max_g, x_wb}}}
    results_full = {}   # {problem_name: {solver_name: {regret, J, time, feasible, max_g, x}}}

    # ─── INNER PROBLEM BENCHMARK ─────────────────────────────────────────
    print("=" * 120)
    print("INNER PROBLEM BENCHMARK (y fixed at optimal x_bb)")
    print("=" * 120)

    solver_names = list(SOLVER_CONFIGS['inner'].keys())
    hdr = f"{'Problem':<25}"
    for s in solver_names:
        hdr += f" | {s:>10} {'Time':>6} {'Feas':>5}"
    print(hdr)
    print("-" * len(hdr))

    for prob in problems:
        sign = -1.0 if prob.maximize else 1.0
        y_opt = prob.fbb(prob.x_bb_optimal)
        results_inner[prob.name] = {}

        row = f"{prob.name:<25}"
        for sname, cfg in SOLVER_CONFIGS['inner'].items():
            sfn = cfg['fn']
            res, elapsed = run_with_timeout(lambda sfn=sfn: sfn(prob, y_opt), TIMEOUT)
            J_val, x_wb, feas, max_g = extract_inner_result(prob, res, y_opt, sign)
            regret = compute_regret(prob, J_val)

            results_inner[prob.name][sname] = {
                'regret': float(regret) if np.isfinite(regret) else None,
                'J': float(J_val) if J_val is not None and np.isfinite(J_val) else None,
                'J_optimal': float(prob.J_optimal),
                'time_s': round(elapsed, 3),
                'feasible': bool(feas),
                'max_g': float(max_g) if np.isfinite(max_g) else None,
                'x_wb': x_wb.tolist() if x_wb is not None else None,
                'timed_out': res is None,
            }

            feas_str = "Y" if feas else f"N({max_g:.0e})"
            row += f" | {regret:>10.2e} {elapsed:>5.1f}s {feas_str:>5}"

        print(row)

    print()

    # ─── FULL-SPACE BENCHMARK ────────────────────────────────────────────
    print("=" * 120)
    print("FULL-SPACE BENCHMARK (regret vs J*)")
    print("=" * 120)

    full_names = list(SOLVER_CONFIGS['full_space'].keys())
    hdr = f"{'Problem':<25} {'J*':>12}"
    for s in full_names:
        hdr += f" | {s:>10} {'Time':>6} {'Feas':>5}"
    print(hdr)
    print("-" * len(hdr))

    for prob in problems:
        sign = -1.0 if prob.maximize else 1.0
        results_full[prob.name] = {}
        row = f"{prob.name:<25} {prob.J_optimal:>12.4f}"

        for sname, cfg in SOLVER_CONFIGS['full_space'].items():
            sfn = cfg['fn']
            res, elapsed = run_with_timeout(lambda sfn=sfn: sfn(prob), TIMEOUT)
            J_val, feas, max_g = extract_full_result(prob, res, sign)
            regret = compute_regret(prob, J_val)

            x_list = None
            if res is not None and hasattr(res, 'x'):
                x_list = np.asarray(res.x).tolist()

            results_full[prob.name][sname] = {
                'regret': float(regret) if np.isfinite(regret) else None,
                'J': float(J_val) if J_val is not None and np.isfinite(J_val) else None,
                'J_optimal': float(prob.J_optimal),
                'time_s': round(elapsed, 3),
                'feasible': bool(feas),
                'max_g': float(max_g) if np.isfinite(max_g) else None,
                'x': x_list,
                'timed_out': res is None,
            }

            feas_str = "Y" if feas else f"N({max_g:.0e})"
            row += f" | {regret:>10.2e} {elapsed:>5.1f}s {feas_str:>5}"

        print(row)

    print()

    # ─── SAVE RESULTS ────────────────────────────────────────────────────
    results_dir = os.path.join(os.path.dirname(__file__), 'results_parallel')
    os.makedirs(results_dir, exist_ok=True)

    # Build hyperparameters dict (without fn keys)
    hyperparams = {}
    for context in ['inner', 'full_space']:
        hyperparams[context] = {}
        for sname, cfg in SOLVER_CONFIGS[context].items():
            hyperparams[context][sname] = cfg['hyperparameters']

    # Save pickle
    save_data = {
        'inner_results': results_inner,
        'full_space_results': results_full,
        'hyperparameters': hyperparams,
        'timeout_s': TIMEOUT,
        'problems': [p.name for p in problems],
    }
    pkl_path = os.path.join(results_dir, 'solver_benchmark.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"Saved pickle: {pkl_path}")

    # Save JSON (human-readable)
    json_path = os.path.join(results_dir, 'solver_benchmark.json')
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"Saved JSON:   {json_path}")

    print("Done.")
