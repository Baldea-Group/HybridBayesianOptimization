"""
Final Basin-Hopping verification: unified config on all 13 problems.

Config: stepsize = half range, T=100, niter=1000, bounded steps, SLSQP local.
Test on both inner (at optimal y) and full-space.
"""

import numpy as np
import time
import signal as _sig
import pickle, json, os
from scipy.optimize import minimize, basinhopping
from functions import get_all_problems, BiLevelProblem

TIMEOUT = 300  # generous for Williams-Otto


class _Timeout(Exception):
    pass

def _handler(s, f):
    raise _Timeout()

def run_with_timeout(fn, timeout=TIMEOUT):
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


class BoundedStep:
    def __init__(self, stepsize, lower, upper, rng):
        self.stepsize = stepsize
        self.lower = np.asarray(lower)
        self.upper = np.asarray(upper)
        self.rng = rng

    def __call__(self, x):
        x_new = x + self.rng.uniform(-self.stepsize, self.stepsize, size=x.shape)
        return np.clip(x_new, self.lower, self.upper)


def compute_regret(prob, J_val):
    if J_val is None or not np.isfinite(J_val):
        return np.inf
    return abs(J_val - prob.J_optimal) / max(abs(prob.J_optimal), 1e-10)


def check_feasibility(prob, x_wb, y):
    if prob.g is None or prob.n_g == 0:
        return True, 0.0
    g = prob.g(x_wb, y)
    return bool(np.all(g <= 1e-6)), float(np.max(g))


def run_bh_inner(prob, y, niter=1000, T=100.0, seed=42):
    sign = -1.0 if prob.maximize else 1.0
    bounds = list(zip(prob.x_wb_lower, prob.x_wb_upper))
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(prob.x_wb_lower, prob.x_wb_upper)

    # Stepsize = half of variable range per dimension
    ranges = prob.x_wb_upper - prob.x_wb_lower
    stepsize = np.max(ranges) / 2.0

    jac = None
    if prob.J_grad_x_wb is not None:
        jac = lambda x: sign * prob.J_grad_x_wb(x, y)

    minimizer_kwargs = {
        'method': 'SLSQP',
        'jac': jac,
        'bounds': bounds,
        'options': {'maxiter': 500, 'ftol': 1e-14},
    }
    if prob.g is not None and prob.n_g > 0:
        con = {'type': 'ineq', 'fun': lambda x: -prob.g(x, y)}
        if prob.g_grad_x_wb is not None:
            con['jac'] = lambda x: -prob.g_grad_x_wb(x, y)
        minimizer_kwargs['constraints'] = [con]

    stepper = BoundedStep(stepsize, prob.x_wb_lower, prob.x_wb_upper, rng)

    res = basinhopping(
        lambda x: sign * prob.J(x, y),
        x0, minimizer_kwargs=minimizer_kwargs,
        niter=niter, T=T, seed=int(seed),
        take_step=stepper,
    )
    return res, sign


def run_bh_full(prob, niter=1000, T=100.0, seed=42):
    sign = -1.0 if prob.maximize else 1.0
    x_lower, x_upper = prob.get_full_bounds()
    bounds = list(zip(x_lower, x_upper))
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(x_lower, x_upper)

    # Stepsize = half of max variable range
    ranges = x_upper - x_lower
    stepsize = np.max(ranges) / 2.0

    def obj(x):
        x_wb, x_bb = prob.split_x(x)
        y = prob.fbb(x_bb)
        return sign * prob.J(x_wb, y)

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

    stepper = BoundedStep(stepsize, x_lower, x_upper, rng)

    res = basinhopping(
        obj, x0, minimizer_kwargs=minimizer_kwargs,
        niter=niter, T=T, seed=int(seed),
        take_step=stepper,
    )
    return res, sign


if __name__ == '__main__':
    problems = list(get_all_problems().values())

    BH_CONFIG = {'niter': 1000, 'T': 100.0}

    results_inner = {}
    results_full = {}

    # ─── INNER PROBLEM ───────────────────────────────────────────────
    print("=" * 90)
    print(f"INNER PROBLEM: Basin-Hopping (niter={BH_CONFIG['niter']}, T={BH_CONFIG['T']}, "
          f"stepsize=range/2)")
    print("=" * 90)
    print(f"{'Problem':<25} | {'Regret':>12} {'J':>16} {'Time':>7} {'Feas':>5} {'max_g':>10}")
    print("-" * 90)

    for prob in problems:
        y_opt = prob.fbb(prob.x_bb_optimal)
        res_tuple, elapsed = run_with_timeout(
            lambda p=prob, y=y_opt: run_bh_inner(p, y, **BH_CONFIG), TIMEOUT)

        if res_tuple is not None:
            res, sign = res_tuple
            J_val = sign * float(res.fun)
            x_wb = np.asarray(res.x)
            feas, max_g = check_feasibility(prob, x_wb, y_opt)
            regret = compute_regret(prob, J_val)

            results_inner[prob.name] = {
                'regret': float(regret) if np.isfinite(regret) else None,
                'J': float(J_val),
                'J_optimal': float(prob.J_optimal),
                'time_s': round(elapsed, 3),
                'feasible': feas,
                'max_g': float(max_g),
                'x_wb': x_wb.tolist(),
            }

            feas_str = "Y" if feas else f"N"
            print(f"{prob.name:<25} | {regret:>12.2e} {J_val:>16.8f} {elapsed:>6.1f}s "
                  f"{feas_str:>5} {max_g:>10.2e}")
        else:
            results_inner[prob.name] = {
                'regret': None, 'J': None, 'time_s': round(elapsed, 3),
                'feasible': False, 'timed_out': True,
            }
            print(f"{prob.name:<25} | {'TIMEOUT':>12} {'':>16} {elapsed:>6.1f}s")

    print()

    # ─── FULL-SPACE ──────────────────────────────────────────────────
    print("=" * 90)
    print(f"FULL-SPACE: Basin-Hopping (niter={BH_CONFIG['niter']}, T={BH_CONFIG['T']}, "
          f"stepsize=range/2)")
    print("=" * 90)
    print(f"{'Problem':<25} | {'Regret':>12} {'J':>16} {'Time':>7} {'Feas':>5} {'max_g':>10}")
    print("-" * 90)

    for prob in problems:
        res_tuple, elapsed = run_with_timeout(
            lambda p=prob: run_bh_full(p, **BH_CONFIG), TIMEOUT)

        if res_tuple is not None:
            res, sign = res_tuple
            J_val = sign * float(res.fun)
            x_full = np.asarray(res.x)
            x_wb, x_bb = prob.split_x(x_full)
            y = prob.fbb(x_bb)
            feas, max_g = check_feasibility(prob, x_wb, y)
            regret = compute_regret(prob, J_val)

            results_full[prob.name] = {
                'regret': float(regret) if np.isfinite(regret) else None,
                'J': float(J_val),
                'J_optimal': float(prob.J_optimal),
                'time_s': round(elapsed, 3),
                'feasible': feas,
                'max_g': float(max_g),
                'x': x_full.tolist(),
                'x_wb': x_wb.tolist(),
                'x_bb': x_bb.tolist(),
            }

            feas_str = "Y" if feas else f"N"
            print(f"{prob.name:<25} | {regret:>12.2e} {J_val:>16.8f} {elapsed:>6.1f}s "
                  f"{feas_str:>5} {max_g:>10.2e}")
        else:
            results_full[prob.name] = {
                'regret': None, 'J': None, 'time_s': round(elapsed, 3),
                'feasible': False, 'timed_out': True,
            }
            print(f"{prob.name:<25} | {'TIMEOUT':>12} {'':>16} {elapsed:>6.1f}s")

    print()

    # ─── SAVE ────────────────────────────────────────────────────────
    results_dir = os.path.join(os.path.dirname(__file__), 'results_parallel')
    os.makedirs(results_dir, exist_ok=True)

    save_data = {
        'solver': 'basinhopping',
        'config': {
            'niter': BH_CONFIG['niter'],
            'T': BH_CONFIG['T'],
            'stepsize': 'max_range / 2 (per-problem adaptive)',
            'local_minimizer': 'SLSQP',
            'local_maxiter': 500,
            'local_ftol': 1e-14,
            'take_step': 'BoundedStep (uniform perturbation clipped to bounds)',
            'uses_gradients': 'Yes (J_grad_x_wb, g_grad_x_wb) for inner; No for full-space',
            'constraint_handling': 'Via SLSQP local minimizer (dict-style ineq constraints)',
        },
        'inner_results': results_inner,
        'full_space_results': results_full,
        'timeout_s': TIMEOUT,
    }

    pkl_path = os.path.join(results_dir, 'bh_benchmark.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"Saved pickle: {pkl_path}")

    json_path = os.path.join(results_dir, 'bh_benchmark.json')
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"Saved JSON:   {json_path}")

    print("Done.")
