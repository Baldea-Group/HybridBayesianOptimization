"""
Tune Basin-Hopping to converge on ALL 13 problems, including Rastrigin.

Key BH parameters to tune:
  - T (temperature): controls acceptance of uphill moves
  - stepsize: perturbation magnitude
  - niter: number of BH iterations
  - minimizer_kwargs: local solver config

Test each configuration on all problems for both inner and full-space.
"""

import numpy as np
import time
import signal as _sig
from scipy.optimize import (
    minimize, basinhopping, NonlinearConstraint, OptimizeResult,
)
from functions import get_all_problems, BiLevelProblem

TIMEOUT = 120


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
        print(f"    ERROR: {e}")
        return None, time.time() - t0
    finally:
        _sig.signal(_sig.SIGALRM, old)


class BoundedStep:
    """Custom step-taking that respects bounds."""
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


def run_bh_inner(prob, y, T=1.0, stepsize=0.5, niter=200, seed=42):
    sign = -1.0 if prob.maximize else 1.0
    bounds = list(zip(prob.x_wb_lower, prob.x_wb_upper))
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(prob.x_wb_lower, prob.x_wb_upper)

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
    return res


def run_bh_full(prob, T=1.0, stepsize=0.5, niter=200, seed=42):
    sign = -1.0 if prob.maximize else 1.0
    x_lower, x_upper = prob.get_full_bounds()
    bounds = list(zip(x_lower, x_upper))
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(x_lower, x_upper)

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
    return res


def check_feasibility(prob, x_wb, y):
    if prob.g is None or prob.n_g == 0:
        return True
    return np.all(prob.g(x_wb, y) <= 1e-6)


if __name__ == '__main__':
    problems = list(get_all_problems().values())

    # Configurations to test
    configs = {
        'default':      {'T': 1.0,  'stepsize': 0.5, 'niter': 200},
        'hot+wide':     {'T': 5.0,  'stepsize': 2.0, 'niter': 200},
        'hot+wider':    {'T': 10.0, 'stepsize': 3.0, 'niter': 200},
        'cold+narrow':  {'T': 0.5,  'stepsize': 0.3, 'niter': 200},
        'wide+long':    {'T': 1.0,  'stepsize': 2.0, 'niter': 500},
        'hot+wide+long':{'T': 5.0,  'stepsize': 2.0, 'niter': 500},
    }

    # ─── FOCUS: Rastrigin full-space first ──────────────────────────────
    print("=" * 100)
    print("RASTRIGIN FULL-SPACE: Tuning Basin-Hopping")
    print("=" * 100)

    rast = [p for p in problems if p.name == 'Rastrigin'][0]
    sign = -1.0 if rast.maximize else 1.0

    for cname, cfg in configs.items():
        regrets = []
        times = []
        for seed in range(10):
            res, elapsed = run_with_timeout(
                lambda: run_bh_full(rast, seed=seed, **cfg), TIMEOUT)
            times.append(elapsed)
            if res is not None:
                J_val = sign * float(res.fun)
                regrets.append(compute_regret(rast, J_val))
            else:
                regrets.append(np.inf)

        regrets = np.array(regrets)
        success = np.sum(regrets < 1e-6)
        print(f"  {cname:>20}: {success}/10 converged, "
              f"median_reg={np.median(regrets):.2e}, "
              f"max_reg={np.max(regrets):.2e}, "
              f"avg_t={np.mean(times):.1f}s")

    print()

    # ─── ALL PROBLEMS: Best config candidates ───────────────────────────
    # Test the two most promising configs on all problems
    test_configs = {
        'hot+wide':      {'T': 5.0,  'stepsize': 2.0, 'niter': 200},
        'hot+wide+long': {'T': 5.0,  'stepsize': 2.0, 'niter': 500},
    }

    for context, run_fn_factory in [
        ('INNER', lambda p, cfg: lambda: run_bh_inner(
            p, p.fbb(p.x_bb_optimal), **cfg)),
        ('FULL-SPACE', lambda p, cfg: lambda: run_bh_full(p, **cfg)),
    ]:
        print("=" * 100)
        print(f"{context} BENCHMARK: Basin-Hopping tuning")
        print("=" * 100)

        hdr = f"{'Problem':<25}"
        for cname in test_configs:
            hdr += f" | {cname:>18} {'Time':>6} {'Feas':>5}"
        print(hdr)
        print("-" * len(hdr))

        for prob in problems:
            sign_p = -1.0 if prob.maximize else 1.0
            row = f"{prob.name:<25}"

            for cname, cfg in test_configs.items():
                fn = run_fn_factory(prob, cfg)
                res, elapsed = run_with_timeout(fn, TIMEOUT)

                if res is not None:
                    J_val = sign_p * float(res.fun)
                    regret = compute_regret(prob, J_val)
                    if context == 'INNER':
                        y_opt = prob.fbb(prob.x_bb_optimal)
                        feas = check_feasibility(prob, np.asarray(res.x), y_opt)
                    else:
                        x_wb, x_bb = prob.split_x(np.asarray(res.x))
                        y = prob.fbb(x_bb)
                        feas = check_feasibility(prob, x_wb, y)
                else:
                    regret = np.inf
                    feas = False

                feas_str = "Y" if feas else "N"
                row += f" | {regret:>18.2e} {elapsed:>5.1f}s {feas_str:>5}"

            print(row)

        print()

    print("Done.")
