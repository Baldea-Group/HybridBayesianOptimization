"""
Aggressive Basin-Hopping tuning for Rastrigin and Distillation full-space.

Strategies:
  1. Many restarts of BH (pick best of N independent BH runs)
  2. Very large niter (1000+)
  3. Stepsize = full range (maximum exploration)
  4. Multiple seeds, report success rate
"""

import numpy as np
import time
import signal as _sig
from scipy.optimize import minimize, basinhopping
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
    return res, sign


def multi_restart_bh_full(prob, n_restarts=10, T=1.0, stepsize=0.5, niter=200, base_seed=42):
    """Run BH n_restarts times, return best result."""
    sign = -1.0 if prob.maximize else 1.0
    best_res = None
    best_fun = np.inf

    for i in range(n_restarts):
        res, _ = run_bh_full(prob, T=T, stepsize=stepsize, niter=niter, seed=base_seed + i)
        if res is not None and res.fun < best_fun:
            best_fun = res.fun
            best_res = res

    return best_res, sign


if __name__ == '__main__':
    problems_dict = get_all_problems()
    rast = problems_dict['Rastrigin']
    dist = problems_dict['Distillation']

    # ─── Rastrigin: diagnose the failure ────────────────────────────
    print("=" * 80)
    print("RASTRIGIN FULL-SPACE: Diagnosing BH failure")
    print("=" * 80)
    print()

    # What does SLSQP find from random starts?
    sign = 1.0  # minimize
    x_lower, x_upper = rast.get_full_bounds()
    rng = np.random.default_rng(42)

    slsqp_vals = []
    for i in range(100):
        x0 = rng.uniform(x_lower, x_upper)
        def obj(x):
            x_wb, x_bb = rast.split_x(x)
            y = rast.fbb(x_bb)
            return rast.J(x_wb, y)
        r = minimize(obj, x0, method='SLSQP',
                     bounds=list(zip(x_lower, x_upper)),
                     options={'maxiter': 500, 'ftol': 1e-14})
        slsqp_vals.append(r.fun)

    slsqp_vals = np.array(slsqp_vals)
    print(f"SLSQP from 100 random starts:")
    print(f"  min={slsqp_vals.min():.6f}, median={np.median(slsqp_vals):.6f}, "
          f"max={slsqp_vals.max():.6f}")
    print(f"  fraction at global min (< 1e-6): {np.mean(slsqp_vals < 1e-6):.0%}")
    print(f"  fraction near global (< 0.01): {np.mean(slsqp_vals < 0.01):.0%}")
    print(f"  unique local minima: ~{len(np.unique(np.round(slsqp_vals, 2)))}")
    print()

    # ─── Strategy 1: More restarts ──────────────────────────────────
    print("Strategy 1: Multiple BH restarts (niter=200 each)")
    for n_restarts in [5, 10, 20, 50]:
        t0 = time.time()
        res, s = multi_restart_bh_full(rast, n_restarts=n_restarts,
                                        T=1.0, stepsize=0.5, niter=200)
        elapsed = time.time() - t0
        J_val = s * float(res.fun) if res else None
        reg = compute_regret(rast, J_val)
        print(f"  {n_restarts:>3} restarts: regret={reg:.2e}, J={J_val:.6f}, time={elapsed:.1f}s")

    print()

    # ─── Strategy 2: Very long single run ───────────────────────────
    print("Strategy 2: Long single BH run")
    for niter in [500, 1000, 2000]:
        res, elapsed = run_with_timeout(
            lambda ni=niter: run_bh_full(rast, T=1.0, stepsize=0.5, niter=ni),
            timeout=300)
        if res is not None:
            res, s = res
            J_val = s * float(res.fun)
            reg = compute_regret(rast, J_val)
            print(f"  niter={niter:>5}: regret={reg:.2e}, J={J_val:.6f}, time={elapsed:.1f}s")
        else:
            print(f"  niter={niter:>5}: timed out ({elapsed:.0f}s)")

    print()

    # ─── Strategy 3: Stepsize = half range ──────────────────────────
    print("Strategy 3: Large stepsize (half of variable range)")
    half_range = (x_upper[0] - x_lower[0]) / 2  # 5.12
    for niter in [200, 500]:
        results = []
        for seed in range(10):
            res, elapsed = run_with_timeout(
                lambda s=seed, ni=niter: run_bh_full(
                    rast, T=1.0, stepsize=half_range, niter=ni, seed=s),
                timeout=120)
            if res is not None:
                res, s = res
                J_val = s * float(res.fun)
                results.append(compute_regret(rast, J_val))
            else:
                results.append(np.inf)
        results = np.array(results)
        success = np.sum(results < 1e-6)
        print(f"  stepsize={half_range:.1f}, niter={niter}: "
              f"{success}/10 converged, median={np.median(results):.2e}, "
              f"max={np.max(results):.2e}")

    print()

    # ─── Strategy 4: Stepsize = full range (pure random restart + SLSQP) ─
    print("Strategy 4: Stepsize = full range (effectively random restart)")
    full_range = x_upper[0] - x_lower[0]  # 10.24
    for niter in [200, 500, 1000]:
        results = []
        for seed in range(10):
            res, elapsed = run_with_timeout(
                lambda s=seed, ni=niter: run_bh_full(
                    rast, T=100.0, stepsize=full_range, niter=ni, seed=s),
                timeout=120)
            if res is not None:
                res, s = res
                J_val = s * float(res.fun)
                results.append(compute_regret(rast, J_val))
            else:
                results.append(np.inf)
        results = np.array(results)
        success = np.sum(results < 1e-6)
        print(f"  niter={niter:>5}: {success}/10 converged, "
              f"median={np.median(results):.2e}, max={np.max(results):.2e}")

    print()

    # ─── Distillation full-space ────────────────────────────────────
    print("=" * 80)
    print("DISTILLATION FULL-SPACE: Diagnosing BH residual regret (0.18)")
    print("=" * 80)

    sign_d = -1.0 if dist.maximize else 1.0
    for niter in [200, 500, 1000]:
        results = []
        for seed in range(5):
            res, elapsed = run_with_timeout(
                lambda s=seed, ni=niter: run_bh_full(
                    dist, T=1.0, stepsize=0.5, niter=ni, seed=s),
                timeout=120)
            if res is not None:
                res, s = res
                J_val = s * float(res.fun)
                results.append(compute_regret(dist, J_val))
            else:
                results.append(np.inf)
        results = np.array(results)
        print(f"  niter={niter:>5}: median={np.median(results):.2e}, "
              f"min={np.min(results):.2e}, max={np.max(results):.2e}")

    print()
    print("Done.")
