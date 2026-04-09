"""
run_single_job.py - Run one (problem, n_init, xi, rep) combination for pylauncher.

Runs a single repetition of all 4 solvers (SLSQP, Basin-Hopping, BB-BO,
Bi-level BO) and saves the result as a pickle file.
"""

import argparse
import pickle
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings('ignore')

from functions import get_all_problems
from solvers import (
    solve_multistart_slsqp,
    solve_basin_hopping,
    solve_blackbox_bo,
    solve_bilevel_bo,
    compute_regret,
)


def run_single(problem_name, n_init, xi, n_iter, rep, base_seed, acq,
               n_starts_nlp, n_inner_starts, kappa, outdir):
    inner_solvers = ['global', 'multistart']

    out_path = Path(outdir)
    fname = out_path / f"{problem_name}_ninit{n_init}_xi{xi}_it{rep}.pkl"

    if fname.exists():
        try:
            with open(fname, 'rb') as f:
                saved = pickle.load(f)
            s = saved.get('settings', {})
            if (s.get('n_iterations') == n_iter and
                    s.get('n_initial') == n_init and
                    s.get('xi') == xi and
                    s.get('rep_index') == rep and
                    s.get('base_seed') == base_seed and
                    s.get('acquisitions') == [acq] and
                    s.get('inner_solvers') == inner_solvers):
                print(f"Skipping — complete results found in {fname}", flush=True)
                return
        except Exception:
            pass  # corrupted file, re-run

    problems = get_all_problems()
    if problem_name not in problems:
        raise ValueError(f"Unknown problem: {problem_name}. "
                         f"Available: {list(problems.keys())}")
    problem = problems[problem_name]

    acquisitions = [acq]

    results = {
        'problem': problem_name,
        'J_optimal': problem.J_optimal,
        'maximize': problem.maximize,
        'acquisitions': acquisitions,
        'multistart_slsqp': {
            'final_regrets': [], 'best_Js': [], 'n_fbb_evals': [], 'wall_times': []
        },
        'basin_hopping': {
            'regrets': [], 'final_regrets': [], 'best_Js': [],
            'n_fbb_evals': [], 'wall_times': [],
            'X_history': [], 'Y_history': []
        },
    }
    for a in acquisitions:
        results[f'blackbox_bo_{a}'] = {
            'regrets': [], 'best_Js': [], 'n_fbb_evals': [], 'wall_times': [],
            'iter_times': [], 'X_history': [], 'Y_history': []
        }
        for isolver in inner_solvers:
            results[f'bilevel_bo_{a}_{isolver}'] = {
                'regrets': [], 'best_Js': [], 'n_fbb_evals': [], 'wall_times': [],
                'iter_times': [], 'X_bb_history': [], 'X_wb_history': [], 'Y_history': []
            }

    seed = base_seed + rep
    print(f"Problem={problem_name}  n_init={n_init}  xi={xi}  "
          f"n_iter={n_iter}  rep={rep}  seed={seed}  acq={acq}", flush=True)

    # Multi-start SLSQP
    t0 = time.perf_counter()
    res_nlp = solve_multistart_slsqp(problem, n_starts=n_starts_nlp,
                                     seed=seed, verbose=False)
    t_nlp = time.perf_counter() - t0
    regret_nlp = (problem.J_optimal - res_nlp['best_J']
                  if problem.maximize
                  else res_nlp['best_J'] - problem.J_optimal)
    results['multistart_slsqp']['final_regrets'].append(regret_nlp)
    results['multistart_slsqp']['best_Js'].append(res_nlp['best_J'])
    results['multistart_slsqp']['n_fbb_evals'].append(res_nlp['n_fbb_evals'])
    results['multistart_slsqp']['wall_times'].append(t_nlp)

    # Basin-Hopping
    t0 = time.perf_counter()
    try:
        res_bh = solve_basin_hopping(problem, seed=seed, verbose=False,
                                     return_history=True)
        t_bh = time.perf_counter() - t0
        G_bh = res_bh.get('G_history', None)
        regret_bh = compute_regret(res_bh['Y_history'], problem.J_optimal,
                                   G_bh, problem.maximize)
        bo_budget = n_iter + n_init
        fair_regret = (regret_bh[bo_budget - 1] if len(regret_bh) >= bo_budget
                       else regret_bh[-1])
        results['basin_hopping']['regrets'].append(regret_bh)
        results['basin_hopping']['final_regrets'].append(fair_regret)
        results['basin_hopping']['best_Js'].append(res_bh['best_J'])
        results['basin_hopping']['n_fbb_evals'].append(res_bh['n_fbb_evals'])
        results['basin_hopping']['wall_times'].append(t_bh)
        results['basin_hopping']['X_history'].append(
            res_bh.get('X_history', np.array([])))
        results['basin_hopping']['Y_history'].append(
            res_bh.get('Y_history', np.array([])))
    except Exception as e:
        t_bh = time.perf_counter() - t0
        print(f"    Basin-Hopping error: {e}", flush=True)
        results['basin_hopping']['regrets'].append(np.array([np.inf]))
        results['basin_hopping']['final_regrets'].append(np.inf)
        results['basin_hopping']['best_Js'].append(
            np.inf if not problem.maximize else -np.inf)
        results['basin_hopping']['n_fbb_evals'].append(0)
        results['basin_hopping']['wall_times'].append(t_bh)
        results['basin_hopping']['X_history'].append(np.array([]))
        results['basin_hopping']['Y_history'].append(np.array([]))

    # BO methods
    for a in acquisitions:
        # Black-box BO
        t0 = time.perf_counter()
        try:
            res_bo = solve_blackbox_bo(
                problem, n_iterations=n_iter, n_initial=n_init,
                acquisition=a, seed=seed, verbose=False,
                return_history=True, xi=xi, kappa=kappa)
            t_bo = time.perf_counter() - t0
            G_bo = res_bo.get('G_history', None)
            regret_bo = compute_regret(res_bo['Y_history'],
                                       problem.J_optimal, G_bo,
                                       problem.maximize)
            results[f'blackbox_bo_{a}']['regrets'].append(regret_bo)
            results[f'blackbox_bo_{a}']['best_Js'].append(res_bo['best_J'])
            results[f'blackbox_bo_{a}']['n_fbb_evals'].append(
                res_bo['n_fbb_evals'])
            results[f'blackbox_bo_{a}']['wall_times'].append(t_bo)
            results[f'blackbox_bo_{a}']['iter_times'].append(
                res_bo.get('iter_times', np.array([])))
            results[f'blackbox_bo_{a}']['X_history'].append(
                res_bo.get('X_history', np.array([])))
            results[f'blackbox_bo_{a}']['Y_history'].append(
                res_bo.get('Y_history', np.array([])))
        except Exception as e:
            t_bo = time.perf_counter() - t0
            print(f"    BB-BO {a} error: {e}", flush=True)
            results[f'blackbox_bo_{a}']['regrets'].append(
                np.full(n_iter + n_init, np.inf))
            results[f'blackbox_bo_{a}']['best_Js'].append(
                np.inf if not problem.maximize else -np.inf)
            results[f'blackbox_bo_{a}']['n_fbb_evals'].append(0)
            results[f'blackbox_bo_{a}']['wall_times'].append(t_bo)
            results[f'blackbox_bo_{a}']['iter_times'].append(np.array([]))
            results[f'blackbox_bo_{a}']['X_history'].append(np.array([]))
            results[f'blackbox_bo_{a}']['Y_history'].append(np.array([]))

        # Bi-level BO (both inner solvers)
        for isolver in inner_solvers:
            key = f'bilevel_bo_{a}_{isolver}'
            t0 = time.perf_counter()
            try:
                res_bi = solve_bilevel_bo(
                    problem, n_iterations=n_iter, n_initial=n_init,
                    n_inner_starts=n_inner_starts, acquisition=a, seed=seed,
                    verbose=False, return_history=True, xi=xi, kappa=kappa,
                    inner_solver=isolver)
                t_bi = time.perf_counter() - t0
                G_bi = res_bi.get('G_history', None)
                regret_bi = compute_regret(res_bi['Y_history'],
                                           problem.J_optimal, G_bi,
                                           problem.maximize)
                results[key]['regrets'].append(regret_bi)
                results[key]['best_Js'].append(res_bi['best_J'])
                results[key]['n_fbb_evals'].append(res_bi['n_fbb_evals'])
                results[key]['wall_times'].append(t_bi)
                results[key]['iter_times'].append(
                    res_bi.get('iter_times', np.array([])))
                results[key]['X_bb_history'].append(
                    res_bi.get('X_bb_history', np.array([])))
                results[key]['X_wb_history'].append(
                    res_bi.get('X_wb_history', []))
                results[key]['Y_history'].append(
                    res_bi.get('Y_history', np.array([])))
            except Exception as e:
                t_bi = time.perf_counter() - t0
                print(f"    Bi-BO {a} ({isolver}) error: {e}", flush=True)
                results[key]['regrets'].append(
                    np.full(n_iter + n_init, np.inf))
                results[key]['best_Js'].append(
                    np.inf if not problem.maximize else -np.inf)
                results[key]['n_fbb_evals'].append(0)
                results[key]['wall_times'].append(t_bi)
                results[key]['iter_times'].append(np.array([]))
                results[key]['X_bb_history'].append(np.array([]))
                results[key]['X_wb_history'].append([])
                results[key]['Y_history'].append(np.array([]))

    # Save
    settings = {
        'n_iterations': n_iter,
        'n_initial': n_init,
        'n_starts_nlp': n_starts_nlp,
        'n_inner_starts': n_inner_starts,
        'n_repetitions': 1,
        'rep_index': rep,
        'base_seed': base_seed,
        'problem_names': [problem_name],
        'acquisitions': acquisitions,
        'xi': xi,
        'kappa': kappa,
        'inner_solvers': inner_solvers,
    }

    out_path.mkdir(parents=True, exist_ok=True)
    with open(fname, 'wb') as f:
        pickle.dump({'settings': settings, 'results': {problem_name: results}}, f)
    print(f"  Saved -> {fname}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run single job for pylauncher')
    parser.add_argument('--problem', type=str, required=True)
    parser.add_argument('--n_init', type=int, required=True)
    parser.add_argument('--xi', type=float, required=True)
    parser.add_argument('--rep', type=int, required=True,
                        help='Repetition index (0-based)')
    parser.add_argument('--n_iter', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--acq', type=str, default='ei')
    parser.add_argument('--n_starts_nlp', type=int, default=20)
    parser.add_argument('--n_inner_starts', type=int, default=20)
    parser.add_argument('--kappa', type=float, default=2.0)
    parser.add_argument('--outdir', type=str, default='results_parallel')
    args = parser.parse_args()

    run_single(
        problem_name=args.problem,
        n_init=args.n_init,
        xi=args.xi,
        n_iter=args.n_iter,
        rep=args.rep,
        base_seed=args.seed,
        acq=args.acq,
        n_starts_nlp=args.n_starts_nlp,
        n_inner_starts=args.n_inner_starts,
        kappa=args.kappa,
        outdir=args.outdir,
    )
