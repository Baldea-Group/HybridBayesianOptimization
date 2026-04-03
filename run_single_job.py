"""
run_single_job.py - Run one (problem, n_init, xi) combination for pylauncher.

Runs all repetitions of all 4 solvers (NLP, DE, BB-BO, Bi-level BO) for a
single problem/n_init/xi combo and saves the result as a pickle file.

Output format matches run_comparison.py so gather_results.py can merge them.
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
    solve_blackbox_nlp,
    solve_global_de,
    solve_blackbox_bo,
    solve_bilevel_bo,
    compute_regret,
)


def run_single(problem_name, n_init, xi, n_iter, n_reps, base_seed, acq,
               inner_solver, n_starts_nlp, n_inner_starts, kappa, outdir):
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
        'blackbox_nlp': {
            'final_regrets': [], 'best_Js': [], 'n_fbb_evals': [], 'wall_times': []
        },
        'global_de': {
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
        results[f'bilevel_bo_{a}'] = {
            'regrets': [], 'best_Js': [], 'n_fbb_evals': [], 'wall_times': [],
            'iter_times': [], 'X_bb_history': [], 'X_wb_history': [], 'Y_history': []
        }

    print(f"Problem={problem_name}  n_init={n_init}  xi={xi}  "
          f"n_iter={n_iter}  n_reps={n_reps}  acq={acq}", flush=True)

    for rep in range(n_reps):
        seed = base_seed + rep
        print(f"  rep {rep+1}/{n_reps} (seed={seed})", flush=True)

        # NLP
        t0 = time.perf_counter()
        res_nlp = solve_blackbox_nlp(problem, n_starts=n_starts_nlp,
                                     seed=seed, verbose=False)
        t_nlp = time.perf_counter() - t0
        regret_nlp = (problem.J_optimal - res_nlp['best_J']
                      if problem.maximize
                      else res_nlp['best_J'] - problem.J_optimal)
        results['blackbox_nlp']['final_regrets'].append(regret_nlp)
        results['blackbox_nlp']['best_Js'].append(res_nlp['best_J'])
        results['blackbox_nlp']['n_fbb_evals'].append(res_nlp['n_fbb_evals'])
        results['blackbox_nlp']['wall_times'].append(t_nlp)

        # DE
        t0 = time.perf_counter()
        try:
            res_de = solve_global_de(problem, seed=seed, verbose=False,
                                     return_history=True)
            t_de = time.perf_counter() - t0
            G_de = res_de.get('G_history', None)
            regret_de = compute_regret(res_de['Y_history'], problem.J_optimal,
                                       G_de, problem.maximize)
            bo_budget = n_iter + n_init
            fair_regret = (regret_de[bo_budget - 1] if len(regret_de) >= bo_budget
                           else regret_de[-1])
            results['global_de']['regrets'].append(regret_de)
            results['global_de']['final_regrets'].append(fair_regret)
            results['global_de']['best_Js'].append(res_de['best_J'])
            results['global_de']['n_fbb_evals'].append(res_de['n_fbb_evals'])
            results['global_de']['wall_times'].append(t_de)
            results['global_de']['X_history'].append(
                res_de.get('X_history', np.array([])))
            results['global_de']['Y_history'].append(
                res_de.get('Y_history', np.array([])))
        except Exception as e:
            t_de = time.perf_counter() - t0
            print(f"    DE error: {e}", flush=True)
            results['global_de']['regrets'].append(np.array([np.inf]))
            results['global_de']['final_regrets'].append(np.inf)
            results['global_de']['best_Js'].append(
                np.inf if not problem.maximize else -np.inf)
            results['global_de']['n_fbb_evals'].append(0)
            results['global_de']['wall_times'].append(t_de)
            results['global_de']['X_history'].append(np.array([]))
            results['global_de']['Y_history'].append(np.array([]))

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

            # Bi-level BO
            t0 = time.perf_counter()
            try:
                res_bi = solve_bilevel_bo(
                    problem, n_iterations=n_iter, n_initial=n_init,
                    n_inner_starts=n_inner_starts, acquisition=a, seed=seed,
                    verbose=False, return_history=True, xi=xi, kappa=kappa,
                    inner_solver=inner_solver)
                t_bi = time.perf_counter() - t0
                G_bi = res_bi.get('G_history', None)
                regret_bi = compute_regret(res_bi['Y_history'],
                                           problem.J_optimal, G_bi,
                                           problem.maximize)
                results[f'bilevel_bo_{a}']['regrets'].append(regret_bi)
                results[f'bilevel_bo_{a}']['best_Js'].append(res_bi['best_J'])
                results[f'bilevel_bo_{a}']['n_fbb_evals'].append(
                    res_bi['n_fbb_evals'])
                results[f'bilevel_bo_{a}']['wall_times'].append(t_bi)
                results[f'bilevel_bo_{a}']['iter_times'].append(
                    res_bi.get('iter_times', np.array([])))
                results[f'bilevel_bo_{a}']['X_bb_history'].append(
                    res_bi.get('X_bb_history', np.array([])))
                results[f'bilevel_bo_{a}']['X_wb_history'].append(
                    res_bi.get('X_wb_history', []))
                results[f'bilevel_bo_{a}']['Y_history'].append(
                    res_bi.get('Y_history', np.array([])))
            except Exception as e:
                t_bi = time.perf_counter() - t0
                print(f"    Bi-BO {a} error: {e}", flush=True)
                results[f'bilevel_bo_{a}']['regrets'].append(
                    np.full(n_iter + n_init, np.inf))
                results[f'bilevel_bo_{a}']['best_Js'].append(
                    np.inf if not problem.maximize else -np.inf)
                results[f'bilevel_bo_{a}']['n_fbb_evals'].append(0)
                results[f'bilevel_bo_{a}']['wall_times'].append(t_bi)
                results[f'bilevel_bo_{a}']['iter_times'].append(np.array([]))
                results[f'bilevel_bo_{a}']['X_bb_history'].append(np.array([]))
                results[f'bilevel_bo_{a}']['X_wb_history'].append([])
                results[f'bilevel_bo_{a}']['Y_history'].append(np.array([]))

    # Save
    settings = {
        'n_iterations': n_iter,
        'n_initial': n_init,
        'n_starts_nlp': n_starts_nlp,
        'n_inner_starts': n_inner_starts,
        'n_repetitions': n_reps,
        'base_seed': base_seed,
        'problem_names': [problem_name],
        'acquisitions': acquisitions,
        'xi': xi,
        'kappa': kappa,
        'inner_solver': inner_solver,
    }

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    fname = out_path / f"{problem_name}_ninit{n_init}_xi{xi}.pkl"
    with open(fname, 'wb') as f:
        pickle.dump({'settings': settings, 'results': {problem_name: results}}, f)
    print(f"  Saved -> {fname}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run single job for pylauncher')
    parser.add_argument('--problem', type=str, required=True)
    parser.add_argument('--n_init', type=int, required=True)
    parser.add_argument('--xi', type=float, required=True)
    parser.add_argument('--n_iter', type=int, default=200)
    parser.add_argument('--n_reps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--acq', type=str, default='ei')
    parser.add_argument('--inner_solver', type=str, default='global',
                        choices=['multistart', 'global'])
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
        n_reps=args.n_reps,
        base_seed=args.seed,
        acq=args.acq,
        inner_solver=args.inner_solver,
        n_starts_nlp=args.n_starts_nlp,
        n_inner_starts=args.n_inner_starts,
        kappa=args.kappa,
        outdir=args.outdir,
    )
