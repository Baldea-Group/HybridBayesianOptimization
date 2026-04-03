"""
run_comparison.py - Run optimization experiments comparing methods

This script compares optimization approaches on bi-level test problems:
1. Black-box NLP (multi-start gradient optimization)
2. Black-box BO (Bayesian Optimization over all variables)
3. Bi-level BO (BO over material vars, NLP over process vars)

With multiple acquisition functions:
- EI: Expected Improvement
- PI: Probability of Improvement
- LCB: Lower Confidence Bound
- mWB2: Modified Watson-Barnes 2 (from COBALT paper)
- Thompson: Thompson Sampling

Features:
- Multiple acquisition functions comparison
- Multiple repetitions with different seeds
- Results caching (only re-runs if settings change)
- Wall time tracking
- Comprehensive output plots
"""

import numpy as np
import time
import pickle
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional

from functions import BiLevelProblem, get_all_problems, get_cobalt_problems
from solvers import (
    solve_blackbox_nlp,
    solve_global_de,
    solve_blackbox_bo,
    solve_bilevel_bo,
    compute_regret,
    get_available_acquisitions,
    AVAILABLE_ACQUISITIONS
)
from plotting import print_summary, plot_results

warnings.filterwarnings('ignore')

# Default acquisition functions to compare
DEFAULT_ACQUISITIONS = ['ei', 'pi', 'lcb', 'mwb2', 'thompson']


def run_experiments(
    problems: Dict[str, BiLevelProblem],
    settings: Dict[str, Any],
    acquisitions: List[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run optimization experiments on all problems with multiple acquisition functions.

    Args:
        problems: Dictionary of problem name -> BiLevelProblem
        settings: Experiment settings dictionary
        acquisitions: List of acquisition function names to compare
        verbose: Whether to print progress

    Returns:
        Dictionary with all results, organized by problem and acquisition function
    """
    if acquisitions is None:
        acquisitions = DEFAULT_ACQUISITIONS

    n_iterations = settings['n_iterations']
    n_initial = settings['n_initial']
    n_starts_nlp = settings['n_starts_nlp']
    n_inner_starts = settings['n_inner_starts']
    n_repetitions = settings['n_repetitions']
    base_seed = settings['base_seed']
    xi = settings.get('xi', 0.01)
    kappa = settings.get('kappa', 2.0)
    inner_solver = settings.get('inner_solver', 'global')

    all_results = {}

    for name, problem in problems.items():
        if verbose:
            print("\n" + "=" * 70)
            print(f"Problem: {name}")
            print(f"  Dimensions: n_x_wb={problem.n_x_wb}, n_x_bb={problem.n_x_bb}")
            print(f"  Constraints: n_g={problem.n_g}")
            print(f"  Maximize: {problem.maximize}")
            print(f"  Optimal: J* = {problem.J_optimal}")
            print(f"  Acquisition functions: {acquisitions}")
            print(f"  Running {n_repetitions} repetitions...")
            print("=" * 70)

        results = {
            'problem': name,
            'J_optimal': problem.J_optimal,
            'maximize': problem.maximize,
            'acquisitions': acquisitions,
            'blackbox_nlp': {'final_regrets': [], 'best_Js': [], 'n_fbb_evals': [], 'wall_times': []},
            'global_de': {
                'regrets': [], 'final_regrets': [], 'best_Js': [],
                'n_fbb_evals': [], 'wall_times': [],
                'X_history': [], 'Y_history': []
            },
        }

        # Initialize storage for each acquisition function
        for acq in acquisitions:
            results[f'blackbox_bo_{acq}'] = {
                'regrets': [], 'best_Js': [], 'n_fbb_evals': [], 'wall_times': [], 'iter_times': [],
                'X_history': [], 'Y_history': []
            }
            results[f'bilevel_bo_{acq}'] = {
                'regrets': [], 'best_Js': [], 'n_fbb_evals': [], 'wall_times': [], 'iter_times': [],
                'X_bb_history': [], 'X_wb_history': [], 'Y_history': []
            }

        for rep in range(n_repetitions):
            seed = base_seed + rep
            if verbose:
                print(f"\n  --- Repetition {rep + 1}/{n_repetitions} (seed={seed}) ---")

            # 1. Black-box NLP (only once per repetition, independent of acquisition)
            if verbose:
                print("    [NLP] Black-box NLP...")
            t0 = time.perf_counter()
            res_nlp = solve_blackbox_nlp(
                problem, n_starts=n_starts_nlp, seed=seed, verbose=False
            )
            t_nlp = time.perf_counter() - t0

            if problem.maximize:
                regret_nlp = problem.J_optimal - res_nlp['best_J']
            else:
                regret_nlp = res_nlp['best_J'] - problem.J_optimal

            results['blackbox_nlp']['final_regrets'].append(regret_nlp)
            results['blackbox_nlp']['best_Js'].append(res_nlp['best_J'])
            results['blackbox_nlp']['n_fbb_evals'].append(res_nlp['n_fbb_evals'])
            results['blackbox_nlp']['wall_times'].append(t_nlp)
            if verbose:
                print(f"           best_J = {res_nlp['best_J']:.6f}, regret = {regret_nlp:.6f}, time = {t_nlp:.2f}s")

            # 2. Global DE (once per repetition, independent of acquisition)
            if verbose:
                print("    [DE] Global Differential Evolution...")
            t0 = time.perf_counter()
            try:
                res_de = solve_global_de(
                    problem, seed=seed, verbose=False, return_history=True
                )
                t_de = time.perf_counter() - t0

                G_de = res_de.get('G_history', None)
                regret_de = compute_regret(
                    res_de['Y_history'], problem.J_optimal, G_de, problem.maximize
                )

                # Extract regret at BO-equivalent budget for fair comparison
                bo_budget = n_iterations + n_initial
                if len(regret_de) >= bo_budget:
                    fair_regret = regret_de[bo_budget - 1]
                else:
                    fair_regret = regret_de[-1]

                results['global_de']['regrets'].append(regret_de)
                results['global_de']['final_regrets'].append(fair_regret)
                results['global_de']['best_Js'].append(res_de['best_J'])
                results['global_de']['n_fbb_evals'].append(res_de['n_fbb_evals'])
                results['global_de']['wall_times'].append(t_de)
                results['global_de']['X_history'].append(res_de.get('X_history', np.array([])))
                results['global_de']['Y_history'].append(res_de.get('Y_history', np.array([])))

                if verbose:
                    print(f"           best_J = {res_de['best_J']:.6f}, regret@budget = {fair_regret:.6f}, "
                          f"total_evals = {res_de['n_fbb_evals']}, time = {t_de:.2f}s")
            except Exception as e:
                t_de = time.perf_counter() - t0
                if verbose:
                    print(f"           ERROR: {type(e).__name__}: {e}")
                results['global_de']['regrets'].append(np.array([np.inf]))
                results['global_de']['final_regrets'].append(np.inf)
                results['global_de']['best_Js'].append(np.inf if not problem.maximize else -np.inf)
                results['global_de']['n_fbb_evals'].append(0)
                results['global_de']['wall_times'].append(t_de)
                results['global_de']['X_history'].append(np.array([]))
                results['global_de']['Y_history'].append(np.array([]))

            # 3. Run BO methods for each acquisition function
            for acq_idx, acq in enumerate(acquisitions):
                # Black-box BO
                if verbose:
                    print(f"    [BB-BO {acq.upper()}] ({acq_idx + 1}/{len(acquisitions)})...")
                t0 = time.perf_counter()
                try:
                    res_bo = solve_blackbox_bo(
                        problem, n_iterations=n_iterations, n_initial=n_initial,
                        acquisition=acq, seed=seed, verbose=False, return_history=True,
                        xi=xi, kappa=kappa
                    )
                    t_bo = time.perf_counter() - t0
                    G_bo = res_bo.get('G_history', None)
                    regret_bo = compute_regret(res_bo['Y_history'], problem.J_optimal, G_bo, problem.maximize)
                    results[f'blackbox_bo_{acq}']['regrets'].append(regret_bo)
                    results[f'blackbox_bo_{acq}']['best_Js'].append(res_bo['best_J'])
                    results[f'blackbox_bo_{acq}']['n_fbb_evals'].append(res_bo['n_fbb_evals'])
                    results[f'blackbox_bo_{acq}']['wall_times'].append(t_bo)
                    results[f'blackbox_bo_{acq}']['iter_times'].append(res_bo.get('iter_times', np.array([])))
                    results[f'blackbox_bo_{acq}']['X_history'].append(res_bo.get('X_history', np.array([])))
                    results[f'blackbox_bo_{acq}']['Y_history'].append(res_bo.get('Y_history', np.array([])))
                    if verbose:
                        print(f"           best_J = {res_bo['best_J']:.6f}, final_regret = {regret_bo[-1]:.6f}, time = {t_bo:.2f}s")
                except Exception as e:
                    t_bo = time.perf_counter() - t0
                    if verbose:
                        print(f"           ERROR: {type(e).__name__}: {e}")
                        print(f"           Skipping this run for BB-BO {acq.upper()}")
                    # Store NaN/inf values to indicate failure
                    results[f'blackbox_bo_{acq}']['regrets'].append(np.full(n_iterations + n_initial, np.inf))
                    results[f'blackbox_bo_{acq}']['best_Js'].append(np.inf if not problem.maximize else -np.inf)
                    results[f'blackbox_bo_{acq}']['n_fbb_evals'].append(0)
                    results[f'blackbox_bo_{acq}']['wall_times'].append(t_bo)
                    results[f'blackbox_bo_{acq}']['iter_times'].append(np.array([]))
                    results[f'blackbox_bo_{acq}']['X_history'].append(np.array([]))
                    results[f'blackbox_bo_{acq}']['Y_history'].append(np.array([]))

                # Bi-level BO
                if verbose:
                    print(f"    [Bi-BO {acq.upper()}] ({acq_idx + 1}/{len(acquisitions)})...")
                t0 = time.perf_counter()
                try:
                    res_bilevel = solve_bilevel_bo(
                        problem, n_iterations=n_iterations, n_initial=n_initial,
                        n_inner_starts=n_inner_starts, acquisition=acq, seed=seed,
                        verbose=False, return_history=True, xi=xi, kappa=kappa,
                        inner_solver=inner_solver
                    )
                    t_bi = time.perf_counter() - t0
                    G_bi = res_bilevel.get('G_history', None)
                    regret_bi = compute_regret(res_bilevel['Y_history'], problem.J_optimal, G_bi, problem.maximize)
                    results[f'bilevel_bo_{acq}']['regrets'].append(regret_bi)
                    results[f'bilevel_bo_{acq}']['best_Js'].append(res_bilevel['best_J'])
                    results[f'bilevel_bo_{acq}']['n_fbb_evals'].append(res_bilevel['n_fbb_evals'])
                    results[f'bilevel_bo_{acq}']['wall_times'].append(t_bi)
                    results[f'bilevel_bo_{acq}']['iter_times'].append(res_bilevel.get('iter_times', np.array([])))
                    results[f'bilevel_bo_{acq}']['X_bb_history'].append(res_bilevel.get('X_bb_history', np.array([])))
                    results[f'bilevel_bo_{acq}']['X_wb_history'].append(res_bilevel.get('X_wb_history', []))
                    results[f'bilevel_bo_{acq}']['Y_history'].append(res_bilevel.get('Y_history', np.array([])))
                    if verbose:
                        print(f"           best_J = {res_bilevel['best_J']:.6f}, final_regret = {regret_bi[-1]:.6f}, time = {t_bi:.2f}s")
                except Exception as e:
                    t_bi = time.perf_counter() - t0
                    if verbose:
                        print(f"           ERROR: {type(e).__name__}: {e}")
                        print(f"           Skipping this run for Bi-BO {acq.upper()}")
                    # Store NaN/inf values to indicate failure
                    results[f'bilevel_bo_{acq}']['regrets'].append(np.full(n_iterations + n_initial, np.inf))
                    results[f'bilevel_bo_{acq}']['best_Js'].append(np.inf if not problem.maximize else -np.inf)
                    results[f'bilevel_bo_{acq}']['n_fbb_evals'].append(0)
                    results[f'bilevel_bo_{acq}']['wall_times'].append(t_bi)
                    results[f'bilevel_bo_{acq}']['iter_times'].append(np.array([]))
                    results[f'bilevel_bo_{acq}']['X_bb_history'].append(np.array([]))
                    results[f'bilevel_bo_{acq}']['X_wb_history'].append([])
                    results[f'bilevel_bo_{acq}']['Y_history'].append(np.array([]))

        all_results[name] = results

    return all_results


def main(
    problem_set: str = 'cobalt',
    acquisitions: List[str] = None,
    n_iterations: int = 50,
    n_initial_values: List[int] = None,
    n_starts_nlp: int = 20,
    n_inner_starts: int = 20,
    n_repetitions: int = 10,
    base_seed: int = 42,
    results_file: Optional[str] = None,
    save_plots: bool = True,
    xi_values: List[float] = None,
    kappa: float = 2.0,
    inner_solver: str = 'global'
):
    """
    Main function to run experiments with multiple acquisition functions.

    Args:
        problem_set: 'cobalt' for minimization benchmarks, 'all' for all problems
        acquisitions: List of acquisition functions to compare (default: all available)
        n_iterations: Number of BO iterations
        n_initial_values: List of initial sample counts to sweep over
        n_starts_nlp: Multi-start for NLP
        n_inner_starts: Inner problem multi-starts
        n_repetitions: Number of independent runs
        base_seed: Base random seed
        results_file: Path to save/load results
        save_plots: Whether to save plots
        xi_values: List of xi values to sweep over (exploration-exploitation trade-off)
        kappa: Exploration parameter for LCB/UCB (higher = more exploration)
    """
    # Set default acquisitions
    if acquisitions is None:
        acquisitions = DEFAULT_ACQUISITIONS

    # Set default xi values
    if xi_values is None:
        xi_values = [0.01]

    # Set default n_initial values
    if n_initial_values is None:
        n_initial_values = [5]

    # Get problems
    if problem_set == 'cobalt':
        problems = get_cobalt_problems()
    elif problem_set == 'all':
        problems = get_all_problems()
    else:
        raise ValueError(f"Unknown problem set: {problem_set}")

    print("=" * 70)
    print("BI-LEVEL OPTIMIZATION COMPARISON")
    print("=" * 70)
    print(f"Problem set: {problem_set}")
    print(f"Acquisition functions: {', '.join([a.upper() for a in acquisitions])}")
    print(f"BO iterations: {n_iterations}")
    if len(n_initial_values) == 1:
        print(f"Initial samples: {n_initial_values[0]}")
    else:
        print(f"Initial samples sweep: {n_initial_values}")
    print(f"Repetitions: {n_repetitions}, Base seed: {base_seed}")
    if len(xi_values) == 1:
        print(f"Exploration params: xi={xi_values[0]} (EI/PI/mWB2), kappa={kappa} (LCB)")
    else:
        print(f"Xi sweep: {xi_values}")
        print(f"Kappa (LCB): {kappa}")
    print("=" * 70)

    # Run experiments for each combination of n_initial and xi
    all_sweep_results = {}

    for n_init_idx, n_initial in enumerate(n_initial_values):
        for xi_idx, xi in enumerate(xi_values):
            sweep_key = (n_initial, xi)

            if len(n_initial_values) > 1 or len(xi_values) > 1:
                print(f"\n{'#' * 70}")
                print(f"# n_init={n_initial} ({n_init_idx + 1}/{len(n_initial_values)}), xi={xi} ({xi_idx + 1}/{len(xi_values)})")
                print(f"{'#' * 70}")

            # Settings for this combination
            settings = {
                'n_iterations': n_iterations,
                'n_initial': n_initial,
                'n_starts_nlp': n_starts_nlp,
                'n_inner_starts': n_inner_starts,
                'n_repetitions': n_repetitions,
                'base_seed': base_seed,
                'problem_names': list(problems.keys()),
                'acquisitions': acquisitions,
                'xi': xi,
                'kappa': kappa,
                'inner_solver': inner_solver,
            }

            # Results file for this combination
            if results_file is None:
                acq_str = '_'.join(acquisitions)
                base_name = f'results_{problem_set}_{acq_str}'
                if len(n_initial_values) > 1:
                    base_name += f'_ninit{n_initial}'
                if len(xi_values) > 1:
                    base_name += f'_xi{xi}'
                results_dir = Path(__file__).parent / "results"
                results_dir.mkdir(exist_ok=True)
                combo_results_file = results_dir / f'{base_name}.pkl'
            else:
                base_path = Path(results_file)
                suffix_parts = []
                if len(n_initial_values) > 1:
                    suffix_parts.append(f'ninit{n_initial}')
                if len(xi_values) > 1:
                    suffix_parts.append(f'xi{xi}')
                if suffix_parts:
                    combo_results_file = base_path.parent / f'{base_path.stem}_{"_".join(suffix_parts)}{base_path.suffix}'
                else:
                    combo_results_file = base_path

            # Check if results file exists and has matching settings
            run_experiments_flag = True
            if combo_results_file.exists():
                try:
                    with open(combo_results_file, 'rb') as f:
                        saved_data = pickle.load(f)

                    saved_settings = saved_data.get('settings', {})
                    saved_results = saved_data.get('results', {})

                    settings_match = all(
                        saved_settings.get(k) == v for k, v in settings.items()
                    )

                    if settings_match:
                        all_complete = True
                        for name in settings['problem_names']:
                            if name not in saved_results:
                                all_complete = False
                                break
                            res = saved_results[name]
                            if len(res['blackbox_nlp']['final_regrets']) < n_repetitions:
                                all_complete = False
                                break
                            # Check global DE
                            de_res = res.get('global_de', {})
                            if len(de_res.get('final_regrets', [])) < n_repetitions:
                                all_complete = False
                                break
                            # Check all acquisition functions
                            for acq in acquisitions:
                                bo_key = f'blackbox_bo_{acq}'
                                bi_key = f'bilevel_bo_{acq}'
                                if bo_key not in res or bi_key not in res:
                                    all_complete = False
                                    break
                                if (len(res[bo_key]['regrets']) < n_repetitions or
                                    len(res[bi_key]['regrets']) < n_repetitions):
                                    all_complete = False
                                    break

                        if all_complete:
                            print("Found existing results with matching settings.")
                            print("Loading cached results...")
                            all_results = saved_results
                            run_experiments_flag = False
                        else:
                            print("Found results but experiments incomplete. Re-running...")
                    else:
                        print("Found results but settings differ. Re-running experiments...")

                except Exception as e:
                    print(f"Error loading results: {e}. Re-running experiments...")

            if run_experiments_flag:
                all_results = run_experiments(problems, settings, acquisitions, verbose=True)

                # Save results
                print("\n" + "=" * 70)
                print(f"Saving results to {combo_results_file}...")
                with open(combo_results_file, 'wb') as f:
                    pickle.dump({'settings': settings, 'results': all_results}, f)
                print("Results saved.")
            else:
                # Results already exist and were loaded - skip this combination
                print(f"Skipping n_init={n_initial}, xi={xi} - results already exist.")
                continue

            # Store results for this combination
            all_sweep_results[sweep_key] = all_results

            # Print summary for this combination
            print_summary(all_results, n_repetitions)

            # Plot results for this combination
            if save_plots:
                print("\n" + "=" * 70)
                print(f"Generating plots for n_init={n_initial}, xi={xi}...")
                print("=" * 70)
                plot_suffix_parts = []
                if len(n_initial_values) > 1:
                    plot_suffix_parts.append(f'ninit{n_initial}')
                if len(xi_values) > 1:
                    plot_suffix_parts.append(f'xi{xi}')
                if plot_suffix_parts:
                    save_dir = Path(__file__).parent / f"plots_{'_'.join(plot_suffix_parts)}"
                else:
                    save_dir = Path(__file__).parent / "plots"
                plot_results(all_results, save_dir=save_dir)

    # Print combined summary if sweeping over multiple values
    if len(n_initial_values) > 1 or len(xi_values) > 1:
        print("\n" + "=" * 70)
        print("SWEEP SUMMARY")
        print("=" * 70)
        print(f"n_initial values tested: {n_initial_values}")
        print(f"Xi values tested: {xi_values}")
        print("\nBest final regret by (n_init, xi) (Bi-level BO, mean across problems):")
        print("-" * 70)
        for (n_init, xi_val), results in all_sweep_results.items():
            total_regret = 0
            n_valid = 0
            for name, res in results.items():
                for acq in res.get('acquisitions', ['ei']):
                    bi_key = f'bilevel_bo_{acq}'
                    if bi_key in res and len(res[bi_key]['regrets']) > 0:
                        final_regrets = [r[-1] for r in res[bi_key]['regrets'] if not np.isinf(r[-1])]
                        if final_regrets:
                            total_regret += np.mean(final_regrets)
                            n_valid += 1
            if n_valid > 0:
                print(f"  n_init={n_init}, xi={xi_val}: avg regret = {total_regret / n_valid:.6f}")
            else:
                print(f"  n_init={n_init}, xi={xi_val}: no valid results")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Run bi-level optimization comparison with multiple acquisition functions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Acquisition functions available:
  ei       - Expected Improvement (default)
  pi       - Probability of Improvement
  lcb      - Lower Confidence Bound
  mwb2     - Modified Watson-Barnes 2 (from COBALT paper)
  thompson - Thompson Sampling

Examples:
  python run_comparison.py --acq ei pi lcb     # Compare EI, PI, and LCB
  python run_comparison.py --acq all           # Run all acquisition functions
  python run_comparison.py --problems all      # Run on all problems
  python run_comparison.py --xi 0.1            # Run with xi=0.1 (more exploration)
  python run_comparison.py --xi_sweep 0.001 0.01 0.1 1.0  # Sweep over xi values
  python run_comparison.py --n_init_sweep 1 5 20 50       # Sweep over initial samples
        """
    )
    parser.add_argument('--problems', type=str, default='all',
                        choices=['cobalt', 'all'],
                        help='Problem set to run')
    parser.add_argument('--acq', type=str, nargs='+', default=["ei"],
                        help='Acquisition functions to compare (default: ei). '
                             'Use "all" for all available, or list specific ones.')
    parser.add_argument('--n_iter', type=int, default=50,
                        help='Number of BO iterations')
    parser.add_argument('--n_init', type=int, default=5,
                        help='Initial samples for BO')
    parser.add_argument('--n_init_sweep', type=int, nargs='+', default=None,
                        help='Run experiments at multiple n_init values (e.g., --n_init_sweep 1 5 20 50)')
    parser.add_argument('--n_reps', type=int, default=10,
                        help='Number of repetitions')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed')
    parser.add_argument('--no_plots', action='store_true',
                        help='Disable plot generation')
    parser.add_argument('--xi', type=float, default=0.1,
                        help='Exploration-exploitation trade-off for EI/PI/mWB2 (default: 0.01, higher = more exploration)')
    parser.add_argument('--xi_sweep', type=float, nargs='+', default=None,
                        help='Run experiments at multiple xi values (e.g., --xi_sweep 0.001 0.01 0.1 1.0)')
    parser.add_argument('--kappa', type=float, default=2.0,
                        help='Exploration parameter for LCB (default: 2.0, higher = more exploration)')
    parser.add_argument('--inner_solver', type=str, default='global',
                        choices=['multistart', 'global'],
                        help='Inner solver for bi-level BO (default: global)')

    args = parser.parse_args()

    # Handle acquisition function argument
    if args.acq is None or args.acq == ['all']:
        acquisitions = DEFAULT_ACQUISITIONS
    else:
        acquisitions = [a.lower() for a in args.acq]
        # Validate
        for acq in acquisitions:
            if acq not in AVAILABLE_ACQUISITIONS:
                parser.error(f"Unknown acquisition function: {acq}. "
                           f"Available: {AVAILABLE_ACQUISITIONS}")

    # Handle xi sweep vs single xi value
    if args.xi_sweep is not None:
        xi_values = args.xi_sweep
    else:
        xi_values = [args.xi]

    # Handle n_init sweep vs single n_init value
    if args.n_init_sweep is not None:
        n_initial_values = args.n_init_sweep
    else:
        n_initial_values = [args.n_init]

    main(
        problem_set=args.problems,
        acquisitions=acquisitions,
        n_iterations=args.n_iter,
        n_initial_values=n_initial_values,
        n_repetitions=args.n_reps,
        base_seed=args.seed,
        save_plots=not args.no_plots,
        xi_values=xi_values,
        kappa=args.kappa,
        inner_solver=args.inner_solver
    )
