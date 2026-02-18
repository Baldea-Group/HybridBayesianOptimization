"""
plotting.py - Plotting functions for bi-level optimization comparison

This module contains visualization functions for comparing optimization methods
and acquisition functions on bi-level test problems.
"""

import numpy as np
import os
from pathlib import Path
from typing import Dict, Any, Optional

import matplotlib.pyplot as plt
from scipy import stats


# Seaborn colorblind palette - distinct colors for colorblind accessibility
COLORBLIND_PALETTE = [
    '#0173B2',  # blue
    '#DE8F05',  # orange
    '#029E73',  # green
    '#D55E00',  # vermillion
    '#CC78BC',  # purple
    '#CA9161',  # brown
    '#FBAFE4',  # pink
    '#949494',  # gray
    '#ECE133',  # yellow
    '#56B4E9',  # sky blue
]

# Map acquisition functions to colorblind palette
ALL_ACQUISITIONS = ['ei', 'pi', 'lcb', 'mwb2', 'thompson']
ACQ_COLORS = {acq: COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)]
              for i, acq in enumerate(ALL_ACQUISITIONS)}

# Default number of rows for multi-problem plots
DEFAULT_NROWS = 3


def compute_grid_layout(n_items: int, n_rows: int = DEFAULT_NROWS) -> tuple:
    """
    Compute grid layout for plotting multiple items in n_rows.

    Args:
        n_items: Number of items to plot
        n_rows: Target number of rows

    Returns:
        Tuple of (n_rows, n_cols) for the grid
    """
    if n_items <= n_rows:
        return n_items, 1
    n_cols = int(np.ceil(n_items / n_rows))
    return n_rows, n_cols

# Method colors for single-acquisition plot
METHOD_COLORS = {
    'blackbox_bo': '#0173B2',   # blue
    'bilevel_bo': '#D55E00',    # vermillion/orange-red
    'nlp': '#949494'            # gray
}


def print_summary(all_results: Dict[str, Any], n_repetitions: int):
    """Print summary tables for multiple acquisition functions."""
    print("\n\n" + "=" * 100)
    print(f"SUMMARY (n={n_repetitions} repetitions, mean ± std)")
    print("=" * 100)

    for name, res in all_results.items():
        J_opt = res['J_optimal']
        acquisitions = res.get('acquisitions', ['ei'])

        print(f"\n{'=' * 100}")
        print(f"Problem: {name} (J* = {J_opt:.4f})")
        print("=" * 100)

        # NLP result (independent of acquisition)
        nlp_Js = np.array(res['blackbox_nlp']['best_Js'])
        nlp_regrets = np.array(res['blackbox_nlp']['final_regrets'])
        print(f"\nBlack-box NLP:")
        print(f"  Best J: {np.mean(nlp_Js):.4f} ± {np.std(nlp_Js):.4f}")
        print(f"  Regret: {np.mean(nlp_regrets):.4f} ± {np.std(nlp_regrets):.4f}")

        # Table header for BO methods
        print(f"\n{'Method':<15} {'Acquisition':<12} {'Best J':<20} {'Final Regret':<20}")
        print("-" * 70)

        for acq in acquisitions:
            # Black-box BO
            bo_key = f'blackbox_bo_{acq}'
            if bo_key in res:
                bo_Js = np.array(res[bo_key]['best_Js'])
                bo_regrets = np.array([r[-1] for r in res[bo_key]['regrets']])
                j_str = f"{np.mean(bo_Js):.4f} ± {np.std(bo_Js):.4f}"
                r_str = f"{np.mean(bo_regrets):.4f} ± {np.std(bo_regrets):.4f}"
                print(f"{'Black-box BO':<15} {acq.upper():<12} {j_str:<20} {r_str:<20}")

        print("-" * 70)

        for acq in acquisitions:
            # Bi-level BO
            bi_key = f'bilevel_bo_{acq}'
            if bi_key in res:
                bi_Js = np.array(res[bi_key]['best_Js'])
                bi_regrets = np.array([r[-1] for r in res[bi_key]['regrets']])
                j_str = f"{np.mean(bi_Js):.4f} ± {np.std(bi_Js):.4f}"
                r_str = f"{np.mean(bi_regrets):.4f} ± {np.std(bi_regrets):.4f}"
                print(f"{'Bi-level BO':<15} {acq.upper():<12} {j_str:<20} {r_str:<20}")

    # Summary table comparing best acquisition for each method
    print("\n\n" + "=" * 100)
    print("BEST ACQUISITION FUNCTION PER PROBLEM (by final regret)")
    print("=" * 100)
    print(f"{'Problem':<20} {'Best BB-BO':<25} {'Best Bi-level BO':<25}")
    print("-" * 70)

    for name, res in all_results.items():
        acquisitions = res.get('acquisitions', ['ei'])

        # Find best BB-BO
        best_bo_acq = None
        best_bo_regret = np.inf
        for acq in acquisitions:
            bo_key = f'blackbox_bo_{acq}'
            if bo_key in res:
                mean_regret = np.mean([r[-1] for r in res[bo_key]['regrets']])
                if mean_regret < best_bo_regret:
                    best_bo_regret = mean_regret
                    best_bo_acq = acq

        # Find best Bi-level BO
        best_bi_acq = None
        best_bi_regret = np.inf
        for acq in acquisitions:
            bi_key = f'bilevel_bo_{acq}'
            if bi_key in res:
                mean_regret = np.mean([r[-1] for r in res[bi_key]['regrets']])
                if mean_regret < best_bi_regret:
                    best_bi_regret = mean_regret
                    best_bi_acq = acq

        bo_str = f"{best_bo_acq.upper()} ({best_bo_regret:.4f})" if best_bo_acq else "N/A"
        bi_str = f"{best_bi_acq.upper()} ({best_bi_regret:.4f})" if best_bi_acq else "N/A"
        print(f"{name:<20} {bo_str:<25} {bi_str:<25}")


def plot_regret_vs_iteration(all_results: Dict[str, Any], save_dir: Optional[Path] = None):
    """
    Plot regret vs iteration for each problem.

    For single acquisition: BB-BO and Bi-level on same plot.
    For multiple acquisitions: BB-BO and Bi-level in separate rows.
    """
    n_problems = len(all_results)

    # Determine if we have single or multiple acquisition functions
    first_res = list(all_results.values())[0]
    acquisitions_used = first_res.get('acquisitions', ['ei'])
    single_acq_mode = len(acquisitions_used) == 1

    if single_acq_mode:
        # Single acquisition: plot both methods on same subplot
        n_rows, n_cols = compute_grid_layout(n_problems)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_problems == 1:
            axes = np.array([[axes]])
        axes = np.atleast_2d(axes)

        acq = acquisitions_used[0]

        for idx, (name, res) in enumerate(all_results.items()):
            row, col = divmod(idx, n_cols)
            J_optimal = res['J_optimal']
            ax = axes[row, col]

            # Collect all y-values for axis limits
            all_y_values = []

            # Black-box BO
            bo_key = f'blackbox_bo_{acq}'
            if bo_key in res and len(res[bo_key]['regrets']) > 0:
                regrets = np.array(res[bo_key]['regrets'])
                valid_mask = ~np.isinf(regrets).any(axis=1)
                if np.any(valid_mask):
                    regrets = regrets[valid_mask]
                    n_runs, n_iters = regrets.shape
                    mean_regret = np.mean(regrets, axis=0)
                    sem = stats.sem(regrets, axis=0)
                    ci = 1.96 * sem

                    valid_y = mean_regret[mean_regret > 0]
                    if len(valid_y) > 0:
                        all_y_values.extend(valid_y)
                        all_y_values.extend((mean_regret + ci)[mean_regret + ci > 0])
                        all_y_values.extend((mean_regret - ci)[(mean_regret - ci) > 0])

                    iters = range(1, n_iters + 1)
                    ax.plot(iters, mean_regret, color=METHOD_COLORS['blackbox_bo'],
                           linestyle='-', label='Black-box BO', linewidth=2)
                    ax.fill_between(iters, np.maximum(mean_regret - ci, 1e-10),
                                   mean_regret + ci, color=METHOD_COLORS['blackbox_bo'], alpha=0.15)

            # Bi-level BO
            bi_key = f'bilevel_bo_{acq}'
            if bi_key in res and len(res[bi_key]['regrets']) > 0:
                regrets = np.array(res[bi_key]['regrets'])
                valid_mask = ~np.isinf(regrets).any(axis=1)
                if np.any(valid_mask):
                    regrets = regrets[valid_mask]
                    n_runs, n_iters = regrets.shape
                    mean_regret = np.mean(regrets, axis=0)
                    sem = stats.sem(regrets, axis=0)
                    ci = 1.96 * sem

                    valid_y = mean_regret[mean_regret > 0]
                    if len(valid_y) > 0:
                        all_y_values.extend(valid_y)
                        all_y_values.extend((mean_regret + ci)[mean_regret + ci > 0])
                        all_y_values.extend((mean_regret - ci)[(mean_regret - ci) > 0])

                    iters = range(1, n_iters + 1)
                    ax.plot(iters, mean_regret, color=METHOD_COLORS['bilevel_bo'],
                           linestyle='-', label='Bi-level BO', linewidth=2)
                    ax.fill_between(iters, np.maximum(mean_regret - ci, 1e-10),
                                   mean_regret + ci, color=METHOD_COLORS['bilevel_bo'], alpha=0.15)

            # Add NLP reference line
            if 'final_regrets' in res['blackbox_nlp']:
                regrets_nlp = np.array(res['blackbox_nlp']['final_regrets'])
                mean_nlp = np.mean(regrets_nlp)
                if mean_nlp > 0:
                    all_y_values.append(mean_nlp)
                ax.axhline(y=mean_nlp, color=METHOD_COLORS['nlp'], linestyle=':',
                          label=f'NLP ({mean_nlp:.4f})', linewidth=1.5, alpha=0.7)

            # Set axis limits
            if len(all_y_values) > 0:
                y_min = max(min(all_y_values) * 0.5, 1e-10)
                y_max = max(all_y_values) * 2.0
                ax.set_ylim(y_min, y_max)

            # Format axis
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Regret')
            ax.set_title(f'{name}\n(J* = {J_optimal:.4f}, Acq: {acq.upper()})')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

        # Hide unused axes
        for idx in range(n_problems, n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes[row, col].set_visible(False)

        fig.suptitle('Regret vs Iteration - Method Comparison', fontsize=14, fontweight='bold')
        fig.tight_layout()

    else:
        # Multiple acquisitions: separate rows for BB-BO and Bi-level BO
        n_rows, n_cols = compute_grid_layout(n_problems)
        fig, axes_bb = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        fig_bi, axes_bi = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_problems == 1:
            axes_bb = np.array([[axes_bb]])
            axes_bi = np.array([[axes_bi]])
        axes_bb = np.atleast_2d(axes_bb)
        axes_bi = np.atleast_2d(axes_bi)

        for idx, (name, res) in enumerate(all_results.items()):
            row, col = divmod(idx, n_cols)
            J_optimal = res['J_optimal']
            acquisitions = res.get('acquisitions', ['ei'])

            # Black-box BO subplot
            ax_bb = axes_bb[row, col]
            # Bi-level BO subplot
            ax_bi = axes_bi[row, col]

            # Collect all y-values for shared axis limits
            all_y_values = []

            for acq in acquisitions:
                color = ACQ_COLORS.get(acq, '#949494')

                # Black-box BO
                bo_key = f'blackbox_bo_{acq}'
                if bo_key in res and len(res[bo_key]['regrets']) > 0:
                    regrets = np.array(res[bo_key]['regrets'])
                    # Filter out inf values for plotting
                    valid_mask = ~np.isinf(regrets).any(axis=1)
                    if np.any(valid_mask):
                        regrets = regrets[valid_mask]
                        n_runs, n_iters = regrets.shape
                        mean_regret = np.mean(regrets, axis=0)
                        sem = stats.sem(regrets, axis=0)
                        ci = 1.96 * sem

                        # Collect for axis limits (filter out non-positive for log scale)
                        valid_y = mean_regret[mean_regret > 0]
                        if len(valid_y) > 0:
                            all_y_values.extend(valid_y)
                            all_y_values.extend((mean_regret + ci)[mean_regret + ci > 0])
                            all_y_values.extend((mean_regret - ci)[(mean_regret - ci) > 0])

                        iters = range(1, n_iters + 1)
                        ax_bb.plot(iters, mean_regret, color=color, linestyle='-',
                                  label=f'{acq.upper()}', linewidth=2)
                        ax_bb.fill_between(iters, np.maximum(mean_regret - ci, 1e-10),
                                          mean_regret + ci, color=color, alpha=0.15)

                # Bi-level BO
                bi_key = f'bilevel_bo_{acq}'
                if bi_key in res and len(res[bi_key]['regrets']) > 0:
                    regrets = np.array(res[bi_key]['regrets'])
                    # Filter out inf values for plotting
                    valid_mask = ~np.isinf(regrets).any(axis=1)
                    if np.any(valid_mask):
                        regrets = regrets[valid_mask]
                        n_runs, n_iters = regrets.shape
                        mean_regret = np.mean(regrets, axis=0)
                        sem = stats.sem(regrets, axis=0)
                        ci = 1.96 * sem

                        # Collect for axis limits
                        valid_y = mean_regret[mean_regret > 0]
                        if len(valid_y) > 0:
                            all_y_values.extend(valid_y)
                            all_y_values.extend((mean_regret + ci)[mean_regret + ci > 0])
                            all_y_values.extend((mean_regret - ci)[(mean_regret - ci) > 0])

                        iters = range(1, n_iters + 1)
                        ax_bi.plot(iters, mean_regret, color=color, linestyle='-',
                                  label=f'{acq.upper()}', linewidth=2)
                        ax_bi.fill_between(iters, np.maximum(mean_regret - ci, 1e-10),
                                          mean_regret + ci, color=color, alpha=0.15)

            # Add NLP reference line to both
            if 'final_regrets' in res['blackbox_nlp']:
                regrets_nlp = np.array(res['blackbox_nlp']['final_regrets'])
                mean_nlp = np.mean(regrets_nlp)
                if mean_nlp > 0:
                    all_y_values.append(mean_nlp)
                for ax in [ax_bb, ax_bi]:
                    ax.axhline(y=mean_nlp, color='black', linestyle=':',
                              label=f'NLP ({mean_nlp:.4f})', linewidth=1.5, alpha=0.7)

            # Set shared y-axis limits for top and bottom plots
            if len(all_y_values) > 0:
                y_min = max(min(all_y_values) * 0.5, 1e-10)
                y_max = max(all_y_values) * 2.0
                ax_bb.set_ylim(y_min, y_max)
                ax_bi.set_ylim(y_min, y_max)

            # Format axes
            for ax, method_name in [(ax_bb, 'Black-box BO'), (ax_bi, 'Bi-level BO')]:
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Regret')
                ax.set_title(f'{name} - {method_name}\n(J* = {J_optimal:.4f})')
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_yscale('log')

        # Hide unused axes
        for idx in range(n_problems, n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes_bb[row, col].set_visible(False)
            axes_bi[row, col].set_visible(False)

        fig.suptitle('Black-box BO - Regret vs Iteration', fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig_bi.suptitle('Bi-level BO - Regret vs Iteration', fontsize=14, fontweight='bold')
        fig_bi.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, 'regret_vs_iteration_bb.png'), dpi=150, bbox_inches='tight')
        if not single_acq_mode:
            fig_bi.savefig(os.path.join(save_dir, 'regret_vs_iteration_bi.png'), dpi=150, bbox_inches='tight')

    return fig if single_acq_mode else (fig, fig_bi)


def plot_final_regret_bars(all_results: Dict[str, Any], save_dir: Optional[Path] = None):
    """Plot final regret bar chart grouped by problem and method."""
    n_problems = len(all_results)
    n_rows, n_cols = compute_grid_layout(n_problems)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 6*n_rows))
    if n_problems == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)

    for idx, (name, res) in enumerate(all_results.items()):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        acquisitions = res.get('acquisitions', ['ei'])
        n_acq = len(acquisitions)

        # Prepare data
        bb_means = []
        bb_stds = []
        bi_means = []
        bi_stds = []

        for acq in acquisitions:
            bo_key = f'blackbox_bo_{acq}'
            bi_key = f'bilevel_bo_{acq}'

            if bo_key in res and len(res[bo_key]['regrets']) > 0:
                final_regrets = [r[-1] for r in res[bo_key]['regrets']]
                bb_means.append(np.mean(final_regrets))
                bb_stds.append(np.std(final_regrets))
            else:
                bb_means.append(0)
                bb_stds.append(0)

            if bi_key in res and len(res[bi_key]['regrets']) > 0:
                final_regrets = [r[-1] for r in res[bi_key]['regrets']]
                bi_means.append(np.mean(final_regrets))
                bi_stds.append(np.std(final_regrets))
            else:
                bi_means.append(0)
                bi_stds.append(0)

        x = np.arange(n_acq)
        width = 0.35

        ax.bar(x - width/2, bb_means, width, yerr=bb_stds,
               label='Black-box BO', color='steelblue', alpha=0.8, capsize=3)
        ax.bar(x + width/2, bi_means, width, yerr=bi_stds,
               label='Bi-level BO', color='coral', alpha=0.8, capsize=3)

        # Add NLP reference
        if 'final_regrets' in res['blackbox_nlp']:
            mean_nlp = np.mean(res['blackbox_nlp']['final_regrets'])
            ax.axhline(y=mean_nlp, color='green', linestyle='--',
                      label=f'NLP ({mean_nlp:.4f})', linewidth=2)

        ax.set_xlabel('Acquisition Function')
        ax.set_ylabel('Final Regret')
        ax.set_title(f'{name}')
        ax.set_xticks(x)
        ax.set_xticklabels([a.upper() for a in acquisitions])
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_yscale('log')

    # Hide unused axes
    for idx in range(n_problems, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle('Final Regret by Acquisition Function', fontsize=14, fontweight='bold')
    fig.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, 'final_regret_by_acq.png'), dpi=150, bbox_inches='tight')

    return fig


def plot_method_comparison(all_results: Dict[str, Any], save_dir: Optional[Path] = None):
    """Plot method comparison using best acquisition for each method."""
    fig, ax = plt.subplots(figsize=(10, 6))

    problem_names = list(all_results.keys())
    x = np.arange(len(problem_names))
    width = 0.25

    nlp_regrets = []
    best_bb_regrets = []
    best_bi_regrets = []
    best_bb_acqs = []
    best_bi_acqs = []

    for name in problem_names:
        res = all_results[name]
        acquisitions = res.get('acquisitions', ['ei'])

        # NLP
        nlp_regrets.append(np.mean(res['blackbox_nlp']['final_regrets']))

        # Best BB-BO
        best_bb = np.inf
        best_bb_acq = 'ei'
        for acq in acquisitions:
            bo_key = f'blackbox_bo_{acq}'
            if bo_key in res and len(res[bo_key]['regrets']) > 0:
                mean_reg = np.mean([r[-1] for r in res[bo_key]['regrets']])
                if mean_reg < best_bb:
                    best_bb = mean_reg
                    best_bb_acq = acq
        best_bb_regrets.append(best_bb)
        best_bb_acqs.append(best_bb_acq)

        # Best Bi-level BO
        best_bi = np.inf
        best_bi_acq = 'ei'
        for acq in acquisitions:
            bi_key = f'bilevel_bo_{acq}'
            if bi_key in res and len(res[bi_key]['regrets']) > 0:
                mean_reg = np.mean([r[-1] for r in res[bi_key]['regrets']])
                if mean_reg < best_bi:
                    best_bi = mean_reg
                    best_bi_acq = acq
        best_bi_regrets.append(best_bi)
        best_bi_acqs.append(best_bi_acq)

    bars1 = ax.bar(x - width, nlp_regrets, width, label='NLP', color='green', alpha=0.7)
    bars2 = ax.bar(x, best_bb_regrets, width, label='Best BB-BO', color='steelblue', alpha=0.7)
    bars3 = ax.bar(x + width, best_bi_regrets, width, label='Best Bi-level', color='coral', alpha=0.7)

    # Add acquisition labels on bars
    for i, (bar, acq) in enumerate(zip(bars2, best_bb_acqs)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                acq.upper(), ha='center', va='bottom', fontsize=8, rotation=45)
    for i, (bar, acq) in enumerate(zip(bars3, best_bi_acqs)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                acq.upper(), ha='center', va='bottom', fontsize=8, rotation=45)

    ax.set_xlabel('Problem')
    ax.set_ylabel('Final Regret')
    ax.set_title('Best Method Comparison (with best acquisition)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(problem_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')

    fig.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, 'best_method_comparison.png'), dpi=150, bbox_inches='tight')

    return fig


def plot_wall_time(all_results: Dict[str, Any], save_dir: Optional[Path] = None):
    """Plot total wall time comparison bar chart."""
    n_problems = len(all_results)
    n_rows, n_cols = compute_grid_layout(n_problems)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 6*n_rows))
    if n_problems == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)

    for idx, (name, res) in enumerate(all_results.items()):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        acquisitions = res.get('acquisitions', ['ei'])
        n_acq = len(acquisitions)

        # Prepare data
        bb_means = []
        bb_stds = []
        bi_means = []
        bi_stds = []

        for acq in acquisitions:
            bo_key = f'blackbox_bo_{acq}'
            bi_key = f'bilevel_bo_{acq}'

            if bo_key in res and len(res[bo_key]['wall_times']) > 0:
                times = np.array(res[bo_key]['wall_times'])
                bb_means.append(np.mean(times))
                bb_stds.append(np.std(times))
            else:
                bb_means.append(0)
                bb_stds.append(0)

            if bi_key in res and len(res[bi_key]['wall_times']) > 0:
                times = np.array(res[bi_key]['wall_times'])
                bi_means.append(np.mean(times))
                bi_stds.append(np.std(times))
            else:
                bi_means.append(0)
                bi_stds.append(0)

        x = np.arange(n_acq)
        width = 0.35

        ax.bar(x - width/2, bb_means, width, yerr=bb_stds,
               label='Black-box BO', color=METHOD_COLORS['blackbox_bo'], alpha=0.8, capsize=3)
        ax.bar(x + width/2, bi_means, width, yerr=bi_stds,
               label='Bi-level BO', color=METHOD_COLORS['bilevel_bo'], alpha=0.8, capsize=3)

        # Add NLP reference line
        if 'wall_times' in res['blackbox_nlp'] and len(res['blackbox_nlp']['wall_times']) > 0:
            mean_nlp = np.mean(res['blackbox_nlp']['wall_times'])
            ax.axhline(y=mean_nlp, color=METHOD_COLORS['nlp'], linestyle='--',
                      label=f'NLP ({mean_nlp:.2f}s)', linewidth=2)

        ax.set_xlabel('Acquisition Function')
        ax.set_ylabel('Wall Time (seconds)')
        ax.set_title(f'{name}')
        ax.set_xticks(x)
        ax.set_xticklabels([a.upper() for a in acquisitions])
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

    # Hide unused axes
    for idx in range(n_problems, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle('Total Wall Time by Acquisition Function', fontsize=14, fontweight='bold')
    fig.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, 'total_wall_time.png'), dpi=150, bbox_inches='tight')

    return fig


def plot_time_per_iteration(all_results: Dict[str, Any], save_dir: Optional[Path] = None):
    """Plot time per iteration vs iteration."""
    n_problems = len(all_results)

    # Determine if we have single or multiple acquisition functions
    first_res = list(all_results.values())[0]
    acquisitions_used = first_res.get('acquisitions', ['ei'])
    single_acq_mode = len(acquisitions_used) == 1

    if single_acq_mode:
        # Single acquisition: plot both methods on same subplot
        n_rows, n_cols = compute_grid_layout(n_problems)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_problems == 1:
            axes = np.array([[axes]])
        axes = np.atleast_2d(axes)

        acq = acquisitions_used[0]

        for idx, (name, res) in enumerate(all_results.items()):
            row, col = divmod(idx, n_cols)
            ax = axes[row, col]

            # Black-box BO
            bo_key = f'blackbox_bo_{acq}'
            if bo_key in res and 'iter_times' in res[bo_key] and len(res[bo_key]['iter_times']) > 0:
                # Stack all iteration times across repetitions
                valid_times = [t for t in res[bo_key]['iter_times'] if len(t) > 0]
                if len(valid_times) > 0:
                    iter_times_arr = np.array(valid_times)
                    n_iters = iter_times_arr.shape[1]
                    mean_times = np.mean(iter_times_arr, axis=0)
                    sem_times = stats.sem(iter_times_arr, axis=0)
                    ci = 1.96 * sem_times

                    iters = range(1, n_iters + 1)
                    ax.plot(iters, mean_times, color=METHOD_COLORS['blackbox_bo'],
                           linestyle='-', label='Black-box BO', linewidth=2)
                    ax.fill_between(iters, np.maximum(mean_times - ci, 0),
                                   mean_times + ci, color=METHOD_COLORS['blackbox_bo'], alpha=0.15)

            # Bi-level BO
            bi_key = f'bilevel_bo_{acq}'
            if bi_key in res and 'iter_times' in res[bi_key] and len(res[bi_key]['iter_times']) > 0:
                valid_times = [t for t in res[bi_key]['iter_times'] if len(t) > 0]
                if len(valid_times) > 0:
                    iter_times_arr = np.array(valid_times)
                    n_iters = iter_times_arr.shape[1]
                    mean_times = np.mean(iter_times_arr, axis=0)
                    sem_times = stats.sem(iter_times_arr, axis=0)
                    ci = 1.96 * sem_times

                    iters = range(1, n_iters + 1)
                    ax.plot(iters, mean_times, color=METHOD_COLORS['bilevel_bo'],
                           linestyle='-', label='Bi-level BO', linewidth=2)
                    ax.fill_between(iters, np.maximum(mean_times - ci, 0),
                                   mean_times + ci, color=METHOD_COLORS['bilevel_bo'], alpha=0.15)

            ax.set_xlabel('Iteration')
            ax.set_ylabel('Time per Iteration (seconds)')
            ax.set_title(f'{name} (Acq: {acq.upper()})')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

        # Hide unused axes
        for idx in range(n_problems, n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes[row, col].set_visible(False)

        fig.suptitle('Time per Iteration - Method Comparison', fontsize=14, fontweight='bold')
        fig.tight_layout()

    else:
        # Multiple acquisitions: separate figures for BB-BO and Bi-level BO
        n_rows, n_cols = compute_grid_layout(n_problems)
        fig, axes_bb = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        fig_bi, axes_bi = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_problems == 1:
            axes_bb = np.array([[axes_bb]])
            axes_bi = np.array([[axes_bi]])
        axes_bb = np.atleast_2d(axes_bb)
        axes_bi = np.atleast_2d(axes_bi)

        for idx, (name, res) in enumerate(all_results.items()):
            row, col = divmod(idx, n_cols)
            acquisitions = res.get('acquisitions', ['ei'])

            ax_bb = axes_bb[row, col]
            ax_bi = axes_bi[row, col]

            for acq in acquisitions:
                color = ACQ_COLORS.get(acq, '#949494')

                # Black-box BO
                bo_key = f'blackbox_bo_{acq}'
                if bo_key in res and 'iter_times' in res[bo_key] and len(res[bo_key]['iter_times']) > 0:
                    valid_times = [t for t in res[bo_key]['iter_times'] if len(t) > 0]
                    if len(valid_times) > 0:
                        iter_times_arr = np.array(valid_times)
                        n_iters = iter_times_arr.shape[1]
                        mean_times = np.mean(iter_times_arr, axis=0)
                        sem_times = stats.sem(iter_times_arr, axis=0)
                        ci = 1.96 * sem_times

                        iters = range(1, n_iters + 1)
                        ax_bb.plot(iters, mean_times, color=color, linestyle='-',
                                  label=f'{acq.upper()}', linewidth=2)
                        ax_bb.fill_between(iters, np.maximum(mean_times - ci, 0),
                                          mean_times + ci, color=color, alpha=0.15)

                # Bi-level BO
                bi_key = f'bilevel_bo_{acq}'
                if bi_key in res and 'iter_times' in res[bi_key] and len(res[bi_key]['iter_times']) > 0:
                    valid_times = [t for t in res[bi_key]['iter_times'] if len(t) > 0]
                    if len(valid_times) > 0:
                        iter_times_arr = np.array(valid_times)
                        n_iters = iter_times_arr.shape[1]
                        mean_times = np.mean(iter_times_arr, axis=0)
                        sem_times = stats.sem(iter_times_arr, axis=0)
                        ci = 1.96 * sem_times

                        iters = range(1, n_iters + 1)
                        ax_bi.plot(iters, mean_times, color=color, linestyle='-',
                                  label=f'{acq.upper()}', linewidth=2)
                        ax_bi.fill_between(iters, np.maximum(mean_times - ci, 0),
                                          mean_times + ci, color=color, alpha=0.15)

            # Format axes
            ax_bb.set_xlabel('Iteration')
            ax_bb.set_ylabel('Time per Iteration (seconds)')
            ax_bb.set_title(f'{name}')
            ax_bb.legend(loc='upper right', fontsize=8)
            ax_bb.grid(True, alpha=0.3)

            ax_bi.set_xlabel('Iteration')
            ax_bi.set_ylabel('Time per Iteration (seconds)')
            ax_bi.set_title(f'{name}')
            ax_bi.legend(loc='upper right', fontsize=8)
            ax_bi.grid(True, alpha=0.3)

        # Hide unused axes
        for idx in range(n_problems, n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes_bb[row, col].set_visible(False)
            axes_bi[row, col].set_visible(False)

        fig.suptitle('Black-box BO - Time per Iteration', fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig_bi.suptitle('Bi-level BO - Time per Iteration', fontsize=14, fontweight='bold')
        fig_bi.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, 'time_per_iteration_bb.png'), dpi=150, bbox_inches='tight')
        if not single_acq_mode:
            fig_bi.savefig(os.path.join(save_dir, 'time_per_iteration_bi.png'), dpi=150, bbox_inches='tight')

    return fig if single_acq_mode else (fig, fig_bi)


def plot_results(all_results: Dict[str, Any], save_dir: Optional[Path] = None):
    """
    Create all plots comparing methods and acquisition functions.

    This is a convenience function that calls all individual plotting functions:
    1. Regret vs iteration for each problem (comparing acquisitions)
    2. Final regret bar chart (comparing methods and acquisitions)
    3. Best acquisition comparison (Black-box BO vs Bi-level BO)
    4. Total wall time comparison
    5. Time per iteration vs iteration

    Args:
        all_results: Dictionary with all experiment results
        save_dir: Optional directory to save plots
    """
    fig1 = plot_regret_vs_iteration(all_results, save_dir)
    fig2 = plot_final_regret_bars(all_results, save_dir)
    fig3 = plot_method_comparison(all_results, save_dir)
    fig4 = plot_wall_time(all_results, save_dir)
    fig5 = plot_time_per_iteration(all_results, save_dir)

    if save_dir:
        print(f"Figures saved to {save_dir}/")

    # Close figures to free memory (don't block with plt.show())
    plt.close('all')

    return fig1, fig2, fig3, fig4, fig5


# =============================================================================
# Animation for Hybrid BO on Rastrigin Problem
# =============================================================================

def generate_rastrigin_ground_truth(n_points: int = 20000, n_inner_starts: int = 10):
    """
    Generate ground truth data for the Rastrigin bi-level problem.

    For each xm value, solve the inner optimization problem to get J*(xm).

    Args:
        n_points: Number of xm points to evaluate
        n_inner_starts: Number of multi-starts for inner optimization

    Returns:
        x_bb_grid: Array of xm values
        J_star: Array of optimal objective values J*(xm)
    """
    from functions import create_rastrigin

    problem = create_rastrigin()
    rng = np.random.default_rng(42)

    # Generate uniform grid of xm values
    x_bb_grid = np.linspace(problem.x_bb_lower[0], problem.x_bb_upper[0], n_points)
    J_star = np.zeros(n_points)

    print(f"Generating ground truth with {n_points} points...")
    for i, x_bb_val in enumerate(x_bb_grid):
        x_bb = np.array([x_bb_val])
        J_star[i] = problem.evaluate_bilevel(x_bb, n_starts=n_inner_starts, rng=rng)
        if (i + 1) % 2000 == 0:
            print(f"  Computed {i + 1}/{n_points} points...")

    print("Ground truth generation complete.")
    return x_bb_grid, J_star


def run_bilevel_bo_with_history(
    n_iterations: int = 30,
    n_initial: int = 5,
    n_inner_starts: int = 10,
    acquisition: str = 'ei',
    seed: int = 42
):
    """
    Run bi-level BO on Rastrigin and return full history for animation.

    Returns detailed history including GP models at each iteration.

    Args:
        n_iterations: Number of BO iterations
        n_initial: Number of initial samples
        n_inner_starts: Multi-starts for inner problem
        acquisition: Acquisition function name
        seed: Random seed

    Returns:
        Dictionary with full optimization history
    """
    from functions import create_rastrigin
    from solvers import create_gp, get_acquisition_function

    problem = create_rastrigin()
    rng = np.random.default_rng(seed)

    n_total = n_initial + n_iterations

    # Storage
    X_bb_history = []  # All xm samples
    Y_history = []   # All objective values
    gp_history = []  # GP model snapshots (params only)
    acq_history = [] # Acquisition values on grid at each iteration
    next_x_bb_history = []  # Next point selected

    # Grid for GP predictions and acquisition
    x_bb_grid = np.linspace(problem.x_bb_lower[0], problem.x_bb_upper[0], 10000).reshape(-1, 1)

    # Initial sampling
    print(f"Generating {n_initial} initial samples...")
    X_bb_sample = []
    Y_sample = []

    for i in range(n_initial):
        x_bb = rng.uniform(problem.x_bb_lower, problem.x_bb_upper)
        J_val = problem.evaluate_bilevel(x_bb, n_starts=n_inner_starts, rng=rng)
        X_bb_sample.append(x_bb)
        Y_sample.append(J_val)

        # Store history for each initial point
        X_bb_history.append(np.array(X_bb_sample).copy())
        Y_history.append(np.array(Y_sample).copy())
        gp_history.append(None)  # No GP for initial points
        acq_history.append(None)
        next_x_bb_history.append(x_bb)

    X_bb_sample = np.array(X_bb_sample)
    Y_sample = np.array(Y_sample)

    # GP setup
    gpr = create_gp(kernel_type='matern', n_dim=problem.n_x_bb)

    print(f"Running {n_iterations} BO iterations...")
    for i in range(n_iterations):
        # Fit GP
        gpr.fit(X_bb_sample, Y_sample)

        # Store GP predictions on grid
        mu, sigma = gpr.predict(x_bb_grid, return_std=True)

        # Get acquisition function
        acq_func = get_acquisition_function(
            acquisition,
            maximize=False,
            n_iter=n_initial + i,
            n_total=n_total,
            rng=rng
        )

        # Compute acquisition on grid
        acq_values = acq_func(x_bb_grid, gpr, np.min(Y_sample))

        # Find next point
        best_idx = np.argmax(acq_values)
        x_bb_next = x_bb_grid[best_idx].flatten()

        # Evaluate
        J_next = problem.evaluate_bilevel(x_bb_next, n_starts=n_inner_starts, rng=rng)

        # Store before updating
        gp_history.append({
            'mu': mu.copy(),
            'sigma': sigma.copy(),
            'kernel_params': gpr.kernel_.get_params() if gpr.kernel_ else None
        })
        acq_history.append(acq_values.copy())
        next_x_bb_history.append(x_bb_next.copy())

        # Update samples
        X_bb_sample = np.vstack([X_bb_sample, x_bb_next])
        Y_sample = np.append(Y_sample, J_next)

        X_bb_history.append(X_bb_sample.copy())
        Y_history.append(Y_sample.copy())

        if (i + 1) % 10 == 0:
            print(f"  Iteration {i + 1}/{n_iterations}: best_J = {np.min(Y_sample):.4f}")

    return {
        'X_bb_history': X_bb_history,
        'Y_history': Y_history,
        'gp_history': gp_history,
        'acq_history': acq_history,
        'next_x_bb_history': next_x_bb_history,
        'x_bb_grid': x_bb_grid.flatten(),
        'n_initial': n_initial,
        'n_iterations': n_iterations,
        'acquisition': acquisition
    }


def animate_bilevel_bo_rastrigin(
    n_iterations: int = 30,
    n_initial: int = 5,
    n_ground_truth: int = 20000,
    acquisition: str = 'ei',
    seed: int = 42,
    interval: int = 500,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Create animation of bi-level BO solving the Rastrigin problem.

    Top subplot shows:
    - Ground truth curve (xm vs J*(xm)) in gray
    - Sampled points as scatter
    - GP posterior mean and 95% confidence interval

    Bottom subplot shows:
    - Acquisition function over xm domain
    - Vertical line at next selected xm point

    Args:
        n_iterations: Number of BO iterations to animate
        n_initial: Number of initial random samples
        n_ground_truth: Number of points for ground truth curve
        acquisition: Acquisition function ('ei', 'pi', 'lcb', 'mwb2', 'thompson')
        seed: Random seed
        interval: Milliseconds between frames
        save_path: Optional path to save animation (e.g., 'animation.gif' or 'animation.mp4')
        show: Whether to display the animation

    Returns:
        matplotlib Animation object
    """
    from matplotlib.animation import FuncAnimation
    from functions import create_rastrigin
    from solvers import create_gp, get_acquisition_function

    print("=" * 60)
    print("Animating Bi-level BO on Rastrigin Problem")
    print("=" * 60)

    problem = create_rastrigin()

    # Generate ground truth
    print("\nStep 1: Generating ground truth...")
    x_bb_truth, J_truth = generate_rastrigin_ground_truth(n_ground_truth, n_inner_starts=10)

    # Run BO with history
    print("\nStep 2: Running bi-level BO...")
    history = run_bilevel_bo_with_history(
        n_iterations=n_iterations,
        n_initial=n_initial,
        n_inner_starts=10,
        acquisition=acquisition,
        seed=seed
    )

    # Setup figure
    print("\nStep 3: Creating animation...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[2, 1])
    fig.suptitle(f'Bi-level BO on Rastrigin (Acquisition: {acquisition.upper()})',
                 fontsize=14, fontweight='bold')

    x_bb_grid = history['x_bb_grid']
    n_frames = len(history['X_bb_history'])

    # Initialize plot elements
    # Top subplot
    ax1.plot(x_bb_truth, J_truth, 'k-', alpha=0.3, linewidth=1, label='Ground Truth J*(xm)')
    ax1.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Optimal J*=0')
    ax1.axvline(x=0, color='green', linestyle='--', alpha=0.5, label='Optimal xm*=0')

    gp_mean_line, = ax1.plot([], [], 'b-', linewidth=2, label='GP Mean')
    gp_fill = ax1.fill_between([], [], [], alpha=0.2, color='blue', label='GP 95% CI')
    sample_scatter = ax1.scatter([], [], c='red', s=80, zorder=5,
                                  edgecolors='black', label='Samples')
    next_point_scatter = ax1.scatter([], [], c='lime', s=150, marker='*',
                                      zorder=6, edgecolors='black', linewidths=1.5)

    ax1.set_xlim(problem.x_bb_lower[0], problem.x_bb_upper[0])
    ax1.set_ylim(-5, 80)
    ax1.set_xlabel('xm (Material Variable)')
    ax1.set_ylabel('J*(xm) = min J(xp, fm(xm))')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    title1 = ax1.set_title('')

    # Bottom subplot
    acq_line, = ax2.plot([], [], 'purple', linewidth=2)
    next_x_bb_line = ax2.axvline(x=0, color='lime', linestyle='-', linewidth=2, alpha=0.8)

    ax2.set_xlim(problem.x_bb_lower[0], problem.x_bb_upper[0])
    ax2.set_xlabel('xm (Material Variable)')
    ax2.set_ylabel(f'{acquisition.upper()} Acquisition')
    ax2.grid(True, alpha=0.3)

    title2 = ax2.set_title('')

    plt.tight_layout()

    def init():
        gp_mean_line.set_data([], [])
        acq_line.set_data([], [])
        sample_scatter.set_offsets(np.empty((0, 2)))
        next_point_scatter.set_offsets(np.empty((0, 2)))
        return gp_mean_line, acq_line, sample_scatter, next_point_scatter

    def update(frame):
        # Get data for this frame
        Xm = history['X_bb_history'][frame]
        Y = history['Y_history'][frame]
        gp_data = history['gp_history'][frame]
        acq_data = history['acq_history'][frame]
        next_x_bb = history['next_x_bb_history'][frame]

        n_initial = history['n_initial']

        # Update sample scatter
        sample_scatter.set_offsets(np.column_stack([Xm.flatten(), Y]))

        # Update titles
        if frame < n_initial:
            title1.set_text(f'Initial Sampling: {frame + 1}/{n_initial} points')
            title2.set_text('Acquisition function (computed after initial sampling)')
        else:
            iter_num = frame - n_initial + 1
            best_J = np.min(Y)
            title1.set_text(f'Iteration {iter_num}/{n_iterations} | Best J = {best_J:.4f} | Samples: {len(Y)}')
            title2.set_text(f'Next xm = {next_x_bb[0]:.3f}')

        # Clear and update GP fill
        for coll in ax1.collections:
            if coll != sample_scatter and coll != next_point_scatter:
                try:
                    coll.remove()
                except (ValueError, AttributeError):
                    pass

        # Update GP predictions
        if gp_data is not None:
            mu = gp_data['mu']
            sigma = gp_data['sigma']

            gp_mean_line.set_data(x_bb_grid, mu)
            ax1.fill_between(x_bb_grid, mu - 1.96*sigma, mu + 1.96*sigma,
                            alpha=0.2, color='blue')

            # Update next point marker
            next_J_pred = mu[np.argmin(np.abs(x_bb_grid - next_x_bb[0]))]
            next_point_scatter.set_offsets([[next_x_bb[0], next_J_pred]])
        else:
            gp_mean_line.set_data([], [])
            next_point_scatter.set_offsets(np.empty((0, 2)))

        # Update acquisition function
        if acq_data is not None:
            acq_line.set_data(x_bb_grid, acq_data)
            ax2.set_ylim(np.min(acq_data) - 0.1 * np.ptp(acq_data),
                        np.max(acq_data) + 0.1 * np.ptp(acq_data))
            next_x_bb_line.set_xdata([next_x_bb[0], next_x_bb[0]])
        else:
            acq_line.set_data([], [])

        # Re-add ground truth (since fill_between clears it)
        # This is handled by not removing the original line plot

        return gp_mean_line, acq_line, sample_scatter, next_point_scatter

    anim = FuncAnimation(
        fig, update, frames=n_frames,
        init_func=init, blit=False, interval=interval, repeat=True
    )

    if save_path:
        print(f"\nSaving animation to {save_path}...")
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=1000//interval)
        elif save_path.endswith('.mp4'):
            anim.save(save_path, writer='ffmpeg', fps=1000//interval)
        else:
            anim.save(save_path, fps=1000//interval)
        print("Animation saved.")

    if show:
        plt.show()

    return anim


if __name__ == "__main__":
    # Demo the animation
    import sys

    # Use non-interactive backend for saving
    import matplotlib
    matplotlib.use('Agg')

    print("Running animation demo...")
    save_file = "plots/bilevel_bo_animation.gif"

    anim = animate_bilevel_bo_rastrigin(
        n_iterations=25,
        n_initial=5,
        n_ground_truth=20000,
        acquisition='ei',
        seed=42,
        interval=800,
        save_path=save_file,
        show=False
    )

    print(f"\nAnimation saved to: {save_file}")
