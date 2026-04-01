"""
analysis_discussion.py - Additional analyses for the Discussion section

Implements three analyses from the OUTLINE.md "Analysis for Discussion" section:
1. Feasibility rate comparison (BB-BO vs Bi-BO)
2. Dimensionality vs. constraint tightness scatter
3. Sample efficiency crossover

All analyses use cached results from run_comparison.py.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional

from functions import get_all_problems, BiLevelProblem

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

# Use the best-xi, n_init=50 results (matches Table 2 in the paper)
RESULTS_FILE = RESULTS_DIR / "results_all_ei_ninit50_xibest.pkl"

# Publication-quality settings
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
})

# Colors consistent with existing paper figures
COLOR_BB = "#0173B2"   # blue (black-box BO)
COLOR_BI = "#D55E00"   # vermillion (bilevel BO)


def load_results(path: Path = RESULTS_FILE) -> Dict[str, Any]:
    """Load cached experiment results."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["results"], data["settings"]


# =========================================================================== #
# Analysis 1: Feasibility Rate Comparison
# =========================================================================== #

def compute_feasibility_rates(
    results: Dict[str, Any],
    problems: Dict[str, BiLevelProblem],
) -> Dict[str, Dict[str, Any]]:
    """
    For each constrained problem, compute per-evaluation feasibility for BB-BO
    and Bi-BO by re-evaluating constraint functions on stored X_history.

    Returns dict keyed by problem name with:
        bb_rate: float (overall feasibility fraction for BB-BO)
        bi_rate: float (overall feasibility fraction for Bi-BO)
        bb_rates_per_rep: np.ndarray shape (n_reps,)
        bi_rates_per_rep: np.ndarray shape (n_reps,)
        bb_cumulative: np.ndarray shape (n_evals,) - cumulative feasibility over iterations
        bi_cumulative: np.ndarray shape (n_evals,) - cumulative feasibility over iterations
    """
    feas = {}

    for prob_name, r in results.items():
        p = problems.get(prob_name)
        if p is None or p.n_g == 0 or p.g is None:
            continue

        bb = r["blackbox_bo_ei"]
        bi = r["bilevel_bo_ei"]
        n_reps = len(bb["X_history"])

        bb_feas_per_rep = []
        bi_feas_per_rep = []
        bb_feas_all = []  # shape will be (n_reps, n_evals)
        bi_feas_all = []

        for rep in range(n_reps):
            # --- BB-BO: reconstruct constraints from full X_history ---
            X_full = np.array(bb["X_history"][rep])  # (n_evals, n_x_wb + n_x_bb)
            n_evals = X_full.shape[0]
            bb_feasible = np.zeros(n_evals, dtype=bool)
            for i in range(n_evals):
                x_wb, x_bb = p.split_x(X_full[i])
                y = p.fbb(x_bb)
                g_val = p.g(x_wb, y)
                bb_feasible[i] = np.all(g_val <= 1e-6)
            bb_feas_per_rep.append(np.mean(bb_feasible))
            bb_feas_all.append(bb_feasible)

            # --- Bi-BO: reconstruct from X_bb_history + X_wb_history ---
            X_bb = np.array(bi["X_bb_history"][rep])  # (n_evals, n_x_bb)
            X_wb_list = bi["X_wb_history"][rep]        # list of arrays
            n_evals_bi = X_bb.shape[0]
            bi_feasible = np.zeros(n_evals_bi, dtype=bool)
            for i in range(n_evals_bi):
                x_bb = X_bb[i]
                x_wb = np.array(X_wb_list[i])
                y = p.fbb(x_bb)
                g_val = p.g(x_wb, y)
                bi_feasible[i] = np.all(g_val <= 1e-6)
            bi_feas_per_rep.append(np.mean(bi_feasible))
            bi_feas_all.append(bi_feasible)

        bb_feas_all = np.array(bb_feas_all)  # (n_reps, n_evals)
        bi_feas_all = np.array(bi_feas_all)

        # Cumulative feasibility rate (mean over reps at each iteration)
        bb_cumulative = np.cumsum(bb_feas_all, axis=1) / np.arange(1, bb_feas_all.shape[1] + 1)
        bi_cumulative = np.cumsum(bi_feas_all, axis=1) / np.arange(1, bi_feas_all.shape[1] + 1)

        feas[prob_name] = {
            "bb_rate": np.mean(bb_feas_per_rep),
            "bi_rate": np.mean(bi_feas_per_rep),
            "bb_rates_per_rep": np.array(bb_feas_per_rep),
            "bi_rates_per_rep": np.array(bi_feas_per_rep),
            "bb_cumulative": bb_cumulative,  # (n_reps, n_evals)
            "bi_cumulative": bi_cumulative,
            "n_g": p.n_g,
        }

    return feas


def plot_feasibility_rates(feas: Dict[str, Dict], save_dir: Path = PLOTS_DIR):
    """
    Plot 1a: Bar chart of overall feasibility rates.
    Plot 1b: Cumulative feasibility over iterations for selected problems.
    """
    # --- Bar chart ---
    prob_names = sorted(feas.keys(), key=lambda n: feas[n]["bb_rate"])
    n = len(prob_names)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(n)
    w = 0.35

    bb_rates = [feas[p]["bb_rate"] * 100 for p in prob_names]
    bi_rates = [feas[p]["bi_rate"] * 100 for p in prob_names]

    bars_bb = ax.bar(x - w / 2, bb_rates, w, label="BB-BO", color=COLOR_BB, edgecolor="white", linewidth=0.5)
    bars_bi = ax.bar(x + w / 2, bi_rates, w, label="Bi-BO", color=COLOR_BI, edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Feasible evaluations (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(prob_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 105)
    ax.axhline(100, color="gray", ls="--", lw=0.5)
    ax.legend(loc="upper left")
    ax.set_title("Fraction of evaluations landing in feasible regions")

    fig.tight_layout()
    fig.savefig(save_dir / "feasibility_rates.eps", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved feasibility_rates.eps")

    # --- Cumulative feasibility over iterations (selected problems) ---
    # Pick problems with lowest BB-BO feasibility rates
    worst_bb = prob_names[:min(6, n)]

    n_panels = len(worst_bb)
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for idx, pname in enumerate(worst_bb):
        ax = axes[idx // ncols, idx % ncols]
        f = feas[pname]

        # Mean and std over reps
        bb_mean = f["bb_cumulative"].mean(axis=0) * 100
        bb_std = f["bb_cumulative"].std(axis=0) * 100
        bi_mean = f["bi_cumulative"].mean(axis=0) * 100
        bi_std = f["bi_cumulative"].std(axis=0) * 100
        iters = np.arange(1, len(bb_mean) + 1)

        ax.plot(iters, bb_mean, color=COLOR_BB, label="BB-BO", lw=1.2)
        ax.fill_between(iters, bb_mean - bb_std, np.minimum(bb_mean + bb_std, 100),
                         color=COLOR_BB, alpha=0.15)
        ax.plot(iters, bi_mean, color=COLOR_BI, label="Bi-BO", lw=1.2)
        ax.fill_between(iters, np.maximum(bi_mean - bi_std, 0), np.minimum(bi_mean + bi_std, 100),
                         color=COLOR_BI, alpha=0.15)

        ax.set_ylim(0, 105)
        ax.set_xlabel("Evaluation")
        ax.set_ylabel("Cumulative feasibility (%)")
        ax.set_title(pname, fontsize=9)
        if idx == 0:
            ax.legend(fontsize=8)

    # Hide empty subplots
    for idx in range(n_panels, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_dir / "feasibility_cumulative.eps", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved feasibility_cumulative.eps")


def print_feasibility_table(feas: Dict[str, Dict]):
    """Print LaTeX-ready feasibility rate table."""
    print("\n  Feasibility Rate Summary")
    print("  " + "-" * 65)
    print(f"  {'Problem':<25s} {'n_g':>3s} {'BB-BO %':>8s} {'Bi-BO %':>8s} {'Wasted':>8s}")
    print("  " + "-" * 65)
    for pname in sorted(feas.keys()):
        f = feas[pname]
        wasted = (1 - f["bb_rate"]) * 100
        print(f"  {pname:<25s} {f['n_g']:>3d} {f['bb_rate']*100:>7.1f}% {f['bi_rate']*100:>7.1f}% {wasted:>7.1f}%")
    print("  " + "-" * 65)


# =========================================================================== #
# Analysis 3: Dimensionality vs. Constraint Tightness Scatter
# =========================================================================== #

def estimate_feasible_fraction(
    problems: Dict[str, BiLevelProblem],
    n_samples: int = 100_000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Estimate the fraction of the full decision space [x^WB, x^BB] that is
    feasible by uniform random sampling within bounds.
    """
    rng = np.random.default_rng(seed)
    fracs = {}

    for name, p in problems.items():
        if p.n_g == 0 or p.g is None:
            fracs[name] = 1.0
            continue

        lower, upper = p.get_full_bounds()
        n_total = p.n_x_wb + p.n_x_bb
        samples = rng.uniform(lower, upper, size=(n_samples, n_total))

        feasible_count = 0
        for i in range(n_samples):
            x_wb, x_bb = p.split_x(samples[i])
            y = p.fbb(x_bb)
            g_val = p.g(x_wb, y)
            if np.all(g_val <= 1e-6):
                feasible_count += 1

        fracs[name] = feasible_count / n_samples

    return fracs


def plot_dim_vs_constraint_scatter(
    results: Dict[str, Any],
    problems: Dict[str, BiLevelProblem],
    feasible_fracs: Dict[str, float],
    save_dir: Path = PLOTS_DIR,
):
    """
    Scatter plot: x = total dimension, y = feasible fraction of domain,
    color/size = log10(BB/Bi regret ratio).
    """
    # Gather data
    names = []
    dims = []
    feas_fracs = []
    ratios = []

    for prob_name, r in results.items():
        p = problems.get(prob_name)
        if p is None:
            continue

        bb_regrets = r["blackbox_bo_ei"]["regrets"]
        bi_regrets = r["bilevel_bo_ei"]["regrets"]

        # Final regret: mean over reps
        bb_final = np.mean([reg[-1] for reg in bb_regrets])
        bi_final = np.mean([reg[-1] for reg in bi_regrets])

        # Avoid division by zero
        if bi_final < 1e-10:
            ratio = bb_final / 1e-10
        else:
            ratio = bb_final / bi_final

        total_dim = p.n_x_wb + p.n_x_bb
        ff = feasible_fracs.get(prob_name, 1.0)

        names.append(prob_name)
        dims.append(total_dim)
        feas_fracs.append(ff)
        ratios.append(ratio)

    dims = np.array(dims)
    feas_fracs = np.array(feas_fracs)
    ratios = np.array(ratios)
    log_ratios = np.log10(ratios)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6, 4.5), layout="constrained")

    scatter = ax.scatter(
        dims, feas_fracs, c=log_ratios, s=80,
        cmap="YlOrRd", edgecolors="black", linewidths=0.5,
        vmin=1, vmax=6, zorder=5,
    )

    # Annotate each point with manual offsets to avoid overlap
    SHORT_NAMES = {
        "Small-Feasible-Region": "SFR",
        "Small-Feasible-Region-2": "SFR-2",
        "Heat-Exchanger": "Heat-Exch.",
        "Batch-Reactor": "Batch-React.",
        "Williams-Otto": "Williams-Otto",
        "Toy-Hydrology": "Toy-Hydrol.",
    }
    # Manual offsets for the crowded dim=5 cluster (and others as needed)
    LABEL_OFFSETS = {
        "Heat-Exchanger":  (8, 10),
        "PSA":             (-45, 12),
        "Membrane":        (8, -15),
        "CSTR":            (-35, -15),
        "Distillation":    (8, -20),
        "Evaporator":      (8, 5),
        "Batch-Reactor":   (8, 5),
        "Williams-Otto":   (8, 5),
        "Rastrigin":       (8, 5),
        "Rosen-Suzuki":    (8, 5),
        "Small-Feasible-Region":   (8, 5),
        "Small-Feasible-Region-2": (-50, 10),
        "Toy-Hydrology":   (8, -15),
    }
    for i, name in enumerate(names):
        short = SHORT_NAMES.get(name, name)
        offset = LABEL_OFFSETS.get(name, (8, 5))
        ax.annotate(short, (dims[i], feas_fracs[i]),
                     textcoords="offset points", xytext=offset,
                     fontsize=7, ha="left", va="bottom")

    cbar = fig.colorbar(scatter, ax=ax, label=r"$\log_{10}$(BB-BO / Bi-BO regret ratio)")
    ax.set_xlabel("Total problem dimension ($n_{\\mathrm{WB}} + n_{\\mathrm{BB}}$)")
    ax.set_ylabel("Feasible fraction of domain")
    ax.set_yscale("log")
    ax.set_ylim(bottom=max(1e-4, min(feas_fracs) / 3))
    ax.set_xticks([2, 3, 4, 5])
    fig.savefig(save_dir / "dim_vs_constraint_scatter.eps", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved dim_vs_constraint_scatter.eps")

    # Print table
    print("\n  Dimensionality vs. Constraint Tightness")
    print("  " + "-" * 75)
    print(f"  {'Problem':<25s} {'dim':>3s} {'n_BB':>4s} {'Feas%':>8s} {'BB/Bi':>12s} {'log10':>6s}")
    print("  " + "-" * 75)
    order = np.argsort(log_ratios)[::-1]
    for i in order:
        print(f"  {names[i]:<25s} {dims[i]:>3d} {problems[names[i]].n_x_bb:>4d} "
              f"{feas_fracs[i]*100:>7.2f}% {ratios[i]:>11.0f}x {log_ratios[i]:>6.1f}")
    print("  " + "-" * 75)


# =========================================================================== #
# Analysis 4: Sample Efficiency Crossover
# =========================================================================== #

def compute_sample_crossover(
    results: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    For each problem, find the iteration at which bilevel BO first achieves
    the final regret of BB-BO (at iteration 200).

    Returns dict keyed by problem name with:
        bb_final_regret: float (mean final regret of BB-BO)
        bi_crossover_iter: float (mean iteration where Bi-BO matches BB-BO's final)
        bi_crossover_iters_per_rep: list of int/None per rep
        bi_final_regret: float
    """
    crossovers = {}

    for prob_name, r in results.items():
        bb = r["blackbox_bo_ei"]
        bi = r["bilevel_bo_ei"]

        bb_final_per_rep = np.array([reg[-1] for reg in bb["regrets"]])
        bb_final_mean = np.mean(bb_final_per_rep)

        bi_crossover_iters = []
        for rep_idx in range(len(bi["regrets"])):
            bi_regret = np.array(bi["regrets"][rep_idx])
            bb_target = bb_final_per_rep[rep_idx]

            # Find first iteration where bi-BO regret <= bb-BO final regret
            matches = np.where(bi_regret <= bb_target)[0]
            if len(matches) > 0:
                bi_crossover_iters.append(matches[0])
            else:
                bi_crossover_iters.append(None)

        # Compute statistics (excluding None)
        valid = [x for x in bi_crossover_iters if x is not None]
        crossover_mean = np.mean(valid) if valid else None
        crossover_frac = len(valid) / len(bi_crossover_iters)

        bi_final_per_rep = np.array([reg[-1] for reg in bi["regrets"]])

        crossovers[prob_name] = {
            "bb_final_regret": bb_final_mean,
            "bi_final_regret": np.mean(bi_final_per_rep),
            "bi_crossover_iter": crossover_mean,
            "bi_crossover_iters_per_rep": bi_crossover_iters,
            "crossover_fraction": crossover_frac,
            "n_total_evals": len(bi["regrets"][0]),
        }

    return crossovers


def plot_sample_crossover(crossovers: Dict[str, Dict], save_dir: Path = PLOTS_DIR):
    """Bar chart showing how many evaluations bilevel BO needs to match BB-BO's final performance."""
    # Filter to problems where crossover happens
    valid = {k: v for k, v in crossovers.items()
             if v["bi_crossover_iter"] is not None}

    if not valid:
        print("  No valid crossover data to plot.")
        return

    # Sort by crossover iteration
    sorted_names = sorted(valid.keys(), key=lambda n: valid[n]["bi_crossover_iter"])

    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(sorted_names))

    crossover_iters = [valid[n]["bi_crossover_iter"] for n in sorted_names]
    total_evals = [valid[n]["n_total_evals"] for n in sorted_names]

    bars = ax.bar(x, crossover_iters, color=COLOR_BI, edgecolor="white", linewidth=0.5)

    # Add total budget line
    ax.axhline(total_evals[0], color="gray", ls="--", lw=0.8, label=f"BB-BO budget ({total_evals[0]} evals)")

    # Add text annotations on bars
    for i, (xi, ci) in enumerate(zip(x, crossover_iters)):
        # Show iteration count and percentage of budget
        pct = ci / total_evals[i] * 100
        ax.text(xi, ci + 3, f"{ci:.0f}\n({pct:.0f}%)", ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("Evaluations for Bi-BO to match BB-BO final regret")
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_names, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.set_title("Sample efficiency: Bi-BO iterations to match BB-BO at 250 evaluations")

    fig.tight_layout()
    fig.savefig(save_dir / "sample_crossover.eps", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved sample_crossover.eps")


def print_crossover_table(crossovers: Dict[str, Dict]):
    """Print table of sample efficiency crossover results."""
    print("\n  Sample Efficiency Crossover")
    print("  " + "-" * 80)
    print(f"  {'Problem':<25s} {'BB final':>10s} {'Bi final':>10s} {'Crossover':>10s} {'%budget':>8s} {'Frac':>6s}")
    print("  " + "-" * 80)
    for pname in sorted(crossovers.keys()):
        c = crossovers[pname]
        ci = c["bi_crossover_iter"]
        ci_str = f"{ci:.0f}" if ci is not None else "never"
        pct = f"{ci / c['n_total_evals'] * 100:.0f}%" if ci is not None else "---"
        frac = f"{c['crossover_fraction']:.0%}"
        print(f"  {pname:<25s} {c['bb_final_regret']:>10.4f} {c['bi_final_regret']:>10.4f} "
              f"{ci_str:>10s} {pct:>8s} {frac:>6s}")
    print("  " + "-" * 80)


# =========================================================================== #
# Main
# =========================================================================== #

def main():
    print("Loading results...")
    results, settings = load_results()
    problems = get_all_problems()
    print(f"  Loaded {len(results)} problems, settings: n_init={settings['n_initial']}, "
          f"n_iter={settings['n_iterations']}, n_reps={settings['n_repetitions']}")

    # --- Analysis 1: Feasibility rates ---
    print("\n" + "=" * 70)
    print("Analysis 1: Feasibility Rate Comparison")
    print("=" * 70)
    feas = compute_feasibility_rates(results, problems)
    print_feasibility_table(feas)
    plot_feasibility_rates(feas)

    # --- Analysis 3: Dimensionality vs. constraint tightness ---
    print("\n" + "=" * 70)
    print("Analysis 3: Dimensionality vs. Constraint Tightness")
    print("=" * 70)
    print("  Estimating feasible fractions (100k random samples per problem)...")
    feasible_fracs = estimate_feasible_fraction(problems, n_samples=100_000)
    plot_dim_vs_constraint_scatter(results, problems, feasible_fracs)

    # --- Analysis 4: Sample efficiency crossover ---
    print("\n" + "=" * 70)
    print("Analysis 4: Sample Efficiency Crossover")
    print("=" * 70)
    crossovers = compute_sample_crossover(results)
    print_crossover_table(crossovers)
    plot_sample_crossover(crossovers)

    print("\nAll analyses complete. Plots saved to:", PLOTS_DIR.resolve())


if __name__ == "__main__":
    main()
