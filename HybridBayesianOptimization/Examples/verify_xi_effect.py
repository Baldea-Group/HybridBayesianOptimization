"""
verify_xi_effect.py - Verify that xi parameter affects exploration/exploitation tradeoff

This script demonstrates:
1. How EI acquisition values change with different xi values
2. Where the acquisition function selects points (exploitation vs exploration regions)
3. The effect on optimization convergence
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from solvers import expected_improvement, probability_of_improvement


def demo_xi_effect_on_acquisition():
    """
    Demonstrate how xi affects EI acquisition values visually.
    """
    print("=" * 70)
    print("VERIFICATION: xi Effect on Expected Improvement")
    print("=" * 70)

    # Create a simple 1D example
    np.random.seed(42)

    # Sample points - sparse to show GP uncertainty
    X_train = np.array([[0.1], [0.3], [0.7], [0.9]])
    # Function with a minimum around 0.5
    y_train = np.array([0.8, 0.3, 0.2, 0.6])

    # Fit GP
    kernel = ConstantKernel(1.0) * RBF(length_scale=0.2) + WhiteKernel(noise_level=1e-5)
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=5)
    gpr.fit(X_train, y_train)

    # Test points
    X_test = np.linspace(0, 1, 200).reshape(-1, 1)
    mu, sigma = gpr.predict(X_test, return_std=True)

    y_best = np.min(y_train)

    # Different xi values to test
    xi_values = [0.0, 0.01, 0.1, 0.5, 1.0]

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: GP mean, uncertainty, and training points
    ax1 = axes[0]
    ax1.fill_between(X_test.flatten(), mu - 2*sigma, mu + 2*sigma, alpha=0.2, color='blue', label='95% CI')
    ax1.plot(X_test, mu, 'b-', label='GP mean')
    ax1.scatter(X_train, y_train, c='red', s=100, zorder=5, label='Training points')
    ax1.axhline(y=y_best, color='green', linestyle='--', label=f'y_best = {y_best:.2f}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y (objective)')
    ax1.set_title('Gaussian Process Model (minimize)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: EI for different xi values
    ax2 = axes[1]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(xi_values)))

    max_locations = []
    for xi, color in zip(xi_values, colors):
        ei = expected_improvement(X_test, gpr, y_best, xi=xi, maximize=False)
        ax2.plot(X_test, ei, color=color, label=f'xi={xi}', linewidth=2)

        # Find max location
        max_idx = np.argmax(ei)
        max_loc = X_test[max_idx, 0]
        max_locations.append((xi, max_loc, ei[max_idx]))
        ax2.axvline(x=max_loc, color=color, linestyle=':', alpha=0.5)

    ax2.set_xlabel('x')
    ax2.set_ylabel('Expected Improvement')
    ax2.set_title('EI Acquisition Function for Different xi Values')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('xi_effect_visual.png', dpi=150)
    print("\nSaved: xi_effect_visual.png")
    plt.close()

    # Print analysis
    print("\n" + "-" * 70)
    print("Analysis: Where does EI select the next point?")
    print("-" * 70)
    print(f"{'xi':<10} {'Max EI Location':<20} {'Max EI Value':<15} {'Interpretation'}")
    print("-" * 70)

    for xi, loc, val in max_locations:
        # Determine if this is exploitation (near known minimum) or exploration
        dist_to_min = abs(loc - 0.7)  # 0.7 is near the observed minimum
        dist_to_unexplored = min(abs(loc - 0.5), abs(loc - 0.0), abs(loc - 1.0))  # unexplored regions

        if dist_to_min < 0.15:
            interp = "Exploitation (near minimum)"
        elif dist_to_unexplored < 0.1:
            interp = "Exploration (high uncertainty)"
        else:
            interp = "Balanced"

        print(f"{xi:<10} {loc:<20.3f} {val:<15.6f} {interp}")


def demo_xi_effect_on_convergence():
    """
    Demonstrate how xi affects optimization convergence on a simple problem.
    """
    print("\n" + "=" * 70)
    print("VERIFICATION: xi Effect on Convergence (1D Rastrigin)")
    print("=" * 70)

    # 1D Rastrigin-like function
    def rastrigin_1d(x):
        return x**2 - 10 * np.cos(2 * np.pi * x) + 10

    x_optimal = 0.0
    y_optimal = 0.0

    xi_values = [0.001, 0.01, 0.1, 0.5, 1.0]
    n_iterations = 30
    n_initial = 3
    n_reps = 20

    results = {xi: {'regrets': [], 'sampled_x': []} for xi in xi_values}

    for xi in xi_values:
        for rep in range(n_reps):
            np.random.seed(42 + rep)

            # Initialize
            X_sample = np.random.uniform(-2, 2, n_initial).reshape(-1, 1)
            y_sample = np.array([rastrigin_1d(x[0]) for x in X_sample])

            kernel = ConstantKernel(1.0) * RBF(length_scale=0.5) + WhiteKernel(noise_level=1e-5)
            gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=3)

            best_so_far = [np.min(y_sample)]
            sampled_x = list(X_sample.flatten())

            for i in range(n_iterations):
                gpr.fit(X_sample, y_sample)

                # Grid search for next point
                X_grid = np.linspace(-2, 2, 500).reshape(-1, 1)
                ei = expected_improvement(X_grid, gpr, np.min(y_sample), xi=xi, maximize=False)
                x_next = X_grid[np.argmax(ei)]
                y_next = rastrigin_1d(x_next[0])

                X_sample = np.vstack([X_sample, x_next])
                y_sample = np.append(y_sample, y_next)
                sampled_x.append(x_next[0])

                best_so_far.append(min(best_so_far[-1], y_next))

            regret = np.array(best_so_far) - y_optimal
            results[xi]['regrets'].append(regret)
            results[xi]['sampled_x'].append(sampled_x)

    # Plot convergence
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(xi_values)))

    # Left: Regret curves
    ax1 = axes[0]
    for xi, color in zip(xi_values, colors):
        regrets = np.array(results[xi]['regrets'])
        mean_regret = np.mean(regrets, axis=0)
        std_regret = np.std(regrets, axis=0)
        iterations = np.arange(len(mean_regret))

        ax1.plot(iterations, mean_regret, color=color, label=f'xi={xi}', linewidth=2)
        ax1.fill_between(iterations, mean_regret - std_regret, mean_regret + std_regret,
                        color=color, alpha=0.2)

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Simple Regret')
    ax1.set_title('Convergence for Different xi Values')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Right: Distribution of sampled points
    ax2 = axes[1]
    for idx, (xi, color) in enumerate(zip(xi_values, colors)):
        all_x = []
        for sampled in results[xi]['sampled_x']:
            all_x.extend(sampled[n_initial:])  # exclude initial random samples

        # Histogram
        ax2.hist(all_x, bins=30, alpha=0.5, color=color, label=f'xi={xi}',
                density=True, histtype='stepfilled')

    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Optimum (x=0)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('Density of sampled points')
    ax2.set_title('Where Does BO Sample? (after initial)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('xi_convergence_effect.png', dpi=150)
    print("\nSaved: xi_convergence_effect.png")
    plt.close()

    # Print statistics
    print("\n" + "-" * 70)
    print("Final Regret Statistics (after {} iterations, {} reps)".format(n_iterations, n_reps))
    print("-" * 70)
    print(f"{'xi':<10} {'Mean Final Regret':<20} {'Std':<15} {'Best Rep':<15}")
    print("-" * 70)

    for xi in xi_values:
        final_regrets = [r[-1] for r in results[xi]['regrets']]
        print(f"{xi:<10} {np.mean(final_regrets):<20.6f} {np.std(final_regrets):<15.6f} {np.min(final_regrets):<15.6f}")

    # Calculate spread of sampling
    print("\n" + "-" * 70)
    print("Sampling Spread Statistics")
    print("-" * 70)
    print(f"{'xi':<10} {'Mean |x|':<15} {'Std of x':<15} {'Interpretation'}")
    print("-" * 70)

    for xi in xi_values:
        all_x = []
        for sampled in results[xi]['sampled_x']:
            all_x.extend(sampled[n_initial:])
        all_x = np.array(all_x)
        mean_abs_x = np.mean(np.abs(all_x))
        std_x = np.std(all_x)

        if mean_abs_x < 0.3:
            interp = "Focused near optimum"
        elif mean_abs_x < 0.6:
            interp = "Moderate exploration"
        else:
            interp = "Wide exploration"

        print(f"{xi:<10} {mean_abs_x:<15.3f} {std_x:<15.3f} {interp}")


def demo_acquisition_shape():
    """
    Show how the acquisition function shape changes with xi.
    """
    print("\n" + "=" * 70)
    print("VERIFICATION: Acquisition Function Shape Analysis")
    print("=" * 70)

    # Create a scenario where exploitation vs exploration choice is clear
    np.random.seed(42)

    # Two observed points: one with good value, one exploring
    X_train = np.array([[0.2], [0.8]])
    y_train = np.array([0.1, 0.5])  # Lower is better, clear minimum at x=0.2

    kernel = ConstantKernel(1.0) * RBF(length_scale=0.3) + WhiteKernel(noise_level=1e-5)
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=5)
    gpr.fit(X_train, y_train)

    X_test = np.linspace(0, 1, 200).reshape(-1, 1)
    mu, sigma = gpr.predict(X_test, return_std=True)
    y_best = np.min(y_train)

    xi_values = [0.0, 0.05, 0.2, 0.5]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, xi in zip(axes.flatten(), xi_values):
        ei = expected_improvement(X_test, gpr, y_best, xi=xi, maximize=False)

        # Normalize for visualization
        ei_norm = ei / (ei.max() + 1e-10)

        ax.fill_between(X_test.flatten(), mu - 2*sigma, mu + 2*sigma, alpha=0.15, color='blue')
        ax.plot(X_test, mu, 'b-', linewidth=1.5, label='GP mean')
        ax.scatter(X_train, y_train, c='red', s=100, zorder=5)
        ax.axhline(y=y_best, color='green', linestyle='--', alpha=0.5)

        # Plot EI on secondary axis
        ax2 = ax.twinx()
        ax2.fill_between(X_test.flatten(), 0, ei_norm, alpha=0.3, color='orange')
        ax2.plot(X_test, ei_norm, 'orange', linewidth=2, label='EI (normalized)')
        ax2.set_ylabel('EI (normalized)', color='orange')

        max_idx = np.argmax(ei)
        ax.axvline(x=X_test[max_idx], color='red', linestyle=':', linewidth=2)

        ax.set_xlabel('x')
        ax.set_ylabel('y', color='blue')
        ax.set_title(f'xi = {xi}\nNext point: x = {X_test[max_idx, 0]:.3f}')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('xi_acquisition_shape.png', dpi=150)
    print("\nSaved: xi_acquisition_shape.png")
    plt.close()

    print("\nKey insight:")
    print("- xi=0: EI is maximized where improvement is likely (near known minimum)")
    print("- Higher xi: Need MORE improvement to overcome the xi threshold")
    print("- This shifts preference toward high-uncertainty regions (exploration)")


if __name__ == "__main__":
    # Run all demonstrations
    demo_xi_effect_on_acquisition()
    demo_xi_effect_on_convergence()
    demo_acquisition_shape()

    print("\n" + "=" * 70)
    print("SUMMARY: xi Parameter Verification")
    print("=" * 70)
    print("""
The xi parameter controls exploration vs exploitation in EI/PI:

  EI = (y_best - mu - xi) * Phi(Z) + sigma * phi(Z)
       ^^^^^^^^^^^^^^^
       This is the "improvement" term

- LOW xi (0.001-0.01): Exploitative
  - Small threshold for improvement
  - Prefers points near the current best
  - Fast convergence when near optimum
  - Risk of getting stuck in local minima

- HIGH xi (0.1-1.0): Explorative
  - Large threshold for improvement
  - Needs big gains to be attractive
  - High-uncertainty regions become relatively more valuable
  - Better global exploration
  - Slower convergence but more robust

Recommendation:
- Start with xi=0.01 (default, slightly exploitative)
- Use xi=0.1-0.2 for multi-modal functions
- Use xi < 0.01 when confident in unimodal structure
""")
