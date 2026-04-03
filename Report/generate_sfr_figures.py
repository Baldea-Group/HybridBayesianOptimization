"""Generate SFR-1 and SFR-2 figures for the paper.

Produces:
  figs/sfr1_search.pdf   – side-by-side contour + scatter for SFR-1
  figs/sfr2_search.pdf   – side-by-side contour + scatter for SFR-2
  figs/sfr_convergence.pdf – convergence curves for both SFR problems
"""
import sys, pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

REPO_ROOT    = Path(__file__).resolve().parent.parent
RESULTS_DIR  = REPO_ROOT / "results"
FIG_DIR      = Path(__file__).resolve().parent / "figs"
sys.path.insert(0, str(REPO_ROOT))

from functions import create_small_feasible_region, create_small_feasible_region2
from solvers import solve_blackbox_bo, solve_bilevel_bo

# ── Matplotlib defaults for publication ───────────────────────────
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'mathtext.fontset': 'dejavusans',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

C_BBBO = '#0C5DA5'   # blue
C_BIBO = '#FF2C00'   # red

N_INIT = 50
N_ITER = 200
SEED   = 0

# =====================================================================
# Helper: draw background (objective contours + feasible region)
# =====================================================================
def draw_background(fig, ax, XW, XB, J_grid, G_grid, xwb_star, xbb_star,
                    title, xlabel=r'$x^{\mathrm{WB}}$',
                    ylabel=r'$x^{\mathrm{BB}}$'):
    levels_J = np.linspace(J_grid.min(), J_grid.max(), 30)
    cf = ax.contourf(XW, XB, J_grid, levels=levels_J, cmap='viridis_r')
    ax.contour(XW, XB, J_grid, levels=levels_J, colors='black',
               linewidths=0.3, alpha=0.3)
    cbar = fig.colorbar(cf, ax=ax, shrink=0.75, pad=0.03)
    cbar.set_label(r'$J(x^{\mathrm{WB}}, y)$', fontsize=10)

    # Gray-out infeasible region
    ax.contourf(XW, XB, G_grid, levels=[0, 1e3], colors=['white'], alpha=0.7)
    # Constraint boundary
    ax.contour(XW, XB, G_grid, levels=[0], colors=['black'], linewidths=1.5)
    # Hatching for feasible region
    ax.contourf(XW, XB, G_grid, levels=[-1e3, 0], colors=['none'],
                hatches=['oo'], alpha=0)

    # Optimum star
    ax.plot(xwb_star, xbb_star, marker='*', color='gold',
            markersize=14, markeredgecolor='black', markeredgewidth=0.8,
            zorder=10, label='Optimum')

    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect('equal')


# =====================================================================
# SFR-1: run solvers, build grids, plot
# =====================================================================
print("Running SFR-1 solvers...")
prob1 = create_small_feasible_region()

res_bb1 = solve_blackbox_bo(prob1, n_iterations=N_ITER, n_initial=N_INIT,
                            seed=SEED, verbose=False, return_history=True, xi=0.01)
res_bi1 = solve_bilevel_bo(prob1, n_iterations=N_ITER, n_initial=N_INIT,
                           seed=SEED, verbose=False, return_history=True, xi=0.01)

bb1_xwb = res_bb1['X_history'][:, 0]
bb1_xbb = res_bb1['X_history'][:, 1]
bi1_xbb = np.asarray(res_bi1['X_bb_history']).ravel()
bi1_xwb = np.asarray([xw[0] for xw in res_bi1['X_wb_history']])

grid = 300
xw = np.linspace(0, 6, grid)
xb = np.linspace(0, 6, grid)
XW1, XB1 = np.meshgrid(xw, xb)
J1_grid = np.sin(XW1) + XB1
G1_grid = np.sin(XW1) * np.sin(XB1) + 0.95
xwb1_star = prob1.x_wb_optimal[0]
xbb1_star = prob1.x_bb_optimal[0]

fig1, (ax_bb1, ax_bi1) = plt.subplots(1, 2, figsize=(7.0, 3.3),
                                       constrained_layout=True)
draw_background(fig1, ax_bb1, XW1, XB1, J1_grid, G1_grid,
                xwb1_star, xbb1_star, 'Black-box BO')
draw_background(fig1, ax_bi1, XW1, XB1, J1_grid, G1_grid,
                xwb1_star, xbb1_star, 'Bilevel BO')

# Initial samples (open circles)
ax_bb1.scatter(bb1_xwb[:N_INIT], bb1_xbb[:N_INIT], s=25, facecolors='none',
               edgecolors=C_BBBO, linewidths=0.8, zorder=5, label='Initial')
ax_bi1.scatter(bi1_xwb[:N_INIT], bi1_xbb[:N_INIT], s=25, facecolors='none',
               edgecolors=C_BIBO, linewidths=0.8, zorder=5, label='Initial')
# BO queries (filled circles)
ax_bb1.scatter(bb1_xwb[N_INIT:], bb1_xbb[N_INIT:], s=25, c=C_BBBO,
               edgecolors='white', linewidths=0.3, zorder=6, label='BO query')
ax_bi1.scatter(bi1_xwb[N_INIT:], bi1_xbb[N_INIT:], s=25, c=C_BIBO,
               edgecolors='white', linewidths=0.3, zorder=6, label='BO query')

for _ax in (ax_bb1, ax_bi1):
    leg = _ax.legend(loc='upper right', fontsize=7, framealpha=1.0,
                     edgecolor='0.7', fancybox=False)
    leg.set_zorder(20)

fig1.savefig(FIG_DIR / 'sfr1_search.eps')
fig1.savefig(FIG_DIR / 'sfr1_search.pdf')
print(f"  Saved sfr1_search.eps/.pdf")
plt.close(fig1)

# =====================================================================
# SFR-2: run solvers, build grids, plot
# =====================================================================
print("Running SFR-2 solvers...")
prob2 = create_small_feasible_region2()

res_bb2 = solve_blackbox_bo(prob2, n_iterations=N_ITER, n_initial=N_INIT,
                            seed=SEED, verbose=False, return_history=True, xi=0.01)
res_bi2 = solve_bilevel_bo(prob2, n_iterations=N_ITER, n_initial=N_INIT,
                           seed=SEED, verbose=False, return_history=True, xi=0.01)

bb2_xwb = res_bb2['X_history'][:, 0]
bb2_xbb = res_bb2['X_history'][:, 1]
bi2_xbb = np.asarray(res_bi2['X_bb_history']).ravel()
bi2_xwb = np.asarray([xw[0] for xw in res_bi2['X_wb_history']])

XW2, XB2 = np.meshgrid(xw, xb)
Y2_grid = (XB2 - 5.5) / 2 * np.exp(XB2 / 2)
J2_grid = np.cos(2 * XW2) * np.cos(Y2_grid) + np.sin(XW2)
G2_grid = np.cos(XW2) * np.cos(Y2_grid) - np.sin(XW2) * np.sin(Y2_grid) - 0.5
xwb2_star = prob2.x_wb_optimal[0]
xbb2_star = prob2.x_bb_optimal[0]

fig2, (ax_bb2, ax_bi2) = plt.subplots(1, 2, figsize=(7.0, 3.3),
                                       constrained_layout=True)
draw_background(fig2, ax_bb2, XW2, XB2, J2_grid, G2_grid,
                xwb2_star, xbb2_star, 'Black-box BO')
draw_background(fig2, ax_bi2, XW2, XB2, J2_grid, G2_grid,
                xwb2_star, xbb2_star, 'Bilevel BO')

ax_bb2.scatter(bb2_xwb[:N_INIT], bb2_xbb[:N_INIT], s=25, facecolors='none',
               edgecolors=C_BBBO, linewidths=0.8, zorder=5, label='Initial')
ax_bi2.scatter(bi2_xwb[:N_INIT], bi2_xbb[:N_INIT], s=25, facecolors='none',
               edgecolors=C_BIBO, linewidths=0.8, zorder=5, label='Initial')
ax_bb2.scatter(bb2_xwb[N_INIT:], bb2_xbb[N_INIT:], s=25, c=C_BBBO,
               edgecolors='white', linewidths=0.3, zorder=6, label='BO query')
ax_bi2.scatter(bi2_xwb[N_INIT:], bi2_xbb[N_INIT:], s=25, c=C_BIBO,
               edgecolors='white', linewidths=0.3, zorder=6, label='BO query')

for _ax in (ax_bb2, ax_bi2):
    leg = _ax.legend(loc='lower right', fontsize=7, framealpha=1.0,
                     edgecolor='0.7', fancybox=False)
    leg.set_zorder(20)

fig2.savefig(FIG_DIR / 'sfr2_search.eps')
fig2.savefig(FIG_DIR / 'sfr2_search.pdf')
print(f"  Saved sfr2_search.eps/.pdf")
plt.close(fig2)

# =====================================================================
# Convergence curves for SFR-1 and SFR-2 (from cached results)
# =====================================================================
print("Generating SFR convergence curves...")
RESULTS_FILE = RESULTS_DIR / 'results_all_ei_ninit50_xibest.pkl'
with open(RESULTS_FILE, 'rb') as f:
    data = pickle.load(f)


def _lighten(color, amount=0.65):
    """Blend *color* toward white by *amount* (0 = unchanged, 1 = white)."""
    import matplotlib.colors as mc
    r, g, b, _ = mc.to_rgba(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def plot_regret_curve(ax, regret_list, color, label):
    if not regret_list:
        return
    R = np.stack(regret_list)
    R = np.where(np.isfinite(R), R, np.nan)
    R = np.clip(R, 1e-12, None)
    logR = np.log10(R)
    with np.errstate(all='ignore'):
        mean = np.nanmean(logR, axis=0)
        std  = np.nanstd(logR, axis=0)
    valid_mask = ~np.isnan(mean)
    if not np.any(valid_mask):
        return
    iters = np.arange(len(mean))
    light = _lighten(color)
    ax.fill_between(iters, mean - std, mean + std,
                    where=valid_mask, color=light, edgecolor='none')
    ax.plot(iters[valid_mask], mean[valid_mask], color=color, label=label)


fig3, (ax_s1, ax_s2) = plt.subplots(1, 2, figsize=(7.0, 2.8),
                                     constrained_layout=True)

for ax, pname in [(ax_s1, 'Small-Feasible-Region'),
                   (ax_s2, 'Small-Feasible-Region-2')]:
    prob = data['results'][pname]
    plot_regret_curve(ax, prob['blackbox_bo_ei']['regrets'], C_BBBO, 'Black-box BO')
    plot_regret_curve(ax, prob['bilevel_bo_ei']['regrets'], C_BIBO, 'Bilevel BO')
    ylims = ax.get_ylim()
    ax.vlines(50, ylims[0], ylims[1], color='black', linestyle='dashed',
              linewidth=0.8, label=r'$n_{\mathrm{init}}$')
    ax.set_ylim(ylims)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'$\log_{10}$ Regret')
    short = pname.replace('Small-Feasible-Region', 'SFR')
    ax.set_title(short if short != 'SFR' else 'SFR-1')
    ax.legend(fontsize=7, loc='upper right')

fig3.savefig(FIG_DIR / 'sfr_convergence.eps')
fig3.savefig(FIG_DIR / 'sfr_convergence.pdf')
print(f"  Saved sfr_convergence.eps/.pdf")
plt.close(fig3)

print("Done.")
