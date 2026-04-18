"""Shared paths and utilities for the multi-scale BO project."""

from pathlib import Path

# Repository root (directory containing this file)
REPO_ROOT = Path(__file__).resolve().parent

# Standard directories
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_PARALLEL_DIR = REPO_ROOT / "results_parallel"
PLOTS_DIR = REPO_ROOT / "plots"
FIGURES_DIR = REPO_ROOT / "figures"
REPORT_DIR = REPO_ROOT / "Report"
REPORT_FIGS_DIR = REPORT_DIR / "figs"
NOTEBOOKS_DIR = REPO_ROOT / "Notebooks"


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_file(problem_set: str, acquisitions: list[str],
                 n_init: int = None, xi: float = None) -> Path:
    """Build a results pickle filename matching the project convention."""
    acq_str = '_'.join(acquisitions)
    name = f"results_{problem_set}_{acq_str}"
    if n_init is not None:
        name += f"_ninit{n_init}"
    if xi is not None:
        name += f"_xi{xi}"
    name += ".pkl"
    return RESULTS_DIR / name


def plots_dir(n_init: int = None, xi: float = None) -> Path:
    """Build a plots directory path, optionally suffixed for sweep parameters."""
    parts = []
    if n_init is not None:
        parts.append(f"ninit{n_init}")
    if xi is not None:
        parts.append(f"xi{xi}")
    if parts:
        return REPO_ROOT / f"plots_{'_'.join(parts)}"
    return PLOTS_DIR