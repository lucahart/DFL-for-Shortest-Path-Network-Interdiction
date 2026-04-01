"""Persistence helpers for SPNI sweep results and derived figures.

This module keeps result-export side effects separate from the core execution
pipeline. Callers that want durable artefacts can pass a typed ``SweepResult``
here and receive deterministic CSV and figure outputs without rescanning the
results directory later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from dflintdpy.simulation.spni.types import SweepResult
from dflintdpy.utils.analyse_results import (
    create_boxplots,
    create_boxplots_by_simulation,
)
from dflintdpy.utils.read_write_results import save_results_to_csv


@dataclass(frozen=True)
class SweepStoragePaths:
    """Filesystem targets produced by ``persist_sweep_outputs(...)``."""

    results_path: Path
    sample_boxplot_path: Path
    simulation_boxplot_path: Path


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read one config value from legacy or typed config objects."""
    getter = getattr(cfg, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(cfg, key, default)


def _project_root() -> Path:
    """Return the repository root for default results/figure directories."""
    return Path(__file__).resolve().parents[4]


def _default_results_filename(cfg: Any, num_seeds: int) -> str:
    """Build the legacy results filename from one config object."""
    m_size, n_size = _cfg_get(cfg, "grid_size", (0, 0))
    return (
        "results_train_"
        f"{_cfg_get(cfg, 'num_train_samples', 0)}"
        "_valid_"
        f"{_cfg_get(cfg, 'num_val_samples', 0)}"
        "_test_"
        f"{_cfg_get(cfg, 'num_test_samples', 0)}"
        "_m_"
        f"{m_size}"
        "_n_"
        f"{n_size}"
        "_deg_"
        f"{_cfg_get(cfg, 'deg', 0)}"
        "_noise_"
        f"{_cfg_get(cfg, 'noise_width', 0)}"
        "_seeds_"
        f"{int(num_seeds)}.csv"
    )


def _resolve_results_path(
    cfg: Any,
    *,
    num_seeds: int,
    output_path: str | Path | None,
) -> Path:
    """Return the CSV destination for one persisted sweep."""
    if output_path is not None:
        return Path(output_path)
    return _project_root() / "results" / _default_results_filename(cfg, num_seeds)


def _resolve_figure_paths(
    results_path: Path,
    *,
    figure_directory: str | Path | None,
) -> tuple[Path, Path]:
    """Return deterministic figure paths that match one CSV output."""
    if figure_directory is None:
        default_results_directory = _project_root() / "results"
        if results_path.parent == default_results_directory:
            resolved_figure_directory = _project_root() / "figures"
        else:
            resolved_figure_directory = results_path.parent
    else:
        resolved_figure_directory = Path(figure_directory)

    base_name = results_path.stem
    return (
        resolved_figure_directory / f"{base_name}_boxplot.png",
        resolved_figure_directory / f"{base_name}_boxplot_sims.png",
    )


def persist_sweep_outputs(
    sweep_result: SweepResult,
    *,
    output_path: str | Path | None = None,
    figure_directory: str | Path | None = None,
) -> SweepStoragePaths:
    """Save one sweep CSV plus both result-summary boxplots.

    The output names are derived from the sweep's normalized run config, so the
    caller does not need to rediscover the right CSV later through
    ``analyze_results()`` directory scans.
    """
    if not sweep_result.results:
        raise ValueError("Cannot persist outputs for an empty sweep result.")

    num_seeds = len(sweep_result.results)
    results_path = _resolve_results_path(
        sweep_result.run_config,
        num_seeds=num_seeds,
        output_path=output_path,
    )
    sample_boxplot_path, simulation_boxplot_path = _resolve_figure_paths(
        results_path,
        figure_directory=figure_directory,
    )

    results_path.parent.mkdir(parents=True, exist_ok=True)
    sample_boxplot_path.parent.mkdir(parents=True, exist_ok=True)

    save_results_to_csv(sweep_result, results_path)

    fig_samples = create_boxplots(
        sweep_result,
        save_path=sample_boxplot_path,
    )
    plt.close(fig_samples)

    fig_sims = create_boxplots_by_simulation(
        sweep_result,
        save_path=simulation_boxplot_path,
    )
    plt.close(fig_sims)

    return SweepStoragePaths(
        results_path=results_path,
        sample_boxplot_path=sample_boxplot_path,
        simulation_boxplot_path=simulation_boxplot_path,
    )
