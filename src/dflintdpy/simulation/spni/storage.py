"""Persistence helpers for SPNI sweep results and derived figures.

This module keeps result-export side effects separate from the core execution
pipeline. Callers that want durable artefacts can pass a typed ``SweepResult``
here and receive deterministic CSV and figure outputs without rescanning the
results directory later.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from dflintdpy.simulation.spni.naming import (
    cfg_get as _cfg_get,
    real_world_graph_filename_suffix,
)
from dflintdpy.simulation.spni.types import SweepResult
from dflintdpy.utils.analyse_results import (
    create_boxplots,
    create_boxplots_by_simulation,
)
from dflintdpy.utils.read_write_results import (
    load_results_from_csv,
    save_results_to_csv,
)


@dataclass(frozen=True)
class SweepStoragePaths:
    """Filesystem targets produced by ``persist_sweep_outputs(...)``."""

    results_path: Path
    sample_boxplot_path: Path
    simulation_boxplot_path: Path


@dataclass(frozen=True)
class ReplotStoragePaths:
    """Filesystem targets produced by ``replot_saved_sweep_outputs(...)``."""

    results_path: Path
    sample_boxplot_path: Path
    simulation_boxplot_path: Path
    results_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class ScenarioSweepStoragePaths:
    """Filesystem targets produced by scenario-sweep presentation helpers."""

    simulation_plot_path: Path
    asym_simulation_plot_path: Path
    sample_plot_path: Path
    asym_sample_plot_path: Path


_SCENARIO_METHOD_ORDER = ("PO", "DFL", "R-DFL", "A-DFL")
_SCENARIO_METHOD_COLORS = {
    "PO": "#FF6B6B",
    "DFL": "#4ECDC4",
    "R-DFL": "#FFA552",
    "A-DFL": "#45B7D1",
}
_SCENARIO_CONDITION_TITLES = {
    "unintd": "Uninterdicted",
    "intd": "Symmetric Interdiction",
    "asym": "Asymmetric Interdiction",
}
_DEFAULT_BOXPLOT_LEGEND_LOCATION = "lower right"
_RESULTS_STEM_WITH_SEEDS = re.compile(r"^(?P<prefix>.+)_seeds_\d+$")


def _project_root() -> Path:
    """Return the repository root for default results/figure directories."""
    return Path(__file__).resolve().parents[4]


def _default_results_filename(cfg: Any, num_seeds: int) -> str:
    """Build the legacy results filename from one config object."""
    m_size, n_size = _cfg_get(cfg, "grid_size", (0, 0))
    graph_suffix = real_world_graph_filename_suffix(cfg)
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
        f"{graph_suffix}"
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
    filename_suffix: str = "",
) -> tuple[Path, Path]:
    """Return deterministic figure paths that match one CSV output."""
    if figure_directory is None:
        resolved_figure_directory = _project_root() / "figures"
    else:
        resolved_figure_directory = Path(figure_directory)

    base_name = f"{results_path.stem}{filename_suffix}"
    return (
        resolved_figure_directory / f"{base_name}_boxplot.png",
        resolved_figure_directory / f"{base_name}_boxplot_sims.png",
    )


def _coerce_results_paths(
    results_path: str | Path | Sequence[str | Path],
) -> tuple[Path, ...]:
    """Return one or more saved CSV paths as normalized Path objects."""
    if isinstance(results_path, (str, Path)):
        return (Path(results_path),)

    resolved_paths = tuple(Path(path) for path in results_path)
    if not resolved_paths:
        raise ValueError("At least one saved SPNI result CSV is required.")
    return resolved_paths


def _combined_replot_stem(
    results_paths: Sequence[Path],
    *,
    num_simulations: int,
) -> str:
    """Return a stable figure stem for one or more replot input CSVs."""
    if len(results_paths) == 1:
        return results_paths[0].stem

    seed_prefixes = []
    for results_path in results_paths:
        match = _RESULTS_STEM_WITH_SEEDS.match(results_path.stem)
        if match is None:
            seed_prefixes = []
            break
        seed_prefixes.append(match.group("prefix"))

    if seed_prefixes and len(set(seed_prefixes)) == 1:
        return f"{seed_prefixes[0]}_seeds_{num_simulations}_combined"

    common_prefix = os.path.commonprefix(
        [results_path.stem for results_path in results_paths]
    ).rstrip("_-.")
    if not common_prefix:
        common_prefix = "combined_results"
    return f"{common_prefix}_combined_{len(results_paths)}_files"


def persist_sweep_outputs(
    sweep_result: SweepResult,
    *,
    output_path: str | Path | None = None,
    figure_directory: str | Path | None = None,
    legend_location: str | None = None,
    exclude_symmetric_interdictions: bool = True,
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
    suffix = "_no_sym" if exclude_symmetric_interdictions else ""
    sample_boxplot_path, simulation_boxplot_path = _resolve_figure_paths(
        results_path,
        figure_directory=figure_directory,
        filename_suffix=suffix,
    )

    results_path.parent.mkdir(parents=True, exist_ok=True)
    sample_boxplot_path.parent.mkdir(parents=True, exist_ok=True)

    save_results_to_csv(sweep_result, results_path)

    fig_samples = create_boxplots(
        sweep_result,
        save_path=sample_boxplot_path,
        include_symmetric_interdiction=not exclude_symmetric_interdictions,
        legend_location=(
            legend_location or _DEFAULT_BOXPLOT_LEGEND_LOCATION
        ),
    )
    plt.close(fig_samples)

    fig_sims = create_boxplots_by_simulation(
        sweep_result,
        save_path=simulation_boxplot_path,
        include_symmetric_interdiction=not exclude_symmetric_interdictions,
        legend_location=(
            legend_location or _DEFAULT_BOXPLOT_LEGEND_LOCATION
        ),
    )
    plt.close(fig_sims)

    return SweepStoragePaths(
        results_path=results_path,
        sample_boxplot_path=sample_boxplot_path,
        simulation_boxplot_path=simulation_boxplot_path,
    )


def replot_saved_sweep_outputs(
    results_path: str | Path | Sequence[str | Path],
    *,
    figure_directory: str | Path | None = None,
    exclude_symmetric_interdictions: bool = True,
    legend_location: str | None = None,
) -> ReplotStoragePaths:
    """Regenerate sweep boxplots from one or more saved result CSVs."""
    resolved_results_paths = _coerce_results_paths(results_path)
    for resolved_results_path in resolved_results_paths:
        if not resolved_results_path.exists():
            raise FileNotFoundError(
                f"Saved SPNI result CSV not found: {resolved_results_path}"
            )

    simulations = [
        simulation
        for resolved_results_path in resolved_results_paths
        for simulation in load_results_from_csv(resolved_results_path)
    ]

    suffix = "_no_sym" if exclude_symmetric_interdictions else ""
    combined_stem = _combined_replot_stem(
        resolved_results_paths,
        num_simulations=len(simulations),
    )
    sample_boxplot_path, simulation_boxplot_path = _resolve_figure_paths(
        resolved_results_paths[0],
        figure_directory=figure_directory,
        filename_suffix="",
    )
    sample_boxplot_path = (
        sample_boxplot_path.parent / f"{combined_stem}{suffix}_boxplot.png"
    )
    simulation_boxplot_path = (
        simulation_boxplot_path.parent
        / f"{combined_stem}{suffix}_boxplot_sims.png"
    )
    sample_boxplot_path.parent.mkdir(parents=True, exist_ok=True)

    fig_samples = create_boxplots(
        simulations,
        save_path=sample_boxplot_path,
        include_symmetric_interdiction=not exclude_symmetric_interdictions,
        legend_location=(
            legend_location or _DEFAULT_BOXPLOT_LEGEND_LOCATION
        ),
    )
    plt.close(fig_samples)

    fig_sims = create_boxplots_by_simulation(
        simulations,
        save_path=simulation_boxplot_path,
        include_symmetric_interdiction=not exclude_symmetric_interdictions,
        legend_location=(
            legend_location or _DEFAULT_BOXPLOT_LEGEND_LOCATION
        ),
    )
    plt.close(fig_sims)

    return ReplotStoragePaths(
        results_path=resolved_results_paths[0],
        sample_boxplot_path=sample_boxplot_path,
        simulation_boxplot_path=simulation_boxplot_path,
        results_paths=resolved_results_paths,
    )


def _default_scenario_sweep_base_name(
    cfg: Any,
    *,
    scenarios: Sequence[int],
    num_seeds: int,
) -> str:
    """Build a deterministic filename stem for one scenario sweep."""
    m_size, n_size = _cfg_get(cfg, "grid_size", (0, 0))
    scenario_label = "-".join(str(int(scenario)) for scenario in scenarios)
    graph_suffix = real_world_graph_filename_suffix(cfg)
    return (
        "scenario_sweep_train_"
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
        f"{graph_suffix}"
        "_seeds_"
        f"{int(num_seeds)}"
        "_scenarios_"
        f"{scenario_label}"
    )


def _resolve_scenario_sweep_figure_paths(
    cfg: Any,
    *,
    scenarios: Sequence[int],
    num_seeds: int,
    figure_directory: str | Path | None,
) -> ScenarioSweepStoragePaths:
    """Return deterministic figure paths for one scenario sweep."""
    if figure_directory is None:
        resolved_figure_directory = _project_root() / "figures"
    else:
        resolved_figure_directory = Path(figure_directory)

    base_name = _default_scenario_sweep_base_name(
        cfg,
        scenarios=scenarios,
        num_seeds=num_seeds,
    )
    return ScenarioSweepStoragePaths(
        simulation_plot_path=(
            resolved_figure_directory / f"{base_name}_by_simulation.png"
        ),
        asym_simulation_plot_path=(
            resolved_figure_directory / f"{base_name}_asym_by_simulation.png"
        ),
        sample_plot_path=(
            resolved_figure_directory / f"{base_name}_by_sample.png"
        ),
        asym_sample_plot_path=(
            resolved_figure_directory / f"{base_name}_asym_by_sample.png"
        ),
    )


def _collect_scenario_mean_std(
    stats: dict[int, dict[str, dict[str, list[float]]]],
    scenarios: Sequence[int],
    condition: str,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the mean and std used by scenario-sweep line plots."""
    means = []
    stds = []
    for scenario in scenarios:
        values = np.asarray(
            stats[int(scenario)][condition][method],
            dtype=float,
        )
        if values.size == 0:
            means.append(0.0)
            stds.append(0.0)
            continue
        means.append(float(values.mean()))
        stds.append(float(values.std()))
    return np.asarray(means), np.asarray(stds)


def plot_scenario_sweep_stats(
    stats: dict[int, dict[str, dict[str, list[float]]]],
    *,
    scenarios: Sequence[int],
    output_path: str | Path,
    title: str,
    conditions: Sequence[str],
    show: bool = False,
) -> None:
    """Render one saved scenario-sweep figure from plot-ready stats."""
    x_values = np.asarray([int(scenario) for scenario in scenarios], dtype=int)
    fig, axes = plt.subplots(
        1,
        len(conditions),
        figsize=(7 * len(conditions), 5),
        sharex=True,
    )
    if len(conditions) == 1:
        axes = [axes]

    for method in _SCENARIO_METHOD_ORDER:
        for index, condition in enumerate(conditions):
            mean_values, std_values = _collect_scenario_mean_std(
                stats,
                scenarios,
                condition,
                method,
            )
            axes[index].errorbar(
                x_values,
                mean_values,
                yerr=std_values,
                marker="o",
                capsize=4,
                linewidth=1.8,
                color=_SCENARIO_METHOD_COLORS[method],
                label=method,
            )

    for index, condition in enumerate(conditions):
        ax = axes[index]
        ax.set_title(_SCENARIO_CONDITION_TITLES[condition])
        ax.set_xlabel("Number of Adverse Scenarios")
        ax.set_ylabel("Percentage cost increase vs oracle (%)")
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend()

    fig.suptitle(title, fontsize=14, fontweight="bold")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def persist_scenario_sweep_outputs(
    *,
    run_cfg: Any,
    scenarios: Sequence[int],
    num_seeds: int,
    sim_stats: dict[int, dict[str, dict[str, list[float]]]],
    sample_stats: dict[int, dict[str, dict[str, list[float]]]],
    figure_directory: str | Path | None = None,
) -> ScenarioSweepStoragePaths:
    """Save the default presentation figures for one scenario sweep."""
    stored_paths = _resolve_scenario_sweep_figure_paths(
        run_cfg,
        scenarios=scenarios,
        num_seeds=num_seeds,
        figure_directory=figure_directory,
    )
    stored_paths.simulation_plot_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_scenario_sweep_stats(
        sim_stats,
        scenarios=scenarios,
        output_path=stored_paths.simulation_plot_path,
        title="Percentage Increase vs Oracle (Symmetric Simulations)",
        conditions=("unintd", "intd"),
    )
    plot_scenario_sweep_stats(
        sim_stats,
        scenarios=scenarios,
        output_path=stored_paths.asym_simulation_plot_path,
        title="Percentage Increase vs Oracle (Asymmetric Simulations)",
        conditions=("asym",),
    )
    plot_scenario_sweep_stats(
        sample_stats,
        scenarios=scenarios,
        output_path=stored_paths.sample_plot_path,
        title="Percentage Increase vs Oracle (Mean+Std over Samples)",
        conditions=("unintd", "intd"),
    )
    plot_scenario_sweep_stats(
        sample_stats,
        scenarios=scenarios,
        output_path=stored_paths.asym_sample_plot_path,
        title=(
            "Asymmetric Interdiction: Percentage Increase vs Oracle "
            "(Mean+Std over Samples)"
        ),
        conditions=("asym",),
    )
    return stored_paths
