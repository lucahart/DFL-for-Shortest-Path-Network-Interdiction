"""Terminal and figure reporting helpers for SPNI simulations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from dflintdpy.simulation.spni.naming import real_world_graph_filename_suffix
from dflintdpy.simulation.spni.types import (
    SimulationResult,
    TrainingLogBundle,
)

_PREDICTOR_LABELS = {
    "o": "Oracle",
    "p": "PFL",
    "s": "DFL",
    "r": "R-DFL",
    "a": "A-DFL",
}

_LEARNING_CURVE_LABELS = {
    "pfl": "PFL",
    "dfl": "DFL",
    "rdfl": "R-DFL",
    "adfl": "A-DFL",
}


def _project_root() -> Path:
    """Return the repository root for default figure directories."""
    return Path(__file__).resolve().parents[4]


def _format_value(value: Any) -> str:
    """Format one summary value for terminal tables."""
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(number):
        return "nan"
    return f"{number:.4f}"


def _run_metadata_rows(result: SimulationResult) -> list[list[Any]]:
    """Build rows describing the completed simulation run."""
    run_cfg = result.run_config
    graph_bundle = result.graph_bundle
    evaluation_bundle = getattr(result, "evaluation_bundle", None)
    diagnostics = getattr(evaluation_bundle, "diagnostics", {}) or {}
    failure_counts = diagnostics.get("asymmetric_failure_counts", {})
    total_asym_failures = sum(int(value) for value in failure_counts.values())
    return [
        ["grid_size", getattr(run_cfg, "grid_size", "-")],
        ["graph_kind", graph_bundle.graph_kind],
        ["graph_source", graph_bundle.graph_source or "-"],
        ["num_train_samples", getattr(run_cfg, "num_train_samples", "-")],
        ["num_val_samples", getattr(run_cfg, "num_val_samples", "-")],
        ["num_test_samples", getattr(run_cfg, "num_test_samples", "-")],
        ["num_scenarios", getattr(run_cfg, "num_scenarios", "-")],
        ["budget", getattr(run_cfg, "budget", "-")],
        ["sweep_seed", result.seed_bundle.sweep_seed],
        ["random_seed", result.seed_bundle.random_seed],
        ["asymmetric_failed_solves", total_asym_failures],
    ]


def _prediction_rows(result: SimulationResult) -> list[list[str]]:
    """Build prediction-stat rows from the summary bundle."""
    stats = result.summary_bundle.prediction_mean_std
    rows = [
        ["Test true costs", stats.get("test_mean"), stats.get("test_std")],
        ["Train true costs", stats.get("train_mean"), stats.get("train_std")],
        ["Interdiction costs", stats.get("intd_mean"), stats.get("intd_std")],
        ["PFL predictions", stats.get("po_mean"), stats.get("po_std")],
        ["DFL predictions", stats.get("spo_mean"), stats.get("spo_std")],
        [
            "R-DFL predictions",
            stats.get("rand_spo_mean"),
            stats.get("rand_spo_std"),
        ],
        [
            "A-DFL predictions",
            stats.get("adv_spo_mean"),
            stats.get("adv_spo_std"),
        ],
    ]
    return [[label, _format_value(mean), _format_value(std)]
            for label, mean, std in rows]


def _objective_rows(result: SimulationResult) -> list[list[str]]:
    """Build the legacy Table 1 objective rows in display order."""
    table_1 = result.summary_bundle.table_1
    rows: list[list[str]] = []
    for prefix, label in _PREDICTOR_LABELS.items():
        rows.append(
            [
                label,
                _format_value(table_1.get(f"t1_{prefix}_n_mean")),
                _format_value(table_1.get(f"t1_{prefix}_s_mean")),
                _format_value(table_1.get(f"t1_{prefix}_s_std")),
                _format_value(table_1.get(f"t1_{prefix}_a_mean")),
                _format_value(table_1.get(f"t1_{prefix}_a_std")),
            ]
        )
    return rows


def _wrong_model_rows(result: SimulationResult) -> list[list[str]]:
    """Build optional wrong-model asymmetric summary rows."""
    table_2 = result.summary_bundle.table_2
    labels = {
        "p": "PFL true response",
        "s": "DFL true response",
        "a": "A-DFL true response",
    }
    rows = []
    for prefix, label in labels.items():
        rows.append(
            [
                label,
                _format_value(table_2.get(f"t2_{prefix}_s_mean")),
                _format_value(table_2.get(f"t2_{prefix}_s_std")),
                _format_value(table_2.get(f"t2_{prefix}_a_mean")),
                _format_value(table_2.get(f"t2_{prefix}_a_std")),
            ]
        )
    return rows


def _metric_rows(result: SimulationResult) -> list[list[str]]:
    """Build scalar metric rows in deterministic order."""
    metrics = result.summary_bundle.metrics
    return [
        [key, _format_value(metrics[key])]
        for key in sorted(metrics)
    ]


def format_simulation_summary_tables(result: SimulationResult) -> list[str]:
    """Return terminal-ready summary tables for one simulation result."""
    sections = [
        (
            "Run metadata",
            tabulate(
                _run_metadata_rows(result),
                headers=["Field", "Value"],
                tablefmt="github",
            ),
        ),
        (
            "Prediction summary",
            tabulate(
                _prediction_rows(result),
                headers=["Quantity", "Mean", "Std"],
                tablefmt="github",
            ),
        ),
        (
            "Table 1",
            tabulate(
                _objective_rows(result),
                headers=[
                    "Model",
                    "No Intd Mean",
                    "Sym Mean",
                    "Sym Std",
                    "Asym Mean",
                    "Asym Std",
                ],
                tablefmt="github",
            ),
        ),
        (
            "Metrics",
            tabulate(
                _metric_rows(result),
                headers=["Metric", "Value"],
                tablefmt="github",
            ),
        ),
    ]
    if result.summary_bundle.table_2:
        sections.insert(
            3,
            (
                "Table 2",
                tabulate(
                    _wrong_model_rows(result),
                    headers=[
                        "Response Model",
                        "False DFL Mean",
                        "False DFL Std",
                        "False A-DFL Mean",
                        "False A-DFL Std",
                    ],
                    tablefmt="github",
                ),
            ),
        )
    return [f"{title}\n{table}" for title, table in sections]


def print_simulation_summary(result: SimulationResult) -> None:
    """Print summary tables for one simulation result."""
    print("\nSPNI Simulation Results")
    tables = format_simulation_summary_tables(result)
    if isinstance(tables, dict):
        table_values = tables.values()
    else:
        table_values = tables
    for table in table_values:
        print()
        print(table)


def _has_curve_data(logs: TrainingLogBundle) -> bool:
    """Return whether one predictor has any train or validation trace."""
    return any(
        bool(values)
        for values in (
            logs.train_loss,
            logs.train_regret,
            logs.val_loss,
            logs.val_regret,
        )
    )


def _resolve_learning_curve_directory(
    figure_directory: str | Path | None,
) -> Path:
    """Return the directory used for saved learning-curve figures."""
    if figure_directory is None:
        return _project_root() / "figures" / "learning_curves"
    return Path(figure_directory) / "learning_curves"


def _configured_epochs_for_family(
    run_cfg: Any,
    family: str,
    logs: TrainingLogBundle,
) -> int:
    """Return the configured total training epochs for one predictor family."""
    fallback = max(len(logs.train_loss), len(logs.train_regret), 1) - 1
    field_name = "po_epochs" if family == "pfl" else "spo_epochs"
    raw_value = getattr(run_cfg, field_name, fallback)
    try:
        return max(0, int(raw_value))
    except (TypeError, ValueError):
        return fallback


def _validation_epoch_positions(
    num_values: int,
    *,
    train_log_length: int,
    total_epochs: int,
) -> np.ndarray:
    """Return actual epoch numbers for sparsely logged validation values."""
    if num_values <= 0:
        return np.asarray([], dtype=int)

    actual_epochs = max(0, int(train_log_length) - 1)
    configured_epochs = max(0, int(total_epochs))
    validation_stride = max(1, configured_epochs // 10)
    positions = [0]
    positions.extend(
        range(validation_stride, actual_epochs + 1, validation_stride)
    )

    if len(positions) < num_values and actual_epochs not in positions:
        positions.append(actual_epochs)
    while len(positions) < num_values:
        positions.append(positions[-1] + validation_stride)

    return np.asarray(positions[:num_values], dtype=int)


def _line_color(line_objects) -> Any:
    """Return the first Matplotlib line color when available."""
    if not line_objects:
        return None
    first_line = line_objects[0]
    get_color = getattr(first_line, "get_color", None)
    if callable(get_color):
        return get_color()
    return None


def _plot_log_bundle(
    axes,
    logs: TrainingLogBundle,
    *,
    label: str,
    total_epochs: int,
) -> None:
    """Plot one predictor's loss and regret traces on existing axes."""
    ax_loss, ax_regret = axes
    loss_color = None
    if logs.train_loss:
        train_loss_x = np.arange(len(logs.train_loss))
        loss_color = _line_color(
            ax_loss.plot(
                train_loss_x,
                logs.train_loss,
                marker=".",
                label=f"{label} train",
            )
        )
    if logs.val_loss:
        val_loss_x = _validation_epoch_positions(
            len(logs.val_loss),
            train_log_length=len(logs.train_loss),
            total_epochs=total_epochs,
        )
        ax_loss.scatter(
            val_loss_x,
            logs.val_loss,
            marker="x",
            color=loss_color,
            label=f"{label} val",
        )
    regret_color = None
    if logs.train_regret:
        train_regret_x = np.arange(len(logs.train_regret))
        regret_color = _line_color(
            ax_regret.plot(
                train_regret_x,
                logs.train_regret,
                marker=".",
                label=f"{label} train",
            )
        )
    if logs.val_regret:
        val_regret_x = _validation_epoch_positions(
            len(logs.val_regret),
            train_log_length=len(logs.train_regret),
            total_epochs=total_epochs,
        )
        ax_regret.scatter(
            val_regret_x,
            logs.val_regret,
            marker="x",
            color=regret_color,
            label=f"{label} val",
        )


def _finalize_learning_curve_figure(fig, axes, title: str, output_path: Path) -> None:
    """Apply common labels and persist one learning-curve figure."""
    ax_loss, ax_regret = axes
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_yscale("log", nonpositive="clip")
    ax_loss.grid(alpha=0.3, linestyle="--")
    ax_loss.legend()
    ax_regret.set_title("Regret")
    ax_regret.set_xlabel("Epoch")
    ax_regret.set_ylabel("Regret")
    ax_regret.set_yscale("log", nonpositive="clip")
    ax_regret.grid(alpha=0.3, linestyle="--")
    ax_regret.legend()
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _learning_curve_stem(result: SimulationResult) -> str:
    """Build a deterministic filename stem for one run's learning curves."""
    graph_suffix = real_world_graph_filename_suffix(result.run_config)
    return (
        f"learning_curves_seed_{result.seed_bundle.sweep_seed}"
        f"_scenarios_{result.run_config.num_scenarios}"
        f"{graph_suffix}"
    )


def save_learning_curve_plots(
    result: SimulationResult,
    *,
    figure_directory: str | Path | None = None,
) -> dict[str, str]:
    """Save combined and per-predictor learning-curve plots for one run."""
    logs_by_family = {
        family: logs
        for family, logs in result.predictor_bundle.logs.items()
        if _has_curve_data(logs)
    }
    if not logs_by_family:
        return {}

    resolved_directory = _resolve_learning_curve_directory(figure_directory)
    stem = _learning_curve_stem(result)
    paths: dict[str, str] = {}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for family, logs in logs_by_family.items():
        _plot_log_bundle(
            axes,
            logs,
            label=_LEARNING_CURVE_LABELS.get(family, family.upper()),
            total_epochs=_configured_epochs_for_family(
                result.run_config,
                family,
                logs,
            ),
        )
    combined_path = resolved_directory / f"{stem}_combined.png"
    _finalize_learning_curve_figure(
        fig,
        axes,
        "Learning Curves by Predictor",
        combined_path,
    )
    paths["learning_curves_combined"] = str(combined_path)

    for family, logs in logs_by_family.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        label = _LEARNING_CURVE_LABELS.get(family, family.upper())
        _plot_log_bundle(
            axes,
            logs,
            label=label,
            total_epochs=_configured_epochs_for_family(
                result.run_config,
                family,
                logs,
            ),
        )
        output_path = resolved_directory / f"{stem}_{family}.png"
        _finalize_learning_curve_figure(
            fig,
            axes,
            f"{label} Learning Curves",
            output_path,
        )
        paths[f"learning_curve_{family}"] = str(output_path)

    artifacts = getattr(result, "artifacts", None)
    figure_paths = getattr(artifacts, "figure_paths", None)
    if isinstance(figure_paths, dict):
        figure_paths.update(paths)

    return paths


def _seed_sweep_learning_curve_stem(results: list[SimulationResult]) -> str:
    """Build a deterministic filename stem for sweep-level curve plots."""
    run_cfg = results[0].run_config
    graph_suffix = real_world_graph_filename_suffix(run_cfg)
    return (
        f"learning_curves_seed_sweep_seeds_{len(results)}"
        f"_scenarios_{run_cfg.num_scenarios}"
        f"{graph_suffix}"
    )


def save_seed_sweep_learning_curve_plots(
    results: list[SimulationResult],
    *,
    figure_directory: str | Path | None = None,
) -> dict[str, str]:
    """Save one seed-comparison learning-curve plot per predictor family."""
    if not results:
        return {}

    resolved_directory = _resolve_learning_curve_directory(figure_directory)
    families = sorted(
        {
            family
            for result in results
            for family, logs in result.predictor_bundle.logs.items()
            if _has_curve_data(logs)
        }
    )
    if not families:
        return {}

    stem = _seed_sweep_learning_curve_stem(results)
    paths: dict[str, str] = {}
    for family in families:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for result in results:
            logs = result.predictor_bundle.logs.get(family)
            if logs is None or not _has_curve_data(logs):
                continue
            _plot_log_bundle(
                axes,
                logs,
                label=f"seed {result.seed_bundle.sweep_seed}",
                total_epochs=_configured_epochs_for_family(
                    result.run_config,
                    family,
                    logs,
                ),
            )
        output_path = resolved_directory / f"{stem}_{family}_by_seed.png"
        label = _LEARNING_CURVE_LABELS.get(family, family.upper())
        _finalize_learning_curve_figure(
            fig,
            axes,
            f"{label} Learning Curves by Seed",
            output_path,
        )
        paths[f"learning_curve_seed_sweep_{family}"] = str(output_path)

    return paths
