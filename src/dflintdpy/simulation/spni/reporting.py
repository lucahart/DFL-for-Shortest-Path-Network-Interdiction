"""Terminal and figure reporting helpers for SPNI simulations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

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


def _plot_log_bundle(
    axes,
    logs: TrainingLogBundle,
    *,
    label: str,
) -> None:
    """Plot one predictor's loss and regret traces on existing axes."""
    ax_loss, ax_regret = axes
    if logs.train_loss:
        ax_loss.plot(logs.train_loss, marker=".", label=f"{label} train")
    if logs.val_loss:
        ax_loss.plot(logs.val_loss, marker=".", linestyle="--",
                     label=f"{label} val")
    if logs.train_regret:
        ax_regret.plot(logs.train_regret, marker=".", label=f"{label} train")
    if logs.val_regret:
        ax_regret.plot(logs.val_regret, marker=".", linestyle="--",
                       label=f"{label} val")


def _finalize_learning_curve_figure(fig, axes, title: str, output_path: Path) -> None:
    """Apply common labels and persist one learning-curve figure."""
    ax_loss, ax_regret = axes
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Logged epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(alpha=0.3, linestyle="--")
    ax_loss.legend()
    ax_regret.set_title("Regret")
    ax_regret.set_xlabel("Logged epoch")
    ax_regret.set_ylabel("Regret")
    ax_regret.grid(alpha=0.3, linestyle="--")
    ax_regret.legend()
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _learning_curve_stem(result: SimulationResult) -> str:
    """Build a deterministic filename stem for one run's learning curves."""
    return (
        f"learning_curves_seed_{result.seed_bundle.sweep_seed}"
        f"_scenarios_{result.run_config.num_scenarios}"
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

    if figure_directory is None:
        resolved_directory = _project_root() / "figures" / "learning_curves"
    else:
        resolved_directory = Path(figure_directory) / "learning_curves"

    stem = _learning_curve_stem(result)
    paths: dict[str, str] = {}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for family, logs in logs_by_family.items():
        _plot_log_bundle(
            axes,
            logs,
            label=_LEARNING_CURVE_LABELS.get(family, family.upper()),
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
        _plot_log_bundle(axes, logs, label=label)
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
