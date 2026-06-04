"""Top-level SPNI pipeline entrypoints.

This module is the canonical orchestration owner for SPNI. Its job is to call
the stage modules in a fixed order, pass typed artifacts between them, and
return one result object that downstream compatibility layers can adapt.

The pipeline deliberately keeps side effects out of the core run path. Legacy
scripts may still persist results or trigger analysis, but that behavior sits
outside this module.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Sequence

from dflintdpy._runtime import configure_terminal_cache_environment

# Configure cache paths before importing modules that may import Matplotlib.
configure_terminal_cache_environment()

from dflintdpy.simulation.spni.build import build_problem_bundle
from dflintdpy.simulation.spni.config import (
    SPNIRunConfig,
    build_run_config,
    derive_seed_bundle,
    derive_seed_sweep,
    describe_run,
)
from dflintdpy.simulation.spni.data import assemble_dataset_bundle
from dflintdpy.simulation.spni.evaluate import evaluate_all
from dflintdpy.simulation.spni.results import (
    aggregate_sweep_results,
    build_summary,
)
from dflintdpy.simulation.spni.reporting import (
    print_simulation_summary,
    save_learning_curve_plots,
    save_seed_sweep_learning_curve_plots,
)
from dflintdpy.simulation.spni.storage import (
    persist_scenario_sweep_outputs,
    persist_sweep_outputs,
    replot_saved_sweep_outputs,
)
from dflintdpy.simulation.spni.train import train_all_predictors
from dflintdpy.simulation.spni.types import SimulationResult, SweepResult

RunHandler = Callable[..., SimulationResult | SweepResult | dict[str, Any]]

_SCENARIO_SWEEP_PERCENTAGE_KEYS = {
    "unintd": {
        "PO": "no_intd_p",
        "DFL": "no_intd_s",
        "R-DFL": "no_intd_r",
        "A-DFL": "no_intd_a",
    },
    "intd": {
        "PO": "sym_intd_p",
        "DFL": "sym_intd_s",
        "R-DFL": "sym_intd_r",
        "A-DFL": "sym_intd_a",
    },
    "asym": {
        "PO": "asym_intd_p",
        "DFL": "asym_intd_s",
        "R-DFL": "asym_intd_r",
        "A-DFL": "asym_intd_a",
    },
}


def _normalize_run_config(
    run_cfg_or_base_cfg: SPNIRunConfig | Any,
    **options,
) -> SPNIRunConfig:
    """Return one normalized run config from either input form.

    Callers may provide either an already-normalized ``SPNIRunConfig`` or a
    legacy config object such as ``HP``. This helper makes both entry paths
    converge before the expensive stages run.
    """
    if isinstance(run_cfg_or_base_cfg, SPNIRunConfig):
        if not options:
            return run_cfg_or_base_cfg
        # Only apply overrides for fields that actually exist on the frozen run
        # config so unrelated keyword arguments are ignored safely.
        updates = {
            key: value
            for key, value in options.items()
            if hasattr(run_cfg_or_base_cfg, key)
        }
        if not updates:
            return run_cfg_or_base_cfg
        return replace(run_cfg_or_base_cfg, **updates)

    return build_run_config(run_cfg_or_base_cfg, **options)


def _build_base_cfg(cfg: Any | None = None):
    """Return a mutable base config for convenience entrypoints.

    The convenience runner accepts either an explicit base config or nothing at
    all. When no config is supplied, the default SPNI ``HP`` object is used.
    A deep copy is returned so caller-owned config objects are not mutated by
    terminal-driven experiments.
    """
    if cfg is None:
        from dflintdpy.data.config import HP

        cfg = HP()
    return deepcopy(cfg)


def _set_cfg_value(cfg: Any, key: str, value: Any) -> None:
    """Set one config attribute using the legacy setter when available."""
    if isinstance(cfg, dict):
        cfg[key] = value
        return
    setter = getattr(cfg, "set", None)
    if callable(setter):
        setter(key, value)
        return
    setattr(cfg, key, value)


def _apply_cfg_overrides(cfg: Any, **overrides) -> Any:
    """Apply config-style overrides to a config object.

    Legacy base configs are mutated on a private copy. Normalized
    ``SPNIRunConfig`` instances are frozen, so overrides are applied by
    returning a replaced copy instead.
    """
    if isinstance(cfg, SPNIRunConfig):
        updates = {
            key: value
            for key, value in overrides.items()
            if hasattr(cfg, key)
        }
        if not updates:
            return cfg
        return replace(cfg, **updates)

    for key, value in overrides.items():
        _set_cfg_value(cfg, key, value)
    return cfg


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read one config-style value from mappings or legacy config objects."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    getter = getattr(cfg, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(cfg, key, default)


def _mode_handlers() -> dict[str, RunHandler]:
    """Return the supported top-level SPNI execution modes."""
    return {
        "replot": run_saved_result_replot,
        "single": run_single_simulation,
        "seed_sweep": run_seed_sweep,
        "scenario_sweep": run_scenario_sweep,
    }


def _resolve_run_options(
    *,
    num_seeds: int | None,
    compute_asym_intd: bool | None,
    compute_wrong_asym_intd: bool | None,
    load_real_world_graph: str | None,
    source_node: int | None = None,
    target_node: int | None = None,
    present_results: bool | None = None,
    figure_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Normalize optional run-mode flags for the convenience entrypoints."""
    options: dict[str, Any] = {}
    if compute_asym_intd is not None:
        options["compute_asym_intd"] = bool(compute_asym_intd)
    if compute_wrong_asym_intd is not None:
        options["compute_wrong_asym_intd"] = bool(compute_wrong_asym_intd)
    if load_real_world_graph is not None:
        options["load_real_world_graph"] = str(load_real_world_graph)
    if source_node is not None:
        options["source_node"] = int(source_node)
    if target_node is not None:
        options["target_node"] = int(target_node)
    if present_results is not None:
        options["present_results"] = bool(present_results)
    if figure_directory is not None:
        options["figure_directory"] = figure_directory
    if num_seeds is not None:
        options["num_seeds"] = int(num_seeds)
    return options


def _normalize_scenarios(scenarios: Sequence[int]) -> list[int]:
    """Return validated integer scenario counts in caller-specified order."""
    resolved = [int(scenario) for scenario in scenarios]
    if not resolved:
        raise ValueError("Scenario sweeps require at least one scenario count.")
    return resolved


def _init_scenario_stats(
    scenarios: Sequence[int],
) -> tuple[dict[int, dict[str, dict[str, list[float]]]], dict[int, dict[str, dict[str, list[float]]]]]:
    """Build the nested plot-stat containers used by scenario sweeps."""
    sim_stats = {
        int(scenario): {
            condition: {
                method: []
                for method in methods
            }
            for condition, methods in _SCENARIO_SWEEP_PERCENTAGE_KEYS.items()
        }
        for scenario in scenarios
    }
    sample_stats = {
        int(scenario): {
            condition: {
                method: []
                for method in methods
            }
            for condition, methods in _SCENARIO_SWEEP_PERCENTAGE_KEYS.items()
        }
        for scenario in scenarios
    }
    return sim_stats, sample_stats


def _copy_percentage_stats(
    target: dict[int, dict[str, dict[str, list[float]]]],
    scenario: int,
    percentages: dict[str, Any],
) -> None:
    """Populate one scenario bucket from aggregated percentage arrays."""
    for condition, method_map in _SCENARIO_SWEEP_PERCENTAGE_KEYS.items():
        for method, percentage_key in method_map.items():
            values = percentages.get(percentage_key, [])
            target[scenario][condition][method] = [
                float(value)
                for value in list(values)
            ]


def _parse_override(raw_override: str) -> tuple[str, Any]:
    """Parse one ``key=value`` CLI override into a typed pair."""
    if "=" not in raw_override:
        raise ValueError(
            f"Override '{raw_override}' must use the form key=value."
        )
    key, raw_value = raw_override.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError("Override keys must be non-empty.")
    raw_value = raw_value.strip()
    try:
        value = ast.literal_eval(raw_value)
    except (SyntaxError, ValueError):
        value = raw_value
    return key, value


def _parse_scenarios_arg(raw_scenarios: str | None) -> list[int] | None:
    """Parse one comma-separated scenario list from the CLI."""
    if raw_scenarios is None:
        return None
    return _normalize_scenarios(
        [
            part.strip()
            for part in raw_scenarios.split(",")
            if part.strip()
        ]
    )


def _parse_input_paths_arg(
    raw_input_paths: Sequence[Sequence[str]] | None,
) -> str | list[str] | None:
    """Flatten and validate one or more CLI input-path groups."""
    if raw_input_paths is None:
        return None

    input_paths = [
        input_path
        for group in raw_input_paths
        for input_path in group
    ]
    if not input_paths:
        return None
    for input_path in input_paths:
        if not Path(input_path).exists():
            raise FileNotFoundError(
                f"Saved SPNI result CSV not found: {input_path}"
            )
    if len(input_paths) == 1:
        return input_paths[0]
    return input_paths


def _apply_seed_bundle(
    run_cfg: SPNIRunConfig,
    *,
    sweep_seed: int,
    random_seed: int,
    intd_seed: int,
    loader_seed: int,
) -> SPNIRunConfig:
    """Return one run config updated with an explicit seed bundle.

    The base run config is immutable, so sweep execution creates per-run
    variants by replacing the seed-related fields with the derived bundle.
    """
    return replace(
        run_cfg,
        seed=int(sweep_seed),
        random_seed=int(random_seed),
        intd_seed=int(intd_seed),
        loader_seed=int(loader_seed),
    )


def main(
    *,
    mode: str = "seed_sweep",
    cfg: Any | None = None,
    input_path: str | Path | Sequence[str | Path] | None = None,
    exclude_symmetric_interdictions: bool = False,
    legend_location: str | None = None,
    scenarios: Sequence[int] | None = None,
    num_seeds: int | None = None,
    compute_asym_intd: bool | None = None,
    compute_wrong_asym_intd: bool | None = None,
    load_real_world_graph: str | None = None,
    source_node: int | None = None,
    target_node: int | None = None,
    present_results: bool | None = None,
    figure_directory: str | Path | None = None,
    **cfg_overrides,
) -> SimulationResult | SweepResult | dict[str, Any]:
    """Run one SPNI entrypoint with optional config overrides.

    This is the convenience function intended for terminal one-liners and
    notebooks. It lets callers choose the execution mode, override config
    fields, and still receive the same typed results returned by the core
    pipeline functions.

    Parameters:
    - ``mode`` selects the top-level execution path. Supported values are
      ``"single"``, ``"seed_sweep"``, ``"scenario_sweep"``, and
      ``"replot"``.
    - ``input_path`` applies only to ``mode="replot"`` and points to a
      previously saved SPNI result CSV.
    - ``cfg`` optionally provides a legacy base config object. When omitted,
      the default ``HP()`` config is used.
    - ``scenarios`` applies only to scenario sweeps.
    - ``num_seeds`` applies to seed and scenario sweeps.
    - the remaining named parameters are forwarded as pipeline run options
    - arbitrary ``cfg_overrides`` are written onto the copied base config
      before the selected run mode executes
    """
    handlers = _mode_handlers()
    if mode not in handlers:
        supported = ", ".join(sorted(handlers))
        raise ValueError(
            f"Unsupported SPNI run mode '{mode}'. "
            f"Expected one of: {supported}."
        )

    if mode == "replot":
        if input_path is None:
            raise ValueError("mode='replot' requires input_path.")
        return handlers[mode](
            input_path=input_path,
            figure_directory=figure_directory,
            exclude_symmetric_interdictions=(
                exclude_symmetric_interdictions
            ),
            legend_location=legend_location,
        )

    resolved_cfg = _apply_cfg_overrides(
        _build_base_cfg(cfg),
        **cfg_overrides,
    )
    run_options = _resolve_run_options(
        num_seeds=num_seeds,
        compute_asym_intd=compute_asym_intd,
        compute_wrong_asym_intd=compute_wrong_asym_intd,
        load_real_world_graph=load_real_world_graph,
        source_node=source_node,
        target_node=target_node,
        present_results=present_results,
        figure_directory=figure_directory,
    )

    handler = handlers[mode]
    if mode == "seed_sweep":
        resolved_num_seeds = int(
            run_options.pop(
                "num_seeds",
                getattr(resolved_cfg, "num_seeds", 1),
            )
        )
        if legend_location is not None:
            run_options["legend_location"] = legend_location
        return handler(
            resolved_cfg,
            num_seeds=resolved_num_seeds,
            **run_options,
        )

    if mode == "scenario_sweep":
        resolved_num_seeds = int(
            run_options.pop(
                "num_seeds",
                getattr(resolved_cfg, "num_seeds", 1),
            )
        )
        resolved_scenarios = scenarios
        if resolved_scenarios is None:
            resolved_scenarios = [
                getattr(resolved_cfg, "num_scenarios", 1)
            ]
        return handler(
            resolved_cfg,
            scenarios=_normalize_scenarios(resolved_scenarios),
            num_seeds=resolved_num_seeds,
            **run_options,
        )

    run_options.pop("num_seeds", None)
    return handler(resolved_cfg, **run_options)


def run_saved_result_replot(
    *,
    input_path: str | Path | Sequence[str | Path],
    figure_directory: str | Path | None = None,
    exclude_symmetric_interdictions: bool = False,
    legend_location: str | None = None,
) -> dict[str, Any]:
    """Regenerate result boxplots from a saved seed-sweep CSV."""
    stored_paths = replot_saved_sweep_outputs(
        input_path,
        figure_directory=figure_directory,
        exclude_symmetric_interdictions=exclude_symmetric_interdictions,
        legend_location=legend_location,
    )
    stored_input_paths = getattr(
        stored_paths,
        "results_paths",
        (stored_paths.results_path,),
    )
    return {
        "diagnostics": {
            "input_path": str(stored_paths.results_path),
            "input_paths": [
                str(results_path)
                for results_path in stored_input_paths
            ],
            "num_input_files": len(stored_input_paths),
            "exclude_symmetric_interdictions": bool(
                exclude_symmetric_interdictions
            ),
            "legend_location": legend_location or "lower right",
            "sample_boxplot_path": str(stored_paths.sample_boxplot_path),
            "simulation_boxplot_path": str(
                stored_paths.simulation_boxplot_path
            ),
        },
    }


def run_single_simulation(
    run_cfg_or_base_cfg: SPNIRunConfig | Any,
    *,
    present_results: bool = True,
    figure_directory: str | Path | None = None,
    **options,
) -> SimulationResult:
    """Run one full SPNI simulation and return a typed result object.

    Stage order:
    1. normalize the input config
    2. derive the deterministic seed bundle for this run
    3. build the graph and optimization model
    4. assemble the dataset bundle
    5. train or load every predictor family
    6. evaluate all enabled experiment families
    7. derive summary structures for reporting and export

    The returned :class:`SimulationResult` keeps each stage artifact available
    so debugging and compatibility wrappers can inspect the whole run.
    """
    run_cfg = _normalize_run_config(run_cfg_or_base_cfg, **options)
    seed_bundle = derive_seed_bundle(run_cfg)
    # Replace the seed fields explicitly so every downstream stage sees the
    # exact resolved seeds that were used for the run.
    seeded_run_cfg = _apply_seed_bundle(
        run_cfg,
        sweep_seed=seed_bundle.sweep_seed,
        random_seed=seed_bundle.random_seed,
        intd_seed=seed_bundle.intd_seed,
        loader_seed=seed_bundle.loader_seed,
    )

    # The pipeline is intentionally linear: each stage consumes the previous
    # stage's typed bundle and produces the next one.
    graph_bundle = build_problem_bundle(seeded_run_cfg)
    dataset_bundle = assemble_dataset_bundle(seeded_run_cfg, graph_bundle)
    predictor_bundle = train_all_predictors(
        seeded_run_cfg,
        graph_bundle,
        dataset_bundle,
    )
    evaluation_bundle = evaluate_all(
        seeded_run_cfg,
        graph_bundle,
        dataset_bundle,
        predictor_bundle,
    )
    summary_bundle = build_summary(
        seeded_run_cfg,
        dataset_bundle,
        predictor_bundle,
        evaluation_bundle,
    )

    result = SimulationResult(
        run_config=seeded_run_cfg,
        seed_bundle=seed_bundle,
        graph_bundle=graph_bundle,
        dataset_bundle=dataset_bundle,
        predictor_bundle=predictor_bundle,
        evaluation_bundle=evaluation_bundle,
        summary_bundle=summary_bundle,
        diagnostics={
            "run_description": describe_run(seeded_run_cfg),
            "present_results": bool(present_results),
            "side_effects_enabled": bool(present_results),
        },
    )
    if present_results:
        print_simulation_summary(result)
        figure_paths = save_learning_curve_plots(
            result,
            figure_directory=figure_directory,
        )
        result.diagnostics["learning_curve_plot_paths"] = figure_paths
    else:
        result.diagnostics["learning_curve_plot_paths"] = {}
    return result


def run_seed_sweep(
    run_cfg_or_base_cfg: SPNIRunConfig | Any,
    *,
    num_seeds: int,
    present_results: bool = True,
    output_path: str | None = None,
    figure_directory: str | None = None,
    legend_location: str | None = None,
    **options,
) -> SweepResult:
    """Run a multi-seed SPNI sweep and return an aggregated result object.

    The sweep path is a thin loop around ``run_single_simulation(...)``. It is
    responsible for deriving the ordered seed bundles, applying them to the run
    config, and aggregating the resulting simulation summaries.
    """
    run_cfg = _normalize_run_config(run_cfg_or_base_cfg, **options)
    seed_sweep = derive_seed_sweep(run_cfg, num_seeds=num_seeds)

    results = []
    for seed_bundle in seed_sweep:
        # Each sweep item becomes one fully-resolved single-run config before
        # the core pipeline is invoked.
        seeded_run_cfg = _apply_seed_bundle(
            run_cfg,
            sweep_seed=seed_bundle.sweep_seed,
            random_seed=seed_bundle.random_seed,
            intd_seed=seed_bundle.intd_seed,
            loader_seed=seed_bundle.loader_seed,
        )
        result = run_single_simulation(
            seeded_run_cfg,
            present_results=present_results,
            figure_directory=figure_directory,
        )
        results.append(result)

    aggregated_summary = aggregate_sweep_results(results)
    sweep_result = SweepResult(
        run_config=run_cfg,
        results=results,
        aggregated_summary=aggregated_summary,
        diagnostics={
            "num_runs": len(results),
            "sweep_seeds": [bundle.sweep_seed for bundle in seed_sweep],
            "present_results": bool(present_results),
            "side_effects_enabled": bool(present_results),
            "plot_filter": aggregated_summary.get(
                "diagnostics",
                {},
            ).get("plot_filter", {}),
        },
    )
    if present_results:
        learning_curve_paths = save_seed_sweep_learning_curve_plots(
            results,
            figure_directory=figure_directory,
        )
        stored_paths = persist_sweep_outputs(
            sweep_result,
            output_path=output_path,
            figure_directory=figure_directory,
            legend_location=legend_location,
        )
        sweep_result.diagnostics.update(
            {
                "legacy_output_path": str(stored_paths.results_path),
                "sample_boxplot_path": str(stored_paths.sample_boxplot_path),
                "simulation_boxplot_path": str(
                    stored_paths.simulation_boxplot_path
                ),
                "learning_curve_seed_sweep_plot_paths": learning_curve_paths,
            }
        )
    else:
        sweep_result.diagnostics.update(
            {
                "legacy_output_path": None,
                "sample_boxplot_path": None,
                "simulation_boxplot_path": None,
                "learning_curve_seed_sweep_plot_paths": {},
            }
        )
    return sweep_result


def run_scenario_sweep(
    run_cfg_or_base_cfg: SPNIRunConfig | Any,
    *,
    scenarios: Sequence[int],
    num_seeds: int | None = None,
    present_results: bool = True,
    figure_directory: str | Path | None = None,
    compute_asym_intd: bool | None = None,
    compute_wrong_asym_intd: bool | None = None,
    load_real_world_graph: str | None = None,
    source_node: int | None = None,
    target_node: int | None = None,
) -> dict[str, Any]:
    """Run seed sweeps across scenario counts and return plot-ready stats.

    This is the canonical replacement for the legacy
    ``scripts.plot_scenario_sweep.run_sweep(...)`` helper. Each scenario count
    reuses ``run_seed_sweep(...)`` and then adapts the aggregated percentage
    metrics into the legacy nested dictionaries expected by the plotting code.
    When ``present_results`` is enabled, the outer sweep saves the summary
    scenario plots after all scenario counts have finished.
    """
    resolved_scenarios = _normalize_scenarios(scenarios)
    resolved_num_seeds = int(
        _cfg_get(run_cfg_or_base_cfg, "num_seeds", 1)
        if num_seeds is None else num_seeds
    )
    sim_stats, sample_stats = _init_scenario_stats(resolved_scenarios)
    sweep_results: dict[int, SweepResult] = {}
    base_cfg = _build_base_cfg(run_cfg_or_base_cfg)
    run_options = _resolve_run_options(
        num_seeds=None,
        compute_asym_intd=compute_asym_intd,
        compute_wrong_asym_intd=compute_wrong_asym_intd,
        load_real_world_graph=load_real_world_graph,
        source_node=source_node,
        target_node=target_node,
    )

    for scenario in resolved_scenarios:
        scenario_cfg = _apply_cfg_overrides(
            _build_base_cfg(base_cfg),
            num_scenarios=int(scenario),
        )
        sweep_result = run_seed_sweep(
            scenario_cfg,
            num_seeds=resolved_num_seeds,
            present_results=False,
            **run_options,
        )
        sweep_results[int(scenario)] = sweep_result
        percentage_increases = sweep_result.aggregated_summary.get(
            "percentage_increases",
            {},
        )
        _copy_percentage_stats(
            sim_stats,
            int(scenario),
            percentage_increases.get("simulations", {}),
        )
        _copy_percentage_stats(
            sample_stats,
            int(scenario),
            percentage_increases.get("samples", {}),
        )

    diagnostics = {
        "num_scenarios_swept": len(resolved_scenarios),
        "scenario_counts": resolved_scenarios,
        "present_results": bool(present_results),
        "side_effects_enabled": bool(present_results),
    }
    if present_results:
        stored_paths = persist_scenario_sweep_outputs(
            run_cfg=base_cfg,
            scenarios=resolved_scenarios,
            num_seeds=resolved_num_seeds,
            sim_stats=sim_stats,
            sample_stats=sample_stats,
            figure_directory=figure_directory,
        )
        diagnostics.update(
            {
                "simulation_plot_path": str(
                    stored_paths.simulation_plot_path
                ),
                "asym_simulation_plot_path": str(
                    stored_paths.asym_simulation_plot_path
                ),
                "sample_plot_path": str(stored_paths.sample_plot_path),
                "asym_sample_plot_path": str(
                    stored_paths.asym_sample_plot_path
                ),
            }
        )
    else:
        diagnostics.update(
            {
                "simulation_plot_path": None,
                "asym_simulation_plot_path": None,
                "sample_plot_path": None,
                "asym_sample_plot_path": None,
            }
        )

    return {
        "scenarios": resolved_scenarios,
        "num_seeds": resolved_num_seeds,
        "sweep_results": sweep_results,
        "sim_stats": sim_stats,
        "sample_stats": sample_stats,
        "diagnostics": diagnostics,
    }


def cli(
    argv: Sequence[str] | None = None,
    *,
    prog: str | None = None,
) -> SimulationResult | SweepResult | dict[str, Any]:
    """Parse one-line terminal arguments and run the requested SPNI mode."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Run SPNI simulations, sweeps, or saved-result replots."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=sorted(_mode_handlers()),
        default="seed_sweep",
        help="Top-level SPNI execution mode.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=None,
        help="Number of seeds to run when mode=seed_sweep or scenario_sweep.",
    )
    parser.add_argument(
        "--input-path",
        action="append",
        nargs="+",
        default=None,
        help=(
            "Saved SPNI result CSV(s) to use when mode=replot. "
            "Pass multiple paths after one flag or repeat the flag."
        ),
    )
    parser.add_argument(
        "--scenarios",
        default=None,
        help=(
            "Comma-separated scenario counts to run when "
            "mode=scenario_sweep."
        ),
    )
    parser.add_argument(
        "--compute-asym-intd",
        dest="compute_asym_intd",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable asymmetric interdiction evaluation.",
    )
    parser.add_argument(
        "--compute-wrong-asym-intd",
        dest="compute_wrong_asym_intd",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable wrong-model asymmetric evaluation.",
    )
    parser.add_argument(
        "--load-real-world-graph",
        default=None,
        help="Optional path to a real-world graph CSV.",
    )
    parser.add_argument(
        "--source-node",
        type=int,
        default=None,
        help="Optional graph source node for shortest-path solves.",
    )
    parser.add_argument(
        "--target-node",
        type=int,
        default=None,
        help="Optional graph target node for shortest-path solves.",
    )
    parser.add_argument(
        "--present-results",
        dest="present_results",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Persist seed-sweep CSV/boxplots or scenario-sweep summary "
            "figures."
        ),
    )
    parser.add_argument(
        "--figure-directory",
        default=None,
        help="Optional directory for generated SPNI figures.",
    )
    parser.add_argument(
        "--exclude-symmetric-interdictions",
        action="store_true",
        default=False,
        help=(
            "When mode=replot, omit symmetric-interdiction boxplot groups "
            "and write *_no_sym figure files."
        ),
    )
    parser.add_argument(
        "--legend-location",
        default=None,
        help=(
            "Matplotlib legend location for generated result boxplots, e.g. "
            "'upper left', 'lower right', or 'best'."
        ),
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Config override written onto the base config before execution.",
    )
    parsed = parser.parse_args(argv)

    cfg_overrides: dict[str, Any] = {}
    for raw_override in parsed.overrides:
        key, value = _parse_override(raw_override)
        cfg_overrides[key] = value

    result = main(
        mode=parsed.mode,
        input_path=_parse_input_paths_arg(parsed.input_path),
        exclude_symmetric_interdictions=(
            parsed.exclude_symmetric_interdictions
        ),
        legend_location=parsed.legend_location,
        scenarios=_parse_scenarios_arg(parsed.scenarios),
        num_seeds=parsed.num_seeds,
        compute_asym_intd=parsed.compute_asym_intd,
        compute_wrong_asym_intd=parsed.compute_wrong_asym_intd,
        load_real_world_graph=parsed.load_real_world_graph,
        source_node=parsed.source_node,
        target_node=parsed.target_node,
        present_results=parsed.present_results,
        figure_directory=parsed.figure_directory,
        **cfg_overrides,
    )
    diagnostics = (
        result.get("diagnostics", {})
        if isinstance(result, dict)
        else result.diagnostics
    )
    print(
        f"Completed SPNI {parsed.mode}: "
        f"{diagnostics}"
    )
    return result


if __name__ == "__main__":
    cli()
