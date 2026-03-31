"""Top-level SPNI pipeline entrypoints.

This module is the canonical orchestration owner for SPNI. Its job is to call
the stage modules in a fixed order, pass typed artifacts between them, and
return one result object that downstream compatibility layers can adapt.

The pipeline deliberately keeps side effects out of the core run path. Legacy
scripts may still persist results or trigger analysis, but that behavior sits
outside this module.
"""

from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from dataclasses import replace
from typing import Any, Callable, Sequence

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
from dflintdpy.simulation.spni.train import train_all_predictors
from dflintdpy.simulation.spni.types import SimulationResult, SweepResult

RunHandler = Callable[..., SimulationResult | SweepResult]


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


def _mode_handlers() -> dict[str, RunHandler]:
    """Return the supported top-level SPNI execution modes."""
    return {
        "single": run_single_simulation,
        "seed_sweep": run_seed_sweep,
    }


def _resolve_run_options(
    *,
    num_seeds: int | None,
    compute_asym_intd: bool | None,
    compute_wrong_asym_intd: bool | None,
    load_real_world_graph: str | None,
) -> dict[str, Any]:
    """Normalize optional run-mode flags for the convenience entrypoints."""
    options: dict[str, Any] = {}
    if compute_asym_intd is not None:
        options["compute_asym_intd"] = bool(compute_asym_intd)
    if compute_wrong_asym_intd is not None:
        options["compute_wrong_asym_intd"] = bool(compute_wrong_asym_intd)
    if load_real_world_graph is not None:
        options["load_real_world_graph"] = str(load_real_world_graph)
    if num_seeds is not None:
        options["num_seeds"] = int(num_seeds)
    return options


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
    num_seeds: int | None = None,
    compute_asym_intd: bool | None = None,
    compute_wrong_asym_intd: bool | None = None,
    load_real_world_graph: str | None = None,
    **cfg_overrides,
) -> SimulationResult | SweepResult:
    """Run one SPNI entrypoint with optional config overrides.

    This is the convenience function intended for terminal one-liners and
    notebooks. It lets callers choose the execution mode, override config
    fields, and still receive the same typed results returned by the core
    pipeline functions.

    Parameters:
    - ``mode`` selects the top-level execution path. Supported values are
      ``"single"`` and ``"seed_sweep"``.
    - ``cfg`` optionally provides a legacy base config object. When omitted,
      the default ``HP()`` config is used.
    - ``num_seeds`` applies only to seed sweeps.
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

    resolved_cfg = _apply_cfg_overrides(
        _build_base_cfg(cfg),
        **cfg_overrides,
    )
    run_options = _resolve_run_options(
        num_seeds=num_seeds,
        compute_asym_intd=compute_asym_intd,
        compute_wrong_asym_intd=compute_wrong_asym_intd,
        load_real_world_graph=load_real_world_graph,
    )

    handler = handlers[mode]
    if mode == "seed_sweep":
        resolved_num_seeds = int(
            run_options.pop(
                "num_seeds",
                getattr(resolved_cfg, "num_seeds", 1),
            )
        )
        return handler(
            resolved_cfg,
            num_seeds=resolved_num_seeds,
            **run_options,
        )

    run_options.pop("num_seeds", None)
    return handler(resolved_cfg, **run_options)


def run_single_simulation(
    run_cfg_or_base_cfg: SPNIRunConfig | Any,
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

    return SimulationResult(
        run_config=seeded_run_cfg,
        seed_bundle=seed_bundle,
        graph_bundle=graph_bundle,
        dataset_bundle=dataset_bundle,
        predictor_bundle=predictor_bundle,
        evaluation_bundle=evaluation_bundle,
        summary_bundle=summary_bundle,
        diagnostics={
            "run_description": describe_run(seeded_run_cfg),
            "side_effects_enabled": False,
        },
    )


def run_seed_sweep(
    run_cfg_or_base_cfg: SPNIRunConfig | Any,
    *,
    num_seeds: int,
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
        result = run_single_simulation(seeded_run_cfg)
        results.append(result)

    return SweepResult(
        run_config=run_cfg,
        results=results,
        aggregated_summary=aggregate_sweep_results(results),
        diagnostics={
            "num_runs": len(results),
            "sweep_seeds": [bundle.sweep_seed for bundle in seed_sweep],
            "side_effects_enabled": False,
        },
    )


def cli(argv: Sequence[str] | None = None) -> SimulationResult | SweepResult:
    """Parse one-line terminal arguments and run the requested SPNI mode."""
    parser = argparse.ArgumentParser(
        description="Run SPNI single simulations or seed sweeps.",
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
        help="Number of seeds to run when mode=seed_sweep.",
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
        num_seeds=parsed.num_seeds,
        compute_asym_intd=parsed.compute_asym_intd,
        compute_wrong_asym_intd=parsed.compute_wrong_asym_intd,
        load_real_world_graph=parsed.load_real_world_graph,
        **cfg_overrides,
    )
    print(
        f"Completed SPNI {parsed.mode}: "
        f"{result.diagnostics}"
    )
    return result


if __name__ == "__main__":
    cli()
