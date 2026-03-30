"""Top-level SPNI pipeline entrypoints.

This module is the future orchestration replacement for the current script
layer. It should be the only place that wires every stage together.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

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


def _normalize_run_config(
    run_cfg_or_base_cfg: SPNIRunConfig | Any,
    **options,
) -> SPNIRunConfig:
    """Return one normalized run config from either input form."""
    if isinstance(run_cfg_or_base_cfg, SPNIRunConfig):
        if not options:
            return run_cfg_or_base_cfg
        updates = {
            key: value
            for key, value in options.items()
            if hasattr(run_cfg_or_base_cfg, key)
        }
        if not updates:
            return run_cfg_or_base_cfg
        return replace(run_cfg_or_base_cfg, **updates)

    return build_run_config(run_cfg_or_base_cfg, **options)


def _apply_seed_bundle(
    run_cfg: SPNIRunConfig,
    *,
    sweep_seed: int,
    random_seed: int,
    intd_seed: int,
    loader_seed: int,
) -> SPNIRunConfig:
    """Return one run config updated with an explicit seed bundle."""
    return replace(
        run_cfg,
        seed=int(sweep_seed),
        random_seed=int(random_seed),
        intd_seed=int(intd_seed),
        loader_seed=int(loader_seed),
    )


def run_single_simulation(
    run_cfg_or_base_cfg: SPNIRunConfig | Any,
    **options,
) -> SimulationResult:
    """Run one full SPNI simulation and return a typed result object.

    Future implementation responsibilities:
    - normalize config input
    - derive a seed bundle
    - build the graph/model bundle
    - build the dataset bundle
    - train or load predictors
    - evaluate all enabled experiment families
    - derive summary outputs
    - return a complete `SimulationResult`
    """
    run_cfg = _normalize_run_config(run_cfg_or_base_cfg, **options)
    seed_bundle = derive_seed_bundle(run_cfg)
    seeded_run_cfg = _apply_seed_bundle(
        run_cfg,
        sweep_seed=seed_bundle.sweep_seed,
        random_seed=seed_bundle.random_seed,
        intd_seed=seed_bundle.intd_seed,
        loader_seed=seed_bundle.loader_seed,
    )

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

    Future implementation responsibilities:
    - derive one seed bundle per run
    - call `run_single_simulation` in a stable order
    - aggregate the resulting summaries into one `SweepResult`
    """
    run_cfg = _normalize_run_config(run_cfg_or_base_cfg, **options)
    seed_sweep = derive_seed_sweep(run_cfg, num_seeds=num_seeds)

    results = []
    for seed_bundle in seed_sweep:
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
