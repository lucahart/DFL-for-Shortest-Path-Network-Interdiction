"""Top-level SPNI pipeline entrypoints.

This module is the future orchestration replacement for the current script
layer. It should be the only place that wires every stage together.
"""

from __future__ import annotations

from typing import Any

from dflintdpy.simulation.spni.config import SPNIRunConfig
from dflintdpy.simulation.spni.types import SimulationResult, SweepResult


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

    raise NotImplementedError("Specification scaffold only.")


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

    raise NotImplementedError("Specification scaffold only.")

