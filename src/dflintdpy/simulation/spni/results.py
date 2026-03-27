"""Summary and export adapters for SPNI simulations.

This module should own all translation from raw numerical outputs into:
- summary metrics
- legacy `all_data` mappings
- flat CSV rows
- multi-run aggregated summaries
"""

from __future__ import annotations

from dflintdpy.simulation.spni.config import SPNIRunConfig
from dflintdpy.simulation.spni.types import (
    DatasetBundle,
    EvaluationBundle,
    PredictorBundle,
    SimulationResult,
    SummaryBundle,
)


def build_summary(
    run_cfg: SPNIRunConfig,
    dataset_bundle: DatasetBundle,
    predictor_bundle: PredictorBundle,
    evaluation_bundle: EvaluationBundle,
) -> SummaryBundle:
    """Build the derived summary structures for one run.

    Future implementation responsibilities:
    - compute the same summary outputs currently produced by the scripts
    - keep metric derivation centralized and testable
    """

    raise NotImplementedError("Specification scaffold only.")


def to_legacy_all_data(summary_bundle: SummaryBundle) -> dict:
    """Expose the current `all_data` structure for compatibility.

    Future implementation responsibilities:
    - preserve current analysis-script expectations during migration
    - keep legacy column names in one module rather than many scripts
    """

    raise NotImplementedError("Specification scaffold only.")


def flatten_result_rows(result: SimulationResult) -> list[dict]:
    """Flatten one simulation result into CSV-ready row dictionaries.

    Future implementation responsibilities:
    - keep simulation and sample indices stable
    - preserve explicit missing-value semantics
    """

    raise NotImplementedError("Specification scaffold only.")


def aggregate_sweep_results(results: list[SimulationResult]) -> dict:
    """Aggregate multiple run results into a sweep-level summary.

    Future implementation responsibilities:
    - compute per-run and per-sample aggregate views
    - expose the exact data needed by later plotting and analysis code
    """

    raise NotImplementedError("Specification scaffold only.")

