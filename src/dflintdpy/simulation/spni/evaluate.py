"""Evaluation orchestration for SPNI simulations.

This module should wrap the existing comparison logic in typed and testable
stage functions. It is responsible for preserving sample alignment and for
recording skipped or failed solves explicitly.
"""

from __future__ import annotations

from dflintdpy.simulation.spni.config import SPNIRunConfig
from dflintdpy.simulation.spni.types import (
    DatasetBundle,
    EvaluationBundle,
    GraphBundle,
    PredictorBundle,
)


def evaluate_uninterdicted(
    run_cfg: SPNIRunConfig,
    graph_bundle: GraphBundle,
    dataset_bundle: DatasetBundle,
    predictor_bundle: PredictorBundle,
):
    """Evaluate predictors on the uninterdicted shortest-path task.

    Future implementation responsibilities:
    - return one aligned output row per test sample
    - keep raw objective arrays available for downstream summaries
    """

    raise NotImplementedError("Specification scaffold only.")


def evaluate_symmetric_interdiction(
    run_cfg: SPNIRunConfig,
    graph_bundle: GraphBundle,
    dataset_bundle: DatasetBundle,
    predictor_bundle: PredictorBundle,
):
    """Evaluate predictors under symmetric SPNI interdiction.

    Future implementation responsibilities:
    - delegate to the current symmetric-comparison logic
    - preserve method labels and array alignment
    """

    raise NotImplementedError("Specification scaffold only.")


def evaluate_asymmetric_interdiction(
    run_cfg: SPNIRunConfig,
    graph_bundle: GraphBundle,
    dataset_bundle: DatasetBundle,
    predictor_bundle: PredictorBundle,
):
    """Evaluate predictors under asymmetric SPNI interdiction.

    Future implementation responsibilities:
    - represent failed solves explicitly instead of silently dropping samples
    - expose failure counts through diagnostics
    """

    raise NotImplementedError("Specification scaffold only.")


def evaluate_wrong_model_asymmetry(
    run_cfg: SPNIRunConfig,
    graph_bundle: GraphBundle,
    dataset_bundle: DatasetBundle,
    predictor_bundle: PredictorBundle,
):
    """Run the wrong-evader-model asymmetric experiments.

    Future implementation responsibilities:
    - execute only when enabled in the run config
    - return a stable structure even when the experiment is skipped
    """

    raise NotImplementedError("Specification scaffold only.")


def evaluate_all(
    run_cfg: SPNIRunConfig,
    graph_bundle: GraphBundle,
    dataset_bundle: DatasetBundle,
    predictor_bundle: PredictorBundle,
) -> EvaluationBundle:
    """Run every evaluation family required for one SPNI simulation.

    Future implementation responsibilities:
    - coordinate the lower-level evaluation helpers
    - store stage diagnostics such as failed-solve counts
    - return one canonical evaluation bundle
    """

    raise NotImplementedError("Specification scaffold only.")

