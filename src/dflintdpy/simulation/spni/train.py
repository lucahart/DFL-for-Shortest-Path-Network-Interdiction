"""Predictor training orchestration for SPNI simulations.

This module should coordinate the existing trainer helpers and predictor setup
functions. It should not own the trainer core or the predictor math itself.
"""

from __future__ import annotations

from dflintdpy.simulation.spni.config import SPNIRunConfig
from dflintdpy.simulation.spni.types import (
    DatasetBundle,
    GraphBundle,
    PredictorBundle,
)


def train_pfl_predictor(
    run_cfg: SPNIRunConfig,
    graph_bundle: GraphBundle,
    dataset_bundle: DatasetBundle,
):
    """Train or load the PFL predictor.

    Future implementation responsibilities:
    - call the existing PFL setup helper
    - return the predictor plus its log bundle
    - keep cache-tag handling explicit and centralized
    """

    raise NotImplementedError("Specification scaffold only.")


def train_dfl_predictor(
    run_cfg: SPNIRunConfig,
    graph_bundle: GraphBundle,
    dataset_bundle: DatasetBundle,
    *,
    variant_name: str,
):
    """Train or load one DFL-family predictor.

    Future implementation responsibilities:
    - support baseline DFL, random adverse DFL, and adversarial DFL
    - keep the mapping from variant name to training data explicit
    - return the predictor plus its log bundle
    """

    raise NotImplementedError("Specification scaffold only.")


def train_all_predictors(
    run_cfg: SPNIRunConfig,
    graph_bundle: GraphBundle,
    dataset_bundle: DatasetBundle,
) -> PredictorBundle:
    """Train or load all predictor families used by the SPNI pipeline.

    Future implementation responsibilities:
    - produce PFL, DFL, R-DFL, and A-DFL models
    - attach structured logs and diagnostics
    - return one canonical predictor bundle
    """

    raise NotImplementedError("Specification scaffold only.")

