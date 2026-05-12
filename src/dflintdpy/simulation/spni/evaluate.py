"""Evaluation orchestration for SPNI simulations.

This module wraps the legacy comparison helpers in typed, stage-specific
functions. The solver logic still lives in the legacy helpers; the refactor
layer is responsible for shaping their inputs and outputs into stable,
sample-aligned payloads for downstream summary code.

Responsibilities:
- adapt the normalized run config and dataset bundle into legacy helper inputs
- preserve sample alignment across every evaluation family
- keep skipped experiments and failed solves explicit in diagnostics
- return one canonical evaluation bundle for the pipeline
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

import dflintdpy.scripts.compare as legacy_compare_module
from dflintdpy.simulation.spni.config import SPNIRunConfig
from dflintdpy.simulation.spni.types import (
    DatasetBundle,
    EvaluationBundle,
    GraphBundle,
    PredictorBundle,
)


class _LegacyConfigAdapter:
    """Expose normalized run-config values through the legacy ``.get(...)`` API.

    The compare helpers still consume config-like objects, so this adapter
    keeps the normalized run config as the source of truth while satisfying
    that older interface.
    """

    def __init__(self, run_cfg: SPNIRunConfig):
        self._run_cfg = run_cfg

    def get(self, key: str, default: Any = None) -> Any:
        """Read one value from the normalized config or its base config."""
        if hasattr(self._run_cfg, key):
            return getattr(self._run_cfg, key)

        base_cfg = self._run_cfg.base_cfg
        getter = getattr(base_cfg, "get", None)
        if callable(getter):
            return getter(key, default)
        if isinstance(base_cfg, Mapping):
            return base_cfg.get(key, default)
        return getattr(base_cfg, key, default)


def _sample_indices(num_samples: int) -> np.ndarray:
    """Return the canonical sample-index vector for evaluation outputs.

    Every evaluation payload carries this index vector so summary/export code
    can prove that arrays remained aligned through the helper calls.
    """
    return np.arange(num_samples, dtype=int)


def _build_testing_view(dataset_bundle: DatasetBundle) -> dict[str, Any]:
    """Build the legacy testing-data mapping expected by compare helpers.

    The legacy compare layer expects plain dictionaries with fixed key names.
    """
    return {
        "feats": dataset_bundle.testing_features,
        "costs": dataset_bundle.testing_costs,
    }


def _build_interdiction_view(dataset_bundle: DatasetBundle) -> dict[str, Any]:
    """Build the legacy interdiction mapping expected by compare helpers.

    Keeping this shape conversion in one helper avoids repeating hard-coded
    key names across the stage functions.
    """
    return {
        "feats": dataset_bundle.interdiction_features,
        "costs": dataset_bundle.interdiction_costs,
    }


def _as_float_array(values: Any) -> np.ndarray:
    """Normalize one helper output into a float NumPy array.

    The compare helpers may return lists or arrays. Converting immediately
    gives the rest of the stage a single predictable numeric representation.
    """
    return np.asarray(values, dtype=float)


def _align_array(
    values: Any,
    *,
    expected_len: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Align one helper output to the test-set length with explicit metadata.

    Some legacy helpers can return fewer rows than the test set when solves
    fail or when an experiment is conditionally skipped. This function pads or
    trims the output into a stable test-set-length vector and records what
    happened in diagnostics.
    """
    arr = _as_float_array(values)
    original_len = int(arr.shape[0]) if arr.ndim > 0 else int(arr.size)
    if arr.ndim == 0:
        arr = arr.reshape(1)

    if original_len == expected_len:
        return arr, {
            "expected_len": expected_len,
            "returned_len": original_len,
            "exact_alignment": True,
            "padded_count": 0,
            "trimmed_count": 0,
        }

    aligned = np.full(expected_len, np.nan, dtype=float)
    copied = min(expected_len, original_len)
    if copied > 0:
        aligned[:copied] = arr[:copied]
    return aligned, {
        "expected_len": expected_len,
        "returned_len": original_len,
        "exact_alignment": False,
        "padded_count": max(0, expected_len - original_len),
        "trimmed_count": max(0, original_len - expected_len),
    }


def _empty_stage(
    *,
    num_samples: int,
    reason: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a stable skipped-stage payload.

    Stages that are disabled by config should still return the same top-level
    structure shape so downstream consumers do not need special-case branches.
    """
    diagnostics = {
        "skipped": True,
        "reason": reason,
        "num_samples": int(num_samples),
    }
    if extra:
        diagnostics.update(dict(extra))
    return {
        "sample_indices": _sample_indices(num_samples),
        "diagnostics": diagnostics,
    }


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
    legacy_cfg = _LegacyConfigAdapter(run_cfg)
    testing_data = _build_testing_view(dataset_bundle)
    num_samples = int(dataset_bundle.testing_features.shape[0])

    # The current helper API can compare only three predictors at a time, so
    # the wrapper makes two calls to cover all four learned model families.
    true_objs, pfl_objs, dfl_objs, adfl_objs = \
        legacy_compare_module.compare_shortest_paths(
            legacy_cfg,
            graph_bundle.opt_model,
            predictor_bundle.pfl,
            predictor_bundle.dfl,
            testing_data,
            predictor_bundle.adfl,
        )
    _, _, _, rdfl_objs = legacy_compare_module.compare_shortest_paths(
        legacy_cfg,
        graph_bundle.opt_model,
        predictor_bundle.pfl,
        predictor_bundle.dfl,
        testing_data,
        predictor_bundle.rdfl,
    )
    # TODO: Unify compare_shortest_paths so that it can compare an arbitrary
    # number of predictors in one call.

    return {
        "sample_indices": _sample_indices(num_samples),
        "objectives": {
            "oracle": _as_float_array(true_objs),
            "pfl": _as_float_array(pfl_objs),
            "dfl": _as_float_array(dfl_objs),
            "rdfl": _as_float_array(rdfl_objs),
            "adfl": _as_float_array(adfl_objs),
        },
        "diagnostics": {
            "num_samples": num_samples,
            "helper_calls": 2,
        },
    }


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
    legacy_cfg = _LegacyConfigAdapter(run_cfg)
    testing_data = _build_testing_view(dataset_bundle)
    interdictions = _build_interdiction_view(dataset_bundle)
    num_samples = int(dataset_bundle.testing_features.shape[0])

    # Symmetric interdiction already returns one dictionary containing every
    # family, so the wrapper mainly renames fields and records diagnostics.
    results = legacy_compare_module.compare_sym_intd(
        legacy_cfg,
        graph_bundle.opt_model,
        predictor_bundle.pfl,
        predictor_bundle.dfl,
        testing_data,
        interdictions,
        dataset_bundle.normalization_constant,
        adfl_predictor=predictor_bundle.adfl,
        rand_adfl_predictor=predictor_bundle.rdfl,
    )

    return {
        "sample_indices": _sample_indices(num_samples),
        "objectives": {
            "oracle": _as_float_array(results["true_objective"]),
            "pfl": _as_float_array(results["po_objective"]),
            "dfl": _as_float_array(results["spo_objective"]),
            "rdfl": _as_float_array(results["rand_adv_spo_objective"]),
            "adfl": _as_float_array(results["adv_spo_objective"]),
        },
        "diagnostics": {
            "num_samples": num_samples,
            "returned_keys": sorted(results),
        },
    }


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
    num_samples = int(dataset_bundle.testing_features.shape[0])
    if not run_cfg.compute_asym_intd:
        return _empty_stage(
            num_samples=num_samples,
            reason="run_config_disabled",
            extra={
                "estimated_objectives": [],
                "oracle_objectives": [],
            },
        ) | {
            "estimated_objectives": {},
            "oracle_objectives": {},
        }

    legacy_cfg = _LegacyConfigAdapter(run_cfg)
    testing_data = _build_testing_view(dataset_bundle)
    interdictions = _build_interdiction_view(dataset_bundle)
    predictor_map = {
        "oracle": None,
        "pfl": predictor_bundle.pfl,
        "dfl": predictor_bundle.dfl,
        "rdfl": predictor_bundle.rdfl,
        "adfl": predictor_bundle.adfl,
    }

    estimated_objectives: dict[str, np.ndarray] = {}
    oracle_objectives: dict[str, np.ndarray] = {}
    failure_counts: dict[str, int] = {}

    for family, predictor in predictor_map.items():
        # Each family is evaluated independently because the legacy helper takes
        # only one predictor at a time.
        estimated, oracle = legacy_compare_module.compare_asym_intd(
            legacy_cfg,
            graph_bundle.opt_model,
            testing_data,
            interdictions,
            dataset_bundle.normalization_constant,
            predictor,
            pred_family=family if predictor is not None else None,
        )
        estimated_arr = _as_float_array(estimated)
        oracle_arr = _as_float_array(oracle)
        estimated_objectives[family] = estimated_arr
        oracle_objectives[family] = oracle_arr
        # NaNs are treated as solve failures and surfaced explicitly rather than
        # being silently ignored by downstream summary code.
        failure_counts[family] = int(
            max(
                np.isnan(estimated_arr).sum(),
                np.isnan(oracle_arr).sum(),
            )
        )

    return {
        "sample_indices": _sample_indices(num_samples),
        "estimated_objectives": estimated_objectives,
        "oracle_objectives": oracle_objectives,
        "diagnostics": {
            "num_samples": num_samples,
            "failure_counts": failure_counts,
        },
    }


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
    num_samples = int(dataset_bundle.testing_features.shape[0])
    if not run_cfg.compute_wrong_asym_intd:
        return _empty_stage(
            num_samples=num_samples,
            reason="run_config_disabled",
            extra={"pair_count": 0},
        ) | {"objectives": {}}

    legacy_cfg = _LegacyConfigAdapter(run_cfg)
    testing_data = _build_testing_view(dataset_bundle)
    interdictions = _build_interdiction_view(dataset_bundle)
    predictor_pairs = {
        "true_dfl_false_pfl": (predictor_bundle.dfl, predictor_bundle.pfl),
        "true_pfl_false_dfl": (predictor_bundle.pfl, predictor_bundle.dfl),
        "true_adfl_false_pfl": (predictor_bundle.adfl, predictor_bundle.pfl),
        "true_pfl_false_adfl": (predictor_bundle.pfl, predictor_bundle.adfl),
        "true_adfl_false_dfl": (predictor_bundle.adfl, predictor_bundle.dfl),
        "true_dfl_false_adfl": (predictor_bundle.dfl, predictor_bundle.adfl),
    }

    objectives: dict[str, np.ndarray] = {}
    pair_diagnostics: dict[str, dict[str, Any]] = {}
    for label, (true_model, false_model) in predictor_pairs.items():
        # Wrong-model experiments can return short arrays, so every pair is
        # normalized back to the full test-set length before leaving the stage.
        raw_values = legacy_compare_module.compare_wrong_asym_intd(
            legacy_cfg,
            graph_bundle.opt_model,
            testing_data,
            interdictions,
            dataset_bundle.normalization_constant,
            true_model=true_model,
            false_model=false_model,
        )
        aligned_values, alignment_diag = _align_array(
            raw_values,
            expected_len=num_samples,
        )
        objectives[label] = aligned_values
        pair_diagnostics[label] = alignment_diag

    return {
        "sample_indices": _sample_indices(num_samples),
        "objectives": objectives,
        "diagnostics": {
            "num_samples": num_samples,
            "skipped": False,
            "pair_count": len(predictor_pairs),
            "pair_diagnostics": pair_diagnostics,
        },
    }


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
    # Run the families in reporting order so the returned bundle mirrors the
    # summary/export expectations used elsewhere in the repository.
    uninterdicted = evaluate_uninterdicted(
        run_cfg,
        graph_bundle,
        dataset_bundle,
        predictor_bundle,
    )
    symmetric = evaluate_symmetric_interdiction(
        run_cfg,
        graph_bundle,
        dataset_bundle,
        predictor_bundle,
    )
    asymmetric = evaluate_asymmetric_interdiction(
        run_cfg,
        graph_bundle,
        dataset_bundle,
        predictor_bundle,
    )
    wrong_model_asymmetry = evaluate_wrong_model_asymmetry(
        run_cfg,
        graph_bundle,
        dataset_bundle,
        predictor_bundle,
    )

    return EvaluationBundle(
        uninterdicted=uninterdicted,
        symmetric=symmetric,
        asymmetric=asymmetric,
        wrong_model_asymmetry=wrong_model_asymmetry,
        diagnostics={
            "num_samples": int(dataset_bundle.testing_features.shape[0]),
            "asymmetric_failure_counts": asymmetric["diagnostics"].get(
                "failure_counts",
                {},
            ),
            "wrong_model_skipped": bool(
                wrong_model_asymmetry["diagnostics"].get("skipped", False)
            ),
        },
    )
