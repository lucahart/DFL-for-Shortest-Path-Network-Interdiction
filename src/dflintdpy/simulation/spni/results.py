"""Summary and export adapters for SPNI simulations.

This module should own all translation from raw numerical outputs into:
- summary metrics
- legacy `all_data` mappings
- flat CSV rows
- multi-run aggregated summaries
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from dflintdpy.simulation.spni.config import SPNIRunConfig
from dflintdpy.simulation.spni.types import (
    DatasetBundle,
    EvaluationBundle,
    PredictorBundle,
    SimulationResult,
    SummaryBundle,
)

LEGACY_ALL_DATA_KEYS = (
    "o_o",
    "o_p",
    "o_s",
    "o_r",
    "o_a",
    "s_o",
    "s_p",
    "s_s",
    "s_r",
    "s_a",
    "a_o",
    "a_p",
    "a_p_o",
    "a_s",
    "a_s_o",
    "a_r",
    "a_r_o",
    "a_a",
    "a_a_o",
)

WRONG_MODEL_ALL_DATA_KEYS = (
    "a_s_p",
    "a_p_s",
    "a_a_p",
    "a_p_a",
    "a_a_s",
    "a_s_a",
)

_UNINTERDICTED_KEY_MAP = {
    "o_o": "oracle",
    "o_p": "pfl",
    "o_s": "dfl",
    "o_r": "rdfl",
    "o_a": "adfl",
}

_SYMMETRIC_KEY_MAP = {
    "s_o": "oracle",
    "s_p": "pfl",
    "s_s": "dfl",
    "s_r": "rdfl",
    "s_a": "adfl",
}

_ASYMMETRIC_EST_KEY_MAP = {
    "a_o": "oracle",
    "a_p": "pfl",
    "a_s": "dfl",
    "a_r": "rdfl",
    "a_a": "adfl",
}

_ASYMMETRIC_ORACLE_KEY_MAP = {
    "a_p_o": "pfl",
    "a_s_o": "dfl",
    "a_r_o": "rdfl",
    "a_a_o": "adfl",
}

_WRONG_MODEL_KEY_MAP = {
    "a_s_p": "true_dfl_false_pfl",
    "a_p_s": "true_pfl_false_dfl",
    "a_a_p": "true_adfl_false_pfl",
    "a_p_a": "true_pfl_false_adfl",
    "a_a_s": "true_adfl_false_dfl",
    "a_s_a": "true_dfl_false_adfl",
}

_TABLE_1_PREFIX_MAP = {
    "oracle": "o",
    "pfl": "p",
    "dfl": "s",
    "rdfl": "r",
    "adfl": "a",
}

_TABLE_2_GROUPS = {
    "p": ("a_p_s", "a_p_a"),
    "s": ("a_s_p", "a_s_a"),
    "a": ("a_a_p", "a_a_s"),
}

_PREDICTION_STAT_KEYS = {
    "pfl": "po",
    "dfl": "spo",
    "rdfl": "rand_spo",
    "adfl": "adv_spo",
}


def _as_float_array(values: Any) -> np.ndarray:
    """Normalize one numerical payload into a float array copy."""
    return np.asarray(values, dtype=float).copy()


def _safe_nanmean(values: Any) -> float:
    """Return a warning-free mean that tolerates empty and NaN-only arrays."""
    arr = _as_float_array(values)
    if arr.size == 0 or np.isnan(arr).all():
        return float("nan")
    return float(np.nanmean(arr))


def _safe_nanstd(values: Any) -> float:
    """Return a warning-free std that tolerates empty and NaN-only arrays."""
    arr = _as_float_array(values)
    if arr.size == 0 or np.isnan(arr).all():
        return float("nan")
    return float(np.nanstd(arr))


def _safe_percentage(num: Any, denom: Any) -> np.ndarray:
    """Compute percentage safely and replace NaN/Inf with zeros."""
    num_arr = _as_float_array(num)
    denom_arr = _as_float_array(denom)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (num_arr - denom_arr) / denom_arr * 100.0
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_percentage_sum(num: Any, denom: Any) -> float:
    """Compute an aggregate percentage safely for one simulation."""
    num_arr = _as_float_array(num)
    denom_arr = _as_float_array(denom)
    denom_sum = float(np.nansum(denom_arr))
    if denom_sum == 0.0:
        return 0.0
    num_sum = float(np.nansum(num_arr - denom_arr))
    with np.errstate(divide="ignore", invalid="ignore"):
        value = num_sum / denom_sum * 100.0
    return float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))


def _array_length(values: Sequence[Any] | np.ndarray) -> int:
    """Return the first-axis length of one legacy export array."""
    return int(_as_float_array(values).shape[0])


def _predictor_output_array(predictor: Any, features) -> np.ndarray:
    """Evaluate one predictor on the testing features when possible."""
    if predictor is None:
        return np.asarray([], dtype=float)

    outputs: Any | None = None
    try:
        outputs = predictor(features)
    except Exception:
        try:
            import torch
        except Exception:
            return np.asarray([], dtype=float)

        with torch.no_grad():
            tensor = torch.tensor(features, dtype=torch.float32)
            outputs = predictor(tensor)

    if hasattr(outputs, "detach"):
        outputs = outputs.detach().cpu().numpy()
    return _as_float_array(outputs)


def _predictor_prediction_stats(predictor: Any, features) -> tuple[float, float]:
    """Return mean/std statistics for one predictor output family."""
    outputs = _predictor_output_array(predictor, features)
    if outputs.size == 0:
        return float("nan"), float("nan")
    return float(outputs.mean()), float(outputs.std())


def _extract_all_data(
    dataset_bundle: DatasetBundle,
    evaluation_bundle: EvaluationBundle,
) -> dict[str, np.ndarray]:
    """Translate the typed evaluation payload into the legacy `all_data` map."""
    normalization_constant = float(dataset_bundle.normalization_constant)
    all_data: dict[str, np.ndarray] = {}

    uninterdicted = evaluation_bundle.uninterdicted["objectives"]
    for key, family in _UNINTERDICTED_KEY_MAP.items():
        all_data[key] = _as_float_array(uninterdicted[family]) \
            * normalization_constant

    symmetric = evaluation_bundle.symmetric["objectives"]
    for key, family in _SYMMETRIC_KEY_MAP.items():
        all_data[key] = _as_float_array(symmetric[family])

    asymmetric_est = evaluation_bundle.asymmetric["estimated_objectives"]
    for key, family in _ASYMMETRIC_EST_KEY_MAP.items():
        all_data[key] = _as_float_array(asymmetric_est[family])

    asymmetric_oracle = evaluation_bundle.asymmetric["oracle_objectives"]
    for key, family in _ASYMMETRIC_ORACLE_KEY_MAP.items():
        all_data[key] = _as_float_array(asymmetric_oracle[family])

    wrong_model = evaluation_bundle.wrong_model_asymmetry
    if not wrong_model.get("diagnostics", {}).get("skipped", False):
        wrong_model_objectives = wrong_model.get("objectives", {})
        for key, pair_name in _WRONG_MODEL_KEY_MAP.items():
            if pair_name in wrong_model_objectives:
                all_data[key] = _as_float_array(wrong_model_objectives[pair_name])

    return all_data


def _build_prediction_mean_std(
    dataset_bundle: DatasetBundle,
    predictor_bundle: PredictorBundle,
) -> dict[str, float]:
    """Derive the legacy prediction-statistics dictionary."""
    train_costs = getattr(
        getattr(dataset_bundle.train_loader_adversarial, "dataset", None),
        "costs",
        np.asarray([], dtype=float),
    )
    stats = {
        "test_mean": float(np.mean(dataset_bundle.testing_costs)),
        "train_mean": _safe_nanmean(train_costs),
        "intd_mean": float(np.mean(dataset_bundle.interdiction_costs)),
        "test_std": float(np.std(dataset_bundle.testing_costs)),
        "train_std": _safe_nanstd(train_costs),
        "intd_std": float(np.std(dataset_bundle.interdiction_costs)),
    }

    for family, legacy_prefix in _PREDICTION_STAT_KEYS.items():
        mean, std = _predictor_prediction_stats(
            getattr(predictor_bundle, family),
            dataset_bundle.testing_features,
        )
        stats[f"{legacy_prefix}_mean"] = mean
        stats[f"{legacy_prefix}_std"] = std

    return stats


def _build_metrics(
    run_cfg: SPNIRunConfig,
    all_data: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Recreate the legacy scalar metric dictionary."""
    metrics = {
        "metric_1": _safe_nanmean(all_data["o_p"]) - _safe_nanmean(all_data["o_s"]),
        "metric_2": _safe_nanmean(all_data["o_p"]) - _safe_nanmean(all_data["o_r"]),
        "metric_3": _safe_nanmean(all_data["o_p"]) - _safe_nanmean(all_data["o_a"]),
        "metric_4": _safe_nanmean(all_data["s_p"]) - _safe_nanmean(all_data["s_r"]),
        "metric_5": _safe_nanmean(all_data["s_p"]) - _safe_nanmean(all_data["s_a"]),
        "metric_6": _safe_nanmean(all_data["a_p"]) - _safe_nanmean(all_data["a_r"]),
        "metric_7": _safe_nanmean(all_data["a_p"]) - _safe_nanmean(all_data["a_a"]),
        "metric_8": (
            _safe_nanmean(all_data.get("a_p_a", np.asarray([], dtype=float)))
            - _safe_nanmean(all_data["s_a"])
            if run_cfg.compute_wrong_asym_intd
            and "a_p_a" in all_data
            else None
        ),
        "asym_nan_rows_oracle": int(np.isnan(all_data["a_o"]).sum()),
        "asym_nan_rows_po": int(np.isnan(all_data["a_p"]).sum()),
        "asym_nan_rows_spo": int(np.isnan(all_data["a_s"]).sum()),
        "asym_nan_rows_rand_spo": int(np.isnan(all_data["a_r"]).sum()),
        "asym_nan_rows_adv_spo": int(np.isnan(all_data["a_a"]).sum()),
    }
    return metrics


def _build_table_1(all_data: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Build the legacy Table 1 summary payload."""
    table_1: dict[str, float] = {}
    for family, prefix in _TABLE_1_PREFIX_MAP.items():
        no_key = next(key for key, value in _UNINTERDICTED_KEY_MAP.items()
                      if value == family)
        sym_key = next(key for key, value in _SYMMETRIC_KEY_MAP.items()
                       if value == family)
        asym_key = next(key for key, value in _ASYMMETRIC_EST_KEY_MAP.items()
                        if value == family)
        table_1[f"t1_{prefix}_n_mean"] = _safe_nanmean(all_data[no_key])
        table_1[f"t1_{prefix}_s_mean"] = _safe_nanmean(all_data[sym_key])
        table_1[f"t1_{prefix}_s_std"] = _safe_nanstd(all_data[sym_key])
        table_1[f"t1_{prefix}_a_mean"] = _safe_nanmean(all_data[asym_key])
        table_1[f"t1_{prefix}_a_std"] = _safe_nanstd(all_data[asym_key])
    return table_1


def _build_table_2(
    run_cfg: SPNIRunConfig,
    all_data: Mapping[str, np.ndarray],
) -> dict[str, float]:
    """Build the legacy Table 2 summary payload when enabled."""
    if not run_cfg.compute_wrong_asym_intd:
        return {}

    table_2: dict[str, float] = {}
    for prefix, (first_key, second_key) in _TABLE_2_GROUPS.items():
        table_2[f"t2_{prefix}_s_mean"] = _safe_nanmean(all_data[first_key])
        table_2[f"t2_{prefix}_s_std"] = _safe_nanstd(all_data[first_key])
        table_2[f"t2_{prefix}_a_mean"] = _safe_nanmean(all_data[second_key])
        table_2[f"t2_{prefix}_a_std"] = _safe_nanstd(all_data[second_key])
    return table_2


def _to_python_value(value: Any) -> Any:
    """Convert NumPy scalar values into plain Python values for rows."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _result_rows(
    result: SimulationResult,
    *,
    simulation_index: int,
) -> list[dict[str, Any]]:
    """Build flat CSV-style rows for one simulation result."""
    all_data = to_legacy_all_data(result.summary_bundle)
    if not all_data:
        return []

    num_samples = _array_length(next(iter(all_data.values())))
    rows: list[dict[str, Any]] = []
    for sample_index in range(num_samples):
        row: dict[str, Any] = {
            "simulation_index": simulation_index,
            "sample_index": sample_index,
        }
        for key, values in all_data.items():
            row[key] = _to_python_value(values[sample_index])
        rows.append(row)
    return rows


def _combine_all_data(simulations: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate per-run `all_data` mappings into one combined export map."""
    if not simulations:
        return {}

    keys = set()
    for simulation in simulations:
        keys.update(simulation.keys())

    combined: dict[str, np.ndarray] = {}
    for key in sorted(keys):
        arrays = [
            _as_float_array(simulation[key])
            for simulation in simulations
            if key in simulation
        ]
        combined[key] = np.concatenate(arrays) if arrays else np.asarray([], dtype=float)
    return combined


def _compute_percentage_increases_from_samples(
    all_data: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Compute safe per-sample percentage increases for one combined export."""
    return {
        "no_intd_p": _safe_percentage(all_data["o_p"], all_data["o_o"]),
        "no_intd_s": _safe_percentage(all_data["o_s"], all_data["o_o"]),
        "no_intd_r": _safe_percentage(all_data["o_r"], all_data["o_o"]),
        "no_intd_a": _safe_percentage(all_data["o_a"], all_data["o_o"]),
        "sym_intd_p": _safe_percentage(all_data["s_p"], all_data["s_o"]),
        "sym_intd_s": _safe_percentage(all_data["s_s"], all_data["s_o"]),
        "sym_intd_r": _safe_percentage(all_data["s_r"], all_data["s_o"]),
        "sym_intd_a": _safe_percentage(all_data["s_a"], all_data["s_o"]),
        "asym_intd_p": _safe_percentage(all_data["a_p"], all_data["a_o"]),
        "asym_intd_s": _safe_percentage(all_data["a_s"], all_data["a_o"]),
        "asym_intd_r": _safe_percentage(all_data["a_r"], all_data["a_o"]),
        "asym_intd_a": _safe_percentage(all_data["a_a"], all_data["a_o"]),
    }


def _compute_percentage_increases_from_simulations(
    simulations: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Compute safe per-simulation aggregate percentage increases."""
    calculations: dict[str, list[float]] = {
        "no_intd_p": [],
        "no_intd_s": [],
        "no_intd_r": [],
        "no_intd_a": [],
        "sym_intd_p": [],
        "sym_intd_s": [],
        "sym_intd_r": [],
        "sym_intd_a": [],
        "asym_intd_p": [],
        "asym_intd_s": [],
        "asym_intd_r": [],
        "asym_intd_a": [],
    }

    for sim in simulations:
        calculations["no_intd_p"].append(_safe_percentage_sum(sim["o_p"], sim["o_o"]))
        calculations["no_intd_s"].append(_safe_percentage_sum(sim["o_s"], sim["o_o"]))
        calculations["no_intd_r"].append(_safe_percentage_sum(sim["o_r"], sim["o_o"]))
        calculations["no_intd_a"].append(_safe_percentage_sum(sim["o_a"], sim["o_o"]))
        calculations["sym_intd_p"].append(_safe_percentage_sum(sim["s_p"], sim["s_o"]))
        calculations["sym_intd_s"].append(_safe_percentage_sum(sim["s_s"], sim["s_o"]))
        calculations["sym_intd_r"].append(_safe_percentage_sum(sim["s_r"], sim["s_o"]))
        calculations["sym_intd_a"].append(_safe_percentage_sum(sim["s_a"], sim["s_o"]))
        calculations["asym_intd_p"].append(_safe_percentage_sum(sim["a_p"], sim["a_o"]))
        calculations["asym_intd_s"].append(_safe_percentage_sum(sim["a_s"], sim["a_o"]))
        calculations["asym_intd_r"].append(_safe_percentage_sum(sim["a_r"], sim["a_o"]))
        calculations["asym_intd_a"].append(_safe_percentage_sum(sim["a_a"], sim["a_o"]))

    return {
        key: np.asarray(values, dtype=float)
        for key, values in calculations.items()
    }


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
    all_data = _extract_all_data(dataset_bundle, evaluation_bundle)
    prediction_mean_std = _build_prediction_mean_std(
        dataset_bundle,
        predictor_bundle,
    )
    metrics = _build_metrics(run_cfg, all_data)
    table_1 = _build_table_1(all_data)
    table_2 = _build_table_2(run_cfg, all_data)

    return SummaryBundle(
        prediction_mean_std=prediction_mean_std,
        metrics=metrics,
        table_1=table_1,
        table_2=table_2,
        all_data=all_data,
        diagnostics={
            "legacy_all_data_keys": sorted(all_data),
            "wrong_model_enabled": bool(run_cfg.compute_wrong_asym_intd),
        },
    )


def to_legacy_all_data(summary_bundle: SummaryBundle) -> dict:
    """Expose the current `all_data` structure for compatibility.

    Future implementation responsibilities:
    - preserve current analysis-script expectations during migration
    - keep legacy column names in one module rather than many scripts
    """
    return {
        key: _as_float_array(values)
        for key, values in summary_bundle.all_data.items()
    }


def flatten_result_rows(result: SimulationResult) -> list[dict]:
    """Flatten one simulation result into CSV-ready row dictionaries.

    Future implementation responsibilities:
    - keep simulation and sample indices stable
    - preserve explicit missing-value semantics
    """
    simulation_index = int(result.diagnostics.get("simulation_index", 0))
    return _result_rows(result, simulation_index=simulation_index)


def aggregate_sweep_results(results: list[SimulationResult]) -> dict:
    """Aggregate multiple run results into a sweep-level summary.

    Future implementation responsibilities:
    - compute per-run and per-sample aggregate views
    - expose the exact data needed by later plotting and analysis code
    """
    simulations = [
        to_legacy_all_data(result.summary_bundle)
        for result in results
    ]
    combined_all_data = _combine_all_data(simulations)
    rows: list[dict[str, Any]] = []
    for simulation_index, result in enumerate(results):
        rows.extend(_result_rows(result, simulation_index=simulation_index))

    metrics: dict[str, np.ndarray] = {}
    table_1: dict[str, np.ndarray] = {}
    table_2: dict[str, np.ndarray] = {}
    for result in results:
        for key, value in result.summary_bundle.metrics.items():
            metrics.setdefault(key, []).append(value)
        for key, value in result.summary_bundle.table_1.items():
            table_1.setdefault(key, []).append(value)
        for key, value in result.summary_bundle.table_2.items():
            table_2.setdefault(key, []).append(value)

    return {
        "num_runs": len(results),
        "all_data": combined_all_data,
        "rows": rows,
        "metrics": {
            key: np.asarray(values, dtype=object)
            for key, values in metrics.items()
        },
        "table_1": {
            key: np.asarray(values, dtype=float)
            for key, values in table_1.items()
        },
        "table_2": {
            key: np.asarray(values, dtype=float)
            for key, values in table_2.items()
        },
        "percentage_increases": {
            "samples": (
                _compute_percentage_increases_from_samples(combined_all_data)
                if combined_all_data else {}
            ),
            "simulations": _compute_percentage_increases_from_simulations(
                simulations
            ),
        },
    }
