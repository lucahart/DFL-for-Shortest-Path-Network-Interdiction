"""Predictor training orchestration for SPNI simulations.

This module should coordinate the existing trainer helpers and predictor setup
functions. It should not own the trainer core or the predictor math itself.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import dflintdpy.scripts.setup as legacy_setup_module
from dflintdpy.simulation.spni.config import SPNIRunConfig
from dflintdpy.simulation.spni.types import (
    DatasetBundle,
    GraphBundle,
    PredictorBundle,
    TrainingLogBundle,
)
from dflintdpy.utils.read_write import CacheReplaceOptions

_PREDICTOR_CACHE_TAGS = {
    "pfl": "pfl",
    "dfl": "dfl",
    "rdfl": "rdfl",
    "adfl": "adfl",
}

_DFL_FAMILY_SPECS = {
    "dfl": {
        "train_loader_attr": "train_loader_baseline",
        "val_loader_attr": "val_loader_baseline",
        "setup_kwargs": {},
    },
    "rdfl": {
        "train_loader_attr": "train_loader_random",
        "val_loader_attr": "val_loader_random",
        "setup_kwargs": {"dfl_variant": "a-dfl"},
    },
    "adfl": {
        "train_loader_attr": "train_loader_adversarial",
        "val_loader_attr": "val_loader_adversarial",
        "setup_kwargs": {"dfl_variant": "a-dfl"},
    },
}


class _LegacyConfigAdapter:
    """Expose normalized run-config values through the legacy `.get(...)` API."""

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


def _build_cache_options(run_cfg: SPNIRunConfig) -> CacheReplaceOptions:
    """Translate the normalized cache policy into legacy cache options."""
    policy = run_cfg.cache_policy
    return CacheReplaceOptions(
        replace_data=policy.replace_data,
        replace_intd_rnd=policy.replace_intd_rnd,
        replace_intd_adv=policy.replace_intd_adv,
        replace_pred=policy.replace_pred,
        replace_result=policy.replace_result,
        replace_fig=policy.replace_fig,
        archive_replaced=policy.archive_replaced,
    )


def _build_training_data(train_loader: Any, val_loader: Any) -> dict[str, Any]:
    """Build the legacy training-data mapping expected by setup helpers."""
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
    }


def _normalize_log_values(values: Sequence[Any] | None) -> list[float] | None:
    """Convert trainer log sequences into plain Python float lists."""
    if values is None:
        return None
    return [float(value) for value in values]


def _build_log_bundle(
    fit_logs: tuple[
        Sequence[Any],
        Sequence[Any],
        Sequence[Any] | None,
        Sequence[Any] | None,
    ] | None,
) -> TrainingLogBundle:
    """Convert one captured trainer fit result into a typed log bundle."""
    if fit_logs is None:
        return TrainingLogBundle(
            train_loss=[],
            train_regret=[],
            val_loss=None,
            val_regret=None,
        )

    train_loss, train_regret, val_loss, val_regret = fit_logs
    return TrainingLogBundle(
        train_loss=_normalize_log_values(train_loss) or [],
        train_regret=_normalize_log_values(train_regret) or [],
        val_loss=_normalize_log_values(val_loss),
        val_regret=_normalize_log_values(val_regret),
    )


@contextmanager
def _capture_trainer_fit_logs(
    trainer_cls: type[Any],
) -> Iterator[dict[str, Any]]:
    """Record the legacy trainer's `fit(...)` return value once per call."""
    capture: dict[str, Any] = {"fit_logs": None}
    original_fit = trainer_cls.fit

    def _recording_fit(self, *args, **kwargs):
        result = original_fit(self, *args, **kwargs)
        capture["fit_logs"] = result
        return result

    trainer_cls.fit = _recording_fit
    try:
        yield capture
    finally:
        trainer_cls.fit = original_fit


def _run_with_fit_capture(
    trainer_cls: type[Any],
    runner,
) -> tuple[Any, TrainingLogBundle, bool]:
    """Run one legacy setup helper and capture trainer logs when available."""
    with _capture_trainer_fit_logs(trainer_cls) as capture:
        predictor = runner()
    fit_logs = capture["fit_logs"]
    return predictor, _build_log_bundle(fit_logs), fit_logs is not None


def train_pfl_predictor(
    run_cfg: SPNIRunConfig,
    graph_bundle: GraphBundle,
    dataset_bundle: DatasetBundle,
) -> tuple[Any, TrainingLogBundle]:
    """Train or load the PFL predictor.

    Future implementation responsibilities:
    - call the existing PFL setup helper
    - return the predictor plus its log bundle
    - keep cache-tag handling explicit and centralized
    """
    legacy_cfg = _LegacyConfigAdapter(run_cfg)
    cache_options = _build_cache_options(run_cfg)
    training_data = _build_training_data(
        dataset_bundle.train_loader_adversarial,
        dataset_bundle.val_loader_adversarial,
    )

    predictor, logs, _ = _run_with_fit_capture(
        legacy_setup_module.PFLTrainer,
        lambda: legacy_setup_module.setup_pfl_predictor(
            legacy_cfg,
            graph_bundle.graph,
            graph_bundle.opt_model,
            training_data,
            cache_tag=_PREDICTOR_CACHE_TAGS["pfl"],
            cache_options=cache_options,
            verbose=False,
        ),
    )
    return predictor, logs


def train_dfl_predictor(
    run_cfg: SPNIRunConfig,
    graph_bundle: GraphBundle,
    dataset_bundle: DatasetBundle,
    *,
    variant_name: str,
) -> tuple[Any, TrainingLogBundle]:
    """Train or load one DFL-family predictor.

    Future implementation responsibilities:
    - support baseline DFL, random adverse DFL, and adversarial DFL
    - keep the mapping from variant name to training data explicit
    - return the predictor plus its log bundle
    """
    if variant_name not in _DFL_FAMILY_SPECS:
        supported = ", ".join(sorted(_DFL_FAMILY_SPECS))
        raise ValueError(
            f"Unsupported DFL predictor family '{variant_name}'. "
            f"Expected one of: {supported}."
        )

    legacy_cfg = _LegacyConfigAdapter(run_cfg)
    cache_options = _build_cache_options(run_cfg)
    spec = _DFL_FAMILY_SPECS[variant_name]
    training_data = _build_training_data(
        getattr(dataset_bundle, spec["train_loader_attr"]),
        getattr(dataset_bundle, spec["val_loader_attr"]),
    )

    predictor, logs, _ = _run_with_fit_capture(
        legacy_setup_module.DFLTrainer,
        lambda: legacy_setup_module.setup_dfl_predictor(
            legacy_cfg,
            graph_bundle.graph,
            graph_bundle.opt_model,
            training_data,
            cache_tag=_PREDICTOR_CACHE_TAGS[variant_name],
            cache_options=cache_options,
            verbose=False,
            **spec["setup_kwargs"],
        ),
    )
    return predictor, logs


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
    pfl, pfl_logs = train_pfl_predictor(run_cfg, graph_bundle, dataset_bundle)
    dfl, dfl_logs = train_dfl_predictor(
        run_cfg,
        graph_bundle,
        dataset_bundle,
        variant_name="dfl",
    )
    rdfl, rdfl_logs = train_dfl_predictor(
        run_cfg,
        graph_bundle,
        dataset_bundle,
        variant_name="rdfl",
    )
    adfl, adfl_logs = train_dfl_predictor(
        run_cfg,
        graph_bundle,
        dataset_bundle,
        variant_name="adfl",
    )

    return PredictorBundle(
        pfl=pfl,
        dfl=dfl,
        rdfl=rdfl,
        adfl=adfl,
        logs={
            "pfl": pfl_logs,
            "dfl": dfl_logs,
            "rdfl": rdfl_logs,
            "adfl": adfl_logs,
        },
        diagnostics={
            "cache_tags": dict(_PREDICTOR_CACHE_TAGS),
            "dfl_family_loaders": {
                family: {
                    "train_loader_attr": spec["train_loader_attr"],
                    "val_loader_attr": spec["val_loader_attr"],
                }
                for family, spec in _DFL_FAMILY_SPECS.items()
            },
        },
    )
