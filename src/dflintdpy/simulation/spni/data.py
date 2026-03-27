"""Dataset assembly for SPNI simulations.

This module should own all orchestration around:
- base synthetic data generation
- train/val/test splitting
- adverse/random data creation
- baseline non-adverse loader creation

It must not train predictors or compute evaluation metrics.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from dflintdpy.data.adverse.adverse_data_generator import (
    SPNIAdverseDataGenerator,
)
from dflintdpy.data.adverse.adverse_dataset import AdvDataset
from dflintdpy.data.adverse.adverse_loader import AdvLoader
from dflintdpy.data.data_gen import gen_syn_data
from dflintdpy.simulation.spni.config import SPNIRunConfig
from dflintdpy.simulation.spni.types import DatasetBundle, GraphBundle
from dflintdpy.utils.read_write import CacheReplaceOptions


@dataclass(frozen=True)
class _BaseData:
    """Internal raw-data payload before normalization or splitting."""

    features: Any
    costs: Any
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _SplitData:
    """Internal split payload preserving the legacy train/val/test flow."""

    trainval_features: Any
    trainval_costs: Any
    test_features: Any
    test_costs: Any
    train_indices: Any
    val_indices: Any
    normalization_constant: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _TrainingDataViews:
    """Internal container for one SPNI loader family."""

    train_loader: Any
    val_loader: Any
    data_generator: Any
    diagnostics: dict[str, Any] = field(default_factory=dict)


class _LegacyConfigAdapter:
    """Expose normalized run-config values through the legacy `.get(...)` API."""

    def __init__(
        self,
        run_cfg: SPNIRunConfig,
        *,
        overrides: Mapping[str, Any] | None = None,
    ):
        self._run_cfg = run_cfg
        self._overrides = dict(overrides or {})

    def get(self, key: str, default: Any = None) -> Any:
        """Read one value using run-config fields before falling back."""
        if key in self._overrides:
            return self._overrides[key]
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
    """Translate the orchestration cache policy into legacy cache options."""

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


def _build_legacy_cfg(
    run_cfg: SPNIRunConfig,
    *,
    random_seed: int | None = None,
) -> _LegacyConfigAdapter:
    """Build a legacy-config adapter for current data helpers."""

    overrides: dict[str, Any] = {}
    if random_seed is not None:
        overrides["random_seed"] = int(random_seed)
    return _LegacyConfigAdapter(run_cfg, overrides=overrides)


def generate_base_data(run_cfg: SPNIRunConfig, graph_bundle: GraphBundle):
    """Generate or load the base feature/cost arrays for one run.

    Future implementation responsibilities:
    - call the existing synthetic-data helper
    - keep base data generation separate from splitting
    - return raw arrays in a small internal bundle or tuple
    """
    legacy_cfg = _build_legacy_cfg(run_cfg)
    features, costs = gen_syn_data(legacy_cfg, opt_model=graph_bundle.opt_model)
    return _BaseData(
        features=features,
        costs=costs,
        diagnostics={
            "num_samples": int(features.shape[0]),
            "num_features": int(features.shape[1]),
            "num_costs": int(costs.shape[1]),
            "random_seed": int(run_cfg.random_seed),
        },
    )


def split_base_data(run_cfg: SPNIRunConfig, features, costs):
    """Split base arrays into train/validation/test partitions.

    Future implementation responsibilities:
    - make split counts explicit
    - define and preserve sample ordering conventions
    - return a structured split result
    """
    normalization_constant = float(np.max(costs))
    normalized_costs = costs / normalization_constant

    trainval_features, test_features, trainval_costs, test_costs = \
        train_test_split(
            features,
            normalized_costs,
            test_size=run_cfg.num_test_samples,
            random_state=run_cfg.random_seed,
        )

    train_indices = np.arange(trainval_features.shape[0])
    train_indices, val_indices = train_test_split(
        train_indices,
        test_size=run_cfg.num_val_samples,
        random_state=run_cfg.random_seed,
    )

    return _SplitData(
        trainval_features=trainval_features,
        trainval_costs=trainval_costs,
        test_features=test_features,
        test_costs=test_costs,
        train_indices=train_indices,
        val_indices=val_indices,
        normalization_constant=normalization_constant,
        diagnostics={
            "train_count": int(train_indices.shape[0]),
            "val_count": int(val_indices.shape[0]),
            "test_count": int(test_features.shape[0]),
            "trainval_count": int(trainval_features.shape[0]),
        },
    )


def build_spni_training_data(
    run_cfg: SPNIRunConfig,
    graph_bundle: GraphBundle,
    split_data,
):
    """Create adverse and random SPNI training loaders.

    Future implementation responsibilities:
    - build adversarial and random scenario generators
    - build training and validation loaders for both variants
    - preserve the generator objects for debugging and cache inspection
    """
    legacy_cfg = _build_legacy_cfg(run_cfg)
    cache_options = _build_cache_options(run_cfg)
    loaders: dict[str, _TrainingDataViews] = {}

    for interdiction_policy in ("adversarial", "random"):
        data_generator = SPNIAdverseDataGenerator(
            legacy_cfg,
            graph_bundle.opt_model,
            budget=run_cfg.budget,
            normalization_constant=split_data.normalization_constant,
            num_scenarios=run_cfg.num_scenarios,
            interdiction_policy=interdiction_policy,
            cache_options=cache_options,
            gen_intd_seed=run_cfg.intd_seed,
            max_cnt=run_cfg.benders_max_count,
            eps=run_cfg.benders_eps,
        )
        features_all, costs_all, interdictions_all = data_generator.generate(
            split_data.trainval_features,
            split_data.trainval_costs,
            cfg=legacy_cfg,
        )

        train_dataset = AdvDataset(
            graph_bundle.opt_model,
            features_all[split_data.train_indices],
            costs_all[split_data.train_indices],
            interdictions_all[split_data.train_indices],
        )
        val_dataset = AdvDataset(
            graph_bundle.opt_model,
            features_all[split_data.val_indices],
            costs_all[split_data.val_indices],
            interdictions_all[split_data.val_indices],
        )
        train_loader = AdvLoader(
            train_dataset,
            batch_size=run_cfg.batch_size,
            seed=run_cfg.loader_seed,
            shuffle=True,
        )
        val_loader = AdvLoader(
            val_dataset,
            batch_size=run_cfg.batch_size,
            seed=run_cfg.loader_seed,
            shuffle=False,
        )
        loaders[interdiction_policy] = _TrainingDataViews(
            train_loader=train_loader,
            val_loader=val_loader,
            data_generator=data_generator,
            diagnostics={
                "interdiction_policy": interdiction_policy,
                "num_scenarios": int(run_cfg.num_scenarios),
                "train_count": int(split_data.train_indices.shape[0]),
                "val_count": int(split_data.val_indices.shape[0]),
            },
        )

    return loaders


def build_nonadverse_views(adverse_train_loader, adverse_val_loader):
    """Create baseline non-adverse loader views from adverse loaders.

    Future implementation responsibilities:
    - derive baseline loaders from scenario-zero data
    - preserve batch size and sampler configuration
    """
    return (
        adverse_train_loader.get_nonadverse_loader(),
        adverse_val_loader.get_nonadverse_loader(),
    )


def assemble_dataset_bundle(
    run_cfg: SPNIRunConfig,
    graph_bundle: GraphBundle,
) -> DatasetBundle:
    """Build the full typed dataset bundle for one SPNI run.

    Future implementation responsibilities:
    - normalize costs once and store the normalization constant
    - create all loader variants required by the training stage
    - create evaluation interdiction arrays for the comparison stage
    """
    base_data = generate_base_data(run_cfg, graph_bundle)
    split_data = split_base_data(run_cfg, base_data.features, base_data.costs)
    training_views = build_spni_training_data(run_cfg, graph_bundle, split_data)

    baseline_train_loader, baseline_val_loader = build_nonadverse_views(
        training_views["adversarial"].train_loader,
        training_views["adversarial"].val_loader,
    )

    interdiction_features, interdiction_costs = gen_syn_data(
        _build_legacy_cfg(run_cfg, random_seed=run_cfg.intd_seed),
        opt_model=graph_bundle.opt_model,
        seed=run_cfg.intd_seed,
    )
    evaluation_interdiction_costs = (
        interdiction_costs / split_data.normalization_constant
    )[: run_cfg.num_test_samples]
    evaluation_interdiction_features = interdiction_features[
        : run_cfg.num_test_samples
    ]

    return DatasetBundle(
        train_loader_adversarial=training_views["adversarial"].train_loader,
        val_loader_adversarial=training_views["adversarial"].val_loader,
        train_loader_random=training_views["random"].train_loader,
        val_loader_random=training_views["random"].val_loader,
        train_loader_baseline=baseline_train_loader,
        val_loader_baseline=baseline_val_loader,
        testing_features=split_data.test_features,
        testing_costs=split_data.test_costs,
        interdiction_features=evaluation_interdiction_features,
        interdiction_costs=evaluation_interdiction_costs,
        normalization_constant=split_data.normalization_constant,
        data_generator_adversarial=training_views["adversarial"].data_generator,
        data_generator_random=training_views["random"].data_generator,
        diagnostics={
            "base_data": base_data.diagnostics,
            "splits": split_data.diagnostics,
            "training_views": {
                "adversarial": training_views["adversarial"].diagnostics,
                "random": training_views["random"].diagnostics,
            },
            "evaluation_interdictions": {
                "count": int(evaluation_interdiction_features.shape[0]),
                "seed": int(run_cfg.intd_seed),
            },
        },
    )
