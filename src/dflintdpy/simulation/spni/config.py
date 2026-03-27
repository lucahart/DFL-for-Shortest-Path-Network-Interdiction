"""Configuration utilities for the SPNI orchestration layer.

This module owns the small amount of logic needed before the simulation
pipeline can do any expensive work:
- reading legacy config objects safely
- validating orchestration-relevant fields
- normalizing optional runtime flags
- making seed and cache behavior explicit

It does not build graphs, train models, or evaluate simulations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CachePolicy:
    """Explicit cache behavior for one simulation run.

    The orchestration layer uses this dataclass to carry cache intent across
    stages without touching the filesystem directly.
    """

    replace_data: bool = False
    replace_intd_adv: bool = False
    replace_intd_rnd: bool = False
    replace_pred: bool = False
    replace_result: bool = False
    replace_fig: bool = False
    archive_replaced: bool = True


@dataclass(frozen=True)
class SeedBundle:
    """Deterministic seed assignment for one simulation run.

    This makes the sweep seed and its derived per-stage seeds explicit so
    reruns, logging, and debugging stay straightforward.
    """

    sweep_seed: int
    random_seed: int
    intd_seed: int
    loader_seed: int


@dataclass(frozen=True)
class SPNIRunConfig:
    """Normalized orchestration config for one SPNI simulation run.

    The rest of the orchestration layer should use this dataclass instead of
    reaching back into a mutable legacy `HP` object.
    """

    base_cfg: Any
    grid_size: tuple[int, int]
    num_features: int
    num_train_samples: int
    num_val_samples: int
    num_test_samples: int
    batch_size: int
    budget: int
    num_scenarios: int
    deg: int
    noise_width: float
    benders_max_count: int
    benders_eps: float
    lsd: float
    seed: int
    random_seed: int
    intd_seed: int
    loader_seed: int
    pred_model: str | None
    po_epochs: int
    spo_epochs: int
    po_lr: float
    spo_lr: float
    compute_asym_intd: bool = True
    compute_wrong_asym_intd: bool = False
    load_real_world_graph: str | None = None
    cache_policy: CachePolicy = field(default_factory=CachePolicy)
    metadata: dict[str, Any] = field(default_factory=dict)


def _read_cfg_value(base_cfg: Any, key: str, default: Any = None) -> Any:
    """Read one key from a config-like object without mutating it.

    Lookup order:
    1. `cfg.get(key, default)` when available
    2. `mapping[key]` / `mapping.get(...)` for mapping-like objects
    3. `getattr(cfg, key, default)` for plain attribute containers
    """

    getter = getattr(base_cfg, "get", None)
    if callable(getter):
        return getter(key, default)
    if isinstance(base_cfg, Mapping):
        return base_cfg.get(key, default)
    return getattr(base_cfg, key, default)


def _require_cfg_value(base_cfg: Any, key: str) -> Any:
    """Return one required config value or raise a precise error."""

    value = _read_cfg_value(base_cfg, key, None)
    if value is None:
        raise ValueError(f"Missing required config field: {key}")
    return value


def _coerce_int(value: Any, key: str, *, minimum: int | None = None) -> int:
    """Coerce one numeric field to `int` and enforce an optional minimum."""

    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer, not a boolean.")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be coercible to int.") from exc
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{key} must be >= {minimum}.")
    return normalized


def _coerce_float(
    value: Any,
    key: str,
    *,
    minimum: float | None = None,
) -> float:
    """Coerce one numeric field to `float` and enforce an optional minimum."""

    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be coercible to float.") from exc
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{key} must be >= {minimum}.")
    return normalized


def _normalize_grid_size(value: Any) -> tuple[int, int]:
    """Normalize `grid_size` into a validated `(rows, cols)` tuple."""

    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("grid_size must be a length-2 tuple or list.")
    rows = _coerce_int(value[0], "grid_size[0]", minimum=1)
    cols = _coerce_int(value[1], "grid_size[1]", minimum=1)
    return (rows, cols)


def _normalize_pred_model(value: Any) -> str | None:
    """Normalize the optional predictor-model label."""

    if value is None:
        return None
    model = str(value).strip()
    if not model:
        raise ValueError("pred_model must be a non-empty string when set.")
    return model


def _normalize_cache_policy(
    policy: CachePolicy | Mapping[str, Any] | None,
) -> CachePolicy:
    """Normalize one cache-policy input into the canonical dataclass."""

    if policy is None:
        return CachePolicy()
    if isinstance(policy, CachePolicy):
        return policy
    if isinstance(policy, Mapping):
        return CachePolicy(**dict(policy))
    raise TypeError("cache_policy must be a CachePolicy, mapping, or None.")


def _seed_triplet_from_sweep_seed(seed: int) -> tuple[int, int, int]:
    """Match the legacy sweep convention for per-run seed triplets.

    This uses a local `RandomState` so it does not disturb NumPy's global RNG
    state used elsewhere in the process.
    """

    rng = np.random.RandomState(seed)
    values = rng.randint(0, 150, 3).tolist()
    return int(values[0]), int(values[1]), int(values[2])


def _resolve_sweep_start_seed(run_cfg: SPNIRunConfig) -> int:
    """Resolve the first sweep seed for `derive_seed_sweep`.

    If the normalized config carries the legacy `seed_sweep_offset`, that
    offset defines the sweep start. Otherwise the current `run_cfg.seed`
    remains the fallback start value.
    """

    start_seed = run_cfg.metadata.get("legacy_seed_sweep_offset", run_cfg.seed)
    return _coerce_int(start_seed, "seed_sweep_start")


def build_run_config(
    base_cfg: Any,
    *,
    compute_asym_intd: bool = True,
    compute_wrong_asym_intd: bool = False,
    load_real_world_graph: str | None = None,
    cache_policy: CachePolicy | Mapping[str, Any] | None = None,
) -> SPNIRunConfig:
    """Normalize a legacy config object into `SPNIRunConfig`.

    This function reads one config-like object, validates the fields the SPNI
    orchestration layer actually uses, and returns a frozen dataclass that
    later stages can consume without depending on ad-hoc `.get(...)` calls.

    It also captures legacy sweep metadata, optional runtime flags, and an
    explicit `CachePolicy` so later stages have one stable source of truth.
    """

    normalized_cache_policy = _normalize_cache_policy(cache_policy)
    metadata = dict(_read_cfg_value(base_cfg, "metadata", {}) or {})
    metadata.setdefault("source_type", type(base_cfg).__name__)

    num_seeds = _read_cfg_value(base_cfg, "num_seeds", None)
    if num_seeds is not None:
        metadata.setdefault(
            "legacy_num_seeds",
            _coerce_int(num_seeds, "num_seeds", minimum=1),
        )

    seed_sweep_offset = _read_cfg_value(base_cfg, "seed_sweep_offset", None)
    if seed_sweep_offset is not None:
        metadata.setdefault(
            "legacy_seed_sweep_offset",
            _coerce_int(seed_sweep_offset, "seed_sweep_offset"),
        )

    if load_real_world_graph is not None:
        load_real_world_graph = str(load_real_world_graph)

    return SPNIRunConfig(
        base_cfg=base_cfg,
        grid_size=_normalize_grid_size(_require_cfg_value(base_cfg, "grid_size")),
        num_features=_coerce_int(
            _require_cfg_value(base_cfg, "num_features"),
            "num_features",
            minimum=1,
        ),
        num_train_samples=_coerce_int(
            _require_cfg_value(base_cfg, "num_train_samples"),
            "num_train_samples",
            minimum=1,
        ),
        num_val_samples=_coerce_int(
            _require_cfg_value(base_cfg, "num_val_samples"),
            "num_val_samples",
            minimum=0,
        ),
        num_test_samples=_coerce_int(
            _require_cfg_value(base_cfg, "num_test_samples"),
            "num_test_samples",
            minimum=1,
        ),
        batch_size=_coerce_int(
            _require_cfg_value(base_cfg, "batch_size"),
            "batch_size",
            minimum=1,
        ),
        budget=_coerce_int(
            _require_cfg_value(base_cfg, "budget"),
            "budget",
            minimum=0,
        ),
        num_scenarios=_coerce_int(
            _require_cfg_value(base_cfg, "num_scenarios"),
            "num_scenarios",
            minimum=1,
        ),
        deg=_coerce_int(
            _require_cfg_value(base_cfg, "deg"),
            "deg",
            minimum=1,
        ),
        noise_width=_coerce_float(
            _require_cfg_value(base_cfg, "noise_width"),
            "noise_width",
            minimum=0.0,
        ),
        benders_max_count=_coerce_int(
            _require_cfg_value(base_cfg, "benders_max_count"),
            "benders_max_count",
            minimum=1,
        ),
        benders_eps=_coerce_float(
            _require_cfg_value(base_cfg, "benders_eps"),
            "benders_eps",
            minimum=0.0,
        ),
        lsd=_coerce_float(
            _require_cfg_value(base_cfg, "lsd"),
            "lsd",
            minimum=0.0,
        ),
        seed=_coerce_int(_require_cfg_value(base_cfg, "seed"), "seed"),
        random_seed=_coerce_int(
            _require_cfg_value(base_cfg, "random_seed"),
            "random_seed",
        ),
        intd_seed=_coerce_int(
            _require_cfg_value(base_cfg, "intd_seed"),
            "intd_seed",
        ),
        loader_seed=_coerce_int(
            _require_cfg_value(base_cfg, "loader_seed"),
            "loader_seed",
        ),
        pred_model=_normalize_pred_model(
            _read_cfg_value(base_cfg, "pred_model", None)
        ),
        po_epochs=_coerce_int(
            _require_cfg_value(base_cfg, "po_epochs"),
            "po_epochs",
            minimum=0,
        ),
        spo_epochs=_coerce_int(
            _require_cfg_value(base_cfg, "spo_epochs"),
            "spo_epochs",
            minimum=0,
        ),
        po_lr=_coerce_float(
            _require_cfg_value(base_cfg, "po_lr"),
            "po_lr",
            minimum=0.0,
        ),
        spo_lr=_coerce_float(
            _require_cfg_value(base_cfg, "spo_lr"),
            "spo_lr",
            minimum=0.0,
        ),
        compute_asym_intd=bool(compute_asym_intd),
        compute_wrong_asym_intd=bool(compute_wrong_asym_intd),
        load_real_world_graph=load_real_world_graph,
        cache_policy=normalized_cache_policy,
        metadata=metadata,
    )


def derive_seed_bundle(
    run_cfg: SPNIRunConfig,
    *,
    sweep_seed: int | None = None,
) -> SeedBundle:
    """Derive the exact seeds used by one run.

    Behavior:
    - with no override, preserve the explicit seed triplet already stored on
      the normalized run config
    - with `sweep_seed=...`, derive a fresh triplet using the legacy sweep
      convention used by the current simulator scripts

    This keeps single-run behavior compatible with the current `HP`-driven
    scripts while still making sweep-derived seeds deterministic.
    """

    if sweep_seed is None:
        return SeedBundle(
            sweep_seed=int(run_cfg.seed),
            random_seed=int(run_cfg.random_seed),
            intd_seed=int(run_cfg.intd_seed),
            loader_seed=int(run_cfg.loader_seed),
        )

    resolved_seed = _coerce_int(sweep_seed, "sweep_seed")
    random_seed, intd_seed, loader_seed = \
        _seed_triplet_from_sweep_seed(resolved_seed)
    return SeedBundle(
        sweep_seed=resolved_seed,
        random_seed=random_seed,
        intd_seed=intd_seed,
        loader_seed=loader_seed,
    )


def derive_seed_sweep(
    run_cfg: SPNIRunConfig,
    *,
    num_seeds: int,
) -> list[SeedBundle]:
    """Build the ordered seed bundles for a multi-run sweep.

    Sweep convention:
    - start from the legacy `seed_sweep_offset` when it is available in
      normalized metadata
    - otherwise start from `run_cfg.seed`
    - generate `num_seeds` consecutive sweep seeds in ascending order
    - derive one deterministic triplet per sweep seed

    The returned list is stable and validated for accidental duplicates so it
    can be used safely for aggregation and CSV row ordering.
    """

    total = _coerce_int(num_seeds, "num_seeds", minimum=1)
    start_seed = _resolve_sweep_start_seed(run_cfg)
    bundles = [
        derive_seed_bundle(run_cfg, sweep_seed=start_seed + offset)
        for offset in range(total)
    ]
    if len(set(bundles)) != len(bundles):
        raise ValueError("Seed sweep produced duplicate seed bundles.")
    return bundles


def describe_run(run_cfg: SPNIRunConfig) -> dict[str, Any]:
    """Return a compact debug-friendly summary of one run config.

    The description intentionally excludes the original `base_cfg` object and
    keeps every value JSON-serializable so it can be printed in logs, test
    failures, or saved into lightweight metadata files.
    """

    return {
        "grid_size": list(run_cfg.grid_size),
        "num_features": int(run_cfg.num_features),
        "num_train_samples": int(run_cfg.num_train_samples),
        "num_val_samples": int(run_cfg.num_val_samples),
        "num_test_samples": int(run_cfg.num_test_samples),
        "batch_size": int(run_cfg.batch_size),
        "budget": int(run_cfg.budget),
        "num_scenarios": int(run_cfg.num_scenarios),
        "deg": int(run_cfg.deg),
        "noise_width": float(run_cfg.noise_width),
        "benders_max_count": int(run_cfg.benders_max_count),
        "benders_eps": float(run_cfg.benders_eps),
        "lsd": float(run_cfg.lsd),
        "pred_model": run_cfg.pred_model,
        "po_epochs": int(run_cfg.po_epochs),
        "spo_epochs": int(run_cfg.spo_epochs),
        "po_lr": float(run_cfg.po_lr),
        "spo_lr": float(run_cfg.spo_lr),
        "compute_asym_intd": bool(run_cfg.compute_asym_intd),
        "compute_wrong_asym_intd": bool(run_cfg.compute_wrong_asym_intd),
        "load_real_world_graph": run_cfg.load_real_world_graph,
        "sweep_start_seed": _resolve_sweep_start_seed(run_cfg),
        "seed_bundle": asdict(derive_seed_bundle(run_cfg)),
        "cache_policy": asdict(run_cfg.cache_policy),
        "metadata": dict(run_cfg.metadata),
    }
