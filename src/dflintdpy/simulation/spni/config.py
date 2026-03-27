"""Configuration interfaces for the SPNI orchestration layer.

This module should become the single place where SPNI simulation runs are
normalized, validated, and described before any expensive work starts.

The design intent is to keep all orchestration-specific logic here:
- selecting optional run features
- validating required config fields
- deriving reproducible seed bundles
- carrying cache policy explicitly

It must not build graphs, train models, or evaluate simulations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CachePolicy:
    """Explicit cache behavior for one simulation run.

    This dataclass is meant to replace hidden global cache policy assumptions
    in the orchestration layer. It should describe only intent, not execute
    file I/O by itself.
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

    Every source of randomness used by the orchestration layer should be made
    explicit through this object so debugging and reruns are straightforward.
    """

    sweep_seed: int
    random_seed: int
    intd_seed: int
    loader_seed: int


@dataclass(frozen=True)
class SPNIRunConfig:
    """Normalized orchestration config for one SPNI simulation run.

    The future pipeline should accept either the legacy `HP` object or this
    dataclass. Once normalized, the rest of the orchestration layer should use
    only this dataclass so field access is explicit and stable.
    """

    base_cfg: Any
    grid_size: tuple[int, int]
    num_features: int
    num_train_samples: int
    num_val_samples: int
    num_test_samples: int
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
    compute_asym_intd_2: bool = False
    load_real_world_graph: str | None = None
    cache_policy: CachePolicy = field(default_factory=CachePolicy)
    metadata: dict[str, Any] = field(default_factory=dict)


def build_run_config(
    base_cfg: Any,
    *,
    compute_asym_intd: bool = True,
    compute_asym_intd_2: bool = False,
    load_real_world_graph: str | None = None,
    cache_policy: CachePolicy | None = None,
) -> SPNIRunConfig:
    """Normalize a legacy config object into `SPNIRunConfig`.

    Future implementation responsibilities:
    - validate all required fields up front
    - copy only orchestration-relevant values
    - attach optional runtime flags in one place
    - return a frozen config object used by every later stage
    """

    raise NotImplementedError("Specification scaffold only.")


def derive_seed_bundle(
    run_cfg: SPNIRunConfig,
    *,
    sweep_seed: int | None = None,
) -> SeedBundle:
    """Derive the exact seeds used by one run.

    Future implementation responsibilities:
    - keep seed derivation deterministic
    - document whether `sweep_seed` overrides or supplements `run_cfg.seed`
    - avoid mutating module-global RNG state unless explicitly requested
    """

    raise NotImplementedError("Specification scaffold only.")


def derive_seed_sweep(
    run_cfg: SPNIRunConfig,
    *,
    num_seeds: int,
) -> list[SeedBundle]:
    """Build the ordered seed bundles for a multi-run sweep.

    Future implementation responsibilities:
    - define the exact sweep indexing convention
    - ensure seed bundles are unique and reproducible
    - make the returned order stable for aggregation and CSV output
    """

    raise NotImplementedError("Specification scaffold only.")


def describe_run(run_cfg: SPNIRunConfig) -> dict[str, Any]:
    """Return a compact debug-friendly summary of one run config.

    Future implementation responsibilities:
    - expose the fields most useful in logs and test failure messages
    - avoid dumping large or unserializable objects
    """

    raise NotImplementedError("Specification scaffold only.")

