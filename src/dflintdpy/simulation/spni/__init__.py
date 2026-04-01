"""Canonical SPNI orchestration API.

This package is the typed façade for the SPNI workflow. New code should import
its entrypoints from here instead of reaching into the legacy script layer.

Export groups:
- configuration contracts such as :class:`SPNIRunConfig` and
  :class:`CachePolicy`
- stage artifact dataclasses passed between pipeline steps
- top-level orchestration functions for single runs and seed sweeps

Legacy script modules may still exist for compatibility, but they are expected
to delegate into this package rather than own orchestration themselves.
"""

from dflintdpy.simulation.spni.config import (
    CachePolicy,
    SPNIRunConfig,
    SeedBundle,
)
from dflintdpy.simulation.spni.pipeline import (
    cli,
    main,
    run_seed_sweep,
    run_single_simulation,
)
from dflintdpy.simulation.spni.storage import (
    SweepStoragePaths,
    persist_sweep_outputs,
)
from dflintdpy.simulation.spni.types import (
    DatasetBundle,
    EvaluationBundle,
    GraphBundle,
    InterdictionSampleBundle,
    PredictorBundle,
    SimulationArtifacts,
    SimulationResult,
    SummaryBundle,
    SweepResult,
    TrainingLogBundle,
)

__all__ = [
    # Config and run-shaping helpers.
    "CachePolicy",
    "DatasetBundle",
    "EvaluationBundle",
    "GraphBundle",
    "InterdictionSampleBundle",
    "PredictorBundle",
    "SPNIRunConfig",
    "SeedBundle",
    "SimulationArtifacts",
    "SimulationResult",
    "SummaryBundle",
    "SweepResult",
    "TrainingLogBundle",
    # Canonical orchestration entrypoints.
    "cli",
    "main",
    "run_seed_sweep",
    "run_single_simulation",
    "SweepStoragePaths",
    "persist_sweep_outputs",
]
