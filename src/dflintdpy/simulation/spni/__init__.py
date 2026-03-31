"""Canonical SPNI orchestration API.

The package-level entrypoints exported here are the supported path for new
single-run and seed-sweep SPNI workflows. Legacy script modules may remain as
compatibility wrappers, but they should delegate into this package rather than
own orchestration themselves.
"""

from dflintdpy.simulation.spni.config import (
    CachePolicy,
    SPNIRunConfig,
    SeedBundle,
)
from dflintdpy.simulation.spni.pipeline import (
    run_seed_sweep,
    run_single_simulation,
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
    "run_seed_sweep",
    "run_single_simulation",
]
