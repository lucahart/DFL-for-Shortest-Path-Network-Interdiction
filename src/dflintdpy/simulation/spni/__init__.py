"""SPNI orchestration scaffold.

The modules in this package are intentionally non-invasive scaffolding for a
future refactor. They document the intended interfaces and responsibilities of
the SPNI orchestration layer without changing the current scripts.
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
    PredictorBundle,
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
    "PredictorBundle",
    "SPNIRunConfig",
    "SeedBundle",
    "SimulationResult",
    "SummaryBundle",
    "SweepResult",
    "TrainingLogBundle",
    "run_seed_sweep",
    "run_single_simulation",
]

