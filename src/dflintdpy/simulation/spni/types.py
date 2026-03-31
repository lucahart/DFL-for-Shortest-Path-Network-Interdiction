"""Typed artifacts passed between SPNI orchestration stages.

The refactor moves SPNI orchestration away from loosely-coupled script locals
and toward explicit objects with stable responsibilities. Each dataclass below
represents the hand-off between two adjacent stages of the pipeline.

Design goals:
- make pipeline inputs and outputs inspectable in tests
- keep compatibility data available without exposing legacy script internals
- preserve diagnostics close to the stage that produced them
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dflintdpy.simulation.spni.config import SPNIRunConfig, SeedBundle

DiagnosticMap = dict[str, Any]


@dataclass
class GraphBundle:
    """Graph-facing objects created for one simulation run.

    This bundle is produced by the build stage and consumed by later data,
    training, and evaluation stages. It intentionally keeps the raw graph and
    its solver model together so later stages do not need to reconstruct or
    re-import them.

    Responsibilities:
    - hold the graph instance and its optimization model
    - record whether the graph was synthetic or real-world
    - make graph provenance visible in debugging and tests
    """

    graph: Any
    opt_model: Any
    graph_kind: str
    graph_source: str | None = None
    diagnostics: DiagnosticMap = field(default_factory=dict)


@dataclass
class InterdictionSampleBundle:
    """Sample-aligned interdiction inputs prepared for one evaluation stage.

    The current pipeline mostly carries these pieces inside broader bundle
    dictionaries, but this dataclass remains available for future callers that
    want one explicit representation of evaluation samples plus their applied
    interdictions.

    Responsibilities:
    - keep evaluation samples grouped in one stable object
    - preserve optional sample indexing for row flattening and debugging
    - expose generation diagnostics without binding to solver behavior
    """

    features: Any
    costs: Any
    interdictions: Any
    sample_indices: Any | None = None
    diagnostics: DiagnosticMap = field(default_factory=dict)


@dataclass
class DatasetBundle:
    """All loaders and arrays required for training and evaluation.

    This is the output of the data stage. It contains both training-time
    loaders and evaluation-time arrays so the later stages can remain pure
    consumers of already-prepared data instead of re-deriving splits.

    Responsibilities:
    - carry the full training/evaluation data state for one run
    - make scenario variants explicit instead of implicit in ad hoc dicts
    - keep normalization and generator provenance accessible
    """

    train_loader_adversarial: Any
    val_loader_adversarial: Any
    train_loader_random: Any
    val_loader_random: Any
    train_loader_baseline: Any
    val_loader_baseline: Any
    testing_features: Any
    testing_costs: Any
    interdiction_features: Any
    interdiction_costs: Any
    normalization_constant: float
    data_generator_adversarial: Any | None = None
    data_generator_random: Any | None = None
    diagnostics: DiagnosticMap = field(default_factory=dict)


@dataclass
class TrainingLogBundle:
    """Training curves for one predictor family.

    The legacy trainer returns multiple parallel log sequences. This dataclass
    normalizes those values into one typed structure so later code can reason
    about them without depending on positional tuples.

    Responsibilities:
    - keep trainer outputs structured and inspectable
    - decouple raw logs from plotting and terminal formatting
    """

    train_loss: list[float]
    train_regret: list[float]
    val_loss: list[float] | None
    val_regret: list[float] | None


@dataclass
class PredictorBundle:
    """All trained or loaded predictors for one SPNI run.

    The training stage returns this bundle after it has either trained new
    models or loaded them from cache. The object keeps the predictor families
    and their logs together so evaluation and reporting do not need to infer
    which loader/cache path produced which model.

    Responsibilities:
    - provide one stable access point for every model family
    - keep training logs associated with the predictor identities
    """

    pfl: Any
    dfl: Any
    rdfl: Any
    adfl: Any
    logs: dict[str, TrainingLogBundle] = field(default_factory=dict)
    diagnostics: DiagnosticMap = field(default_factory=dict)


@dataclass
class EvaluationBundle:
    """Raw numerical outputs of the evaluation stage.

    Each field corresponds to one evaluation family. The stage functions keep
    their outputs in dict form because the legacy helper functions already
    return dict-like payloads, but this outer dataclass gives the pipeline a
    stable top-level contract.

    Responsibilities:
    - hold per-sample outputs before aggregation
    - preserve sample alignment across evaluation families
    - expose skipped or failed solve counts explicitly
    """

    uninterdicted: dict[str, Any]
    symmetric: dict[str, Any]
    asymmetric: dict[str, Any]
    wrong_model_asymmetry: dict[str, Any]
    diagnostics: DiagnosticMap = field(default_factory=dict)


@dataclass
class SummaryBundle:
    """Derived summary tables and legacy export payloads.

    The results stage converts raw arrays into downstream-facing structures
    such as summary tables, scalar metrics, and the legacy ``all_data`` export
    map expected by existing analysis code.

    Responsibilities:
    - provide the exact structures needed by current reporting code
    - isolate formatting and metric derivation from numerical evaluation
    """

    prediction_mean_std: dict[str, Any]
    metrics: dict[str, Any]
    table_1: dict[str, Any]
    table_2: dict[str, Any]
    all_data: dict[str, Any]
    diagnostics: DiagnosticMap = field(default_factory=dict)


@dataclass
class SimulationArtifacts:
    """Optional side-effect outputs produced by the pipeline.

    The current SPNI pipeline is intentionally side-effect light, but this
    container keeps room for persisted model paths, CSV exports, or figures
    without mixing those filesystem details into the numerical result objects.

    Responsibilities:
    - record where models, results, or figures were persisted
    - keep side effects separate from numerical results
    """

    predictor_paths: dict[str, str] = field(default_factory=dict)
    result_path: str | None = None
    figure_paths: dict[str, str] = field(default_factory=dict)
    diagnostics: DiagnosticMap = field(default_factory=dict)


@dataclass
class SimulationResult:
    """Top-level return type for one SPNI simulation run.

    This is the canonical output of ``run_single_simulation(...)``. It keeps
    the full stage-by-stage artifact chain available so debugging, reporting,
    and compatibility wrappers can all consume the same source object.

    Responsibilities:
    - hold every stage output required for debugging, analysis, and export
    - give later scripts one canonical object to consume
    """

    run_config: SPNIRunConfig
    seed_bundle: SeedBundle
    graph_bundle: GraphBundle
    dataset_bundle: DatasetBundle
    predictor_bundle: PredictorBundle
    evaluation_bundle: EvaluationBundle
    summary_bundle: SummaryBundle
    artifacts: SimulationArtifacts = field(default_factory=SimulationArtifacts)
    diagnostics: DiagnosticMap = field(default_factory=dict)


@dataclass
class SweepResult:
    """Top-level return type for a multi-seed SPNI sweep.

    This is the canonical output of ``run_seed_sweep(...)``. It preserves both
    the ordered per-run results and the aggregated summary derived across the
    sweep so callers can choose the level of detail they need.

    Responsibilities:
    - preserve the ordered list of single-run results
    - carry any aggregated summary derived across runs
    """

    run_config: SPNIRunConfig
    results: list[SimulationResult]
    aggregated_summary: dict[str, Any] = field(default_factory=dict)
    diagnostics: DiagnosticMap = field(default_factory=dict)
