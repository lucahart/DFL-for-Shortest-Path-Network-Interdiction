"""Typed artifacts passed between SPNI orchestration stages.

This module defines the future public data contracts of the orchestration
layer. The aim is to replace the current pattern of passing nested dicts and
parallel arrays between scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dflintdpy.simulation.spni.config import SPNIRunConfig, SeedBundle


@dataclass
class GraphBundle:
    """Graph-facing objects created for one simulation run.

    Responsibilities:
    - hold the graph instance and its optimization model
    - record whether the graph was synthetic or real-world
    - make graph provenance visible in debugging and tests
    """

    graph: Any
    opt_model: Any
    graph_kind: str
    graph_source: str | None = None


@dataclass
class DatasetBundle:
    """All loaders and arrays required for training and evaluation.

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
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingLogBundle:
    """Training curves for one predictor family.

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

    Responsibilities:
    - provide one stable access point for every model family
    - keep training logs associated with the predictor identities
    """

    pfl: Any
    dfl: Any
    rdfl: Any
    adfl: Any
    logs: dict[str, TrainingLogBundle] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationBundle:
    """Raw numerical outputs of the evaluation stage.

    Responsibilities:
    - hold per-sample outputs before aggregation
    - preserve sample alignment across evaluation families
    - expose skipped or failed solve counts explicitly
    """

    uninterdicted: dict[str, Any]
    symmetric: dict[str, Any]
    asymmetric: dict[str, Any]
    wrong_model_asymmetry: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class SummaryBundle:
    """Derived summary tables and legacy export payloads.

    Responsibilities:
    - provide the exact structures needed by current reporting code
    - isolate formatting and metric derivation from numerical evaluation
    """

    prediction_mean_std: dict[str, Any]
    metrics: dict[str, Any]
    table_1: dict[str, Any]
    table_2: dict[str, Any]
    all_data: dict[str, Any]


@dataclass
class SimulationArtifacts:
    """Optional side-effect outputs produced by the pipeline.

    Responsibilities:
    - record where models, results, or figures were persisted
    - keep side effects separate from numerical results
    """

    predictor_paths: dict[str, str] = field(default_factory=dict)
    result_path: str | None = None
    figure_paths: dict[str, str] = field(default_factory=dict)


@dataclass
class SimulationResult:
    """Top-level return type for one SPNI simulation run.

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


@dataclass
class SweepResult:
    """Top-level return type for a multi-seed SPNI sweep.

    Responsibilities:
    - preserve the ordered list of single-run results
    - carry any aggregated summary derived across runs
    """

    run_config: SPNIRunConfig
    results: list[SimulationResult]
    aggregated_summary: dict[str, Any] = field(default_factory=dict)

