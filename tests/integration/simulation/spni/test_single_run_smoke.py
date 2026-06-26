from types import SimpleNamespace

import numpy as np

from dflintdpy.simulation.spni.types import (
    DatasetBundle,
    EvaluationBundle,
    GraphBundle,
    PredictorBundle,
    SummaryBundle,
)

import dflintdpy.simulation.spni.pipeline as pipeline_module


############################
### Helper functionality ###
############################


def _dataset_bundle(label: str) -> DatasetBundle:
    """Return a compact dataset bundle for pipeline smoke tests."""
    array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    return DatasetBundle(
        train_loader_adversarial=SimpleNamespace(label=f"{label}-train-adv"),
        val_loader_adversarial=SimpleNamespace(label=f"{label}-val-adv"),
        train_loader_random=SimpleNamespace(label=f"{label}-train-rand"),
        val_loader_random=SimpleNamespace(label=f"{label}-val-rand"),
        train_loader_baseline=SimpleNamespace(label=f"{label}-train-base"),
        val_loader_baseline=SimpleNamespace(label=f"{label}-val-base"),
        testing_features=array,
        testing_costs=array,
        interdiction_features=array,
        interdiction_costs=array,
        normalization_constant=1.0,
    )


###############################
### test single-run smoke ###
###############################


def test_spni_pipeline_single_run_smoke_with_tiny_config(monkeypatch):
    """Verify that a tiny single-run pipeline completes end to end."""
    # Arrange a compact base config plus lightweight stage stubs.
    base_cfg = {
        "grid_size": (2, 2),
        "num_features": 2,
        "num_train_samples": 4,
        "num_val_samples": 1,
        "num_test_samples": 2,
        "batch_size": 2,
        "budget": 1,
        "num_scenarios": 2,
        "deg": 1,
        "noise_width": 0.1,
        "benders_max_count": 2,
        "benders_eps": 1e-4,
        "lsd": 1e-5,
        "seed": 5,
        "random_seed": 7,
        "intd_seed": 11,
        "loader_seed": 13,
        "pred_model": "linear",
        "pfl_epochs": 1,
        "dfl_epochs": 1,
        "pfl_lr": 1e-3,
        "dfl_lr": 1e-3,
    }

    monkeypatch.setattr(
        pipeline_module,
        "build_problem_bundle",
        lambda cfg: GraphBundle(
            graph=SimpleNamespace(name="graph"),
            opt_model=SimpleNamespace(name="opt-model"),
            graph_kind="synthetic",
        ),
    )
    monkeypatch.setattr(
        pipeline_module,
        "assemble_dataset_bundle",
        lambda cfg, graph: _dataset_bundle("smoke"),
    )
    monkeypatch.setattr(
        pipeline_module,
        "train_all_predictors",
        lambda cfg, graph, dataset: PredictorBundle(
            pfl=SimpleNamespace(label="pfl"),
            dfl=SimpleNamespace(label="dfl"),
            rdfl=SimpleNamespace(label="rdfl"),
            adfl=SimpleNamespace(label="adfl"),
        ),
    )
    monkeypatch.setattr(
        pipeline_module,
        "evaluate_all",
        lambda cfg, graph, dataset, predictors: EvaluationBundle(
            uninterdicted={"objectives": {"oracle": np.array([1.0, 2.0])}},
            symmetric={"objectives": {"oracle": np.array([3.0, 4.0])}},
            asymmetric={"diagnostics": {"failure_counts": {}}},
            wrong_model_asymmetry={"diagnostics": {"skipped": True}},
        ),
    )
    monkeypatch.setattr(
        pipeline_module,
        "build_summary",
        lambda cfg, dataset, predictors, evaluation: SummaryBundle(
            prediction_mean_std={"test_mean": 1.5},
            metrics={"metric_1": 0.1},
            table_1={"t1_o_n_mean": 1.5},
            table_2={},
            all_data={"o_o": np.array([1.0, 2.0], dtype=float)},
        ),
    )

    # Act by running the tiny single-run pipeline.
    result = pipeline_module.run_single_simulation(base_cfg)

    # Assert that the smoke result is populated and internally consistent.
    assert result.predictor_bundle.pfl.label == "pfl", \
        "The smoke run should include the PFL predictor family."
    assert result.predictor_bundle.rdfl.label == "rdfl", \
        "The smoke run should include the R-DFL predictor family."
    assert result.summary_bundle.metrics["metric_1"] == 0.1, \
        "The smoke run should produce a non-empty summary payload."
    assert result.seed_bundle.sweep_seed == 5, \
        "The smoke run should preserve the normalized single-run seed bundle."
    pass
