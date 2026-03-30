from types import SimpleNamespace

import numpy as np
import pytest

from dflintdpy.simulation.spni.config import CachePolicy, SPNIRunConfig, SeedBundle
from dflintdpy.simulation.spni.types import (
    DatasetBundle,
    EvaluationBundle,
    GraphBundle,
    PredictorBundle,
    SimulationResult,
    SummaryBundle,
    SweepResult,
)

import dflintdpy.simulation.spni.pipeline as pipeline_module


################
### Fixtures ###
################


@pytest.fixture
def run_cfg() -> SPNIRunConfig:
    """Return a compact run config for pipeline-stage tests."""
    return SPNIRunConfig(
        base_cfg=SimpleNamespace(label="base"),
        grid_size=(3, 4),
        num_features=3,
        num_train_samples=4,
        num_val_samples=2,
        num_test_samples=2,
        batch_size=5,
        budget=1,
        num_scenarios=3,
        deg=2,
        noise_width=0.25,
        benders_max_count=7,
        benders_eps=1e-4,
        lsd=1e-5,
        seed=11,
        random_seed=13,
        intd_seed=17,
        loader_seed=19,
        pred_model="linear",
        po_epochs=8,
        spo_epochs=9,
        po_lr=1e-3,
        spo_lr=2e-3,
        compute_asym_intd=True,
        compute_wrong_asym_intd=False,
        load_real_world_graph=None,
        cache_policy=CachePolicy(replace_pred=True),
        metadata={"source_type": "SimpleNamespace"},
    )


############################
### Helper functionality ###
############################


def _dataset_bundle(label: str) -> DatasetBundle:
    """Return a compact dataset bundle for pipeline tests."""
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


#######################################
### test run_single_simulation(...) ###
#######################################


def test_spni_pipeline_run_single_simulation_wires_stages_in_order(
    monkeypatch,
    run_cfg,
):
    """Verify that the single-run pipeline calls each stage in order."""
    # Arrange stage stubs that record call order and forwarded objects.
    calls: list[str] = []
    graph_bundle = GraphBundle(
        graph=SimpleNamespace(name="graph"),
        opt_model=SimpleNamespace(name="opt-model"),
        graph_kind="synthetic",
    )
    dataset_bundle = _dataset_bundle("unit")
    predictor_bundle = PredictorBundle(
        pfl=SimpleNamespace(label="pfl"),
        dfl=SimpleNamespace(label="dfl"),
        rdfl=SimpleNamespace(label="rdfl"),
        adfl=SimpleNamespace(label="adfl"),
    )
    evaluation_bundle = EvaluationBundle(
        uninterdicted={"diagnostics": {}},
        symmetric={"diagnostics": {}},
        asymmetric={"diagnostics": {}},
        wrong_model_asymmetry={"diagnostics": {}},
    )
    summary_bundle = SummaryBundle(
        prediction_mean_std={"test_mean": 1.0},
        metrics={"metric_1": 2.0},
        table_1={"rows": 1},
        table_2={},
        all_data={"o_o": np.array([1.0, 2.0], dtype=float)},
    )

    monkeypatch.setattr(
        pipeline_module,
        "derive_seed_bundle",
        lambda cfg: SeedBundle(
            sweep_seed=cfg.seed,
            random_seed=101,
            intd_seed=102,
            loader_seed=103,
        ),
    )

    def _fake_build_problem_bundle(cfg):
        calls.append("build")
        assert cfg.random_seed == 101, \
            "run_single_simulation should apply the derived seed bundle."
        return graph_bundle

    def _fake_assemble_dataset_bundle(cfg, graph):
        calls.append("data")
        assert graph is graph_bundle, \
            "Dataset assembly should receive the graph bundle unchanged."
        return dataset_bundle

    def _fake_train_all_predictors(cfg, graph, dataset):
        calls.append("train")
        assert dataset is dataset_bundle, \
            "Predictor training should receive the dataset bundle unchanged."
        return predictor_bundle

    def _fake_evaluate_all(cfg, graph, dataset, predictors):
        calls.append("evaluate")
        assert predictors is predictor_bundle, \
            "Evaluation should receive the predictor bundle unchanged."
        return evaluation_bundle

    def _fake_build_summary(cfg, dataset, predictors, evaluation):
        calls.append("results")
        assert evaluation is evaluation_bundle, \
            "Summary construction should receive evaluation outputs unchanged."
        return summary_bundle

    monkeypatch.setattr(pipeline_module, "build_problem_bundle", _fake_build_problem_bundle)
    monkeypatch.setattr(
        pipeline_module,
        "assemble_dataset_bundle",
        _fake_assemble_dataset_bundle,
    )
    monkeypatch.setattr(
        pipeline_module,
        "train_all_predictors",
        _fake_train_all_predictors,
    )
    monkeypatch.setattr(pipeline_module, "evaluate_all", _fake_evaluate_all)
    monkeypatch.setattr(pipeline_module, "build_summary", _fake_build_summary)

    # Act by running the full single-run pipeline.
    result = pipeline_module.run_single_simulation(run_cfg)

    # Assert that the result is fully populated and stage order is fixed.
    assert isinstance(result, SimulationResult), \
        "run_single_simulation should return a SimulationResult."
    assert calls == ["build", "data", "train", "evaluate", "results"], \
        "run_single_simulation should wire the stages in the documented order."
    assert result.graph_bundle is graph_bundle, \
        "run_single_simulation should preserve the graph bundle."
    assert result.summary_bundle is summary_bundle, \
        "run_single_simulation should preserve the summary bundle."
    assert result.seed_bundle.random_seed == 101, \
        "run_single_simulation should store the derived seed bundle."
    assert result.diagnostics["side_effects_enabled"] is False, \
        "run_single_simulation should keep side effects explicitly disabled."
    pass


#################################
### test run_seed_sweep(...) ###
#################################


def test_spni_pipeline_run_seed_sweep_uses_ordered_seed_bundles(
    monkeypatch,
    run_cfg,
):
    """Verify that the sweep pipeline reuses single-run execution in order."""
    # Arrange an ordered seed sweep and a recording single-run stub.
    calls: list[int] = []
    seed_bundles = [
        SeedBundle(sweep_seed=31, random_seed=41, intd_seed=51, loader_seed=61),
        SeedBundle(sweep_seed=32, random_seed=42, intd_seed=52, loader_seed=62),
    ]

    monkeypatch.setattr(
        pipeline_module,
        "derive_seed_sweep",
        lambda cfg, num_seeds: seed_bundles,
    )

    def _fake_run_single_simulation(cfg):
        calls.append(cfg.seed)
        summary_bundle = SummaryBundle(
            prediction_mean_std={"test_mean": float(cfg.seed)},
            metrics={"metric_1": float(cfg.seed)},
            table_1={"t1_o_n_mean": float(cfg.seed)},
            table_2={},
            all_data={
                "o_o": np.array([float(cfg.seed)], dtype=float),
                "o_p": np.array([float(cfg.seed) + 1.0], dtype=float),
                "o_s": np.array([float(cfg.seed) + 2.0], dtype=float),
                "o_r": np.array([float(cfg.seed) + 3.0], dtype=float),
                "o_a": np.array([float(cfg.seed) + 4.0], dtype=float),
                "s_o": np.array([float(cfg.seed)], dtype=float),
                "s_p": np.array([float(cfg.seed) + 1.0], dtype=float),
                "s_s": np.array([float(cfg.seed) + 2.0], dtype=float),
                "s_r": np.array([float(cfg.seed) + 3.0], dtype=float),
                "s_a": np.array([float(cfg.seed) + 4.0], dtype=float),
                "a_o": np.array([float(cfg.seed)], dtype=float),
                "a_p": np.array([float(cfg.seed) + 1.0], dtype=float),
                "a_s": np.array([float(cfg.seed) + 2.0], dtype=float),
                "a_r": np.array([float(cfg.seed) + 3.0], dtype=float),
                "a_a": np.array([float(cfg.seed) + 4.0], dtype=float),
            },
        )
        return SimulationResult(
            run_config=cfg,
            seed_bundle=SeedBundle(
                sweep_seed=cfg.seed,
                random_seed=cfg.random_seed,
                intd_seed=cfg.intd_seed,
                loader_seed=cfg.loader_seed,
            ),
            graph_bundle=GraphBundle(
                graph=SimpleNamespace(name=f"graph-{cfg.seed}"),
                opt_model=SimpleNamespace(name=f"opt-{cfg.seed}"),
                graph_kind="synthetic",
            ),
            dataset_bundle=_dataset_bundle(str(cfg.seed)),
            predictor_bundle=PredictorBundle(
                pfl=SimpleNamespace(label="pfl"),
                dfl=SimpleNamespace(label="dfl"),
                rdfl=SimpleNamespace(label="rdfl"),
                adfl=SimpleNamespace(label="adfl"),
            ),
            evaluation_bundle=EvaluationBundle(
                uninterdicted={},
                symmetric={},
                asymmetric={},
                wrong_model_asymmetry={},
            ),
            summary_bundle=summary_bundle,
        )

    monkeypatch.setattr(
        pipeline_module,
        "run_single_simulation",
        _fake_run_single_simulation,
    )

    # Act by running the multi-seed sweep.
    result = pipeline_module.run_seed_sweep(run_cfg, num_seeds=2)

    # Assert that the sweep preserves seed order and aggregates results.
    assert isinstance(result, SweepResult), \
        "run_seed_sweep should return a SweepResult."
    assert calls == [31, 32], \
        "run_seed_sweep should call run_single_simulation in sweep order."
    assert [item.seed_bundle.sweep_seed for item in result.results] == [31, 32], \
        "run_seed_sweep should preserve one SimulationResult per seed bundle."
    assert result.aggregated_summary["num_runs"] == 2, \
        "run_seed_sweep should aggregate the resulting run summaries."
    pass
