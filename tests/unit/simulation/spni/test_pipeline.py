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


def _percentage_increases(offset: float) -> dict[str, dict[str, np.ndarray]]:
    """Return deterministic percentage arrays for scenario-sweep tests."""
    return {
        "simulations": {
            "no_intd_p": np.array([offset + 1.0], dtype=float),
            "no_intd_s": np.array([offset + 2.0], dtype=float),
            "no_intd_r": np.array([offset + 3.0], dtype=float),
            "no_intd_a": np.array([offset + 4.0], dtype=float),
            "sym_intd_p": np.array([offset + 5.0], dtype=float),
            "sym_intd_s": np.array([offset + 6.0], dtype=float),
            "sym_intd_r": np.array([offset + 7.0], dtype=float),
            "sym_intd_a": np.array([offset + 8.0], dtype=float),
            "asym_intd_p": np.array([offset + 9.0], dtype=float),
            "asym_intd_s": np.array([offset + 10.0], dtype=float),
            "asym_intd_r": np.array([offset + 11.0], dtype=float),
            "asym_intd_a": np.array([offset + 12.0], dtype=float),
        },
        "samples": {
            "no_intd_p": np.array([offset + 13.0, offset + 14.0], dtype=float),
            "no_intd_s": np.array([offset + 15.0, offset + 16.0], dtype=float),
            "no_intd_r": np.array([offset + 17.0, offset + 18.0], dtype=float),
            "no_intd_a": np.array([offset + 19.0, offset + 20.0], dtype=float),
            "sym_intd_p": np.array([offset + 21.0, offset + 22.0], dtype=float),
            "sym_intd_s": np.array([offset + 23.0, offset + 24.0], dtype=float),
            "sym_intd_r": np.array([offset + 25.0, offset + 26.0], dtype=float),
            "sym_intd_a": np.array([offset + 27.0, offset + 28.0], dtype=float),
            "asym_intd_p": np.array([offset + 29.0, offset + 30.0], dtype=float),
            "asym_intd_s": np.array([offset + 31.0, offset + 32.0], dtype=float),
            "asym_intd_r": np.array([offset + 33.0, offset + 34.0], dtype=float),
            "asym_intd_a": np.array([offset + 35.0, offset + 36.0], dtype=float),
        },
    }


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
    persisted_calls: list[SweepResult] = []
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
                "a_p_o": np.array([float(cfg.seed)], dtype=float),
                "a_s_o": np.array([float(cfg.seed)], dtype=float),
                "a_r_o": np.array([float(cfg.seed)], dtype=float),
                "a_a_o": np.array([float(cfg.seed)], dtype=float),
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
    monkeypatch.setattr(
        pipeline_module,
        "persist_sweep_outputs",
        lambda sweep_result, **kwargs: (
            persisted_calls.append(sweep_result) or SimpleNamespace(
                results_path="results.csv",
                sample_boxplot_path="sample_boxplot.png",
                simulation_boxplot_path="simulation_boxplot.png",
            )
        ),
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
    assert persisted_calls == [result], \
        "run_seed_sweep should persist outputs by default."
    assert result.diagnostics["present_results"] is True, \
        "run_seed_sweep should record that presentation was enabled."
    assert result.diagnostics["legacy_output_path"] == "results.csv", \
        "run_seed_sweep should record the saved CSV path."
    pass


def test_spni_pipeline_run_seed_sweep_skips_persistence_when_disabled(
    monkeypatch,
    run_cfg,
):
    """Verify that explicit persistence opt-out suppresses saved outputs."""
    # Arrange one deterministic sweep and a persistence sentinel.
    seed_bundles = [
        SeedBundle(sweep_seed=31, random_seed=41, intd_seed=51, loader_seed=61),
    ]
    persisted_calls: list[SweepResult] = []

    monkeypatch.setattr(
        pipeline_module,
        "derive_seed_sweep",
        lambda cfg, num_seeds: seed_bundles,
    )
    monkeypatch.setattr(
        pipeline_module,
        "run_single_simulation",
        lambda cfg: SimulationResult(
            run_config=cfg,
            seed_bundle=SeedBundle(
                sweep_seed=cfg.seed,
                random_seed=cfg.random_seed,
                intd_seed=cfg.intd_seed,
                loader_seed=cfg.loader_seed,
            ),
            graph_bundle=GraphBundle(
                graph=SimpleNamespace(name="graph"),
                opt_model=SimpleNamespace(name="opt-model"),
                graph_kind="synthetic",
            ),
            dataset_bundle=_dataset_bundle("unit"),
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
            summary_bundle=SummaryBundle(
                prediction_mean_std={},
                metrics={},
                table_1={},
                table_2={},
                all_data={
                    "o_o": np.array([1.0], dtype=float),
                    "o_p": np.array([2.0], dtype=float),
                    "o_s": np.array([3.0], dtype=float),
                    "o_r": np.array([4.0], dtype=float),
                    "o_a": np.array([5.0], dtype=float),
                    "s_o": np.array([1.0], dtype=float),
                    "s_p": np.array([2.0], dtype=float),
                    "s_s": np.array([3.0], dtype=float),
                    "s_r": np.array([4.0], dtype=float),
                    "s_a": np.array([5.0], dtype=float),
                    "a_o": np.array([1.0], dtype=float),
                    "a_p": np.array([2.0], dtype=float),
                    "a_s": np.array([3.0], dtype=float),
                    "a_r": np.array([4.0], dtype=float),
                    "a_a": np.array([5.0], dtype=float),
                    "a_p_o": np.array([1.0], dtype=float),
                    "a_s_o": np.array([1.0], dtype=float),
                    "a_r_o": np.array([1.0], dtype=float),
                    "a_a_o": np.array([1.0], dtype=float),
                },
            ),
        )
    )
    monkeypatch.setattr(
        pipeline_module,
        "persist_sweep_outputs",
        lambda sweep_result, **kwargs: persisted_calls.append(sweep_result),
    )

    # Act by disabling persistence on the typed seed sweep.
    result = pipeline_module.run_seed_sweep(
        run_cfg,
        num_seeds=1,
        present_results=False,
    )

    # Assert that no storage side effects were triggered.
    assert persisted_calls == [], \
        "run_seed_sweep should skip persistence when disabled explicitly."
    assert result.diagnostics["present_results"] is False, \
        "run_seed_sweep should record when presentation is disabled."
    assert result.diagnostics["legacy_output_path"] is None, \
        "run_seed_sweep should leave the CSV path unset when skipped."
    pass


#####################################
### test run_scenario_sweep(...) ###
#####################################


def test_spni_pipeline_run_scenario_sweep_collects_plot_ready_stats(
    monkeypatch,
):
    """Verify that scenario sweeps delegate per scenario and map stats."""
    # Arrange a compact base config and a recording sweep stub.
    base_cfg = SimpleNamespace(num_seeds=4, num_scenarios=9, budget=1)
    calls: list[dict[str, object]] = []
    persisted_calls: list[dict[str, object]] = []

    def _fake_run_seed_sweep(
        cfg,
        *,
        num_seeds,
        present_results,
        figure_directory=None,
        **options,
    ):
        calls.append(
            {
                "cfg": cfg,
                "num_seeds": num_seeds,
                "present_results": present_results,
                "figure_directory": figure_directory,
                "options": options,
            }
        )
        return SweepResult(
            run_config=cfg,
            results=[],
            aggregated_summary={
                "percentage_increases": _percentage_increases(
                    float(cfg.num_scenarios)
                ),
            },
            diagnostics={"num_runs": num_seeds},
        )

    monkeypatch.setattr(
        pipeline_module,
        "run_seed_sweep",
        _fake_run_seed_sweep,
    )
    monkeypatch.setattr(
        pipeline_module,
        "persist_scenario_sweep_outputs",
        lambda **kwargs: (
            persisted_calls.append(kwargs) or SimpleNamespace(
                simulation_plot_path="scenario-sim.png",
                asym_simulation_plot_path="scenario-asym-sim.png",
                sample_plot_path="scenario-sample.png",
                asym_sample_plot_path="scenario-asym-sample.png",
            )
        ),
    )

    # Act by running the scenario sweep over two counts.
    result = pipeline_module.run_scenario_sweep(
        base_cfg,
        scenarios=[2, 5],
        num_seeds=3,
        compute_asym_intd=False,
        present_results=True,
    )

    # Assert that each scenario delegated to a non-persisting seed sweep.
    assert [call["cfg"].num_scenarios for call in calls] == [2, 5], \
        "run_scenario_sweep should override num_scenarios per sweep item."
    assert all(call["num_seeds"] == 3 for call in calls), \
        "run_scenario_sweep should forward the requested seed count."
    assert all(call["present_results"] is False for call in calls), \
        "run_scenario_sweep should suppress seed-sweep persistence."
    assert all(call["figure_directory"] is None for call in calls), \
        "run_scenario_sweep should keep inner figure directories unset."
    assert all(
        call["options"] == {"compute_asym_intd": False}
        for call in calls
    ), "run_scenario_sweep should forward run options unchanged."
    assert len(persisted_calls) == 1, \
        "run_scenario_sweep should persist one outer scenario summary."
    assert persisted_calls[0]["scenarios"] == [2, 5], \
        "run_scenario_sweep should persist the swept scenario counts."
    assert persisted_calls[0]["num_seeds"] == 3, \
        "run_scenario_sweep should persist the resolved seed count."
    assert persisted_calls[0]["figure_directory"] is None, \
        "run_scenario_sweep should use the default figure directory."
    assert persisted_calls[0]["run_cfg"] is not base_cfg, \
        "run_scenario_sweep should persist against a copied config."
    assert persisted_calls[0]["sim_stats"] is result["sim_stats"], \
        "run_scenario_sweep should persist the computed simulation stats."
    assert persisted_calls[0]["sample_stats"] is result["sample_stats"], \
        "run_scenario_sweep should persist the computed sample stats."
    assert result["sim_stats"][2]["unintd"]["PO"] == [3.0], \
        "run_scenario_sweep should map simulation percentages by scenario."
    assert result["sim_stats"][5]["asym"]["A-DFL"] == [17.0], \
        "run_scenario_sweep should map asymmetric simulation stats."
    assert result["sample_stats"][2]["intd"]["DFL"] == [25.0, 26.0], \
        "run_scenario_sweep should map per-sample symmetric stats."
    assert result["sample_stats"][5]["asym"]["R-DFL"] == [38.0, 39.0], \
        "run_scenario_sweep should map per-sample asymmetric stats."
    assert result["diagnostics"]["scenario_counts"] == [2, 5], \
        "run_scenario_sweep should record the swept scenario counts."
    assert result["diagnostics"]["present_results"] is True, \
        "run_scenario_sweep should record when presentation is enabled."
    assert result["diagnostics"]["simulation_plot_path"] == \
        "scenario-sim.png", \
        "run_scenario_sweep should record the symmetric plot path."
    assert result["diagnostics"]["asym_sample_plot_path"] == \
        "scenario-asym-sample.png", \
        "run_scenario_sweep should record the asymmetric sample plot path."
    assert base_cfg.num_scenarios == 9, \
        "run_scenario_sweep should not mutate the caller-owned config."
    pass


def test_spni_pipeline_run_scenario_sweep_records_result_presentation_paths(
    monkeypatch,
):
    """Verify that scenario sweeps record output paths when presentation runs."""
    # Arrange a compact base config and a recording sweep stub.
    base_cfg = SimpleNamespace(num_seeds=4, num_scenarios=9, budget=1)
    expected_plot_dir = "scenario-figures"
    calls: list[dict[str, object]] = []
    persisted_calls: list[dict[str, object]] = []

    def _fake_run_seed_sweep(
        cfg,
        *,
        num_seeds,
        present_results,
        figure_directory=None,
        **options,
    ):
        calls.append(
            {
                "cfg": cfg,
                "num_seeds": num_seeds,
                "present_results": present_results,
                "figure_directory": figure_directory,
                "options": options,
            }
        )
        return SweepResult(
            run_config=cfg,
            results=[],
            aggregated_summary={
                "percentage_increases": {
                    "simulations": {
                        "no_intd_p": [1.0],
                        "no_intd_s": [2.0],
                        "no_intd_r": [3.0],
                        "no_intd_a": [4.0],
                        "sym_intd_p": [5.0],
                        "sym_intd_s": [6.0],
                        "sym_intd_r": [7.0],
                        "sym_intd_a": [8.0],
                        "asym_intd_p": [9.0],
                        "asym_intd_s": [10.0],
                        "asym_intd_r": [11.0],
                        "asym_intd_a": [12.0],
                    },
                    "samples": {
                        "no_intd_p": [13.0],
                        "no_intd_s": [14.0],
                        "no_intd_r": [15.0],
                        "no_intd_a": [16.0],
                        "sym_intd_p": [17.0],
                        "sym_intd_s": [18.0],
                        "sym_intd_r": [19.0],
                        "sym_intd_a": [20.0],
                        "asym_intd_p": [21.0],
                        "asym_intd_s": [22.0],
                        "asym_intd_r": [23.0],
                        "asym_intd_a": [24.0],
                    },
                }
            },
            diagnostics={"num_runs": num_seeds},
        )

    monkeypatch.setattr(
        pipeline_module,
        "run_seed_sweep",
        _fake_run_seed_sweep,
    )
    monkeypatch.setattr(
        pipeline_module,
        "persist_scenario_sweep_outputs",
        lambda **kwargs: (
            persisted_calls.append(kwargs) or SimpleNamespace(
                simulation_plot_path=(
                    "scenario-figures/custom_by_simulation.png"
                ),
                asym_simulation_plot_path=(
                    "scenario-figures/custom_asym_by_simulation.png"
                ),
                sample_plot_path="scenario-figures/custom_by_sample.png",
                asym_sample_plot_path=(
                    "scenario-figures/custom_asym_by_sample.png"
                ),
            )
        ),
    )

    # Act by running the scenario sweep with presentation enabled.
    result = pipeline_module.run_scenario_sweep(
        base_cfg,
        scenarios=[2],
        num_seeds=3,
        present_results=True,
        figure_directory=expected_plot_dir,
    )

    # Assert that the presentation branch recorded saved plot locations.
    assert calls[0]["present_results"] is False, \
        "run_scenario_sweep should still suppress inner seed persistence."
    assert calls[0]["figure_directory"] is None, \
        "run_scenario_sweep should not pass figure directories to seed sweeps."
    assert len(persisted_calls) == 1, \
        "run_scenario_sweep should persist one set of scenario figures."
    assert persisted_calls[0]["scenarios"] == [2], \
        "run_scenario_sweep should persist the requested scenario list."
    assert persisted_calls[0]["num_seeds"] == 3, \
        "run_scenario_sweep should persist the requested seed count."
    assert persisted_calls[0]["figure_directory"] == expected_plot_dir, \
        "run_scenario_sweep should forward the outer figure directory."
    assert persisted_calls[0]["run_cfg"] is not base_cfg, \
        "run_scenario_sweep should persist against a copied config."
    assert persisted_calls[0]["sim_stats"] is result["sim_stats"], \
        "run_scenario_sweep should persist the computed simulation stats."
    assert persisted_calls[0]["sample_stats"] is result["sample_stats"], \
        "run_scenario_sweep should persist the computed sample stats."
    assert result["diagnostics"]["present_results"] is True, \
        "run_scenario_sweep should record that presentation was enabled."
    assert result["diagnostics"]["simulation_plot_path"] == (
        "scenario-figures/custom_by_simulation.png"
    ), "run_scenario_sweep should record the simulation plot path."
    assert result["diagnostics"]["asym_simulation_plot_path"] == (
        "scenario-figures/custom_asym_by_simulation.png"
    ), "run_scenario_sweep should record the asymmetric plot path."
    assert result["diagnostics"]["sample_plot_path"] == (
        "scenario-figures/custom_by_sample.png"
    ), "run_scenario_sweep should record the sample plot path."
    assert result["diagnostics"]["asym_sample_plot_path"] == (
        "scenario-figures/custom_asym_by_sample.png"
    ), "run_scenario_sweep should record the asymmetric sample plot path."
    pass


####################
### test main(...) ###
####################


def test_spni_pipeline_main_dispatches_seed_sweep_with_cfg_overrides(
    monkeypatch,
):
    """Verify that main dispatches seed sweeps with config overrides."""
    # Arrange a seed-sweep runner stub and a mutable base config.
    base_cfg = SimpleNamespace(num_seeds=3, budget=1, grid_size=(2, 2))
    recorded: dict[str, object] = {}
    expected_result = SimpleNamespace(kind="seed-sweep")

    def _fake_run_seed_sweep(cfg, *, num_seeds, **options):
        recorded["cfg"] = cfg
        recorded["num_seeds"] = num_seeds
        recorded["options"] = options
        return expected_result

    monkeypatch.setattr(
        pipeline_module,
        "run_seed_sweep",
        _fake_run_seed_sweep,
    )

    # Act by running the convenience entrypoint in seed-sweep mode.
    result = pipeline_module.main(
        mode="seed_sweep",
        cfg=base_cfg,
        num_seeds=7,
        budget=9,
        compute_wrong_asym_intd=True,
    )

    # Assert that the dispatch preserved immutability and forwarded overrides.
    assert result is expected_result, \
        "main should return the result from run_seed_sweep."
    assert recorded["num_seeds"] == 7, \
        "main should forward the explicit num_seeds override."
    assert recorded["options"] == {"compute_wrong_asym_intd": True}, \
        "main should forward run options separately from config overrides."
    assert recorded["cfg"] is not base_cfg, \
        "main should deep-copy the caller's base config before mutation."
    assert recorded["cfg"].budget == 9, \
        "main should apply config overrides to the copied config."
    assert base_cfg.budget == 1, \
        "main should not mutate the caller-owned base config."
    pass


def test_spni_pipeline_main_dispatches_single_run_mode(monkeypatch):
    """Verify that main can dispatch directly to single-run execution."""
    # Arrange a single-run stub and a small config override.
    recorded: dict[str, object] = {}
    expected_result = SimpleNamespace(kind="single-run")

    def _fake_run_single_simulation(cfg, **options):
        recorded["cfg"] = cfg
        recorded["options"] = options
        return expected_result

    monkeypatch.setattr(
        pipeline_module,
        "run_single_simulation",
        _fake_run_single_simulation,
    )

    # Act by running the convenience entrypoint in single-run mode.
    result = pipeline_module.main(
        mode="single",
        cfg=SimpleNamespace(num_seeds=4, deg=2),
        deg=5,
        compute_asym_intd=False,
    )

    # Assert that single-run dispatch keeps num_seeds out of the call.
    assert result is expected_result, \
        "main should return the result from run_single_simulation."
    assert recorded["cfg"].deg == 5, \
        "main should apply config overrides before single-run dispatch."
    assert recorded["options"] == {"compute_asym_intd": False}, \
        "main should forward only run options to single-run execution."
    pass


def test_spni_pipeline_main_dispatches_scenario_sweep_mode(monkeypatch):
    """Verify that main can dispatch directly to scenario-sweep execution."""
    # Arrange a scenario-sweep stub and a mutable base config.
    base_cfg = SimpleNamespace(num_seeds=3, num_scenarios=4, budget=1)
    recorded: dict[str, object] = {}
    expected_result = {"diagnostics": {"scenario_counts": [2, 3]}}

    def _fake_run_scenario_sweep(cfg, *, scenarios, num_seeds, **options):
        recorded["cfg"] = cfg
        recorded["scenarios"] = scenarios
        recorded["num_seeds"] = num_seeds
        recorded["options"] = options
        return expected_result

    monkeypatch.setattr(
        pipeline_module,
        "run_scenario_sweep",
        _fake_run_scenario_sweep,
    )

    # Act by running the convenience entrypoint in scenario-sweep mode.
    result = pipeline_module.main(
        mode="scenario_sweep",
        cfg=base_cfg,
        scenarios=[2, 3],
        num_seeds=7,
        budget=9,
        compute_wrong_asym_intd=True,
        present_results=True,
    )

    # Assert that main forwards scenarios and drops seed-only presentation.
    assert result is expected_result, \
        "main should return the result from run_scenario_sweep."
    assert recorded["scenarios"] == [2, 3], \
        "main should forward explicit scenario counts unchanged."
    assert recorded["num_seeds"] == 7, \
        "main should forward the explicit num_seeds override."
    assert recorded["options"] == {
        "compute_wrong_asym_intd": True,
        "present_results": True,
    }, \
        "main should forward only supported run options to scenario sweeps."
    assert recorded["cfg"] is not base_cfg, \
        "main should deep-copy the caller's config before dispatch."
    assert recorded["cfg"].budget == 9, \
        "main should apply config overrides to the copied config."
    assert base_cfg.budget == 1, \
        "main should not mutate the caller-owned base config."
    pass


def test_spni_pipeline_main_rejects_unknown_mode():
    """Verify that main rejects unsupported top-level modes."""
    # Act and assert that unsupported modes fail clearly.
    with pytest.raises(ValueError, match="Unsupported SPNI run mode"):
        pipeline_module.main(mode="future_sweep")
    pass


###################
### test cli(...) ###
###################


def test_spni_pipeline_cli_parses_overrides_and_dispatches(monkeypatch):
    """Verify that cli parses one-line overrides before calling main."""
    # Arrange a main stub and silence the terminal print.
    recorded: dict[str, object] = {}
    expected_result = SimpleNamespace(diagnostics={"num_runs": 2})

    def _fake_main(**kwargs):
        recorded["kwargs"] = kwargs
        return expected_result

    monkeypatch.setattr(pipeline_module, "main", _fake_main)
    monkeypatch.setattr(
        pipeline_module,
        "print",
        lambda *args, **kwargs: None,
        raising=False,
    )

    # Act by executing the CLI with typed override values.
    result = pipeline_module.cli(
        [
            "--mode",
            "seed_sweep",
            "--num-seeds",
            "2",
            "--compute-wrong-asym-intd",
            "--set",
            "budget=5",
            "--set",
            "grid_size=(4, 6)",
        ]
    )

    # Assert that the parsed values are forwarded with the right types.
    assert result is expected_result, \
        "cli should return the result from main."
    assert recorded["kwargs"]["mode"] == "seed_sweep", \
        "cli should forward the requested mode."
    assert recorded["kwargs"]["num_seeds"] == 2, \
        "cli should parse num_seeds as an integer."
    assert recorded["kwargs"]["compute_wrong_asym_intd"] is True, \
        "cli should parse boolean flags for wrong-model evaluation."
    assert recorded["kwargs"]["budget"] == 5, \
        "cli should parse integer config overrides."
    assert recorded["kwargs"]["grid_size"] == (4, 6), \
        "cli should parse tuple config overrides with literal_eval."
    pass


def test_spni_pipeline_cli_parses_scenario_sweep_arguments(monkeypatch):
    """Verify that cli parses scenario-sweep arguments before dispatch."""
    # Arrange a main stub and silence the terminal print.
    recorded: dict[str, object] = {}
    expected_result = {"diagnostics": {"scenario_counts": [2, 3, 5]}}

    def _fake_main(**kwargs):
        recorded["kwargs"] = kwargs
        return expected_result

    monkeypatch.setattr(pipeline_module, "main", _fake_main)
    monkeypatch.setattr(
        pipeline_module,
        "print",
        lambda *args, **kwargs: None,
        raising=False,
    )

    # Act by executing the CLI in scenario-sweep mode.
    result = pipeline_module.cli(
        [
            "--mode",
            "scenario_sweep",
            "--scenarios",
            "2,3,5",
            "--num-seeds",
            "4",
            "--compute-asym-intd",
            "--set",
            "budget=5",
        ]
    )

    # Assert that the parsed values are forwarded with the right types.
    assert result is expected_result, \
        "cli should return the result from main for scenario sweeps."
    assert recorded["kwargs"]["mode"] == "scenario_sweep", \
        "cli should forward the requested scenario-sweep mode."
    assert recorded["kwargs"]["scenarios"] == [2, 3, 5], \
        "cli should parse comma-separated scenario counts as integers."
    assert recorded["kwargs"]["num_seeds"] == 4, \
        "cli should parse the scenario-sweep seed count as an integer."
    assert recorded["kwargs"]["compute_asym_intd"] is True, \
        "cli should parse boolean flags for scenario sweeps as well."
    assert recorded["kwargs"]["budget"] == 5, \
        "cli should keep config overrides available in scenario-sweep mode."
    pass
