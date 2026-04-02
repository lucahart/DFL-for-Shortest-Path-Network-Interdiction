from pathlib import Path
from types import SimpleNamespace

import numpy as np

from dflintdpy.simulation.spni.config import CachePolicy, SPNIRunConfig, SeedBundle
from dflintdpy.simulation.spni.storage import persist_sweep_outputs
from dflintdpy.simulation.spni.types import (
    DatasetBundle,
    EvaluationBundle,
    GraphBundle,
    PredictorBundle,
    SimulationResult,
    SummaryBundle,
    SweepResult,
)
from dflintdpy.utils.analyse_results import (
    combine_simulations,
    compute_percentage_increases_from_simulations,
)
from dflintdpy.utils.read_write_results import (
    load_results_from_csv,
    save_results_to_csv,
)


############################
### Helper functionality ###
############################


def _simulation_result(seed: int, offset: float) -> SimulationResult:
    """Return one compact simulation result for save-load-analyze tests."""
    all_data = {
        "o_o": np.array([10.0 + offset, 20.0 + offset], dtype=float),
        "o_p": np.array([12.0 + offset, 24.0 + offset], dtype=float),
        "o_s": np.array([11.0 + offset, 22.0 + offset], dtype=float),
        "o_r": np.array([13.0 + offset, 25.0 + offset], dtype=float),
        "o_a": np.array([14.0 + offset, 26.0 + offset], dtype=float),
        "s_o": np.array([15.0 + offset, 30.0 + offset], dtype=float),
        "s_p": np.array([18.0 + offset, 36.0 + offset], dtype=float),
        "s_s": np.array([17.0 + offset, 34.0 + offset], dtype=float),
        "s_r": np.array([19.0 + offset, 37.0 + offset], dtype=float),
        "s_a": np.array([16.0 + offset, 32.0 + offset], dtype=float),
        "a_o": np.array([30.0 + offset, 50.0 + offset], dtype=float),
        "a_p": np.array([25.0 + offset, 42.0 + offset], dtype=float),
        "a_s": np.array([24.0 + offset, 43.0 + offset], dtype=float),
        "a_r": np.array([26.0 + offset, 44.0 + offset], dtype=float),
        "a_a": np.array([23.0 + offset, 41.0 + offset], dtype=float),
        "a_p_o": np.array([20.0 + offset, 40.0 + offset], dtype=float),
        "a_s_o": np.array([19.0 + offset, 39.0 + offset], dtype=float),
        "a_r_o": np.array([21.0 + offset, 41.0 + offset], dtype=float),
        "a_a_o": np.array([18.0 + offset, 38.0 + offset], dtype=float),
        "a_s_p": np.array([1.0 + offset, 2.0 + offset], dtype=float),
    }
    summary_bundle = SummaryBundle(
        prediction_mean_std={"test_mean": 1.0 + offset},
        metrics={"metric_1": 2.0 + offset},
        table_1={"t1_o_n_mean": 3.0 + offset},
        table_2={"t2_p_s_mean": 4.0 + offset},
        all_data=all_data,
    )
    return SimulationResult(
        run_config=SPNIRunConfig(
            base_cfg=SimpleNamespace(label="base"),
            grid_size=(2, 2),
            num_features=2,
            num_train_samples=4,
            num_val_samples=1,
            num_test_samples=2,
            batch_size=2,
            budget=1,
            num_scenarios=2,
            deg=1,
            noise_width=0.1,
            benders_max_count=2,
            benders_eps=1e-4,
            lsd=1e-5,
            seed=seed,
            random_seed=seed + 1,
            intd_seed=seed + 2,
            loader_seed=seed + 3,
            pred_model="linear",
            po_epochs=1,
            spo_epochs=1,
            po_lr=1e-3,
            spo_lr=1e-3,
            compute_asym_intd=True,
            compute_wrong_asym_intd=True,
            load_real_world_graph=None,
            cache_policy=CachePolicy(),
            metadata={"source_type": "SimpleNamespace"},
        ),
        seed_bundle=SeedBundle(
            sweep_seed=seed,
            random_seed=seed + 1,
            intd_seed=seed + 2,
            loader_seed=seed + 3,
        ),
        graph_bundle=GraphBundle(
            graph=SimpleNamespace(name="graph"),
            opt_model=SimpleNamespace(name="opt-model"),
            graph_kind="synthetic",
        ),
        dataset_bundle=DatasetBundle(
            train_loader_adversarial=SimpleNamespace(label="train-adv"),
            val_loader_adversarial=SimpleNamespace(label="val-adv"),
            train_loader_random=SimpleNamespace(label="train-rand"),
            val_loader_random=SimpleNamespace(label="val-rand"),
            train_loader_baseline=SimpleNamespace(label="train-base"),
            val_loader_baseline=SimpleNamespace(label="val-base"),
            testing_features=np.array([[1.0], [2.0]], dtype=float),
            testing_costs=np.array([[3.0], [4.0]], dtype=float),
            interdiction_features=np.array([[5.0], [6.0]], dtype=float),
            interdiction_costs=np.array([[7.0], [8.0]], dtype=float),
            normalization_constant=1.0,
        ),
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


###############################
### test save-load-analyze ###
###############################


def test_utils_results_analysis_save_load_analyze_flow_works_for_typed_sweeps(
    tmp_path: Path,
):
    """Verify that typed sweep results can round-trip through CSV analysis."""
    # Arrange a tiny typed sweep result and an output path.
    results = [
        _simulation_result(10, 0.0),
        _simulation_result(11, 5.0),
    ]
    sweep_result = SweepResult(
        run_config=results[0].run_config,
        results=results,
        aggregated_summary={},
    )
    output_path = tmp_path / "results_train_4_valid_1_test_2_m_2_n_2_deg_1_noise_0.1_seeds_2.csv"

    # Act by saving typed results, loading the CSV back, and analyzing both.
    save_results_to_csv(sweep_result, output_path)
    loaded = load_results_from_csv(output_path)
    typed_percentages = compute_percentage_increases_from_simulations(
        sweep_result,
    )
    loaded_percentages = compute_percentage_increases_from_simulations(
        loaded,
    )
    typed_all_data = combine_simulations(sweep_result)
    loaded_all_data = combine_simulations(loaded)

    # Assert that the CSV workflow preserves analysis-relevant data.
    assert output_path.exists(), \
        "save_results_to_csv should write a CSV for typed sweep results."
    assert len(loaded) == 2, \
        "load_results_from_csv should preserve the number of simulations."
    np.testing.assert_allclose(
        typed_all_data["o_o"],
        loaded_all_data["o_o"],
        err_msg="The CSV round-trip should preserve oracle costs.",
    )
    np.testing.assert_allclose(
        typed_all_data["a_s_p"],
        loaded_all_data["a_s_p"],
        err_msg="The CSV round-trip should preserve wrong-model columns.",
    )
    np.testing.assert_allclose(
        typed_percentages["asym_intd_p"],
        loaded_percentages["asym_intd_p"],
        err_msg="Typed and loaded simulations should yield identical analysis.",
    )
    assert np.all(typed_percentages["asym_intd_p"] > 0.0), \
        "Asymmetric percentages should compare predictors to `a_p_o`, not `a_o`."
    pass


def test_utils_results_analysis_persist_sweep_outputs_saves_csv_and_figures(
    tmp_path: Path,
):
    """Verify that typed sweep persistence saves both CSV and figure outputs."""
    # Arrange one compact typed sweep and explicit output locations.
    results = [
        _simulation_result(10, 0.0),
        _simulation_result(11, 5.0),
    ]
    sweep_result = SweepResult(
        run_config=results[0].run_config,
        results=results,
        aggregated_summary={},
    )
    output_path = tmp_path / "results" / "typed_results.csv"
    figure_directory = tmp_path / "figures"

    # Act by persisting the typed sweep end-to-end.
    stored_paths = persist_sweep_outputs(
        sweep_result,
        output_path=output_path,
        figure_directory=figure_directory,
    )
    loaded = load_results_from_csv(stored_paths.results_path)

    # Assert that the storage pipeline created the expected artefacts.
    assert stored_paths.results_path == output_path, \
        "persist_sweep_outputs should honor an explicit CSV output path."
    assert stored_paths.results_path.exists(), \
        "persist_sweep_outputs should create the CSV results file."
    assert stored_paths.sample_boxplot_path.exists(), \
        "persist_sweep_outputs should save the sample boxplot."
    assert stored_paths.simulation_boxplot_path.exists(), \
        "persist_sweep_outputs should save the simulation boxplot."
    assert stored_paths.sample_boxplot_path.parent == figure_directory, \
        "persist_sweep_outputs should honor the figure directory override."
    assert stored_paths.simulation_boxplot_path.parent == figure_directory, \
        "persist_sweep_outputs should keep both figures together."
    assert len(loaded) == 2, \
        "persist_sweep_outputs should preserve the number of simulations."
    np.testing.assert_allclose(
        loaded[0]["o_o"],
        results[0].summary_bundle.all_data["o_o"],
        err_msg="Persisted CSV results should remain analysis-compatible.",
    )
    pass
