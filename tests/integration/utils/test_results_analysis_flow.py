from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend import Legend

import dflintdpy.simulation.spni.storage as storage_module
from dflintdpy.simulation.spni.config import CachePolicy, SPNIRunConfig, SeedBundle
from dflintdpy.simulation.spni.storage import (
    persist_scenario_sweep_outputs,
    persist_sweep_outputs,
    replot_saved_sweep_outputs,
)
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
    create_boxplots_from_calculations,
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


def _real_world_simulation_result(
    seed: int,
    offset: float,
    graph_path: str,
) -> SimulationResult:
    """Return one compact result configured with a real-world graph path."""
    result = _simulation_result(seed, offset)
    run_config = replace(
        result.run_config,
        load_real_world_graph=graph_path,
    )
    graph_bundle = replace(
        result.graph_bundle,
        graph_kind="real_world",
        graph_source=graph_path,
    )
    return replace(
        result,
        run_config=run_config,
        graph_bundle=graph_bundle,
    )


def _scenario_sweep_stats(
    scenarios: list[int],
) -> dict[int, dict[str, dict[str, list[float]]]]:
    """Return compact plot-ready scenario-sweep stats for storage tests."""
    conditions = ("unintd", "intd", "asym")
    methods = ("PO", "DFL", "R-DFL", "A-DFL")
    return {
        scenario: {
            condition: {
                method: [float(scenario)]
                for method in methods
            }
            for condition in conditions
        }
        for scenario in scenarios
    }


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


def test_utils_results_analysis_persist_sweep_outputs_defaults_to_figures_dir(
    tmp_path: Path,
    monkeypatch,
):
    """Verify that default sweep figures are saved under figures/."""
    # Arrange a private project root and an explicit CSV path.
    monkeypatch.setattr(storage_module, "_project_root", lambda: tmp_path)
    result = _simulation_result(10, 0.0)
    sweep_result = SweepResult(
        run_config=result.run_config,
        results=[result],
        aggregated_summary={},
    )
    output_path = tmp_path / "custom-results" / "typed_results.csv"
    expected_figure_directory = tmp_path / "figures"

    # Act by persisting without a figure-directory override.
    stored_paths = persist_sweep_outputs(
        sweep_result,
        output_path=output_path,
    )

    # Assert that figures use the default project figures directory.
    assert stored_paths.results_path == output_path, \
        "persist_sweep_outputs should still honor the explicit CSV path."
    assert stored_paths.sample_boxplot_path.parent == expected_figure_directory, \
        "persist_sweep_outputs should default figures to project figures/."
    assert stored_paths.simulation_boxplot_path.parent == (
        expected_figure_directory
    ), "persist_sweep_outputs should keep default figures together."
    assert stored_paths.sample_boxplot_path.exists(), \
        "persist_sweep_outputs should save the default sample boxplot."
    assert stored_paths.simulation_boxplot_path.exists(), \
        "persist_sweep_outputs should save the default simulation boxplot."
    pass


def test_utils_results_analysis_persist_sweep_outputs_marks_real_world_graphs(
    tmp_path: Path,
    monkeypatch,
):
    """Verify that default real-world sweep outputs include the graph marker."""
    # Arrange a real-world sweep that relies on default result naming.
    monkeypatch.setattr(storage_module, "_project_root", lambda: tmp_path)
    result = _real_world_simulation_result(
        10,
        0.0,
        "real_world_spni_data/Town Level Arcs.csv",
    )
    sweep_result = SweepResult(
        run_config=result.run_config,
        results=[result],
        aggregated_summary={},
    )

    # Act by persisting without explicit result or figure paths.
    stored_paths = persist_sweep_outputs(sweep_result)

    # Assert that both CSV and figure names carry the real-world graph marker.
    expected_stem = (
        "results_train_4_valid_1_test_2_m_2_n_2_deg_1_noise_0.1"
        "_real_world_town_level_arcs_seeds_1"
    )
    assert stored_paths.results_path.name == f"{expected_stem}.csv", \
        "Real-world result CSVs should include the graph filename marker."
    assert stored_paths.sample_boxplot_path.name == (
        f"{expected_stem}_boxplot.png"
    ), "Real-world sample boxplots should match the marked CSV stem."
    assert stored_paths.simulation_boxplot_path.name == (
        f"{expected_stem}_boxplot_sims.png"
    ), "Real-world simulation boxplots should match the marked CSV stem."
    pass


def test_utils_results_analysis_scenario_sweep_outputs_mark_real_world_graphs(
    tmp_path: Path,
):
    """Verify that scenario-sweep figures include the real-world graph marker."""
    # Arrange a real-world config and compact scenario-sweep plot inputs.
    result = _real_world_simulation_result(
        10,
        0.0,
        "real_world_data/transportation_networks/Anaheim_net.tntp",
    )
    scenarios = [2, 3]
    stats = _scenario_sweep_stats(scenarios)

    # Act by persisting the scenario-sweep figures.
    stored_paths = persist_scenario_sweep_outputs(
        run_cfg=result.run_config,
        scenarios=scenarios,
        num_seeds=4,
        sim_stats=stats,
        sample_stats=stats,
        figure_directory=tmp_path,
    )

    # Assert that every scenario-sweep figure carries the graph marker.
    for figure_path in (
        stored_paths.simulation_plot_path,
        stored_paths.asym_simulation_plot_path,
        stored_paths.sample_plot_path,
        stored_paths.asym_sample_plot_path,
    ):
        assert "real_world_anaheim_net" in figure_path.name, (
            "Real-world scenario-sweep figures should include the graph "
            "filename marker."
        )
    pass


def test_utils_results_analysis_replot_saved_outputs_from_csv(
    tmp_path: Path,
):
    """Verify that saved CSV results can regenerate boxplot figures."""
    # Arrange a persisted typed sweep result with one failed asymmetric row.
    result = _simulation_result(10, 0.0)
    all_data = {
        key: value.copy()
        for key, value in result.summary_bundle.all_data.items()
    }
    all_data["a_p"][0] = np.nan
    result.summary_bundle.all_data.update(all_data)
    sweep_result = SweepResult(
        run_config=result.run_config,
        results=[result],
        aggregated_summary={},
    )
    output_path = tmp_path / "results" / "typed_results.csv"
    figure_directory = tmp_path / "figures"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results_to_csv(sweep_result, output_path)

    # Act by regenerating figures directly from the saved CSV.
    stored_paths = replot_saved_sweep_outputs(
        output_path,
        figure_directory=figure_directory,
        exclude_symmetric_interdictions=True,
    )

    # Assert the replay command writes both plot types without rerunning.
    assert stored_paths.results_path == output_path, \
        "replot_saved_sweep_outputs should report the source CSV path."
    assert stored_paths.sample_boxplot_path.exists(), \
        "replot_saved_sweep_outputs should save the sample boxplot."
    assert stored_paths.simulation_boxplot_path.exists(), \
        "replot_saved_sweep_outputs should save the simulation boxplot."
    assert stored_paths.sample_boxplot_path.parent == figure_directory, \
        "replot_saved_sweep_outputs should honor the figure directory."
    assert stored_paths.simulation_boxplot_path.parent == figure_directory, \
        "replot_saved_sweep_outputs should keep both plots together."
    assert stored_paths.sample_boxplot_path.name.endswith(
        "_no_sym_boxplot.png"
    ), "Symmetric-excluding replots should use a no-sym sample filename."
    assert stored_paths.simulation_boxplot_path.name.endswith(
        "_no_sym_boxplot_sims.png"
    ), "Symmetric-excluding replots should use a no-sym simulation filename."
    pass


def test_utils_results_analysis_replot_saved_outputs_defaults_to_figures_dir(
    tmp_path: Path,
    monkeypatch,
):
    """Verify that default replot figures are saved under figures/."""
    # Arrange a saved CSV outside the default figures directory.
    monkeypatch.setattr(storage_module, "_project_root", lambda: tmp_path)
    result = _simulation_result(10, 0.0)
    sweep_result = SweepResult(
        run_config=result.run_config,
        results=[result],
        aggregated_summary={},
    )
    output_path = tmp_path / "custom-results" / "typed_results.csv"
    expected_figure_directory = tmp_path / "figures"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results_to_csv(sweep_result, output_path)

    # Act by regenerating figures without a figure-directory override.
    stored_paths = replot_saved_sweep_outputs(output_path)

    # Assert that replots use the default project figures directory.
    assert stored_paths.results_path == output_path, \
        "replot_saved_sweep_outputs should report the source CSV path."
    assert stored_paths.sample_boxplot_path.parent == expected_figure_directory, \
        "replot_saved_sweep_outputs should default figures to project figures/."
    assert stored_paths.simulation_boxplot_path.parent == (
        expected_figure_directory
    ), "replot_saved_sweep_outputs should keep default figures together."
    assert stored_paths.sample_boxplot_path.exists(), \
        "replot_saved_sweep_outputs should save the default sample boxplot."
    assert stored_paths.simulation_boxplot_path.exists(), \
        "replot_saved_sweep_outputs should save the default simulation boxplot."
    pass


def test_utils_results_analysis_replot_saved_outputs_combines_multiple_csvs(
    tmp_path: Path,
    monkeypatch,
):
    """Verify that replotting can combine several saved result CSV files."""
    # Arrange two compatible saved CSV files and capture plot inputs.
    output_path_a = tmp_path / (
        "results_train_4_valid_1_test_2_m_2_n_2_deg_1_noise_0.1_seeds_3.csv"
    )
    output_path_b = tmp_path / (
        "results_train_4_valid_1_test_2_m_2_n_2_deg_1_noise_0.1_seeds_2.csv"
    )
    save_results_to_csv(
        SweepResult(
            run_config=_simulation_result(10, 0.0).run_config,
            results=[_simulation_result(10, 0.0)],
            aggregated_summary={},
        ),
        output_path_a,
    )
    save_results_to_csv(
        SweepResult(
            run_config=_simulation_result(11, 5.0).run_config,
            results=[_simulation_result(11, 5.0)],
            aggregated_summary={},
        ),
        output_path_b,
    )
    recorded: list[dict[str, object]] = []

    def _fake_boxplot(simulations, *, save_path, **kwargs):
        recorded.append(
            {
                "num_simulations": len(simulations),
                "save_path": save_path,
                "kwargs": kwargs,
            }
        )
        return plt.figure()

    monkeypatch.setattr(storage_module, "create_boxplots", _fake_boxplot)
    monkeypatch.setattr(
        storage_module,
        "create_boxplots_by_simulation",
        _fake_boxplot,
    )

    # Act by regenerating one combined plot bundle from both CSV files.
    stored_paths = replot_saved_sweep_outputs(
        [output_path_a, output_path_b],
        figure_directory=tmp_path / "figures",
    )

    # Assert that both plotting calls receive the concatenated simulations.
    assert stored_paths.results_path == output_path_a, \
        "replot_saved_sweep_outputs should keep the first CSV as primary."
    assert stored_paths.results_paths == (output_path_a, output_path_b), \
        "replot_saved_sweep_outputs should report every source CSV."
    assert stored_paths.sample_boxplot_path.name == (
        "results_train_4_valid_1_test_2_m_2_n_2_deg_1_noise_0.1"
        "_seeds_2_combined_boxplot.png"
    ), "Combined replots should use a filename showing the total simulations."
    assert stored_paths.simulation_boxplot_path.name == (
        "results_train_4_valid_1_test_2_m_2_n_2_deg_1_noise_0.1"
        "_seeds_2_combined_boxplot_sims.png"
    ), "Combined simulation replots should use the same combined stem."
    assert len(recorded) == 2, \
        "replot_saved_sweep_outputs should create both boxplot variants."
    assert all(call["num_simulations"] == 2 for call in recorded), \
        "replot_saved_sweep_outputs should concatenate all CSV simulations."
    pass


def test_utils_results_analysis_replot_saved_outputs_forwards_legend_location(
    tmp_path: Path,
    monkeypatch,
):
    """Verify that replotting forwards the requested legend location."""
    # Arrange a saved CSV and plotting stubs that record storage kwargs.
    result = _simulation_result(10, 0.0)
    sweep_result = SweepResult(
        run_config=result.run_config,
        results=[result],
        aggregated_summary={},
    )
    output_path = tmp_path / "results" / "typed_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results_to_csv(sweep_result, output_path)
    recorded: list[dict[str, object]] = []

    def _fake_boxplot(simulations, *, save_path, **kwargs):
        recorded.append(
            {
                "simulations": simulations,
                "save_path": save_path,
                "kwargs": kwargs,
            }
        )
        return plt.figure()

    monkeypatch.setattr(storage_module, "create_boxplots", _fake_boxplot)
    monkeypatch.setattr(
        storage_module,
        "create_boxplots_by_simulation",
        _fake_boxplot,
    )

    # Act by regenerating figures with a non-default legend location.
    replot_saved_sweep_outputs(
        output_path,
        legend_location="upper left",
    )

    # Assert that both boxplot variants receive the requested location.
    assert len(recorded) == 2, \
        "replot_saved_sweep_outputs should create both boxplot variants."
    assert all(
        call["kwargs"]["legend_location"] == "upper left"
        for call in recorded
    ), "replot_saved_sweep_outputs should forward the legend location."
    pass


def test_utils_results_analysis_boxplots_can_exclude_symmetric_group(
    tmp_path: Path,
):
    """Verify that result boxplots can omit symmetric interdiction results."""
    # Arrange one finite value for every available percentage series.
    calculations = {
        "no_intd_p": np.array([1.0], dtype=float),
        "no_intd_s": np.array([2.0], dtype=float),
        "no_intd_r": np.array([3.0], dtype=float),
        "no_intd_a": np.array([4.0], dtype=float),
        "sym_intd_p": np.array([5.0], dtype=float),
        "sym_intd_s": np.array([6.0], dtype=float),
        "sym_intd_r": np.array([7.0], dtype=float),
        "sym_intd_a": np.array([8.0], dtype=float),
        "asym_intd_p": np.array([9.0], dtype=float),
        "asym_intd_s": np.array([10.0], dtype=float),
        "asym_intd_r": np.array([11.0], dtype=float),
        "asym_intd_a": np.array([12.0], dtype=float),
    }

    # Act by creating a figure with the symmetric group disabled.
    fig = create_boxplots_from_calculations(
        calculations,
        save_path=tmp_path / "no_sym.png",
        include_symmetric_interdiction=False,
        legend_location="upper left",
    )
    labels = [
        tick.get_text()
        for tick in fig.axes[0].get_xticklabels()
    ]
    legend = fig.axes[0].get_legend()
    plt.close(fig)

    # Assert that only uninterdicted and asymmetric groups are shown.
    assert labels == ["no intd", "asym intd"], \
        "Boxplots should omit the symmetric-interdiction group when requested."
    assert legend._loc == Legend.codes["upper left"], \
        "Boxplots should use the requested legend location."
    pass
