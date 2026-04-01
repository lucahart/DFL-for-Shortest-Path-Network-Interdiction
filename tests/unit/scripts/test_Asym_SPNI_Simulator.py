from types import SimpleNamespace

import numpy as np

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

import dflintdpy.scripts.Asym_SPNI_Simulator as script_module


############################
### Helper functionality ###
############################


def _simulation_result(seed: int, values: list[float]) -> SimulationResult:
    """Return one compact single-run result for sweep-wrapper tests."""
    value_array = np.asarray(values, dtype=float)
    summary_bundle = SummaryBundle(
        prediction_mean_std={"test_mean": float(value_array.mean())},
        metrics={"metric_1": float(value_array.sum())},
        table_1={"t1_o_n_mean": float(value_array[0])},
        table_2={"t2_p_s_mean": float(value_array[-1])},
        all_data={
            "o_o": value_array,
            "o_p": value_array + 1.0,
        },
    )
    return SimulationResult(
        run_config=SPNIRunConfig(
            base_cfg=SimpleNamespace(label="base"),
            grid_size=(2, 3),
            num_features=2,
            num_train_samples=4,
            num_val_samples=1,
            num_test_samples=2,
            batch_size=2,
            budget=1,
            num_scenarios=2,
            deg=5,
            noise_width=0.25,
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
            opt_model=SimpleNamespace(name="model"),
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


def _sweep_result() -> SweepResult:
    """Return one compact sweep result for wrapper-compatibility tests."""
    cfg = SimpleNamespace(
        num_seeds=2,
        num_train_samples=4,
        num_val_samples=1,
        num_test_samples=2,
        grid_size=(2, 3),
        deg=5,
        noise_width=0.25,
        get=lambda key, default=None: getattr(
            SimpleNamespace(
                num_seeds=2,
                num_train_samples=4,
                num_val_samples=1,
                num_test_samples=2,
                grid_size=(2, 3),
                deg=5,
                noise_width=0.25,
            ),
            key,
            default,
        ),
    )
    return SweepResult(
        run_config=cfg,
        results=[
            _simulation_result(10, [1.0, 2.0]),
            _simulation_result(11, [3.0, 4.0]),
        ],
        aggregated_summary={"num_runs": 2},
        diagnostics={"side_effects_enabled": False},
    )


#####################
### test run_sweep ###
#####################


def test_scripts_Asym_SPNI_Simulator_run_sweep_delegates_without_side_effects(
    monkeypatch,
):
    """Verify that `run_sweep(...)` delegates and leaves side effects off."""
    # Arrange a pipeline stub plus side-effect sentinels.
    calls: list[dict[str, object]] = []
    analyzed_calls: list[str] = []
    result = _sweep_result()

    def _fake_run_seed_sweep(cfg, *, num_seeds, **options):
        result.diagnostics.update(
            {
                "present_results": bool(options["present_results"]),
                "legacy_output_path": None,
                "sample_boxplot_path": None,
                "simulation_boxplot_path": None,
            }
        )
        calls.append(
            {
                "cfg": cfg,
                "num_seeds": num_seeds,
                "options": options,
            }
        )
        return result

    def _fake_analyze_results():
        analyzed_calls.append("called")

    monkeypatch.setattr(script_module, "run_seed_sweep", _fake_run_seed_sweep)
    monkeypatch.setattr(
        script_module,
        "analyze_results",
        _fake_analyze_results,
    )
    cfg = SimpleNamespace(
        label="legacy-cfg",
        num_seeds=2,
        get=lambda key, default=None: getattr(
            SimpleNamespace(label="legacy-cfg", num_seeds=2),
            key,
            default,
        ),
    )

    # Act by calling the compatibility wrapper with explicit no-side-effect
    # options.
    sweep_result = script_module.run_sweep(
        cfg,
        present_results=False,
        compute_asym_intd_2=False,
        compute_asym_intd=True,
    )

    # Assert that delegation happened and persistence hooks stayed off.
    assert sweep_result is result, \
        "run_sweep should return the typed SweepResult unchanged."
    assert len(calls) == 1, \
        "run_sweep should delegate to run_seed_sweep exactly once."
    assert calls[0]["cfg"] is cfg, \
        "run_sweep should forward the original config object unchanged."
    assert calls[0]["num_seeds"] == 2, \
        "run_sweep should derive num_seeds from the legacy config by default."
    assert calls[0]["options"] == {
        "present_results": False,
        "output_path": None,
        "compute_asym_intd": True,
        "compute_wrong_asym_intd": False,
    }, "run_sweep should map legacy runtime flags onto pipeline options."
    assert analyzed_calls == [], \
        "run_sweep should not analyze results unless requested explicitly."
    assert sweep_result.diagnostics["present_results"] is False, \
        "run_sweep should record that persistence was disabled."
    assert sweep_result.diagnostics["legacy_output_path"] is None, \
        "run_sweep should leave the legacy output path unset when skipped."
    assert sweep_result.diagnostics["sample_boxplot_path"] is None, \
        "run_sweep should leave sample boxplot output unset when skipped."
    assert sweep_result.diagnostics["simulation_boxplot_path"] is None, \
        "run_sweep should leave simulation boxplot output unset when skipped."
    pass


def test_scripts_Asym_SPNI_Simulator_run_sweep_persists_legacy_results(
    monkeypatch,
):
    """Verify that `run_sweep(...)` persists sweep outputs on demand."""
    # Arrange a pipeline stub and side-effect capture hooks.
    calls: list[dict[str, object]] = []
    analyzed_calls: list[str] = []
    result = _sweep_result()

    def _fake_run_seed_sweep(cfg, *, num_seeds, **options):
        result.diagnostics.update(
            {
                "present_results": bool(options["present_results"]),
                "legacy_output_path": str(options["output_path"]),
                "sample_boxplot_path": "custom-results_boxplot.png",
                "simulation_boxplot_path": "custom-results_boxplot_sims.png",
            }
        )
        calls.append(
            {
                "cfg": cfg,
                "num_seeds": num_seeds,
                "options": options,
            }
        )
        return result

    def _fake_analyze_results():
        analyzed_calls.append("called")

    monkeypatch.setattr(script_module, "run_seed_sweep", _fake_run_seed_sweep)
    monkeypatch.setattr(
        script_module,
        "analyze_results",
        _fake_analyze_results,
    )
    output_path = "custom-results.csv"

    # Act by enabling explicit persistence and analysis.
    sweep_result = script_module.run_sweep(
        result.run_config,
        num_seeds=2,
        present_results=True,
        analyze=True,
        output_path=output_path,
    )

    # Assert that the new storage pipeline saw the typed sweep and paths.
    assert sweep_result is result, \
        "run_sweep should return the same SweepResult after persistence."
    assert len(calls) == 1, \
        "run_sweep should delegate to the typed seed sweep once."
    assert calls[0]["options"] == {
        "present_results": True,
        "output_path": output_path,
        "compute_asym_intd": True,
        "compute_wrong_asym_intd": True,
    }, "run_sweep should forward persistence options to run_seed_sweep."
    assert calls[0]["num_seeds"] == 2, \
        "run_sweep should forward the resolved seed count."
    assert calls[0]["cfg"] is result.run_config, \
        "run_sweep should forward the chosen config to the typed pipeline."
    assert sweep_result.diagnostics["legacy_output_path"] == output_path, \
        "run_sweep should honor an explicit legacy output path."
    assert analyzed_calls == ["called"], \
        "run_sweep should trigger analysis only when requested explicitly."
    assert sweep_result.diagnostics["sample_boxplot_path"] == (
        "custom-results_boxplot.png"
    ), "run_sweep should record the saved sample-boxplot path."
    assert sweep_result.diagnostics["simulation_boxplot_path"] == (
        "custom-results_boxplot_sims.png"
    ), "run_sweep should record the saved simulation-boxplot path."
    assert sweep_result.diagnostics["legacy_results_count"] == 2, \
        "run_sweep should record the number of legacy payload entries."
    pass
