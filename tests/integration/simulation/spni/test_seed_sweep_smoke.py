from types import SimpleNamespace

import numpy as np

from dflintdpy.simulation.spni.config import SeedBundle
from dflintdpy.simulation.spni.types import (
    DatasetBundle,
    EvaluationBundle,
    GraphBundle,
    PredictorBundle,
    SimulationResult,
    SummaryBundle,
)

import dflintdpy.simulation.spni.pipeline as pipeline_module


############################
### Helper functionality ###
############################


def _dataset_bundle(label: str) -> DatasetBundle:
    """Return a compact dataset bundle for sweep smoke tests."""
    array = np.array([[1.0], [2.0]], dtype=float)
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
### test seed-sweep smoke ###
###############################


def test_spni_pipeline_seed_sweep_smoke_returns_two_ordered_runs(monkeypatch):
    """Verify that a tiny two-seed sweep stays ordered and aggregates cleanly."""
    # Arrange a compact base config and a lightweight single-run stub.
    persisted_calls: list[object] = []
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
        "po_epochs": 1,
        "spo_epochs": 1,
        "po_lr": 1e-3,
        "spo_lr": 1e-3,
    }

    def _fake_run_single_simulation(cfg, **options):
        del options
        all_data = {
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
        }
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
            summary_bundle=SummaryBundle(
                prediction_mean_std={"test_mean": float(cfg.seed)},
                metrics={"metric_1": float(cfg.seed)},
                table_1={"t1_o_n_mean": float(cfg.seed)},
                table_2={},
                all_data=all_data,
            ),
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

    # Act by running the tiny two-seed sweep.
    result = pipeline_module.run_seed_sweep(base_cfg, num_seeds=2)

    # Assert that the ordered runs and aggregate summary stay consistent.
    assert [item.seed_bundle.sweep_seed for item in result.results] == [5, 6], \
        "The sweep smoke run should preserve the ordered sweep seeds."
    assert result.results[0].seed_bundle.random_seed != \
        result.results[1].seed_bundle.random_seed, \
        "The derived per-run seed bundles should differ across sweep runs."
    assert result.aggregated_summary["num_runs"] == 2, \
        "The sweep smoke run should aggregate both single-run results."
    assert np.array_equal(
        result.aggregated_summary["all_data"]["o_o"],
        np.array([5.0, 6.0], dtype=float),
    ), "Sweep aggregation should combine ordered run outputs without drift."
    assert persisted_calls == [result], \
        "Seed sweeps should persist outputs by default."
    pass
