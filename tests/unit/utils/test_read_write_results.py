from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

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
from dflintdpy.utils.read_write_results import load_results_from_csv, save_results_to_csv


DATA_KEYS = [
    "o_o",
    "o_p",
    "o_s",
    "o_r",
    "o_mr",
    "o_ma",
    "o_m",
    "o_a",
    "s_o",
    "s_p",
    "s_s",
    "s_r",
    "s_mr",
    "s_ma",
    "s_m",
    "s_a",
    "a_o",
    "a_p",
    "a_s",
    "a_r",
    "a_mr",
    "a_ma",
    "a_m",
    "a_a",
    "a_p_o",
    "a_s_o",
    "a_r_o",
    "a_a_o",
]


def _make_result(sim_offset: float, size: int = 3) -> dict:
    all_data = {}
    for idx, key in enumerate(DATA_KEYS):
        base = sim_offset + idx
        all_data[key] = np.array([base + i * 0.1 for i in range(size)], dtype=np.float64)
    return {"all_data": all_data}


def _simulation_result(seed: int) -> SimulationResult:
    all_data = {
        "o_o": np.array([1.0, 2.0], dtype=float),
        "o_p": np.array([2.0, 3.0], dtype=float),
        "o_s": np.array([3.0, 4.0], dtype=float),
        "o_r": np.array([4.0, 5.0], dtype=float),
        "o_a": np.array([5.0, 6.0], dtype=float),
        "s_o": np.array([6.0, 7.0], dtype=float),
        "s_p": np.array([7.0, 8.0], dtype=float),
        "s_s": np.array([8.0, 9.0], dtype=float),
        "s_r": np.array([9.0, 10.0], dtype=float),
        "s_a": np.array([10.0, 11.0], dtype=float),
        "a_o": np.array([11.0, 12.0], dtype=float),
        "a_p": np.array([12.0, 13.0], dtype=float),
        "a_s": np.array([13.0, 14.0], dtype=float),
        "a_r": np.array([14.0, 15.0], dtype=float),
        "a_a": np.array([15.0, 16.0], dtype=float),
        "a_p_o": np.array([16.0, 17.0], dtype=float),
        "a_s_o": np.array([17.0, 18.0], dtype=float),
        "a_r_o": np.array([18.0, 19.0], dtype=float),
        "a_a_o": np.array([19.0, 20.0], dtype=float),
        "a_s_p": np.array([21.0, 22.0], dtype=float),
    }
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
        summary_bundle=SummaryBundle(
            prediction_mean_std={},
            metrics={},
            table_1={},
            table_2={},
            all_data=all_data,
        ),
    )


def test_save_and_load_results_round_trip(tmp_path: Path):
    results = [_make_result(0.0), _make_result(10.0)]
    output_path = tmp_path / "results.csv"

    save_results_to_csv(results, output_path)
    assert output_path.exists()

    loaded = load_results_from_csv(output_path)
    assert len(loaded) == 2
    np.testing.assert_allclose(loaded[1]["s_a"], results[1]["all_data"]["s_a"])


def test_load_results_fills_missing_columns_with_nan(tmp_path: Path):
    output_path = tmp_path / "results_missing_cols.csv"
    minimal_results = [{"all_data": {"o_o": np.array([1.0, 2.0, 3.0])}}]

    save_results_to_csv(minimal_results, output_path)
    loaded = load_results_from_csv(output_path)

    assert np.isnan(loaded[0]["a_a"]).all()


def test_save_results_to_csv_accepts_typed_sweep_results(tmp_path: Path):
    results = [_simulation_result(10), _simulation_result(11)]
    sweep_result = SweepResult(
        run_config=results[0].run_config,
        results=results,
        aggregated_summary={},
    )
    output_path = tmp_path / "typed_results.csv"

    save_results_to_csv(sweep_result, output_path)
    loaded = load_results_from_csv(output_path)

    assert output_path.exists()
    assert len(loaded) == 2
    np.testing.assert_allclose(loaded[0]["o_o"], np.array([1.0, 2.0]))
    np.testing.assert_allclose(loaded[0]["a_s_p"], np.array([21.0, 22.0]))
