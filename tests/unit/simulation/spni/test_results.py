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
)

import dflintdpy.simulation.spni.results as results_module


################
### Fixtures ###
################


@pytest.fixture
def run_cfg() -> SPNIRunConfig:
    """Return a compact config for summary-stage contract tests."""
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
        compute_wrong_asym_intd=True,
        load_real_world_graph=None,
        cache_policy=CachePolicy(replace_pred=True),
        metadata={"source_type": "SimpleNamespace"},
    )


@pytest.fixture
def dataset_bundle() -> DatasetBundle:
    """Return a dataset bundle with deterministic arrays and train costs."""
    train_loader = SimpleNamespace(
        dataset=SimpleNamespace(
            costs=np.array(
                [
                    [[0.1, 0.2], [0.3, 0.4]],
                    [[0.5, 0.6], [0.7, 0.8]],
                ],
                dtype=float,
            )
        )
    )
    features = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    costs = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=float)
    return DatasetBundle(
        train_loader_adversarial=train_loader,
        val_loader_adversarial=SimpleNamespace(label="val-adv"),
        train_loader_random=SimpleNamespace(label="train-rand"),
        val_loader_random=SimpleNamespace(label="val-rand"),
        train_loader_baseline=SimpleNamespace(label="train-base"),
        val_loader_baseline=SimpleNamespace(label="val-base"),
        testing_features=features,
        testing_costs=costs,
        interdiction_features=features + 10.0,
        interdiction_costs=np.array([[2.0, 2.5], [3.0, 3.5]], dtype=float),
        normalization_constant=10.0,
        data_generator_adversarial=None,
        data_generator_random=None,
    )


@pytest.fixture
def predictor_bundle() -> PredictorBundle:
    """Return a predictor bundle with deterministic callable stubs."""
    return PredictorBundle(
        pfl=_PredictorStub([[1.0, 2.0], [3.0, 4.0]]),
        dfl=_PredictorStub([[2.0, 3.0], [4.0, 5.0]]),
        rdfl=_PredictorStub([[3.0, 4.0], [5.0, 6.0]]),
        adfl=_PredictorStub([[4.0, 5.0], [6.0, 7.0]]),
    )


@pytest.fixture
def evaluation_bundle() -> EvaluationBundle:
    """Return a compact evaluation bundle with aligned raw arrays."""
    return EvaluationBundle(
        uninterdicted={
            "sample_indices": np.array([0, 1], dtype=int),
            "objectives": {
                "oracle": np.array([1.0, 2.0], dtype=float),
                "pfl": np.array([1.2, 2.2], dtype=float),
                "dfl": np.array([1.1, 2.1], dtype=float),
                "rdfl": np.array([1.3, 2.3], dtype=float),
                "adfl": np.array([1.4, 2.4], dtype=float),
            },
            "diagnostics": {},
        },
        symmetric={
            "sample_indices": np.array([0, 1], dtype=int),
            "objectives": {
                "oracle": np.array([10.0, 12.0], dtype=float),
                "pfl": np.array([11.0, 13.0], dtype=float),
                "dfl": np.array([10.5, 12.5], dtype=float),
                "rdfl": np.array([12.0, 14.0], dtype=float),
                "adfl": np.array([9.0, 11.0], dtype=float),
            },
            "diagnostics": {},
        },
        asymmetric={
            "sample_indices": np.array([0, 1], dtype=int),
            "estimated_objectives": {
                "oracle": np.array([20.0, np.nan], dtype=float),
                "pfl": np.array([22.0, np.nan], dtype=float),
                "dfl": np.array([21.0, 22.0], dtype=float),
                "rdfl": np.array([23.0, 24.0], dtype=float),
                "adfl": np.array([24.0, 25.0], dtype=float),
            },
            "oracle_objectives": {
                "oracle": np.array([20.0, np.nan], dtype=float),
                "pfl": np.array([19.0, np.nan], dtype=float),
                "dfl": np.array([18.0, 20.0], dtype=float),
                "rdfl": np.array([17.0, 19.0], dtype=float),
                "adfl": np.array([16.0, 18.0], dtype=float),
            },
            "diagnostics": {"failure_counts": {"oracle": 1, "pfl": 1}},
        },
        wrong_model_asymmetry={
            "sample_indices": np.array([0, 1], dtype=int),
            "objectives": {
                "true_dfl_false_pfl": np.array([7.0, 8.0], dtype=float),
                "true_pfl_false_dfl": np.array([6.0, 7.0], dtype=float),
                "true_adfl_false_pfl": np.array([5.0, 6.0], dtype=float),
                "true_pfl_false_adfl": np.array([4.0, 5.0], dtype=float),
                "true_adfl_false_dfl": np.array([3.0, 4.0], dtype=float),
                "true_dfl_false_adfl": np.array([2.0, 3.0], dtype=float),
            },
            "diagnostics": {"skipped": False},
        },
    )


############################
### Helper functionality ###
############################


class _PredictorStub:
    """Return a deterministic batch of predicted costs."""

    def __init__(self, outputs):
        self.outputs = np.array(outputs, dtype=float)

    def __call__(self, features):
        return self.outputs


def _summary_bundle(
    all_data: dict[str, np.ndarray],
    *,
    metric_1: float = 0.0,
) -> SummaryBundle:
    """Return a compact summary bundle for row and sweep tests."""
    return SummaryBundle(
        prediction_mean_std={"test_mean": 1.0},
        metrics={"metric_1": metric_1},
        table_1={"t1_o_n_mean": 1.0},
        table_2={},
        all_data=all_data,
    )


def _simulation_result(
    run_cfg: SPNIRunConfig,
    summary_bundle: SummaryBundle,
    *,
    simulation_index: int,
) -> SimulationResult:
    """Return a compact simulation result for flatten/aggregate tests."""
    dataset_bundle = DatasetBundle(
        train_loader_adversarial=SimpleNamespace(label="train-adv"),
        val_loader_adversarial=SimpleNamespace(label="val-adv"),
        train_loader_random=SimpleNamespace(label="train-rand"),
        val_loader_random=SimpleNamespace(label="val-rand"),
        train_loader_baseline=SimpleNamespace(label="train-base"),
        val_loader_baseline=SimpleNamespace(label="val-base"),
        testing_features=np.array([[1.0], [2.0]], dtype=float),
        testing_costs=np.array([[1.0], [2.0]], dtype=float),
        interdiction_features=np.array([[3.0], [4.0]], dtype=float),
        interdiction_costs=np.array([[5.0], [6.0]], dtype=float),
        normalization_constant=1.0,
    )
    predictor_bundle = PredictorBundle(
        pfl=SimpleNamespace(label="pfl"),
        dfl=SimpleNamespace(label="dfl"),
        rdfl=SimpleNamespace(label="rdfl"),
        adfl=SimpleNamespace(label="adfl"),
    )
    evaluation_bundle = EvaluationBundle(
        uninterdicted={},
        symmetric={},
        asymmetric={},
        wrong_model_asymmetry={},
    )
    return SimulationResult(
        run_config=run_cfg,
        seed_bundle=SeedBundle(
            sweep_seed=simulation_index,
            random_seed=1,
            intd_seed=2,
            loader_seed=3,
        ),
        graph_bundle=GraphBundle(
            graph=SimpleNamespace(name="graph"),
            opt_model=SimpleNamespace(name="opt-model"),
            graph_kind="synthetic",
        ),
        dataset_bundle=dataset_bundle,
        predictor_bundle=predictor_bundle,
        evaluation_bundle=evaluation_bundle,
        summary_bundle=summary_bundle,
        diagnostics={"simulation_index": simulation_index},
    )


###########################
### test build_summary ###
###########################


def test_spni_results_build_summary_derives_legacy_tables_and_metrics(
    run_cfg,
    dataset_bundle,
    predictor_bundle,
    evaluation_bundle,
):
    """Verify that build_summary recreates the legacy summary payloads."""
    # Act by building the typed summary bundle from stage outputs.
    bundle = results_module.build_summary(
        run_cfg,
        dataset_bundle,
        predictor_bundle,
        evaluation_bundle,
    )

    # Assert that legacy all_data keys and summary scalars are preserved.
    assert isinstance(bundle, SummaryBundle), \
        "build_summary should return a SummaryBundle."
    assert np.array_equal(
        bundle.all_data["o_o"],
        np.array([10.0, 20.0], dtype=float),
    ), "Uninterdicted oracle outputs should be rescaled by normalization."
    assert bundle.prediction_mean_std["po_mean"] == 2.5, \
        "Prediction stats should use the PFL predictor outputs."
    assert bundle.metrics["metric_1"] == 1.0, \
        "metric_1 should match the legacy PO minus SPO no-intd mean gap."
    assert bundle.metrics["asym_nan_rows_po"] == 1, \
        "Asymmetric NaN counts should be surfaced in metrics."
    assert bundle.table_1["t1_p_a_mean"] == 22.0, \
        "Table 1 should use NaN-safe asymmetric means."
    assert bundle.table_2["t2_p_s_mean"] == 6.5, \
        "Table 2 should map wrong-model pairings into the legacy keys."
    pass


#################################
### test to_legacy_all_data ###
#################################


def test_spni_results_to_legacy_all_data_preserves_export_keys(
    run_cfg,
    dataset_bundle,
    predictor_bundle,
    evaluation_bundle,
):
    """Verify that the legacy all_data export remains directly accessible."""
    # Arrange a summary bundle using the production builder.
    summary_bundle = results_module.build_summary(
        run_cfg,
        dataset_bundle,
        predictor_bundle,
        evaluation_bundle,
    )

    # Act by asking for the compatibility export payload.
    all_data = results_module.to_legacy_all_data(summary_bundle)

    # Assert that the expected legacy schema is present and aligned.
    assert set(results_module.LEGACY_ALL_DATA_KEYS).issubset(all_data), \
        "Legacy all_data export should contain the standard baseline keys."
    assert set(results_module.WRONG_MODEL_ALL_DATA_KEYS).issubset(all_data), \
        "Legacy all_data export should include wrong-model keys when enabled."
    assert np.array_equal(
        all_data["a_p_o"],
        np.array([19.0, np.nan], dtype=float),
        equal_nan=True,
    ), "Legacy all_data export should preserve oracle asymmetry arrays."
    pass


##################################
### test flatten_result_rows ###
##################################


def test_spni_results_flatten_result_rows_preserves_indices(run_cfg):
    """Verify that row flattening keeps simulation and sample indices stable."""
    # Arrange a simulation result with a compact legacy export payload.
    summary_bundle = _summary_bundle(
        {
            "o_o": np.array([1.0, 2.0], dtype=float),
            "o_p": np.array([3.0, 4.0], dtype=float),
        }
    )
    result = _simulation_result(
        run_cfg,
        summary_bundle,
        simulation_index=7,
    )

    # Act by flattening the result into CSV-style rows.
    rows = results_module.flatten_result_rows(result)

    # Assert that row indices and per-sample values are preserved.
    assert rows == [
        {
            "simulation_index": 7,
            "sample_index": 0,
            "o_o": 1.0,
            "o_p": 3.0,
        },
        {
            "simulation_index": 7,
            "sample_index": 1,
            "o_o": 2.0,
            "o_p": 4.0,
        },
    ], "Flattened rows should keep both simulation and sample indices."
    pass


########################################
### test aggregate_sweep_results ###
########################################


def test_spni_results_aggregate_sweep_results_preserves_rows_and_safe_math(
    run_cfg,
):
    """Verify that sweep aggregation keeps row order and safe percentages."""
    # Arrange two simulation results, including zero-denominator rows.
    result_1 = _simulation_result(
        run_cfg,
        _summary_bundle(
            {
                "o_o": np.array([0.0, 0.0], dtype=float),
                "o_p": np.array([0.0, 1.0], dtype=float),
                "o_s": np.array([0.0, 2.0], dtype=float),
                "o_r": np.array([0.0, 3.0], dtype=float),
                "o_a": np.array([0.0, 4.0], dtype=float),
                "s_o": np.array([0.0, 0.0], dtype=float),
                "s_p": np.array([0.0, 1.0], dtype=float),
                "s_s": np.array([0.0, 2.0], dtype=float),
                "s_r": np.array([0.0, 3.0], dtype=float),
                "s_a": np.array([0.0, 4.0], dtype=float),
                "a_o": np.array([0.0, 0.0], dtype=float),
                "a_p": np.array([0.0, 1.0], dtype=float),
                "a_s": np.array([0.0, 2.0], dtype=float),
                "a_r": np.array([0.0, 3.0], dtype=float),
                "a_a": np.array([0.0, 4.0], dtype=float),
                "a_p_o": np.array([0.0, 0.0], dtype=float),
                "a_s_o": np.array([0.0, 0.0], dtype=float),
                "a_r_o": np.array([0.0, 0.0], dtype=float),
                "a_a_o": np.array([0.0, 0.0], dtype=float),
            },
            metric_1=1.0,
        ),
        simulation_index=9,
    )
    result_2 = _simulation_result(
        run_cfg,
        _summary_bundle(
            {
                "o_o": np.array([2.0, 2.0], dtype=float),
                "o_p": np.array([3.0, 4.0], dtype=float),
                "o_s": np.array([4.0, 5.0], dtype=float),
                "o_r": np.array([5.0, 6.0], dtype=float),
                "o_a": np.array([6.0, 7.0], dtype=float),
                "s_o": np.array([2.0, 2.0], dtype=float),
                "s_p": np.array([3.0, 4.0], dtype=float),
                "s_s": np.array([4.0, 5.0], dtype=float),
                "s_r": np.array([5.0, 6.0], dtype=float),
                "s_a": np.array([6.0, 7.0], dtype=float),
                "a_o": np.array([6.0, 6.0], dtype=float),
                "a_p": np.array([3.0, 4.0], dtype=float),
                "a_s": np.array([4.0, 5.0], dtype=float),
                "a_r": np.array([5.0, 6.0], dtype=float),
                "a_a": np.array([6.0, 7.0], dtype=float),
                "a_p_o": np.array([2.0, 2.0], dtype=float),
                "a_s_o": np.array([3.0, 3.0], dtype=float),
                "a_r_o": np.array([4.0, 4.0], dtype=float),
                "a_a_o": np.array([5.0, 5.0], dtype=float),
            },
            metric_1=2.0,
        ),
        simulation_index=11,
    )

    # Act by aggregating the sweep-level summary payload.
    aggregated = results_module.aggregate_sweep_results([result_1, result_2])

    # Assert that rows are reindexed by sweep order and percentages stay safe.
    assert aggregated["num_runs"] == 2, \
        "Sweep aggregation should record the number of runs."
    assert [row["simulation_index"] for row in aggregated["rows"]] == [
        0, 0, 1, 1,
    ], "Aggregated rows should use stable sweep-order indices."
    assert np.array_equal(
        aggregated["all_data"]["o_p"],
        np.array([0.0, 1.0, 3.0, 4.0], dtype=float),
    ), "Sweep aggregation should concatenate all_data arrays in run order."
    assert np.all(np.isfinite(
        aggregated["percentage_increases"]["samples"]["no_intd_p"]
    )), "Sample-level safe percentage math should avoid NaN and Inf outputs."
    assert np.array_equal(
        aggregated["percentage_increases"]["simulations"]["no_intd_p"],
        np.array([0.0, 75.0], dtype=float),
    ), "Simulation-level safe percentage math should match aggregate sums."
    assert np.array_equal(
        aggregated["percentage_increases"]["samples"]["asym_intd_p"],
        np.array([0.0, 0.0, 50.0, 100.0], dtype=float),
    ), "Asymmetric sample percentages should use predictor-specific oracles."
    assert np.array_equal(
        aggregated["percentage_increases"]["simulations"]["asym_intd_p"],
        np.array([0.0, 75.0], dtype=float),
    ), "Asymmetric simulation percentages should not reuse `a_o`."
    pass
