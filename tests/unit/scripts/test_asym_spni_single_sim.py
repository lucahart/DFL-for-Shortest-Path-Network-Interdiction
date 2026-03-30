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
)

import dflintdpy.scripts.asym_spni_single_sim as script_module


############################
### Helper functionality ###
############################


def _simulation_result() -> SimulationResult:
    """Return a compact simulation result for wrapper-compatibility tests."""
    summary_bundle = SummaryBundle(
        prediction_mean_std={"test_mean": 1.0},
        metrics={"metric_1": 2.0},
        table_1={"t1_o_n_mean": 3.0},
        table_2={"t2_p_s_mean": 4.0},
        all_data={
            "o_o": np.array([5.0, 6.0], dtype=float),
            "o_p": np.array([7.0, 8.0], dtype=float),
        },
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
            seed=5,
            random_seed=7,
            intd_seed=11,
            loader_seed=13,
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
            sweep_seed=5,
            random_seed=7,
            intd_seed=11,
            loader_seed=13,
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


################################
### test single_sim(...) ###
################################


def test_scripts_asym_spni_single_sim_single_sim_delegates_and_preserves_shape(
    monkeypatch,
):
    """Verify that `single_sim(...)` remains a thin compatibility wrapper."""
    # Arrange a pipeline stub that returns a compact typed simulation result.
    calls: list[dict] = []
    result = _simulation_result()

    def _fake_run_single_simulation(cfg, **options):
        calls.append(
            {
                "cfg": cfg,
                "options": options,
            }
        )
        return result

    monkeypatch.setattr(
        script_module,
        "run_single_simulation",
        _fake_run_single_simulation,
    )
    cfg = SimpleNamespace(label="legacy-cfg")

    # Act by calling the legacy entrypoint.
    payload = script_module.single_sim(
        cfg,
        visualize=True,
        compute_asym_intd_2=False,
        compute_asym_intd=True,
    )

    # Assert that the wrapper delegates once and returns the legacy 5-tuple.
    assert len(calls) == 1, \
        "single_sim should delegate to run_single_simulation exactly once."
    assert calls[0]["cfg"] is cfg, \
        "single_sim should forward the original config object unchanged."
    assert calls[0]["options"] == {
        "compute_asym_intd": True,
        "compute_wrong_asym_intd": False,
    }, "single_sim should map legacy runtime flags onto pipeline options."
    assert isinstance(payload, tuple), \
        "single_sim should preserve the legacy tuple return type."
    assert len(payload) == 5, \
        "single_sim should preserve the legacy 5-value return shape."
    assert payload[0] == {"test_mean": 1.0}, \
        "single_sim should expose prediction_mean_std unchanged."
    assert payload[1] == {"metric_1": 2.0}, \
        "single_sim should expose metrics unchanged."
    assert payload[2] == {"t1_o_n_mean": 3.0}, \
        "single_sim should expose table_1 unchanged."
    assert payload[3] == {"t2_p_s_mean": 4.0}, \
        "single_sim should expose table_2 unchanged."
    assert np.array_equal(payload[4]["o_o"], np.array([5.0, 6.0])), \
        "single_sim should expose legacy all_data through the compatibility adapter."
    pass
