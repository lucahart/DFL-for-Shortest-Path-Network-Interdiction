from types import SimpleNamespace

import numpy as np
import pytest

from dflintdpy.simulation.spni.config import SPNIRunConfig
from dflintdpy.simulation.spni.types import DatasetBundle, EvaluationBundle
from dflintdpy.simulation.spni.types import GraphBundle, PredictorBundle

import dflintdpy.simulation.spni.evaluate as evaluate_module


################
### Fixtures ###
################


@pytest.fixture
def run_cfg() -> SPNIRunConfig:
    """Return a compact config for evaluation-stage contract tests."""
    return SPNIRunConfig(
        base_cfg=SimpleNamespace(label="base"),
        grid_size=(3, 4),
        num_features=3,
        num_train_samples=4,
        num_val_samples=2,
        num_test_samples=3,
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
        metadata={"source_type": "SimpleNamespace"},
    )


@pytest.fixture
def graph_bundle() -> GraphBundle:
    """Return a lightweight graph bundle for evaluation-stage tests."""
    return GraphBundle(
        graph=SimpleNamespace(name="graph"),
        opt_model=SimpleNamespace(name="opt-model"),
        graph_kind="synthetic",
    )


@pytest.fixture
def dataset_bundle() -> DatasetBundle:
    """Return a dataset bundle with deterministic evaluation arrays."""
    testing_features = np.arange(9, dtype=float).reshape(3, 3)
    testing_costs = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=float,
    )
    interdiction_features = testing_features + 10.0
    interdiction_costs = testing_costs + 10.0
    return DatasetBundle(
        train_loader_adversarial=SimpleNamespace(label="train-adv"),
        val_loader_adversarial=SimpleNamespace(label="val-adv"),
        train_loader_random=SimpleNamespace(label="train-rand"),
        val_loader_random=SimpleNamespace(label="val-rand"),
        train_loader_baseline=SimpleNamespace(label="train-base"),
        val_loader_baseline=SimpleNamespace(label="val-base"),
        testing_features=testing_features,
        testing_costs=testing_costs,
        interdiction_features=interdiction_features,
        interdiction_costs=interdiction_costs,
        normalization_constant=12.5,
        data_generator_adversarial=None,
        data_generator_random=None,
    )


@pytest.fixture
def predictor_bundle() -> PredictorBundle:
    """Return a predictor bundle with distinct family sentinels."""
    return PredictorBundle(
        pfl=SimpleNamespace(label="pfl"),
        dfl=SimpleNamespace(label="dfl"),
        rdfl=SimpleNamespace(label="rdfl"),
        adfl=SimpleNamespace(label="adfl"),
    )


############################
### Helper functionality ###
############################


def _array(values: list[float]) -> np.ndarray:
    """Return one float NumPy array from a short list."""
    return np.array(values, dtype=float)


###################################
### test evaluate_uninterdicted ###
###################################


def test_spni_evaluate_uninterdicted_preserves_sample_alignment(
    monkeypatch,
    run_cfg,
    graph_bundle,
    dataset_bundle,
    predictor_bundle,
):
    """Verify that uninterdicted outputs stay aligned to the test samples."""
    # Arrange a compare stub that records the data view for both helper calls.
    calls: list[dict] = []

    def _fake_compare_shortest_paths(
        cfg,
        opt_model,
        pfl_predictor,
        dfl_predictor,
        test_data,
        adfl_predictor=None,
    ):
        calls.append(
            {
                "cfg": cfg,
                "opt_model": opt_model,
                "pfl_predictor": pfl_predictor,
                "dfl_predictor": dfl_predictor,
                "test_data": test_data,
                "adfl_predictor": adfl_predictor,
            }
        )
        if adfl_predictor is predictor_bundle.adfl:
            return (
                _array([10.0, 20.0, 30.0]),
                _array([11.0, 21.0, 31.0]),
                _array([12.0, 22.0, 32.0]),
                _array([13.0, 23.0, 33.0]),
            )
        return (
            _array([10.0, 20.0, 30.0]),
            _array([11.0, 21.0, 31.0]),
            _array([12.0, 22.0, 32.0]),
            _array([14.0, 24.0, 34.0]),
        )

    monkeypatch.setattr(
        evaluate_module.legacy_compare_module,
        "compare_shortest_paths",
        _fake_compare_shortest_paths,
    )

    # Act by evaluating the uninterdicted objective arrays.
    results = evaluate_module.evaluate_uninterdicted(
        run_cfg,
        graph_bundle,
        dataset_bundle,
        predictor_bundle,
    )

    # Assert that both helper calls saw the aligned testing view.
    assert len(calls) == 2, \
        "evaluate_uninterdicted should call compare_shortest_paths twice."
    assert calls[0]["cfg"].get("num_test_samples") == 3, \
        "evaluate_uninterdicted should expose run config values via cfg.get."
    assert calls[0]["opt_model"] is graph_bundle.opt_model, \
        "evaluate_uninterdicted should forward the optimization model."
    assert np.array_equal(
        calls[0]["test_data"]["feats"],
        dataset_bundle.testing_features,
    ), "The testing features should be forwarded unchanged."
    assert np.array_equal(
        calls[0]["test_data"]["costs"],
        dataset_bundle.testing_costs,
    ), "The testing costs should be forwarded unchanged."
    assert np.array_equal(
        results["sample_indices"],
        np.array([0, 1, 2], dtype=int),
    ), "Uninterdicted outputs should preserve one index per test sample."
    assert np.array_equal(
        results["objectives"]["oracle"],
        _array([10.0, 20.0, 30.0]),
    ), "The oracle objective array should remain aligned."
    assert np.array_equal(
        results["objectives"]["rdfl"],
        _array([14.0, 24.0, 34.0]),
    ), "The R-DFL objective array should come from the second helper call."
    assert results["diagnostics"]["helper_calls"] == 2, \
        "Diagnostics should record the two legacy helper calls."
    pass


############################################
### test evaluate_symmetric_interdiction ###
############################################


def test_spni_evaluate_symmetric_interdiction_maps_family_outputs(
    monkeypatch,
    run_cfg,
    graph_bundle,
    dataset_bundle,
    predictor_bundle,
):
    """Verify that symmetric outputs map legacy labels to family labels."""
    # Arrange a compare stub that returns one aligned row per test sample.
    calls: list[dict] = []

    def _fake_compare_sym_intd(
        cfg,
        opt_model,
        pfl_predictor,
        dfl_predictor,
        test_data,
        interdictions,
        normalization_constant,
        **kwargs,
    ):
        calls.append(
            {
                "cfg": cfg,
                "opt_model": opt_model,
                "test_data": test_data,
                "interdictions": interdictions,
                "normalization_constant": normalization_constant,
                "kwargs": kwargs,
            }
        )
        return {
            "true_objective": _array([1.0, 2.0, 3.0]),
            "po_objective": _array([4.0, 5.0, 6.0]),
            "spo_objective": _array([7.0, 8.0, 9.0]),
            "rand_adv_spo_objective": _array([10.0, 11.0, 12.0]),
            "adv_spo_objective": _array([13.0, 14.0, 15.0]),
        }

    monkeypatch.setattr(
        evaluate_module.legacy_compare_module,
        "compare_sym_intd",
        _fake_compare_sym_intd,
    )

    # Act by evaluating symmetric interdiction.
    results = evaluate_module.evaluate_symmetric_interdiction(
        run_cfg,
        graph_bundle,
        dataset_bundle,
        predictor_bundle,
    )

    # Assert that family outputs are aligned and renamed explicitly.
    assert len(calls) == 1, \
        "evaluate_symmetric_interdiction should call compare_sym_intd once."
    assert np.array_equal(
        calls[0]["interdictions"]["costs"],
        dataset_bundle.interdiction_costs,
    ), "The interdiction costs should be forwarded unchanged."
    assert calls[0]["normalization_constant"] == \
        dataset_bundle.normalization_constant, \
        "The normalization constant should be forwarded unchanged."
    assert np.array_equal(
        results["objectives"]["oracle"],
        _array([1.0, 2.0, 3.0]),
    ), "The oracle symmetric objectives should stay aligned."
    assert np.array_equal(
        results["objectives"]["pfl"],
        _array([4.0, 5.0, 6.0]),
    ), "The PFL symmetric objectives should map from `po_objective`."
    assert np.array_equal(
        results["objectives"]["adfl"],
        _array([13.0, 14.0, 15.0]),
    ), "The A-DFL symmetric objectives should map from the adverse key."
    pass


#############################################
### test evaluate_asymmetric_interdiction ###
#############################################


def test_spni_evaluate_asymmetric_interdiction_records_failure_counts(
    monkeypatch,
    run_cfg,
    graph_bundle,
    dataset_bundle,
    predictor_bundle,
):
    """Verify that asymmetric NaN slots become explicit failure counts."""
    # Arrange a compare stub with family-specific aligned NaN patterns.
    calls: list[dict] = []
    outputs = {
        None: (
            _array([1.0, np.nan, 3.0]),
            _array([1.5, np.nan, 3.5]),
        ),
        predictor_bundle.pfl.label: (
            _array([4.0, 5.0, np.nan]),
            _array([4.5, 5.5, np.nan]),
        ),
        predictor_bundle.dfl.label: (
            _array([6.0, 7.0, 8.0]),
            _array([6.5, 7.5, 8.5]),
        ),
        predictor_bundle.rdfl.label: (
            _array([9.0, np.nan, np.nan]),
            _array([9.5, np.nan, np.nan]),
        ),
        predictor_bundle.adfl.label: (
            _array([10.0, 11.0, 12.0]),
            _array([10.5, 11.5, 12.5]),
        ),
    }

    def _fake_compare_asym_intd(
        cfg,
        opt_model,
        test_data,
        interdictions,
        normalization_constant,
        pred_model=None,
        **kwargs,
    ):
        del kwargs
        calls.append({"pred_model": pred_model})
        key = None if pred_model is None else pred_model.label
        return outputs[key]

    monkeypatch.setattr(
        evaluate_module.legacy_compare_module,
        "compare_asym_intd",
        _fake_compare_asym_intd,
    )

    # Act by evaluating asymmetric interdiction.
    results = evaluate_module.evaluate_asymmetric_interdiction(
        run_cfg,
        graph_bundle,
        dataset_bundle,
        predictor_bundle,
    )

    # Assert that NaN-preserving arrays remain aligned and diagnostic counts.
    assert len(calls) == 5, \
        "evaluate_asymmetric_interdiction should cover five predictor views."
    assert np.array_equal(
        results["sample_indices"],
        np.array([0, 1, 2], dtype=int),
    ), "Asymmetric outputs should preserve one index per input sample."
    assert np.array_equal(
        np.isnan(results["estimated_objectives"]["rdfl"]),
        np.array([False, True, True]),
    ), "R-DFL asymmetric results should preserve failure slots as NaNs."
    assert results["diagnostics"]["failure_counts"] == {
        "oracle": 1,
        "pfl": 1,
        "dfl": 0,
        "rdfl": 2,
        "adfl": 0,
    }, "Failure diagnostics should report NaN-aligned solve failures."
    pass


################################################
### test evaluate_wrong_model_asymmetry(...) ###
################################################


def test_spni_evaluate_wrong_model_asymmetry_skips_cleanly_when_disabled(
    monkeypatch,
    run_cfg,
    graph_bundle,
    dataset_bundle,
    predictor_bundle,
):
    """Verify that wrong-model evaluation returns a stable skipped payload."""
    # Arrange a failing helper so the test proves the skip path avoids calls.
    def _fail_compare_wrong_asym_intd(*args, **kwargs):
        raise AssertionError("compare_wrong_asym_intd should not be called.")

    monkeypatch.setattr(
        evaluate_module.legacy_compare_module,
        "compare_wrong_asym_intd",
        _fail_compare_wrong_asym_intd,
    )

    # Act by evaluating with wrong-model experiments disabled.
    results = evaluate_module.evaluate_wrong_model_asymmetry(
        run_cfg,
        graph_bundle,
        dataset_bundle,
        predictor_bundle,
    )

    # Assert that the skipped payload stays stable and explicit.
    assert results["diagnostics"]["skipped"] is True, \
        "Wrong-model evaluation should mark disabled runs as skipped."
    assert results["diagnostics"]["reason"] == "run_config_disabled", \
        "Wrong-model evaluation should record why it was skipped."
    assert results["objectives"] == {}, \
        "Skipped wrong-model evaluation should return no objective arrays."
    pass


def test_spni_evaluate_wrong_model_asymmetry_aligns_short_legacy_outputs(
    monkeypatch,
    graph_bundle,
    dataset_bundle,
    predictor_bundle,
):
    """Verify that wrong-model outputs are padded and diagnosed explicitly."""
    # Arrange an enabled config and a helper that returns short arrays.
    run_cfg = SPNIRunConfig(
        base_cfg=SimpleNamespace(label="base"),
        grid_size=(3, 4),
        num_features=3,
        num_train_samples=4,
        num_val_samples=2,
        num_test_samples=3,
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
        metadata={"source_type": "SimpleNamespace"},
    )
    calls: list[dict] = []

    def _fake_compare_wrong_asym_intd(
        cfg,
        opt_model,
        test_data,
        interdictions,
        normalization_constant,
        true_model,
        false_model,
    ):
        calls.append(
            {
                "true_model": true_model,
                "false_model": false_model,
            }
        )
        return _array([1.0, 2.0])

    monkeypatch.setattr(
        evaluate_module.legacy_compare_module,
        "compare_wrong_asym_intd",
        _fake_compare_wrong_asym_intd,
    )

    # Act by evaluating the enabled wrong-model experiments.
    results = evaluate_module.evaluate_wrong_model_asymmetry(
        run_cfg,
        graph_bundle,
        dataset_bundle,
        predictor_bundle,
    )

    # Assert that short legacy outputs are padded back to sample alignment.
    assert len(calls) == 6, \
        "Wrong-model evaluation should cover all six predictor pairings."
    assert results["diagnostics"]["pair_count"] == 6, \
        "Diagnostics should record the number of wrong-model pairings."
    assert np.array_equal(
        np.isnan(results["objectives"]["true_dfl_false_pfl"]),
        np.array([False, False, True]),
    ), "Short legacy wrong-model outputs should be padded with trailing NaNs."
    assert results["diagnostics"]["pair_diagnostics"][
        "true_dfl_false_pfl"
    ]["padded_count"] == 1, \
        "Padding diagnostics should expose the recovered alignment gap."
    pass


###########################
### test evaluate_all ###
###########################


def test_spni_evaluate_evaluate_all_collects_stage_outputs(
    monkeypatch,
    run_cfg,
    graph_bundle,
    dataset_bundle,
    predictor_bundle,
):
    """Verify that evaluate_all coordinates the stage wrappers explicitly."""
    # Arrange stage stubs so bundle assembly can be isolated.
    calls: list[str] = []

    def _fake_uninterdicted(cfg, graph, data, predictors):
        calls.append("uninterdicted")
        return {"diagnostics": {"num_samples": 3}}

    def _fake_symmetric(cfg, graph, data, predictors):
        calls.append("symmetric")
        return {"diagnostics": {"num_samples": 3}}

    def _fake_asymmetric(cfg, graph, data, predictors):
        calls.append("asymmetric")
        return {"diagnostics": {"failure_counts": {"pfl": 1}}}

    def _fake_wrong_model(cfg, graph, data, predictors):
        calls.append("wrong_model_asymmetry")
        return {"diagnostics": {"skipped": True}}

    monkeypatch.setattr(
        evaluate_module,
        "evaluate_uninterdicted",
        _fake_uninterdicted,
    )
    monkeypatch.setattr(
        evaluate_module,
        "evaluate_symmetric_interdiction",
        _fake_symmetric,
    )
    monkeypatch.setattr(
        evaluate_module,
        "evaluate_asymmetric_interdiction",
        _fake_asymmetric,
    )
    monkeypatch.setattr(
        evaluate_module,
        "evaluate_wrong_model_asymmetry",
        _fake_wrong_model,
    )

    # Act by aggregating the stage outputs.
    bundle = evaluate_module.evaluate_all(
        run_cfg,
        graph_bundle,
        dataset_bundle,
        predictor_bundle,
    )

    # Assert that evaluate_all returns the canonical bundle and diagnostics.
    assert isinstance(bundle, EvaluationBundle), \
        "evaluate_all should return an EvaluationBundle."
    assert calls == [
        "uninterdicted",
        "symmetric",
        "asymmetric",
        "wrong_model_asymmetry",
    ], "evaluate_all should coordinate the four stage wrappers in order."
    assert bundle.diagnostics["asymmetric_failure_counts"] == {"pfl": 1}, \
        "Bundle diagnostics should surface asymmetric failure counts."
    assert bundle.diagnostics["wrong_model_skipped"] is True, \
        "Bundle diagnostics should surface wrong-model skip state."
    pass
