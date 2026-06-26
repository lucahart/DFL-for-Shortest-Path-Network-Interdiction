from types import SimpleNamespace

import numpy as np
import pytest

from dflintdpy.simulation.spni.config import CachePolicy, SPNIRunConfig
from dflintdpy.simulation.spni.types import (
    DatasetBundle,
    GraphBundle,
    PredictorBundle,
    TrainingLogBundle,
)
from dflintdpy.utils.read_write import _unique_pred_hash

import dflintdpy.simulation.spni.train as train_module


################
### Fixtures ###
################


@pytest.fixture
def run_cfg() -> SPNIRunConfig:
    """Return a compact config for predictor-training contract tests."""
    return SPNIRunConfig(
        base_cfg=SimpleNamespace(label="base", lam=0.0, anchor="none"),
        grid_size=(3, 4),
        num_features=5,
        num_train_samples=10,
        num_val_samples=2,
        num_test_samples=3,
        batch_size=4,
        budget=1,
        num_scenarios=6,
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
        pfl_epochs=8,
        dfl_epochs=9,
        pfl_lr=1e-3,
        dfl_lr=2e-3,
        compute_asym_intd=True,
        compute_wrong_asym_intd=False,
        load_real_world_graph=None,
        cache_policy=CachePolicy(replace_pred=True, replace_data=True),
        metadata={"source_type": "SimpleNamespace"},
    )


@pytest.fixture
def graph_bundle() -> GraphBundle:
    """Return a lightweight graph bundle for training-stage tests."""
    return GraphBundle(
        graph=SimpleNamespace(name="graph"),
        opt_model=SimpleNamespace(name="opt-model"),
        graph_kind="synthetic",
        graph_source=None,
    )


@pytest.fixture
def dataset_bundle() -> DatasetBundle:
    """Return a dataset bundle with distinct loader sentinels per family."""
    features = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    costs = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=float)
    return DatasetBundle(
        train_loader_adversarial=SimpleNamespace(label="train-adv"),
        val_loader_adversarial=SimpleNamespace(label="val-adv"),
        train_loader_random=SimpleNamespace(label="train-rand"),
        val_loader_random=SimpleNamespace(label="val-rand"),
        train_loader_baseline=SimpleNamespace(label="train-base"),
        val_loader_baseline=SimpleNamespace(label="val-base"),
        testing_features=features,
        testing_costs=costs,
        interdiction_features=features + 10.0,
        interdiction_costs=costs + 10.0,
        normalization_constant=12.5,
        data_generator_adversarial=None,
        data_generator_random=None,
    )


############################
### Helper functionality ###
############################


def _training_log_bundle(offset: float = 0.0) -> TrainingLogBundle:
    """Return a distinct typed log bundle for one predictor family."""
    return TrainingLogBundle(
        train_loss=[1.0 + offset, 0.5 + offset],
        train_regret=[2.0 + offset, 1.0 + offset],
        val_loss=[1.5 + offset, 0.75 + offset],
        val_regret=[2.5 + offset, 1.5 + offset],
    )


def _predictor_stub(tag: str) -> SimpleNamespace:
    """Return a unique predictor sentinel for one family."""
    return SimpleNamespace(tag=tag)


def test_spni_train_legacy_config_adapter_exposes_hashable_pred_keys(run_cfg):
    """Verify that the training adapter exposes keys needed for pred hashes."""
    # Arrange a legacy config adapter for the predictor cache path.
    cfg = train_module._LegacyConfigAdapter(run_cfg)

    # Act by hashing the adapter with the PFL predictor key subset.
    hash_value = _unique_pred_hash(cfg, artifact_tag="pfl")

    # Assert that the adapter exposes the expected predictor fields.
    assert isinstance(hash_value, str) and len(hash_value) == 64, \
        "The training adapter should support non-empty predictor hashing."
    assert cfg.num_features == run_cfg.num_features, \
        "The adapter should expose normalized predictor settings as attributes."
    assert cfg.lam == 0.0, \
        "The adapter should also expose legacy base-config keys for helpers."
    pass


#################################
### test train_pfl_predictor ###
#################################


def test_spni_train_train_pfl_predictor_delegates_to_legacy_helper(
    monkeypatch,
    run_cfg,
    graph_bundle,
    dataset_bundle,
):
    """Verify that PFL training delegates through the legacy helper once."""
    # Arrange a fake capture wrapper so helper delegation stays observable.
    calls: list[dict] = []

    def _fake_setup_pfl_predictor(cfg, graph, opt_model, training_data, **kwargs):
        calls.append(
            {
                "cfg": cfg,
                "graph": graph,
                "opt_model": opt_model,
                "training_data": training_data,
                "kwargs": kwargs,
            }
        )
        return _predictor_stub("pfl")

    def _fake_run_with_fit_capture(trainer_cls, runner):
        return runner(), _training_log_bundle(), True

    monkeypatch.setattr(
        train_module.legacy_setup_module,
        "setup_pfl_predictor",
        _fake_setup_pfl_predictor,
    )
    monkeypatch.setattr(
        train_module,
        "_run_with_fit_capture",
        _fake_run_with_fit_capture,
    )

    # Act by training the PFL predictor through the wrapper layer.
    predictor, log_bundle = train_module.train_pfl_predictor(
        run_cfg,
        graph_bundle,
        dataset_bundle,
    )

    # Assert that the helper saw the adversarial training view and cache tag.
    assert len(calls) == 1, \
        "train_pfl_predictor should delegate to setup_pfl_predictor once."
    assert calls[0]["cfg"].get("pfl_epochs") == run_cfg.pfl_epochs, \
        "PFL training should expose normalized config values via cfg.get(...)."
    assert calls[0]["cfg"].get("lam") == 0.0, \
        "PFL training should fall back to the base config for legacy keys."
    assert calls[0]["graph"] is graph_bundle.graph, \
        "PFL training should forward the graph object unchanged."
    assert calls[0]["opt_model"] is graph_bundle.opt_model, \
        "PFL training should forward the optimization model unchanged."
    assert calls[0]["training_data"]["train_loader"] is \
        dataset_bundle.train_loader_adversarial, \
        "PFL training should use the adversarial train loader."
    assert calls[0]["training_data"]["val_loader"] is \
        dataset_bundle.val_loader_adversarial, \
        "PFL training should use the adversarial validation loader."
    assert calls[0]["kwargs"]["cache_tag"] == "pfl", \
        "PFL training should use the stable `pfl` cache tag."
    assert calls[0]["kwargs"]["verbose"] is False, \
        "PFL training should keep visualization disabled in the wrapper."
    assert calls[0]["kwargs"]["cache_options"].replace_pred is True, \
        "PFL training should translate the cache policy into cache options."
    assert predictor.tag == "pfl", \
        "train_pfl_predictor should return the predictor from the helper."
    assert isinstance(log_bundle, TrainingLogBundle), \
        "train_pfl_predictor should return a typed TrainingLogBundle."
    assert log_bundle.train_loss == [1.0, 0.5], \
        "PFL training should preserve the training loss curve."
    assert log_bundle.val_regret == [2.5, 1.5], \
        "PFL training should preserve the validation regret curve."
    pass


def test_spni_train_train_pfl_predictor_repairs_missing_read_cache(
    monkeypatch,
    run_cfg,
    graph_bundle,
    dataset_bundle,
):
    """Verify that the wrapper patches read_cache into the legacy setup module."""
    # Arrange a legacy helper stub that expects read_cache to exist globally.
    sentinel_predictor = _predictor_stub("pfl-cache")
    monkeypatch.delattr(
        train_module.legacy_setup_module,
        "read_cache",
        raising=False,
    )

    def _fake_setup_pfl_predictor(cfg, graph, opt_model, training_data, **kwargs):
        assert train_module.legacy_setup_module.read_cache is \
            train_module.read_cache, \
            "The wrapper should repair setup.py with the shared read_cache."
        return sentinel_predictor

    def _fake_run_with_fit_capture(trainer_cls, runner):
        return runner(), _training_log_bundle(), False

    monkeypatch.setattr(
        train_module.legacy_setup_module,
        "setup_pfl_predictor",
        _fake_setup_pfl_predictor,
    )
    monkeypatch.setattr(
        train_module,
        "_run_with_fit_capture",
        _fake_run_with_fit_capture,
    )

    # Act by training through the compatibility wrapper.
    predictor, log_bundle = train_module.train_pfl_predictor(
        run_cfg,
        graph_bundle,
        dataset_bundle,
    )

    # Assert that the predictor path succeeds after the namespace repair.
    assert predictor is sentinel_predictor, \
        "The wrapper should still return the helper's predictor."
    assert log_bundle.train_loss == [1.0, 0.5], \
        "The repaired setup path should still return the captured logs."
    pass


#################################
### test train_dfl_predictor ###
#################################


@pytest.mark.parametrize(
    (
        "variant_name",
        "expected_tag",
        "expected_train_loader_attr",
        "expected_val_loader_attr",
        "expected_dfl_variant",
    ),
    [
        ("dfl", "dfl", "train_loader_baseline", "val_loader_baseline", None),
        ("rdfl", "rdfl", "train_loader_random", "val_loader_random", "a-dfl"),
        (
            "adfl",
            "adfl",
            "train_loader_adversarial",
            "val_loader_adversarial",
            "a-dfl",
        ),
    ],
)
def test_spni_train_train_dfl_predictor_uses_explicit_family_mapping(
    monkeypatch,
    run_cfg,
    graph_bundle,
    dataset_bundle,
    variant_name,
    expected_tag,
    expected_train_loader_attr,
    expected_val_loader_attr,
    expected_dfl_variant,
):
    """Verify that DFL-family training maps families to stable loaders/tags."""
    # Arrange a fake capture wrapper so the helper arguments are inspectable.
    calls: list[dict] = []

    def _fake_setup_dfl_predictor(
        cfg,
        graph,
        opt_model,
        training_data,
        **kwargs,
    ):
        calls.append(
            {
                "cfg": cfg,
                "graph": graph,
                "opt_model": opt_model,
                "training_data": training_data,
                "kwargs": kwargs,
            }
        )
        return _predictor_stub(expected_tag)

    def _fake_run_with_fit_capture(trainer_cls, runner):
        return runner(), _training_log_bundle(offset=1.0), True

    monkeypatch.setattr(
        train_module.legacy_setup_module,
        "setup_dfl_predictor",
        _fake_setup_dfl_predictor,
    )
    monkeypatch.setattr(
        train_module,
        "_run_with_fit_capture",
        _fake_run_with_fit_capture,
    )

    # Act by training the requested DFL-family predictor.
    predictor, log_bundle = train_module.train_dfl_predictor(
        run_cfg,
        graph_bundle,
        dataset_bundle,
        variant_name=variant_name,
    )

    # Assert that the helper saw the expected family-specific mapping.
    assert len(calls) == 1, \
        "train_dfl_predictor should delegate to setup_dfl_predictor once."
    assert calls[0]["cfg"].get("dfl_epochs") == run_cfg.dfl_epochs, \
        "DFL training should expose normalized config values via cfg.get(...)."
    assert calls[0]["graph"] is graph_bundle.graph, \
        "DFL training should forward the graph object unchanged."
    assert calls[0]["opt_model"] is graph_bundle.opt_model, \
        "DFL training should forward the optimization model unchanged."
    assert calls[0]["training_data"]["train_loader"] is \
        getattr(dataset_bundle, expected_train_loader_attr), \
        "DFL-family training should use the variant-specific train loader."
    assert calls[0]["training_data"]["val_loader"] is \
        getattr(dataset_bundle, expected_val_loader_attr), \
        "DFL-family training should use the variant-specific val loader."
    assert calls[0]["kwargs"]["cache_tag"] == expected_tag, \
        "The DFL-family cache tag should match the requested family."
    assert calls[0]["kwargs"]["cache_options"].replace_pred is True, \
        "DFL training should translate the cache policy into cache options."
    if expected_dfl_variant is None:
        assert "dfl_variant" not in calls[0]["kwargs"], \
            "Baseline DFL should preserve the legacy helper default."
    else:
        assert calls[0]["kwargs"]["dfl_variant"] == expected_dfl_variant, \
            "Adverse DFL families should forward the explicit variant."
    assert predictor.tag == expected_tag, \
        "train_dfl_predictor should return the predictor from the helper."
    assert isinstance(log_bundle, TrainingLogBundle), \
        "train_dfl_predictor should return a typed TrainingLogBundle."
    assert log_bundle.train_regret == [3.0, 2.0], \
        "DFL-family training should preserve the regret curve."
    pass


def test_spni_train_train_dfl_predictor_rejects_unknown_family(
    run_cfg,
    graph_bundle,
    dataset_bundle,
):
    """Verify that unknown DFL families fail with a precise error."""
    # Act and assert that the family validation rejects unsupported names.
    with pytest.raises(ValueError, match="Unsupported DFL predictor family"):
        train_module.train_dfl_predictor(
            run_cfg,
            graph_bundle,
            dataset_bundle,
            variant_name="mystery",
        )

    pass


def test_spni_train_train_pfl_predictor_captures_fit_logs_from_trainer(
    monkeypatch,
    run_cfg,
    graph_bundle,
    dataset_bundle,
):
    """Verify that PFL training captures trainer fit logs into a bundle."""
    # Arrange a trainer stub whose fit output mimics the legacy trainer API.
    fit_calls: list[dict] = []

    class _TrainerStub:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def fit(self, train_loader, val_loader):
            fit_calls.append(
                {
                    "train_loader": train_loader,
                    "val_loader": val_loader,
                }
            )
            return (
                np.array([4.0, 2.0], dtype=float),
                np.array([3.0, 1.0], dtype=float),
                np.array([2.5, 1.5], dtype=float),
                np.array([5.0, 2.5], dtype=float),
            )

    def _fake_setup_pfl_predictor(cfg, graph, opt_model, training_data, **kwargs):
        trainer = train_module.legacy_setup_module.PFLTrainer()
        trainer.fit(
            training_data["train_loader"],
            training_data["val_loader"],
        )
        return _predictor_stub("pfl-captured")

    monkeypatch.setattr(
        train_module.legacy_setup_module,
        "PFLTrainer",
        _TrainerStub,
    )
    monkeypatch.setattr(
        train_module.legacy_setup_module,
        "setup_pfl_predictor",
        _fake_setup_pfl_predictor,
    )

    # Act by training through the real fit-capture wrapper.
    predictor, log_bundle = train_module.train_pfl_predictor(
        run_cfg,
        graph_bundle,
        dataset_bundle,
    )

    # Assert that the captured fit result became a typed log bundle.
    assert predictor.tag == "pfl-captured", \
        "The wrapper should return the predictor produced by the helper."
    assert fit_calls == [
        {
            "train_loader": dataset_bundle.train_loader_adversarial,
            "val_loader": dataset_bundle.val_loader_adversarial,
        }
    ], \
        "The fake helper should invoke trainer.fit on the adversarial view."
    assert log_bundle.train_loss == [4.0, 2.0], \
        "Captured train-loss logs should be normalized into Python floats."
    assert log_bundle.train_regret == [3.0, 1.0], \
        "Captured train-regret logs should be preserved."
    assert log_bundle.val_loss == [2.5, 1.5], \
        "Captured validation-loss logs should be preserved."
    assert log_bundle.val_regret == [5.0, 2.5], \
        "Captured validation-regret logs should be preserved."
    pass


#####################################
### test train_all_predictors(...) ###
#####################################


def test_spni_train_train_all_predictors_returns_full_predictor_bundle(
    monkeypatch,
    run_cfg,
    graph_bundle,
    dataset_bundle,
):
    """Verify that all four predictor families are assembled once."""
    # Arrange family-specific wrappers so aggregation stays isolated.
    calls: list[tuple[str, str | None]] = []

    def _fake_train_pfl_predictor(cfg, graph, data):
        calls.append(("pfl", None))
        return _predictor_stub("pfl"), _training_log_bundle()

    def _fake_train_dfl_predictor(cfg, graph, data, *, variant_name):
        calls.append(("dfl", variant_name))
        offsets = {"dfl": 1.0, "rdfl": 2.0, "adfl": 3.0}
        return _predictor_stub(variant_name), _training_log_bundle(
            offset=offsets[variant_name]
        )

    monkeypatch.setattr(
        train_module,
        "train_pfl_predictor",
        _fake_train_pfl_predictor,
    )
    monkeypatch.setattr(
        train_module,
        "train_dfl_predictor",
        _fake_train_dfl_predictor,
    )

    # Act by building the complete predictor bundle for one run.
    bundle = train_module.train_all_predictors(
        run_cfg,
        graph_bundle,
        dataset_bundle,
    )

    # Assert that all predictor families and diagnostics are present.
    assert isinstance(bundle, PredictorBundle), \
        "train_all_predictors should return a PredictorBundle."
    assert bundle.pfl.tag == "pfl", \
        "The PFL predictor should be stored on the bundle."
    assert bundle.dfl.tag == "dfl", \
        "The DFL predictor should be stored on the bundle."
    assert bundle.rdfl.tag == "rdfl", \
        "The random DFL predictor should be stored on the bundle."
    assert bundle.adfl.tag == "adfl", \
        "The adversarial DFL predictor should be stored on the bundle."
    assert calls == [
        ("pfl", None),
        ("dfl", "dfl"),
        ("dfl", "rdfl"),
        ("dfl", "adfl"),
    ], \
        "train_all_predictors should cover PFL, DFL, R-DFL, and A-DFL once."
    assert set(bundle.logs) == {"pfl", "dfl", "rdfl", "adfl"}, \
        "train_all_predictors should retain logs for every family."
    assert bundle.logs["adfl"].val_loss == [4.5, 3.75], \
        "The adversarial log bundle should be preserved in the bundle."
    assert bundle.diagnostics["cache_tags"] == {
        "pfl": "pfl",
        "dfl": "dfl",
        "rdfl": "rdfl",
        "adfl": "adfl",
    }, \
        "The predictor bundle should expose the explicit cache-tag mapping."
    pass
