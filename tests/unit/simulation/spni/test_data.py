from types import SimpleNamespace

import numpy as np
import pytest

from dflintdpy.simulation.spni.config import SPNIRunConfig
from dflintdpy.simulation.spni.types import GraphBundle
from dflintdpy.utils.read_write import _unique_hash

import dflintdpy.simulation.spni.data as data_module


################
### Fixtures ###
################


@pytest.fixture
def run_cfg() -> SPNIRunConfig:
    """Return a compact run config for dataset-stage tests."""
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
        metadata={"source_type": "SimpleNamespace"},
    )


@pytest.fixture
def graph_bundle() -> GraphBundle:
    """Return a lightweight graph bundle for dataset-stage tests."""
    return GraphBundle(
        graph=SimpleNamespace(name="graph"),
        opt_model=SimpleNamespace(name="opt-model"),
        graph_kind="synthetic",
    )


@pytest.fixture
def base_features() -> np.ndarray:
    """Return deterministic synthetic features for split tests."""
    return np.arange(24, dtype=float).reshape(8, 3)


@pytest.fixture
def base_costs() -> np.ndarray:
    """Return deterministic synthetic costs for split tests."""
    return np.array(
        [
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
            [4.0, 8.0, 12.0],
            [5.0, 10.0, 15.0],
            [6.0, 12.0, 18.0],
            [7.0, 14.0, 21.0],
            [8.0, 16.0, 24.0],
            [9.0, 18.0, 27.0],
        ],
        dtype=float,
    )


############################
### Helper functionality ###
############################


def test_spni_data_legacy_config_adapter_exposes_hashable_intd_keys(run_cfg):
    """Verify that the data-stage adapter exposes keys needed for intd hashes."""
    # Arrange a legacy config adapter with one temporary seed override.
    cfg = data_module._build_legacy_cfg(
        run_cfg,
        random_seed=run_cfg.intd_seed,
    )

    # Act by hashing the adapter with the legacy interdiction key subset.
    hash_value = _unique_hash(cfg, type="intd")

    # Assert that the adapter exposes the expected public config fields.
    assert isinstance(hash_value, str) and len(hash_value) == 64, \
        "The data-stage adapter should support non-empty intd hashing."
    assert cfg.budget == run_cfg.budget, \
        "The adapter should expose normalized budget values as attributes."
    assert cfg.random_seed == run_cfg.intd_seed, \
        "Temporary overrides should be visible as real adapter attributes."
    pass


class _GeneratorStub:
    """Record generator construction and emit grouped scenario arrays."""

    init_calls: list[dict] = []
    generate_calls: list[dict] = []

    def __init__(
        self,
        cfg,
        opt_model,
        budget,
        normalization_constant,
        **kwargs,
    ):
        self.cfg = cfg
        self.opt_model = opt_model
        self.budget = budget
        self.normalization_constant = normalization_constant
        self.kwargs = kwargs
        self.policy = kwargs["interdiction_policy"]
        type(self).init_calls.append(
            {
                "cfg": cfg,
                "opt_model": opt_model,
                "budget": budget,
                "normalization_constant": normalization_constant,
                "kwargs": kwargs,
            }
        )

    @classmethod
    def reset(cls) -> None:
        """Clear recorded constructor and generate calls."""
        cls.init_calls = []
        cls.generate_calls = []

    def generate(self, features, costs, cfg=None):
        """Return grouped scenario arrays keyed by the interdiction policy."""
        type(self).generate_calls.append(
            {
                "policy": self.policy,
                "features": np.array(features, dtype=float),
                "costs": np.array(costs, dtype=float),
                "cfg": cfg,
            }
        )
        n_samples, n_cost = costs.shape
        num_scenarios = self.kwargs["num_scenarios"]
        grouped_costs = np.repeat(costs[:, np.newaxis, :], num_scenarios, axis=1)
        grouped_intds = np.zeros_like(grouped_costs)
        offset = 5.0 if self.policy == "adversarial" else 9.0
        grouped_costs[:, 1:, :] = grouped_costs[:, 1:, :] + offset
        grouped_intds[:, 1:, :] = offset
        return np.array(features, dtype=float), grouped_costs, grouped_intds


class _DatasetRecorder:
    """Record dataset-construction arguments for loader-stage tests."""

    init_calls: list[dict] = []

    def __init__(self, opt_model, feats, costs_grouped, intds_grouped):
        type(self).init_calls.append(
            {
                "opt_model": opt_model,
                "feats": np.array(feats, dtype=float),
                "costs_grouped": np.array(costs_grouped, dtype=float),
                "intds_grouped": np.array(intds_grouped, dtype=float),
            }
        )
        self.opt_model = opt_model
        self.feats = np.array(feats, dtype=float)
        self.costs = np.array(costs_grouped, dtype=float)
        self.intds = np.array(intds_grouped, dtype=float)

    @classmethod
    def reset(cls) -> None:
        """Clear recorded dataset-construction calls."""
        cls.init_calls = []


class _LoaderStub:
    """Record loader construction and expose a nonadverse-view helper."""

    init_calls: list[dict] = []

    def __init__(self, dataset, *, batch_size, seed, shuffle):
        type(self).init_calls.append(
            {
                "dataset": dataset,
                "batch_size": batch_size,
                "seed": seed,
                "shuffle": shuffle,
            }
        )
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.loader = SimpleNamespace(batch_size=batch_size)
        self.sampler = SimpleNamespace(seed=seed, shuffle=shuffle)

    @classmethod
    def reset(cls) -> None:
        """Clear recorded loader-construction calls."""
        cls.init_calls = []

    def get_nonadverse_loader(self):
        """Return a deterministic baseline-view sentinel."""
        return SimpleNamespace(
            label=f"baseline-{id(self)}",
            source_loader=self,
            batch_size=self.batch_size,
        )


class _NonadverseLoaderStub:
    """Minimal loader stub for direct nonadverse-view tests."""

    def __init__(self, label: str):
        self.label = label
        self.calls = 0

    def get_nonadverse_loader(self):
        """Return a deterministic nonadverse loader sentinel."""
        self.calls += 1
        return SimpleNamespace(label=f"{self.label}-baseline", source=self)


##############################
### test generate_base_data ###
##############################


def test_spni_data_generate_base_data_delegates_to_gen_syn_data(
    monkeypatch,
    run_cfg,
    graph_bundle,
    base_features,
    base_costs,
):
    """Verify that base-data generation delegates to the legacy helper."""
    calls: list[dict] = []

    def _fake_gen_syn_data(cfg, opt_model=None, seed=None):
        calls.append(
            {
                "cfg": cfg,
                "opt_model": opt_model,
                "seed": seed,
            }
        )
        return base_features, base_costs

    monkeypatch.setattr(data_module, "gen_syn_data", _fake_gen_syn_data)

    base_data = data_module.generate_base_data(run_cfg, graph_bundle)

    assert len(calls) == 1, \
        "generate_base_data should call gen_syn_data exactly once."
    assert calls[0]["opt_model"] is graph_bundle.opt_model, \
        "generate_base_data should pass the graph bundle opt model through."
    assert calls[0]["seed"] is None, \
        "generate_base_data should rely on the normalized random seed."
    assert calls[0]["cfg"].get("random_seed") == run_cfg.random_seed, \
        "generate_base_data should expose run_cfg values via cfg.get(...)."
    assert np.array_equal(base_data.features, base_features), \
        "generate_base_data should return the generated feature array."
    assert np.array_equal(base_data.costs, base_costs), \
        "generate_base_data should return the generated cost array."
    pass


##########################
### test split_base_data ###
##########################


def test_spni_data_split_base_data_returns_exact_counts_and_normalization(
    run_cfg,
    base_features,
    base_costs,
):
    """Verify that the split stage preserves legacy split counts."""
    split_data = data_module.split_base_data(
        run_cfg,
        base_features,
        base_costs,
    )

    assert split_data.normalization_constant == 27.0, \
        "split_base_data should store the max cost as the normalizer."
    assert split_data.trainval_features.shape == (6, 3), \
        "split_base_data should keep train+val samples before scenario gen."
    assert split_data.test_features.shape == (2, 3), \
        "split_base_data should allocate exactly num_test_samples to test."
    assert split_data.train_indices.shape == (4,), \
        "split_base_data should allocate exactly num_train_samples to train."
    assert split_data.val_indices.shape == (2,), \
        "split_base_data should allocate exactly num_val_samples to val."
    assert np.isclose(split_data.trainval_costs.max(), 1.0), \
        "split_base_data should normalize train+val costs by the max cost."
    assert np.isclose(split_data.test_costs.max(), 18/27), \
        "split_base_data should normalize test costs by the same constant."
    assert split_data.diagnostics["train_count"] == run_cfg.num_train_samples, \
        "split diagnostics should record the exact training count."
    assert split_data.diagnostics["val_count"] == run_cfg.num_val_samples, \
        "split diagnostics should record the exact validation count."
    pass


###################################
### test build_spni_training_data ###
###################################


def test_spni_data_build_spni_training_data_builds_both_loader_families(
    monkeypatch,
    run_cfg,
    graph_bundle,
    base_features,
    base_costs,
):
    """Verify that the loader stage builds adversarial and random views."""
    _GeneratorStub.reset()
    _DatasetRecorder.reset()
    _LoaderStub.reset()
    monkeypatch.setattr(
        data_module,
        "SPNIAdverseDataGenerator",
        _GeneratorStub,
    )
    monkeypatch.setattr(data_module, "AdvDataset", _DatasetRecorder)
    monkeypatch.setattr(data_module, "AdvLoader", _LoaderStub)
    split_data = data_module.split_base_data(run_cfg, base_features, base_costs)

    training_views = data_module.build_spni_training_data(
        run_cfg,
        graph_bundle,
        split_data,
    )

    policies = [call["kwargs"]["interdiction_policy"] for call in
                _GeneratorStub.init_calls]
    assert policies == ["adversarial", "random"], \
        "build_spni_training_data should build adversarial and random views."
    assert len(_GeneratorStub.generate_calls) == 2, \
        "build_spni_training_data should generate grouped costs per policy."
    assert np.array_equal(
        _GeneratorStub.generate_calls[0]["features"],
        split_data.trainval_features,
    ), \
        "Loader generation should run on the combined train+val features."
    assert len(_DatasetRecorder.init_calls) == 4, \
        "build_spni_training_data should build train and val datasets twice."
    assert _DatasetRecorder.init_calls[0]["feats"].shape[0] == 4, \
        "The adversarial train dataset should contain num_train_samples."
    assert _DatasetRecorder.init_calls[1]["feats"].shape[0] == 2, \
        "The adversarial val dataset should contain num_val_samples."
    assert _LoaderStub.init_calls[0]["shuffle"] is True, \
        "Training loaders should enable shuffled sampling."
    assert _LoaderStub.init_calls[1]["shuffle"] is False, \
        "Validation loaders should disable shuffled sampling."
    assert training_views["adversarial"].data_generator.policy == \
        "adversarial", \
        "The adversarial view should retain its generator object."
    assert training_views["random"].data_generator.policy == "random", \
        "The random view should retain its generator object."
    pass


#################################
### test build_nonadverse_views ###
#################################


def test_spni_data_build_nonadverse_views_uses_loader_helpers():
    """Verify that baseline views come from loader nonadverse helpers."""
    adverse_train_loader = _NonadverseLoaderStub("train")
    adverse_val_loader = _NonadverseLoaderStub("val")

    train_loader, val_loader = data_module.build_nonadverse_views(
        adverse_train_loader,
        adverse_val_loader,
    )

    assert adverse_train_loader.calls == 1, \
        "build_nonadverse_views should request one train baseline view."
    assert adverse_val_loader.calls == 1, \
        "build_nonadverse_views should request one val baseline view."
    assert train_loader.label == "train-baseline", \
        "build_nonadverse_views should return the train baseline loader."
    assert val_loader.label == "val-baseline", \
        "build_nonadverse_views should return the val baseline loader."
    pass


####################################
### test assemble_dataset_bundle ###
####################################


def test_spni_data_assemble_dataset_bundle_collects_loaders_and_intd_views(
    monkeypatch,
    run_cfg,
    graph_bundle,
    base_features,
):
    """Verify that dataset assembly returns every training/eval view."""
    split_data = SimpleNamespace(
        test_features=base_features[: run_cfg.num_test_samples],
        test_costs=np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            dtype=float,
        ),
        normalization_constant=20.0,
        diagnostics={
            "train_count": 4,
            "val_count": 2,
            "test_count": 2,
        },
    )
    training_views = {
        "adversarial": SimpleNamespace(
            train_loader=SimpleNamespace(label="adv-train"),
            val_loader=SimpleNamespace(label="adv-val"),
            data_generator=SimpleNamespace(label="adv-generator"),
            diagnostics={"policy": "adversarial"},
        ),
        "random": SimpleNamespace(
            train_loader=SimpleNamespace(label="rnd-train"),
            val_loader=SimpleNamespace(label="rnd-val"),
            data_generator=SimpleNamespace(label="rnd-generator"),
            diagnostics={"policy": "random"},
        ),
    }
    base_data = SimpleNamespace(
        features=base_features,
        costs=np.ones((8, 3), dtype=float),
        diagnostics={"num_samples": 8},
    )
    gen_calls: list[dict] = []

    def _fake_generate_base_data(cfg, bundle):
        return base_data

    def _fake_split_base_data(cfg, features, costs):
        return split_data

    def _fake_build_spni_training_data(cfg, bundle, split_result):
        return training_views

    def _fake_build_nonadverse_views(train_loader, val_loader):
        return (
            SimpleNamespace(label="baseline-train", source=train_loader),
            SimpleNamespace(label="baseline-val", source=val_loader),
        )

    def _fake_gen_syn_data(cfg, opt_model=None, seed=None):
        gen_calls.append(
            {
                "cfg": cfg,
                "opt_model": opt_model,
                "seed": seed,
            }
        )
        return (
            np.array(
                [
                    [100.0, 101.0, 102.0],
                    [200.0, 201.0, 202.0],
                    [300.0, 301.0, 302.0],
                ],
                dtype=float,
            ),
            np.array(
                [
                    [10.0, 20.0, 30.0],
                    [40.0, 50.0, 60.0],
                    [70.0, 80.0, 90.0],
                ],
                dtype=float,
            ),
        )

    monkeypatch.setattr(
        data_module,
        "generate_base_data",
        _fake_generate_base_data,
    )
    monkeypatch.setattr(
        data_module,
        "split_base_data",
        _fake_split_base_data,
    )
    monkeypatch.setattr(
        data_module,
        "build_spni_training_data",
        _fake_build_spni_training_data,
    )
    monkeypatch.setattr(
        data_module,
        "build_nonadverse_views",
        _fake_build_nonadverse_views,
    )
    monkeypatch.setattr(data_module, "gen_syn_data", _fake_gen_syn_data)

    bundle = data_module.assemble_dataset_bundle(run_cfg, graph_bundle)

    assert bundle.train_loader_adversarial.label == "adv-train", \
        "assemble_dataset_bundle should retain the adversarial train loader."
    assert bundle.val_loader_random.label == "rnd-val", \
        "assemble_dataset_bundle should retain the random val loader."
    assert bundle.train_loader_baseline.label == "baseline-train", \
        "assemble_dataset_bundle should attach the baseline train loader."
    assert bundle.val_loader_baseline.label == "baseline-val", \
        "assemble_dataset_bundle should attach the baseline val loader."
    assert np.array_equal(bundle.testing_features, split_data.test_features), \
        "assemble_dataset_bundle should keep the test feature slice."
    assert np.array_equal(bundle.testing_costs, split_data.test_costs), \
        "assemble_dataset_bundle should keep the normalized test costs."
    assert bundle.normalization_constant == 20.0, \
        "assemble_dataset_bundle should expose the split normalizer."
    assert len(gen_calls) == 1, \
        "assemble_dataset_bundle should generate evaluation interdictions."
    assert gen_calls[0]["seed"] == run_cfg.intd_seed, \
        "Evaluation interdictions should use the configured intd seed."
    assert gen_calls[0]["cfg"].get("random_seed") == run_cfg.intd_seed, \
        "Evaluation interdictions should adapt cfg.get(random_seed)."
    assert np.array_equal(
        bundle.interdiction_features,
        np.array(
            [[100.0, 101.0, 102.0], [200.0, 201.0, 202.0]],
            dtype=float,
        ),
    ), \
        "assemble_dataset_bundle should keep only the evaluation slice."
    assert np.allclose(
        bundle.interdiction_costs,
        np.array([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]], dtype=float),
    ), \
        "assemble_dataset_bundle should normalize interdiction costs once."
    assert bundle.data_generator_adversarial.label == "adv-generator", \
        "assemble_dataset_bundle should retain the adversarial generator."
    assert bundle.data_generator_random.label == "rnd-generator", \
        "assemble_dataset_bundle should retain the random generator."
    pass
