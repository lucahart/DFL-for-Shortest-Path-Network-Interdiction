import json
from pathlib import Path

import numpy as np
import pytest

from dflintdpy.data.config import HP
from dflintdpy.simulation.spni.config import (
    CachePolicy,
    build_run_config,
    describe_run,
    derive_seed_bundle,
    derive_seed_sweep,
)


################
### Fixtures ###
################

@pytest.fixture
def hp() -> HP:
    """Return a compact legacy config for SPNI config-layer tests."""
    cfg = HP()
    cfg.set("grid_size", [4, 6])
    cfg.set("num_features", 7)
    cfg.set("num_train_samples", 123)
    cfg.set("num_val_samples", 11)
    cfg.set("num_test_samples", 29)
    cfg.set("batch_size", 8)
    cfg.set("budget", 2)
    cfg.set("num_scenarios", 5)
    cfg.set("deg", 3)
    cfg.set("noise_width", 0.125)
    cfg.set("benders_max_count", 77)
    cfg.set("benders_eps", 1e-4)
    cfg.set("lsd", 2e-5)
    cfg.set("seed", 104)
    cfg.set("random_seed", 17)
    cfg.set("intd_seed", 31)
    cfg.set("loader_seed", 47)
    cfg.set("pred_model", "linear")
    cfg.set("po_epochs", 9)
    cfg.set("spo_epochs", 12)
    cfg.set("po_lr", 5e-4)
    cfg.set("spo_lr", 8e-4)
    cfg.set("num_seeds", 4)
    cfg.set("seed_sweep_offset", 104)
    cfg.set("metadata", {"label": "phase1"})
    return cfg


############################
### Helper functionality ###
############################

def _seed_triplet(seed: int) -> tuple[int, int, int]:
    """Return the legacy sweep-derived triplet for one sweep seed."""
    rng = np.random.RandomState(seed)
    values = rng.randint(0, 150, 3).tolist()
    return int(values[0]), int(values[1]), int(values[2])


########################
### Regression tests ###
########################

def test_spni_config_hp_default_disables_underprediction_safeguard():
    """Verify that SPO+ underprediction flooring is opt-in by default."""
    # Arrange a fresh legacy config.
    cfg = HP()

    # Act by reading the safeguard weight used by setup_dfl_predictor.
    weight = cfg.get("surrogate_underprediction_penalty_weight")

    # Assert that ordinary simulations do not activate the safeguard.
    assert weight == 0.0, \
        "HP should leave the SPO+ underprediction safeguard disabled."
    pass


#################################
### test build_run_config(...) ###
#################################

def test_spni_config_build_run_config_normalizes_hp_and_flags(hp: HP):
    """Verify that `build_run_config` normalizes the legacy HP object."""
    # Arrange runtime-only orchestration overrides.
    cache_policy = CachePolicy(replace_data=True, replace_pred=True)

    # Act by normalizing the legacy config.
    run_cfg = build_run_config(
        hp,
        compute_asym_intd=False,
        compute_wrong_asym_intd=True,
        load_real_world_graph=Path("graphs/sample.csv"),
        cache_policy=cache_policy,
    )

    # Assert that the normalized config keeps the expected values.
    assert run_cfg.base_cfg is hp, \
        "The normalized config should retain the original base config."
    assert run_cfg.grid_size == (4, 6), \
        "grid_size should be normalized into a tuple of ints."
    assert run_cfg.batch_size == 8, \
        "batch_size should be copied into the run config."
    assert run_cfg.compute_asym_intd is False, \
        "compute_asym_intd should reflect the orchestration override."
    assert run_cfg.compute_wrong_asym_intd is True, \
        "compute_asym_intd_2 should reflect the orchestration override."
    assert run_cfg.load_real_world_graph == "graphs/sample.csv", \
        "The real-world graph path should be preserved as a string."
    assert run_cfg.cache_policy == cache_policy, \
        "The provided cache policy should be stored unchanged."
    assert run_cfg.metadata["label"] == "phase1", \
        "Custom metadata should be preserved during normalization."
    assert run_cfg.metadata["source_type"] == "HP", \
        "The source config type should be recorded for debugging."
    assert run_cfg.metadata["legacy_num_seeds"] == 4, \
        "The legacy sweep width should be captured in metadata."
    assert run_cfg.metadata["legacy_seed_sweep_offset"] == 104, \
        "The legacy sweep offset should be captured in metadata."
    pass


def test_spni_config_build_run_config_supports_mapping_inputs():
    """Verify that mapping-style configs are normalized correctly."""
    # Arrange a minimal mapping-based legacy config.
    raw_cfg = {
        "grid_size": [2, 3],
        "num_features": 5,
        "num_train_samples": 10,
        "num_val_samples": 0,
        "num_test_samples": 4,
        "batch_size": 2,
        "budget": 1,
        "num_scenarios": 3,
        "deg": 2,
        "noise_width": 0.0,
        "benders_max_count": 5,
        "benders_eps": 1e-3,
        "lsd": 1e-5,
        "seed": 7,
        "random_seed": 8,
        "intd_seed": 9,
        "loader_seed": 10,
        "pred_model": None,
        "po_epochs": 1,
        "spo_epochs": 2,
        "po_lr": 1e-3,
        "spo_lr": 2e-3,
    }

    # Act by building the normalized config.
    run_cfg = build_run_config(raw_cfg)

    # Assert that mapping values were read and normalized.
    assert run_cfg.grid_size == (2, 3), \
        "Mapping-based grid_size should be normalized correctly."
    assert run_cfg.pred_model is None, \
        "A None pred_model should remain unset."
    assert run_cfg.metadata["source_type"] == "dict", \
        "The metadata should record mapping-based inputs cleanly."
    pass


def test_spni_config_build_run_config_accepts_mapping_cache_policy(hp: HP):
    """Verify that mapping-based cache policies are normalized correctly."""
    # Arrange a mapping-shaped cache policy.
    raw_policy = {
        "replace_data": True,
        "replace_pred": True,
        "archive_replaced": False,
    }

    # Act by normalizing the config with that policy.
    run_cfg = build_run_config(hp, cache_policy=raw_policy)

    # Assert that the policy became the canonical dataclass.
    assert isinstance(run_cfg.cache_policy, CachePolicy), \
        "Cache policies should be normalized into CachePolicy."
    assert run_cfg.cache_policy.replace_data is True, \
        "replace_data should be copied from the mapping policy."
    assert run_cfg.cache_policy.replace_pred is True, \
        "replace_pred should be copied from the mapping policy."
    assert run_cfg.cache_policy.archive_replaced is False, \
        "archive_replaced should be copied from the mapping policy."
    pass


def test_spni_config_build_run_config_rejects_missing_required_fields(hp: HP):
    """Verify that missing required config fields fail fast."""
    # Arrange a config with one required field removed.
    hp.set("batch_size", None)

    # Act / Assert that normalization fails with a targeted error.
    with pytest.raises(ValueError, match="Missing required config field"):
        build_run_config(hp)
    pass


def test_spni_config_build_run_config_rejects_invalid_grid_size(hp: HP):
    """Verify that invalid grid_size values are rejected clearly."""
    # Arrange a malformed grid_size value.
    hp.set("grid_size", [4])

    # Act / Assert that the malformed shape is rejected.
    with pytest.raises(ValueError, match="grid_size"):
        build_run_config(hp)
    pass


#################################
### test derive_seed_bundle ###
#################################

def test_spni_config_derive_seed_bundle_uses_explicit_cfg_seeds(hp: HP):
    """Verify that no-override seed bundles preserve explicit config seeds."""
    # Arrange the normalized config.
    run_cfg = build_run_config(hp)

    # Act by deriving the default bundle.
    bundle = derive_seed_bundle(run_cfg)

    # Assert that single-run semantics preserve the stored seed fields.
    assert bundle.sweep_seed == 104, \
        "The default sweep seed should come from run_cfg.seed."
    assert bundle.random_seed == 17, \
        "The default random_seed should preserve the config value."
    assert bundle.intd_seed == 31, \
        "The default intd_seed should preserve the config value."
    assert bundle.loader_seed == 47, \
        "The default loader_seed should preserve the config value."
    pass


def test_spni_config_derive_seed_bundle_override_matches_legacy_triplet(hp: HP):
    """Verify that override seeds use the legacy sweep-triplet rule."""
    # Arrange the normalized config and expected sweep-derived triplet.
    run_cfg = build_run_config(hp)
    expected = _seed_triplet(111)

    # Act by overriding the sweep seed.
    bundle = derive_seed_bundle(run_cfg, sweep_seed=111)

    # Assert that the legacy seed convention is reproduced exactly.
    assert bundle.sweep_seed == 111, \
        "The override sweep seed should be stored in the bundle."
    assert (
        bundle.random_seed,
        bundle.intd_seed,
        bundle.loader_seed,
    ) == expected, \
        "The override triplet should match the legacy sweep convention."
    pass


def test_spni_config_derive_seed_bundle_does_not_mutate_global_rng(hp: HP):
    """Verify that sweep derivation leaves NumPy's global RNG untouched."""
    # Arrange a normalized config and a saved RNG state.
    run_cfg = build_run_config(hp)
    np.random.seed(999)
    state_before = np.random.get_state()

    # Act by deriving a sweep-based bundle.
    derive_seed_bundle(run_cfg, sweep_seed=111)
    state_after = np.random.get_state()

    # Assert that the global RNG state is unchanged.
    assert state_before[0] == state_after[0], \
        "The RNG bit-generator name should remain unchanged."
    assert np.array_equal(state_before[1], state_after[1]), \
        "The RNG state array should remain unchanged."
    assert state_before[2] == state_after[2], \
        "The RNG position should remain unchanged."
    assert state_before[3] == state_after[3], \
        "The RNG cached Gaussian flag should remain unchanged."
    assert state_before[4] == state_after[4], \
        "The RNG cached Gaussian value should remain unchanged."
    pass


################################
### test derive_seed_sweep ###
################################

def test_spni_config_derive_seed_sweep_is_stable_and_ordered(hp: HP):
    """Verify that sweep bundles are reproducible and ordered by seed."""
    # Arrange the normalized config.
    run_cfg = build_run_config(hp)

    # Act by deriving the same sweep twice.
    first = derive_seed_sweep(run_cfg, num_seeds=3)
    second = derive_seed_sweep(run_cfg, num_seeds=3)

    # Assert deterministic order and legacy triplet contents.
    assert first == second, \
        "Repeated sweep derivation should be deterministic."
    assert [bundle.sweep_seed for bundle in first] == [104, 105, 106], \
        "Sweep seeds should increment from the run seed in order."
    assert (
        first[0].random_seed,
        first[0].intd_seed,
        first[0].loader_seed,
    ) == _seed_triplet(104), \
        "The first sweep bundle should match the legacy seed triplet."
    assert len(set(first)) == 3, \
        "The sweep should produce unique bundles for distinct sweep seeds."
    pass


def test_spni_config_derive_seed_sweep_uses_legacy_seed_offset(hp: HP):
    """Verify that sweep generation starts from the legacy seed offset."""
    # Arrange a config where the current seed differs from the sweep offset.
    hp.set("seed", 999)
    hp.set("seed_sweep_offset", 200)
    run_cfg = build_run_config(hp)

    # Act by deriving a short sweep.
    bundles = derive_seed_sweep(run_cfg, num_seeds=3)

    # Assert that the sweep starts from the legacy offset.
    assert [bundle.sweep_seed for bundle in bundles] == [200, 201, 202], \
        "The sweep should start from seed_sweep_offset when available."
    assert (
        bundles[0].random_seed,
        bundles[0].intd_seed,
        bundles[0].loader_seed,
    ) == _seed_triplet(200), \
        "The first offset-based seed bundle should match the legacy rule."
    pass


def test_spni_config_derive_seed_sweep_rejects_non_positive_counts(hp: HP):
    """Verify that invalid sweep widths fail fast."""
    # Arrange the normalized config.
    run_cfg = build_run_config(hp)

    # Act / Assert that an empty sweep is rejected.
    with pytest.raises(ValueError, match="num_seeds"):
        derive_seed_sweep(run_cfg, num_seeds=0)
    pass


############################
### test describe_run ###
############################

def test_spni_config_describe_run_returns_compact_serializable_view(hp: HP):
    """Verify that `describe_run` returns a JSON-serializable debug view."""
    # Arrange the normalized config.
    run_cfg = build_run_config(hp)

    # Act by building the debug description.
    description = describe_run(run_cfg)

    # Assert that the view is compact and serializable.
    assert "base_cfg" not in description, \
        "describe_run should not expose the original config object."
    assert description["grid_size"] == [4, 6], \
        "describe_run should emit a JSON-friendly grid_size list."
    assert description["compute_wrong_asym_intd"] is False, \
        "describe_run should expose the wrong-model flag by name."
    assert description["sweep_start_seed"] == 104, \
        "describe_run should expose the resolved sweep start seed."
    assert description["seed_bundle"] == {
        "sweep_seed": 104,
        "random_seed": 17,
        "intd_seed": 31,
        "loader_seed": 47,
    }, "describe_run should expose the current seed bundle explicitly."
    assert json.loads(json.dumps(description)) == description, \
        "describe_run output should round-trip through JSON."
    pass
