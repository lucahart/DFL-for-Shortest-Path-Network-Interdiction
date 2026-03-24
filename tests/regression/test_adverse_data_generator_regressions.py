import warnings

import numpy as np
import pytest

import dflintdpy.data.adverse.adverse_data_generator as adg_module
from dflintdpy.data.adverse.adverse_data_generator import AdvDataGenerator
from dflintdpy.data.config import HP


################
### Fixtures ###
################


@pytest.fixture
def cfg() -> HP:
    return HP(num_features=3, deg=5, noise_width=0.2, benders_eps=1e-4)


@pytest.fixture
def opt_model_stub():
    class _OptModelStub:
        def __init__(self):
            self.num_cost = 3
            self._graph = {"name": "graph-stub"}
            self.c = np.array([4.0, 2.0, 1.0], dtype=float)
            self.Sigma = np.eye(3, dtype=float)
            self.gamma = 0.3

    return _OptModelStub()


############################
### Helper functionality ###
############################


class _SymStub:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def _patch_lightweight_spni_dependencies(monkeypatch):
    # Keep constructor tests lightweight by replacing heavy collaborators.
    monkeypatch.setattr(adg_module, "SymmetricInterdictor", _SymStub)
    monkeypatch.setattr(
        AdvDataGenerator,
        "gen_interdictions",
        staticmethod(lambda *args, **kwargs: np.ones((2, 3), dtype=float)),
    )


#####################
### test __init__ ###
#####################


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Constructor forwards n_training_interdictions via **kwargs into "
        "gen_interdictions and should not."
    ),
)
def test_adv_data_generator_regression_init_filters_training_count_kwarg(
    cfg, opt_model_stub, monkeypatch
):
    """Document that n_training_interdictions leaks into gen_interdictions."""
    captured = {}

    def _fake_gen(*args, **kwargs):
        captured.update(kwargs)
        return np.ones((2, 3), dtype=float)

    # Patch constructor dependencies and capture forwarded kwargs.
    monkeypatch.setattr(adg_module, "SymmetricInterdictor", _SymStub)
    monkeypatch.setattr(
        AdvDataGenerator, "gen_interdictions", staticmethod(_fake_gen)
    )

    # Construct with n_training_interdictions and inspect forwarded kwargs.
    AdvDataGenerator(
        cfg,
        opt_model_stub,
        budget=2,
        normalization_constant=1.0,
        adverse_problem="SPNI",
        n_training_interdictions=5,
    )

    # Expectation: this key should not be forwarded to gen_interdictions.
    assert "n_training_interdictions" not in captured, \
        "Constructor leaked n_training_interdictions into gen_interdictions."
    pass


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Constructor calls Warning(...) instead of warnings.warn(...), so "
        "no warning is emitted when scenarios exceed available interdictions."
    ),
)
def test_adv_data_generator_regression_init_emits_warning_for_scenario_clamp(
    cfg, opt_model_stub, monkeypatch
):
    """Document missing warning emission when scenario count is clamped."""
    _patch_lightweight_spni_dependencies(monkeypatch)

    # Capture runtime warnings while forcing scenario clamp behavior.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        AdvDataGenerator(
            cfg,
            opt_model_stub,
            budget=2,
            normalization_constant=1.0,
            num_scenarios=150,
            adverse_problem="SPNI",
        )

    # Expectation: at least one warning should be emitted.
    assert len(caught) >= 1, \
        "Expected warning for scenario clamp was not emitted."
    pass


######################
### test generate ###
######################

@pytest.mark.xfail(
    strict=True,
    reason=(
        "BPPO generate(cfg=...) always raises NotImplementedError because "
        "_save_interdictions_to_cache has no BPPO implementation."
    ),
)
def test_adv_data_generator_regression_generate_bppo_with_cfg_avoids_crash():
    """Document that BPPO generate with cfg currently crashes on cache save."""
    generator = object.__new__(AdvDataGenerator)
    generator.adverse_problem = "BPPO"
    generator.num_scenarios = 2
    generator.interdiction_policy = "adversarial"
    generator._cache_options = adg_module.get_cache_replace_options()
    feats = np.array([[1.0]], dtype=float)
    costs = np.array([[2.0, 3.0]], dtype=float)

    # Stub cache load/generation so generate reaches BPPO save path.
    generator._load_interdictions_from_cache = (
        lambda cfg_arg, costs_arg, feats_arg: None
    )
    generator._generate_bppo_interdictions = (
        lambda feats_arg, costs_arg, versatile=False: (
            feats_arg,
            np.array([[[2.0, 3.0], [1.0, 1.5]]], dtype=float),
            np.array([[[0.0, 0.0], [1.0, 1.5]]], dtype=float),
        )
    )

    # Expectation: generate should complete without raising.
    result = generator.generate(feats, costs, cfg=object(), versatile=False)
    assert result[1].shape == (1, 2, 2), \
        "BPPO generation did not return grouped cost scenarios."
    pass
