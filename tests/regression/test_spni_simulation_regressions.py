import numpy as np
import pytest
import torch

import dflintdpy.data.data_gen as data_gen_module
import dflintdpy.scripts.compare as compare_module
from dflintdpy.data.config import HP


pytestmark = [pytest.mark.regression, pytest.mark.pyepo, pytest.mark.torch]


################
### Fixtures ###
################


@pytest.fixture
def syn_cfg() -> HP:
    """Return a compact config for synthetic-data regression tests."""
    hp = HP()
    hp.set("num_features", 4)
    hp.set("num_train_samples", 2)
    hp.set("num_val_samples", 1)
    hp.set("num_test_samples", 1)
    hp.set("deg", 3)
    hp.set("noise_width", 0.25)
    hp.set("random_seed", 17)
    return hp


@pytest.fixture
def compare_cfg() -> HP:
    """Return a compact config for compare-function regression tests."""
    hp = HP()
    hp.set("num_test_samples", 1)
    hp.set("budget", 1)
    hp.set("lsd", 1e-3)
    return hp


############################
### Helper functionality ###
############################


class _OptModelStub:
    """Minimal opt-model stub for synthetic-data regression tests."""

    def __init__(self, num_cost: int = 3):
        self.num_cost = num_cost


class _GraphEvaluatorStub:
    """Callable graph stub that records evaluation requests."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, path, interdictions=None):
        """Record the requested path evaluation."""
        self.calls.append(
            {
                "path": np.array(path, dtype=float),
                "interdictions": None
                if interdictions is None
                else np.array(interdictions, dtype=float),
            }
        )
        return float(np.sum(path))


class _CompareOptModelStub:
    """Minimal shortest-path model stub for compare-function tests."""

    def __init__(self):
        self._graph = _GraphEvaluatorStub()
        self.set_obj_calls: list[np.ndarray] = []

    def setObj(self, cost):
        """Record objective updates."""
        self.set_obj_calls.append(np.array(cost, dtype=float))


class _SolvingCompareOptModelStub(_CompareOptModelStub):
    """Compare-model stub with a deterministic solve sequence."""

    def __init__(self, planned_paths: list[list[float]]):
        super().__init__()
        self._planned_paths = [
            np.array(path, dtype=float) for path in planned_paths
        ]

    def solve(self):
        """Return the next planned path and a dummy objective value."""
        return self._planned_paths.pop(0), 0.0


class _FailedAsymmetricInterdictorStub:
    """Stub asymmetric interdictor that signals an infeasible solve."""

    init_calls: list[dict] = []

    def __init__(
        self,
        graph,
        budget,
        true_costs,
        true_delays,
        est_costs,
        est_delays,
        lsd,
    ):
        type(self).init_calls.append(
            {
                "graph": graph,
                "budget": budget,
                "true_costs": np.array(true_costs, dtype=float),
                "true_delays": np.array(true_delays, dtype=float),
                "est_costs": np.array(est_costs, dtype=float),
                "est_delays": np.array(est_delays, dtype=float),
                "lsd": lsd,
            }
        )

    def solve(self):
        """Return the sentinel failure pair used by the regression case."""
        return None, None


class _TimeoutAsymmetricInterdictorStub:
    """Stub asymmetric interdictor that raises a timeout-style error."""

    def __init__(
        self,
        graph,
        budget,
        true_costs,
        true_delays,
        est_costs,
        est_delays,
        lsd,
    ):
        del graph, budget, true_costs, true_delays, est_costs, est_delays, lsd

    def solve(self):
        """Raise the error shape produced by a solver time limit."""
        raise RuntimeError("SPNI solve did not complete successfully.")


class _SequencedAsymmetricInterdictorStub:
    """Return a planned sequence of asymmetric-solver outcomes."""

    planned_results: list[tuple[np.ndarray | None, float | None]] = []

    def __init__(
        self,
        graph,
        budget,
        true_costs,
        true_delays,
        est_costs,
        est_delays,
        lsd,
    ):
        del graph, budget, true_costs, true_delays, est_costs, est_delays, lsd

    @classmethod
    def reset(
        cls,
        planned_results: list[tuple[np.ndarray | None, float | None]],
    ) -> None:
        """Load the solver outcomes returned by future instances."""
        cls.planned_results = list(planned_results)

    def solve(self):
        """Return the next planned asymmetric-solver result."""
        return type(self).planned_results.pop(0)


class _ConstantPredictor(torch.nn.Module):
    """Predict a fixed cost vector regardless of the input feature."""

    def __init__(self, output: list[float]):
        super().__init__()
        self.output = torch.tensor(output, dtype=torch.float32)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Return the stored prediction vector."""
        del feats
        return self.output.clone()


########################
### test gen_syn_data ###
########################


def test_data_gen_gen_syn_data_uses_explicit_seed_when_opt_model_is_provided(
    syn_cfg,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify that the explicit seed reaches pyepo's data generator."""
    recorded = {}

    def fake_gen_data(
        n_samples,
        num_features,
        grid_size,
        *,
        deg,
        noise_width,
        seed,
    ):
        """Record the synthetic-data parameters and return fixed arrays."""
        recorded.update(
            {
                "n_samples": n_samples,
                "num_features": num_features,
                "grid_size": grid_size,
                "deg": deg,
                "noise_width": noise_width,
                "seed": seed,
            }
        )
        feats = np.zeros((n_samples, num_features), dtype=float)
        costs = np.full((n_samples, grid_size[1]), 2.0, dtype=float)
        return feats, costs

    # Arrange: patch the pyepo generator and provide a stub opt model.
    monkeypatch.setattr(data_gen_module, "genData", fake_gen_data)
    opt_model = _OptModelStub(num_cost=3)

    # Act: call the helper with an explicit seed override.
    features, costs = data_gen_module.gen_syn_data(
        syn_cfg,
        opt_model=opt_model,
        seed=99,
    )

    # Assert: the override should win over cfg.random_seed.
    assert recorded["seed"] == 99, (
        "gen_syn_data did not forward the explicit seed to pyepo."
    )
    assert recorded["grid_size"] == (1, 4), (
        "gen_syn_data did not use the opt-model cost dimension."
    )
    assert features.shape == (4, 4), (
        "gen_syn_data returned features with the wrong shape."
    )
    assert costs.shape == (4, 4), (
        "gen_syn_data returned costs with the wrong shape."
    )
    pass


##################################
### test compare_asym_intd ###
##################################


def test_compare_asym_intd_preserves_alignment_when_solver_times_out(
    compare_cfg,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify that a runtime timeout becomes NaN placeholders."""
    # Arrange: make the asymmetric solver raise the timeout-style error.
    monkeypatch.setattr(
        compare_module,
        "AsymmetricInterdictor",
        _TimeoutAsymmetricInterdictorStub,
    )
    opt_model = _CompareOptModelStub()
    test_data = {
        "feats": np.array([[1.0, 2.0]], dtype=float),
        "costs": np.array([[3.0, 4.0]], dtype=float),
    }
    interdictions = {
        "costs": np.array([[0.5, 0.25]], dtype=float),
    }

    # Act: run the asymmetric comparison on one sample.
    est_objs, true_objs = compare_module.compare_asym_intd(
        compare_cfg,
        opt_model,
        test_data,
        interdictions,
        normalization_constant=1.0,
    )

    # Assert: the failed sample should stay aligned as NaN placeholders.
    assert est_objs.shape == (1,), (
        "compare_asym_intd did not preserve the estimated axis."
    )
    assert true_objs.shape == (1,), (
        "compare_asym_intd did not preserve the true axis."
    )
    assert np.isnan(est_objs[0]), (
        "compare_asym_intd did not replace the estimated value with NaN."
    )
    assert np.isnan(true_objs[0]), (
        "compare_asym_intd did not replace the true value with NaN."
    )
    pass


def test_compare_asym_intd_skips_failed_asymmetric_solve(
    compare_cfg,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify that failed asymmetric solves produce NaN placeholders."""
    # Arrange: replace the solver with one that returns the failure sentinel.
    monkeypatch.setattr(
        compare_module,
        "AsymmetricInterdictor",
        _FailedAsymmetricInterdictorStub,
    )
    opt_model = _CompareOptModelStub()
    test_data = {
        "feats": np.array([[1.0, 2.0]], dtype=float),
        "costs": np.array([[3.0, 4.0]], dtype=float),
    }
    interdictions = {
        "costs": np.array([[0.5, 0.25]], dtype=float),
    }

    # Act: run the asymmetric comparison on one sample.
    est_objs, true_objs = compare_module.compare_asym_intd(
        compare_cfg,
        opt_model,
        test_data,
        interdictions,
        normalization_constant=1.0,
    )

    # Assert: failed solves should preserve shape with NaN placeholders.
    assert est_objs.shape == (1,), (
        "compare_asym_intd did not preserve the estimated-objective axis."
    )
    assert true_objs.shape == (1,), (
        "compare_asym_intd did not preserve the true-objective axis."
    )
    assert np.isnan(est_objs[0]), (
        "compare_asym_intd did not mark the failed estimated sample."
    )
    assert np.isnan(true_objs[0]), (
        "compare_asym_intd did not mark the failed true-objective sample."
    )
    pass


def test_compare_asym_intd_preserves_sample_alignment_after_failed_solve(
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify that failed asymmetric solves preserve sample alignment."""
    # Arrange two samples where only the first asymmetric solve fails.
    cfg = HP(num_test_samples=2, budget=1, lsd=1e-3)
    _SequencedAsymmetricInterdictorStub.reset(
        [
            (None, None),
            (np.array([1.0, 0.0], dtype=float), 0.0),
        ]
    )
    monkeypatch.setattr(
        compare_module,
        "AsymmetricInterdictor",
        _SequencedAsymmetricInterdictorStub,
    )
    opt_model = _SolvingCompareOptModelStub(
        planned_paths=[[1.0, 0.0], [0.0, 1.0]]
    )
    test_data = {
        "feats": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        "costs": np.array([[3.0, 4.0], [5.0, 6.0]], dtype=float),
    }
    interdictions = {
        "costs": np.array([[0.5, 0.25], [0.75, 0.5]], dtype=float),
    }

    # Act by running the asymmetric comparison across both samples.
    est_objs, true_objs = compare_module.compare_asym_intd(
        cfg,
        opt_model,
        test_data,
        interdictions,
        normalization_constant=1.0,
    )

    # Assert failed samples should remain present as NaN placeholders.
    assert est_objs.shape == (2,), (
        "compare_asym_intd did not preserve the estimated-objective axis."
    )
    assert true_objs.shape == (2,), (
        "compare_asym_intd did not preserve the true-objective axis."
    )
    assert np.isnan(est_objs[0]), (
        "compare_asym_intd did not mark the failed estimated sample."
    )
    assert np.isnan(true_objs[0]), (
        "compare_asym_intd did not mark the failed true-objective sample."
    )
    assert est_objs[1] == pytest.approx(1.0), (
        "compare_asym_intd returned the wrong estimated objective."
    )
    assert true_objs[1] == pytest.approx(1.0), (
        "compare_asym_intd returned the wrong true objective."
    )
    pass


#######################################
### test compare_wrong_asym_intd ###
#######################################


@pytest.mark.skip(
    reason="compare_wrong_asym_intd does not handle failed asymmetric solves.",
)
def test_compare_wrong_asym_intd_skips_failed_asymmetric_solve(
    compare_cfg,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify that failed asymmetric solves are skipped cleanly."""
    # Arrange: replace the solver with one that returns the failure sentinel.
    monkeypatch.setattr(
        compare_module,
        "AsymmetricInterdictor",
        _FailedAsymmetricInterdictorStub,
    )
    opt_model = _CompareOptModelStub()
    test_data = {
        "feats": np.array([[1.0, 2.0]], dtype=float),
        "costs": np.array([[3.0, 4.0]], dtype=float),
    }
    interdictions = {
        "costs": np.array([[0.5, 0.25]], dtype=float),
    }
    true_model = _ConstantPredictor([1.0, 1.5])
    false_model = _ConstantPredictor([2.0, 2.5])

    # Act: run the comparison on one sample.
    result = compare_module.compare_wrong_asym_intd(
        compare_cfg,
        opt_model,
        test_data,
        interdictions,
        normalization_constant=1.0,
        true_model=true_model,
        false_model=false_model,
    )

    # Assert: failed solves should be skipped rather than crashing.
    assert result.size == 0, (
        "compare_wrong_asym_intd did not skip the failed solve."
    )
    pass
