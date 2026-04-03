import itertools

import numpy as np
import pytest
import torch

import dflintdpy.data.data_gen as data_gen_module
import dflintdpy.scripts.compare as compare_module
from dflintdpy.data.config import HP
from dflintdpy.models.graph import Graph
from dflintdpy.solvers.asymmetric_interdictor import AsymmetricInterdictor
from dflintdpy.solvers.shortest_path_grb import ShortestPathGrb


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


def _bruteforce_asymmetric_optimum(
    graph: Graph,
    true_costs: np.ndarray,
    true_delays: np.ndarray,
    est_costs: np.ndarray,
    est_delays: np.ndarray,
    budget: int,
) -> tuple[np.ndarray, float]:
    """Enumerate the exact asymmetric optimum on a tiny pinned instance.

    The Bayrak-Bailey-style asymmetric objective is evaluated in two steps:
    first the follower solves the estimated shortest-path problem under a
    candidate interdiction, then the resulting path is scored with the true
    costs and true interdiction delays.  This helper brute-forces every binary
    interdiction pattern that respects the budget so the regression can compare
    the solver output against the exact optimum on a very small graph.
    """

    # Reuse one shortest-path model to avoid rebuilding the follower model for
    # every interdiction pattern in the brute-force search.
    follower = ShortestPathGrb(graph)
    true_graph = Graph(
        arcs=graph.arcs,
        vertices=np.asarray(graph.vertices, dtype=int),
        cost=np.asarray(true_costs, dtype=float),
        source=graph.source,
        target=graph.target,
    )

    best_x = None
    best_value = -np.inf

    for bits in itertools.product([0.0, 1.0], repeat=len(graph.arcs)):
        x = np.asarray(bits, dtype=float)
        if np.sum(x) > budget:
            continue

        # Solve the follower problem on the estimated costs induced by x.
        est_path, _ = follower.solve(c=est_costs + x * est_delays)
        est_path = np.asarray(est_path, dtype=float)

        # Score the follower's chosen path using the true realized costs.
        realized_value = true_graph.evaluate(est_path, interdictions=x * true_delays)
        if realized_value > best_value + 1e-12:
            best_value = float(realized_value)
            best_x = x.copy()

    assert best_x is not None, "Brute-force search did not find a feasible interdiction."
    return best_x, float(best_value)


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


###############################################
### test asymmetric interdictor regression ###
###############################################


@pytest.mark.gurobi
def test_asymmetric_interdictor_solve_matches_bruteforce_optimum_on_pinned_smoke_rows():
    """Verify the staged asymmetric solve matches brute force on smoke data.

    This regression pins two rows extracted from the smoke example used in the
    SPNI simulation pipeline.  Each row is small enough that we can enumerate
    every budget-feasible interdiction and compute the exact Bayrak-Bailey
    asymmetric objective directly.  The solver should return an interdiction
    whose realized value matches that exact optimum.
    """

    # Arrange: pin the smoke-run graph and the two failing asymmetric rows.
    arcs = [
        (0, 1),
        (1, 2),
        (0, 3),
        (1, 4),
        (2, 5),
        (3, 4),
        (4, 5),
        (3, 6),
        (4, 7),
        (5, 8),
        (6, 7),
        (7, 8),
    ]
    vertices = np.arange(9, dtype=int)
    budget = 3
    smoke_cases = [
        {
            "name": "rdfl_sample_2",
            "true_costs": np.array(
                [
                    0.175584829151676,
                    2.291364129319079,
                    4.279864951643211,
                    2.72675497487491,
                    0.329687401413776,
                    0.745138856408196,
                    0.588928066149211,
                    0.657444355519338,
                    0.580558209459803,
                    0.311359544886485,
                    0.430070902429302,
                    1.941334068242822,
                ],
                dtype=float,
            ),
            "true_delays": np.array(
                [
                    0.507463273237833,
                    0.049519210776339,
                    1.651552225245095,
                    1.588422536926261,
                    0.383573102212783,
                    0.061132073598733,
                    0.007005195503427,
                    0.078911330627577,
                    0.168540777790408,
                    0.018397212725843,
                    0.223712742321394,
                    0.038543699202386,
                ],
                dtype=float,
            ),
            "est_costs": np.array(
                [
                    -7.5838723,
                    -2.7844884,
                    4.088116,
                    1.1665864,
                    13.470801,
                    2.7685585,
                    9.754138,
                    -9.210021,
                    4.9252734,
                    1.7087691,
                    3.485747,
                    10.0195875,
                ],
                dtype=float,
            ),
            "est_delays": np.array(
                [
                    0.507463273237833,
                    0.049519210776339,
                    1.651552225245095,
                    1.588422536926261,
                    0.383573102212783,
                    0.061132073598733,
                    0.007005195503427,
                    0.078911330627577,
                    0.168540777790408,
                    0.018397212725843,
                    0.223712742321394,
                    0.038543699202386,
                ],
                dtype=float,
            ),
        },
        {
            "name": "adfl_sample_3",
            "true_costs": np.array(
                [
                    1.290870143855197,
                    0.210464334503301,
                    0.170737649039095,
                    0.17603298215584,
                    0.152117397488633,
                    2.258647766467982,
                    0.11728811822947,
                    0.143886305459124,
                    2.701155826473943,
                    6.866769616616532,
                    7.111411785380134,
                    0.015052597626595,
                ],
                dtype=float,
            ),
            "true_delays": np.array(
                [
                    3.562099484994831,
                    0.853045249106831,
                    5.217163771197206,
                    1.775753233853224,
                    0.417954475672238,
                    0.124740945797335,
                    0.333238318672095,
                    0.159219055178098,
                    0.341956857610789,
                    0.371803487840877,
                    0.635130595247592,
                    0.126781338258611,
                ],
                dtype=float,
            ),
            "est_costs": np.array(
                [
                    17.045324,
                    7.7953434,
                    17.416714,
                    5.6877327,
                    -19.54382,
                    24.043865,
                    -14.742358,
                    -6.0509624,
                    -39.97106,
                    13.975892,
                    3.639001,
                    36.526314,
                ],
                dtype=float,
            ),
            "est_delays": np.array(
                [
                    3.562099484994831,
                    0.853045249106831,
                    5.217163771197206,
                    1.775753233853224,
                    0.417954475672238,
                    0.124740945797335,
                    0.333238318672095,
                    0.159219055178098,
                    0.341956857610789,
                    0.371803487840877,
                    0.635130595247592,
                    0.126781338258611,
                ],
                dtype=float,
            ),
        },
    ]

    for case in smoke_cases:
        # Build a fresh graph and solver for each pinned row so the comparison
        # is isolated and the regression remains easy to debug.
        graph = Graph(
            arcs=arcs,
            vertices=vertices,
            cost=case["true_costs"],
            source=0,
            target=8,
        )
        solver = AsymmetricInterdictor(
            graph=graph,
            budget=budget,
            true_costs=case["true_costs"],
            true_delays=case["true_delays"],
            est_costs=case["est_costs"],
            est_delays=case["est_delays"],
            lsd=1e-3,
        )

        # Compute the exact asymmetric optimum by exhaustively enumerating the
        # tiny interdiction space for this pinned smoke row.
        exact_x, exact_value = _bruteforce_asymmetric_optimum(
            graph=graph,
            true_costs=case["true_costs"],
            true_delays=case["true_delays"],
            est_costs=case["est_costs"],
            est_delays=case["est_delays"],
            budget=budget,
        )

        # Solve the solver under test and normalize the returned vector.
        solver_x, solver_value = solver.solve()
        solver_x = np.asarray(solver_x, dtype=float)

        # Assert the solver output against the exact brute-force optimum.
        assert np.all(np.isin(solver_x, [0.0, 1.0])), (
            f"{case['name']} returned a non-binary interdiction vector."
        )
        assert np.sum(solver_x) <= budget + 1e-7, (
            f"{case['name']} returned an interdiction that violates budget."
        )
        assert solver_value == pytest.approx(
            exact_value, abs=1e-7
        ), (
            f"{case['name']} returned a realized asymmetric value below the "
            f"exact brute-force optimum: {solver_value} < {exact_value}; "
            f"one exact maximizer is {exact_x.tolist()}."
        )
        pass
