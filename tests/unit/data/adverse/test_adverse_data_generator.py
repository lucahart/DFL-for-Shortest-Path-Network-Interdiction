from types import MethodType, SimpleNamespace

import numpy as np
import pytest

import dflintdpy.data.adverse.adverse_data_generator as \
    adverse_data_generator_module
from dflintdpy.data.adverse.adverse_data_generator import AdvDataGenerator
from dflintdpy.data.config import HP
from dflintdpy.utils.read_write import CacheReplaceOptions


################
### Fixtures ###
################

@pytest.fixture
def cfg() -> HP:
    """Return a compact configuration for deterministic generator tests."""
    hp = HP()
    hp.set("num_features", 4)
    hp.set("deg", 3)
    hp.set("noise_width", 0.25)
    hp.set("benders_eps", 1e-4)
    return hp


@pytest.fixture
def opt_model() -> "_OptModelStub":
    """Return a lightweight optimization-model stub for constructor tests."""
    return _OptModelStub()


@pytest.fixture
def feats() -> np.ndarray:
    """Return a small feature matrix shared across generation tests."""
    return np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)


@pytest.fixture
def costs() -> np.ndarray:
    """Return a small cost matrix shared across generation tests."""
    return np.array([[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]], dtype=float)


############################
### Helper functionality ###
############################

class _GraphStub:
    """Minimal graph-like object that can be deep-copied safely."""

    def __init__(self):
        self.label = "graph"


class _OptModelStub:
    """Minimal optimization-model stub used by AdvDataGenerator tests."""

    def __init__(self):
        self._graph = _GraphStub()
        self.num_cost = 3
        self.c = np.array([2.0, 4.0, 6.0], dtype=float)
        self.Sigma = np.eye(3, dtype=float)
        self.gamma = 1.5


class _SetObjRecorder:
    """Record each cost vector pushed into a follower model."""

    def __init__(self):
        self.set_obj_calls: list[np.ndarray] = []

    def setObj(self, cost: np.ndarray) -> None:
        """Record one objective update."""
        self.set_obj_calls.append(np.array(cost, dtype=float))


class _SymmetricInterdictorStub:
    """Stub symmetric interdictor for constructor and SPNI tests."""

    init_calls: list[dict] = []
    planned_solutions: list[np.ndarray] = []

    def __init__(self, graph, **kwargs):
        self.graph = graph
        self.k = kwargs["k"]
        self.max_cnt = kwargs["max_cnt"]
        self.eps = kwargs["eps"]
        self.opt_model = _SetObjRecorder()
        self.benders_calls: list[dict] = []
        type(self).init_calls.append(
            {
                "graph": graph,
                "kwargs": kwargs,
            }
        )

    @classmethod
    def reset(cls, planned_solutions: list[np.ndarray] | None = None) -> None:
        """Clear captured constructor and solve state."""
        cls.init_calls = []
        cls.planned_solutions = list(planned_solutions or [])

    def benders_decomposition(
        self,
        *,
        interdiction_cost: np.ndarray,
        versatile: bool,
    ) -> tuple[np.ndarray, float, dict]:
        """Return the next planned interdiction pattern."""
        self.benders_calls.append(
            {
                "interdiction_cost": np.array(interdiction_cost, dtype=float),
                "versatile": versatile,
            }
        )
        idx = len(self.benders_calls) - 1
        sol = np.array(type(self).planned_solutions[idx], dtype=float)
        return sol, float(sol.sum()), {"scenario": idx}


class _FastSolverStub:
    """Stub fast solver for BPPO constructor and generation tests."""

    init_calls: list[dict] = []
    solve_calls: list[dict] = []
    planned_results: list[np.ndarray] = []

    def __init__(self, cost, Sigma, gamma, budget):
        self.cost = np.array(cost, dtype=float)
        self.Sigma = np.array(Sigma, dtype=float)
        self.gamma = gamma
        self.budget = float(budget)
        type(self).init_calls.append(
            {
                "cost": self.cost.copy(),
                "Sigma": self.Sigma.copy(),
                "gamma": self.gamma,
                "budget": self.budget,
            }
        )

    @classmethod
    def reset(cls, planned_results: list[np.ndarray] | None = None) -> None:
        """Clear captured constructor and solve state."""
        cls.init_calls = []
        cls.solve_calls = []
        cls.planned_results = list(planned_results or [])

    def solve(self, *, n_starts: int, verbose: bool) -> dict:
        """Return the next planned pricing solution."""
        type(self).solve_calls.append(
            {
                "n_starts": n_starts,
                "verbose": verbose,
            }
        )
        idx = len(type(self).solve_calls) - 1
        return {"p_opt": np.array(type(self).planned_results[idx], dtype=float)}


class _ChoiceStub:
    """Deterministic replacement for numpy's random choice method."""

    def __init__(self, outputs: list[np.ndarray]):
        self.outputs = [np.array(output) for output in outputs]
        self.calls: list[dict] = []

    def choice(self, a, size=None, replace=True):
        """Return the next planned choice result."""
        self.calls.append(
            {
                "a": a,
                "size": size,
                "replace": replace,
            }
        )
        return self.outputs.pop(0)


def _build_generator(**overrides) -> AdvDataGenerator:
    """Create a partially initialized generator for method-level tests."""
    generator = object.__new__(AdvDataGenerator)
    generator.opt_model = _OptModelStub()
    generator.num_scenarios = 3
    generator.interdictions = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=float,
    )
    generator._sym_interdictor = SimpleNamespace(
        k=2,
        opt_model=_SetObjRecorder(),
        Sigma=np.eye(3, dtype=float),
        gamma=1.5,
        budget=2.0,
    )
    generator._rng = _ChoiceStub([np.array([0, 1])])
    generator.adverse_problem = "SPNI"
    generator.interdiction_policy = "adversarial"
    generator._cache_options = CacheReplaceOptions()
    generator._base_seed = 0
    for key, value in overrides.items():
        setattr(generator, key, value)
    return generator


def _bind_method(instance, func):
    """Bind a function as an instance method for orchestration tests."""
    return MethodType(func, instance)


#####################
### test __init__ ###
#####################

def test_adv_data_generator_init_rejects_unknown_adverse_problem(
    cfg,
    opt_model,
):
    """Verify that __init__ rejects unsupported adverse-problem values."""
    # Act / Assert: construction should fail for unknown problem types.
    with pytest.raises(
        ValueError,
        match="Unknown adverse problem type: OTHER",
    ):
        AdvDataGenerator(
            cfg,
            opt_model,
            budget=2,
            normalization_constant=2.0,
            adverse_problem="OTHER",
        )
    pass


def test_adv_data_generator_init_rejects_unknown_interdiction_policy(
    cfg,
    opt_model,
):
    """Verify that __init__ rejects unsupported interdiction policies."""
    # Act / Assert: construction should fail for unknown policy types.
    with pytest.raises(
        ValueError,
        match="Unknown interdiction policy: OTHER",
    ):
        AdvDataGenerator(
            cfg,
            opt_model,
            budget=2,
            normalization_constant=2.0,
            interdiction_policy="OTHER",
        )
    pass


def test_adv_data_generator_init_spni_deepcopies_model_and_builds_helpers(
    cfg,
    opt_model,
    monkeypatch,
):
    """Verify that SPNI initialization copies the model and builds helpers."""
    # Arrange: replace heavy collaborators with deterministic stubs.
    _SymmetricInterdictorStub.reset()
    planned_intds = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 1.5]])
    monkeypatch.setattr(
        adverse_data_generator_module.AdvDataGenerator,
        "gen_interdictions",
        staticmethod(lambda *args, **kwargs: planned_intds.copy()),
    )
    monkeypatch.setattr(
        adverse_data_generator_module,
        "SymmetricInterdictor",
        _SymmetricInterdictorStub,
    )

    # Act: construct the generator under SPNI mode.
    generator = AdvDataGenerator(
        cfg,
        opt_model,
        budget=3,
        normalization_constant=2.0,
        num_scenarios=4,
        seed=9,
    )

    # Assert: the optimization model was deep-copied and helper state stored.
    assert generator.opt_model is not opt_model, \
        "SPNI initialization reused the caller optimization model."
    assert generator.opt_model._graph is not opt_model._graph, \
        "SPNI initialization reused the caller graph instance."
    assert generator.num_scenarios == 4, \
        "SPNI initialization changed the requested scenario count."
    assert np.array_equal(generator.interdictions, planned_intds), \
        "SPNI initialization did not store the generated interdictions."
    assert len(_SymmetricInterdictorStub.init_calls) == 1, \
        "SPNI initialization did not build one symmetric interdictor."
    assert _SymmetricInterdictorStub.init_calls[0]["kwargs"]["k"] == 3, \
        "SPNI initialization passed the wrong interdiction budget."
    assert _SymmetricInterdictorStub.init_calls[0]["kwargs"]["eps"] == 1e-4, \
        "SPNI initialization passed the wrong Benders tolerance."
    pass


def test_adv_data_generator_init_spni_uses_defaults_and_caps_scenarios(
    cfg,
    opt_model,
    monkeypatch,
):
    """Verify that SPNI initialization uses default counts and caps scenarios."""
    # Arrange: replace heavy collaborators with lightweight stubs.
    _SymmetricInterdictorStub.reset()
    monkeypatch.setattr(
        adverse_data_generator_module.AdvDataGenerator,
        "gen_interdictions",
        staticmethod(lambda *args, **kwargs: np.ones((100, 3), dtype=float)),
    )
    monkeypatch.setattr(
        adverse_data_generator_module,
        "SymmetricInterdictor",
        _SymmetricInterdictorStub,
    )

    # Act: request more scenarios than the default training pool can support.
    generator = AdvDataGenerator(
        cfg,
        opt_model,
        budget=2,
        normalization_constant=1.0,
        num_scenarios=102,
    )

    # Assert: defaults are retained and scenarios are capped at 101.
    assert generator.n_training_intds == 100, \
        "SPNI initialization did not keep the default training pool size."
    assert generator.num_scenarios == 101, \
        "SPNI initialization did not cap the scenario count correctly."
    pass


##########################################
### test _load_interdictions_from_cache ###
##########################################

def test_adv_data_generator_load_interdictions_from_cache_skips_read_on_replace(
    cfg,
    feats,
    costs,
    monkeypatch,
):
    """Verify that cache loading is skipped when replacement is requested."""
    # Arrange: build a generator configured to replace adversarial cache data.
    generator = _build_generator(
        _cache_options=CacheReplaceOptions(replace_intd_adv=True),
        interdiction_policy="adversarial",
    )
    read_calls: list[tuple] = []

    def _read_cache(*args, **kwargs):
        read_calls.append((args, kwargs))
        return np.zeros((2, 3), dtype=float)

    monkeypatch.setattr(
        adverse_data_generator_module,
        "read_cache",
        _read_cache,
    )

    # Act: attempt to load cached interdictions.
    result = generator._load_interdictions_from_cache(cfg, costs, feats)

    # Assert: the method returns None without touching the cache reader.
    assert result is None, \
        "Cache loading did not short-circuit when replacement was requested."
    assert read_calls == [], \
        "Cache loading still called the cache reader during replacement."
    pass


def test_adv_data_generator_load_interdictions_from_cache_groups_spni_data(
    cfg,
    feats,
    costs,
    monkeypatch,
):
    """Verify that SPNI cache loading reshapes and groups flat interdictions."""
    # Arrange: provide flat cached interdictions for two samples and two
    # interdicted scenarios per sample.
    generator = _build_generator(
        adverse_problem="SPNI",
        num_scenarios=3,
        interdiction_policy="adversarial",
    )
    cached_intd = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 2.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0],
        ],
        dtype=float,
    )
    monkeypatch.setattr(
        adverse_data_generator_module,
        "read_cache",
        lambda *args, **kwargs: cached_intd.copy(),
    )

    # Act: load the cached interdictions.
    _, costs_grouped, interdictions_grouped = (
        generator._load_interdictions_from_cache(cfg, costs, feats)
    )

    # Assert: the original costs stay in scenario zero and the cached
    # interdictions populate later scenarios.
    expected_intd = cached_intd.reshape(2, 2, 3)
    expected_costs = np.zeros((2, 3, 3), dtype=float)
    expected_costs[:, 0, :] = costs
    expected_costs[:, 1, :] = costs + expected_intd[:, 0, :]
    expected_costs[:, 2, :] = costs + expected_intd[:, 1, :]
    expected_grouped = np.zeros_like(expected_costs)
    expected_grouped[:, 1:, :] = expected_intd

    assert np.array_equal(costs_grouped, expected_costs), \
        "SPNI cache loading grouped scenario costs incorrectly."
    assert np.array_equal(interdictions_grouped, expected_grouped), \
        "SPNI cache loading grouped interdictions incorrectly."
    pass


def test_adv_data_generator_load_interdictions_from_cache_returns_none_on_error(
    cfg,
    feats,
    costs,
    monkeypatch,
):
    """Verify that cache loading falls back to generation on reshape errors."""
    # Arrange: return cached data with an incompatible shape.
    generator = _build_generator(
        adverse_problem="SPNI",
        num_scenarios=3,
    )
    monkeypatch.setattr(
        adverse_data_generator_module,
        "read_cache",
        lambda *args, **kwargs: np.ones((3, 3), dtype=float),
    )

    # Act: attempt to load malformed cached interdictions.
    result = generator._load_interdictions_from_cache(cfg, costs, feats)

    # Assert: malformed cached data is treated like a cache miss.
    assert result is None, \
        "Malformed cached interdictions were not treated like a cache miss."
    pass


##########################################
### test _save_interdictions_to_cache ###
##########################################

def test_adv_data_generator_save_interdictions_to_cache_flattens_spni_adv_data(
    cfg,
    monkeypatch,
):
    """Verify that SPNI adversarial cache saves flatten later scenarios."""
    # Arrange: create grouped interdictions with one original scenario.
    generator = _build_generator(
        adverse_problem="SPNI",
        interdiction_policy="adversarial",
        _cache_options=CacheReplaceOptions(
            replace_intd_adv=True,
            archive_replaced=False,
        ),
    )
    grouped = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 2.0, 0.0]],
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 4.0]],
        ],
        dtype=float,
    )
    write_calls: list[dict] = []

    def _write_adv_intd(
        cfg_arg,
        intd_flat,
        *,
        replace,
        archive_replaced,
    ) -> None:
        write_calls.append(
            {
                "cfg": cfg_arg,
                "intd_flat": np.array(intd_flat, dtype=float),
                "replace": replace,
                "archive_replaced": archive_replaced,
            }
        )

    monkeypatch.setattr(
        adverse_data_generator_module,
        "write_adv_intd",
        _write_adv_intd,
    )

    # Act: save the grouped interdictions through the cache helper.
    generator._save_interdictions_to_cache(cfg, grouped)

    # Assert: later scenarios were flattened and sent to the adversarial writer.
    assert len(write_calls) == 1, \
        "SPNI adversarial cache save did not call the writer exactly once."
    assert np.array_equal(
        write_calls[0]["intd_flat"], grouped[:, 1:, :].reshape(-1, 3)
    ), "SPNI adversarial cache save did not flatten later scenarios."
    assert write_calls[0]["replace"] is True, \
        "SPNI adversarial cache save passed the wrong replace flag."
    assert write_calls[0]["archive_replaced"] is False, \
        "SPNI adversarial cache save passed the wrong archive flag."
    pass


def test_adv_data_generator_save_interdictions_to_cache_flattens_spni_rnd_data(
    cfg,
    monkeypatch,
):
    """Verify that SPNI random cache saves flatten later scenarios."""
    # Arrange: create grouped random interdictions with one original scenario.
    generator = _build_generator(
        adverse_problem="SPNI",
        interdiction_policy="random",
        _cache_options=CacheReplaceOptions(
            replace_intd_rnd=True,
            archive_replaced=True,
        ),
    )
    grouped = np.array(
        [
            [[0.0, 0.0, 0.0], [4.0, 0.0, 6.0], [0.0, 2.0, 3.0]],
            [[0.0, 0.0, 0.0], [7.0, 0.0, 0.0], [0.0, 0.0, 5.0]],
        ],
        dtype=float,
    )
    write_calls: list[dict] = []

    def _write_rnd_intd(
        cfg_arg,
        intd_flat,
        *,
        replace,
        archive_replaced,
    ) -> None:
        write_calls.append(
            {
                "cfg": cfg_arg,
                "intd_flat": np.array(intd_flat, dtype=float),
                "replace": replace,
                "archive_replaced": archive_replaced,
            }
        )

    monkeypatch.setattr(
        adverse_data_generator_module,
        "write_rnd_intd",
        _write_rnd_intd,
    )

    # Act: save the grouped interdictions through the random cache helper.
    generator._save_interdictions_to_cache(cfg, grouped)

    # Assert: later scenarios were flattened and sent to the random writer.
    assert len(write_calls) == 1, \
        "SPNI random cache save did not call the writer exactly once."
    assert np.array_equal(
        write_calls[0]["intd_flat"], grouped[:, 1:, :].reshape(-1, 3)
    ), "SPNI random cache save did not flatten later scenarios."
    assert write_calls[0]["replace"] is True, \
        "SPNI random cache save passed the wrong replace flag."
    assert write_calls[0]["archive_replaced"] is True, \
        "SPNI random cache save passed the wrong archive flag."
    pass


##########################################
### test _generate_bppo_interdictions ###
##########################################

def test_adv_data_generator_generate_bppo_interdictions_clips_and_stores_results(
    feats,
    monkeypatch,
):
    """Verify that BPPO generation clips costs and stores solver results."""
    # Arrange: replace the fast solver and progress reporter with stubs.
    _FastSolverStub.reset(
        planned_results=[
            np.array([0.5, 1.0, 0.25], dtype=float),
            np.array([0.1, 0.2, 0.3], dtype=float),
        ]
    )
    progress_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        adverse_data_generator_module,
        "FastBilevelPricingSolver",
        _FastSolverStub,
    )
    monkeypatch.setattr(
        adverse_data_generator_module,
        "print_progress",
        lambda idx, total: progress_calls.append((idx, total)),
    )
    generator = _build_generator(
        adverse_problem="BPPO",
        num_scenarios=2,
        _sym_interdictor=SimpleNamespace(
            Sigma=np.eye(3, dtype=float),
            gamma=1.25,
            budget=2.5,
        ),
    )
    input_costs = np.array([[1.0, -2.0, 4.0], [0.0, 3.0, -1.0]], dtype=float)

    # Act: generate BPPO interdictions for two samples.
    _, costs_grouped, interdictions_grouped = (
        generator._generate_bppo_interdictions(feats, input_costs, versatile=True)
    )

    # Assert: the solver saw clipped costs and the grouped outputs match the
    # solver decisions.
    expected_solver_costs = [
        np.array([1.0, 0.0, 4.0], dtype=float),
        np.array([0.0, 3.0, 0.0], dtype=float),
    ]
    expected_new_costs = np.array(
        [[0.5, -1.0, 3.75], [-0.1, 2.8, -0.3]],
        dtype=float,
    )

    assert np.array_equal(_FastSolverStub.init_calls[0]["cost"], expected_solver_costs[0]), \
        "BPPO generation did not clip negative costs before solving sample 0."
    assert np.array_equal(_FastSolverStub.init_calls[1]["cost"], expected_solver_costs[1]), \
        "BPPO generation did not clip negative costs before solving sample 1."
    assert np.array_equal(costs_grouped[:, 0, :], input_costs), \
        "BPPO generation did not preserve the original scenario costs."
    assert np.allclose(costs_grouped[:, 1, :], expected_new_costs), \
        "BPPO generation stored the interdicted costs incorrectly."
    assert np.allclose(
        interdictions_grouped[:, 1, :],
        np.array(_FastSolverStub.planned_results),
    ), "BPPO generation stored the solver interdictions incorrectly."
    assert progress_calls == [(0, 2), (1, 2)], \
        "BPPO generation did not report progress for every sample."
    pass


##########################################
### test _generate_spni_interdictions ###
##########################################

def test_adv_data_generator_generate_spni_interdictions_uses_benders_results(
    feats,
    costs,
    monkeypatch,
):
    """Verify that adversarial SPNI generation stores Benders solutions."""
    # Arrange: build a deterministic interdictor, RNG, and progress recorder.
    _SymmetricInterdictorStub.reset(
        planned_solutions=[
            np.array([1.0, 0.0, 1.0], dtype=float),
            np.array([0.0, 1.0, 0.0], dtype=float),
        ]
    )
    progress_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        adverse_data_generator_module,
        "print_progress",
        lambda idx, total: progress_calls.append((idx, total)),
    )
    sym_interdictor = _SymmetricInterdictorStub(
        _GraphStub(),
        k=2,
        max_cnt=3,
        eps=1e-4,
    )
    generator = _build_generator(
        num_scenarios=3,
        interdictions=np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=float,
        ),
        _sym_interdictor=sym_interdictor,
        _rng=_ChoiceStub([np.array([1, 0])]),
        interdiction_policy="adversarial",
    )

    # Act: generate SPNI interdictions for one sample.
    _, costs_grouped, interdictions_grouped = (
        generator._generate_spni_interdictions(feats[:1], costs[:1], versatile=True)
    )

    # Assert: Benders was called with the selected interdiction costs and the
    # returned interdictions were stored in grouped form.
    expected_costs = np.array(
        [
            [10.0, 20.0, 30.0],
            [14.0, 20.0, 36.0],
            [10.0, 22.0, 30.0],
        ],
        dtype=float,
    )
    expected_intd = np.array(
        [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 6.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=float,
    )

    assert np.array_equal(
        sym_interdictor.opt_model.set_obj_calls[0], costs[0]
    ), "SPNI generation did not push sample costs into the follower model."
    assert np.array_equal(
        sym_interdictor.benders_calls[0]["interdiction_cost"],
        np.array([4.0, 5.0, 6.0], dtype=float),
    ), "SPNI generation selected the wrong first interdiction vector."
    assert np.array_equal(costs_grouped[0], expected_costs), \
        "SPNI generation stored adversarial scenario costs incorrectly."
    assert np.array_equal(interdictions_grouped[0], expected_intd), \
        "SPNI generation stored adversarial interdictions incorrectly."
    assert progress_calls == [(0, 1)], \
        "SPNI generation did not report progress for the sample."
    pass


def test_adv_data_generator_generate_spni_interdictions_draws_random_patterns(
    feats,
    costs,
    monkeypatch,
):
    """Verify that random SPNI generation applies budget-feasible patterns."""
    # Arrange: provide deterministic choices for scenario selection and edge
    # sampling.
    progress_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        adverse_data_generator_module,
        "print_progress",
        lambda idx, total: progress_calls.append((idx, total)),
    )
    generator = _build_generator(
        num_scenarios=3,
        interdictions=np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=float,
        ),
        _sym_interdictor=SimpleNamespace(k=2),
        _rng=_ChoiceStub(
            [
                np.array([1, 0]),
                np.array([2, 0]),
                np.array([1, 2]),
            ]
        ),
        interdiction_policy="random",
    )

    # Act: generate random SPNI interdictions for one sample.
    _, costs_grouped, interdictions_grouped = (
        generator._generate_spni_interdictions(feats[:1], costs[:1])
    )

    # Assert: each random scenario uses at most the interdiction budget and
    # stores the weighted interdiction vector.
    expected_intd = np.array(
        [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 6.0],
            [0.0, 2.0, 3.0],
        ],
        dtype=float,
    )
    expected_costs = np.array(
        [
            [10.0, 20.0, 30.0],
            [14.0, 20.0, 36.0],
            [10.0, 22.0, 33.0],
        ],
        dtype=float,
    )

    assert np.array_equal(interdictions_grouped[0], expected_intd), \
        "Random SPNI generation stored the wrong weighted interdictions."
    assert np.array_equal(costs_grouped[0], expected_costs), \
        "Random SPNI generation stored the wrong scenario costs."
    assert np.count_nonzero(interdictions_grouped[0, 1]) == 2, \
        "Random SPNI generation exceeded the budget in scenario 1."
    assert np.count_nonzero(interdictions_grouped[0, 2]) == 2, \
        "Random SPNI generation exceeded the budget in scenario 2."
    assert progress_calls == [(0, 1)], \
        "Random SPNI generation did not report progress for the sample."
    pass


######################
### test generate ###
######################

def test_adv_data_generator_generate_returns_cached_result_when_available(
    cfg,
    feats,
    monkeypatch,
):
    """Verify that generate returns cached data without regenerating it."""
    # Arrange: prepare a cached result and fail fast if regeneration is used.
    generator = _build_generator(adverse_problem="SPNI")
    cached_result = (
        feats,
        np.zeros((2, 2, 3), dtype=float),
        np.ones((2, 2, 3), dtype=float),
    )
    monkeypatch.setattr(
        generator,
        "_load_interdictions_from_cache",
        _bind_method(
            generator,
            lambda self, cfg_arg, costs_arg, feats_arg: cached_result,
        ),
    )
    monkeypatch.setattr(
        generator,
        "_generate_spni_interdictions",
        _bind_method(
            generator,
            lambda self, *args, **kwargs: pytest.fail(
                "generate called SPNI generation despite a cache hit."
            ),
        ),
    )

    # Act: ask the generator to produce data with a cache configuration.
    result = generator.generate(feats, np.zeros((2, 3), dtype=float), cfg=cfg)

    # Assert: the cached tuple is returned unchanged.
    assert result[0] is cached_result[0], \
        "generate did not return the cached feature matrix."
    assert result[1] is cached_result[1], \
        "generate did not return the cached grouped costs."
    assert result[2] is cached_result[2], \
        "generate did not return the cached grouped interdictions."
    pass


def test_adv_data_generator_generate_rejects_unknown_problem_type(feats):
    """Verify that generate rejects unsupported adverse-problem values."""
    # Arrange: build a minimal generator with an invalid problem type.
    generator = _build_generator(adverse_problem="OTHER")

    # Act / Assert: dispatch should fail before any generation occurs.
    with pytest.raises(
        ValueError,
        match="Unknown adverse_problem: OTHER",
    ):
        generator.generate(feats, np.zeros((2, 3), dtype=float))
    pass


########################
### Regression tests ###
########################

# Regression-focused coverage lives in
# tests/regression/test_adverse_data_generator_regressions.py.


#########################
### Integration tests ###
#########################

def test_adv_data_generator_generate_orchestrates_spni_generation_and_save(
    cfg,
    feats,
    costs,
    monkeypatch,
):
    """Verify that generate wires cache miss, generation, and save together."""
    # Arrange: replace the cache load, SPNI generation, and cache save steps
    # with deterministic method stubs.
    generator = _build_generator(adverse_problem="SPNI")
    expected_result = (
        feats,
        np.full((2, 2, 3), 9.0, dtype=float),
        np.full((2, 2, 3), 4.0, dtype=float),
    )
    generate_calls: list[dict] = []
    save_calls: list[dict] = []
    monkeypatch.setattr(
        generator,
        "_load_interdictions_from_cache",
        _bind_method(generator, lambda self, cfg_arg, costs_arg, feats_arg: None),
    )

    def _generate(self, feats_arg, costs_arg, versatile=False):
        generate_calls.append(
            {
                "feats": feats_arg.copy(),
                "costs": costs_arg.copy(),
                "versatile": versatile,
            }
        )
        return expected_result

    def _save(self, cfg_arg, grouped_arg):
        save_calls.append(
            {
                "cfg": cfg_arg,
                "grouped": grouped_arg.copy(),
            }
        )

    monkeypatch.setattr(
        generator,
        "_generate_spni_interdictions",
        _bind_method(generator, _generate),
    )
    monkeypatch.setattr(
        generator,
        "_save_interdictions_to_cache",
        _bind_method(generator, _save),
    )

    # Act: generate data on a cache miss.
    result = generator.generate(feats, costs, cfg=cfg, versatile=True)

    # Assert: generation used the SPNI path and saved the resulting
    # interdictions.
    assert len(generate_calls) == 1, \
        "generate did not dispatch to SPNI generation on a cache miss."
    assert generate_calls[0]["versatile"] is True, \
        "generate did not forward the versatile flag to SPNI generation."
    assert len(save_calls) == 1, \
        "generate did not save the newly generated interdictions."
    assert np.array_equal(save_calls[0]["grouped"], expected_result[2]), \
        "generate did not save the grouped interdictions it produced."
    assert result[0] is expected_result[0], \
        "generate did not return the generated feature matrix."
    assert result[1] is expected_result[1], \
        "generate did not return the generated grouped costs."
    assert result[2] is expected_result[2], \
        "generate did not return the generated grouped interdictions."
    pass


##############################
### test gen_interdictions ###
##############################

def test_adv_data_generator_gen_interdictions_normalizes_generated_costs(
    cfg,
    monkeypatch,
):
    """Verify that gen_interdictions normalizes the generated cost matrix."""
    # Arrange: replace pyepo's data generator with a deterministic stub.
    gen_calls: list[dict] = []
    raw_costs = np.array([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]], dtype=float)

    def _gen_data(
        n_interdictions,
        num_features,
        shape,
        *,
        deg,
        noise_width,
        seed,
    ):
        gen_calls.append(
            {
                "n_interdictions": n_interdictions,
                "num_features": num_features,
                "shape": shape,
                "deg": deg,
                "noise_width": noise_width,
                "seed": seed,
            }
        )
        return np.zeros((2, 4), dtype=float), raw_costs.copy()

    monkeypatch.setattr(
        adverse_data_generator_module.pyepo.data.shortestpath,
        "genData",
        _gen_data,
    )

    # Act: generate and normalize interdiction costs.
    intd = AdvDataGenerator.gen_interdictions(
        cfg,
        normalization_constant=2.0,
        num_cost=2,
        gen_intd_seed=17,
        n_interdictions=2,
    )

    # Assert: the raw costs were requested with the right arguments and scaled.
    assert len(gen_calls) == 1, \
        "gen_interdictions did not call the pyepo data generator once."
    assert gen_calls[0]["shape"] == (3, 1), \
        "gen_interdictions requested the wrong shortest-path data shape."
    assert gen_calls[0]["seed"] == 17, \
        "gen_interdictions forwarded the wrong random seed."
    assert np.array_equal(intd, raw_costs / 2.0), \
        "gen_interdictions did not normalize the generated costs."
    pass
