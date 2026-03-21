from copy import deepcopy
import warnings

import numpy as np
import pytest

import dflintdpy.solvers.symmetric_interdictor as symmetric_interdictor_module
from dflintdpy.models.graph import Graph
from dflintdpy.models.grid import Grid
from dflintdpy.solvers.shortest_path_grb import ShortestPathGrb
from dflintdpy.solvers.symmetric_interdictor import SymmetricInterdictor


################
### Fixtures ###
################

@pytest.fixture
def graph() -> Graph:
    arcs = [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (2, 5), (3, 5)]
    vertices = [0, 1, 2, 3, 4, 5]
    cost = np.array([1.0, 4.0, 2.0, 3.0, 1.0, 6.0, 2.0])
    return Graph(arcs, vertices, cost)


@pytest.fixture
def grid() -> Grid:
    m, n = 3, 3
    n_costs = m * (n - 1) + (m - 1) * n
    cost = np.arange(1.0, n_costs + 1.0, 1.0)
    return Grid(m, n, cost)


@pytest.fixture
def interdiction_cost(graph: Graph) -> np.ndarray:
    return np.array([5.0, 1.0, 4.0, 3.0, 2.0, 6.0, 2.0])


@pytest.fixture
def solver(graph: Graph, interdiction_cost: np.ndarray) -> SymmetricInterdictor:
    return SymmetricInterdictor(
        graph,
        k=2,
        interdiction_cost=interdiction_cost,
        max_cnt=5,
        eps=1e-3,
    )


def _assert_flow_balance(graph: Graph, sol: np.ndarray) -> None:
    """Check that a solution satisfies the source-target flow constraints."""
    flow = np.asarray(sol, dtype=float)

    # Assert the returned flow has one nonnegative entry per arc.
    assert flow.shape == (len(graph.arcs),), \
        f"Flow vector length does not match the graph arc count."
    assert np.all(flow >= -1e-7), \
        f"Follower solution uses negative flow on at least one arc."

    # Assert source, sink, and intermediate vertices satisfy conservation.
    for vertex in graph.vertices:
        balance = 0.0
        for idx, (tail, head) in enumerate(graph.arcs):
            if vertex == head:
                balance += flow[idx]
            elif vertex == tail:
                balance -= flow[idx]

        if vertex == graph.source:
            expected = -1.0
        elif vertex == graph.target:
            expected = 1.0
        else:
            expected = 0.0

        assert balance == pytest.approx(expected), \
            f"Flow conservation is violated at this vertex."
        pass


class _FollowerStub:
    """Minimal follower stub for Benders and solve orchestration tests."""

    def __init__(
        self,
        cost: np.ndarray,
        solve_results: list[tuple[np.ndarray, float]],
    ):
        self.cost = np.array(cost, dtype=float)
        self.solve_results = list(solve_results)
        self.solve_calls = 0
        self.set_obj_calls: list[np.ndarray] = []
        self.visualize_calls: list[dict] = []

    def solve(self) -> tuple[np.ndarray, float]:
        """Return the next canned follower solve result."""
        result = self.solve_results[self.solve_calls]
        self.solve_calls += 1
        return result

    def setObj(self, cost: np.ndarray) -> None:
        """Track each follower objective update."""
        new_cost = np.array(cost, dtype=float)
        self.cost = new_cost
        self.set_obj_calls.append(new_cost.copy())

    def visualize(self, **kwargs) -> None:
        """Record visualization requests."""
        self.visualize_calls.append(kwargs)
        pass


#####################
### test __init__ ###
#####################

def test_sym_init_builds_independent_follower_model_from_graph_copy(graph):
    """
    Verify that __init__ wraps the input graph in an independent
    ShortestPathGrb instance.
    """
    # Arrange: build the interdictor around the input graph.
    interdictor = SymmetricInterdictor(graph)

    # Assert: the follower solver is the expected type and owns a copied graph.
    assert isinstance(interdictor.opt_model, ShortestPathGrb), \
        f"Follower model is not a ShortestPathGrb instance."
    assert interdictor.opt_model._graph is not graph, \
        f"Solver reused the caller graph object instead of copying it."
    assert np.array_equal(interdictor.opt_model._graph.cost, graph.cost), \
        f"Copied graph cost vector differs from the input graph."
    assert np.array_equal(
        interdictor.opt_model._graph.vertices, graph.vertices
    ), "Copied graph vertices differ from the input graph."
    assert np.array_equal(interdictor.opt_model._graph.arcs, graph.arcs), \
        f"Copied graph arcs differ from the input graph."

    # Assert: later mutations to the caller's graph do not leak into the
    # solver copy.
    graph.setObj(graph.cost + 10.0)
    assert not np.array_equal(interdictor.opt_model._graph.cost, graph.cost), \
        f"Mutating the caller graph also mutated the solver graph copy."
    pass


def test_sym_init_uses_zero_interdiction_cost_when_none_is_provided(graph):
    """
    Verify that __init__ defaults the interdiction-cost vector to zeros
    with one entry per edge.
    """
    # Arrange: build the interdictor without an explicit interdiction-cost
    # vector.
    interdictor = SymmetricInterdictor(graph)

    # Assert: the stored interdiction costs match the graph edge
    # dimension and start at zero.
    assert interdictor.interdiction_cost.shape == (len(graph.arcs),), \
        f"Default interdiction-cost vector has the wrong length."
    assert np.array_equal(
        interdictor.interdiction_cost, np.zeros(len(graph.arcs))
    ), "Default interdiction costs are not all zero."
    pass


def test_sym_init_stores_budget_hyperparameters_and_custom_interdiction_cost(
    graph, interdiction_cost
):
    """
    Verify that __init__ stores k, max_cnt, eps, and the provided
    interdiction-cost vector unchanged.
    """
    # Arrange: choose non-default settings to confirm they are stored.
    k = 2
    max_cnt = 7
    eps = 0.25

    # Act: construct the interdictor with explicit hyperparameters.
    interdictor = SymmetricInterdictor(
        graph,
        k=k,
        interdiction_cost=interdiction_cost,
        max_cnt=max_cnt,
        eps=eps,
    )

    # Assert: scalar settings and interdiction costs are preserved on the
    # instance.
    assert interdictor.k == k, "Interdiction budget was not stored."
    assert interdictor.max_cnt == max_cnt, \
        f"Benders iteration cap was not stored."
    assert interdictor.eps == eps, "Convergence tolerance was not stored."
    assert np.array_equal(
        interdictor.interdiction_cost, interdiction_cost
    ), "Provided interdiction-cost vector was not preserved."
    pass


def test_sym_init_rejects_interdiction_cost_with_wrong_length(graph):
    """
    Verify that __init__ raises ValueError when the interdiction-cost
    vector length does not match the edge count.
    """
    # Arrange: create an interdiction vector that is one entry too short.
    wrong_size_cost = np.ones(len(graph.arcs) - 1)

    # Act / Assert: construction should fail with the documented
    # validation error.
    with pytest.raises(
        ValueError,
        match="Interdiction cost must match the number of edges in the graph.",
    ):
        SymmetricInterdictor(graph, interdiction_cost=wrong_size_cost)
    pass


def test_sym_init_respects_output_flag_setting(graph):
    """
    Verify that __init__ suppresses Gurobi output by default and leaves
    it enabled when output_flag=True.
    """
    # Arrange: construct one solver with default logging and one with
    # logging enabled.
    silent_interdictor = SymmetricInterdictor(graph)
    verbose_interdictor = SymmetricInterdictor(graph, output_flag=True)

    # Assert: the Gurobi model parameter matches the requested logging behavior.
    assert silent_interdictor._model.Params.OutputFlag == 0, \
        f"Logging is not disabled by default."
    assert verbose_interdictor._model.Params.OutputFlag == 1, \
        f"Logging is not enabled when output_flag=True."
    pass


#########################
### test __deepcopy__ ###
#########################

def test_sym_deepcopy_creates_new_solver_with_matching_configuration(
    graph, interdiction_cost
):
    """
    Verify that __deepcopy__ returns a distinct solver with the same
    budget, tolerance, and interdiction costs.
    """
    # Arrange: build a configured solver that can be copied.
    interdictor = SymmetricInterdictor(
        graph,
        k=2,
        interdiction_cost=interdiction_cost,
        max_cnt=6,
        eps=0.5,
    )

    # Act: deep-copy the solver.
    copied = deepcopy(interdictor)

    # Assert: the copy is distinct but preserves the original configuration.
    assert copied is not interdictor, \
        f"Deepcopy returned the original solver instead of a new instance."
    assert isinstance(copied.opt_model, ShortestPathGrb), \
        f"Copied solver does not keep a ShortestPathGrb follower."
    assert copied.k == interdictor.k, \
        f"Deepcopy changed the interdiction budget."
    assert copied.max_cnt == interdictor.max_cnt, \
        f"Deepcopy changed the Benders iteration limit."
    assert copied.eps == interdictor.eps, \
        f"Deepcopy changed the convergence tolerance."
    assert np.array_equal(
        copied.interdiction_cost, interdictor.interdiction_cost
    ), "Deepcopy changed interdiction costs."
    assert copied.interdiction_cost is not interdictor.interdiction_cost, (
        "Deepcopy shares the interdiction-cost array with the original "
        "solver."
    )
    assert copied.opt_model is not interdictor.opt_model, \
        f"Deepcopy shares the follower solver with the original solver."
    assert copied.opt_model._graph is not interdictor.opt_model._graph, \
        f"Deepcopy shares the follower graph with the original solver."
    assert np.array_equal(copied.opt_model.cost, interdictor.opt_model.cost), \
        f"Deepcopy changed the follower cost vector."
    pass


def test_sym_deepcopy_decouples_mutable_state_from_the_original_solver(
    graph, interdiction_cost
):
    """
    Verify that mutating the copied solver does not change the original
    solver's follower model or interdiction costs.
    """
    # Arrange: deep-copy a configured solver.
    interdictor = SymmetricInterdictor(
        graph,
        k=2,
        interdiction_cost=interdiction_cost,
        max_cnt=6,
        eps=0.5,
    )
    copied = deepcopy(interdictor)

    # Act: mutate the copy's mutable arrays and follower objective.
    copied.interdiction_cost[0] += 99.0
    copied.opt_model.setObj(copied.opt_model.cost + 7.0)

    # Assert: the original solver state stays unchanged.
    assert copied.interdiction_cost[0] != interdictor.interdiction_cost[0], (
        "Changing copied interdiction costs also changed the original "
        "solver."
    )
    assert not np.array_equal(
        copied.opt_model.cost, interdictor.opt_model.cost
    ), (
        "Changing the copied follower objective also changed the "
        "original follower."
    )
    assert np.array_equal(interdictor.interdiction_cost, interdiction_cost), \
        f"Original interdiction costs were modified through the copy."
    assert np.array_equal(interdictor.opt_model.cost, graph.cost), \
        f"Original follower cost vector changed after mutating the copy."
    pass


#####################
### test __call__ ###
#####################

def test_sym_call_delegates_directly_to_solve(solver, monkeypatch):
    """
    Verify that calling the solver instance forwards to solve() and
    returns the same tuple.
    """
    # Arrange: replace solve() with a sentinel return value.
    expected_x = np.array([1, 0, 0, 1, 0, 0, 0])
    expected_y = np.array([1, 0, 1, 1, 0, 0, 1])
    expected_z = 9.0
    calls = []

    def fake_solve():
        calls.append(True)
        return expected_x, expected_y, expected_z

    monkeypatch.setattr(solver, "solve", fake_solve)

    # Act: call the solver instance directly.
    actual_x, actual_y, actual_z = solver()

    # Assert: __call__ delegated once and returned solve()'s tuple unchanged.
    assert calls == [True], "__call__ did not invoke solve() exactly once."
    assert np.array_equal(actual_x, expected_x), \
        f"__call__ changed the interdiction vector returned by solve()."
    assert np.array_equal(actual_y, expected_y), \
        f"__call__ changed the path vector returned by solve()."
    assert actual_z == expected_z, \
        f"__call__ changed the objective returned by solve()."
    pass


##################################
### test solve_maxmin_knapsack ###
##################################

def test_sym_solve_maxmin_knapsack_solves_single_scenario_instance(graph):
    """
    Verify that solve_maxmin_knapsack returns the optimal binary
    decision vector and objective on a simple instance.
    """
    # Arrange: build a one-scenario max-min instance with three
    # profitable items.
    interdictor = SymmetricInterdictor(graph, k=2)
    A = np.array([[1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]])
    b = np.array([2.0])

    # Act: solve the leader problem.
    x_opt, z_opt = interdictor.solve_maxmin_knapsack(A, b)

    # Assert: the returned solution is binary, budget-feasible, and
    # optimal for the scenario.
    assert x_opt.shape == (len(graph.arcs),), \
        f"Leader solution has the wrong dimension."
    assert np.all(np.isin(x_opt, [0, 1])), "Leader solution is not binary."
    assert np.sum(x_opt) == 2, (
        "Leader solution violates the cardinality budget on this "
        "instance."
    )
    assert A[0] @ x_opt + b[0] == pytest.approx(z_opt), \
        f"Reported objective does not match the chosen scenario value."
    assert z_opt == pytest.approx(4.0), \
        f"Simple single-scenario instance was not solved optimally."
    pass


def test_sym_solve_maxmin_knapsack_respects_cardinality_budget(graph):
    """
    Verify that solve_maxmin_knapsack never selects more than k
    interdictions.
    """
    # Arrange: create a scenario with a unique best single item under a
    # budget of one.
    interdictor = SymmetricInterdictor(graph, k=1)
    A = np.array([[3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    b = np.array([0.0])

    # Act: solve the max-min knapsack.
    x_opt, z_opt = interdictor.solve_maxmin_knapsack(A, b)

    # Assert: at most one item is selected, and the best item is chosen.
    assert np.sum(x_opt) <= interdictor.k, \
        f"solve_maxmin_knapsack exceeded the cardinality budget."
    assert x_opt[0] == 1, "Most profitable item was not selected."
    assert np.sum(x_opt) == 1, "One-item budget was not used as expected."
    assert z_opt == pytest.approx(3.0), \
        f"Single-item instance objective is not optimal."
    pass


def test_sym_solve_maxmin_knapsack_optimizes_worst_case_scenarios(
    graph,
):
    """
    Verify that solve_maxmin_knapsack maximizes the minimum scenario
    value when several scenarios are present.
    """
    # Arrange: create two scenarios where either of the first two items
    # yields the same worst-case value.
    interdictor = SymmetricInterdictor(graph, k=1)
    A = np.array([
        [5.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    b = np.array([0.0, 0.0])

    # Act: solve the multi-scenario leader problem.
    x_opt, z_opt = interdictor.solve_maxmin_knapsack(A, b)

    # Assert: the solution maximizes the minimum scenario payoff under
    # the budget.
    scenario_values = A @ x_opt + b
    assert np.sum(x_opt) == 1, \
        f"Multi-scenario solution violates the one-item budget."
    assert x_opt[0] + x_opt[1] == 1, \
        f"Chosen item is not one of the two valid symmetric optima."
    assert np.min(scenario_values) == pytest.approx(z_opt), \
        f"z_opt does not equal the realized worst-case scenario value."
    assert z_opt == pytest.approx(1.0), \
        f"Worst-case optimum for the two-scenario instance is wrong."
    pass


def test_sym_solve_maxmin_knapsack_raises_on_nonoptimal_gurobi_status(
    graph, monkeypatch
):
    """
    Verify that solve_maxmin_knapsack raises RuntimeError when Gurobi
    does not terminate optimally.
    """
    # Arrange: build a fake Gurobi model that reports a non-optimal status.
    interdictor = SymmetricInterdictor(graph)

    class FakeVar:
        def __init__(self):
            self.X = 0.0

        def __rmul__(self, other):
            return 0.0

        def __radd__(self, other):
            return other

        def __add__(self, other):
            return other

        def __le__(self, other):
            return ("<=", other)

    class FakeModel:
        def __init__(self, *args, **kwargs):
            self.Params = type("Params", (), {"OutputFlag": 0})()
            self.Status = symmetric_interdictor_module.GRB.INFEASIBLE

        def addVars(self, n, **kwargs):
            return {j: FakeVar() for j in range(n)}

        def addVar(self, **kwargs):
            return FakeVar()

        def addConstr(self, *args, **kwargs):
            return None

        def setObjective(self, *args, **kwargs):
            return None

        def optimize(self):
            return None

    monkeypatch.setattr(symmetric_interdictor_module.gp, "Model", FakeModel)
    monkeypatch.setattr(
        symmetric_interdictor_module.gp,
        "quicksum",
        lambda terms: sum(terms, 0.0),
    )

    # Act / Assert: the solver surfaces the non-optimal status as a
    # RuntimeError.
    A = np.ones((1, len(graph.arcs)))
    b = np.array([0.0])
    with pytest.raises(RuntimeError, match="Gurobi ended with status"):
        interdictor.solve_maxmin_knapsack(A, b)
    pass


##################################
### test benders_decomposition ###
##################################

def test_sym_benders_decomposition_returns_leader_and_follower_solutions(
    graph, interdiction_cost
):
    """
    Verify that benders_decomposition returns an interdiction vector, a
    shortest-path vector, and the follower objective.
    """
    # Arrange: replace the follower and leader subproblems with
    # deterministic stubs.
    interdictor = SymmetricInterdictor(
        graph,
        interdiction_cost=interdiction_cost,
        max_cnt=2,
        eps=0.0,
    )
    initial_path = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0])
    updated_path = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    chosen_x = np.array([0, 0, 0, 1, 0, 1, 0])
    interdictor.opt_model = _FollowerStub(
        graph.cost,
        [(initial_path, 8.0), (updated_path, 11.0)],
    )

    def fake_knapsack(A, b):
        return chosen_x, 11.0

    interdictor.solve_maxmin_knapsack = fake_knapsack

    # Act: run the Benders loop once.
    actual_x, actual_y, actual_z = interdictor.benders_decomposition(
        interdiction_cost, versatile=False
    )

    # Assert: the method returns the leader decision, final follower
    # path, and final follower objective.
    assert np.array_equal(actual_x, chosen_x), (
        "Benders returned a different interdiction vector than the "
        "leader subproblem chose."
    )
    assert np.array_equal(actual_y, updated_path), (
        "Benders did not return the final follower path from the last "
        "follower solve."
    )
    assert actual_z == pytest.approx(11.0), \
        f"Benders did not return the final follower objective value."
    pass


def test_sym_benders_updates_follower_objective_each_iteration(
    graph,
    interdiction_cost,
):
    """
    Verify that each Benders iteration resolves the follower after
    applying the current interdiction decision.
    """
    # Arrange: configure two Benders iterations with different leader solutions.
    interdictor = SymmetricInterdictor(
        graph,
        interdiction_cost=interdiction_cost,
        max_cnt=3,
        eps=0.5,
    )
    y0 = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0])
    y1 = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    y2 = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    x1 = np.array([1, 0, 0, 0, 0, 0, 0])
    x2 = np.array([0, 1, 0, 0, 0, 0, 0])
    follower = _FollowerStub(graph.cost, [(y0, 5.0), (y1, 3.0), (y2, 4.0)])
    interdictor.opt_model = follower
    knapsack_calls = []

    def fake_knapsack(A, b):
        knapsack_calls.append(
            (np.array(A, dtype=float), np.array(b, dtype=float).reshape(-1))
        )
        if len(knapsack_calls) == 1:
            return x1, 10.0
        return x2, 4.2

    interdictor.solve_maxmin_knapsack = fake_knapsack

    # Act: run the decomposition until the second iteration satisfies eps.
    interdictor.benders_decomposition(interdiction_cost, versatile=False)

    # Assert: each iteration updates the follower objective, then the
    # original cost is restored.
    assert len(knapsack_calls) == 2, (
        "Leader problem was not solved once per Benders iteration in "
        "this scenario."
    )
    assert np.array_equal(
        follower.set_obj_calls[0], graph.cost + interdiction_cost * x1
    ), "First interdiction decision was not applied to the follower objective."
    assert np.array_equal(
        follower.set_obj_calls[1], graph.cost + interdiction_cost * x2
    ), "Second interdiction decision was not applied to the follower objective."
    assert np.array_equal(follower.set_obj_calls[2], graph.cost), (
        "Original follower cost vector was not restored after Benders "
        "completed."
    )
    pass


def test_sym_benders_decomposition_accumulates_scenarios_across_iterations(
    graph, interdiction_cost
):
    """
    Verify that each newly found follower path is appended as an
    additional max-min scenario.
    """
    # Arrange: configure two iterations and capture the leader problem
    # data each time.
    interdictor = SymmetricInterdictor(
        graph,
        interdiction_cost=interdiction_cost,
        max_cnt=3,
        eps=0.5,
    )
    y0 = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0])
    y1 = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    y2 = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    follower = _FollowerStub(graph.cost, [(y0, 5.0), (y1, 3.0), (y2, 4.0)])
    interdictor.opt_model = follower
    calls = []

    def fake_knapsack(A, b):
        calls.append(
            (np.array(A, dtype=float), np.array(b, dtype=float).reshape(-1))
        )
        if len(calls) == 1:
            return np.array([1, 0, 0, 0, 0, 0, 0]), 10.0
        return np.array([0, 1, 0, 0, 0, 0, 0]), 4.2

    interdictor.solve_maxmin_knapsack = fake_knapsack

    # Act: run the decomposition through two iterations.
    interdictor.benders_decomposition(interdiction_cost, versatile=False)

    # Assert: each new follower path becomes a new scenario in A and b.
    assert calls[0][0].shape == (1, len(graph.arcs)), \
        f"First leader solve did not receive exactly one scenario."
    assert np.array_equal(calls[0][0][0], interdiction_cost * y0), (
        "First scenario coefficients were not built from the initial "
        "follower path."
    )
    assert np.array_equal(calls[0][1], np.array([graph.cost @ y0])), \
        f"First scenario constant term is not the base path cost."
    assert calls[1][0].shape == (2, len(graph.arcs)), \
        f"Second leader solve did not receive both accumulated scenarios."
    assert np.array_equal(calls[1][0][0], interdiction_cost * y0), \
        f"Original scenario was not retained in the second leader solve."
    assert np.array_equal(calls[1][0][1], interdiction_cost * y1), \
        f"New follower path was not appended as the second scenario."
    assert np.array_equal(
        calls[1][1], np.array([graph.cost @ y0, graph.cost @ y1])
    ), "Second leader solve used the wrong scenario constants."
    pass


def test_sym_benders_decomposition_stops_when_epsilon_gap_is_met(
    graph, interdiction_cost
):
    """
    Verify that benders_decomposition terminates once z_max - z_min is
    within eps.
    """
    # Arrange: configure a single iteration whose optimality gap is
    # already within eps.
    interdictor = SymmetricInterdictor(
        graph,
        interdiction_cost=interdiction_cost,
        max_cnt=5,
        eps=0.2,
    )
    y0 = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0])
    y1 = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    follower = _FollowerStub(graph.cost, [(y0, 5.0), (y1, 3.0)])
    interdictor.opt_model = follower
    calls = []

    def fake_knapsack(A, b):
        calls.append(True)
        return np.array([0, 0, 0, 1, 0, 1, 0]), 3.1

    interdictor.solve_maxmin_knapsack = fake_knapsack

    # Act: run the decomposition.
    interdictor.benders_decomposition(interdiction_cost, versatile=False)

    # Assert: the loop stops after one leader solve once the gap is
    # small enough.
    assert calls == [True], (
        "Benders did not stop after the first leader solve once the "
        "epsilon gap was satisfied."
    )
    assert follower.solve_calls == 2, (
        "Follower was not solved exactly for the initial and "
        "first-updated objectives."
    )
    pass


def test_sym_benders_decomposition_stops_at_max_iterations_when_gap_remains(
    graph, interdiction_cost
):
    """
    Verify that benders_decomposition respects the max_cnt cap when
    convergence is not reached sooner.
    """
    # Arrange: configure two iterations where the gap never falls below eps.
    interdictor = SymmetricInterdictor(
        graph,
        interdiction_cost=interdiction_cost,
        max_cnt=2,
        eps=0.0,
    )
    y0 = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0])
    y1 = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    y2 = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    follower = _FollowerStub(graph.cost, [(y0, 5.0), (y1, 3.0), (y2, 4.0)])
    interdictor.opt_model = follower
    calls = []

    def fake_knapsack(A, b):
        calls.append(True)
        return np.array([0, 0, 0, 1, 0, 1, 0]), 10.0

    interdictor.solve_maxmin_knapsack = fake_knapsack

    # Act: run the decomposition to the iteration limit.
    actual_x, actual_y, actual_z = interdictor.benders_decomposition(
        interdiction_cost, versatile=False
    )

    # Assert: the loop performs exactly max_cnt leader solves and
    # returns the last follower solution.
    assert len(calls) == interdictor.max_cnt, \
        f"Benders did not honor the max iteration cap."
    assert follower.solve_calls == interdictor.max_cnt + 1, \
        f"Follower solve count does not match the bounded Benders loop."
    assert np.array_equal(actual_y, y2), (
        "Benders did not return the last follower path found before "
        "stopping."
    )
    assert actual_z == pytest.approx(4.0), (
        "Benders did not return the last follower objective found "
        "before stopping."
    )
    assert np.array_equal(actual_x, np.array([0, 0, 0, 1, 0, 1, 0])), (
        "Benders did not return the last leader decision found before "
        "stopping."
    )
    pass


def test_sym_benders_restores_original_follower_costs_after_completion(
    graph,
    interdiction_cost,
):
    """
    Verify that the follower model's original edge costs are restored
    before benders_decomposition returns.
    """
    # Arrange: configure a one-iteration run with a stubbed follower.
    interdictor = SymmetricInterdictor(
        graph,
        interdiction_cost=interdiction_cost,
        max_cnt=2,
        eps=0.0,
    )
    y0 = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0])
    y1 = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    follower = _FollowerStub(graph.cost, [(y0, 5.0), (y1, 11.0)])
    interdictor.opt_model = follower
    interdictor.solve_maxmin_knapsack = lambda A, b: (
        np.array([0, 0, 0, 1, 0, 1, 0]),
        11.0,
    )

    # Act: run the decomposition.
    interdictor.benders_decomposition(interdiction_cost, versatile=False)

    # Assert: the follower objective is restored to the original graph
    # cost before returning.
    assert np.array_equal(follower.cost, graph.cost), \
        f"Follower model kept interdicted costs after Benders returned."
    assert np.array_equal(follower.set_obj_calls[-1], graph.cost), (
        "Final objective reset call did not restore the original graph "
        "cost."
    )
    pass


def test_sym_benders_decomposition_controls_console_output_with_versatile_flag(
    graph, interdiction_cost, capsys
):
    """Verify that progress messages are printed only when versatile=True."""
    # Arrange: build two equivalent stubbed solvers, one quiet and one verbose.
    quiet_interdictor = SymmetricInterdictor(
        graph,
        interdiction_cost=interdiction_cost,
        max_cnt=2,
        eps=0.0,
    )
    quiet_interdictor.opt_model = _FollowerStub(
        graph.cost,
        [
            (np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]), 5.0),
            (np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]), 11.0),
        ],
    )
    quiet_interdictor.solve_maxmin_knapsack = lambda A, b: (
        np.array([0, 0, 0, 1, 0, 1, 0]),
        11.0,
    )

    verbose_interdictor = SymmetricInterdictor(
        graph,
        interdiction_cost=interdiction_cost,
        max_cnt=2,
        eps=0.0,
    )
    verbose_interdictor.opt_model = _FollowerStub(
        graph.cost,
        [
            (np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]), 5.0),
            (np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]), 11.0),
        ],
    )
    verbose_interdictor.solve_maxmin_knapsack = lambda A, b: (
        np.array([0, 0, 0, 1, 0, 1, 0]),
        11.0,
    )
    capsys.readouterr()

    # Act: run one quiet solve and one verbose solve.
    quiet_interdictor.benders_decomposition(interdiction_cost, versatile=False)
    quiet_out = capsys.readouterr().out
    verbose_interdictor.benders_decomposition(interdiction_cost, versatile=True)
    verbose_out = capsys.readouterr().out

    # Assert: only the verbose run emits progress messages.
    assert quiet_out == "", "Quiet Benders run printed progress output."
    assert "Bender's decomposition running" in verbose_out, \
        f"Verbose Benders run omitted the startup progress message."
    assert "Iteration 1" in verbose_out, \
        f"Verbose Benders run omitted per-iteration progress output."
    pass


##################
### test solve ###
##################

def test_sym_solve_delegates_to_benders_with_stored_interdiction_costs(
    solver, monkeypatch
):
    """
    Verify that solve() passes the stored interdiction-cost vector into
    benders_decomposition.
    """
    # Arrange: replace benders_decomposition with a recorder.
    expected = (
        np.array([1, 0, 0, 1, 0, 0, 0]),
        np.array([1, 0, 1, 1, 0, 0, 1]),
        9.0,
    )
    calls = []

    def fake_benders(interdiction_cost, versatile=True):
        calls.append((interdiction_cost.copy(), versatile))
        return expected

    monkeypatch.setattr(solver, "benders_decomposition", fake_benders)

    # Act: solve the interdictor.
    actual = solver.solve(versatile=False)

    # Assert: solve() forwards the stored interdiction-cost vector into Benders.
    assert len(calls) == 1, \
        f"solve() did not call benders_decomposition exactly once."
    assert np.array_equal(calls[0][0], solver.interdiction_cost), (
        "solve() did not pass the stored interdiction-cost vector into "
        "Benders."
    )
    assert calls[0][1] is False, \
        f"solve() did not forward the versatile flag into Benders."
    assert np.array_equal(actual[0], expected[0]), \
        f"solve() changed the interdiction vector returned by Benders."
    assert np.array_equal(actual[1], expected[1]), \
        f"solve() changed the follower path returned by Benders."
    assert actual[2] == expected[2], \
        f"solve() changed the objective returned by Benders."
    pass


def test_sym_solve_returns_benders_solution_tuple(solver, monkeypatch):
    """
    Verify that solve() returns the exact interdiction vector, path
    vector, and objective from benders_decomposition.
    """
    # Arrange: make benders_decomposition return a known tuple.
    expected_x = np.array([0, 0, 0, 1, 0, 1, 0])
    expected_y = np.array([1, 0, 1, 1, 0, 0, 1])
    expected_z = 11.0
    monkeypatch.setattr(
        solver,
        "benders_decomposition",
        lambda interdiction_cost, versatile=True: (
            expected_x,
            expected_y,
            expected_z,
        ),
    )

    # Act: call solve().
    actual_x, actual_y, actual_z = solver.solve()

    # Assert: the result matches the Benders output exactly.
    assert np.array_equal(actual_x, expected_x), (
        "solve() returned a different interdiction vector than Benders "
        "produced."
    )
    assert np.array_equal(actual_y, expected_y), (
        "solve() returned a different path vector than Benders "
        "produced."
    )
    assert actual_z == expected_z, \
        f"solve() returned a different objective than Benders produced."
    pass


def test_sym_solve_visualizes_solution_when_requested(solver, monkeypatch):
    """
    Verify that solve(visualize=True) forwards the path and
    interdiction masks to the follower visualization method.
    """
    # Arrange: stub the Benders result and record visualization calls.
    expected_x = np.array([0, 0, 0, 1, 0, 1, 0])
    expected_y = np.array([1, 0, 1, 1, 0, 0, 1])
    calls = []
    monkeypatch.setattr(
        solver,
        "benders_decomposition",
        lambda interdiction_cost, versatile=True: (
            expected_x,
            expected_y,
            11.0,
        ),
    )
    monkeypatch.setattr(
        solver.opt_model, "visualize", lambda **kwargs: calls.append(kwargs)
    )

    # Act: solve with visualization enabled.
    solver.solve(visualize=True)

    # Assert: solve() forwards the returned leader and follower masks to
    # visualize().
    assert len(calls) == 1, (
        "solve(visualize=True) did not invoke the visualization hook "
        "exactly once."
    )
    assert np.array_equal(calls[0]["colored_edges"], expected_y), (
        "visualize() did not receive the follower path mask as colored "
        "edges."
    )
    assert np.array_equal(calls[0]["dashed_edges"], expected_x), (
        "visualize() did not receive the interdiction mask as dashed "
        "edges."
    )
    pass


def test_sym_solve_forwards_visualization_kwargs(solver, monkeypatch):
    """
    Verify that solve() passes any extra visualization keyword
    arguments through to opt_model.visualize().
    """
    # Arrange: stub the Benders result and record visualization kwargs.
    expected_x = np.array([0, 0, 0, 1, 0, 1, 0])
    expected_y = np.array([1, 0, 1, 1, 0, 0, 1])
    calls = []
    monkeypatch.setattr(
        solver,
        "benders_decomposition",
        lambda interdiction_cost, versatile=True: (
            expected_x,
            expected_y,
            11.0,
        ),
    )
    monkeypatch.setattr(
        solver.opt_model, "visualize", lambda **kwargs: calls.append(kwargs)
    )

    # Act: solve with visualization kwargs.
    solver.solve(visualize=True, versatile=False, node_size=50, title="demo")

    # Assert: extra visualization kwargs are passed through unchanged.
    assert len(calls) == 1, (
        "solve() did not make exactly one visualization call when "
        "visualization was enabled."
    )
    assert calls[0]["node_size"] == 50, \
        f"solve() did not forward the node_size visualization kwarg."
    assert calls[0]["title"] == "demo", \
        f"solve() did not forward the title visualization kwarg."
    assert np.array_equal(calls[0]["colored_edges"], expected_y), \
        f"solve() did not forward the follower mask to visualization."
    assert np.array_equal(calls[0]["dashed_edges"], expected_x), \
        f"solve() did not forward the interdiction mask to visualization."
    pass


########################
### Regression Tests ###
########################

def test_sym_regression_deepcopy_preserves_constructor_contract(
    graph, interdiction_cost
):
    """
    Guard against deepcopy failures caused by passing the wrong object
    type or keyword-only arguments incorrectly.
    """
    # Arrange: build a configured solver that previously triggered a
    # deepcopy TypeError.
    interdictor = SymmetricInterdictor(
        graph,
        k=3,
        interdiction_cost=interdiction_cost,
        max_cnt=4,
        eps=0.1,
    )

    # Act: deep-copy the solver.
    copied = deepcopy(interdictor)

    # Assert: deepcopy succeeds and preserves the constructor-
    # controlled configuration.
    assert isinstance(copied, SymmetricInterdictor), \
        f"Deepcopy no longer returns a SymmetricInterdictor instance."
    assert copied.k == 3, \
        f"Deepcopy no longer preserves the constructor-provided budget."
    assert copied.max_cnt == 4, (
        "Deepcopy no longer preserves the constructor-provided "
        "iteration cap."
    )
    assert copied.eps == pytest.approx(0.1), (
        "Deepcopy no longer preserves the constructor-provided "
        "tolerance."
    )
    assert np.array_equal(copied.interdiction_cost, interdiction_cost), (
        "Deepcopy no longer preserves the constructor-provided "
        "interdiction costs."
    )
    pass


def test_sym_regression_repeated_maxmin_solves_do_not_accumulate_state(
    graph,
):
    """
    Guard against solve_maxmin_knapsack reusing old variables and
    constraints across repeated calls on one solver instance.
    """
    # Arrange: solve the same leader instance twice on one solver.
    interdictor = SymmetricInterdictor(graph, k=2)
    A = np.array([[1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]])
    b = np.array([2.0])

    # Act: solve once, record model size, then solve again and record
    # model size again.
    first_x, first_z = interdictor.solve_maxmin_knapsack(A, b)
    interdictor._model.update()
    first_num_vars = interdictor._model.NumVars
    first_num_constrs = interdictor._model.NumConstrs

    second_x, second_z = interdictor.solve_maxmin_knapsack(A, b)
    interdictor._model.update()
    second_num_vars = interdictor._model.NumVars
    second_num_constrs = interdictor._model.NumConstrs

    # Assert: repeated solves return the same answer without growing the model.
    assert np.array_equal(second_x, first_x), (
        "Repeated max-min solves changed the leader solution on the "
        "same instance."
    )
    assert second_z == pytest.approx(first_z), (
        "Repeated max-min solves changed the objective on the same "
        "instance."
    )
    assert second_num_vars == first_num_vars, (
        "Repeated max-min solves leaked stale variables into the "
        "Gurobi model."
    )
    assert second_num_constrs == first_num_constrs, (
        "Repeated max-min solves leaked stale constraints into the "
        "Gurobi model."
    )
    pass


def test_sym_regression_benders_zero_iteration_has_defined_behavior(
    graph,
    interdiction_cost,
):
    """
    Guard against unbound return values when max_cnt is zero before any
    Benders iteration runs.
    """
    # Arrange: build a solver that is not allowed to run any Benders iteration.
    interdictor = SymmetricInterdictor(
        graph,
        k=2,
        interdiction_cost=interdiction_cost,
        max_cnt=0,
        eps=1e-3,
    )
    expected_y, expected_z = interdictor.opt_model.solve()
    original_cost = interdictor.opt_model.cost.copy()

    # Act: run the decomposition with zero allowed iterations.
    actual_x, actual_y, actual_z = interdictor.benders_decomposition(
        interdiction_cost, versatile=False
    )

    # Assert: the method returns the initial follower solution and a
    # zero interdiction vector.
    assert np.array_equal(actual_x, np.zeros(len(graph.arcs), dtype=int)), (
        "Zero allowed iterations did not return the all-zero "
        "interdiction vector."
    )
    assert np.array_equal(actual_y, expected_y), (
        "Zero allowed iterations did not return the initial follower "
        "path."
    )
    assert actual_z == pytest.approx(expected_z), (
        "Zero allowed iterations did not return the initial follower "
        "objective."
    )
    assert np.array_equal(interdictor.opt_model.cost, original_cost), \
        f"Zero allowed iterations still mutated the follower objective."
    pass


def test_sym_regression_solve_restores_follower_costs_after_visualized_run(
    graph, interdiction_cost, monkeypatch
):
    """
    Guard against solve() leaving the follower objective mutated after a
    full solve-and-visualize cycle.
    """
    # Arrange: build a real solver and suppress plotting side effects.
    interdictor = SymmetricInterdictor(
        graph,
        k=2,
        interdiction_cost=interdiction_cost,
        max_cnt=5,
        eps=1e6,
    )
    original_cost = interdictor.opt_model.cost.copy()
    monkeypatch.setattr(
        interdictor.opt_model, "visualize", lambda **kwargs: None
    )

    # Act: run a full solve with visualization enabled.
    interdictor.solve(visualize=True, versatile=False)

    # Assert: the follower objective is restored after the solve finishes.
    assert np.array_equal(interdictor.opt_model.cost, original_cost), (
        "solve(visualize=True) left the follower cost vector mutated "
        "afterward."
    )
    pass


def test_sym_regression_constructor_copies_interdiction_cost_input(graph):
    """
    Guard against external mutation of the caller-owned
    interdiction-cost vector after construction.
    """
    # Arrange: build the solver from a caller-owned interdiction-cost
    # array.
    interdiction_cost = np.array([5.0, 1.0, 4.0, 3.0, 2.0, 6.0, 2.0])
    interdictor = SymmetricInterdictor(graph, interdiction_cost=interdiction_cost)

    # Act: mutate the caller's original array after construction.
    interdiction_cost[0] += 99.0

    # Assert: the solver should have taken its own copy.
    assert interdictor.interdiction_cost[0] == pytest.approx(5.0), \
        (
        "Mutating the caller-owned interdiction-cost array still "
        "changes the solver state."
    )
    pass


def test_sym_regression_solve_maxmin_knapsack_avoids_numpy_scalar_deprecation(
    graph,
):
    """
    Guard against the 2D offset shape produced by Benders triggering
    NumPy scalar-conversion deprecations.
    """
    # Arrange: build the same offset shape that Benders produces after
    # two scenarios.
    interdictor = SymmetricInterdictor(graph, k=1)
    A = np.array([
        [5.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    b = np.array([[2.0], [3.0]])

    # Act: solve while capturing deprecation warnings.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        interdictor.solve_maxmin_knapsack(A, b)

    # Assert: no NumPy scalar-conversion deprecation should be emitted.
    dep_warnings = [warning for warning in caught if issubclass(
        warning.category, DeprecationWarning
    )]
    assert dep_warnings == [], \
        (
        "solve_maxmin_knapsack still emits a NumPy scalar-conversion "
        "deprecation warning for 2D b inputs."
    )
    pass


def test_sym_regression_leader_model_rebuild_preserves_custom_gurobi_params(
    graph,
):
    """
    Guard against repeated leader solves dropping previously configured
    Gurobi parameters.
    """
    # Arrange: configure a non-default model parameter before solving.
    interdictor = SymmetricInterdictor(graph, k=1)
    interdictor._model.Params.TimeLimit = 12.0
    A = np.array([[3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    b = np.array([0.0])

    # Act: solve the leader problem, which rebuilds the model internally.
    interdictor.solve_maxmin_knapsack(A, b)

    # Assert: custom model parameters should survive the rebuild.
    assert interdictor._model.Params.TimeLimit == pytest.approx(12.0), \
        (
        "Leader-model rebuild dropped the configured Gurobi TimeLimit."
    )
    pass


########################
### Integration Tests ###
########################

def test_sym_integration_graph_solve_returns_budget_feasible_interdiction(
    graph,
    interdiction_cost,
):
    """
    Verify end-to-end that solve() returns a feasible interdiction plan
    and follower path on a small graph instance.
    """
    # Arrange: build a real solver on the sample graph.
    interdictor = SymmetricInterdictor(
        graph,
        k=2,
        interdiction_cost=interdiction_cost,
        max_cnt=5,
        eps=1e6,
    )

    # Act: solve the interdiction problem end to end.
    actual_x, actual_y, actual_z = interdictor.solve(versatile=False)

    # Assert: the leader solution is binary and budget-feasible, and the
    # follower path is a feasible flow.
    assert actual_x.shape == (len(graph.arcs),), \
        f"End-to-end interdiction vector has the wrong dimension."
    assert np.all(np.isin(actual_x, [0, 1])), \
        f"End-to-end interdiction vector is not binary."
    assert np.sum(actual_x) <= interdictor.k, \
        f"End-to-end interdiction vector violates the budget."
    _assert_flow_balance(graph, actual_y)
    assert actual_z >= 0.0, (
        "End-to-end follower objective is negative on this "
        "nonnegative-cost instance."
    )
    pass


def test_sym_integration_end_to_end_solution_matches_follower_evaluation(
    graph, interdiction_cost
):
    """
    Verify end-to-end that the returned follower objective matches
    evaluating the returned path under the interdicted costs.
    """
    # Arrange: solve a real interdiction instance on the sample graph.
    interdictor = SymmetricInterdictor(
        graph,
        k=2,
        interdiction_cost=interdiction_cost,
        max_cnt=5,
        eps=1e6,
    )

    # Act: solve the problem and re-evaluate the returned follower path
    # under the chosen interdictions.
    actual_x, actual_y, actual_z = interdictor.solve(versatile=False)
    reevaluated = interdictor.opt_model.evaluate(
        actual_y, interdiction_cost * actual_x
    )

    # Assert: the reported follower objective matches direct evaluation
    # of the returned solution pair.
    assert actual_z == pytest.approx(reevaluated), (
        "Reported objective does not match direct evaluation of the "
        "returned solution pair."
    )
    pass


def test_sym_integration_end_to_end_solve_on_grid_instance(grid):
    """
    Verify end-to-end that the solver also works with Grid inputs and
    preserves the graph-grid integration path.
    """
    # Arrange: build a real solver on a small grid instance.
    interdiction_cost = np.ones(len(grid.arcs))
    interdictor = SymmetricInterdictor(
        grid,
        k=3,
        interdiction_cost=interdiction_cost,
        max_cnt=5,
        eps=1e6,
    )

    # Act: solve the interdiction problem on the grid.
    actual_x, actual_y, actual_z = interdictor.solve(versatile=False)

    # Assert: the returned leader and follower solutions are
    # dimensionally valid and feasible.
    assert actual_x.shape == (len(grid.arcs),), \
        f"Grid interdiction vector has the wrong dimension."
    assert np.all(np.isin(actual_x, [0, 1])), \
        f"Grid interdiction vector is not binary."
    assert np.sum(actual_x) <= interdictor.k, \
        f"Grid interdiction vector violates the budget."
    _assert_flow_balance(grid, actual_y)
    assert actual_z >= 0.0, (
        "Grid follower objective is negative on this nonnegative-cost "
        "instance."
    )
    pass


def test_sym_integration_visualization_receives_consistent_solution_masks(
    graph, interdiction_cost, monkeypatch
):
    """
    Verify end-to-end that the visualization layer receives masks
    consistent with the final leader and follower solutions.
    """
    # Arrange: build a real solver and record visualization inputs.
    interdictor = SymmetricInterdictor(
        graph,
        k=2,
        interdiction_cost=interdiction_cost,
        max_cnt=5,
        eps=1e6,
    )
    calls = []
    monkeypatch.setattr(
        interdictor.opt_model,
        "visualize",
        lambda **kwargs: calls.append(kwargs),
    )

    # Act: solve end to end with visualization enabled.
    actual_x, actual_y, _ = interdictor.solve(
        visualize=True,
        versatile=False,
        title="graph",
    )

    # Assert: visualize() receives the final leader and follower masks
    # from the solve.
    assert len(calls) == 1, \
        f"Integration solve did not make exactly one visualization call."
    assert np.array_equal(calls[0]["colored_edges"], actual_y), \
        f"Integration visualization received the wrong follower mask."
    assert np.array_equal(calls[0]["dashed_edges"], actual_x), (
        "Integration visualization received the wrong interdiction "
        "mask."
    )
    assert calls[0]["title"] == "graph", (
        "Integration visualization did not receive the forwarded title "
        "kwarg."
    )
    pass
