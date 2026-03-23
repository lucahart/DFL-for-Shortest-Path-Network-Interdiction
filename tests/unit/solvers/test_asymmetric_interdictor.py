import numpy as np
import pytest

import dflintdpy.solvers.asymmetric_interdictor as asymmetric_interdictor_module
from dflintdpy.models.graph import Graph
from dflintdpy.models.grid import Grid
from dflintdpy.solvers.asymmetric_interdictor import AsymmetricInterdictor
from gurobipy import GRB


pytestmark = pytest.mark.gurobi


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
def true_costs() -> np.ndarray:
    return np.array([1.0, 4.0, 2.0, 3.0, 1.0, 6.0, 2.0])


@pytest.fixture
def true_delays() -> np.ndarray:
    return np.array([3.0, 1.0, 4.0, 2.0, 5.0, 2.0, 1.0])


@pytest.fixture
def est_costs() -> np.ndarray:
    return np.array([1.5, 3.5, 2.5, 2.5, 1.0, 5.0, 2.0])


@pytest.fixture
def est_delays() -> np.ndarray:
    return np.array([2.0, 1.5, 3.0, 2.0, 4.0, 2.5, 1.0])


@pytest.fixture
def solver(
    graph: Graph,
    true_costs: np.ndarray,
    true_delays: np.ndarray,
    est_costs: np.ndarray,
    est_delays: np.ndarray,
) -> AsymmetricInterdictor:
    return AsymmetricInterdictor(
        graph=graph,
        budget=2,
        true_costs=true_costs,
        true_delays=true_delays,
        est_costs=est_costs,
        est_delays=est_delays,
        lsd=1e-2,
    )


############################
### Helper functionality ###
############################

def _objective_coefficients(model) -> dict[str, float]:
    """Return the linear objective coefficient on each model variable."""
    model.update()
    return {var.VarName: var.Obj for var in model.getVars()}


def _constraint_coefficients(model, constr) -> dict[str, float]:
    """Return one row of the model as a variable-to-coefficient map."""
    row = model.getRow(constr)
    return {
        row.getVar(i).VarName: row.getCoeff(i)
        for i in range(row.size())
    }


def _assert_binary_and_budget_feasible(
    values: list[float],
    budget: int,
) -> None:
    """Check that a returned interdiction vector is binary and feasible."""
    arr = np.asarray(values, dtype=float)

    assert np.all(np.isin(arr, [0.0, 1.0])), \
        "Interdiction vector contains non-binary entries."
    assert np.sum(arr) <= budget + 1e-7, \
        "Interdiction vector violates the interdiction budget."
    pass


class _StageVar:
    """Minimal variable stub for two-stage orchestration tests."""

    def __init__(self, x_value: float):
        self.X = x_value
        self.Start = None


class _StageModel:
    """Minimal model stub for solve_spnia_LG orchestration tests."""

    def __init__(
        self,
        *,
        status: int,
        obj_val: float,
        label: str,
        call_order: list[str],
    ):
        self.Status = status
        self.ObjVal = obj_val
        self.label = label
        self.call_order = call_order
        self.optimize_calls = 0
        self.param_calls: list[tuple[str, float]] = []
        self.added_constraints: list[tuple[str | None, object]] = []

    def setParam(self, name: str, value: float) -> None:
        """Record each parameter assignment."""
        self.param_calls.append((name, value))

    def optimize(self) -> None:
        """Record solve order without performing optimization."""
        self.optimize_calls += 1
        self.call_order.append(self.label)

    def addConstr(self, expr, name: str | None = None) -> None:
        """Record added constraints for later inspection."""
        self.added_constraints.append((name, expr))
        return None


class _LoadedThenTimedOutModel:
    """Model stub that reports TIME_LIMIT only after optimize()."""

    def __init__(self):
        self.Status = GRB.LOADED
        self.optimize_calls = 0
        self.param_calls: list[tuple[str, float]] = []

    def setParam(self, name: str, value: float) -> None:
        """Record each parameter assignment."""
        self.param_calls.append((name, value))

    def optimize(self) -> None:
        """Simulate a live solve that ends by hitting a time limit."""
        self.optimize_calls += 1
        self.Status = GRB.TIME_LIMIT


def _build_second_stage_data(graph: Graph, x_values: dict) -> tuple:
    """Build simple second-stage return data for solve_spnia_LG stubs."""
    zeros_by_arc = {arc: 0.0 for arc in graph.arcs}
    zeros_by_node = {node: 0.0 for node in graph.vertices}
    return x_values, zeros_by_arc, zeros_by_arc.copy(), zeros_by_node


#####################
### test __init__ ###
#####################

def test_asymmetric_interdictor_init_deepcopies_graph_and_precomputes_adjacency(
    solver,
    graph,
):
    # Check that the solver owns a copied graph instance.
    assert solver.graph is not graph, \
        "Solver reused the caller graph instead of copying it."
    assert solver.graph.arcs == graph.arcs, \
        "Copied graph arcs do not match the input graph."
    assert np.array_equal(solver.graph.vertices, graph.vertices), \
        "Copied graph vertices do not match the input graph."

    # Check that the adjacency maps were precomputed in graph arc order.
    assert solver._out_edges[0] == [(0, 1), (0, 2)], \
        "Outgoing adjacency for node 0 was not precomputed correctly."
    assert solver._out_edges[2] == [(2, 3), (2, 4), (2, 5)], \
        "Outgoing adjacency for node 2 was not precomputed correctly."
    assert solver._in_edges[2] == [(0, 2), (1, 2)], \
        "Incoming adjacency for node 2 was not precomputed correctly."
    assert solver._in_edges[5] == [(2, 5), (3, 5)], \
        "Incoming adjacency for node 5 was not precomputed correctly."
    pass


def test_asymmetric_interdictor_init_stores_arc_data_as_edge_dicts(
    solver,
    graph,
    true_costs,
    true_delays,
    est_costs,
    est_delays,
):
    # Check that every edge is mapped to the correct input values.
    for idx, arc in enumerate(graph.arcs):
        assert solver.true_costs[arc] == pytest.approx(true_costs[idx]), \
            "True cost dictionary is not aligned with the arc ordering."
        assert solver.true_delays[arc] == pytest.approx(true_delays[idx]), \
            "True delay dictionary is not aligned with the arc ordering."
        assert solver.est_costs[arc] == pytest.approx(est_costs[idx]), \
            "Estimated cost dictionary is not aligned with arc ordering."
        assert solver.est_delays[arc] == pytest.approx(est_delays[idx]), \
            "Estimated delay dictionary is not aligned with arc ordering."
    pass


def test_asymmetric_interdictor_init_computes_theta_from_longest_path(
    graph,
    true_costs,
    true_delays,
    est_costs,
    est_delays,
    monkeypatch,
):
    class _ShortestPathStub:
        """Stub that records the cost vector and returns a fixed objective."""

        last_instance = None

        def __init__(self, graph_arg):
            self.graph_arg = graph_arg
            self.set_obj_arg = None
            _ShortestPathStub.last_instance = self

        def setObj(self, cost_arg):
            self.set_obj_arg = np.asarray(cost_arg, dtype=float)

        def solve(self):
            return np.zeros(len(self.graph_arg.arcs)), -12.5

    # Replace the longest-path helper with a deterministic stub.
    monkeypatch.setattr(
        asymmetric_interdictor_module,
        "ShortestPathGrb",
        _ShortestPathStub,
    )

    # Build the solver and inspect the stored theta computation inputs.
    interdictor = AsymmetricInterdictor(
        graph=graph,
        budget=2,
        true_costs=true_costs,
        true_delays=true_delays,
        est_costs=est_costs,
        est_delays=est_delays,
        lsd=0.5,
    )

    expected_cost = -(true_costs + true_delays)
    assert np.array_equal(
        _ShortestPathStub.last_instance.set_obj_arg,
        expected_cost,
    ), "Theta setup did not negate the true cost-plus-delay vector."
    assert interdictor.theta == pytest.approx(25.0), \
        "Theta was not computed as -objective divided by lsd."
    pass


@pytest.mark.regression
def test_asymmetric_interdictor_init_accepts_list_inputs_for_theta_path(
    graph,
    monkeypatch,
):
    class _ShortestPathStub:
        """Stub that should be usable for both lists and numpy arrays."""

        last_instance = None

        def __init__(self, graph_arg):
            self.graph_arg = graph_arg
            self.set_obj_arg = None
            _ShortestPathStub.last_instance = self

        def setObj(self, cost_arg):
            self.set_obj_arg = np.asarray(cost_arg, dtype=float)

        def solve(self):
            return np.zeros(len(self.graph_arg.arcs)), -12.5

    # Replace the longest-path helper with a deterministic stub.
    monkeypatch.setattr(
        asymmetric_interdictor_module,
        "ShortestPathGrb",
        _ShortestPathStub,
    )

    # Build the solver using list inputs, which the public type hints allow.
    true_costs = [1.0, 4.0, 2.0, 3.0, 1.0, 6.0, 2.0]
    true_delays = [3.0, 1.0, 4.0, 2.0, 5.0, 2.0, 1.0]
    est_costs = [1.5, 3.5, 2.5, 2.5, 1.0, 5.0, 2.0]
    est_delays = [2.0, 1.5, 3.0, 2.0, 4.0, 2.5, 1.0]
    interdictor = AsymmetricInterdictor(
        graph=graph,
        budget=2,
        true_costs=true_costs,
        true_delays=true_delays,
        est_costs=est_costs,
        est_delays=est_delays,
        lsd=0.5,
    )

    # Check that list inputs use the same longest-path theta computation.
    assert np.array_equal(
        _ShortestPathStub.last_instance.set_obj_arg,
        -(np.asarray(true_costs) + np.asarray(true_delays)),
    ), "List inputs did not reach the documented longest-path theta branch."
    assert interdictor.theta == pytest.approx(25.0), \
        "List inputs did not produce the expected theta value."
    pass


def test_asymmetric_interdictor_init_uses_sum_fallback_when_path_setup_fails(
    graph,
    true_costs,
    true_delays,
    est_costs,
    est_delays,
    monkeypatch,
):
    class _FailingShortestPath:
        """Stub that always fails during longest-path setup."""

        def __init__(self, graph_arg):
            raise RuntimeError("boom")

    # Force the fallback branch for theta.
    monkeypatch.setattr(
        asymmetric_interdictor_module,
        "ShortestPathGrb",
        _FailingShortestPath,
    )

    # Build the solver and check the fallback scaling.
    interdictor = AsymmetricInterdictor(
        graph=graph,
        budget=2,
        true_costs=true_costs,
        true_delays=true_delays,
        est_costs=est_costs,
        est_delays=est_delays,
        lsd=0.5,
    )

    expected_theta = np.sum(true_costs + true_delays) / 0.5
    assert interdictor.theta == pytest.approx(expected_theta), \
        "Theta fallback did not use the summed true cost-plus-delay value."
    pass


######################
### test out_edges ###
######################

def test_asymmetric_interdictor_out_edges_returns_precomputed_arc_list(solver):
    # Query one node with multiple outgoing arcs.
    outgoing = solver.out_edges(2)

    # Check both the contents and that the cached list is returned directly.
    assert outgoing == [(2, 3), (2, 4), (2, 5)], \
        "out_edges returned the wrong outgoing arcs for node 2."
    assert outgoing is solver._out_edges[2], \
        "out_edges did not return the cached outgoing adjacency list."
    pass


#####################
### test in_edges ###
#####################

def test_asymmetric_interdictor_in_edges_returns_precomputed_arc_list(solver):
    # Query one node with multiple incoming arcs.
    incoming = solver.in_edges(5)

    # Check both the contents and that the cached list is returned directly.
    assert incoming == [(2, 5), (3, 5)], \
        "in_edges returned the wrong incoming arcs for node 5."
    assert incoming is solver._in_edges[5], \
        "in_edges did not return the cached incoming adjacency list."
    pass


##########################
### test build_spnia_L ###
##########################

def test_asymmetric_interdictor_build_spnia_l_creates_expected_model_objects(
    solver,
):
    # Build the optimistic model and synchronize the live Gurobi state.
    model, x = solver.build_spnia_L()
    model.update()

    # Check the model name, variable families, and output flag.
    assert model.ModelName == "SPNIA_L", \
        "Optimistic model has the wrong Gurobi model name."
    assert len(x) == len(solver.graph.arcs), \
        "Optimistic model did not create one binary x variable per arc."
    assert all(x[arc].VType == GRB.BINARY for arc in solver.graph.arcs), \
        "Optimistic model x variables are not binary."
    assert len(model.getVars()) == 3 * len(solver.graph.arcs) + \
        len(solver.graph.vertices), \
        "Optimistic model created the wrong total number of variables."
    assert model.Params.OutputFlag == 0, \
        "Optimistic model did not suppress Gurobi output."
    pass


def test_asymmetric_interdictor_build_spnia_l_uses_true_objective_data(solver):
    # Build the optimistic model and inspect its linear objective.
    model, _ = solver.build_spnia_L()
    coeffs = _objective_coefficients(model)

    # Check that v and w variables use the documented true coefficients.
    assert coeffs["v[0,1]"] == pytest.approx(solver.true_costs[(0, 1)]), \
        "v objective coefficient did not use the true cost."
    assert coeffs["v[2,5]"] == pytest.approx(solver.true_costs[(2, 5)]), \
        "v objective coefficient did not use the true cost."
    assert coeffs["w[0,1]"] == pytest.approx(
        solver.true_costs[(0, 1)] + solver.true_delays[(0, 1)]
    ), "w objective coefficient did not use true cost plus true delay."
    assert coeffs["w[2,4]"] == pytest.approx(
        solver.true_costs[(2, 4)] + solver.true_delays[(2, 4)]
    ), "w objective coefficient did not use true cost plus true delay."
    pass


def test_asymmetric_interdictor_build_spnia_l_adds_expected_constraints(solver):
    # Build the optimistic model and collect the named constraints.
    model, _ = solver.build_spnia_L()
    model.update()
    names = {constr.ConstrName for constr in model.getConstrs()}

    # Check that each major constraint family was created in full.
    expected_total = (
        len(solver.graph.vertices)
        + len(solver.graph.arcs)
        + 1
        + 2 * len(solver.graph.arcs)
        + 1
    )
    assert len(model.getConstrs()) == expected_total, \
        "Optimistic model created the wrong number of constraints."
    assert "flow_0" in names and "flow_5" in names, \
        "Optimistic model is missing source or sink flow constraints."
    assert "dual_0_1" in names and "dual_3_5" in names, \
        "Optimistic model is missing per-arc dual constraints."
    assert "dual_link" in names, \
        "Optimistic model is missing the dual-link equality."
    assert "link1_(2, 5)" in names and "link2_(2, 5)" in names, \
        "Optimistic model is missing arc-linking constraints."
    assert "budget" in names, \
        "Optimistic model is missing the interdiction budget constraint."
    pass


def test_asymmetric_interdictor_build_spnia_l_uses_estimated_dual_data(solver):
    # Build the optimistic model and inspect one dual row and the budget row.
    model, _ = solver.build_spnia_L()
    model.update()
    dual = model.getConstrByName("dual_0_1")
    dual_link = model.getConstrByName("dual_link")
    budget = model.getConstrByName("budget")

    dual_coeffs = _constraint_coefficients(model, dual)
    dual_link_coeffs = _constraint_coefficients(model, dual_link)
    budget_coeffs = _constraint_coefficients(model, budget)

    # Check that the dual constraints use estimated costs and delays.
    assert dual.RHS == pytest.approx(solver.est_costs[(0, 1)]), \
        "Optimistic dual row RHS did not use the estimated cost."
    assert dual_coeffs["x[0,1]"] == pytest.approx(
        -solver.est_delays[(0, 1)]
    ), "Optimistic dual row x coefficient did not use estimated delay."
    assert dual_coeffs["u[0]"] == pytest.approx(1.0), \
        "Optimistic dual row is missing the +u[i] coefficient."
    assert dual_coeffs["u[1]"] == pytest.approx(-1.0), \
        "Optimistic dual row is missing the -u[j] coefficient."

    # Check that the dual-link row uses the estimated data on v and w.
    assert dual_link_coeffs["v[0,1]"] == pytest.approx(
        solver.est_costs[(0, 1)]
    ), "Optimistic dual-link row did not use estimated v coefficients."
    assert dual_link_coeffs["w[1,2]"] == pytest.approx(
        solver.est_costs[(1, 2)] + solver.est_delays[(1, 2)]
    ), "Optimistic dual-link row did not use estimated w coefficients."
    assert dual_link_coeffs["u[0]"] == pytest.approx(-1.0), \
        "Optimistic dual-link row is missing the -u[source] coefficient."
    assert dual_link_coeffs["u[5]"] == pytest.approx(1.0), \
        "Optimistic dual-link row is missing the +u[target] coefficient."

    # Check that the budget row charges every x variable once.
    assert budget.RHS == pytest.approx(solver.budget), \
        "Optimistic budget row RHS does not equal the budget."
    assert all(value == pytest.approx(1.0) for value in budget_coeffs.values()), \
        "Optimistic budget row does not sum the x variables correctly."
    pass


###########################
### test build_spnia_LG ###
###########################

def test_asymmetric_interdictor_build_spnia_lg_creates_expected_model_objects(
    solver,
):
    # Build the pessimistic model and synchronize the live Gurobi state.
    model, x, _, _, _ = solver.build_spnia_LG()
    model.update()

    # Check the model name, variable families, and output flag.
    assert model.ModelName == "SPNIA_LG", \
        "Pessimistic model has the wrong Gurobi model name."
    assert len(x) == len(solver.graph.arcs), \
        "Pessimistic model did not create one binary x variable per arc."
    assert all(x[arc].VType == GRB.BINARY for arc in solver.graph.arcs), \
        "Pessimistic model x variables are not binary."
    assert len(model.getVars()) == 3 * len(solver.graph.arcs) + \
        len(solver.graph.vertices), \
        "Pessimistic model created the wrong total number of variables."
    assert model.Params.OutputFlag == 0, \
        "Pessimistic model did not suppress Gurobi output."
    pass


def test_asymmetric_interdictor_build_spnia_lg_uses_theta_weighted_objective(
    solver,
):
    # Build the pessimistic model and inspect its linear objective.
    model, _, _, _, _ = solver.build_spnia_LG()
    coeffs = _objective_coefficients(model)

    # Check the theta-scaled v and w coefficients and source-target terms.
    assert coeffs["v[0,1]"] == pytest.approx(
        -solver.theta * solver.est_costs[(0, 1)]
    ), "Pessimistic v coefficient did not use theta-scaled estimate."
    assert coeffs["w[0,1]"] == pytest.approx(
        -solver.theta * (
            solver.est_costs[(0, 1)] + solver.est_delays[(0, 1)]
        )
    ), "Pessimistic w coefficient did not use theta-scaled estimate."
    assert coeffs["u[0]"] == pytest.approx(1.0), \
        "Pessimistic objective is missing the +u[source] term."
    assert coeffs["u[5]"] == pytest.approx(-1.0), \
        "Pessimistic objective is missing the -u[target] term."
    pass


def test_asymmetric_interdictor_build_spnia_lg_uses_true_and_estimated_arc_data(
    solver,
):
    # Build the pessimistic model and inspect the first arc row.
    model, _, _, _, _ = solver.build_spnia_LG()
    model.update()
    first_dual = model.getConstrs()[len(solver.graph.vertices)]
    coeffs = _constraint_coefficients(model, first_dual)

    expected_x = -(
        solver.theta * solver.est_delays[(0, 1)]
        + solver.true_delays[(0, 1)]
    )
    expected_rhs = (
        solver.theta * solver.est_costs[(0, 1)]
        + solver.true_costs[(0, 1)]
    )

    # Check that the arc row mixes estimated and true data correctly.
    assert first_dual.RHS == pytest.approx(expected_rhs), \
        "Pessimistic arc row RHS did not combine theta and true costs."
    assert coeffs["x[0,1]"] == pytest.approx(expected_x), \
        "Pessimistic arc row x coefficient did not combine both delays."
    assert coeffs["u[0]"] == pytest.approx(1.0), \
        "Pessimistic arc row is missing the +u[i] coefficient."
    assert coeffs["u[1]"] == pytest.approx(-1.0), \
        "Pessimistic arc row is missing the -u[j] coefficient."
    pass


def test_asymmetric_interdictor_build_spnia_lg_adds_expected_constraints(
    solver,
):
    # Build the pessimistic model and count its constraint families.
    model, _, _, _, _ = solver.build_spnia_LG()
    model.update()

    # Check the total number of constraints and the final budget row.
    expected_total = (
        len(solver.graph.vertices)
        + len(solver.graph.arcs)
        + 2 * len(solver.graph.arcs)
        + 1
    )
    budget = model.getConstrs()[-1]
    budget_coeffs = _constraint_coefficients(model, budget)

    assert len(model.getConstrs()) == expected_total, \
        "Pessimistic model created the wrong number of constraints."
    assert budget.RHS == pytest.approx(solver.budget), \
        "Pessimistic budget row RHS does not equal the budget."
    assert all(value == pytest.approx(1.0) for value in budget_coeffs.values()), \
        "Pessimistic budget row does not sum the x variables correctly."
    pass


###########################
### test solve_spnia_LG ###
###########################

def test_asymmetric_interdictor_solve_spnia_lg_solves_stage_models_in_order(
    solver,
    monkeypatch,
):
    # Build deterministic first-stage and second-stage stubs.
    call_order: list[str] = []
    x_l = {arc: _StageVar(float(idx % 2)) for idx, arc in enumerate(solver.graph.arcs)}
    x_lg = {
        arc: _StageVar(float((idx + 1) % 2))
        for idx, arc in enumerate(solver.graph.arcs)
    }
    model_l = _StageModel(
        status=GRB.OPTIMAL,
        obj_val=7.5,
        label="L",
        call_order=call_order,
    )
    model_lg = _StageModel(
        status=GRB.OPTIMAL,
        obj_val=11.0,
        label="LG",
        call_order=call_order,
    )
    second_stage = _build_second_stage_data(solver.graph, x_lg)

    # Replace both builders with the deterministic stage stubs.
    monkeypatch.setattr(solver, "build_spnia_L", lambda: (model_l, x_l))
    monkeypatch.setattr(
        solver,
        "build_spnia_LG",
        lambda: (model_lg, *second_stage),
    )

    # Solve the two-stage procedure.
    x_star, z_star = solver.solve_spnia_LG()

    # Check solve order, return values, and time-limit parameter setup.
    assert call_order == ["L", "LG"], \
        "Two-stage solve did not optimize the stages in the expected order."
    assert x_star == {arc: x_lg[arc].X for arc in solver.graph.arcs}, \
        "Two-stage solve did not return the second-stage interdictions."
    assert z_star == pytest.approx(model_lg.ObjVal), \
        "Two-stage solve did not return the second-stage objective."
    assert model_l.param_calls == [("TimeLimit", 120.0)], \
        "First-stage solve did not set the documented time limit."
    assert model_lg.param_calls == [("TimeLimit", 120.0)], \
        "Second-stage solve did not set the documented time limit."
    pass


def test_asymmetric_interdictor_solve_spnia_lg_warm_starts_second_stage(
    solver,
    monkeypatch,
):
    # Build deterministic stage stubs with known first-stage x values.
    call_order: list[str] = []
    x_l = {arc: _StageVar(float(idx % 2)) for idx, arc in enumerate(solver.graph.arcs)}
    x_lg = {arc: _StageVar(0.0) for arc in solver.graph.arcs}
    model_l = _StageModel(
        status=GRB.OPTIMAL,
        obj_val=7.5,
        label="L",
        call_order=call_order,
    )
    model_lg = _StageModel(
        status=GRB.OPTIMAL,
        obj_val=11.0,
        label="LG",
        call_order=call_order,
    )
    second_stage = _build_second_stage_data(solver.graph, x_lg)

    # Replace both builders so the warm-start values can be inspected.
    monkeypatch.setattr(solver, "build_spnia_L", lambda: (model_l, x_l))
    monkeypatch.setattr(
        solver,
        "build_spnia_LG",
        lambda: (model_lg, *second_stage),
    )

    # Solve the two-stage procedure.
    solver.solve_spnia_LG()

    # Check that every second-stage Start value matches the first stage.
    for arc in solver.graph.arcs:
        assert x_lg[arc].Start == pytest.approx(x_l[arc].X), \
            "Second-stage warm start did not copy the first-stage solution."
    pass


def test_asymmetric_interdictor_solve_spnia_lg_adds_warm_cut_to_second_stage(
    solver,
    monkeypatch,
):
    # Build deterministic stage stubs that allow constraint inspection.
    call_order: list[str] = []
    x_l = {arc: _StageVar(0.0) for arc in solver.graph.arcs}
    x_lg = {arc: _StageVar(0.0) for arc in solver.graph.arcs}
    model_l = _StageModel(
        status=GRB.OPTIMAL,
        obj_val=7.5,
        label="L",
        call_order=call_order,
    )
    model_lg = _StageModel(
        status=GRB.OPTIMAL,
        obj_val=11.0,
        label="LG",
        call_order=call_order,
    )
    second_stage = _build_second_stage_data(solver.graph, x_lg)

    # Replace both builders so the second-stage added constraints are visible.
    monkeypatch.setattr(solver, "build_spnia_L", lambda: (model_l, x_l))
    monkeypatch.setattr(
        solver,
        "build_spnia_LG",
        lambda: (model_lg, *second_stage),
    )

    # Solve the two-stage procedure.
    solver.solve_spnia_LG()

    # Check that the named warm-cut row was added before the second solve.
    added_names = [name for name, _ in model_lg.added_constraints]
    assert "warm_cut" in added_names, \
        "Second-stage model did not receive the warm-cut constraint."
    pass


def test_asymmetric_interdictor_solve_spnia_lg_returns_none_on_first_stage_timeout(
    solver,
    monkeypatch,
):
    # Build a first-stage stub that times out only after optimize().
    model_l = _LoadedThenTimedOutModel()
    x_l = {arc: _StageVar(0.0) for arc in solver.graph.arcs}
    second_stage_called = {"value": False}

    def _unexpected_second_stage():
        second_stage_called["value"] = True
        raise AssertionError("Second stage should not be built on timeout.")

    # Replace the builders and exercise a realistic timeout branch.
    monkeypatch.setattr(solver, "build_spnia_L", lambda: (model_l, x_l))
    monkeypatch.setattr(solver, "build_spnia_LG", _unexpected_second_stage)
    result = solver.solve_spnia_LG()

    # Check that the timeout was detected after optimize() and stopped stage 2.
    assert result == (None, None), \
        "First-stage timeout did not return the documented None pair."
    assert model_l.optimize_calls == 1, \
        "First-stage timeout was not detected after optimize()."
    assert model_l.param_calls == [("TimeLimit", 120.0)], \
        "First-stage timeout branch did not set the documented time limit."
    assert not second_stage_called["value"], \
        "First-stage timeout still attempted to build the second stage."
    pass


def test_asymmetric_interdictor_solve_spnia_lg_returns_none_on_second_stage_timeout(
    solver,
    monkeypatch,
):
    # Build deterministic stubs where the second stage times out.
    call_order: list[str] = []
    x_l = {arc: _StageVar(0.0) for arc in solver.graph.arcs}
    x_lg = {arc: _StageVar(0.0) for arc in solver.graph.arcs}
    model_l = _StageModel(
        status=GRB.OPTIMAL,
        obj_val=7.5,
        label="L",
        call_order=call_order,
    )
    model_lg = _StageModel(
        status=GRB.TIME_LIMIT,
        obj_val=11.0,
        label="LG",
        call_order=call_order,
    )
    second_stage = _build_second_stage_data(solver.graph, x_lg)

    # Replace both builders and exercise the second-stage timeout branch.
    monkeypatch.setattr(solver, "build_spnia_L", lambda: (model_l, x_l))
    monkeypatch.setattr(
        solver,
        "build_spnia_LG",
        lambda: (model_lg, *second_stage),
    )
    result = solver.solve_spnia_LG()

    # Check that both stages ran and the timeout produced the None pair.
    assert call_order == ["L", "LG"], \
        "Second-stage timeout branch did not attempt both stage solves."
    assert result == (None, None), \
        "Second-stage timeout did not return the documented None pair."
    pass


#################
### test solve ###
#################

def test_asymmetric_interdictor_solve_returns_arc_ordered_list(
    solver,
    monkeypatch,
):
    # Stub the two-stage solve with a fixed arc-keyed solution dictionary.
    x_star = {
        arc: float(idx % 2)
        for idx, arc in enumerate(solver.graph.arcs)
    }
    monkeypatch.setattr(solver, "solve_spnia_LG", lambda: (x_star, 13.0))

    # Solve through the public method.
    x_list, z_star = solver.solve()

    # Check that the dictionary was converted into graph arc order.
    assert x_list == [x_star[arc] for arc in solver.graph.arcs], \
        "solve did not convert the arc dictionary into arc-order list form."
    assert z_star == pytest.approx(13.0), \
        "solve changed the objective returned by solve_spnia_LG."
    pass


def test_asymmetric_interdictor_solve_preserves_objective_from_two_stage_solver(
    solver,
    monkeypatch,
):
    # Stub the two-stage solve with a known objective value.
    x_star = {arc: 0.0 for arc in solver.graph.arcs}
    monkeypatch.setattr(solver, "solve_spnia_LG", lambda: (x_star, 9.25))

    # Solve through the public method.
    _, z_star = solver.solve()

    # Check that solve only changes the x representation.
    assert z_star == pytest.approx(9.25), \
        "solve did not preserve the objective from the two-stage solver."
    pass


@pytest.mark.regression
def test_asymmetric_interdictor_solve_propagates_none_pair_gracefully(
    solver,
    monkeypatch,
):
    # Force the public solve() method to receive the solver's None contract.
    monkeypatch.setattr(solver, "solve_spnia_LG", lambda: (None, None))

    # Check that solve() now raises the documented failure error.
    with pytest.raises(RuntimeError, match="did not complete successfully"):
        solver.solve()
    pass


####################################
### test regression coverage ###
####################################

@pytest.mark.regression
def test_asymmetric_interdictor_regression_theta_fallback_is_stable(
    graph,
    true_costs,
    true_delays,
    est_costs,
    est_delays,
    monkeypatch,
):
    class _FailingShortestPath:
        """Stub that always fails during longest-path setup."""

        def __init__(self, graph_arg):
            raise RuntimeError("boom")

    # Force the fallback branch and pin the returned theta value.
    monkeypatch.setattr(
        asymmetric_interdictor_module,
        "ShortestPathGrb",
        _FailingShortestPath,
    )
    interdictor = AsymmetricInterdictor(
        graph=graph,
        budget=2,
        true_costs=true_costs,
        true_delays=true_delays,
        est_costs=est_costs,
        est_delays=est_delays,
        lsd=1e-2,
    )

    # Check the exact fallback theta used by this fixture instance.
    assert interdictor.theta == pytest.approx(3700.0), \
        "Fallback theta changed on the pinned regression instance."
    pass


@pytest.mark.regression
def test_asymmetric_interdictor_regression_small_instance_solution_matches_baseline(
    solver,
):
    # Solve the pinned small asymmetric instance end to end.
    x_star, z_star = solver.solve()

    # Check the known interdiction pattern and objective baseline.
    assert x_star == [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0], \
        "Pinned small-instance interdiction pattern changed unexpectedly."
    assert z_star == pytest.approx(11.0), \
        "Pinned small-instance objective changed unexpectedly."
    pass


@pytest.mark.regression
def test_asymmetric_interdictor_regression_zero_budget_matches_uninterdicted_case(
    graph,
    true_costs,
    true_delays,
    est_costs,
    est_delays,
):
    # Build a version of the fixture problem with no available interdictions.
    interdictor = AsymmetricInterdictor(
        graph=graph,
        budget=0,
        true_costs=true_costs,
        true_delays=true_delays,
        est_costs=est_costs,
        est_delays=est_delays,
        lsd=1e-2,
    )

    # Solve the zero-budget instance end to end.
    x_star, z_star = interdictor.solve()

    # Check that no arcs are interdicted and the baseline objective is stable.
    assert x_star == [0.0] * len(graph.arcs), \
        "Zero-budget solve still interdicted one or more arcs."
    assert z_star == pytest.approx(9.0), \
        "Zero-budget baseline objective changed unexpectedly."
    pass


#####################################
### test integration coverage ###
#####################################

@pytest.mark.integration
def test_asymmetric_interdictor_integration_solves_end_to_end_with_live_models(
    solver,
):
    # Solve the fixture instance with the live optimistic and pessimistic models.
    x_star, z_star = solver.solve()

    # Check the returned vector shape, domain, budget, and objective.
    assert len(x_star) == len(solver.graph.arcs), \
        "End-to-end solve returned the wrong number of arc decisions."
    _assert_binary_and_budget_feasible(x_star, solver.budget)
    assert z_star == pytest.approx(11.0), \
        "End-to-end live solve returned the wrong objective value."
    pass


@pytest.mark.integration
def test_asymmetric_interdictor_integration_solve_matches_solve_spnia_lg(
    solver,
):
    # Solve once through each public interface on the same fixture instance.
    x_dict, z_dict = solver.solve_spnia_LG()
    x_list, z_list = solver.solve()

    # Check that solve() only converts the solution representation.
    assert x_list == [x_dict[arc] for arc in solver.graph.arcs], \
        "solve and solve_spnia_LG disagree on the chosen interdictions."
    assert z_list == pytest.approx(z_dict), \
        "solve and solve_spnia_LG disagree on the objective value."
    pass


@pytest.mark.integration
def test_asymmetric_interdictor_integration_returns_arc_aligned_grid_solution():
    # Build a small grid instance with distinct true and estimated data.
    grid = Grid(2, 3)
    true_costs = np.asarray(grid.cost, dtype=float)
    true_delays = np.arange(1.0, len(grid.arcs) + 1.0)
    est_costs = true_costs + 0.5
    est_delays = true_delays + 0.25
    interdictor = AsymmetricInterdictor(
        graph=grid,
        budget=1,
        true_costs=true_costs,
        true_delays=true_delays,
        est_costs=est_costs,
        est_delays=est_delays,
        lsd=1e-2,
    )

    # Solve the structured instance end to end.
    x_star, z_star = interdictor.solve()

    # Check that the returned vector stays arc aligned and budget feasible.
    assert len(x_star) == len(grid.arcs), \
        "Grid integration solve returned the wrong number of arc decisions."
    _assert_binary_and_budget_feasible(x_star, interdictor.budget)
    assert z_star == pytest.approx(3.0), \
        "Grid integration solve returned the wrong objective value."
    pass
