# test_shortest_path_grb.py
import numpy as np
import pytest
from gurobipy import GRB

from copy import deepcopy

from dflintdpy.models.graph import Graph
from dflintdpy.models.grid import Grid
from dflintdpy.models.dgrid import DGrid
from dflintdpy.solvers.shortest_path_grb import ShortestPathGrb
import torch


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
    n_costs = m * (n-1) + (m-1) * n
    cost = np.arange(1.0, n_costs+1.0, 1.0)
    return Grid(m, n, cost)

@pytest.fixture
def graph_path(graph: Graph) -> tuple[list[int], np.ndarray[float]]:
    path = [0, 1, 2, 3, 5]

    one_hot = np.zeros(graph.num_cost)
    one_hot[0] = 1  # (0, 1)
    one_hot[2] = 1  # (1, 2)
    one_hot[3] = 1  # (2, 3)
    one_hot[6] = 1  # (3, 5)

    return path, one_hot


def _assert_flow_balance(graph: Graph, sol: np.ndarray[float]) -> None:
    """Check that a solution satisfies the source-target flow constraints."""
    flow = np.asarray(sol, dtype=float)

    # Check that the solution has one entry per arc and stays nonnegative.
    assert flow.shape == (len(graph.arcs),), "Solution has the wrong shape"
    assert np.all(flow >= -1e-7), "Solution contains negative arc flow"

    # Check the flow conservation equation on every vertex.
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
            f"Flow conservation is violated at vertex {vertex}"


#####################
### test __init__ ###
#####################

# integration with graph
def test_sp_init_graph_cost(graph, grid):
    """Test that the cost of the underlying graph is correctly assigned."""
    sp_graph = ShortestPathGrb(graph)

    # test graph attributes directly
    assert np.array_equal(sp_graph._graph.cost, graph.cost), \
        "Graph cost not assigned correctly in __init__"

    # test the cost property
    assert np.array_equal(sp_graph.cost, graph.cost), \
        "Graph cost property not assigned correctly in __init__"


def test_sp_init_grid_copies_grid_data_into_the_live_model(grid):
    """Test that __init__ preserves Grid data and applies it to the Gurobi model."""
    sp_grid = ShortestPathGrb(grid)
    sp_grid._model.update()

    assert isinstance(sp_grid._graph, Grid), \
        "Grid input was not preserved when copied into the solver"
    assert sp_grid._graph is not grid, \
        "Grid was not deep copied in __init__"
    assert sp_grid._graph.m == grid.m and sp_grid._graph.n == grid.n, \
        "Grid dimensions were not copied correctly in __init__"
    assert np.array_equal(sp_grid._graph.cost, grid.cost), \
        "Grid cost not assigned correctly in __init__"
    assert np.array_equal(sp_grid.cost, grid.cost), \
        "Grid cost property not assigned correctly in __init__"
    assert [var.Obj for var in sp_grid._model.getVars()] == pytest.approx(grid.cost), \
        "Grid cost was not pushed into the Gurobi model during __init__"


def test_sp_init_graph_deepcopy(graph):
    """Test that the graph is deep copied in the __init__ method."""
    org_graph = deepcopy(graph)
    sp_graph = ShortestPathGrb(org_graph)

    assert sp_graph._graph is not org_graph, \
        "Graph was not deep copied in __init__"
    assert np.array_equal(sp_graph._graph.cost, org_graph.cost), \
        "Graph cost was not deep copied in __init__"
    assert np.array_equal(sp_graph._graph.vertices, org_graph.vertices), \
        "Graph vertices were not deep copied in __init__"
    assert np.array_equal(sp_graph._graph.arcs, org_graph.arcs), \
        "Graph arcs were not deep copied in __init__"



#######################
### test empty_grid ###
#######################

def test_sp_empty_grid(grid):
    m, n = 3, 3
    grid = ShortestPathGrb.empty_grid(m, n)

    assert isinstance(grid, ShortestPathGrb), \
        "empty_grid did not return an instance of ShortestPathGrb"
    assert np.array_equal(grid._graph.cost, Grid(m, n).cost), \
        "empty_grid did not initialize the grid cost correctly"
    assert np.array_equal(grid._graph.vertices, Grid(m, n).vertices), \
        "empty_grid did not initialize the grid vertices correctly"
    assert np.array_equal(grid._graph.arcs, Grid(m, n).arcs), \
        "empty_grid did not initialize the grid arcs correctly"
    pass


#####################
### test evaluate ###
#####################

def test_sp_evaluate_types(graph, graph_path):
    """Test that the evaluate method can handle lists, numpy arrays, and tensors."""
    path_np = graph_path[1]
    intd_np = np.array([0.0, 0, 2, 3, 0, 1, 0])
    path_list = path_np.tolist()
    intd_list = intd_np.tolist()
    path_tensor = torch.tensor(path_np)
    intd_tensor = torch.tensor(intd_np)

    sp = ShortestPathGrb(graph)
    cost = sp.cost

    obj_list = sp.evaluate(path_list, intd_list)
    obj_np = sp.evaluate(path_np, intd_np)
    obj_tensor = sp.evaluate(path_tensor, intd_tensor)

    true_obj = path_np @ cost + path_np @ intd_np

    assert obj_list == true_obj, "evaluate did not return the correct objective for list inputs"
    assert obj_np == true_obj, "evaluate did not return the correct objective for numpy array inputs"
    assert obj_tensor == true_obj, "evaluate did not return the correct objective for tensor inputs"
    pass


def test_sp_evaluate_batched_tensor_paths(graph):
    """Test that evaluate returns one objective per row for batched path tensors."""
    batch_paths = torch.tensor([
        [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ], dtype=torch.float64)
    sp = ShortestPathGrb(graph)

    objs = sp.evaluate(batch_paths)
    expected = torch.sum(batch_paths * torch.tensor(graph.cost, dtype=batch_paths.dtype), dim=1)

    assert isinstance(objs, torch.Tensor), \
        "evaluate should return a tensor for batched tensor paths"
    assert objs.shape == expected.shape, \
        "evaluate returned batched objectives with the wrong shape"
    assert torch.allclose(objs, expected), \
        "evaluate returned the wrong objectives for batched tensor paths"


def test_sp_evaluate_batched_tensor_paths_with_batched_interdictions(graph):
    """Test that evaluate handles one interdiction vector per batched path."""
    batch_paths = torch.tensor([
        [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ], dtype=torch.float64)
    batch_intd = torch.tensor([
        [0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 4.0, 0.0, 0.0, 2.0],
        [0.0, 5.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ], dtype=torch.float64)
    sp = ShortestPathGrb(graph)

    objs = sp.evaluate(batch_paths, batch_intd)
    cost = torch.tensor(graph.cost, dtype=batch_paths.dtype)
    expected = torch.sum(batch_paths * (cost + batch_intd), dim=1)

    assert isinstance(objs, torch.Tensor), \
        "evaluate should return a tensor for batched tensor inputs"
    assert objs.shape == expected.shape, \
        "evaluate returned batched objectives with the wrong shape for batched interdictions"
    assert torch.allclose(objs, expected), \
        "evaluate returned the wrong objectives for batched tensor interdictions"


def test_sp_evaluate_batched_tensor_paths_with_shared_interdictions(graph):
    """Test that evaluate broadcasts one interdiction vector across batched tensor paths."""
    batch_paths = torch.tensor([
        [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ], dtype=torch.float64)
    shared_intd = torch.tensor([0.0, 1.0, 2.0, 3.0, 0.0, 4.0, 5.0], dtype=torch.float64)
    sp = ShortestPathGrb(graph)

    objs = sp.evaluate(batch_paths, shared_intd)
    cost = torch.tensor(graph.cost, dtype=batch_paths.dtype)
    expected = torch.sum(batch_paths * (cost + shared_intd), dim=1)

    assert isinstance(objs, torch.Tensor), \
        "evaluate should return a tensor when broadcasting interdictions across batched paths"
    assert objs.shape == expected.shape, \
        "evaluate returned batched objectives with the wrong shape for shared interdictions"
    assert torch.allclose(objs, expected), \
        "evaluate returned the wrong objectives when broadcasting interdictions"


def test_sp_evaluate_raises_type_error_on_invalid_path_type(graph):
    """Test that evaluate rejects path inputs of unsupported types."""
    sp = ShortestPathGrb(graph)

    with pytest.raises(TypeError, match="Expected vector to be a numpy array"):
        sp.evaluate("not a path")


def test_sp_evaluate_raises_value_error_on_wrong_path_length(graph):
    """Test that evaluate rejects 1D paths whose length does not match the arc count."""
    sp = ShortestPathGrb(graph)

    with pytest.raises(ValueError, match=f"Expected vector to have length {graph.num_cost}"):
        sp.evaluate(np.ones(graph.num_cost - 1))


def test_sp_evaluate_raises_value_error_on_batched_path_with_wrong_columns(graph):
    """Test that evaluate rejects batched tensor paths with the wrong number of columns."""
    bad_batch_paths = torch.ones((3, graph.num_cost - 1), dtype=torch.float64)
    sp = ShortestPathGrb(graph)

    with pytest.raises(ValueError, match=f"Expected batched paths to have {graph.num_cost} columns"):
        sp.evaluate(bad_batch_paths)


def test_sp_evaluate_raises_value_error_on_batched_interdictions_with_wrong_length(graph):
    """Test that evaluate rejects shared interdictions whose length does not match the arc count."""
    batch_paths = torch.ones((3, graph.num_cost), dtype=torch.float64)
    bad_shared_intd = torch.ones(graph.num_cost - 1, dtype=torch.float64)
    sp = ShortestPathGrb(graph)

    with pytest.raises(ValueError, match=f"Expected interdictions to have length {graph.num_cost}"):
        sp.evaluate(batch_paths, bad_shared_intd)


def test_sp_evaluate_raises_value_error_on_batched_interdictions_with_wrong_shape(graph):
    """Test that evaluate rejects batched interdictions that do not match the path batch shape."""
    batch_paths = torch.ones((3, graph.num_cost), dtype=torch.float64)
    bad_batch_intd = torch.ones((2, graph.num_cost), dtype=torch.float64)
    sp = ShortestPathGrb(graph)

    with pytest.raises(ValueError, match="Expected batched interdictions to match path shape"):
        sp.evaluate(batch_paths, bad_batch_intd)


def test_sp_evaluate_raises_value_error_on_batched_interdictions_with_wrong_rank(graph):
    """Test that evaluate rejects interdictions with rank higher than two for batched paths."""
    batch_paths = torch.ones((3, graph.num_cost), dtype=torch.float64)
    bad_rank_intd = torch.ones((3, graph.num_cost, 1), dtype=torch.float64)
    sp = ShortestPathGrb(graph)

    with pytest.raises(ValueError, match="Expected interdictions to be 1D or 2D"):
        sp.evaluate(batch_paths, bad_rank_intd)


###################
### test setObj ###
###################

def test_sp_setObj_types(graph):
    """Test that the setObj method can handle lists, numpy arrays, and tensors."""
    cost_np = np.arange(1.0, graph.num_cost + 1.0, 1.0)
    cost_list = cost_np.tolist()
    cost_tensor = torch.tensor(cost_np)

    sp = ShortestPathGrb(graph)

    sp.setObj(cost_list)
    assert np.array_equal(sp.cost, cost_np), "setObj did not update the cost correctly for list input" 

    sp.setObj(cost_np)
    assert np.array_equal(sp.cost, cost_np), "setObj did not update the cost correctly for numpy array input"

    sp.setObj(cost_tensor)
    assert np.array_equal(sp.cost, cost_np), "setObj did not update the cost correctly for tensor input"
    pass


def test_sp_setObj_updates_gurobi_objective_coefficients(graph):
    """Test that setObj updates the objective coefficients stored in the Gurobi model."""
    new_cost = np.array([10.0, 1.0, 10.0, 1.0, 10.0, 10.0, 1.0])
    sp = ShortestPathGrb(graph)

    sp.setObj(new_cost)
    sp._model.update()

    assert [var.Obj for var in sp._model.getVars()] == pytest.approx(new_cost), \
        "setObj did not update the Gurobi model objective coefficients"


def test_sp_setObj_changes_the_next_solution_without_rebuilding(graph):
    """Test that solve immediately uses the updated objective after setObj."""
    new_cost = np.array([10.0, 1.0, 10.0, 1.0, 10.0, 10.0, 1.0])
    expected_solution = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    sp = ShortestPathGrb(graph)

    sp.setObj(new_cost)
    sol, obj = sp.solve()

    assert np.array_equal(sol, expected_solution), \
        "solve did not use the updated Gurobi objective after setObj"
    assert obj == pytest.approx(expected_solution @ new_cost), \
        "solve returned the wrong objective value after setObj"


#####################
### test deepcopy ###
#####################

def test_sp_deepcopy(graph):
    """Test that deepcopy creates an independent solver with an equivalent model."""
    sp = ShortestPathGrb(graph)
    new_cost = np.arange(10.0, 10.0 + graph.num_cost, 1.0)
    sp.setObj(new_cost)

    sp_copy = deepcopy(sp)

    # Check that new instances are created
    assert sp_copy is not sp, "deepcopy returned the original solver instance"
    assert sp_copy._graph is not sp._graph, "Graph was not deep copied"
    assert sp_copy._model is not sp._model, "Gurobi model was not deep copied"

    # Check that the graphs are equivalent
    assert np.array_equal(sp_copy._graph.cost, sp._graph.cost), \
        "Graph cost does not match after deepcopy"
    assert np.array_equal(sp_copy._graph.vertices, sp._graph.vertices), \
        "Graph vertices do not match after deepcopy"
    assert sp_copy._graph.arcs == sp._graph.arcs, \
        "Graph arcs do not match after deepcopy"
    assert sp_copy._graph.source == sp._graph.source, \
        "Graph source does not match after deepcopy"
    assert sp_copy._graph.target == sp._graph.target, \
        "Graph target does not match after deepcopy"

    sp._model.update()
    sp_copy._model.update()

    original_vars = sp._model.getVars()
    copied_vars = sp_copy._model.getVars()

    # Check that the Gurobi models are equivalent
    assert sp_copy._model.ModelSense == sp._model.ModelSense, \
        "Model sense does not match after deepcopy"
    assert sp_copy._model.NumVars == sp._model.NumVars, \
        "Number of model variables does not match after deepcopy"
    assert sp_copy._model.NumConstrs == sp._model.NumConstrs, \
        "Number of model constraints does not match after deepcopy"
    assert [var.VarName for var in copied_vars] == [var.VarName for var in original_vars], \
        "Model variable names do not match after deepcopy"
    assert [var.Obj for var in copied_vars] == pytest.approx([var.Obj for var in original_vars]), \
        "Model objective coefficients do not match after deepcopy"



######################
### test _getModel ###
######################

def test_sp_getModel_sets_minimization_objective_on_arc_variables(graph):
    """Test that the built Gurobi model minimizes the graph arc costs."""
    sp = ShortestPathGrb(graph)
    sp._model.update()

    vars_ = sp._model.getVars()

    assert sp._model.ModelSense == GRB.MINIMIZE, \
        "_getModel did not create a minimization model"
    assert sp._model.NumVars == len(graph.arcs), \
        "_getModel did not create one decision variable per arc"
    assert [var.VType for var in vars_] == [GRB.CONTINUOUS] * len(graph.arcs), \
        "_getModel did not create continuous arc variables"
    assert [var.Obj for var in vars_] == pytest.approx(graph.cost), \
        "The Gurobi model objective coefficients do not match the graph cost vector"


def test_sp_getModel_adds_flow_balance_constraints_for_each_vertex(graph):
    """Test that each vertex gets the correct flow conservation constraint."""
    sp = ShortestPathGrb(graph)
    sp._model.update()

    constrs = sp._model.getConstrs()

    assert len(constrs) == len(graph.vertices), \
        "_getModel did not create one flow-balance constraint per vertex"

    for vertex, constr in zip(graph.vertices, constrs):
        row = sp._model.getRow(constr)
        coeffs = {
            row.getVar(i).VarName: row.getCoeff(i)
            for i in range(row.size())
        }
        expected_coeffs = {}

        for tail, head in graph.arcs:
            if vertex == head:
                expected_coeffs[f"x[{tail},{head}]"] = 1.0
            elif vertex == tail:
                expected_coeffs[f"x[{tail},{head}]"] = -1.0

        if vertex == graph.source:
            expected_rhs = -1.0
        elif vertex == graph.target:
            expected_rhs = 1.0
        else:
            expected_rhs = 0.0

        assert constr.Sense == GRB.EQUAL, \
            f"Vertex {vertex} constraint is not an equality constraint"
        assert constr.RHS == pytest.approx(expected_rhs), \
            f"Vertex {vertex} constraint has the wrong right-hand side"
        assert coeffs == pytest.approx(expected_coeffs), \
            f"Vertex {vertex} constraint has the wrong flow-balance coefficients"



##################
### test solve ###
##################

def test_sp_solve_with_cost_uses_temp_obj(graph):
    """Test that solve uses a temporary cost vector and restores the model afterward."""
    temp_cost = np.array([10.0, 1.0, 10.0, 1.0, 10.0, 10.0, 1.0])
    expected_sol = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    sp = ShortestPathGrb(graph)

    # Solve with a temporary objective and keep the original cost for comparison.
    original_cost = graph.cost.copy()
    sol, obj = sp.solve(c=temp_cost)
    sp._model.update()

    # Check that solve used the temporary objective but restored the stored one.
    assert np.array_equal(sol, expected_sol), \
        "solve did not use the provided temporary cost vector"
    assert obj == pytest.approx(expected_sol @ temp_cost), \
        "solve returned the wrong objective for the temporary cost vector"
    assert np.array_equal(sp.cost, original_cost), \
        "solve did not restore the graph cost after using a temporary objective"
    assert [var.Obj for var in sp._model.getVars()] == pytest.approx(original_cost), \
        "solve did not restore the Gurobi objective after using a temporary cost vector"


def test_sp_solve_graph(graph, graph_path):
    """Test that solve returns the shortest path on a generic graph."""
    _, expected_sol = graph_path
    sp = ShortestPathGrb(graph)

    # Solve the graph instance with its default objective.
    sol, obj = sp.solve()

    # Check that the expected shortest path and objective are returned.
    assert np.array_equal(sol, expected_sol), \
        "solve did not return the expected shortest path on the graph"
    assert obj == pytest.approx(graph.cost @ expected_sol), \
        "solve returned the wrong shortest-path objective on the graph"


def test_sp_solve_negative_costs():
    """Test that solve handles negative costs when no negative cycle exists."""
    arcs = [(0, 1), (0, 2), (1, 3), (2, 3)]
    vertices = [0, 1, 2, 3]
    cost = np.array([2.0, -3.0, 1.0, 1.0])
    expected_sol = np.array([0.0, 1.0, 0.0, 1.0])
    sp = ShortestPathGrb(Graph(arcs, vertices, cost))

    # Solve a graph with negative edge weights but no cycle.
    sol, obj = sp.solve()

    # Check that the negative-cost path is chosen correctly.
    assert np.array_equal(sol, expected_sol), \
        "solve did not return the expected path with negative edge costs"
    assert obj == pytest.approx(cost @ expected_sol), \
        "solve returned the wrong objective on a graph with negative costs"


def test_sp_solve_grid(grid):
    """Test that solve returns the shortest path on a grid."""
    expected_sol = np.zeros(len(grid.arcs))
    expected_sol[[0, 1, 4, 9]] = 1.0
    sp = ShortestPathGrb(grid)

    # Solve the grid instance with its default arc costs.
    sol, obj = sp.solve()

    # Check that the unique shortest grid path is returned.
    assert np.array_equal(sol, expected_sol), \
        "solve did not return the expected shortest path on the grid"
    assert obj == pytest.approx(grid.cost @ expected_sol), \
        "solve returned the wrong shortest-path objective on the grid"


def test_sp_solve_fails_on_negative_cycle():
    """Test that solve raises an error on a reachable negative-cost cycle."""
    arcs = [(0, 1), (1, 2), (2, 1), (2, 3)]
    vertices = [0, 1, 2, 3]
    cost = np.array([0.0, -2.0, 1.0, 0.0])
    sp = ShortestPathGrb(Graph(arcs, vertices, cost))

    # Solve a graph whose cycle can decrease the objective without bound.
    with pytest.raises(RuntimeError, match="unbounded"):
        sp.solve()


def test_sp_solve_skips_positive_cycle():
    """Test that solve does not traverse a positive-cost cycle."""
    arcs = [(0, 1), (1, 2), (2, 1), (2, 3), (0, 3)]
    vertices = [0, 1, 2, 3]
    cost = np.array([1.0, 1.0, 2.0, 1.0, 10.0])
    expected_sol = np.array([1.0, 1.0, 0.0, 1.0, 0.0])
    sp = ShortestPathGrb(Graph(arcs, vertices, cost))

    # Solve a graph that contains a positive cycle along the useful path.
    sol, obj = sp.solve()

    # Check that the cycle edge is not used in the shortest path.
    assert np.array_equal(sol, expected_sol), \
        "solve used a positive-cost cycle in the returned path"
    assert obj == pytest.approx(cost @ expected_sol), \
        "solve returned the wrong objective on a graph with a positive cycle"


def test_sp_solve_returns_feasible_flow(graph):
    """Test that solve returns a feasible source-target flow."""
    sp = ShortestPathGrb(graph)

    # Solve the graph and verify the path feasibility directly.
    sol, _ = sp.solve()

    # Check that the source-target flow satisfies all conservation constraints.
    _assert_flow_balance(graph, sol)


def test_sp_solve_obj_matches_evaluate(graph):
    """Test that the returned objective matches evaluate on the solution."""
    sp = ShortestPathGrb(graph)

    # Solve the graph and re-evaluate the returned solution.
    sol, obj = sp.solve()

    # Check that the reported and recomputed objectives agree.
    assert obj == pytest.approx(sp.evaluate(sol)), \
        "solve returned an objective that does not match evaluate(solution)"


def test_sp_solve_fails_on_disconnected_graph():
    """Test that solve raises an error when source and target are disconnected."""
    arcs = [(0, 1), (2, 3)]
    vertices = [0, 1, 2, 3]
    cost = np.array([1.0, 1.0])
    sp = ShortestPathGrb(Graph(arcs, vertices, cost))

    # Solve a graph with no path from the default source to the default target.
    with pytest.raises(RuntimeError, match="infeasible"):
        sp.solve()


def test_sp_solve_multi_opt_checks_obj_and_flow():
    """Test that solve checks value and feasibility when multiple optima exist."""
    arcs = [(0, 1), (1, 3), (0, 2), (2, 3)]
    vertices = [0, 1, 2, 3]
    cost = np.array([1.0, 1.0, 1.0, 1.0])
    sp = ShortestPathGrb(Graph(arcs, vertices, cost))

    # Solve a graph with two equally short source-target paths.
    sol, obj = sp.solve()

    # Check the optimal value and feasibility without fixing one exact path.
    _assert_flow_balance(sp._graph, sol)
    assert obj == pytest.approx(2.0), \
        "solve returned the wrong objective when multiple shortest paths exist"
    assert obj == pytest.approx(sp.evaluate(sol)), \
        "solve returned an objective that does not match evaluate(solution)"


def test_sp_solve_with_cost_kwarg_uses_temp_obj(graph):
    """Test that solve accepts a temporary objective through the cost kwarg."""
    temp_cost = np.array([10.0, 1.0, 10.0, 1.0, 10.0, 10.0, 1.0])
    expected_sol = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    sp = ShortestPathGrb(graph)

    # Solve with the documented cost kwarg instead of the c argument.
    sol, obj = sp.solve(cost=temp_cost)

    # Check that the kwarg path is handled exactly like c.
    assert np.array_equal(sol, expected_sol), \
        "solve did not use the temporary objective passed through cost"
    assert obj == pytest.approx(expected_sol @ temp_cost), \
        "solve returned the wrong objective for the temporary cost kwarg"


def test_sp_setObj_copies_external_cost_on_all_graph_types(graph, grid):
    """Test that setObj copies ndarray, list, and tensor inputs."""
    dgrid = DGrid(3, 3, cost=np.arange(1.0, 17.0, 1.0))
    graph_types = [
        (ShortestPathGrb(graph), np.arange(11.0, 18.0, 1.0)),
        (ShortestPathGrb(grid), np.arange(21.0, 33.0, 1.0)),
        (ShortestPathGrb(dgrid), np.arange(31.0, 47.0, 1.0)),
    ]

    for sp, base_cost in graph_types:
        cost_inputs = [
            np.array(base_cost, copy=True),
            base_cost.tolist(),
            torch.tensor(base_cost, dtype=torch.float32),
        ]

        for new_cost in cost_inputs:
            sp.setObj(new_cost)
            stored_cost = np.array(base_cost, copy=True)

            new_cost[0] = -999.0
            sp._model.update()

            if isinstance(new_cost, np.ndarray):
                assert not np.shares_memory(sp.cost, new_cost), \
                    "setObj kept a shared reference to the caller's cost array"

            assert np.array_equal(sp.cost, stored_cost), \
                "setObj did not preserve an internal copy of the provided cost array"
            assert [var.Obj for var in sp._model.getVars()] == pytest.approx(stored_cost), \
                "setObj did not preserve the copied cost in the Gurobi model"


def test_sp_solve_batched_torch_costs_returns_one_solution_and_objective_per_row(graph):
    """Test that solve returns one shortest path and one objective per tensor row."""
    batch_costs = torch.tensor(np.vstack([
        graph.cost,
        [10.0, 1.0, 10.0, 1.0, 10.0, 10.0, 1.0],
        [10.0, 1.0, 10.0, 10.0, 10.0, 1.0, 10.0],
    ]), dtype=torch.float64)
    expected_solutions = torch.tensor([
        [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ], dtype=torch.float64)
    expected_objectives = torch.sum(batch_costs * expected_solutions, dim=1)
    sp = ShortestPathGrb(graph)

    sols, objs = sp.solve(c=batch_costs)

    assert isinstance(sols, torch.Tensor), \
        "solve should return batched solutions as a tensor for batched tensor costs"
    assert isinstance(objs, torch.Tensor), \
        "solve should return batched objectives as a tensor for batched tensor costs"
    assert sols.shape == expected_solutions.shape, \
        "solve returned batched solutions with the wrong shape"
    assert objs.shape == expected_objectives.shape, \
        "solve returned batched objectives with the wrong shape"
    assert torch.allclose(sols, expected_solutions), \
        "solve returned the wrong batched shortest-path solutions"
    assert torch.allclose(objs, expected_objectives), \
        "solve returned the wrong batched objective values"

    for cost_row, sol, obj in zip(batch_costs, sols, objs):
        _assert_flow_balance(graph, sol.detach().cpu().numpy())
        assert torch.dot(cost_row, sol) == pytest.approx(obj.item()), \
            "A batched solution/objective pair does not agree with evaluate"


def test_sp_solve_batched_torch_costs_accepts_cost_kwarg(graph):
    """Test that solve accepts batched tensor costs through the documented cost kwarg."""
    batch_costs = torch.tensor(np.vstack([
        graph.cost,
        [10.0, 1.0, 10.0, 1.0, 10.0, 10.0, 1.0],
        [10.0, 1.0, 10.0, 10.0, 10.0, 1.0, 10.0],
    ]), dtype=torch.float64)
    sp_c = ShortestPathGrb(graph)
    sp_kwarg = ShortestPathGrb(graph)

    sols_from_c, objs_from_c = sp_c.solve(c=batch_costs)
    sols_from_kwarg, objs_from_kwarg = sp_kwarg.solve(cost=batch_costs)

    assert torch.allclose(sols_from_kwarg, sols_from_c), \
        "solve(cost=...) returned different batched solutions than solve(c=...)"
    assert torch.allclose(objs_from_kwarg, objs_from_c), \
        "solve(cost=...) returned different batched objectives than solve(c=...)"


def test_sp_solve_batched_torch_costs_restores_original_objective(graph):
    """Test that solve restores the stored graph and Gurobi objectives after a batched solve."""
    batch_costs = torch.tensor(np.vstack([
        graph.cost,
        [10.0, 1.0, 10.0, 1.0, 10.0, 10.0, 1.0],
        [10.0, 1.0, 10.0, 10.0, 10.0, 1.0, 10.0],
    ]), dtype=torch.float64)
    original_cost = graph.cost.copy()
    sp = ShortestPathGrb(graph)

    sp.solve(c=batch_costs)
    sp._model.update()

    assert np.array_equal(sp.cost, original_cost), \
        "solve did not restore the graph cost after using batched tensor costs"
    assert [var.Obj for var in sp._model.getVars()] == pytest.approx(original_cost), \
        "solve did not restore the Gurobi objective after using batched tensor costs"

# TODO: Check how the SPO+ loss function in pyepo works, given that shortest_path_grb wasn't able to handle tensors until now.
