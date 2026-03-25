# test_grid.py

from copy import deepcopy

import numpy as np
import pytest
import networkx as nx

pytest.importorskip("pyepo")
pytest.importorskip("gurobipy")

from dflintdpy.models.grid import Grid
from dflintdpy.models.graph import Graph


def test_grid_arc_order():
    """Test if the arcs are generated in the correct order for a 3x3 grid."""

    costs = np.arange(12)
    grid = Grid(3, 3, cost=costs)
    expected_arcs = [
        (0, 1), (1, 2), 
        (0, 3), (1, 4), (2, 5), 
        (3, 4), (4, 5), 
        (3, 6), (4, 7), (5, 8),
        (6, 7), (7, 8), 
    ]
    expected_costs = np.arange(12)
    assert grid.arcs == expected_arcs, "Arc order does not match expected order."
    assert np.array_equal(grid.cost, expected_costs), "Cost array does not match expected costs."

def test_grid_init_cost_length_mismatch():
    """Test if an error is raised when the cost array length does not match the number of arcs."""
    with pytest.raises(ValueError):
        Grid(2, 2, cost=np.array([1, 2, 3]))  # Should have length 4 for a 2x2 grid

def test_grid_init_no_cost():
    """Test if the grid initializes correctly when no cost is provided."""
    grid = Grid(2, 2)
    expected_arcs = [
        (0, 1), 
        (0, 2), (1, 3), 
        (2, 3)
    ]
    expected_costs = [1] * 4  # Default cost should be one for all arcs
    assert grid.arcs == expected_arcs, "Arc order does not match expected order."
    assert np.array_equal(grid.cost, expected_costs), "Cost array does not match expected costs."

def test_grid_init_cost_types():
    """Test if the grid initializes correctly with cost arrays of the types list and ndarray."""
    costs_list = [1, 2, 3, 4]
    costs_array = np.array(costs_list)
    
    grid_list_cost = Grid(2, 2, cost=costs_list)
    grid_array_cost = Grid(2, 2, cost=costs_array)
    
    assert np.array_equal(grid_list_cost.cost, costs_array), "Cost array from list does not match expected costs."
    assert np.array_equal(grid_array_cost.cost, costs_array), "Cost array from numpy array does not match expected costs."

def test_grid_deepcopy():
    """Test if the deepcopy method creates a correct copy of the grid."""
    costs = np.arange(12)
    grid = Grid(3, 3, cost=costs)
    grid_copy = deepcopy(grid)

    assert grid is not grid_copy, "Deepcopy did not create a new instance."
    assert grid.arcs == grid_copy.arcs, "Arcs do not match after deepcopy."
    assert np.array_equal(grid.cost, grid_copy.cost), "Cost arrays do not match after deepcopy."
    assert grid.m == grid_copy.m and grid.n == grid_copy.n, "Grid dimensions do not match after deepcopy."
    assert grid.graph is not grid_copy.graph, "Graph object was not deepcopied properly."
    assert nx.utils.misc.graphs_equal(grid.graph, grid_copy.graph), "Graphs are not exactly equal."
    assert grid.source == grid_copy.source and grid.target == grid_copy.target, "Source and sink nodes do not match after deepcopy."

def test_grid_arcs_one_hot():
    """Test if the _arcs_one_hot method correctly encodes a shortest path."""
    m, n = 3, 3
    total_arcs = m * (n - 1) + (m - 1) * n
    costs = np.arange(1, 1 + total_arcs)
    grid = Grid(m, n, cost=costs)
    path = [0, 1, 4, 7, 8]  # A valid path from top-left to bottom-right

    one_hot, obj = grid._arcs_one_hot(path)

    expected_one_hot = np.zeros(total_arcs)
    expected_one_hot[0] = 1  # (0, 1)
    expected_one_hot[3] = 1  # (1, 4)
    expected_one_hot[8] = 1  # (4, 7)
    expected_one_hot[11] = 1  # (7, 8)

    expected_obj = costs[0] + costs[3] + costs[8] + costs[11]

    assert np.array_equal(one_hot, expected_one_hot), "One-hot encoding does not match expected encoding."
    assert obj == expected_obj, "Objective value does not match expected value."

def test_arcs_one_hot_matches_base():
    """Test if the _arcs_one_hot method in Grid matches the implementation in the base Graph class."""
    grid = Grid(3, 4)
    path = [0, 1, 5, 9, 10, 11]

    one_hot, obj = grid._arcs_one_hot(path)
    expected_one_hot, expected_obj = Graph._arcs_one_hot(grid, path)

    assert np.array_equal(one_hot, expected_one_hot)
    assert obj == expected_obj
    pass

def test_grid_arcs_one_hot_invalid_arc_error():
    """Test if the _arcs_one_hot method raises an error if there is an invalid arc."""
    grid = Grid(3, 3)
    path = [0, 4, 5, 5, 8]

    with pytest.raises(ValueError):
        _, _ = grid._arcs_one_hot(path)
    pass

@pytest.mark.skip(reason=(
    "Test outdated. A fixed number of nodes would not take into account "
    "different start/end nodes and thus fails on, e.g., vertical-only paths."
))
def test_grid_arcs_one_hot_invalid_nodes():
    """Test if the _arcs_one_hot method raises an error when given invalid nodes."""
    grid = Grid(2, 2)
    too_high_numbers = [0, 1, 4]  # 4 is not a valid node in a 2x2 grid
    too_small_numbers = [-1, 0, 1]  # -1 is not a valid node in a 2x2 grid
    too_many_nodes = [0, 1, 2, 3]  # A path with 4 nodes is not valid for a 2x2 grid
    too_few_nodes = [0, 1]  # A path with only 2 nodes is not valid for a 2x2 grid

    with pytest.raises(ValueError):
        grid._arcs_one_hot(too_high_numbers)
    with pytest.raises(ValueError):
        grid._arcs_one_hot(too_small_numbers)
    with pytest.raises(ValueError):
        grid._arcs_one_hot(too_many_nodes)
    with pytest.raises(ValueError):
        grid._arcs_one_hot(too_few_nodes)
