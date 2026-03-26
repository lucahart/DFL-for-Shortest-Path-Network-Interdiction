import pytest
import numpy as np
from copy import deepcopy
# from networkx import graphs_equal

pytest.importorskip("pyepo")

from dflintdpy.models.grid import Grid
    
def test_copy_parameters():
    """
    Test if the copy method creates a new instance with the same parameters.
    """
    # Generate grid instance
    m, n = (5,5)  # grid size
    cost = np.arange((m-1)*n + m*(n-1))
    original = Grid(m, n, cost=cost)

    # Create a copy of the original instance
    copy = deepcopy(original)

    # Check if the copy is a new instance and has the same parameters
    assert isinstance(copy, Grid), "Copy should be an instance of Grid."
    assert copy.m == original.m, "Copy should have the same number of rows."
    assert copy.n == original.n, "Copy should have the same number of columns."
    assert np.array_equal(copy.arcs, original.arcs), "Copy should have the same arcs."
    assert np.array_equal(copy.vertices, original.vertices), "Copy should have the same vertices."
    assert np.array_equal(copy.cost, original.cost), "Copy should have the same cost vector."
    # assert graphs_equal(copy.graph, original.graph), "Graph structure should be the same in both instances."
    assert copy.source == original.source, "Source should be the same in both instances."
    assert copy.target == original.target, "Target should be the same in both instances."
    pass


def test_copy_new_instance():
    """
    Test if the copy method creates a new instance.
    """
    # Generate grid instance
    m, n = (5,5)  # grid size
    cost = np.arange((m-1)*n + m*(n-1))
    original = Grid(m, n, cost=cost)

    # Create a copy of the original instance
    copy = deepcopy(original)

    # Check if attributes of copy are new instances
    assert copy.arcs is not original.arcs, "Arcs should be a new instance in the copy."
    assert copy.vertices is not original.vertices, "Vertices should be a new instance in the copy."
    assert copy.cost is not original.cost, "Cost should be a new instance in the copy."
    assert copy.graph is not original.graph, "Graph should be a new instance in the copy."

    # Change the objective of the original instance
    original.setObj(np.zeros_like(original.cost), source = 1, target = 2)

    # Check if the copy is a new instance
    assert not np.array_equal(copy.cost, original.cost), "Cost should not be the same after changing the original."
    assert copy.target != original.target, "Target should not be the same after changing the original."
    # assert not graphs_equal(copy.graph, original.graph), "Graph structure should be different in both instances."
    pass


def test_arcs_one_hot():
    """
    Test the arcs_one_hot method of ShortestPathGrid.
    """
    # Create a grid instance
    m, n = (2, 3)  # grid size
    cost = np.arange((m - 1) * n + m * (n - 1))
    graph = Grid(m, n, cost=cost)

    # Define a characteristic path by its traversed nodes
    shortest_path_nodes = [0, 1, 2, 5]

    # Get the one-hot encoded vector
    one_hot_vector, _ = graph._arcs_one_hot(shortest_path_nodes)

    # Check the length of the one-hot vector
    expected_length = len(graph.arcs)
    assert len(one_hot_vector) == expected_length, f"One-hot vector length should be {expected_length}."

    # Check if the one-hot vector is correctly encoded
    expected_vector = np.array([1, 1, 0, 0, 1, 0, 0])
    assert np.array_equal(one_hot_vector, expected_vector), "One-hot vector should match the expected encoding."
