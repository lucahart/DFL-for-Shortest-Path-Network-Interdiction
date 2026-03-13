# test_graph.py
from copy import deepcopy

import networkx as nx
import numpy as np
import pytest

from dflintdpy.models.graph import Graph
import torch


@pytest.fixture
def triangle_graph() -> Graph:
    arcs = [(0, 1), (0, 2), (1, 2)]
    vertices = [0, 1, 2]
    cost = np.array([1.0, 2.0, 3.0])
    return Graph(arcs, vertices, cost)

@pytest.fixture
def larger_graph() -> Graph:
    arcs = [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (2, 5), (3, 5)]
    vertices = [0, 1, 2, 3, 4, 5]
    cost = np.arange(1.0, 8.0, 1.0)
    return Graph(arcs, vertices, cost)

@pytest.fixture
def larger_graph_path(larger_graph: Graph) -> tuple[list[int], np.ndarray[float]]:
    path = [0, 1, 2, 3, 5]

    one_hot = np.zeros(larger_graph.num_cost)
    one_hot[0] = 1  # (0, 1)
    one_hot[2] = 1  # (1, 2)
    one_hot[3] = 1  # (2, 3)
    one_hot[6] = 1  # (3, 5)

    return path, one_hot


####################################
### Test fixture initializations ###
####################################
# These tests ensure that the fixtures are initialized as expected.

def test_graph_triangle_graph(triangle_graph: Graph):
    """Test that the triangle graph is initialized correctly."""
    assert triangle_graph.arcs == [(0, 1), (0, 2), (1, 2)]
    assert np.array_equal(triangle_graph.vertices, [0, 1, 2])
    assert np.array_equal(triangle_graph.cost, [1.0, 2.0, 3.0])
    assert triangle_graph.source == 0
    assert triangle_graph.target == 2
    pass

def test_graph_larger_graph(larger_graph: Graph):
    """Test that the larger graph is initialized correctly."""
    assert larger_graph.arcs == [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (2, 5), (3, 5)]
    assert np.array_equal(larger_graph.vertices, [0, 1, 2, 3, 4, 5])
    assert np.array_equal(larger_graph.cost, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    assert larger_graph.source == 0
    assert larger_graph.target == 5
    pass


######################################
### Initialization tests: __init__ ###
######################################

def test_graph_init_no_vertices():
    """Test that the graph initializes correctly when no vertices are provided."""
    arcs = [(0, 1), (0, 2), (1, 2), (2, 4)]
    cost = np.array([1.0, 2.0, 3.0, 4.0])
    graph = Graph(arcs, cost=cost)
    assert np.array_equal(graph.vertices, [0, 1, 2, 3, 4])
    pass

def test_graph_init_no_cost():
    """Test that the graph initializes correctly when no cost is provided."""
    arcs = [(0, 1), (0, 2), (1, 2)]
    vertices = [0, 1, 2]
    graph = Graph(arcs, vertices)
    assert np.array_equal(graph.cost, [1.0, 1.0, 1.0])
    pass

def test_graph_init_source_target():
    """Test that source and target are set correctly at initialization."""
    arcs = [(0, 1), (0, 2), (1, 2)]
    vertices = [0, 1, 2]
    cost = np.array([1.0, 2.0, 3.0])
    graph = Graph(arcs, vertices, cost, source=1, target=0)
    assert graph.source == 1, f"Source should be 1 but was {graph.source}"
    assert graph.target == 0, f"Target should be 0 but was {graph.target}"
    pass

def test_graph_init_networkx_graph():
    """Test if networkx graph is created correctly."""
    # Define triangle graph independent of fixture to avoid backcompatibility issues
    arcs = [(0, 1), (0, 2), (1, 2)]
    vertices = [0, 1, 2]
    cost = np.array([1.0, 2.0, 3.0])
    triangle_graph = Graph(arcs, vertices, cost)
    
    # Create networkx graph
    G = nx.DiGraph()
    G.add_nodes_from(vertices)
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(0, 2, weight=2.0)
    G.add_edge(1, 2, weight=3.0)

    # Check if graphs are equal
    assert isinstance(triangle_graph.graph, nx.DiGraph), f"Graph must be networknx instance, not {type(triangle_graph.graph)}."
    assert nx.utils.misc.graphs_equal(triangle_graph.graph, G)


###################
### test setObj ###
###################

def test_graph_setObj_source_target(triangle_graph: Graph):
    """Test that source and target are set correctly at setObj."""
    triangle_graph.setObj(triangle_graph.cost, source=1, target=0)
    assert triangle_graph.source == 1, f"Source should be 1 but was {triangle_graph.source}."
    assert triangle_graph.target == 0, f"Target should be 0 but was {triangle_graph.target}."
    pass

def test_graph_setObj_torch_tensor_type_error(triangle_graph: Graph):
    """Test that the cost is of type list of ndarray."""
    cost = torch.tensor([1.0, 2.0, 3.0])
    with pytest.raises(TypeError):
        triangle_graph.setObj(cost)
    pass

def test_graph_setObj_cost_squeeze(triangle_graph: Graph):
    """Test that the cost is squeezed."""
    cost = np.array([[1.0, 2.0, 3.0]])
    # Arrays are squeezed and no error should be raised
    triangle_graph.setObj(cost)
    assert triangle_graph.cost.ndim == 1, f"Expected cost to be 1D but got {triangle_graph.cost.ndim}D."
    assert len(triangle_graph.cost) == len(triangle_graph.arcs), \
        f"Expected cost to have length {len(triangle_graph.arcs)} but got {len(triangle_graph.cost)}."
    pass

def test_graph_setObj_cost_dimension_error():
    """Test that the cost has the correct dimension."""
    arcs = [(0, 1), (0, 2), (1, 2), (2, 3)]
    vertices = [0, 1, 2, 3]
    cost = np.array([1.0, 2.0, 3.0, 4.0])
    graph = Graph(arcs, vertices, cost)

    cost_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError):
        graph.setObj(cost_2d)
    pass

def test_graph_setObj_cost_length_error(triangle_graph: Graph):
    """Test that the cost has the correct length."""
    cost_short = np.array([1.0, 2.0])
    cost_long = np.array([1,2,3,4])
    with pytest.raises(ValueError):
        triangle_graph.setObj(cost_short)
    with pytest.raises(ValueError):
        triangle_graph.setObj(cost_long)
    pass

def test_graph_setObj_cost_values(triangle_graph: Graph):
    """Test that the cost values are set correctly."""
    # Try list and ndarray
    cost_list = [1.0, 2.0, 3.0]
    cost_ndarray = np.array([1.0, 2.0, 3.0])
    triangle_graph.setObj(cost_list)
    assert np.array_equal(triangle_graph.cost, np.array(cost_list))
    triangle_graph.setObj(cost_ndarray)
    assert np.array_equal(triangle_graph.cost, cost_ndarray)
    pass


##########################
### test _arcs_one_hot ###
##########################

def test_graph_arcs_one_hot(
        larger_graph: Graph, 
        larger_graph_path: tuple[list[int], np.ndarray[float]]
    ):
    """Test if the _arcs_one_hot method correctly encodes a shortest path."""
    # Define arbitrary path. This method does not check if the path is the shortest.
    path, expected_one_hot = larger_graph_path

    one_hot, obj = larger_graph._arcs_one_hot(path)

    expected_obj = expected_one_hot @ larger_graph.cost

    assert np.array_equal(one_hot, expected_one_hot), "One-hot encoding does not match expected encoding."
    assert obj == expected_obj, "Objective value does not match expected value."


def test_graph_arcs_one_hot_invalid_nodes(triangle_graph: Graph):
    """Test if the _arcs_one_hot method raises an error when given invalid nodes."""
    too_high_numbers = [0, 1, 3]  # triangle graph only has nodes 0, 1, 2
    too_small_numbers = [-1, 0, 1] # triangle graph only has nodes 0, 1, 2

    with pytest.raises(ValueError):
        triangle_graph._arcs_one_hot(too_high_numbers)
    with pytest.raises(ValueError):
        triangle_graph._arcs_one_hot(too_small_numbers)
    pass


############################
### test one_hot_to_arcs ###
############################

def test_graph_one_hot_to_arcs(
        larger_graph: Graph,
        larger_graph_path: tuple[list[int], np.ndarray[float]]
    ):
    """Test if the _one_hot_to_arcs method correctly decodes a one-hot vector."""
    _, one_hot = larger_graph_path

    arcs = Graph.one_hot_to_arcs(larger_graph, one_hot)

    expected_arcs = [(0, 1), (1, 2), (2, 3), (3, 5)]

    assert arcs == expected_arcs, "Decoded arcs do not match expected arcs."
    pass


##################
### test solve ###
##################

def test_graph_solve(
        larger_graph: Graph,
        larger_graph_path: tuple[list[int], np.ndarray[float]]
    ):
    """Test if the solve method correctly solves the shortest path problem."""
    new_cost = [1.0, 4.0, 2.0, 3.0, 1.0, 6.0, 2.0]
    larger_graph.setObj(new_cost)
    solution, objective = larger_graph.solve()

    _, expected_solution = larger_graph_path
    expected_objective = new_cost @ expected_solution

    assert np.array_equal(solution, expected_solution), \
        f"Solved path {solution} does not match expected path {expected_solution}."
    assert objective == expected_objective, \
        f"Objective value {objective} does not match expected value {expected_objective}."
    pass


#####################
### test evaluate ###
#####################

def test_graph_evaluate_unintd(
        larger_graph: Graph,
        larger_graph_path: tuple[list[int], np.ndarray[float]]
    ):
    """Test if the evaluate method correctly evaluates a given uninterdicted path."""
    _, one_hot = larger_graph_path
    objective = larger_graph.evaluate(one_hot)

    expected_objective = larger_graph.cost @ one_hot

    assert objective == expected_objective, \
        f"Evaluated objective {objective} does not match expected value {expected_objective}."
    pass

def test_graph_evaluate_intd(
        larger_graph: Graph,
        larger_graph_path: tuple[list[int], np.ndarray[float]]
    ):
    """Test if the evaluate method correctly evaluates a given interdicted path."""
    _, one_hot = larger_graph_path
    intd = np.array([0.0, 0, 2, 3, 0, 1, 0])

    objective = larger_graph.evaluate(one_hot, intd)

    expected_objective = (larger_graph.cost @ one_hot + intd @ one_hot)

    assert objective == expected_objective, \
        f"Evaluated objective {objective} does not match expected value {expected_objective}."
    pass


########################
### test to_1d_numpy ###
########################

def test_graph_to_1d_numpy_squeeze(triangle_graph: Graph):
    """Test that the vector is squeezed."""
    vector = np.array([[1.0, 2.0, 3.0]])
    # Arrays are squeezed and no error should be raised
    new_vector = triangle_graph._to_1d_numpy(vector)
    assert new_vector.ndim == 1, f"Expected new vector to be 1D but got {new_vector.ndim}D."
    assert len(new_vector) == len(triangle_graph.arcs), \
        f"Expected new vector to have length {len(triangle_graph.arcs)} but got {len(new_vector)}."
    pass

def test_graph_to_1d_numpy_dimension_error():
    """Test that the vector has the correct dimension."""
    arcs = [(0, 1), (0, 2), (1, 2), (2, 3)]
    vertices = [0, 1, 2, 3]
    vector = np.array([1.0, 2.0, 3.0, 4.0])
    graph = Graph(arcs, vertices, vector)

    vector_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError):
        graph._to_1d_numpy(vector_2d)
    pass

def test_graph_to_1d_numpy_length_error(triangle_graph: Graph):
    """Test that the vector has the correct length."""
    vector_single = np.array([1.0])
    vector_short = np.array([1.0, 2.0])
    vector_long = np.array([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError):
        triangle_graph._to_1d_numpy(vector_single)
    with pytest.raises(ValueError):
        triangle_graph._to_1d_numpy(vector_short)
    with pytest.raises(ValueError):
        triangle_graph._to_1d_numpy(vector_long)
    pass

def test_graph_to_1d_numpy_ndarray_conversion(triangle_graph: Graph):
    """Test that the vector is converted to a numpy array if it's a list or tensor."""
    vector_list = [1.0, 2.0, 3.0]
    vector_tensor = torch.tensor([1.0, 2.0, 3.0])
    vector_from_list = triangle_graph._to_1d_numpy(vector_list)
    vector_from_tensor = triangle_graph._to_1d_numpy(vector_tensor)
    assert isinstance(vector_from_list, np.ndarray), f"Expected new vector to be a numpy array but got {type(vector_from_list)}."
    assert np.array_equal(vector_from_list, vector_list), "Converted vector does not match original list."
    assert isinstance(vector_from_tensor, np.ndarray), f"Expected new vector to be a numpy array but got {type(vector_from_tensor)}."
    assert np.array_equal(vector_from_tensor, vector_tensor.numpy()), "Converted vector does not match original tensor."
    pass


#########################
### test __deepcopy__ ###
#########################
def test_graph_deepcopy(triangle_graph: Graph):
    """Test if the deepcopy method creates a correct copy of the graph."""
    graph_copy = deepcopy(triangle_graph)

    assert triangle_graph is not graph_copy, "Deepcopy did not create a new instance."
    assert triangle_graph.arcs == graph_copy.arcs, "Arcs do not match after deepcopy."
    assert np.array_equal(triangle_graph.vertices, graph_copy.vertices), "Vertices do not match after deepcopy."
    assert np.array_equal(triangle_graph.cost, graph_copy.cost), "Cost arrays do not match after deepcopy."
    assert triangle_graph.source == graph_copy.source, "Source does not match after deepcopy."
    assert triangle_graph.target == graph_copy.target, "Target does not match after deepcopy."
    assert triangle_graph.graph is not graph_copy.graph, "Graph object was not deepcopied properly."
    assert nx.utils.misc.graphs_equal(triangle_graph.graph, graph_copy.graph), "Graphs are not exactly equal."
    pass


###############################
### Source and target tests ###
###############################

def test_graph_source_target_type_errors(triangle_graph: Graph):
    """Test that source and target must be integers."""
    with pytest.raises(TypeError):
        triangle_graph._set_source_target("0", 2)  # Source is a string, should raise error
    with pytest.raises(TypeError):
        triangle_graph._set_source_target(0, "2")  # Target is a string, should raise error
    pass

def test_graph_source_target_value_errors(triangle_graph: Graph):
    """Test that source and target must be valid vertices."""
    with pytest.raises(ValueError):
        triangle_graph._set_source_target(-1, 2)  # Source is negative, should raise error
    with pytest.raises(ValueError):
        triangle_graph._set_source_target(0, 3)  # Target is greater than max vertex, should raise error
    pass

def test_graph_source_target_default_values():
    """Test that source defaults to the first vertex and target defaults to the largest vertex if not provided."""
    # Define graph with different vertex names
    arcs = [(0, 3), (0, 2), (3, 2)]
    vertices = [0, 3, 2]
    cost = np.array([1.0, 2.0, 3.0])
    triangle_graph = Graph(arcs, vertices, cost)

    # Test that the source is the first not lowest vertex by default
    assert triangle_graph.source == 0
    # Test that the target is not the last but highest vertex by default
    assert triangle_graph.target == 3
    pass

def test_graph_source_target_storage():
    """Test that the source and target are stored if provided, and that they are of the correct type."""
    # Define graph
    arcs = [(0, 1), (0, 2), (1, 2)]
    vertices = [0, 1, 2]
    cost = np.array([1.0, 2.0, 3.0])
    triangle_graph_with_source = Graph(arcs, vertices, cost, source=1)
    triangle_graph_with_target = Graph(arcs, vertices, cost, target=1)
    triangle_graph_with_both = Graph(arcs, vertices, cost, source=1, target=0)

    # Test that the source and target are stored correctly
    assert triangle_graph_with_source.source == 1, f"Source should be 1 but was {triangle_graph_with_source.source}"
    assert triangle_graph_with_source.target == 2, f"Target should be 2 but was {triangle_graph_with_source.target}"
    assert triangle_graph_with_target.source == 0, f"Source should be 0 but was {triangle_graph_with_target.source}"
    assert triangle_graph_with_target.target == 1, f"Target should be 1 but was {triangle_graph_with_target.target}"
    assert triangle_graph_with_both.source == 1, f"Source should be 1 but was {triangle_graph_with_both.source}"
    assert triangle_graph_with_both.target == 0, f"Target should be 0 but was {triangle_graph_with_both.target}"
    pass

def test_graph_solve_source_target_not_overwritten():
    """Test that source and target are not overwritten at solve."""
    pass

def test_graph_solve_source_target_used():
    """Test that source and target are used correctly at solve."""
    pass
