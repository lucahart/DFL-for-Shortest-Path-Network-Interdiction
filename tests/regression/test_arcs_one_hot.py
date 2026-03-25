import numpy as np
import pytest

pytest.importorskip("pyepo")

from dflintdpy.models.grid import Grid
from dflintdpy.models.graph import Graph

def test_arcs_one_hot_matches_base():
    """Tests that the one-hot encoding of arcs for a path in the Grid model 
    matches the one-hot encoding of arcs for the same path in the base Graph model.
    This ensures that the Grid model's _arcs_one_hot method is consistent with 
    the base Graph model's _arcs_one_hot method, which is important for the 
    correctness of the Grid model"""
    grid = Grid(3, 4)
    path = [0, 1, 5, 9, 10, 11]

    one_hot, obj = grid._arcs_one_hot(path)
    expected_one_hot, expected_obj = Graph._arcs_one_hot(grid, path)

    assert np.array_equal(one_hot, expected_one_hot), \
        "One-hot encoding of arcs in Grid model does not match base Graph model."
    assert obj == expected_obj, \
        "Objective value from one-hot encoding in Grid model does not match base Graph model."

def test_vertical_only_path():
    """Tests that the one-hot encoding of arcs for a vertical-only path in the Grid model 
    matches the one-hot encoding of arcs for the same path in the base Graph model.
    This test includes the case where the path is not from 0 -> m*n-1."""
    grid = Grid(3, 4)
    path = [0, 4, 8]

    try:
        one_hot, _ = grid._arcs_one_hot(path)
    except Exception as e:
        # Likely cause for exception was legacy check on # nodes in path,
        # which ignored different start/end nodes and thus fails on a vertical-only path.
        pytest.fail(f"Grid model _arcs_one_hot raised an exception: {e}")
    expected_one_hot, _ = Graph._arcs_one_hot(grid, path)

    assert np.array_equal(one_hot, expected_one_hot), \
        "One-hot encoding of vertical-only path in Grid model does not match base Graph model."

