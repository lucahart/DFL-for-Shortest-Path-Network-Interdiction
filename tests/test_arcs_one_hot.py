import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models.ShortestPathGrid import ShortestPathGrid
from src.models.ShortestPath import ShortestPath

def test_arcs_one_hot_matches_base():
    grid = ShortestPathGrid(3, 4)
    path = [0, 1, 5, 9, 10, 11]

    one_hot, obj = grid._arcs_one_hot(path)
    expected_one_hot, expected_obj = ShortestPath._arcs_one_hot(grid, path)

    assert np.array_equal(one_hot, expected_one_hot)
    assert obj == expected_obj

def test_vertical_only_path():
    grid = ShortestPathGrid(3, 4)
    path = [0, 4, 8]

    one_hot, _ = grid._arcs_one_hot(path)
    expected_one_hot, _ = ShortestPath._arcs_one_hot(grid, path)

    assert np.array_equal(one_hot, expected_one_hot)
