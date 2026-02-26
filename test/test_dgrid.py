import os
import sys
import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

pytest.importorskip("pyepo")

from src.dflintdpy.models.dgrid import DGrid
from src.dflintdpy.models.graph import Graph
from src.dflintdpy.solvers.shortest_path_grb import ShortestPathGrb


def test_dgrid_arcs_order():
    grid = DGrid(2, 3)

    expected_arcs = [
        (0, 1),
        (1, 2),
        (0, 3),
        (1, 4),
        (2, 5),
        (0, 4),
        (1, 5),
        (3, 4),
        (4, 5),
    ]

    assert grid.arcs == expected_arcs


# def test_dgrid_arcs_one_hot_matches_graph():
#     m, n = 3, 3
#     total_arcs = m * (n - 1) + (m - 1) * n + (m - 1) * (n - 1)
#     grid = DGrid(m, n, cost=np.arange(1, 1 + total_arcs))
#     path = [0, 4, 8]

#     one_hot, obj = grid._arcs_one_hot(path)
#     expected_one_hot, expected_obj = Graph._arcs_one_hot(grid, path)

#     assert np.array_equal(one_hot, expected_one_hot)
#     assert obj == expected_obj


def test_dgrid_scale_diagonal_edges():
    m, n = 3, 3
    total_arcs = m * (n - 1) + (m - 1) * n + (m - 1) * (n - 1)
    cost = np.ones(total_arcs)
    grid = DGrid(m, n, cost=cost)

    expected_scaled_cost = np.array(
        [1.0, 1.0, 
         1.0, 1.0, 1.0, 
         0.5, 0.5, 
         1.0, 1.0, 
         1.0, 1.0, 1.0, 
         0.5, 0.5, 
         1.0, 1.0]
    )

    assert np.array_equal(
        expected_scaled_cost, 
        grid.scale_diagonal_edges(scale=0.5)
    )


def test_dgrid_shortest_path_integration():
    m, n = 3, 3
    total_arcs = m * (n - 1) + (m - 1) * n + (m - 1) * (n - 1)
    cost = np.arange(1, 1 + total_arcs)
    grid = DGrid(m, n, cost=cost)

    # source = 0
    # target = m * n - 1
    
    opt_model = ShortestPathGrb(graph=grid)
    shortest_path, objective = opt_model.solve()

    expected_path = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    assert shortest_path == expected_path


if __name__ == "__main__":
    pytest.main([__file__])
