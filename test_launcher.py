import pytest

test_dgrid = "/Users/lucahartmann/Documents/Professional/Research/Prof_Parinaz_Naghizadeh/Code/Shortest_Path_Interdiction/tests/unit/models/test_dgrid.py"
test_graph = "/Users/lucahartmann/Documents/Professional/Research/Prof_Parinaz_Naghizadeh/Code/Shortest_Path_Interdiction/tests/unit/models/test_graph.py"
test_grid = "/Users/lucahartmann/Documents/Professional/Research/Prof_Parinaz_Naghizadeh/Code/Shortest_Path_Interdiction/tests/unit/models/test_grid.py"
test_sp = "/Users/lucahartmann/Documents/Professional/Research/Prof_Parinaz_Naghizadeh/Code/Shortest_Path_Interdiction/tests/unit/solvers/test_shortest_path_grb.py"

pytest.main([
    test_dgrid, 
    test_graph,
    test_grid,
    # test_sp,
])
pass

