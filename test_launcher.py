import pytest

cvxpy_hackathon = "/Users/lucahartmann/Documents/Professional/Code/CVXPY-Hackathon-2026/cvxpylayers"
unit_test_models = "/Users/lucahartmann/Documents/Professional/Research/Prof_Parinaz_Naghizadeh/Code/Shortest_Path_Interdiction/tests/unit/models/"

pytest.main([
    unit_test_models + "test_grid.py"
])
pass

