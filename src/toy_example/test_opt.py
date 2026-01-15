
from toy_example.opt import ToyOptModel
import numpy as np

def test_init():
    cost = np.array([1, 1, 2, 2])
    opt_model = ToyOptModel(cost)
    assert np.array_equal(opt_model.cost, cost), \
        f"Initialization failed. Expected cost {cost}, but got {opt_model.cost}"
    pass

def test_setObj():
    cost = np.array([1, 1, 2, 2])
    opt_model = ToyOptModel(cost)
    new_cost = np.array([2, 2, 1, 1])
    opt_model.setObj(new_cost)
    assert np.array_equal(opt_model.cost, new_cost), \
        f"setObj failed. Expected cost {new_cost}, but got {opt_model.cost}"
    pass

def test_solve():
    cost = np.array([1, 1, 2, 2])
    opt_model = ToyOptModel(cost)
    y, obj_val = opt_model.solve()
    expected_y = np.array([1, 1, 0, 0])
    expected_obj_val = 2
    assert np.array_equal(y, expected_y), \
        f"solve failed. Expected solution {expected_y}, but got {y}"
    assert obj_val == expected_obj_val, \
        f"solve failed. Expected objective value {expected_obj_val}, but got {obj_val}"
    pass

def run_tests():
    test_init()
    test_setObj()
    test_solve()
    print("All tests passed.")
    pass

if __name__ == "__main__":
    run_tests()