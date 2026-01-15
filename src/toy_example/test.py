import numpy as np
import torch

from toy_example.main import (
    feature_cost_mapping,
    optimizer,
    interdictor,
)

def test_intd_pipeline():
    w = -1
    c = feature_cost_mapping(w)
    assert c == [2, 2, 1, 1], \
        f"Mapping failed for w={w}. Expected [2, 2, 1, 1], but got {c}"
    y = optimizer(c)
    assert y == [0, 0, 1, 1], \
        f"Optimizer failed for c={c}. Expected [0, 0, 1, 1], but got {y}"
    g_c = interdictor(c)
    assert g_c == [0, 0, 0, 3], \
        f"Interdictor failed for c={c}. Expected [0, 0, 0, 3], but got {g_c}"
    c_intd = [c[i] + g_c[i] for i in range(len(c))]
    assert c_intd == [2, 2, 1, 4], \
        f"Optimizer failed for c_intd={c_intd}. " \
            f"Expected [2, 2, 1, 4], but got {c_intd}."
    y_intd = optimizer(c_intd)
    assert y_intd == [1, 1, 0, 0], \
        f"Interdictor failed for c_intd={c_intd}. " \
            f"Expected [1, 1, 0, 0], but got {y_intd}"
    pass

def test_feature_cost_mapping_scalar():
    w = 2
    c = feature_cost_mapping(w)
    assert c == [2, 2, 4, 4], \
        f"Mapping failed for w={w}. Expected [2, 2, 4, 4], but got {c}"
    w = -3
    c = feature_cost_mapping(w)
    assert c == [6, 6, 3, 3], \
        f"Mapping failed for w={w}. Expected [6, 6, 3, 3], but got {c}"
    pass

def test_feature_cost_mapping_tensor_vector():
    w = torch.tensor([1.0, -2.0, 0.5])
    c = feature_cost_mapping(w)
    expected = torch.tensor([
        [1.0, 1.0, 2.0, 2.0],
        [4.0, 4.0, 2.0, 2.0],
        [0.5, 0.5, 1.0, 1.0],
    ])
    assert torch.allclose(c, expected), \
        f"Tensor mapping failed. Expected {expected}, but got {c}"
    pass

def test_feature_cost_mapping_tensor_column_vector():
    w = torch.tensor([1.0, -2.0, 0.5]).unsqueeze(-1)
    c = feature_cost_mapping(w)
    expected = torch.tensor([
        [1.0, 1.0, 2.0, 2.0],
        [4.0, 4.0, 2.0, 2.0],
        [0.5, 0.5, 1.0, 1.0],
    ])
    assert torch.allclose(c, expected), \
        f"Tensor column mapping failed. Expected {expected}, but got {c}"
    pass

def test_feature_cost_mapping_numpy_vector():
    w = np.array([1.0, -2.0, 0.5])
    c = feature_cost_mapping(w)
    expected = np.array([
        [1.0, 1.0, 2.0, 2.0],
        [4.0, 4.0, 2.0, 2.0],
        [0.5, 0.5, 1.0, 1.0],
    ])
    assert np.allclose(c, expected), \
        f"Numpy mapping failed. Expected {expected}, but got {c}"
    pass

def test_feature_cost_mapping_numpy_column_vector():
    w = np.array([1.0, -2.0, 0.5]).reshape(-1, 1)
    c = feature_cost_mapping(w)
    expected = np.array([
        [1.0, 1.0, 2.0, 2.0],
        [4.0, 4.0, 2.0, 2.0],
        [0.5, 0.5, 1.0, 1.0],
    ])
    assert np.allclose(c, expected), \
        f"Numpy column mapping failed. Expected {expected}, but got {c}"
    pass

def test_optimizer_tensor_costs():
    # case: w = torch.tensor([1.0, -2.0, 0.5])
    c = torch.tensor([
        [1.0, 1.0, 2.0, 2.0],
        [4.0, 4.0, 2.0, 2.0],
        [0.5, 0.5, 1.0, 1.0],
    ])
    y = optimizer(c)
    expected = torch.tensor([
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
    ])
    assert torch.allclose(y, expected), \
        f"Tensor optimizer failed. Expected {expected}, but got {y}"
    pass

def test_optimizer_numpy_costs():
    # case: w = np.array([1.0, -2.0, 0.5])
    c = np.array([
        [1.0, 1.0, 2.0, 2.0],
        [4.0, 4.0, 2.0, 2.0],
        [0.5, 0.5, 1.0, 1.0],
    ])
    y = optimizer(c)
    expected = np.array([
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
    ])
    assert np.allclose(y, expected), \
        f"Numpy optimizer failed. Expected {expected}, but got {y}"

def test_interdictor_tensor_costs():
    # case: w = torch.tensor([1.0, -2.0, 0.5])
    c = torch.tensor([
        [1.0, 1.0, 2.0, 2.0],
        [4.0, 4.0, 2.0, 2.0],
        [0.5, 0.5, 1.0, 1.0],
    ])
    g_c = interdictor(c)
    expected = torch.tensor([
        [0.0, 3.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 3.0],
        [0.0, 3.0, 0.0, 0.0],
    ])
    assert torch.allclose(g_c, expected), \
        f"Tensor interdictor failed. Expected {expected}, but got {g_c}"

def test_interdictor_numpy_costs():
    # case: w = np.array([1.0, -2.0, 0.5])
    c = np.array([
        [1.0, 1.0, 2.0, 2.0],
        [4.0, 4.0, 2.0, 2.0],
        [0.5, 0.5, 1.0, 1.0],
    ])
    g_c = interdictor(c)
    expected = np.array([
        [0.0, 3.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 3.0],
        [0.0, 3.0, 0.0, 0.0],
    ])
    assert np.allclose(g_c, expected), \
        f"Numpy interdictor failed. Expected {expected}, but got {g_c}"
    pass

def run_tests():
    test_feature_cost_mapping_scalar()
    test_feature_cost_mapping_tensor_vector()
    test_feature_cost_mapping_tensor_column_vector()
    test_feature_cost_mapping_numpy_vector()
    test_feature_cost_mapping_numpy_column_vector()
    test_optimizer_tensor_costs()
    test_optimizer_numpy_costs()
    test_interdictor_tensor_costs()
    test_interdictor_numpy_costs()
    test_intd_pipeline()

if __name__ == "__main__":
    run_tests()
    print("All tests passed.")
