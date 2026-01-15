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
    print("All tests passed.")

if __name__ == "__main__":
    test_intd_pipeline()