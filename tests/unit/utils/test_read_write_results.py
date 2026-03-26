from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from dflintdpy.utils.read_write_results import load_results_from_csv, save_results_to_csv


DATA_KEYS = [
    "o_o",
    "o_p",
    "o_s",
    "o_r",
    "o_mr",
    "o_ma",
    "o_m",
    "o_a",
    "s_o",
    "s_p",
    "s_s",
    "s_r",
    "s_mr",
    "s_ma",
    "s_m",
    "s_a",
    "a_o",
    "a_p",
    "a_s",
    "a_r",
    "a_mr",
    "a_ma",
    "a_m",
    "a_a",
    "a_p_o",
    "a_s_o",
    "a_r_o",
    "a_a_o",
]


def _make_result(sim_offset: float, size: int = 3) -> dict:
    all_data = {}
    for idx, key in enumerate(DATA_KEYS):
        base = sim_offset + idx
        all_data[key] = np.array([base + i * 0.1 for i in range(size)], dtype=np.float64)
    return {"all_data": all_data}


def test_save_and_load_results_round_trip(tmp_path: Path):
    results = [_make_result(0.0), _make_result(10.0)]
    output_path = tmp_path / "results.csv"

    save_results_to_csv(results, output_path)
    assert output_path.exists()

    loaded = load_results_from_csv(output_path)
    assert len(loaded) == 2
    np.testing.assert_allclose(loaded[1]["s_a"], results[1]["all_data"]["s_a"])


def test_load_results_fills_missing_columns_with_nan(tmp_path: Path):
    output_path = tmp_path / "results_missing_cols.csv"
    minimal_results = [{"all_data": {"o_o": np.array([1.0, 2.0, 3.0])}}]

    save_results_to_csv(minimal_results, output_path)
    loaded = load_results_from_csv(output_path)

    assert np.isnan(loaded[0]["a_a"]).all()
