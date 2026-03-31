import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import List, Dict, Any

from dflintdpy.simulation.spni.results import (
    LEGACY_ALL_DATA_KEYS,
    WRONG_MODEL_ALL_DATA_KEYS,
    aggregate_sweep_results,
)
from dflintdpy.simulation.spni.types import SimulationResult, SweepResult


LEGACY_CSV_DATA_KEYS = tuple(
    dict.fromkeys((*LEGACY_ALL_DATA_KEYS, *WRONG_MODEL_ALL_DATA_KEYS))
)


def _to_python_value(value: Any) -> Any:
    """Convert NumPy values into CSV-safe native Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _rows_from_legacy_results(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Convert legacy result dictionaries into flat CSV row dictionaries."""
    rows: list[dict[str, Any]] = []
    for sim_idx, result in enumerate(results):
        all_data = result["all_data"]
        num_samples = len(all_data["o_o"])
        for sample_idx in range(num_samples):
            row = {
                "simulation_index": sim_idx,
                "sample_index": sample_idx,
            }
            for key, values in all_data.items():
                row[key] = _to_python_value(values[sample_idx])
            rows.append(row)
    return rows


def _coerce_results_rows(
    results: SweepResult | Sequence[SimulationResult] | Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Normalize typed or legacy results into flat CSV row dictionaries."""
    if isinstance(results, SweepResult):
        rows = aggregate_sweep_results(results.results)["rows"]
        return rows, len(results.results)

    result_list = list(results)
    if not result_list:
        return [], 0

    first = result_list[0]
    if isinstance(first, SimulationResult):
        rows = aggregate_sweep_results(result_list)["rows"]
        return rows, len(result_list)

    rows = _rows_from_legacy_results(result_list)
    return rows, len(result_list)


def save_results_to_csv(results: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save simulation results to a CSV file.
    
    Parameters:
    -----------
    results : List[Dict[str, Any]]
        List of result dictionaries, each containing 'other_data' and 'all_data'
    output_path : str
        Path where the CSV file will be saved
    """
    rows, num_results = _coerce_results_rows(results)

    # Create DataFrame and save
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    num_samples = 0 if df.empty else int(df["sample_index"].max()) + 1
    print(
        f"Saved {num_results} simulations with {num_samples} "
        f"samples each to {output_path}"
    )


def load_results_from_csv(input_path: str) -> List[Dict[str, Any]]:
    """
    Load simulation results from a CSV file.
    
    Parameters:
    -----------
    input_path : str
        Path to the CSV file
        
    Returns:
    --------
    List[Dict[str, Any]]
        List of result dictionaries with the original structure
    """
    df = pd.read_csv(input_path)
    
    # Group by simulation_index
    results = []
    dynamic_keys = [
        key for key in df.columns
        if key not in {"simulation_index", "sample_index"}
    ]
    data_keys = list(dict.fromkeys((*LEGACY_CSV_DATA_KEYS, *dynamic_keys)))
    
    for sim_idx in sorted(df['simulation_index'].unique()):
        sim_data = df[df['simulation_index'] == sim_idx].sort_values('sample_index')
        
        # Reconstruct all_data
        all_data = {}
        for key in data_keys:
            if key not in sim_data.columns:
                all_data[key] = np.full(len(sim_data), np.nan)
                continue
            values = sim_data[key].values
            # Try to parse JSON strings back to lists/arrays
            parsed_values = []
            for val in values:
                if isinstance(val, str):
                    try:
                        parsed_values.append(json.loads(val))
                    except json.JSONDecodeError:
                        parsed_values.append(val)
                else:
                    parsed_values.append(val)
            all_data[key] = np.array(parsed_values)

        results.append(all_data)

    print(f"Loaded {len(results)} simulations from {input_path}")
    return results
