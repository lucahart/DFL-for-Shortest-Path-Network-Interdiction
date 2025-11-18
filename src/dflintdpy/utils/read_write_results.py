# read_write_results.py
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any


def save_results_to_csv(summary: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save simulation results to a CSV file.
    
    Parameters:
    -----------
    results : List[Dict[str, Any]]
        List of result dictionaries, each containing 'other_data' and 'all_data'
    output_path : str
        Path where the CSV file will be saved
    """
    rows = []
    
    all_data = summary['all_data']
    
    # Process each of the 100 samples
    num_samples = len(all_data['np_t'])
    
    for sample_idx in range(num_samples):
        row = {
            'simulation_index': 1,
            'sample_index': sample_idx,
        }
        
        # Add all_data entries
        for key in all_data.keys():
            value = all_data[key][sample_idx]
            # Convert numpy types to native Python types
            if isinstance(value, np.ndarray):
                row[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                row[key] = value.item()
            else:
                row[key] = value
        
        rows.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved {num_samples} samples each to {output_path}")


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
    data_keys = ['o_o', 'o_p', 'o_s', 'o_a', 
                 's_o', 's_p', 's_s', 's_a',
                 'a_o', 'a_p', 'a_s', 'a_a']
    
    for sim_idx in sorted(df['simulation_index'].unique()):
        sim_data = df[df['simulation_index'] == sim_idx].sort_values('sample_index')
        
        # Reconstruct all_data
        all_data = {}
        for key in data_keys:
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
