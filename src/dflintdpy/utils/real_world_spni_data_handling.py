import csv
import pandas as pd
import numpy as np
from pathlib import Path
from dflintdpy.models.graph import Graph

def reformat_to_csv(input_file, output_file, n_cols):
    """
    Reformat a text file into CSV format.
    First n_cols lines are column headers.
    Remaining lines are data values, where every n_cols consecutive values form one row.
    """
    with open(input_file, 'r') as f:
        lines = [line.strip().strip("'") for line in f.readlines() if line.strip()]
    
    # First n_cols lines are the column headers
    headers = lines[:n_cols]
    
    # Remaining lines are data values
    data_values = lines[n_cols:]
    
    # Group every n_cols values into a row
    rows = []
    for i in range(0, len(data_values), n_cols):
        if i + n_cols - 1 < len(data_values):
            row = [data_values[i + j] for j in range(n_cols)]
            rows.append(row)
    
    # Write to CSV file
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header row
        writer.writerow(headers)
        
        # Write data rows
        writer.writerows(rows)
    
    print(f"Successfully created {output_file} with {len(rows)} data rows")


def csv_to_graph(file_path: str):
    data = pd.read_csv(file_path)
    arcs = [(data['From'].values[i], data['To'].values[i]) 
            for i in range(data.shape[0])]
    graph =  Graph(arcs)
    return graph


def test_csv_to_graph():
    # Find path to files
    file_path = Path(__file__).parent.parent.parent.parent / 'real_world_spni_data' / 'county_level_arcs.csv'
    # Run function
    csv_to_graph(file_path = file_path)


def convert_txt_to_csv():
    # Find path to files
    path_dir = Path(__file__).parent.parent.parent.parent / 'real_world_spni_data'

    # County-level nodes
    input_file = path_dir / 'county_level_nodes.txt'
    output_file = path_dir / 'county_level_nodes.csv'
    reformat_to_csv(input_file, output_file, n_cols=4)

    # County-level arcs
    input_file = path_dir / 'county_level_arcs.txt'
    output_file = path_dir / 'county_level_arcs.csv'
    reformat_to_csv(input_file, output_file, n_cols=6)

    # Town-level nodes
    input_file = path_dir / 'town_level_nodes.txt'
    output_file = path_dir / 'town_level_nodes.csv'
    reformat_to_csv(input_file, output_file, n_cols=4)

    # County-level arcs
    input_file = path_dir / 'town_level_arcs.txt'
    output_file = path_dir / 'town_level_arcs.csv'
    reformat_to_csv(input_file, output_file, n_cols=6)


def main():
    test_csv_to_graph()


if __name__ == '__main__':
    main()