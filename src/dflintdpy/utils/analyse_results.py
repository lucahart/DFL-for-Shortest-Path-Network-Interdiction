import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

def parse_filename(filename):
    """Extract parameters from filename."""
    pattern = r'results_train_(\d+)_valid_(\d+)_test_(\d+)_m_(\d+)_n_(\d+)_deg_(\d+)_noise_([\d.]+)_seeds_(\d+)\.csv'
    match = re.match(pattern, filename)
    
    if match:
        return {
            'train': int(match.group(1)),
            'valid': int(match.group(2)),
            'test': int(match.group(3)),
            'm': int(match.group(4)),
            'n': int(match.group(5)),
            'deg': int(match.group(6)),
            'noise': float(match.group(7)),
            'num_seeds': int(match.group(8))
        }
    return None

def scan_available_data(directory='.'):
    """Scan directory for available data combinations."""
    files = [f for f in os.listdir(directory) if f.startswith('results_train_') and f.endswith('.csv')]
    
    combinations = defaultdict(list)
    
    for filename in files:
        params = parse_filename(filename)
        if params:
            key = (
                params['train'], 
                (params['m'], params['n']), 
                params['deg'],
                params['noise']
            )
            combinations[key].append({
                'filename': filename,
                'params': params
            })
    
    return dict(combinations)

def load_data(
        directory='.', 
        train_values=None, 
        mn_values=None, 
        degrees=None,
        noise_values=None
    ):
    """
    Load data for specified combinations.
    
    Parameters:
    -----------
    directory : str
        Directory containing the CSV files
    train_values : list of int or None
        List of train values to load. If None, load all.
    mn_values : list of tuples or None
        List of (m, n) tuples to load. If None, load all.
    
    Returns:
    --------
    dict : Dictionary with keys (train, (m,n)) and values as DataFrame
    """
    available = scan_available_data(directory)
    
    loaded_data = {}
    
    for (train, mn, deg, noise), file_info in available.items():
        # Filter by train values if specified
        if train_values is not None and train not in train_values:
            continue

        if degrees is not None and deg not in degrees:
            continue

        if noise is not None and noise not in noise_values:
            continue
        
        # Filter by (m,n) values if specified
        if mn_values is not None and mn not in mn_values:
            continue
        
        # Load the first file for this combination (assuming one file per combination)
        filepath = os.path.join(directory, file_info[0]['filename'])
        df = pd.read_csv(filepath)
        loaded_data[(train, mn)] = df
        
        print(f"Loaded: train={train}, (m,n)={mn} from {file_info[0]['filename']}")
    
    return loaded_data

def create_boxplots(all_data, save_path=None):
    """
    Create boxplots showing percentage cost increase.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
    save_path : str or None
        Path to save the figure. If None, display instead.
    """
    
    # Calculate percentage increases
    calculations = {}
    
    # No intervention (o_*)
    calculations['no_intd_p'] = all_data['o_p'] # (all_data['o_p'] - all_data['o_o']) / all_data['o_o'] * 100
    calculations['no_intd_s'] = all_data['o_s'] # (all_data['o_s'] - all_data['o_o']) / all_data['o_o'] * 100
    calculations['no_intd_a'] = all_data['o_a'] # (all_data['o_a'] - all_data['o_o']) / all_data['o_o'] * 100

    # Symmetric intervention (s_*)
    calculations['sym_intd_p'] = all_data['s_p'] # (all_data['s_p'] - all_data['s_o']) / all_data['s_o'] * 100
    calculations['sym_intd_s'] = all_data['s_s'] # (all_data['s_s'] - all_data['s_o']) / all_data['s_o'] * 100
    calculations['sym_intd_a'] = all_data['s_a'] # (all_data['s_a'] - all_data['s_o']) / all_data['s_o'] * 100

    # Asymmetric intervention (a_*)
    calculations['asym_intd_p'] = all_data['a_p'] # (all_data['a_p'] - all_data['a_o']) / all_data['a_o'] * 100
    calculations['asym_intd_s'] = all_data['a_s'] # (all_data['a_s'] - all_data['a_o']) / all_data['a_o'] * 100
    calculations['asym_intd_a'] = all_data['a_a'] # (all_data['a_a'] - all_data['a_o']) / all_data['a_o'] * 100

    # Prepare data for boxplot
    data_to_plot = [
        # No intervention
        calculations['no_intd_p'], calculations['no_intd_s'], calculations['no_intd_a'],
        # Symmetric intervention
        calculations['sym_intd_p'], calculations['sym_intd_s'], calculations['sym_intd_a'],
        # Asymmetric intervention
        calculations['asym_intd_p'], calculations['asym_intd_s'], calculations['asym_intd_a']
    ]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create boxplots
    positions = [1, 2, 3, 5, 6, 7, 9, 10, 11]
    bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True,
                     showfliers=True, flierprops=dict(marker='o', markersize=3, alpha=0.5))
    
    # Color the boxes
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'] * 3
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Set x-axis labels
    ax.set_xticks([2, 6, 10])
    ax.set_xticklabels(['no intd', 'sym intd', 'asym intd'], fontsize=12)

    # Set y-axis label
    ax.set_ylabel('Percentage cost increase (%)', fontsize=13)

    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', alpha=0.7, label='x_p vs x_o'),
        Patch(facecolor='#4ECDC4', alpha=0.7, label='x_s vs x_o'),
        Patch(facecolor='#45B7D1', alpha=0.7, label='x_a vs x_o')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    # Add vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # Set title
    total_samples = len(all_data)
    title = f'Cost Increase Comparison'#\n(n={total_samples} samples across '
    # title += f'{len(all_data)} combinations of train and (m,n))'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    return fig

def print_available_combinations(directory='.'):
    """Print available data combinations."""
    combinations = scan_available_data(directory)
    
    print(f"\nFound {len(combinations)} data combinations:\n")
    print(f"{'Train':<10} {'(m, n)':<15} {'Files'}")
    print("-" * 40)

    for (train, mn, deg, noise), files in sorted(combinations.items()):
        print(f"{train:<10} {str(mn):<15} {len(files)}")

    print("\n")
    return combinations

# Main execution
if __name__ == "__main__":
    import sys
    
    # Directory containing the CSV files (default: current directory)
    parent_directory = Path(__file__).parent.parent.parent.parent
    data_directory = parent_directory / 'results'
    figure_directory = parent_directory / 'figures'
    
    print("=" * 50)
    print("Data Analysis Script")
    print("=" * 50)
    
    # Step 1: Scan and print available combinations
    combinations = print_available_combinations(data_directory)
    
    if not combinations:
        print("No data files found! Please check the directory.")
        sys.exit(1)
    
    # Step 2: Load all data (or specify filters)
    # To filter, you can specify:
    # loaded_data = load_data(data_directory, train_values=[100, 200], mn_values=[(5, 3), (10, 5)])
    loaded_data = load_data(
        data_directory, 
        degrees=[7],
        noise_values=[0.05]
    )
    
    if not loaded_data:
        print("No data loaded! Please check your filters.")
        sys.exit(1)
    
    # Step 3: Create boxplots
    print("\nCreating boxplots...")
    for keys, df in loaded_data.items():
        fig = create_boxplots(df, save_path='cost_comparison_boxplots.png')
        fig.savefig(
            figure_directory / "boxplot_train_{train}_graph__{m}_{n}.png"\
                .format(train=keys[0], m=keys[1][0], n=keys[1][1]),
            dpi=300, 
            bbox_inches="tight")
    
    print("\nAnalysis complete!")
    print(f"Total combinations analyzed: {len(loaded_data)}")