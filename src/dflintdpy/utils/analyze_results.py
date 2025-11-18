# analyze results
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

def parse_filename(filename):
    """Extract parameters from filename."""
    pattern = r'results_BPPO_N_(\d+)_noise_([\d.]+)_deg_(\d+)\.csv'
    match = re.match(pattern, filename)
    
    if match:
        return {
            'N': int(match.group(1)),
            'noise': float(match.group(2)),
            'deg': int(match.group(3))
        }
    return None

def scan_available_data(directory='.'):
    """Scan directory for available data combinations."""
    files = [f for f in os.listdir(directory) if f.startswith('results_BPPO_') and f.endswith('.csv')]
    
    combinations = defaultdict(list)
    
    for filename in files:
        params = parse_filename(filename)
        if params:
            key = (
                params['N'], 
                params['noise'],
                params['deg']
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

    for (N, noise, deg), file_info in available.items():
        # if degrees is not None and deg not in degrees:
        #     continue

        # if noise is not None and noise not in noise_values:
        #     continue
        
        # Load the first file for this combination (assuming one file per combination)
        filepath = os.path.join(directory, file_info[0]['filename'])
        df = pd.read_csv(filepath)
        loaded_data[(N, noise, deg)] = df

        print(f"Loaded: N={N}, noise={noise}, deg={deg} from {file_info[0]['filename']}")

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
    
    # No intervention (np_*)
    calculations['no_intd_pfl'] = (all_data['np_pfl'] - all_data['np_t']) / all_data['np_t'] * 100 # all_data['o_p'] # 
    calculations['no_intd_dfl'] = (all_data['np_dfl'] - all_data['np_t']) / all_data['np_t'] * 100 # all_data['o_s'] #
    calculations['no_intd_adfl'] = (all_data['np_adfl'] - all_data['np_t']) / all_data['np_t'] * 100 # all_data['o_a'] #

    # Symmetric intervention (s_*)
    calculations['sym_intd_pfl'] = (all_data['p_pfl'] - all_data['p_t']) / all_data['p_t'] * 100 # all_data['s_p'] #
    calculations['sym_intd_dfl'] = (all_data['p_dfl'] - all_data['p_t']) / all_data['p_t'] * 100 # all_data['s_s'] #
    calculations['sym_intd_adfl'] = (all_data['p_adfl'] - all_data['p_t']) / all_data['p_t'] * 100 # all_data['s_a'] #

    # # Asymmetric intervention (a_*)
    # calculations['asym_intd_p'] = (all_data['a_p'] - all_data['a_o']) / all_data['a_o'] * 100 # all_data['a_p'] #
    # calculations['asym_intd_s'] = (all_data['a_s'] - all_data['a_o']) / all_data['a_o'] * 100 # all_data['a_s'] #
    # calculations['asym_intd_a'] = (all_data['a_a'] - all_data['a_o']) / all_data['a_o'] * 100 # all_data['a_a'] #

    # Prepare data for boxplot
    data_to_plot = [
        # No intervention
        calculations['no_intd_pfl'], calculations['no_intd_dfl'], calculations['no_intd_adfl'],
        # Symmetric intervention
        calculations['sym_intd_pfl'], calculations['sym_intd_dfl'], calculations['sym_intd_adfl'],
        # # Asymmetric intervention
        # calculations['asym_intd_p'], calculations['asym_intd_s'], calculations['asym_intd_a']
    ]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create boxplots
    positions = [1, 2, 3, 5, 6, 7]
    bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True,
                     showfliers=False, flierprops=dict(marker='o', markersize=3, alpha=0.5))
    
    # Color the boxes
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'] * 3
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Set x-axis labels
    ax.set_xticks([2, 6])
    ax.set_xlim([0, 8])
    ax.set_xticklabels(['no pricing', 'pricing'], fontsize=12)

    # Set y-axis label
    ax.set_ylabel('Percentage profit decrease vs. oracle [%]', fontsize=13)

    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', alpha=0.7, label='PFL vs Oracle'),
        Patch(facecolor='#4ECDC4', alpha=0.7, label='DFL vs Oracle'),
        Patch(facecolor='#45B7D1', alpha=0.7, label='ADFL vs Oracle')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11)
    
    # Add vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # Set title
    total_samples = len(all_data)
    title = f'Profit Decrease Comparison'#\n(n={total_samples} samples across '
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

    for (N, noise, deg), files in sorted(combinations.items()):
        print(f"{N:<10} {noise:<15} {deg:<10} {len(files)}")

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
        degrees=[7, 8],
        noise_values=[0.5]
    )
    
    if not loaded_data:
        print("No data loaded! Please check your filters.")
        sys.exit(1)
    
    # Step 3: Create boxplots
    print("\nCreating boxplots...")
    for keys, df in loaded_data.items():
        fig = create_boxplots(df, save_path='cost_comparison_boxplots.png')
        fig.savefig(
            figure_directory / "boxplot_N_{N}_noise_{noise}_deg_{deg}.png"\
                .format(N=keys[0], noise=keys[1], deg=keys[2]),
            dpi=300, 
            bbox_inches="tight")
    
    print("\nAnalysis complete!")
    print(f"Total combinations analyzed: {len(loaded_data)}")
