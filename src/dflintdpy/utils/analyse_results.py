import os
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections.abc import Mapping, Sequence
from collections import defaultdict
from dflintdpy.simulation.spni.results import (
    aggregate_sweep_results,
    filter_all_data_for_comparable_asymmetric_plots,
    to_legacy_all_data,
)
from dflintdpy.simulation.spni.types import SimulationResult, SweepResult
from dflintdpy.utils.read_write_results import load_results_from_csv


_PERCENTAGE_BASELINES = {
    'no_intd_p': ('o_p', 'o_o'),
    'no_intd_s': ('o_s', 'o_o'),
    'no_intd_r': ('o_r', 'o_o'),
    'no_intd_a': ('o_a', 'o_o'),
    'sym_intd_p': ('s_p', 's_o'),
    'sym_intd_s': ('s_s', 's_o'),
    'sym_intd_r': ('s_r', 's_o'),
    'sym_intd_a': ('s_a', 's_o'),
    # Asymmetric percentages must use the oracle follower cost under the same
    # predictor-specific interdiction, not the separate oracle-vs-oracle run.
    'asym_intd_p': ('a_p', 'a_p_o'),
    'asym_intd_s': ('a_s', 'a_s_o'),
    'asym_intd_r': ('a_r', 'a_r_o'),
    'asym_intd_a': ('a_a', 'a_a_o'),
}


def _coerce_legacy_simulations(
    simulations: SweepResult | Sequence[SimulationResult] | Sequence[Mapping],
):
    """Normalize typed or legacy simulation payloads into legacy all-data dicts."""
    if isinstance(simulations, SweepResult):
        return [to_legacy_all_data(result.summary_bundle)
                for result in simulations.results]

    simulation_list = list(simulations)
    if not simulation_list:
        return []

    first = simulation_list[0]
    if isinstance(first, SimulationResult):
        return [
            to_legacy_all_data(result.summary_bundle)
            for result in simulation_list
        ]
    if isinstance(first, Mapping) and "all_data" in first:
        return [simulation["all_data"] for simulation in simulation_list]
    return simulation_list


def _typed_aggregate_summary(
    simulations: SweepResult | Sequence[SimulationResult] | Sequence[Mapping],
):
    """Return a typed aggregate summary when simulations come from the pipeline."""
    if isinstance(simulations, SweepResult):
        return aggregate_sweep_results(simulations.results)

    simulation_list = list(simulations)
    if simulation_list and isinstance(simulation_list[0], SimulationResult):
        return aggregate_sweep_results(simulation_list)
    return None

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
                params['valid'],
                params['test'],
                (params['m'], params['n']),
                params['deg'],
                params['noise'],
                params['num_seeds']
            )
            combinations[key].append({
                'filename': filename,
                'params': params
            })
    
    return dict(combinations)

def load_data(
        directory='.', 
        train_values=None, 
        valid_values=None,
        test_values=None,
        mn_values=None, 
        degrees=None,
        noise_values=None,
        num_seeds_values=None
    ):
    """
    Load data for specified combinations.
    
    Parameters:
    -----------
    directory : str
        Directory containing the CSV files
    train_values : list of int or None
        List of train values to load. If None, load all.
    valid_values : list of int or None
        List of valid values to load. If None, load all.
    test_values : list of int or None
        List of test values to load. If None, load all.
    mn_values : list of tuples or None
        List of (m, n) tuples to load. If None, load all.
    degrees : list of int or None
        List of degree values to load. If None, load all.
    noise_values : list of float or None
        List of noise values to load. If None, load all.
    num_seeds_values : list of int or None
        List of seed counts to load. If None, load all.
    
    Returns:
    --------
    dict : Dictionary with keys
        (train, valid, test, (m,n), deg, noise, num_seeds) and values as
        a list of simulations (each entry is a dict of arrays)
    """
    available = scan_available_data(directory)
    
    loaded_data = {}
    
    for (train, valid, test, mn, deg, noise, num_seeds), file_info in available.items():
        # Filter by train values if specified
        if train_values is not None and train not in train_values:
            continue

        if valid_values is not None and valid not in valid_values:
            continue

        if test_values is not None and test not in test_values:
            continue

        if degrees is not None and deg not in degrees:
            continue

        if noise_values is not None and noise not in noise_values:
            continue

        if num_seeds_values is not None and num_seeds not in num_seeds_values:
            continue
        
        # Filter by (m,n) values if specified
        if mn_values is not None and mn not in mn_values:
            continue
        
        # Load the first file for this combination (assuming one file per combination)
        filepath = os.path.join(directory, file_info[0]['filename'])
        simulations = load_results_from_csv(filepath)
        loaded_data[(train, valid, test, mn, deg, noise, num_seeds)] = simulations
    
    return loaded_data

def combine_simulations(simulations):
    """Combine per-simulation arrays into a single all-data dictionary."""
    typed_summary = _typed_aggregate_summary(simulations)
    if typed_summary is not None:
        return typed_summary["all_data"]

    simulations = _coerce_legacy_simulations(simulations)
    if not simulations:
        return {}

    combined = {}
    for key in simulations[0].keys():
        combined[key] = np.concatenate([sim[key] for sim in simulations])
    return combined


def _safe_percentage(num, denom):
    """Compute percentage safely and replace NaN/Inf with zeros."""
    with np.errstate(divide='ignore', invalid='ignore'):
        out = (num - denom) / denom * 100
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_percentage_sum(num, denom):
    """Safe percentage increase for scalar aggregated values."""
    denom_sum = np.sum(denom)
    if denom_sum == 0:
        return 0.0
    num_sum = np.sum(num - denom)
    return float(np.nan_to_num(num_sum / denom_sum * 100, nan=0.0, posinf=0.0, neginf=0.0))


def compute_percentage_increases_from_samples(all_data):
    """Compute per-sample percentage increases for boxplots."""
    return {
        key: _safe_percentage(all_data[num_key], all_data[denom_key])
        for key, (num_key, denom_key) in _PERCENTAGE_BASELINES.items()
    }

def compute_percentage_increases_from_simulations(simulations):
    """Compute per-simulation percentage increases aggregated by sums."""
    typed_summary = _typed_aggregate_summary(simulations)
    if typed_summary is not None:
        return typed_summary["percentage_increases"]["simulations"]

    simulations = _coerce_legacy_simulations(simulations)
    simulations, _ = filter_all_data_for_comparable_asymmetric_plots(
        simulations
    )
    calculations = defaultdict(list)

    for sim_data in simulations:
        for key, (num_key, denom_key) in _PERCENTAGE_BASELINES.items():
            calculations[key].append(
                _safe_percentage_sum(sim_data[num_key], sim_data[denom_key])
            )

    return dict(calculations)

def create_boxplots_from_calculations(calculations, save_path=None):
    """Create boxplots from precomputed calculations."""
    data_to_plot = [
        calculations['no_intd_p'], calculations['no_intd_s'], calculations['no_intd_r'], calculations['no_intd_a'],
        calculations['sym_intd_p'], calculations['sym_intd_s'], calculations['sym_intd_r'], calculations['sym_intd_a'],
        calculations['asym_intd_p'], calculations['asym_intd_s'], calculations['asym_intd_r'], calculations['asym_intd_a']
    ]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    positions = [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14]
    bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True,
                     showfliers=False, flierprops=dict(marker='o', markersize=3, alpha=0.5))
    
    colors = ['#FF6B6B', '#4ECDC4', '#FFA552', '#45B7D1'] * 3
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xticks([2.5, 7.5, 12.5])
    ax.set_xlim(0, 15)
    ax.set_xticklabels(['no intd', 'sym intd', 'asym intd'], fontsize=20)

    ax.set_ylabel('Percentage cost increase vs. oracle (%)', fontsize=22)

    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', alpha=0.7, label='PFL'),
        Patch(facecolor='#4ECDC4', alpha=0.7, label='DFL'),
        Patch(facecolor='#FFA552', alpha=0.7, label='R-DFL'),
        Patch(facecolor='#45B7D1', alpha=0.7, label='A-DFL')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=20)
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    title = 'Cost Increase Comparison'
    ax.set_title(title, fontsize=24, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    return fig

def create_boxplots(simulations, save_path=None):
    """
    Create boxplots showing percentage cost increase.
    
    Parameters:
    -----------
    simulations : list
        List of simulation data dictionaries
    save_path : str or None
        Path to save the figure. If None, display instead.
    """
    typed_summary = _typed_aggregate_summary(simulations)
    if typed_summary is not None:
        calculations = typed_summary["percentage_increases"]["samples"]
    else:
        plot_simulations, _ = filter_all_data_for_comparable_asymmetric_plots(
            _coerce_legacy_simulations(simulations)
        )
        all_data = combine_simulations(plot_simulations)
        calculations = (
            compute_percentage_increases_from_samples(all_data)
            if all_data else {
                key: np.asarray([], dtype=float)
                for key in _PERCENTAGE_BASELINES
            }
        )
    return create_boxplots_from_calculations(calculations, save_path=save_path)

def create_boxplots_by_simulation(simulations, save_path=None):
    """
    Create boxplots with one value per simulation (aggregated by sums).
    """
    calculations = compute_percentage_increases_from_simulations(simulations)
    return create_boxplots_from_calculations(calculations, save_path=save_path)

def print_available_combinations(directory='.'):
    """Print available data combinations."""
    combinations = scan_available_data(directory)
    
    print(f"\nFound {len(combinations)} data combinations:\n")
    print(f"{'Train':<7} {'Valid':<7} {'Test':<7} {'(m, n)':<15} {'Deg':<5} {'Noise':<7} {'Seeds':<7} {'Files'}")
    print("-" * 70)

    for (train, valid, test, mn, deg, noise, num_seeds), files in sorted(combinations.items()):
        print(
            f"{train:<7} {valid:<7} {test:<7} {str(mn):<15} {deg:<5} {noise:<7} "
            f"{num_seeds:<7} {len(files)}"
        )

    print("\n")
    return combinations

def analyze_results():
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
    loaded_data = load_data(
        data_directory,
        noise_values=[0.5],
        train_values=[1000],
        valid_values=[250],
        num_seeds_values=[5]
    )
    
    if not loaded_data:
        print("No data loaded! Please check your filters.")
        sys.exit(1)
    
    # Step 3: Create boxplots
    print("\nCreating boxplots...")
    for keys, simulations in loaded_data.items():
        fig_samples = create_boxplots(simulations, save_path='cost_comparison_boxplots.png')
        fig_samples.savefig(
            figure_directory / (
                "boxplot_train_{train}_valid_{valid}_test_{test}_m_{m}_n_{n}_deg_{deg}"
                "_noise_{noise}_seeds_{num_seeds}.png"
            ).format(
                train=keys[0],
                valid=keys[1],
                test=keys[2],
                m=keys[3][0],
                n=keys[3][1],
                deg=keys[4],
                noise=keys[5],
                num_seeds=keys[6]
            ),
            dpi=300, 
            bbox_inches="tight")

        fig_sims = create_boxplots_by_simulation(
            simulations,
            save_path='cost_comparison_boxplots_sims.png'
        )
        fig_sims.savefig(
            figure_directory / (
                "boxplot_sims_train_{train}_valid_{valid}_test_{test}_m_{m}_n_{n}_deg_{deg}"
                "_noise_{noise}_seeds_{num_seeds}.png"
            ).format(
                train=keys[0],
                valid=keys[1],
                test=keys[2],
                m=keys[3][0],
                n=keys[3][1],
                deg=keys[4],
                noise=keys[5],
                num_seeds=keys[6]
            ),
            dpi=300, 
            bbox_inches="tight")
    print(f"Plots saved to: {figure_directory}")
    
    print("\nAnalysis complete!")
    print(f"Total combinations analyzed: {len(loaded_data)}")

# Main execution
if __name__ == "__main__":
    analyze_results()
