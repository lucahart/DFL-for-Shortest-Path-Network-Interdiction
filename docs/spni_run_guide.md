# SPNI Run Guide

This guide documents the canonical command-line path for running SPNI
simulations from the repository root.

## Base Command

Most SPNI runs start with this shape:

```bash
source ./.venv/bin/activate

MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp \
python -m dflintdpy.simulation.spni.pipeline \
  --mode <mode>
```

The `MPLCONFIGDIR` and `XDG_CACHE_HOME` assignments keep Matplotlib and other
libraries from writing cache files into locations that may not be writable.

If you want to see the full CLI help:

```bash
source ./.venv/bin/activate

MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp \
python -m dflintdpy.simulation.spni.pipeline --help
```

## Modes

The `--mode` flag selects the top-level action.

- `single`: run one full SPNI simulation.
- `seed_sweep`: run several simulations with different seeds, then aggregate
  results. This is the default mode if `--mode` is omitted.
- `scenario_sweep`: run seed sweeps for multiple values of `num_scenarios`,
  then save scenario-comparison figures.
- `replot`: regenerate boxplots from an already saved result CSV without
  rerunning training or evaluation.

Examples:

```bash
MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp \
python -m dflintdpy.simulation.spni.pipeline \
  --mode single
```

```bash
MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp \
python -m dflintdpy.simulation.spni.pipeline \
  --mode seed_sweep \
  --num-seeds 5
```

```bash
MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp \
python -m dflintdpy.simulation.spni.pipeline \
  --mode scenario_sweep \
  --scenarios 2,3,5 \
  --num-seeds 3
```

```bash
MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp \
python -m dflintdpy.simulation.spni.pipeline \
  --mode replot \
  --input-path results/results_train_500_valid_25_test_250_m_5_n_5_deg_16_noise_0.5_seeds_2.csv
```

## Other CLI Options

These are the main flags outside of `--set`.

- `--num-seeds N`: number of seeds for `seed_sweep` and `scenario_sweep`.
- `--scenarios A,B,C`: comma-separated scenario counts for `scenario_sweep`.
- `--input-path PATH [PATH ...]`: saved CSV file or files to read when
  `--mode replot`. You can pass several paths after one flag or repeat the
  flag.
- `--exclude-symmetric-interdictions`: with `--mode replot`, omit the
  symmetric-interdiction group from the regenerated boxplots. The output files
  use a `_no_sym` suffix.
- `--legend-location LOC`: place the result-boxplot legend with a Matplotlib
  location string such as `upper left`, `lower right`, or `best`.
- `--figure-directory PATH`: directory for generated figures. Learning curves
  are written under a `learning_curves/` subdirectory inside this path.
- `--present-results` / `--no-present-results`: enable or disable side effects
  such as saved CSVs, boxplots, learning curves, and scenario figures.
- `--compute-asym-intd` / `--no-compute-asym-intd`: enable or disable
  asymmetric interdiction evaluation.
- `--compute-wrong-asym-intd` / `--no-compute-wrong-asym-intd`: enable or
  disable wrong-model asymmetric evaluation.
- `--load-real-world-graph PATH`: load a real-world graph instead of using the
  synthetic grid.
- `--source-node NODE`: source node for real-world graph shortest-path solves.
- `--target-node NODE`: target node for real-world graph shortest-path solves.

Use hyphenated CLI flag names, for example `--num-seeds`, not
`--num_seeds`.

## Config Parameters

The pipeline starts from `dflintdpy.data.config.HP()` and lets you override
configuration fields from the command line with:

```bash
--set 'param=value'
```

Values are parsed with Python literal syntax. This means:

- tuples use Python tuple syntax: `--set 'grid_size=(5, 5)'`
- strings can be unquoted or quoted: `--set 'pred_model=linear'`
- booleans use Python spelling: `--set 'some_flag=True'`
- floats and integers are written normally: `--set 'spo_lr=0.001'`

Commonly adjusted parameters:

- Data size: `num_train_samples`, `num_val_samples`, `num_test_samples`
- Graph shape: `grid_size`, `deg`, `noise_width`
- Synthetic feature count: `num_features`
- Interdiction setup: `budget`, `num_scenarios`
- Training setup: `batch_size`, `po_epochs`, `spo_epochs`, `po_lr`, `spo_lr`,
  `pred_model`
- Solver tolerances: `benders_max_count`, `benders_eps`, `lsd`
- Seeds: `seed`, `random_seed`, `intd_seed`, `loader_seed`,
  `seed_sweep_offset`
- Learning-rate stop rule: `max_lr_reductions`
- SPO+ safety knobs: `surrogate_underprediction_penalty_weight`,
  `surrogate_underprediction_margin`. The underprediction safeguard is off by
  default (`surrogate_underprediction_penalty_weight=0.0`). Opt in with, for
  example, `--set 'surrogate_underprediction_penalty_weight=1.0'`.

Small smoke run:

```bash
source ./.venv/bin/activate

MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp \
python -m dflintdpy.simulation.spni.pipeline \
  --mode seed_sweep \
  --num-seeds 2 \
  --set 'grid_size=(3, 3)' \
  --set 'num_train_samples=8' \
  --set 'num_val_samples=4' \
  --set 'num_test_samples=4' \
  --set 'num_scenarios=2' \
  --set 'po_epochs=1' \
  --set 'spo_epochs=1'
```

Normal synthetic grid seed sweep:

```bash
source ./.venv/bin/activate

MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp \
python -m dflintdpy.simulation.spni.pipeline \
  --mode seed_sweep \
  --num-seeds 5 \
  --set 'grid_size=(5, 5)' \
  --set 'deg=12' \
  --set 'num_train_samples=500' \
  --set 'num_val_samples=100' \
  --set 'num_test_samples=250' \
  --set 'num_scenarios=3' \
  --set 'budget=5'
```

Real-world graph structure with synthetic data:

```bash
source ./.venv/bin/activate

MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp \
python -m dflintdpy.simulation.spni.pipeline \
  --mode seed_sweep \
  --num-seeds 3 \
  --load-real-world-graph real_world_spni_data/town_level_arcs.csv \
  --source-node 1 \
  --target-node 10 \
  --set 'num_train_samples=100' \
  --set 'num_val_samples=25' \
  --set 'num_test_samples=100' \
  --set 'num_scenarios=3'
```

If `--load-real-world-graph` is omitted, the run uses the synthetic grid path.

## Outputs

For `seed_sweep` with presentation enabled:

- result CSVs are saved under `results/`
- sample-level and simulation-level boxplots are saved under `figures/`
- per-run learning curves are saved under `figures/learning_curves/`
- seed-comparison learning curves are saved under `figures/learning_curves/`
- real-world graph runs add a graph marker such as
  `_real_world_town_level_arcs` to default result and figure filenames

For `scenario_sweep` with presentation enabled:

- the outer scenario-comparison figures are saved under `figures/`
- inner seed sweeps run with `present_results=False`, so they do not each
  save their own CSV and figure bundle

For `replot`, regenerated boxplots are also saved under `figures/` by default.

Use `--figure-directory PATH` to place figures somewhere else.

## Replot Saved Results

If one or more seed sweeps have already saved CSVs, you can regenerate the
result boxplots without rerunning simulation, training, or evaluation. When
you provide several CSV files, the replot command concatenates their
simulation results before plotting.

CLI form:

```bash
source ./.venv/bin/activate

MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp \
python -m dflintdpy.simulation.spni.pipeline \
  --mode replot \
  --input-path results/results_train_500_valid_25_test_250_m_5_n_5_deg_16_noise_0.5_seeds_2.csv \
  --legend-location "upper left"
```

Multiple saved files:

```bash
source ./.venv/bin/activate

MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp \
python -m dflintdpy.simulation.spni.pipeline \
  --mode replot \
  --input-path \
    results/results_train_500_valid_100_test_250_m_5_n_5_deg_10_noise_0.5_seeds_3.csv \
    results/results_train_500_valid_100_test_250_m_5_n_5_deg_10_noise_0.5_seeds_2.csv \
  --legend-location "upper left"
```

To regenerate only the uninterdicted and asymmetric groups, add:

```bash
source ./.venv/bin/activate

MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp \
python -m dflintdpy.simulation.spni.pipeline \
  --mode replot \
  --input-path results/results_train_500_valid_25_test_250_m_5_n_5_deg_16_noise_0.5_seeds_2.csv \
  --exclude-symmetric-interdictions
```

Python helper form:

```python
from dflintdpy.simulation.spni import replot_saved_sweep_outputs

paths = replot_saved_sweep_outputs(
    [
        "results/results_train_500_valid_100_test_250_m_5_n_5_deg_10_noise_0.5_seeds_3.csv",
        "results/results_train_500_valid_100_test_250_m_5_n_5_deg_10_noise_0.5_seeds_2.csv",
    ],
    exclude_symmetric_interdictions=True,
    legend_location="upper left",
)
print(paths)
```

This direct figure compiler recreates these files in `figures/` unless you pass
`figure_directory=...`:

- `<figure_directory>/<csv_stem>_boxplot.png`
- `<figure_directory>/<csv_stem>_boxplot_sims.png`

For multiple compatible result filenames, the combined output stem uses the
shared parameter prefix and the total number of loaded simulations, for example
`results_train_..._seeds_5_combined_boxplot.png`.

When `exclude_symmetric_interdictions=True`, the files are instead named:

- `<figure_directory>/<csv_stem>_no_sym_boxplot.png`
- `<figure_directory>/<csv_stem>_no_sym_boxplot_sims.png`

It can only regenerate plots that are supported by the saved CSV. It cannot
recreate learning-curve figures from old CSVs because training logs are not
stored in the result CSV.

## Troubleshooting

If the module command is not found, install the package in editable mode:

```bash
source ./.venv/bin/activate
python -m pip install -e .
```

If Matplotlib warns that `~/.matplotlib` is not writable, keep using:

```bash
MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp
```

If you forget the available flags:

```bash
source ./.venv/bin/activate

MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp \
python -m dflintdpy.simulation.spni.pipeline --help
```
