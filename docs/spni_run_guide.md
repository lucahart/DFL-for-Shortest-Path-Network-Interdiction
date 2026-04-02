# SPNI Run Guide

This guide collects the terminal commands for the SPNI pipeline so you do not
have to reconstruct them from memory each time.

## Prerequisites

- Work from the repository root.
- Activate the project environment or call the venv Python directly.
- If you want `python -m dflintdpy...` commands to work, install the package in
  editable mode with `pip install -e .`.

Examples below use:

```bash
./.venv/bin/python
```

If your environment is already activated, you can replace that with `python`.

You can also activate the environment first and then run the module commands
in the shorter form:

```bash
source ./.venv/bin/activate
python -m dflintdpy.simulation.spni.pipeline --help
```

So, in general, these two styles are equivalent:

```bash
./.venv/bin/python -m dflintdpy.simulation.spni.pipeline
```

```bash
source ./.venv/bin/activate
python -m dflintdpy.simulation.spni.pipeline
```

If you are viewing this guide on GitHub, fenced code blocks already get a
built-in copy button in the rendered UI. This markdown file itself cannot add a
custom copy button without a separate docs site or frontend layer.

## Recommended default workflow

The simplest way to run SPNI is:

```bash
source ./.venv/bin/activate
python -m dflintdpy.simulation.spni.pipeline ...
```

By default, the typed pipeline starts from the repository config defaults in
`dflintdpy.data.config.HP()`. That means:

- you usually do not need to write Python one-liners
- you usually do not need to rebuild a config object manually
- you only add terminal flags when you want to override part of the default cfg

So the default mental model should be:

1. activate the venv
2. run `python -m dflintdpy.simulation.spni.pipeline`
3. add `--mode`, `--num-seeds`, or `--set ...` only when needed

## Main entrypoints

### Typed SPNI pipeline CLI

Use this for most runs:

```bash
source ./.venv/bin/activate
python -m dflintdpy.simulation.spni.pipeline
```

Supported modes:

- `--mode seed_sweep`: multi-seed SPNI sweep
- `--mode scenario_sweep`: scenario-count sweep over seed sweeps
- `--mode single`: one SPNI run

## Common CLI options

These apply to the typed CLI:

```bash
python -m dflintdpy.simulation.spni.pipeline [options]
```

Important flags:

- `--mode {single,seed_sweep,scenario_sweep}`: choose the top-level pipeline
  mode
- `--num-seeds N`: number of sweep seeds for `seed_sweep` or
  `scenario_sweep`
- `--scenarios A,B,C`: comma-separated scenario counts for `scenario_sweep`
- `--compute-asym-intd` or `--no-compute-asym-intd`: enable or disable
  asymmetric interdiction evaluation
- `--compute-wrong-asym-intd` or `--no-compute-wrong-asym-intd`: enable or
  disable wrong-model asymmetric evaluation
- `--present-results` or `--no-present-results`: save the sweep CSV and the two
  boxplots for seed sweeps, or save the scenario-sweep summary figures
- `--load-real-world-graph PATH`: run on a real graph CSV instead of a
  synthetic graph
- `--set KEY=VALUE`: override config fields before the run starts

Notes on `--set`:

- Values are parsed with Python literal syntax.
- Tuples should be written like `--set 'grid_size=(5, 5)'`.
- Strings can be written either as `linear` or quoted.
- Booleans should be written as `True` or `False`.

## Where outputs go

For seed sweeps with `present_results=True`:

- CSV results are written under `results/`
- boxplots are written under `figures/`

The returned diagnostics include:

- `legacy_output_path`
- `sample_boxplot_path`
- `simulation_boxplot_path`

For scenario sweeps with `present_results=True`:

- summary figures are written under `figures/`
- inner seed sweeps still run with `present_results=False`
- only the outer scenario sweep saves figures

The returned diagnostics include:

- `simulation_plot_path`
- `asym_simulation_plot_path`
- `sample_plot_path`
- `asym_sample_plot_path`

## Quick test runs

These are small smoke-style commands for fast checks.

### Default seed sweep from cfg

This uses the cfg defaults from `HP()`:

```bash
source ./.venv/bin/activate
python -m dflintdpy.simulation.spni.pipeline
```

### Default single run from cfg

```bash
source ./.venv/bin/activate
python -m dflintdpy.simulation.spni.pipeline --mode single
```

### Small seed sweep test run

```bash
source ./.venv/bin/activate
python -m dflintdpy.simulation.spni.pipeline \
  --num-seeds 2 \
  --no-present-results
```

### Small scenario sweep test run

```bash
source ./.venv/bin/activate
python -m dflintdpy.simulation.spni.pipeline \
  --mode scenario_sweep \
  --scenarios 2,3 \
  --num-seeds 2 \
  --present-results \
  --set 'grid_size=(3, 3)' \
  --set 'num_train_samples=8' \
  --set 'num_val_samples=4' \
  --set 'num_test_samples=4' \
  --set 'po_epochs=1' \
  --set 'spo_epochs=1'
```

### Small single-run test run

```bash
source ./.venv/bin/activate
python -m dflintdpy.simulation.spni.pipeline \
  --mode single
```

### Small custom smoke run with a few cfg overrides

Use this when the cfg defaults are too large and you want a short run without
dropping into Python.

```bash
source ./.venv/bin/activate
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

## Common terminal patterns

### Run with cfg defaults

```bash
source ./.venv/bin/activate
python -m dflintdpy.simulation.spni.pipeline
```

### Run one single simulation with cfg defaults

```bash
source ./.venv/bin/activate
python -m dflintdpy.simulation.spni.pipeline --mode single
```

### Run a seed sweep with a different number of seeds

```bash
source ./.venv/bin/activate
python -m dflintdpy.simulation.spni.pipeline --num-seeds 5
```

### Run a scenario sweep with explicit scenario counts

```bash
source ./.venv/bin/activate
python -m dflintdpy.simulation.spni.pipeline \
  --mode scenario_sweep \
  --scenarios 1,2,4,8 \
  --num-seeds 3
```

### Run without saving CSVs and boxplots

```bash
source ./.venv/bin/activate
python -m dflintdpy.simulation.spni.pipeline --no-present-results
```

### Run with a few cfg overrides

```bash
source ./.venv/bin/activate
python -m dflintdpy.simulation.spni.pipeline \
  --num-seeds 3 \
  --set 'grid_size=(5, 5)' \
  --set 'num_scenarios=3' \
  --set 'num_train_samples=100' \
  --set 'num_val_samples=25' \
  --set 'num_test_samples=25'
```

## Advanced Python-driven runs

Most users should prefer the terminal commands above.

Use the examples below only when you explicitly want to script against the
Python API.

### Seed sweep from `HP()`

```bash
python -c "
from dflintdpy.data.config import HP
from dflintdpy.simulation.spni.pipeline import run_seed_sweep

cfg = HP()
cfg.set('grid_size', (5, 5))
cfg.set('num_train_samples', 100)
cfg.set('num_val_samples', 25)
cfg.set('num_test_samples', 25)
cfg.set('num_scenarios', 3)
cfg.set('po_epochs', 5)
cfg.set('spo_epochs', 5)

result = run_seed_sweep(cfg, num_seeds=3, present_results=True)
print(result.diagnostics)
"
```

### Single run from `HP()`

```bash
python -c "
from dflintdpy.data.config import HP
from dflintdpy.simulation.spni.pipeline import run_single_simulation

cfg = HP()
cfg.set('grid_size', (5, 5))
cfg.set('num_train_samples', 100)
cfg.set('num_val_samples', 25)
cfg.set('num_test_samples', 25)
cfg.set('num_scenarios', 3)
cfg.set('po_epochs', 5)
cfg.set('spo_epochs', 5)

result = run_single_simulation(cfg)
print(result.diagnostics)
"
```

### Scenario sweep from `HP()`

```bash
python -c "
from dflintdpy.data.config import HP
from dflintdpy.simulation.spni.pipeline import run_scenario_sweep

cfg = HP()
cfg.set('grid_size', (5, 5))
cfg.set('num_train_samples', 100)
cfg.set('num_val_samples', 25)
cfg.set('num_test_samples', 25)
cfg.set('po_epochs', 5)
cfg.set('spo_epochs', 5)

result = run_scenario_sweep(cfg, scenarios=[1, 2, 4, 8], num_seeds=3)
print(result['diagnostics'])
"
```

Important:

- `run_scenario_sweep(...)` calls `run_seed_sweep(...)` internally with
  `present_results=False`
- this avoids generating a CSV and figures for every scenario count
- the outer scenario sweep still saves its summary figures by default

## Real-graph example

```bash
source ./.venv/bin/activate
python -m dflintdpy.simulation.spni.pipeline \
  --mode seed_sweep \
  --num-seeds 3 \
  --load-real-world-graph real_world_spni_data/my_graph.csv \
  --set 'num_train_samples=100' \
  --set 'num_val_samples=25' \
  --set 'num_test_samples=25' \
  --set 'num_scenarios=3'
```

## Recommended command patterns

For quick debugging:

```bash
source ./.venv/bin/activate
python -m dflintdpy.simulation.spni.pipeline \
  --mode seed_sweep \
  --num-seeds 2 \
  --no-present-results \
  --set 'grid_size=(3, 3)' \
  --set 'num_train_samples=8' \
  --set 'num_val_samples=4' \
  --set 'num_test_samples=4' \
  --set 'po_epochs=1' \
  --set 'spo_epochs=1'
```

For experiment runs you want saved:

```bash
source ./.venv/bin/activate
python -m dflintdpy.simulation.spni.pipeline \
  --mode seed_sweep \
  --num-seeds 5 \
  --present-results \
  --set 'grid_size=(5, 5)' \
  --set 'num_train_samples=1000' \
  --set 'num_val_samples=250' \
  --set 'num_test_samples=250' \
  --set 'num_scenarios=5'
```

## Troubleshooting

If a command fails immediately:

- confirm the venv exists: `ls .venv`
- confirm the package is installed: `python -m pip show dflintdpy`
- confirm imports work:

```bash
python -c "import dflintdpy; print('ok')"
```

If you forget the CLI flags:

```bash
python -m dflintdpy.simulation.spni.pipeline --help
```

## Legacy wrapper examples

These are last on purpose. Prefer the typed pipeline commands above unless you
specifically need the compatibility layer.

### Default legacy sweep

```bash
source ./.venv/bin/activate
python -m dflintdpy.scripts.Asym_SPNI_Simulator
```

### Legacy wrapper from a one-liner

```bash
python -c "
from dflintdpy.data.config import HP
from dflintdpy.scripts.Asym_SPNI_Simulator import run_sweep

cfg = HP()
cfg.set('grid_size', (4, 4))
cfg.set('num_train_samples', 20)
cfg.set('num_val_samples', 8)
cfg.set('num_test_samples', 8)
cfg.set('po_epochs', 1)
cfg.set('spo_epochs', 1)

result = run_sweep(cfg, num_seeds=2, present_results=True)
print(result.diagnostics)
"
```
