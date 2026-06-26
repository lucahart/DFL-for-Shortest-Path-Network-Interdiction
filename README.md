# Decision-Focused Learning for Network Interdiction

A research codebase for experimenting with decision-focused learning (DFL) applied to shortest-path network interdiction (SPNI) games. The project integrates DFL training methods with a Gurobi-based graph optimization framework and evaluates predictor families under symmetric and asymmetric interdiction scenarios.

The library provides graph models, Gurobi-based solvers for SPNI problems, DFL and predict-then-optimize (PO) training utilities, adversarial data generation, a typed simulation pipeline, and a CLI for running reproducible experiments.

## Prerequisites

- **Python** 3.9 or later
- **Gurobi** with a valid license — most solvers rely on `gurobipy`. Install the Python package separately (`pip install gurobipy`) and activate a license before running experiments.
- **PyTorch** — install separately to match your CUDA version if needed, or let `pip` resolve a CPU build.
- **Core libraries** — `numpy`, `networkx`, `pyepo`, `matplotlib`, `scikit-learn`, `pandas`, `cvxpy` (installed via `requirements.txt`).

## Installation

1. **Clone the repository.**
   ```bash
   git clone git@github.com:<your-org>/DFL-for-Shortest-Path-Network-Interdiction.git
   cd DFL-for-Shortest-Path-Network-Interdiction
   ```

2. **Create and activate a virtual environment.**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

3. **Install Python dependencies.**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Install the package in editable mode.**
   This registers the `dflintd` and `dflintd-spni` console entry points.
   ```bash
   pip install -e .
   ```

5. **(Optional) Strip notebook outputs from version control.**
   ```bash
   pip install nbstripout
   nbstripout --install
   ```

## Running SPNI Simulations

The primary entry point for running SPNI experiments is the `dflintd-spni` CLI. Full documentation of all flags and config parameters is in [docs/spni_run_guide.md](docs/spni_run_guide.md).

**Quick smoke test** (runs in under a minute):
```bash
source .venv/bin/activate

dflintd-spni \
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

**Grid seed sweep** (paper default settings):
```bash
dflintd-spni \
  --mode seed_sweep \
  --num-seeds 5 \
  --set 'grid_size=(5, 5)' \
  --set 'deg=8' \
  --set 'num_train_samples=500' \
  --set 'num_val_samples=100' \
  --set 'num_test_samples=250' \
  --set 'num_scenarios=3' \
  --set 'budget=5'
```

**Real-world graph topology with synthetic costs:**
```bash
dflintd-spni \
  --mode seed_sweep \
  --num-seeds 5 \
  --load-real-world-graph real_world_spni_data/transportation_networks/Anaheim_net.tntp \
  --source-node 108 \
  --target-node 410 \
  --set 'deg=12' \
  --set 'num_train_samples=500' \
  --set 'num_val_samples=100' \
  --set 'num_test_samples=250' \
  --set 'num_scenarios=3' \
  --set "surrogate_underprediction_penalty_weight=1.0" \
  --set "pred_model=\"nn\""

```

See [docs/spni_run_guide.md](docs/spni_run_guide.md) for all modes (`single`, `seed_sweep`, `scenario_sweep`, `replot`), config parameters, and output paths.

## Python API

The simulation pipeline is also accessible directly from Python:

```python
from dflintdpy.data.config import HP
from dflintdpy.simulation.spni import run_seed_sweep, persist_sweep_outputs

cfg = HP()
cfg.set("grid_size", (5, 5))
cfg.set("num_train_samples", 500)
cfg.set("num_scenarios", 3)

result = run_seed_sweep(cfg, num_seeds=5)
persist_sweep_outputs(result)
```

Individual components — graph construction, solvers, data generators, and trainers — can be imported from their respective subpackages under `dflintdpy`.

## Project Structure

```
├── README.md
├── requirements.txt
├── pyproject.toml
├── docs/
│   └── spni_run_guide.md         # CLI reference and run guide
├── Notebooks/                    # Research notebooks
│   ├── 5_Asym_SPNI_Toy_Example.ipynb
│   ├── 7_Asym_SPNI.ipynb
│   ├── 9_real_world_example.ipynb
│   ├── 10-real_world_example_reworked.ipynb
│   └── ...
├── real_world_spni_data/         # Graph topology files for real-world experiments
├── src/
│   ├── dflintdpy/
│   │   ├── cli/                  # CLI entry points (dflintd, dflintd-spni)
│   │   ├── data/                 # Config (HP), synthetic data generation, adversarial loaders
│   │   ├── models/               # Graph and Grid models
│   │   ├── predictors/           # Predictor architectures and hybrid SPO+ loss
│   │   ├── scripts/              # Legacy experiment scripts (compatibility wrappers)
│   │   ├── simulation/
│   │   │   └── spni/             # Typed SPNI pipeline (build, data, train, evaluate, results)
│   │   ├── solvers/              # Gurobi-based shortest-path and interdiction solvers
│   │   └── utils/                # Result I/O, analysis helpers, trainers
│   └── toy_example/              # Standalone toy SPNI example
└── tests/                        # Pytest suite (unit, integration, regression)
```

## Testing

```bash
pytest
```

Integration and regression tests require Gurobi. Unit tests mock heavy solver calls and run without a license.

## Citation

If you use this code, please cite:

```
Luca M. Hartmann, Parinaz Naghizadeh,
"Decision-Focused Learning in Network Interdiction Games,"
Working paper, June 2026.
```
Bibtex citation:

```bibtex
@misc{hartmann2026dfl_spni,
  author       = {Hartmann, Luca M. and Naghizadeh, Parinaz},
  title        = {Decision-Focused Learning in Network Interdiction Games},
  year         = {2026},
  month        = {June},
  note         = {Working paper},
}
```

**Key references used in this codebase:**
- Israeli & Wood (2002). "Shortest-path network interdiction." *Networks*, 40(2):97–111.
- Bayrak & Bailey (2008). "Shortest Path Network Interdiction with Asymmetric Information." *Networks*, 52(3):133–140.
- Elmachtoub & Grigas (2022). "Smart 'Predict, then Optimize'." *Management Science*, 68(7):5152–5171.
- PyEPO: [https://github.com/khalil-research/PyEPO](https://github.com/khalil-research/PyEPO)
- Gurobi Optimizer: [https://www.gurobi.com/](https://www.gurobi.com/)

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgements

Original research code by Luca Hartmann under supervision of Parinaz Naghizadeh at UC San Diego. Built on top of the PyEPO differentiable optimisation library and the Gurobi optimizer.

## Contact

- GitHub issues: bug reports and feature requests
- Email: [lhartmann@ucsd.edu](mailto:lhartmann@ucsd.edu)
