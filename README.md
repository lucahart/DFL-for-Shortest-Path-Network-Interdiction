# Decision-Focused Learning for Network Interdiction
A research codebase for experimenting with decision-focused learning (DFL) for shortest path network interdiction (SPNI) problems. The project builds up on the [PyEPO library](https://github.com/khalil-research/PyEPO) and integrates it with a more general graph optimization focused framework. We apply the code base to SPNI games, but encourage using it for other projects with DFL for graphs. 

This library includes different graph models, Gurobi-based solvers for different optimization problems, DFL training methods where possible, PyTorch training utilities, data wrappers, and application code for SPNI games as an application.


# Installation Instructions
> ⚠️ **Work in progress:** Several scripts are scaffolds or research prototypes. Expect to adapt components for your particular experiments. 
<!-- and see the [Roadmap](#roadmap--todo) for known gaps.-->

## Prerequisites & Requirements
General Information:
- **Python**: 3.10 or later is recommended.
- **Core libraries**: install via `requirements.txt` (`numpy`, `networkx`, `matplotlib`, `pyepo`, `torch`, `scikit-learn`, `tabulate`).
- **Mathematical programming**: many solvers rely on [Gurobi](https://www.gurobi.com/) (`gurobipy`). Ensure you have a valid license and have installed the Python package separately (`pip install gurobipy`).
- **GPU support (optional)**: PyTorch automatically detects CUDA if available.
- **Jupyter (optional)**: install `jupyterlab` or similar to run notebooks.


## Installation & Setup

1. **Clone the repository.**
   Follow the below installation instructions to install the environment. Start by copying the github repository. We recommend using SSH (see [here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh) for details).
   ```bash
    git clone git@github.com:lucahart/DFL-for-Shortest-Path-Network-Interdiction.git
    cd DFL-for-Shortest-Path-Network-Interdiction
   ```
2. **Create and activate a virtual environment.**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. **Install Python dependencies.**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. **Install the package in editable mode.** 
   This exposes the console script `shortest-path` that forwards to `src/Main.py`.
   ```bash
   pip install -e .
   ```
5. **(Optional) Notebook hygiene.** 
   If you plan on making contributions, run this command to avoid adding notebook outputs to version control.
   ```bash
   pip install nbstripout
   nbstripout --install
   ```

## Usage

<!-- ### Running the Bayrak & Bailey sweep prototype
```bash
shortest-path --lr 1e-3 --batch 64 --epochs 25
```
The CLI prints the active hyperparameter configuration (from `src/data/config.py`) and dispatches to `scripts.bayrak08`. The current implementation sets up the experiment scaffolding; extend `src/scripts/bayrak08.py` to complete the simulation for your study.

### Training a SPO model
```python
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from src.models import SPOTrainer
from src.models.ShortestPathGrid import ShortestPathGrid
from src.models.ShortestPathGrb import shortestPathGrb

# Synthetic graph and optimiser
grid = ShortestPathGrid(6, 8)
opt_layer = shortestPathGrb(grid)

# Simple predictive model
predictor = nn.Sequential(nn.Linear(5, grid.num_cost), nn.ReLU())
loss_fn = nn.MSELoss()
optimizer = optim.Adam(predictor.parameters(), lr=1e-3)
trainer = SPOTrainer(predictor, opt_layer, optimizer, loss_fn, method_name="spo+")

# Dummy dataset with (features, true_costs, shortest_paths, objectives, interdictions)
data = TensorDataset(
    torch.randn(128, 5),              # features
    torch.randn(128, grid.num_cost),  # costs
    torch.randn(128, grid.num_cost),  # sols
    torch.randn(128, 1),              # objs
    torch.zeros(128, 1, grid.num_cost)  # interdictions
)
loader = DataLoader(data, batch_size=32)
trainer.train_epoch(loader)
```
Adapt the dataset layout to your generator (`src/data/DataGenerator.py`) or adversarial loaders.

### Evaluating interdiction policies
```python
from src.solvers.BendersDecomposition import BendersDecomposition
from src.models.ShortestPathGrid import ShortestPathGrid
from src.models.ShortestPathGrb import shortestPathGrb
import numpy as np

m, n = 6, 8
cost = np.random.rand(m * (n - 1) + (m - 1) * n)
interdiction_cost = np.random.rand(cost.size)

grid = ShortestPathGrid(m, n, cost)
opt_layer = shortestPathGrb(grid)

benders = BendersDecomposition(opt_layer, k=5, interdiction_cost=interdiction_cost,
                               max_cnt=50, eps=1e-3)
x_intd, y_path, gap = benders.solve()
print(f"Budget used: {x_intd.sum()} edges\nObjective gap: {gap:.4f}")
```
Leverage `src/scripts/compare_po_spo.py` for end-to-end comparisons between PO and SPO policies (symmetric and asymmetric interdiction scenarios).

## Project Structure

```
├── README.md
├── requirements.txt
├── setup.py
├── src/
│   ├── Main.py                 # CLI entry point
│   ├── data/                   # Synthetic generators, loaders, config
│   ├── models/                 # PyTorch models, SPO trainers, graph layers
│   ├── solvers/                # Benders & SPNI interdiction solvers
│   ├── scripts/                # Experiment utilities and comparisons
│   ├── utils/                  # Miscellaneous helpers (progress bars, etc.)
│   └── Notebooks/              # Exploratory research notebooks
└── tests/                      # Pytest suites for calibration and grids
``` -->

## Testing & Validation

1. Install optional test dependencies (`pytest`, `torch`, `pyepo`, `gurobipy`).
2. Run the test suite from the project root:
   ```bash
   pytest
   ```
   Individual tests can be targeted, e.g. `pytest tests/test_arcs_one_hot.py`. Some tests automatically skip when optional dependencies are unavailable.

For manual experiments, visualise grid paths via `ShortestPathGrid.visualize()` or `shortestPathGrb.visualize()`.

<!-- 
## Contributing

1. Fork and create a feature branch.
2. Ensure tests pass (`pytest`) and follow Python type hints/docstring style already present in `src/models/`.
3. Use informative commit messages and submit a pull request describing the motivation, approach, and verification steps.
4. For large contributions (new solvers, dataset pipelines), open an issue to discuss design choices before implementation. -->

## License

No license has been specified yet. Until one is added, treat the repository as "all rights reserved" and contact the maintainers before reusing code.

## Citations & References
**Cite us when using this code base:**
```
Luca M. Hartmann, Parinaz Naghizadeh, "Decision-Focused Learning meets Network Interdiction: The Cost of Staying Behind," September 2025.
```

**Main References used in code base:**
- Israeli, E., & Wood, K. R. (2002). "Shortest-path network interdiction". *Networks*, 40(2):97–111. (Symmetric Interdictor).
- Bayrak, Ö., & Bailey, M. D. (2008). "Shortest Path Network Interdiction with Asymmetric Information." *Networks*, 52(3):133–140, 2008 (Asymmetric Interdictor).
- Elmachtoub, A. N., & Grigas, P. (2022). "Smart 'Predict, then Optimize'." *Management Science*, 68(7): 5152–5171. (Methodological background for SPO training).
- PyEPO: [https://github.com/khalil-research/PyEPO](https://github.com/khalil-research/PyEPO).
- Gurobi Optimizer: [https://www.gurobi.com/](https://www.gurobi.com/).

## Acknowledgements / Credits

- Original research code by Luca Hartmann under supervision of Parinaz Naghizadeh.
- Built on top of the PyEPO differentiable optimisation library and Gurobi optimiser.

## Contact & Support

- Open a GitHub issue for bug reports or feature requests.
- For private questions reach out to [lhartmann@ucsd.edu](mailto:lhartmann@ucsd.edu)
