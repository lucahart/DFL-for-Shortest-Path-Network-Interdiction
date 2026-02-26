
from dataclasses import dataclass
from typing import Any, Tuple

@dataclass
class HP:
    # Hyperparameters
    num_seeds : int = 5 # number of simulations to run and random seed sets to use
    seed_sweep_offset : int = 100 # Offset to ensure different random seeds for different runs

    # Data parameters
    num_features : int = 5
    num_train_samples : int = 1000
    num_val_samples : int = 100
    num_test_samples : int = 1000

    grid_size : Tuple[int, int] = (5, 5)
    deg : int = 8
    noise_width : float = 0.5

    seed : int = seed_sweep_offset
    random_seed : int = 31
    loader_seed : int = 17

    # Interdiction parameters
    budget : int = 10
    num_scenarios : int = 3

    benders_max_count : int = 100
    benders_eps : float = 1e-3
    lsd : float = 1e-5

    intd_seed : int = 53

    # ML hyperparameters
    batch_size : int = 32
    po_epochs : int = 400
    spo_epochs : int = 200 # 300
    po_lr : float = 2e-4 # 2e-3
    spo_lr : float = 3.5e-4 # 5e-3

    # Deprecated parameters. TODO: Remove them in the future.
    lam = 0.0
    anchor = "mse"
    spo_po_epochs = 0

    # -------- Convenience methods --------
    def set(self, key: str, value: Any) -> None:
        """Set a configuration parameter by name."""
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a configuration parameter by name."""
        return getattr(self, key, default)

