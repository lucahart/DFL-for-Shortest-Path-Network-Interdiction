
from dataclasses import dataclass
from typing import Any

@dataclass
class HP:
    # Define hyperparameters
    budget = 10
    grid_size = (5, 5)
    num_scenarios = 2
    num_seeds = 5

    random_seed = 31
    intd_seed = 53
    loader_seed = 17


    # ML hyperparameters
    num_features = 5
    num_train_samples = 1000
    num_val_samples = 100
    num_test_samples = 1000
    batch_size = 32
    po_epochs = 400
    spo_epochs = 200
    po_lr = 2e-4
    spo_lr = 3.5e-4
    lam = 0
    deg = 8
    anchor = "mse"
    spo_po_epochs = 0
    noise_width = 0.5

    # Interdictor parameters
    benders_max_count = 100
    benders_eps = 1e-3
    lsd = 1e-5

    # -------- Convenience methods --------
    def set(self, key: str, value: Any) -> None:
        """Set a configuration parameter by name."""
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a configuration parameter by name."""
        return getattr(self, key, default)

