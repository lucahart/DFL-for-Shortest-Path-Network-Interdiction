
from dataclasses import dataclass
from typing import Any

@dataclass
class HP:
    # Define hyperparameters
    budget = 5
    grid_size = (6, 8)
    random_seed = 31
    intd_seed = 53
    loader_seed = 17
    num_seeds = 1

    # ML hyperparameters
    num_features = 5
    num_train_samples = 200
    num_val_samples = 50
    num_test_samples = 100
    batch_size = 32
    po_epochs = 150
    spo_epochs = 75
    po_lr = 1e-2
    spo_lr = 1e-3
    lam = 0
    deg = 7
    anchor = "mse"
    spo_po_epochs = 0
    noise_width = 1

    # Interdictor parameters
    benders_max_count = 50
    benders_eps = 1e-3
    lsd = 1e-5

    # -------- Convenience methods --------
    def set(self, key: str, value: Any) -> None:
        """Set a configuration parameter by name."""
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a configuration parameter by name."""
        return getattr(self, key, default)

