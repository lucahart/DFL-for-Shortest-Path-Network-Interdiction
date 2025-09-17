
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
    num_train_samples = 50
    num_val_samples = 25
    num_test_samples = 100
    batch_size = 32
    po_epochs = 150
    spo_epochs = 50
    po_lr = 1e-2
    spo_lr = 1e-3
    lam = 0
    deg = 7
    anchor = "mse"
    spo_po_epochs = 0
    noise_width = 0.05

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


    # # -------- Postprocessing parameters --------
    # # Functions to draw random costs and delays
    # @staticmethod
    # def draw_cost():
    #     return random() * (HP.c_max - HP.c_min) + HP.c_min

    # @staticmethod
    # def draw_delay():
    #     return random() * (HP.d_max - HP.d_min) + HP.d_min
    
    # @staticmethod
    # def draw_cost_est(cost: float):
    #     return random() * 2 * (1 - HP.Q) + HP.Q * cost 
    
    # @staticmethod
    # def draw_delay_est(delay: float):
    #     return random() * 2 * (1 - HP.Q) + HP.Q * delay
