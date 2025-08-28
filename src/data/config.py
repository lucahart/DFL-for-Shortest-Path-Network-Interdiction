# config.py
from dataclasses import dataclass
from random import random
from typing import Any, Optional, Tuple

@dataclass
class HP:
    # Testing: To be removed
    lr: float = 1e-3
    batch: int = 64
    epochs: int = 20

    # -------- General simulation parameters --------
    random_seed: int = 31

    # -------- Bailey & Bayrak (2008) simulation parameters --------
    c_min: float = 1.0
    c_max: float = 10.0
    d_min: float = 1.0
    d_max: float = 10.0
    Q = 0.6
    B = 5
    network = (6, 8)

    # -------- ML hyperparameters --------
    num_features: int = 5
    num_data_samples: int = 100
    test_size: float = 0.2
    data_loader_batch_size: int = 32
    epochs: int = 10
    deg: int = 3
    noise_width: float = 0.05

    # -------- PO & SPO comparison --------
    sim_data_samples: int = 1000 # number of training data


    # -------- Postprocessing parameters --------
    # Functions to draw random costs and delays
    @staticmethod
    def draw_cost():
        return random() * (HP.c_max - HP.c_min) + HP.c_min

    @staticmethod
    def draw_delay():
        return random() * (HP.d_max - HP.d_min) + HP.d_min
    
    @staticmethod
    def draw_cost_est(cost: float):
        return random() * 2 * (1 - HP.Q) + HP.Q * cost 
    
    @staticmethod
    def draw_delay_est(delay: float):
        return random() * 2 * (1 - HP.Q) + HP.Q * delay

    # -------- Convenience methods --------
    def set(self, key: str, value: Any) -> None:
        """Set a configuration parameter by name."""
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a configuration parameter by name."""
        return getattr(self, key, default)
