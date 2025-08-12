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
    random_seed: int = 42
    # Parameters for dataset generation and model configuration
    test_data_samples: Optional[int] = None
    deg: Optional[int] = None
    noise_width: Optional[float] = None
    seed: Optional[int] = None
    num_features: Optional[int] = None
    grid_size: Optional[Tuple[int, int]] = None

    # -------- Bailey & Bayrak (2008) simulation parameters --------
    c_min: float = 1.0
    c_max: float = 10.0
    d_min: float = 1.0
    d_max: float = 10.0
    Q = .6


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
