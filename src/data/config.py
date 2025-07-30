# config.py
from dataclasses import dataclass
from random import random

@dataclass
class HP:
    # Testing: To be removed
    lr: float = 1e-3
    batch: int = 64
    epochs: int = 20

    # -------- General simulation parameters --------
    random_seed: int = 42

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
