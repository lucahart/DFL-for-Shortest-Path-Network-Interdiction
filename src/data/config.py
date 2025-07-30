# config.py
from dataclasses import dataclass

@dataclass
class HP:
    lr: float = 1e-3
    batch: int = 64
    epochs: int = 20
