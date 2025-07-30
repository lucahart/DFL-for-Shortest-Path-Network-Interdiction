from random import random
import numpy as np

from models.ShortestPathGrid import ShortestPathGrid

def bayrak08(cfg):
    """
    Simulate the simulation sweep from Bayrak & Bailey (2008).
    
    Parameters
    ----------
    cfg : HP
        Hyperparameters for the simulation.
    """

    Q = [.6, .7, .8]
    B = [5, 10, 15, 20]
    networks = [(6, 8), (6, 10), (6, 12), (8, 8), (8, 10), (8, 12), (10, 10)]

    costs = np.random.rand(networks[0][0] * (networks[0][1] - 1) + (networks[0][0] - 1) * networks[0][1])
    delays = np.random.rand(networks[0][0] * (networks[0][1] - 1) + (networks[0][0] - 1) * networks[0][1])

    graph = ShortestPathGrid(networks[0], networks[1], costs)
    pass

