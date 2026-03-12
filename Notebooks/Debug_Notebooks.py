# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import random
from tabulate import tabulate

import dflintdpy.scripts as scripts
from dflintdpy.data.config import HP
from dflintdpy.models.grid import Grid
from dflintdpy.solvers.shortest_path_grb import ShortestPathGrb
from dflintdpy.solvers.symmetric_interdictor import SymmetricInterdictor
from dflintdpy.solvers.asymmetric_interdictor import AsymmetricInterdictor
from dflintdpy.utils.real_world_spni_data_handling import csv_to_graph

from dflintdpy.scripts.compare import (compare_shortest_paths, 
                                       compare_sym_intd,
                                       compare_asym_intd, 
                                       compare_wrong_asym_intd)
from dflintdpy.scripts.setup import (gen_data, 
                                     gen_train_data, 
                                     setup_pfl_predictor, 
                                     setup_dfl_predictor)

seed_number = 0
np.random.seed(seed_number)
seed1, seed2, seed3 = np.random.randint(0, 150, 3).tolist()

# Initialize the configuration class
cfg = HP()
# Change cfg parameters here
cfg.set("random_seed", seed1)
cfg.set("intd_seed", seed2)
cfg.set("data_loader_seed", seed3)

cfg.set("num_train_samples", 150)
cfg.set("num_val_samples", 50)
cfg.set("num_test_samples", 100)

# Set the random seed for reproducibility
np.random.seed(cfg.get("random_seed"))
random.seed(cfg.get("random_seed"))
torch.manual_seed(cfg.get("random_seed"))
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.get("random_seed"))

# Define a graph with appropriate dimensions and an opt_model 
# for solving the shortest path problem on the graph
file_path = os.path.join(
    os.getcwd(), 
    'real_world_spni_data/county_level_arcs.csv'
)
graph = csv_to_graph(file_path) # Grid(m=6, n=8) #
cfg.set("grid_size", (graph.num_cost+1, 1))
opt_model = ShortestPathGrb(graph)
# graph.visualize()

# Generate normalized training and testing data
training_data, testing_data, normalization_constant = gen_train_data(cfg, opt_model)

