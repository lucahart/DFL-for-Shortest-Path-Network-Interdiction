
import pyepo
import torch
import numpy as np

from src.models.ShortestPathGrid import ShortestPathGrid
from src.models.ShortestPathGrb import shortestPathGrb


def compare_po_spo(cfg,
                   opt_model: 'shortestPathGrb',
                   po_model: torch.nn.Module,
                   spo_model: torch.nn.Module,
                   test_data: dict
                   ) -> None:
    """
    Compare the performance of the PO and SPO models on a shortest path problem.
    """

    # Retrieve configuration parameters
    data_samples = cfg.get("num_test_samples") # number of training data
    m, n = cfg.get("grid_size")

    # Initialize lists to store results
    true_objs = []
    po_objs = []
    spo_objs = []

    # Iterate over the generated data samples
    for i in range(data_samples):
        # Store temporary values
        cost = test_data["costs"][i]
        feature = test_data["feats"][i]

        # Set the cost for the grid (Optionally specify the source and target nodes)
        opt_model.setObj(cost)

        # Solve shortest path problem
        _, obj = opt_model.solve()

        # Predict the shortest path with predict-then-optimize framework
        predicted_costs = po_model(torch.tensor(feature, dtype=torch.float32)).detach().numpy()
        po_graph = ShortestPathGrid(m, n, cost=predicted_costs)
        po_shortest_path = shortestPathGrb(po_graph)
        po_path, _ = po_shortest_path.solve()

        # Predict shortest path with smart predict-then-optimize framework
        predicted_costs = spo_model(torch.tensor(feature, dtype=torch.float32)).detach().numpy()
        spo_graph = ShortestPathGrid(m, n, cost=predicted_costs)
        spo_shortest_path = shortestPathGrb(spo_graph)
        spo_path, _ = spo_shortest_path.solve()

        # Evaluate the estimated paths
        po_obj = opt_model.evaluate(po_path)
        spo_obj = opt_model.evaluate(spo_path)

        # Store the results
        true_objs.append(obj)
        po_objs.append(po_obj)
        spo_objs.append(spo_obj)
    
    return true_objs, po_objs, spo_objs
