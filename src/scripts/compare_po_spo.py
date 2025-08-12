import math as m
import pyepo
import torch
import numpy as np

from src.models.ShortestPathGrid import ShortestPathGrid
from src.models.ShortestPathGrb import shortestPathGrb


def compare_po_spo(cfg,
                   opt_model: 'shortestPathGrb',
                   po_model: torch.nn.Module,
                   spo_model: torch.nn.Module
                   ) -> None:
    """
    Compare the performance of the PO and SPO models on a shortest path problem.
    """

    # Retrieve configuration parameters
    data_samples = cfg.get("test_data_samples") # number of training data
    m, n = cfg.get("grid_size")

    # Generate data for shortest path problem
    features, costs = pyepo.data.shortestpath.genData(
        data_samples,
        cfg.get("num_features"),
        (m, n),
        deg=cfg.get("deg"),
        noise_width=cfg.get("noise_width"),
        seed=cfg.get("seed")
    )

    # deg=3
    # noise_width=0.05
    # seed=31
    # data_samples = 10
    # m,n = (6, 8)
    # num_features = 5
    
    # features, costs = pyepo.data.shortestpath.genData(
    #     data_samples,
    #     num_features,
    #     (m, n),
    #     deg=deg,
    #     noise_width=noise_width,
    #     seed=seed
    # )

    # Initialize lists to store results
    true_objs = []
    po_objs = []
    spo_objs = []

    # Iterate over the generated data samples
    for i in range(data_samples):
        # Store temporary values
        cost = costs[i]
        feature = features[i]

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
