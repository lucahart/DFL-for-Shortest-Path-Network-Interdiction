
from pyepo.data.shortestpath import genData

from dflintdpy.data.config import HP
from dflintdpy.solvers.shortest_path_grb import ShortestPathGrb

def gen_syn_data(cfg: HP, opt_model: ShortestPathGrb):
    """
    Generates synthetic data using pyepo's shortest path data generator.

    The config class 'cfg' should contain the following fields:
    - num_train_samples
    - num_val_samples
    - num_test_samples
    - num_features
    - deg
    - noise_width
    - random_seed (optional)

    The generated data will use opt_model to determine the # of edges.
    """
    features, costs = genData(
        cfg.get("num_train_samples") + cfg.get("num_val_samples") + cfg.get("num_test_samples"), 
        cfg.get("num_features"), 
        (1, opt_model.num_cost+1), 
        deg=cfg.get("deg"), 
        noise_width=cfg.get("noise_width"), 
        seed=cfg.get("random_seed")
    )

    return features, costs

