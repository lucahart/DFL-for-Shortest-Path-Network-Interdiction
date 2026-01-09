from dflintdpy.models.dgrid import DGrid
from dflintdpy.solvers.shortest_path_grb import ShortestPathGrb
from dflintdpy.solvers.symmetric_interdictor import SymmetricInterdictor
from dflintdpy.scripts.setup import gen_train_data, setup_pfl_predictor, setup_dfl_predictor
from dflintdpy.data.config import HP
import numpy as np
import torch


def single_shortest_path_example():
    m, n = 5, 5
    costs = np.random.rand(m * (n - 1) + (m - 1) * n + (m - 1) * (n - 1))
    dgrid = DGrid(m, n, cost=costs)
    opt_model = ShortestPathGrb(graph=dgrid)
    shortest_path, objective = opt_model.solve()
    opt_model.visualize(colored_edges=shortest_path)

    dgrid.scale_diagonal_edges(scale=np.sqrt(2))
    opt_model_scaled = ShortestPathGrb(graph=dgrid)
    shortest_path_scaled, objective_scaled = opt_model_scaled.solve()
    opt_model_scaled.visualize(colored_edges=shortest_path_scaled)
    pass

def shortest_path_heat_map():
    m, n = 5, 5
    n_costs = m * (n - 1) + (m - 1) * n + (m - 1) * (n - 1)
    n_trials = 100
    costs = np.random.rand(n_trials, n_costs)
    dgrid = DGrid(m, n)
    opt_model = ShortestPathGrb(graph=dgrid)
    _create_heat_map(opt_model, costs)
    pass

def _create_heat_map(opt_model, costs):

    n_trials, n_costs = costs.shape
    arc_count = np.zeros(n_costs)
    for i in range(n_trials):
        opt_model.setObj(costs[i])
        shortest_path, objective = opt_model.solve()
        arc_count += shortest_path
    arc_count /= n_trials

    # Visualize heat map
    opt_model.setObj(arc_count)
    opt_model.visualize(heat_map=arc_count, title="Shortest Paths Heat Map")

def intd_shortest_path_heat_map():
    m, n = 5, 5
    n_costs = m * (n - 1) + (m - 1) * n + (m - 1) * (n - 1)
    n_trials = 100
    costs = np.random.rand(n_trials, n_costs)
    intd_costs = np.random.rand(n_trials, n_costs)
    arc_count = np.zeros(n_costs)
    for i in range(n_trials):
        dgrid = DGrid(m, n, cost=costs[i])
        opt_model = ShortestPathGrb(graph=dgrid)
        intder = SymmetricInterdictor(
            dgrid,
            k=15,
            interdiction_cost=intd_costs[i],
        )
        intd, shortest_path, _ = intder.solve(versatile=False)
        # shortest_path, objective = opt_model.solve()
        arc_count += shortest_path
    arc_count /= n_trials
    # Visualize heat map
    opt_model.setObj(arc_count)
    opt_model.visualize(heat_map=arc_count, 
                        title="Interdicted Shortest Paths Heat Map")
    pass

def compare_shortest_path_heat_maps():
    shortest_path_heat_map()
    intd_shortest_path_heat_map()
    pass

def train_dfl_on_shortest_path():

    # Setup parameters
    cfg = HP()
    cfg.set("num_train_samples", 1000)
    cfg.set("num_val_samples", 10)
    cfg.set("num_test_samples", 1000)

    # Define grid network
    m, n = 5, 5
    costs = np.random.rand(m * (n - 1) + (m - 1) * n + (m - 1) * (n - 1))
    dgrid = DGrid(m, n, cost=costs)
    opt_model = ShortestPathGrb(graph=dgrid)

    # Generate synthetic training data
    cfg.set("num_scenarios", 1)
    train_loaders, test_data, norm_const = gen_train_data(cfg, opt_model)

    # Show shortest paths on training data
    train_costs = train_loaders['train_loader'].dataset.costs[:,0,:].squeeze()
    _create_heat_map(opt_model, train_costs)

    # Train the PFL predictor
    po_predictor = setup_dfl_predictor(
        cfg,
        dgrid,
        opt_model,
        train_loaders,
        versatile=False
    )

    # Evaluate predictor with heat map
    cost_diff = np.zeros((1, opt_model.num_cost))
    for idx in range(cfg.get("num_test_samples")):
        po_predictor.eval()  # important if you have dropout / batchnorm
        x = torch.from_numpy(test_data["feats"][idx])
        if x.dtype != torch.float32:
            x = x.float()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        device = next(po_predictor.parameters()).device
        x = x.to(device)
        with torch.no_grad():
            y = po_predictor(x)
        y_np = y.detach().cpu().numpy()
        y_np = y_np * (test_data["costs"][idx].mean() / y_np.mean())  # renormalize
        cost_diff += (y_np - test_data["costs"][idx]) / abs(y_np)

    # Visualize PFL heat map
    cost_diff /= cfg.get("num_test_samples") * 100 # compute mean and percentage
    opt_model.setObj(cost_diff.flatten())
    cost_diff /= max(abs(cost_diff.squeeze()))
    opt_model.visualize(heat_map=cost_diff.flatten())
    pass

# TODO: Learn the edges with DFL and show what edges are more over or underestimated

def main():
    # single_shortest_path_example()
    # shortest_path_heat_map()
    # compare_shortest_path_heat_maps()
    train_dfl_on_shortest_path()
    pass

if __name__ == "__main__":
    main()