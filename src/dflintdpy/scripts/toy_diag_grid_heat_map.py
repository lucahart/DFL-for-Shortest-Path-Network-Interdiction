from dflintdpy.models.dgrid import DGrid
from dflintdpy.solvers.shortest_path_grb import ShortestPathGrb
from dflintdpy.solvers.symmetric_interdictor import SymmetricInterdictor
from dflintdpy.scripts.setup import gen_train_data, setup_pfl_predictor, setup_dfl_predictor
from dflintdpy.data.config import HP
import matplotlib.pyplot as plt
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
    dgrid = DGrid(m, n, cost=costs[0])
    # opt_model = ShortestPathGrb(graph=dgrid)
    intd_model = SymmetricInterdictor(
        dgrid,
        k=15,
        interdiction_cost=intd_costs[0],
    )
    _create_heat_map_intd(intd_model, costs, intd_costs)
    pass

def _create_heat_map_intd(intd_model, costs, intd_costs):

    n_trials, n_costs = costs.shape
    arc_count = np.zeros(n_costs)
    for i in range(n_trials):
        intd_model.opt_model.setObj(costs[i])
        intd, shortest_path, _ = intd_model.benders_decomposition(intd_costs[i], versatile=False)
        # shortest_path, objective = opt_model.solve()
        arc_count += shortest_path
    arc_count /= n_trials
    # Visualize heat map
    intd_model.opt_model.setObj(arc_count)
    intd_model.opt_model.visualize(heat_map=arc_count, 
                        title="Interdicted Shortest Paths Heat Map"),
                        # xlabel="Edge weights represent frequency of usage [%]")

def compare_shortest_path_heat_maps():
    shortest_path_heat_map()
    intd_shortest_path_heat_map()
    pass

def train_dfl_on_shortest_path(cfg):

    # Setup parameters
    n_test = 100
    cfg.set("num_train_samples", 100)
    cfg.set("num_val_samples", 10)
    cfg.set("num_test_samples", n_test)
    cfg.set("deg", 4)
    cfg.set("pred_model", "nn")  # "nn" or "linear"
    # cfg.set("po_epochs", 50)
    # cfg.set("spo_epochs", 50)
    cfg.set("renormalize_predictions", False)

    # Define grid network
    m, n = 3, 3
    costs = np.random.rand(m * (n - 1) + (m - 1) * n + (m - 1) * (n - 1))
    dgrid = DGrid(m, n, cost=costs)
    opt_model = ShortestPathGrb(graph=dgrid)

    # Generate synthetic training data
    train_loaders, test_data, _, extra_data = gen_train_data(cfg, opt_model)
    data_gen = extra_data["data_generator"]
    nonadv_t_data = train_loaders["train_loader"].get_nonadverse_loader()
    nonadv_v_data = train_loaders["val_loader"].get_nonadverse_loader()
    nonadv_train_loaders = {
        "train_loader": nonadv_t_data,
        "val_loader": nonadv_v_data
    }

    # Show shortest paths on training data
    train_costs = train_loaders['train_loader'].dataset.costs[:,0,:].squeeze()
    _create_heat_map(opt_model, train_costs)
    intd_model = SymmetricInterdictor(
        dgrid,
        k=15,
        interdiction_cost=data_gen.interdictions[0],
    )
    _create_heat_map_intd(intd_model, train_costs, data_gen.interdictions)

    # # Train the PFL predictor
    # po_predictor = setup_pfl_predictor(
    #     cfg,
    #     dgrid,
    #     opt_model,
    #     train_loaders,
    #     versatile=False
    # )
    # _create_predictor_heat_map(cfg, opt_model, po_predictor, test_data)

    # Train the DFL predictor
    spo_predictor = setup_dfl_predictor(
        cfg,
        dgrid,
        opt_model,
        nonadv_train_loaders,
        versatile=False
    )
    # _create_predictor_heat_map(cfg, opt_model, spo_predictor, test_data)

    # Train the A-DFL predictor
    aspo_predictor = setup_dfl_predictor(
        cfg,
        dgrid,
        opt_model,
        train_loaders,
        versatile=False
    )
    # _create_predictor_heat_map(cfg, opt_model, aspo_predictor, test_data)


    sample_idx = 5
    fig, ax = plt.subplots(2, 3, figsize=(12,8))
    opt_model.setObj(test_data["costs"][sample_idx])
    opt_model.visualize(dashed_edges=opt_model.solve()[0], ax=ax[0,0], title="True Shortest Path")
    cfg.set("renormalize_predictions", False)
    _create_predictor_heat_map_single_sample(cfg, opt_model, spo_predictor, test_data, sample_idx=sample_idx, ax=ax[0,1], title="SPO Predictor")
    _create_predictor_heat_map_single_sample(cfg, opt_model, aspo_predictor, test_data, sample_idx=sample_idx, ax=ax[0,2], title="A-SPO Predictor")
    cfg.set("renormalize_predictions", True)
    _create_predictor_heat_map_single_sample(cfg, opt_model, spo_predictor, test_data, sample_idx=sample_idx, ax=ax[1,1])
    _create_predictor_heat_map_single_sample(cfg, opt_model, aspo_predictor, test_data, sample_idx=sample_idx, ax=ax[1,2])
    pass


def _create_predictor_heat_map(
        cfg, 
        opt_model,
        pred_model, 
        test_data
    ):

    # Retrieve data
    n_test = cfg.get("num_test_samples")

    # Evaluate predictor with heat map
    cost_diff = np.zeros((n_test, opt_model.num_cost))
    for idx in range(n_test):
        pred_model.eval()  # important if you have dropout / batchnorm
        x = torch.from_numpy(test_data["feats"][idx])
        if x.dtype != torch.float32:
            x = x.float()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        device = next(pred_model.parameters()).device
        x = x.to(device)
        with torch.no_grad():
            y = pred_model(x)
        y_np = y.detach().cpu().numpy()
        if cfg.get("renormalize_predictions"):
            y_np = y_np * (test_data["costs"][idx].mean() / y_np.mean())  # renormalize
        cost_diff[idx] = (y_np - test_data["costs"][idx]) / abs(y_np) * 100 # percentage error

    # Visualize PFL heat map
    mean = cost_diff.mean(axis=0)
    std = cost_diff.std(axis=0)
    opt_model._graph.cost = [f"{m:.0f} ± {s:.0f}" for m, s in zip(mean, std)]
    opt_model.visualize(heat_map=mean / max(abs(mean)))
    pass


def _create_predictor_heat_map_single_sample(
    cfg,
    opt_model,
    pred_model,
    test_data,
    sample_idx=0,
    **kwargs
):
    true_cost = test_data["costs"][sample_idx]

    # Evaluate predictor with heat map
    pred_model.eval()  # important if you have dropout / batchnorm
    x = torch.from_numpy(test_data["feats"][sample_idx])
    if x.dtype != torch.float32:
        x = x.float()
    if x.ndim == 1:
        x = x.unsqueeze(0)
    device = next(pred_model.parameters()).device
    x = x.to(device)
    with torch.no_grad():
        y = pred_model(x)
    y_np = y.detach().cpu().numpy()
    if cfg.get("renormalize_predictions"):
        y_np = y_np * (true_cost.mean() / y_np.mean())  # renormalize
    cost_diff = (y_np - true_cost) / abs(y_np) * 100 # percentage error
    cost_diff = cost_diff.squeeze()

    # Visualize PFL heat map
    opt_model.setObj(y_np.squeeze())
    shortest_path, _ = opt_model.solve()
    opt_model.setObj(y_np.squeeze().round(2))
    opt_model.visualize(
        heat_map=cost_diff / max(abs(cost_diff)),
        dashed_edges=shortest_path,
        **kwargs
    )
    pass

# TODO: Learn the edges with DFL and show what edges are more over or underestimated

def main():
    cfg = HP()
    # single_shortest_path_example()
    # shortest_path_heat_map()
    # intd_shortest_path_heat_map()
    # compare_shortest_path_heat_maps()
    train_dfl_on_shortest_path(cfg)
    pass

if __name__ == "__main__":
    main()