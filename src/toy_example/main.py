#!/usr/bin/env python3
import copy
import numpy as np
import torch
import torch.nn as nn
try:
    import cvxpy as cp
    from cvxpylayers.torch import CvxpyLayer
except ImportError:
    cp = None
    CvxpyLayer = None
try:
    import pyepo
except ImportError:
    pyepo = None
from toy_example.main_funcs import (
    feature_cost_mapping, 
    optimizer, 
    sol_value,
    interdictor
)
from toy_example.plotting import (
    plot_predictor_sweep,
    plot_dfl_init_vs_trained,
    plot_dfl_init_vs_cvx_vs_spo,
)

# PARAMETERS
SEED = 5
N_EPOCHS = 100
SAMPLE_MAX = 6.0
D_TEST = 5.0
LR_DEFAULT = 2e-2
N_TRAIN_SAMPLES = 100

# Constants
torch.manual_seed(SEED)
W_TRAIN = (torch.rand(N_TRAIN_SAMPLES, 1)-.5)*2*SAMPLE_MAX # torch.tensor([-1.0, 1.0]).unsqueeze(-1)  # training features
W_TEST = torch.tensor([-SAMPLE_MAX, -1.0, 1.0, SAMPLE_MAX]).unsqueeze(-1)  # test features

# Utility Functions
_CVXPY_PATH_LAYER = None

def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)

def _require_cvxpy_layers():
    if cp is None or CvxpyLayer is None:
        raise ImportError(
            "cvxpy and cvxpylayers are required for DFL backward pass. "
            "Install with: pip install cvxpy cvxpylayers"
        )

def _require_pyepo():
    if pyepo is None:
        raise ImportError(
            "pyepo is required for SPO+ training. "
            "Install with: pip install pyepo"
        )

def _get_cvxpy_path_layer():
    global _CVXPY_PATH_LAYER
    _require_cvxpy_layers()
    if _CVXPY_PATH_LAYER is None:
        path_cost = cp.Parameter(2)
        path_mix = cp.Variable(2)
        objective = cp.Minimize(
            path_cost @ path_mix #+ 1e-3 * cp.sum_squares(path_mix)
        )
        constraints = [cp.sum(path_mix) >= 1, path_mix >= 0]
        problem = cp.Problem(objective, constraints)
        if not problem.is_dpp():
            raise RuntimeError("CVXPY problem is not DPP-compliant for CvxpyLayer.")
        _CVXPY_PATH_LAYER = CvxpyLayer(
            problem,
            parameters=[path_cost],
            variables=[path_mix],
            # solver="DIFFCP",
            # solver_args={"eps": 1e-8, "max_iters": 5000}
        )
    return _CVXPY_PATH_LAYER

def _edge_to_path_costs(c):
    return torch.stack((c[..., 0] + c[..., 1], c[..., 2] + c[..., 3]), dim=-1)

def _path_mix_to_edge_solution(path_mix):
    return torch.stack(
        (path_mix[..., 0], path_mix[..., 0], path_mix[..., 1], path_mix[..., 1]),
        dim=-1
    )

def dfl_loss_cvxpy(c_pred, c_true, path_layer):
    pred_path_costs = _edge_to_path_costs(c_pred)
    path_mix_pred, = path_layer(
        pred_path_costs
    )
    y_pred = _path_mix_to_edge_solution(path_mix_pred)
    return torch.mean(torch.sum(c_true * y_pred, dim=-1))

def dfl_loss_spoplus(c_pred, c_true, y_true, z_true, spo_plus_loss):
    return spo_plus_loss(c_pred, c_true, y_true, z_true)

def build_toy_dataset_dfl(w_values):
    c = feature_cost_mapping(w_values)
    y = optimizer(c)
    z = sol_value(c, y)
    return w_values, c, y, z

def weights_init(m, value: float = 0.0):
    assert isinstance(m, nn.Module), \
        "Expected nn.Module, got {}".format(type(m))
    assert isinstance(value, float), \
        "Expected float value, got {}".format(type(value))
    if isinstance(m, nn.Linear):
        with torch.no_grad():
            m.weight.fill_(value)
            if m.bias is not None:
                m.bias.fill_(value)

def new_predictor():
    predictor = nn.Sequential(
        nn.Linear(1, 4),
        # nn.ReLU(),
        # nn.Linear(4, 4)
    )
    # predictor.apply(lambda m: weights_init(m, value=1.0))
    # with torch.no_grad():
    #     predictor[0].weight = torch.nn.Parameter(torch.tensor([.1, .1, -.1, -.1], dtype=torch.float).unsqueeze(1))
    #     predictor[0].bias.zero_()
    # with torch.no_grad():
    #     predictor[0].weight = torch.nn.Parameter(torch.tensor([1, -2, -1, 2], dtype=torch.float).unsqueeze(1))
    #     predictor[0].bias.zero_()
    #     predictor[2].weight = torch.nn.Parameter(torch.tensor([[1.0, 2.0, 0.0, 0.0], [1, 2, 0, 0], [0, 0, 1, 2], [0, 0, 1, 2]], dtype=torch.float))
    #     predictor[2].bias.zero_()
    return predictor

# Learning logic
def build_toy_dataset_pfl(w_values):
    c = feature_cost_mapping(w_values)
    return w_values, c

def train_pfl_predictor(predictor, x_train, y_train, epochs=N_EPOCHS, lr=LR_DEFAULT, seed=SEED):
    set_seed(seed)
    predictor.train()
    criterion = nn.MSELoss()
    optimizer_ = torch.optim.Adam(predictor.parameters(), lr=lr)
    print_every = max(1, epochs // 10)
    for epoch in range(epochs):
        optimizer_.zero_grad()
        preds = predictor(x_train)
        loss = criterion(preds, y_train)
        if (epoch + 1) % print_every == 0 or epoch == epochs - 1:
            print(f"[PFL] epoch {epoch + 1}/{epochs} loss: {loss.item():.6f}")
        loss.backward()
        optimizer_.step()
    return predictor

def toy_example_pfl(seed=SEED):
    set_seed(seed)
    predictor = new_predictor()
    w_values = W_TRAIN  # training features
    x_train, y_train = build_toy_dataset_pfl(w_values)
    train_pfl_predictor(predictor, x_train, y_train, seed=seed)
    return predictor

def test_pfl_predictor(seed=SEED):
    pred_pfl = toy_example_pfl(seed=seed)
    w = W_TEST  # test features
    c_true = feature_cost_mapping(w) # true test costs
    with torch.no_grad():
        c_pred = pred_pfl(w) # predicted test costs
    intd = interdictor(c_true) # interdictions
    c_intd_true = c_true + intd # true interdicted costs
    c_intd_pred = c_pred + intd # predicted interdicted costs
    y_true = optimizer(c_intd_true) # true interdicted opt. sol.
    y_pred = optimizer(c_intd_pred) # predicted interdicted opt. sol.
    print(f"True costs:\n{c_true}")
    print(f"Predicted costs:\n{c_pred}")
    print(f"Interdictions:\n{intd}")
    print(f"True interdicted costs:\n{c_intd_true}")
    print(f"Predicted interdicted costs:\n{c_intd_pred}")
    print(f"True interdicted opt. sol.:\n{y_true}")
    print(f"Predicted interdicted opt. sol.:\n{y_pred}")
    pass

def train_dfl_predictor_cvx(predictor, w_train, c_train, y_train, z_train, epochs=N_EPOCHS, lr=LR_DEFAULT, seed=SEED):
    set_seed(seed)
    predictor.train()
    path_layer = _get_cvxpy_path_layer()
    optimizer_ = torch.optim.Adam(predictor.parameters(), lr=lr)
    print_every = max(1, epochs // 10)
    for epoch in range(epochs):
        optimizer_.zero_grad()
        c_pred = predictor(w_train)
        loss = dfl_loss_cvxpy(c_pred, c_train, path_layer)
        if (epoch + 1) % print_every == 0 or epoch == epochs - 1:
            print(f"[DFL-CVXPY] epoch {epoch + 1}/{epochs} loss: {loss.item():.6f}")
        loss.backward()
        optimizer_.step()
    return predictor

def train_dfl_predictor_spo(predictor, w_train, c_train, y_train, z_train, epochs=N_EPOCHS, lr=LR_DEFAULT, seed=SEED):
    _require_pyepo()
    from toy_example.opt import ToyOptModel

    set_seed(seed)
    predictor.train()
    opt_model = ToyOptModel(None)
    spo_plus_loss = pyepo.func.SPOPlus(opt_model, processes=1)
    optimizer_ = torch.optim.Adam(predictor.parameters(), lr=lr)
    print_every = max(1, epochs // 10)
    for epoch in range(epochs):
        optimizer_.zero_grad()
        c_pred = predictor(w_train)
        loss = dfl_loss_spoplus(c_pred, c_train, y_train, z_train, spo_plus_loss)
        if (epoch + 1) % print_every == 0 or epoch == epochs - 1:
            print(f"[DFL-SPO+] epoch {epoch + 1}/{epochs} loss: {loss.item():.6f}")
        loss.backward()
        optimizer_.step()
    return predictor

def train_dfl_predictor(predictor, w_train, c_train, y_train, z_train, epochs=N_EPOCHS, lr=LR_DEFAULT, seed=SEED, method="cvx"):
    if method == "cvx":
        return train_dfl_predictor_cvx(
            predictor, w_train, c_train, y_train, z_train, epochs=epochs, lr=lr, seed=seed
        )
    if method == "spo":
        return train_dfl_predictor_spo(
            predictor, w_train, c_train, y_train, z_train, epochs=epochs, lr=lr, seed=seed
        )
    raise ValueError(f"Unknown DFL training method '{method}'. Expected 'cvx' or 'spo'.")

def toy_example_dfl(seed=SEED, return_initialized=False, method="cvx"):
    set_seed(seed)
    predictor = new_predictor()
    predictor_init = copy.deepcopy(predictor) if return_initialized else None
    # predictor = toy_example_pfl(seed=seed)
    w_values = W_TRAIN  # training features
    w_train, c_train, y_train, z_train = build_toy_dataset_dfl(w_values)
    train_dfl_predictor(predictor, w_train, c_train, y_train, z_train, method=method, seed=seed)
    if return_initialized:
        return predictor_init, predictor
    return predictor

def test_dfl_predictor(
    seed=SEED,
    num_points_per_1pu=100,
    save_path=None,
    show=True,
    comparison_view="all"
):
    aliases = {
        "init+cvxpylayers trained": "init_cvx",
        "init+spo+ trained": "init_spo",
        "init, cvxpylayers trained, spo+ trained": "all",
    }
    comparison_view = aliases.get(comparison_view, comparison_view)
    valid_views = {"init_cvx", "init_spo", "all"}
    if comparison_view not in valid_views:
        raise ValueError(
            f"Unknown comparison_view '{comparison_view}'. "
            f"Expected one of {sorted(valid_views)}."
        )

    set_seed(seed)
    pred_dfl_init = new_predictor()
    need_cvx = comparison_view in {"init_cvx", "all"}
    need_spo = comparison_view in {"init_spo", "all"}
    pred_dfl_cvx = copy.deepcopy(pred_dfl_init) if need_cvx else None
    pred_dfl_spo = copy.deepcopy(pred_dfl_init) if need_spo else None

    w_train, c_train, y_train, z_train = build_toy_dataset_dfl(W_TRAIN)
    if need_cvx:
        train_dfl_predictor_cvx(pred_dfl_cvx, w_train, c_train, y_train, z_train, seed=seed)
    if need_spo:
        train_dfl_predictor_spo(pred_dfl_spo, w_train, c_train, y_train, z_train, seed=seed)

    num_points = num_points_per_1pu*2*int(SAMPLE_MAX)+1
    w_sweep = torch.linspace(-SAMPLE_MAX, SAMPLE_MAX, steps=num_points).unsqueeze(-1)
    c_sweep_true = feature_cost_mapping(w_sweep)
    with torch.no_grad():
        c_sweep_pred_init = pred_dfl_init(w_sweep)
        c_sweep_pred_cvx = pred_dfl_cvx(w_sweep) if need_cvx else None
        c_sweep_pred_spo = pred_dfl_spo(w_sweep) if need_spo else None
    y_sweep_true = optimizer(c_sweep_true)
    y_sweep_pred_init = optimizer(c_sweep_pred_init)
    y_sweep_pred_cvx = optimizer(c_sweep_pred_cvx) if need_cvx else None
    y_sweep_pred_spo = optimizer(c_sweep_pred_spo) if need_spo else None

    if comparison_view == "init_cvx":
        plot_dfl_init_vs_trained(
            w_sweep,
            c_sweep_true,
            c_sweep_pred_init,
            c_sweep_pred_cvx,
            y_sweep_true,
            y_sweep_pred_init,
            y_sweep_pred_cvx,
            save_path=save_path,
            show=show,
            # data_train=(W_TRAIN, c_train),
            title_init="Initialized Model",
            title_trained="CVXPYLayers-Trained Model",
            label_init="Init",
            label_trained="CVXPY",
            color_init="tab:gray",
            color_trained="tab:blue",
        )
    elif comparison_view == "init_spo":
        plot_dfl_init_vs_trained(
            w_sweep,
            c_sweep_true,
            c_sweep_pred_init,
            c_sweep_pred_spo,
            y_sweep_true,
            y_sweep_pred_init,
            y_sweep_pred_spo,
            save_path=save_path,
            show=show,
            # data_train=(W_TRAIN, c_train),
            title_init="Initialized Model",
            title_trained="SPO+-Trained Model",
            label_init="Init",
            label_trained="SPO+",
            color_init="tab:gray",
            color_trained="tab:orange",
        )
    else:
        plot_dfl_init_vs_cvx_vs_spo(
            w_sweep,
            c_sweep_true,
            c_sweep_pred_init,
            c_sweep_pred_cvx,
            c_sweep_pred_spo,
            y_sweep_true,
            y_sweep_pred_init,
            y_sweep_pred_cvx,
            y_sweep_pred_spo,
            save_path=save_path,
            show=show,
            # data_train=(W_TRAIN, c_train),
        )

    w = W_TEST  # test features
    c_true = feature_cost_mapping(w) # true test costs
    with torch.no_grad():
        c_pred_cvx = pred_dfl_cvx(w) if need_cvx else None
        c_pred_spo = pred_dfl_spo(w) if need_spo else None
    intd = interdictor(c_true) # interdictions
    c_intd_true = c_true + intd # true interdicted costs
    c_intd_pred_cvx = c_pred_cvx + intd if need_cvx else None
    c_intd_pred_spo = c_pred_spo + intd if need_spo else None
    y_true = optimizer(c_intd_true) # true interdicted opt. sol.
    y_pred_cvx = optimizer(c_intd_pred_cvx) if need_cvx else None
    y_pred_spo = optimizer(c_intd_pred_spo) if need_spo else None
    print("="*30)
    print(f"Test DFL Predictor Results (comparison_view={comparison_view}):")
    print("="*30)
    print(f"Features:\n{w}")
    print(f"True costs:\n{c_true}")
    if need_cvx:
        print(f"Predicted costs (CVXPY):\n{c_pred_cvx}")
    if need_spo:
        print(f"Predicted costs (SPO+):\n{c_pred_spo}")
    print(f"Interdictions:\n{intd}")
    print(f"True interdicted costs:\n{c_intd_true}")
    if need_cvx:
        print(f"Predicted interdicted costs (CVXPY):\n{c_intd_pred_cvx}")
    if need_spo:
        print(f"Predicted interdicted costs (SPO+):\n{c_intd_pred_spo}")
    print(f"True interdicted opt. sol.:\n{y_true}")
    if need_cvx:
        print(f"Predicted interdicted opt. sol. (CVXPY):\n{y_pred_cvx}")
    if need_spo:
        print(f"Predicted interdicted opt. sol. (SPO+):\n{y_pred_spo}")
    pass

def train_adfl_predictor(predictor, w_train, c_train, y_train, z_train, i_train, epochs=N_EPOCHS, lr=LR_DEFAULT, seed=SEED):
    set_seed(seed)
    predictor.train()
    path_layer = _get_cvxpy_path_layer()
    optimizer_ = torch.optim.Adam(predictor.parameters(), lr=lr)
    print_every = max(1, epochs // 10)
    for epoch in range(epochs):
        # Train without interdiction
        optimizer_.zero_grad()
        c_pred = predictor(w_train)
        loss = dfl_loss_cvxpy(c_pred, c_train, path_layer)
        loss.backward()
        optimizer_.step()
        # Train with interdiction
        optimizer_.zero_grad()
        c_intd_true = c_train + i_train
        c_intd_pred = predictor(w_train) + i_train
        loss_intd = dfl_loss_cvxpy(c_intd_pred, c_intd_true, path_layer)
        if (epoch + 1) % print_every == 0 or epoch == epochs - 1:
            print(
                f"[A-DFL] epoch {epoch + 1}/{epochs} "
                f"loss: {loss.item():.6f} loss_intd: {loss_intd.item():.6f}"
            )
        loss_intd.backward()
        optimizer_.step()
    return predictor

def build_toy_dataset_adfl(w_values):
    c = feature_cost_mapping(w_values)
    y = optimizer(c)
    z = sol_value(c, y)
    i = interdictor(c)
    return w_values, c, y, z, i

def toy_example_adfl(seed=SEED):
    set_seed(seed)
    predictor = new_predictor()
    w_values = W_TRAIN  # training features
    w_train, c_train, y_train, z_train, i_train = build_toy_dataset_adfl(w_values)
    train_adfl_predictor(predictor, w_train, c_train, y_train, z_train, i_train, seed=seed)
    return predictor

def test_adfl_predictor(seed=SEED):
    pred_adfl = toy_example_adfl(seed=seed)
    w = W_TEST  # test features
    c_true = feature_cost_mapping(w) # true test costs
    with torch.no_grad():
        c_pred = pred_adfl(w) # predicted test costs
    intd = interdictor(c_true) # interdictions
    c_intd_true = c_true + intd # true interdicted costs
    c_intd_pred = c_pred + intd # predicted interdicted costs
    y_true = optimizer(c_intd_true) # true interdicted opt. sol.
    y_pred = optimizer(c_intd_pred) # predicted interdicted opt. sol.
    print("="*30)
    print(f"Test A-DFL Predictor Results:")
    print("="*30)
    print(f"Features:\n{w}")
    print(f"True costs:\n{c_true}")
    print(f"Predicted costs:\n{c_pred}")
    print(f"Interdictions:\n{intd}")
    print(f"True interdicted costs:\n{c_intd_true}")
    print(f"Predicted interdicted costs:\n{c_intd_pred}")
    print(f"True interdicted opt. sol.:\n{y_true}")
    print(f"Predicted interdicted opt. sol.:\n{y_pred}")
    pass


def test_predictors_sweep(seed=SEED, num_points_per_1pu=100, save_path=None, show=True):
    pred_dfl = toy_example_dfl(seed=seed)
    pred_adfl = toy_example_adfl(seed=seed)
    num_points = num_points_per_1pu*2*int(SAMPLE_MAX)+1
    w = torch.linspace(-SAMPLE_MAX, SAMPLE_MAX, steps=num_points).unsqueeze(-1)
    c_true = feature_cost_mapping(w)
    c_train = feature_cost_mapping(W_TRAIN)
    i_train = interdictor(c_train)
    i_true = interdictor(c_true, d=D_TEST)
    with torch.no_grad():
        c_pred_dfl = pred_dfl(w)
        c_pred_adfl = pred_adfl(w)

    # Without Interdiction
    y_true = optimizer(c_true)
    y_pred_dfl = optimizer(c_pred_dfl)
    y_pred_adfl = optimizer(c_pred_adfl)
    fig_unintd = plot_predictor_sweep(
        w,
        c_true,
        c_pred_dfl,
        c_pred_adfl,
        y_true,
        y_pred_dfl,
        y_pred_adfl,
        save_path=save_path,
        show=show,
        # data_train = (W_TRAIN, c_train),
        intd=False
    )

    # With Interdiction
    y_true = optimizer(c_true+i_true)
    y_pred_dfl = optimizer(c_pred_dfl+i_true)
    y_pred_adfl = optimizer(c_pred_adfl+i_true)
    fig_intd = plot_predictor_sweep(
        w,
        c_true+i_true,
        c_pred_dfl+i_true,
        c_pred_adfl+i_true,
        y_true,
        y_pred_dfl,
        y_pred_adfl,
        save_path=save_path,
        show=show,
        # data_train = (W_TRAIN, c_train + i_train),
        intd=True
    )

    return fig_unintd, fig_intd

def test_predictors_sweep_uninterdicted(seed=SEED, num_points_per_1pu=100, save_path=None, show=True):
    pred_dfl = toy_example_dfl(seed=seed)
    pred_adfl = toy_example_adfl(seed=seed)
    num_points = num_points_per_1pu*2*int(SAMPLE_MAX)+1
    w = torch.linspace(-SAMPLE_MAX, SAMPLE_MAX, steps=num_points).unsqueeze(-1)
    c_true = feature_cost_mapping(w)
    c_train = feature_cost_mapping(W_TRAIN)
    with torch.no_grad():
        c_pred_dfl = pred_dfl(w)
        c_pred_adfl = pred_adfl(w)
    y_true = optimizer(c_true)
    y_pred_dfl = optimizer(c_pred_dfl)
    y_pred_adfl = optimizer(c_pred_adfl)

    return plot_predictor_sweep(
        w,
        c_true,
        c_pred_dfl,
        c_pred_adfl,
        y_true,
        y_pred_dfl,
        y_pred_adfl,
        save_path=save_path,
        show=show,
        # data_train = (W_TRAIN, c_train),
        intd=False
    )

def main():
    # test_predictors_sweep_uninterdicted()
    # test_predictors_sweep()
    test_dfl_predictor(comparison_view="init_cvx")
    # test_adfl_predictor()
    pass

if __name__ == "__main__":
    main()
