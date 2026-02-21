#!/usr/bin/env python3
import copy
import numpy as np
import torch
import torch.nn as nn
import pyepo
from toy_example.main_funcs import (
    feature_cost_mapping, 
    optimizer, 
    sol_value,
    interdictor
)
from toy_example.opt import ToyOptModel
from toy_example.plotting import (
    plot_predictor_sweep,
    plot_dfl_init_vs_trained,
)

# PARAMETERS
SEED = 5
N_EPOCHS = 500
SAMPLE_MAX = 6.0
D_TEST = 5.0
LR_DEFAULT = 2e-2
N_TRAIN_SAMPLES = 500

# Constants
torch.manual_seed(SEED)
W_TRAIN = (torch.rand(N_TRAIN_SAMPLES, 1)-.5)*2*SAMPLE_MAX # torch.tensor([-1.0, 1.0]).unsqueeze(-1)  # training features
W_TEST = torch.tensor([-SAMPLE_MAX, -1.0, 1.0, SAMPLE_MAX]).unsqueeze(-1)  # test features

# Utility Functions
def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)

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

def train_dfl_predictor(predictor, w_train, c_train, y_train, z_train, epochs=N_EPOCHS, lr=LR_DEFAULT, seed=SEED):
    set_seed(seed)
    predictor.train()
    opt_model = ToyOptModel(None)
    criterion = pyepo.func.SPOPlus(opt_model, processes=1)
    optimizer_ = torch.optim.Adam(predictor.parameters(), lr=lr)
    print_every = max(1, epochs // 10)
    for epoch in range(epochs):
        optimizer_.zero_grad()
        c_pred = predictor(w_train)
        loss = criterion(c_pred, c_train, y_train, z_train)
        if (epoch + 1) % print_every == 0 or epoch == epochs - 1:
            print(f"[DFL] epoch {epoch + 1}/{epochs} loss: {loss.item():.6f}")
        loss.backward()
        optimizer_.step()
    return predictor

def toy_example_dfl(seed=SEED, return_initialized=False):
    set_seed(seed)
    predictor = new_predictor()
    predictor_init = copy.deepcopy(predictor) if return_initialized else None
    # predictor = toy_example_pfl(seed=seed)
    w_values = W_TRAIN  # training features
    w_train, c_train, y_train, z_train = build_toy_dataset_dfl(w_values)
    train_dfl_predictor(predictor, w_train, c_train, y_train, z_train, seed=seed)
    if return_initialized:
        return predictor_init, predictor
    return predictor

def test_dfl_predictor(seed=SEED, num_points_per_1pu=100, save_path=None, show=True):
    pred_dfl_init, pred_dfl = toy_example_dfl(seed=seed, return_initialized=True)
    num_points = num_points_per_1pu*2*int(SAMPLE_MAX)+1
    w_sweep = torch.linspace(-SAMPLE_MAX, SAMPLE_MAX, steps=num_points).unsqueeze(-1)
    c_sweep_true = feature_cost_mapping(w_sweep)
    with torch.no_grad():
        c_sweep_pred_init = pred_dfl_init(w_sweep)
        c_sweep_pred = pred_dfl(w_sweep)
    y_sweep_true = optimizer(c_sweep_true)
    y_sweep_pred_init = optimizer(c_sweep_pred_init)
    y_sweep_pred = optimizer(c_sweep_pred)
    plot_dfl_init_vs_trained(
        w_sweep,
        c_sweep_true,
        c_sweep_pred_init,
        c_sweep_pred,
        y_sweep_true,
        y_sweep_pred_init,
        y_sweep_pred,
        save_path=save_path,
        show=show,
    )

    w = W_TEST  # test features
    c_true = feature_cost_mapping(w) # true test costs
    with torch.no_grad():
        c_pred = pred_dfl(w) # predicted test costs
    intd = interdictor(c_true) # interdictions
    c_intd_true = c_true + intd # true interdicted costs
    c_intd_pred = c_pred + intd # predicted interdicted costs
    y_true = optimizer(c_intd_true) # true interdicted opt. sol.
    y_pred = optimizer(c_intd_pred) # predicted interdicted opt. sol.
    print("="*30)
    print(f"Test DFL Predictor Results:")
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

def train_adfl_predictor(predictor, w_train, c_train, y_train, z_train, i_train, epochs=N_EPOCHS, lr=LR_DEFAULT, seed=SEED):
    set_seed(seed)
    predictor.train()
    opt_model = ToyOptModel(None)
    criterion = pyepo.func.SPOPlus(opt_model, processes=1)
    optimizer_ = torch.optim.Adam(predictor.parameters(), lr=lr)
    print_every = max(1, epochs // 10)
    for epoch in range(epochs):
        # Train without interdiction
        optimizer_.zero_grad()
        c_pred = predictor(w_train)
        loss = criterion(c_pred, c_train, y_train, z_train)
        loss.backward()
        optimizer_.step()
        # Train with interdiction
        optimizer_.zero_grad()
        c_intd_pred = predictor(w_train) + i_train
        y_intd = optimizer(c_train + i_train)
        loss_intd = criterion(c_intd_pred, c_train + i_train, y_intd, z_train)
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
    test_dfl_predictor()
    # test_adfl_predictor()
    pass

if __name__ == "__main__":
    main()
