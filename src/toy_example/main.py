#!/usr/bin/env python3
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

# Constants
W_TRAIN = torch.tensor([-1.0, 1.0]).unsqueeze(-1)  # training features
W_TEST = torch.tensor([-3.0, -1.0, 1.0, 3.0]).unsqueeze(-1)  # test features

# Utility Functions
def set_seed(seed=0):
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
        nn.ReLU(),
        nn.Linear(4, 4)
    )
    # predictor.apply(lambda m: weights_init(m, value=1.0))
    with torch.no_grad():
        predictor[0].weight = torch.nn.Parameter(torch.tensor([1, -2, -1, 2], dtype=torch.float).unsqueeze(1))
        predictor[0].bias.zero_()
        predictor[2].weight = torch.nn.Parameter(torch.tensor([[1.0, 2.0, 0.0, 0.0], [1, 2, 0, 0], [0, 0, 1, 2], [0, 0, 1, 2]], dtype=torch.float))
        predictor[2].bias.zero_()
    return predictor

# Learning logic
def build_toy_dataset_pfl(w_values):
    c = feature_cost_mapping(w_values)
    return w_values, c

def train_pfl_predictor(predictor, x_train, y_train, epochs=200, lr=1e-2, seed=0):
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

def toy_example_pfl(seed=0):
    set_seed(seed)
    predictor = new_predictor()
    w_values = W_TRAIN  # training features
    x_train, y_train = build_toy_dataset_pfl(w_values)
    train_pfl_predictor(predictor, x_train, y_train, seed=seed)
    return predictor

def test_pfl_predictor(seed=0):
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

def train_dfl_predictor(predictor, w_train, c_train, y_train, z_train, epochs=200, lr=1e-2, seed=0):
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

def toy_example_dfl(seed=0):
    set_seed(seed)
    predictor = new_predictor()
    # predictor = toy_example_pfl(seed=seed)
    w_values = W_TRAIN  # training features
    w_train, c_train, y_train, z_train = build_toy_dataset_dfl(w_values)
    train_dfl_predictor(predictor, w_train, c_train, y_train, z_train, seed=seed)
    return predictor

def test_dfl_predictor(seed=0):
    pred_dfl = toy_example_dfl(seed=seed)
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

def train_adfl_predictor(predictor, w_train, c_train, y_train, z_train, i_train, epochs=200, lr=1e-2, seed=0):
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

def toy_example_adfl(seed=0):
    set_seed(seed)
    predictor = new_predictor()
    w_values = W_TRAIN  # training features
    w_train, c_train, y_train, z_train, i_train = build_toy_dataset_adfl(w_values)
    train_adfl_predictor(predictor, w_train, c_train, y_train, z_train, i_train, seed=seed)
    return predictor

def test_adfl_predictor(seed=0):
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

def main():
    test_dfl_predictor()
    test_adfl_predictor()

if __name__ == "__main__":
    main()
