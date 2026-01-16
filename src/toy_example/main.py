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

def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)

def build_toy_dataset_pfl(w_values):
    c = feature_cost_mapping(w_values)
    return w_values, c

def train_pfl_predictor(predictor, x_train, y_train, epochs=200, lr=1e-2, seed=0):
    set_seed(seed)
    predictor.train()
    criterion = nn.MSELoss()
    optimizer_ = torch.optim.Adam(predictor.parameters(), lr=lr)
    for _ in range(epochs):
        optimizer_.zero_grad()
        preds = predictor(x_train)
        loss = criterion(preds, y_train)
        loss.backward()
        optimizer_.step()
    return predictor

def toy_example_pfl(seed=0):
    set_seed(seed)
    predictor = nn.Sequential(
        nn.Linear(1, 4),
        nn.ReLU(),
        nn.Linear(4, 4)
    )
    w_values = torch.tensor([-2.0, -1.0, 0.5, 1.0, 2.0, 5.0]).unsqueeze(-1)
    x_train, y_train = build_toy_dataset_pfl(w_values)
    train_pfl_predictor(predictor, x_train, y_train, seed=seed)
    return predictor

def test_pfl_predictor(seed=0):
    pred_pfl = toy_example_pfl(seed=seed)
    w = torch.tensor([-1.5, 1.0, 3.0]).unsqueeze(-1) # test features
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
    for _ in range(epochs):
        optimizer_.zero_grad()
        c_pred = predictor(w_train)
        loss = criterion(c_pred, c_train, y_train, z_train)
        loss.backward()
        optimizer_.step()
    return predictor

def build_toy_dataset_dfl(w_values):
    c = feature_cost_mapping(w_values)
    y = optimizer(c)
    z = sol_value(c, y)
    return w_values, c, y, z

def toy_example_dfl(seed=0):
    set_seed(seed)
    predictor = nn.Sequential(
        nn.Linear(1, 4),
        nn.ReLU(),
        nn.Linear(4, 4)
    )
    w_values = torch.tensor([-5.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 5.0]).unsqueeze(-1)
    w_train, c_train, y_train, z_train = build_toy_dataset_dfl(w_values)
    train_dfl_predictor(predictor, w_train, c_train, y_train, z_train, seed=seed)
    return predictor

def test_dfl_predictor(seed=0):
    pred_dfl = toy_example_dfl(seed=seed)
    w = torch.tensor([-1.5, 1.0, 3.0]).unsqueeze(-1) # test features
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

def train_adfl_predictor():
    pass

def main():
    # test_pfl_predictor()
    test_dfl_predictor()

if __name__ == "__main__":
    main()
