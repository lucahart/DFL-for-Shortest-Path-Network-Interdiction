#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn


def feature_cost_mapping(w):
    """
    Docstring for feature_cost_mapping

    :param w: Description
    :return: Description
    """
    if torch.is_tensor(w):
        w_t = w.squeeze(-1) if w.ndim > 0 and w.shape[-1] == 1 else w
        c_pos = torch.stack((w_t, w_t, 2 * w_t, 2 * w_t), dim=-1)
        c_neg = torch.stack((-2 * w_t, -2 * w_t, -w_t, -w_t), dim=-1)
        return torch.where((w_t >= 0).unsqueeze(-1), c_pos, c_neg)
    if isinstance(w, np.ndarray):
        w_arr = np.squeeze(w, axis=-1) if w.ndim > 0 and w.shape[-1] == 1 else w
        c_pos = np.stack((w_arr, w_arr, 2 * w_arr, 2 * w_arr), axis=-1)
        c_neg = np.stack((-2 * w_arr, -2 * w_arr, -w_arr, -w_arr), axis=-1)
        return np.where(w_arr[..., None] >= 0, c_pos, c_neg)
    if w >= 0:
        return [w, w, 2 * w, 2 * w]
    return [-2 * w, -2 * w, -w, -w]

def optimizer(c):
    """
    Docstring for optimizer
    
    :param c: Description
    :return: Description
    """
    if torch.is_tensor(c):
        lhs = c[..., 0] + c[..., 1]
        rhs = c[..., 2] + c[..., 3]
        cond = lhs <= rhs
        ones = torch.ones_like(lhs)
        zeros = torch.zeros_like(lhs)
        y_left = torch.stack((ones, ones, zeros, zeros), dim=-1)
        y_right = torch.stack((zeros, zeros, ones, ones), dim=-1)
        return torch.where(cond.unsqueeze(-1), y_left, y_right)
    if isinstance(c, np.ndarray):
        lhs = c[..., 0] + c[..., 1]
        rhs = c[..., 2] + c[..., 3]
        cond = lhs <= rhs
        ones = np.ones_like(lhs)
        zeros = np.zeros_like(lhs)
        y_left = np.stack((ones, ones, zeros, zeros), axis=-1)
        y_right = np.stack((zeros, zeros, ones, ones), axis=-1)
        return np.where(cond[..., None], y_left, y_right)
    if c[0] + c[1] <= c[2] + c[3]:
        return [1, 1, 0, 0]
    return [0, 0, 1, 1]
    
def interdictor(c):
    """
    Docstring for interdictor
    
    :param c: Description
    :return: Description
    """
    if torch.is_tensor(c):
        lhs = c[..., 0] + c[..., 1]
        rhs = c[..., 2] + c[..., 3]
        cond = lhs <= rhs
        zeros = torch.zeros_like(lhs)
        threes = torch.full_like(lhs, 3)
        g_left = torch.stack((zeros, threes, zeros, zeros), dim=-1)
        g_right = torch.stack((zeros, zeros, zeros, threes), dim=-1)
        return torch.where(cond.unsqueeze(-1), g_left, g_right)
    if isinstance(c, np.ndarray):
        lhs = c[..., 0] + c[..., 1]
        rhs = c[..., 2] + c[..., 3]
        cond = lhs <= rhs
        zeros = np.zeros_like(lhs)
        threes = np.full_like(lhs, 3)
        g_left = np.stack((zeros, threes, zeros, zeros), axis=-1)
        g_right = np.stack((zeros, zeros, zeros, threes), axis=-1)
        return np.where(cond[..., None], g_left, g_right)
    if c[0] + c[1] <= c[2] + c[3]:
        return [0, 3, 0, 0]
    return [0, 0, 0, 3]
    
def test_intd_pipeline_volatile():
    w = -1
    print(f"Input                w:  {w}")
    c = feature_cost_mapping(w)
    print(f"Cost                 c: {c}")
    y = optimizer(c)
    print(f"Opt. sol.            y: {y}")
    g_c = interdictor(c)
    print(f"Intd.              g_c: {g_c}")
    c_intd = [c[i] + g_c[i] for i in range(len(c))]
    print(f"Intd. cost     c + g_c: {c_intd}")
    y_intd = optimizer(c_intd)
    print(f"Intd. opt. sol. y_intd: {y_intd}")
    pass

def toy_example_pfl():
    predictor = nn.Sequential(
        nn.Linear(1, 4),
        nn.ReLU(),
        nn.Linear(4, 4)
    )
    w_values = torch.tensor([-2.0, -1.0, 0.5, 1.0, 2.0, 5.0]).unsqueeze(-1)
    x_train, y_train = build_toy_dataset_pfl(w_values)
    train_pfl_predictor(predictor, x_train, y_train)
    return predictor

def test_pfl_predictor():
    pred_pfl = toy_example_pfl()
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


def build_toy_dataset_pfl(w_values):
    c = feature_cost_mapping(w_values)
    return w_values, c

def train_pfl_predictor(predictor, x_train, y_train, epochs=200, lr=1e-2):
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

def train_dfl_predictor():

    pass

def train_adfl_predictor():
    pass

def main():
    test_pfl_predictor()

if __name__ == "__main__":
    main()
