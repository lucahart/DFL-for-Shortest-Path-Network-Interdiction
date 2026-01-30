#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import pyepo
import matplotlib.pyplot as plt
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

def _to_numpy(values):
    if torch.is_tensor(values):
        return values.detach().cpu().numpy()
    return np.asarray(values)

def _solution_match_mask(y_true, y_pred):
    if torch.is_tensor(y_true) and torch.is_tensor(y_pred):
        return torch.all(y_true == y_pred, dim=-1)
    return np.all(y_true == y_pred, axis=-1)

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

def plot_predictor_sweep(
    w_values,
    c_true,
    c_pred_dfl,
    c_pred_adfl,
    y_true,
    y_pred_dfl,
    y_pred_adfl,
    save_path=None,
    show=True
):
    
    match_dfl = _solution_match_mask(y_true, y_pred_dfl)
    match_adfl = _solution_match_mask(y_true, y_pred_adfl)
    w_np = _to_numpy(w_values).squeeze(-1)
    true_np = _to_numpy(c_true)
    dfl_np = _to_numpy(c_pred_dfl)
    adfl_np = _to_numpy(c_pred_adfl)
    y_np = _to_numpy(y_true)
    y_dfl_np = _to_numpy(y_pred_dfl)
    y_adfl_np = _to_numpy(y_pred_adfl)
    match_dfl_np = _to_numpy(match_dfl).astype(bool)
    match_adfl_np = _to_numpy(match_adfl).astype(bool)

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.35], hspace=0.35)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    # match_ax = fig.add_subplot(gs[1, :])

    for idx, ax in enumerate(axes):
        if idx >= 2:
            continue
        # Plot true costs
        c12_true = np.sum(true_np[:, [0, 1]], axis=1)
        c34_true = np.sum(true_np[:, [2, 3]], axis=1)
        ax.plot(w_np, c12_true, color="black", linewidth=2, linestyle="-", label="c_1 + c_2 + d_2 True")
        ax.plot(w_np, c34_true, color="black", linewidth=2, linestyle="--", label="c_3 + c_4 + d_4 True")
        # Plot predicted costs
        if idx == 0:
            c_dfl = np.sum(y_dfl_np * dfl_np, axis=1)
            ax.plot(w_np, c_dfl, color="tab:blue", alpha=0.8, label="DFL")
            ax.scatter(
                w_np[match_dfl_np],
                c_dfl[match_dfl_np],
                color="tab:green",
                s=12,
                alpha=0.7,
                marker="o",
                label="DFL Match"
            )
            ax.scatter(
                w_np[~match_dfl_np],
                c_dfl[~match_dfl_np],
                color="tab:red",
                s=12,
                alpha=0.7,
                marker="x"
            )
            ax.set_title(f"DFL Predictions")
        else:
            c_adfl = np.sum(y_adfl_np * adfl_np, axis=1)
            ax.plot(w_np, c_adfl, color="tab:orange", alpha=0.8, label="A-DFL")
            ax.scatter(
                w_np[match_adfl_np],
                c_adfl[match_adfl_np],
                color="tab:green",
                s=14,
                alpha=0.7,
                marker="o",
                label="A-DFL Match"
            )
            ax.scatter(
                w_np[~match_adfl_np],
                c_adfl[~match_adfl_np],
                color="tab:red",
                s=14,
                alpha=0.7,
                marker="x"
            )
            ax.set_title(f"A-DFL Predictions")
        ax.grid(axis="y", alpha=0.2)
        if idx in (2, 3):
            ax.set_xlabel("w")
        if idx in (0, 2):
            ax.set_ylabel("Cost")

    axes[0].legend(loc="upper left", fontsize=9)
    axes[1].legend(loc="upper left", fontsize=9)

    axes[2].scatter(
        w_np,
        np.full_like(w_np, 0.0),
        c=np.where(match_dfl_np, "tab:green", "tab:red"),
        s=14,
        marker="o",
        alpha=0.8
    )
    axes[2].set_yticks([0.0])
    axes[2].set_yticklabels(["DFL"])
    axes[2].set_ylim(-0.5, 1.5)
    axes[2].set_xlabel("w")
    axes[2].set_title("Solution agreement (green=match, red=mismatch)")
    axes[2].grid(axis="x", alpha=0.2)

    axes[3].scatter(
        w_np,
        np.full_like(w_np, 0.0),
        c=np.where(match_adfl_np, "tab:green", "tab:red"),
        s=16,
        marker="o",
        alpha=0.8
    )
    axes[3].set_yticks([0.0])
    axes[3].set_yticklabels(["A-DFL"])
    axes[3].set_ylim(-0.5, 1.5)
    axes[3].set_xlabel("w")
    axes[3].set_title("Solution agreement (green=match, red=mismatch)")
    axes[3].grid(axis="x", alpha=0.2)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved sweep plot to {save_path}")
    if show:
        plt.show()

    return fig

def test_predictors_sweep(seed=0, num_points=601, save_path=None, show=True):
    pred_dfl = toy_example_dfl(seed=seed)
    pred_adfl = toy_example_adfl(seed=seed)
    w = torch.linspace(-3.0, 3.0, steps=num_points).unsqueeze(-1)
    c_true = feature_cost_mapping(w)
    i_true = interdictor(c_true)
    with torch.no_grad():
        c_pred_dfl = pred_dfl(w)
        c_pred_adfl = pred_adfl(w)
    y_true = optimizer(c_true+i_true)
    y_pred_dfl = optimizer(c_pred_dfl+i_true)
    y_pred_adfl = optimizer(c_pred_adfl+i_true)

    return plot_predictor_sweep(
        w,
        c_true+i_true,
        c_pred_dfl+i_true,
        c_pred_adfl+i_true,
        y_true,
        y_pred_dfl,
        y_pred_adfl,
        save_path=save_path,
        show=show
    )

def main():
    test_predictors_sweep()
    # test_dfl_predictor()
    # test_adfl_predictor()

if __name__ == "__main__":
    main()
