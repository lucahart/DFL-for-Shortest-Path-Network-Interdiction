import numpy as np
import torch
import matplotlib.pyplot as plt

def _to_numpy(values):
    if torch.is_tensor(values):
        return values.detach().cpu().numpy()
    return np.asarray(values)

def _solution_match_mask(y_true, y_pred):
    if torch.is_tensor(y_true) and torch.is_tensor(y_pred):
        return torch.all(y_true == y_pred, dim=-1)
    return np.all(y_true == y_pred, axis=-1)

def _prepare_sweep_plot_data(
    w_values,
    c_true,
    c_pred_dfl,
    c_pred_adfl,
    y_true,
    y_pred_dfl,
    y_pred_adfl,
    data_train=None
):
    match_dfl = _solution_match_mask(y_true, y_pred_dfl)
    match_adfl = _solution_match_mask(y_true, y_pred_adfl)
    train_w = None
    train_c = None
    if data_train is not None:
        w_train, c_train = data_train
        train_w = _to_numpy(w_train).squeeze(-1)
        train_c = _to_numpy(c_train)
    return {
        "w": _to_numpy(w_values).squeeze(-1),
        "true": _to_numpy(c_true),
        "dfl": _to_numpy(c_pred_dfl),
        "adfl": _to_numpy(c_pred_adfl),
        "y_true": _to_numpy(y_true),
        "y_dfl": _to_numpy(y_pred_dfl),
        "y_adfl": _to_numpy(y_pred_adfl),
        "match_dfl": _to_numpy(match_dfl).astype(bool),
        "match_adfl": _to_numpy(match_adfl).astype(bool),
        "train_w": train_w,
        "train_c": train_c,
    }

def _make_sweep_figure():
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.35], hspace=0.35)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    return fig, axes

def _plot_true_costs(ax, w_np, true_np, intd=False):
    c12_true = np.sum(true_np[:, [0, 1]], axis=1)
    c34_true = np.sum(true_np[:, [2, 3]], axis=1)
    ax.plot(w_np, c12_true, color="black", linewidth=2, linestyle="-", label="c_1" + (" + d_2(c) True" if intd else " True"))
    ax.plot(w_np, c34_true, color="black", linewidth=2, linestyle="--", label="c_2" + (" + d_2(c) True" if intd else " True"))

def _plot_predicted_costs(ax, w_np, pred_np, y_pred_np, match_np, color, label_suffix, intd=False):
    c_total = np.sum(y_pred_np * pred_np, axis=1)
    c12_pred = np.sum(pred_np[:, [0, 1]], axis=1)
    c34_pred = np.sum(pred_np[:, [2, 3]], axis=1)
    ax.plot(
        w_np,
        c12_pred,
        color=color,
        alpha=0.8,
        linestyle="-",
        label=f"c_1 + d_2(c) {label_suffix}" if intd else f"c_1 {label_suffix}"
    )
    ax.plot(
        w_np,
        c34_pred,
        color=color,
        alpha=0.8,
        linestyle="--",
        label=f"c_2 + d_2(c) {label_suffix}" if intd else f"c_2 {label_suffix}"
    )
    ax.scatter(
        w_np[match_np],
        c_total[match_np],
        color="tab:green",
        s=12,
        alpha=0.7,
        marker="o",
    )
    ax.scatter(
        w_np[~match_np],
        c_total[~match_np],
        color="tab:red",
        s=12,
        alpha=0.7,
        marker="x",
    )

def _plot_train_lines(ax, w_train_np):
    if w_train_np is None or w_train_np.size == 0:
        return
    for w in np.unique(w_train_np):
        ax.axvline(
            w,
            color="tab:gray",
            alpha=0.2,
            linewidth=2.0,
            zorder=0
        )

def _plot_cost_panel(ax, plot_data, pred_key, y_key, match_key, color, title, label_suffix, intd=False):
    _plot_train_lines(ax, plot_data["train_w"])
    _plot_true_costs(ax, plot_data["w"], plot_data["true"], intd=intd)
    _plot_predicted_costs(
        ax,
        plot_data["w"],
        plot_data[pred_key],
        plot_data[y_key],
        plot_data[match_key],
        color,
        label_suffix,
        intd=intd
    )
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    ax.set_xlabel("w")

def _plot_agreement_panel(ax, w_np, match_np, label, w_train_np=None):
    _plot_train_lines(ax, w_train_np)
    ax.scatter(
        w_np,
        np.full_like(w_np, 0.0),
        c=np.where(match_np, "tab:green", "tab:red"),
        s=14,
        marker="o",
        alpha=0.8
    )
    ax.set_yticks([0.0])
    ax.set_yticklabels([label])
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel("w")
    ax.set_title("Solution agreement (green=match, red=mismatch)")
    ax.grid(axis="x", alpha=0.2)

def plot_predictor_sweep(
    w_values,
    c_true,
    c_pred_dfl,
    c_pred_adfl,
    y_true,
    y_pred_dfl,
    y_pred_adfl,
    save_path=None,
    show=True,
    data_train=None,
    intd=False
):
    plot_data = _prepare_sweep_plot_data(
        w_values,
        c_true,
        c_pred_dfl,
        c_pred_adfl,
        y_true,
        y_pred_dfl,
        y_pred_adfl,
        data_train=data_train
    )

    fig, axes = _make_sweep_figure()
    _plot_cost_panel(
        axes[0],
        plot_data,
        pred_key="dfl",
        y_key="y_dfl",
        match_key="match_dfl",
        color="tab:blue",
        title="DFL Predictions" + (" with Interdiction" if intd else " without Interdiction"),
        label_suffix="DFL",
        intd=intd
    )
    _plot_cost_panel(
        axes[1],
        plot_data,
        pred_key="adfl",
        y_key="y_adfl",
        match_key="match_adfl",
        color="tab:orange",
        title="A-DFL Predictions" + (" with Interdiction" if intd else " without Interdiction"),
        label_suffix="A-DFL",
        intd=intd
    )
    axes[0].set_ylabel("Cost")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[1].legend(loc="upper left", fontsize=9)

    _plot_agreement_panel(
        axes[2],
        plot_data["w"],
        plot_data["match_dfl"],
        "DFL",
        w_train_np=plot_data["train_w"]
    )
    _plot_agreement_panel(
        axes[3],
        plot_data["w"],
        plot_data["match_adfl"],
        "A-DFL",
        w_train_np=plot_data["train_w"]
    )

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved sweep plot to {save_path}")
    if show:
        plt.show()

    return fig

def plot_dfl_init_vs_trained(
    w_values,
    c_true,
    c_pred_init,
    c_pred_trained,
    y_true,
    y_pred_init,
    y_pred_trained,
    save_path=None,
    show=True,
    data_train=None,
    intd=False
):
    plot_data = _prepare_sweep_plot_data(
        w_values,
        c_true,
        c_pred_init,
        c_pred_trained,
        y_true,
        y_pred_init,
        y_pred_trained,
        data_train=data_train
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    _plot_cost_panel(
        axes[0],
        plot_data,
        pred_key="dfl",
        y_key="y_dfl",
        match_key="match_dfl",
        color="tab:gray",
        title="Initialized Model" + (" with Interdiction" if intd else ""),
        label_suffix="Init",
        intd=intd
    )
    _plot_cost_panel(
        axes[1],
        plot_data,
        pred_key="adfl",
        y_key="y_adfl",
        match_key="match_adfl",
        color="tab:blue",
        title="Trained Model" + (" with Interdiction" if intd else ""),
        label_suffix="Trained",
        intd=intd
    )

    axes[0].set_ylabel("Cost")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[1].legend(loc="upper left", fontsize=9)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved init-vs-trained plot to {save_path}")
    if show:
        plt.show()

    return fig
