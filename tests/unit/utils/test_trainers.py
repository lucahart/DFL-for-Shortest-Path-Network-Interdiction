from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

import dflintdpy.utils.base_trainer as base_trainer_module
from dflintdpy.utils.dfl_trainer import DFLTrainer
from dflintdpy.utils.pfl_trainer import PFLTrainer

import pytest


pytestmark = [pytest.mark.unit, pytest.mark.torch, pytest.mark.pyepo]


################
### Fixtures ###
################


@pytest.fixture
def scalar_model() -> "_ScalarModel":
    """Return a minimal single-parameter predictor."""
    return _ScalarModel(init_weight=0.0)


@pytest.fixture
def optimizer(scalar_model: "_ScalarModel") -> torch.optim.Optimizer:
    """Return a simple SGD optimizer for the scalar model."""
    return torch.optim.SGD(scalar_model.parameters(), lr=0.25)


############################
### Helper functionality ###
############################


class _ScalarModel(nn.Module):
    """Single-parameter model used to make optimization predictable."""

    def __init__(self, init_weight: float):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[init_weight]], dtype=torch.float))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Broadcast the learned scalar to the batch shape."""
        return self.weight.expand(feats.shape[0], 1)


class _BatchLoader:
    """Small loader stub with explicit mode tracking."""

    def __init__(self, batches: list[tuple[torch.Tensor, ...]]):
        self._batches = batches
        self.dataset = list(range(len(batches)))
        self.mode = "adverse"
        self.normal_mode_calls = 0
        self.adverse_mode_calls = 0

    def __iter__(self):
        """Yield the same deterministic batches on every pass."""
        return iter(self._batches)

    def normal_mode(self) -> None:
        """Record a transition into normal mode."""
        self.mode = "normal"
        self.normal_mode_calls += 1

    def adverse_mode(self) -> None:
        """Record a transition into adverse mode."""
        self.mode = "adverse"
        self.adverse_mode_calls += 1


class _ScalarSpoLoss(nn.Module):
    """Mean-squared loss with the SPO+ call signature."""

    def forward(
        self,
        costs_pred: torch.Tensor,
        costs: torch.Tensor,
        sols: torch.Tensor,
        objs: torch.Tensor,
    ) -> torch.Tensor:
        """Compare predicted and observed costs."""
        del sols, objs
        return torch.mean((costs_pred - costs) ** 2)


class _RecordingAxis:
    """Capture plotting calls without using a real matplotlib axis."""

    def __init__(self):
        self.plot_calls: list[tuple[tuple, dict]] = []
        self.scatter_calls: list[tuple[tuple, dict]] = []
        self.xlabel = None
        self.ylabel = None
        self.yscale = None
        self.title = None
        self.legend_calls = 0

    def plot(self, *args, **kwargs):
        """Record a line plot call."""
        self.plot_calls.append((args, kwargs))

    def scatter(self, *args, **kwargs):
        """Record a scatter plot call."""
        self.scatter_calls.append((args, kwargs))

    def set_xlabel(self, value):
        """Record the x-axis label."""
        self.xlabel = value

    def set_ylabel(self, value):
        """Record the y-axis label."""
        self.ylabel = value

    def set_yscale(self, value):
        """Record the y-axis scale."""
        self.yscale = value

    def set_title(self, value):
        """Record the plot title."""
        self.title = value

    def legend(self):
        """Record that a legend was requested."""
        self.legend_calls += 1


class _RecordingLoss(nn.Module):
    """Loss stub that records the arity used by a dispatch branch."""

    def __init__(self):
        super().__init__()
        self.call_sizes: list[int] = []

    def forward(self, *args):
        """Record the number of dispatched arguments."""
        self.call_sizes.append(len(args))
        return torch.tensor(float(len(args)))


class _TrainerPlotStub:
    """Minimal trainer stub for plotting tests."""

    def __init__(self, n_epochs: int):
        self.n_epochs = n_epochs


class _MutableOptModel:
    """Opt-model stub with nested mutable state for deepcopy checks."""

    def __init__(self):
        self.meta = {
            "thresholds": [1.0, 2.0],
            "flags": {"enabled": True},
        }


def _make_pfl_loader(cost_value: float = 2.0) -> _BatchLoader:
    """Build a one-batch loader for PFL trainer tests."""
    return _BatchLoader(
        [
            (
                torch.ones((1, 1), dtype=torch.float),
                torch.tensor([[cost_value]], dtype=torch.float),
                torch.zeros((1, 1), dtype=torch.float),
                torch.zeros((1, 1), dtype=torch.float),
            )
        ]
    )


def _make_dfl_loader(cost_value: float = 2.0) -> _BatchLoader:
    """Build a one-batch loader for DFL trainer tests."""
    return _BatchLoader(
        [
            (
                torch.ones((1, 1), dtype=torch.float),
                torch.tensor([[[cost_value]]], dtype=torch.float),
                torch.zeros((1, 1, 1), dtype=torch.float),
                torch.zeros((1, 1, 1), dtype=torch.float),
                torch.zeros((1, 1, 1), dtype=torch.float),
            )
        ]
    )


###############################
### test base_trainer_vis ###
###############################


def test_base_trainer_vis_learning_curve_saves_png_and_plots_testing_data(
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify that the plotting helper writes the PNG and plots test points."""
    # Arrange lightweight stand-ins for matplotlib axes and figure creation.
    train_loss_log = np.array([1.0, 0.5], dtype=float)
    train_regret_log = np.array([2.0, 1.0], dtype=float)
    test_loss_log = np.array([0.75], dtype=float)
    test_regret_log = np.array([1.5], dtype=float)
    ax1 = _RecordingAxis()
    ax2 = _RecordingAxis()
    subplots_calls: list[tuple[int, int, tuple]] = []
    savefig_calls: list[tuple[str, int, str]] = []
    tight_layout_calls = []
    close_calls = []

    def fake_subplots(*args, **kwargs):
        """Return the fake axes for plotting."""
        subplots_calls.append((args[0], args[1], tuple(sorted(kwargs.items()))))
        return object(), (ax1, ax2)

    def fake_tight_layout():
        """Record the layout call."""
        tight_layout_calls.append(True)

    def fake_savefig(file_name, *, dpi, bbox_inches):
        """Record the figure save request."""
        savefig_calls.append((file_name, dpi, bbox_inches))

    def fake_close():
        """Record that the figure was closed."""
        close_calls.append(True)

    monkeypatch.setattr(base_trainer_module.plt, "subplots", fake_subplots)
    monkeypatch.setattr(base_trainer_module.plt, "tight_layout", fake_tight_layout)
    monkeypatch.setattr(base_trainer_module.plt, "savefig", fake_savefig)
    monkeypatch.setattr(base_trainer_module.plt, "close", fake_close)
    trainer = _TrainerPlotStub(n_epochs=3)

    # Act by rendering the learning curve to the fake plotting surface.
    base_trainer_module.BaseTrainer.vis_learning_curve(
        trainer,
        train_loss_log,
        train_regret_log,
        test_loss_log=test_loss_log,
        test_regret_log=test_regret_log,
        file_name="curve",
    )

    # Assert that both panels were populated and the PNG was saved.
    assert subplots_calls == [
        (1, 2, (("figsize", (16, 4)),))
    ], "BaseTrainer did not request the expected subplot layout."
    assert savefig_calls == [
        ("curve.png", 300, "tight")
    ], "BaseTrainer did not save the learning curve as a PNG."
    assert len(tight_layout_calls) == 1, (
        "BaseTrainer did not tighten the layout before saving."
    )
    assert len(close_calls) == 1, "BaseTrainer did not close the saved figure."
    assert ax1.plot_calls[0][1]["label"] == "Training Regret", (
        "Regret panel did not plot the training trace."
    )
    assert ax1.scatter_calls[0][0][0].tolist() == [0], (
        "Regret panel did not place the testing point at epoch zero."
    )
    assert ax2.plot_calls[0][1]["label"] == "Training Loss", (
        "Loss panel did not plot the training trace."
    )
    assert ax2.scatter_calls[0][0][0].tolist() == [0], (
        "Loss panel did not place the testing point at epoch zero."
    )
    pass


################################
### test trainer_ownership ###
################################


@pytest.mark.parametrize("trainer_cls", [PFLTrainer, DFLTrainer])
def test_trainer_init_deep_copies_opt_model_and_keeps_nested_state_isolated(
    trainer_cls,
    scalar_model: "_ScalarModel",
    optimizer: torch.optim.Optimizer,
):
    """Verify that trainer construction deep-copies the opt model."""
    # Arrange a mutable opt-model stub so nested state can be inspected.
    original_opt_model = _MutableOptModel()
    trainer_kwargs = dict(
        pred_model=scalar_model,
        opt_model=original_opt_model,
        optimizer=optimizer,
        loss_fn=nn.MSELoss() if trainer_cls is PFLTrainer else _ScalarSpoLoss(),
    )
    if trainer_cls is DFLTrainer:
        trainer_kwargs["method_name"] = "spo+"

    # Act by constructing the trainer.
    trainer = trainer_cls(**trainer_kwargs)

    # Mutate the caller-owned opt model after construction.
    original_opt_model.meta["thresholds"].append(3.0)
    original_opt_model.meta["flags"]["enabled"] = False

    # Assert that the trainer kept its own copy of the opt model.
    assert trainer.opt_model is not original_opt_model, (
        f"{trainer_cls.__name__} stored the original opt model instance."
    )
    assert trainer.opt_model.meta["thresholds"] == [1.0, 2.0], (
        f"{trainer_cls.__name__} did not isolate nested opt-model state."
    )
    assert trainer.opt_model.meta["flags"]["enabled"] is True, (
        f"{trainer_cls.__name__} shared nested opt-model state by reference."
    )
    pass


def test_pfl_trainer_train_epoch_updates_original_pred_model_in_place(
    scalar_model: "_ScalarModel",
    optimizer: torch.optim.Optimizer,
):
    """Verify that PFL training mutates the caller-owned predictor."""
    # Arrange a trainer with a single batch that produces a non-zero loss.
    train_loader = _make_pfl_loader(cost_value=2.0)
    trainer = PFLTrainer(
        pred_model=scalar_model,
        opt_model=SimpleNamespace(),
        optimizer=optimizer,
        loss_fn=nn.MSELoss(),
    )

    # Act by running one optimization step.
    trainer.train_epoch(train_loader)

    # Assert the original model instance was updated in place.
    assert trainer.pred_model is scalar_model, (
        "PFLTrainer copied the predictor instead of keeping the original object."
    )
    assert scalar_model.weight.item() == pytest.approx(1.0), (
        "PFLTrainer did not update the original predictor in place."
    )
    pass


def test_dfl_trainer_train_epoch_updates_original_pred_model_in_place():
    """Verify that DFL training mutates the caller-owned predictor."""
    # Arrange a DFL trainer with a single batch that produces a non-zero loss.
    scalar_model = _ScalarModel(init_weight=0.0)
    trainer = DFLTrainer(
        pred_model=scalar_model,
        opt_model=SimpleNamespace(),
        optimizer=torch.optim.SGD(scalar_model.parameters(), lr=0.25),
        loss_fn=_ScalarSpoLoss(),
        method_name="spo+",
    )
    train_loader = _make_dfl_loader(cost_value=2.0)

    # Act by running one optimization step.
    trainer.train_epoch(train_loader)

    # Assert the original model instance was updated in place.
    assert trainer.pred_model is scalar_model, (
        "DFLTrainer copied the predictor instead of keeping the original object."
    )
    assert scalar_model.weight.item() == pytest.approx(1.0), (
        "DFLTrainer did not update the original predictor in place."
    )
    pass


###########################
### test pfl_trainer_fit ###
###########################


def test_pfl_trainer_train_epoch_uses_normal_mode_for_loss(
    scalar_model: "_ScalarModel",
    optimizer: torch.optim.Optimizer,
):
    """Verify that PFL training switches loaders into normal mode."""
    # Arrange a one-batch loader with an exact MSE loss.
    train_loader = _BatchLoader(
        [
            (
                torch.ones((1, 1), dtype=torch.float),
                torch.tensor([[2.0]], dtype=torch.float),
                torch.zeros((1, 1), dtype=torch.float),
                torch.zeros((1, 1), dtype=torch.float),
            )
        ]
    )
    trainer = PFLTrainer(
        pred_model=scalar_model,
        opt_model=SimpleNamespace(),
        optimizer=optimizer,
        loss_fn=nn.MSELoss(),
    )

    # Act by running one training epoch.
    loss = trainer.train_epoch(train_loader)

    # Assert that the loader mode and model update match normal-mode training.
    assert train_loader.normal_mode_calls == 1, (
        "PFLTrainer did not switch the training loader into normal mode."
    )
    assert train_loader.mode == "normal", (
        "PFLTrainer left the training loader in the wrong mode."
    )
    assert loss == pytest.approx(4.0), (
        "PFLTrainer returned an unexpected training loss."
    )
    assert scalar_model.weight.item() == pytest.approx(1.0), (
        "PFLTrainer did not update the predictor as expected."
    )
    pass


def test_pfl_trainer_evaluate_uses_normal_mode_for_regret(
    scalar_model: "_ScalarModel",
    optimizer: torch.optim.Optimizer,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify that PFL evaluation uses normal-mode data for regret."""
    # Arrange a loader and a regret stub that records the observed mode.
    eval_loader = _BatchLoader(
        [
            (
                torch.ones((1, 1), dtype=torch.float),
                torch.tensor([[2.0]], dtype=torch.float),
                torch.zeros((1, 1), dtype=torch.float),
                torch.zeros((1, 1), dtype=torch.float),
            )
        ]
    )
    regret_modes: list[str] = []

    def fake_regret(pred_model, opt_model, loader):
        """Capture the loader mode used for regret computation."""
        del pred_model, opt_model
        regret_modes.append(loader.mode)
        return 7.5

    monkeypatch.setattr(base_trainer_module.pyepo.metric, "regret", fake_regret)
    trainer = PFLTrainer(
        pred_model=scalar_model,
        opt_model=SimpleNamespace(),
        optimizer=optimizer,
        loss_fn=nn.MSELoss(),
    )

    # Act by evaluating the loader once.
    loss, regret = trainer.evaluate(eval_loader)

    # Assert that evaluation used normal mode and returned the expected values.
    assert eval_loader.normal_mode_calls == 1, (
        "PFLTrainer did not switch the evaluation loader into normal mode."
    )
    assert eval_loader.mode == "normal", (
        "PFLTrainer left the evaluation loader in the wrong mode."
    )
    assert regret_modes == ["normal"], (
        "PFLTrainer did not compute regret from normal-mode data."
    )
    assert loss == pytest.approx(4.0), (
        "PFLTrainer returned an unexpected evaluation loss."
    )
    assert regret == pytest.approx(7.5), (
        "PFLTrainer returned an unexpected regret value."
    )
    pass


def test_pfl_trainer_fit_without_validation_sets_explicit_n_epochs(
    scalar_model: "_ScalarModel",
    optimizer: torch.optim.Optimizer,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify that PFL fit handles no-validation runs and keeps n_epochs."""
    # Arrange a one-batch loader and a deterministic regret stub.
    train_loader = _BatchLoader(
        [
            (
                torch.ones((1, 1), dtype=torch.float),
                torch.tensor([[2.0]], dtype=torch.float),
                torch.zeros((1, 1), dtype=torch.float),
                torch.zeros((1, 1), dtype=torch.float),
            )
        ]
    )
    monkeypatch.setattr(
        base_trainer_module.pyepo.metric,
        "regret",
        lambda pred_model, opt_model, loader: 0.0,
    )
    trainer = PFLTrainer(
        pred_model=scalar_model,
        opt_model=SimpleNamespace(),
        optimizer=optimizer,
        loss_fn=nn.MSELoss(),
    )

    # Act by fitting without a validation loader.
    train_loss_log, train_regret_log, val_loss_log, val_regret_log = trainer.fit(
        train_loader,
        val_loader=None,
        epochs=1,
        n_epochs=3,
    )

    # Assert the explicit print cadence and no-validation return shape.
    assert trainer.n_epochs == 3, "PFLTrainer ignored the explicit n_epochs value."
    assert len(train_loss_log) == 2, "PFLTrainer did not log the initial and final training losses."
    assert len(train_regret_log) == 2, "PFLTrainer did not log the initial and final training regrets."
    assert val_loss_log is None, "PFLTrainer unexpectedly returned validation losses."
    assert val_regret_log is None, "PFLTrainer unexpectedly returned validation regrets."
    pass


#############################
### test trainer_cadence ###
#############################


def test_pfl_trainer_fit_tracks_validation_loss_every_epoch_and_regret_on_logging_cadence(
    scalar_model: "_ScalarModel",
    optimizer: torch.optim.Optimizer,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify that validation loss stays per-epoch while regret stays sparse."""
    # Arrange loaders with tags so the patched hooks can identify them.
    train_loader = _make_pfl_loader(cost_value=2.0)
    train_loader.tag = "train"
    val_loader = _make_pfl_loader(cost_value=2.0)
    val_loader.tag = "val"
    trainer = PFLTrainer(
        pred_model=scalar_model,
        opt_model=SimpleNamespace(),
        optimizer=optimizer,
        loss_fn=nn.MSELoss(),
    )
    loss_calls: list[str] = []
    regret_calls: list[str] = []

    def fake_evaluate_loss(self, loader):
        """Record which loader was used for validation loss."""
        del self
        loss_calls.append(loader.tag)
        return 4.0

    def fake_regret(pred_model, opt_model, loader):
        """Record which loader was used for regret."""
        del pred_model, opt_model
        regret_calls.append(loader.tag)
        return 0.0

    monkeypatch.setattr(PFLTrainer, "_evaluate_loss", fake_evaluate_loss)
    monkeypatch.setattr(base_trainer_module.pyepo.metric, "regret", fake_regret)

    # Act with epochs below the logging cadence.
    trainer.fit(train_loader, val_loader=val_loader, epochs=2, n_epochs=3)

    # Assert that validation loss ran every epoch while regret stayed sparse.
    assert loss_calls == ["train", "val", "val", "val"], (
        "Validation loss was not evaluated every epoch."
    )
    assert regret_calls == ["train", "val", "train", "train"], (
        "Validation regret was not restricted to the logging cadence."
    )
    pass


def test_pfl_trainer_fit_keeps_lr_when_validation_loss_improves(
    scalar_model: "_ScalarModel",
    optimizer: torch.optim.Optimizer,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify that an improving validation loss does not trigger LR decay."""
    # Arrange deterministic train and validation loaders.
    train_loader = _make_pfl_loader(cost_value=2.0)
    train_loader.tag = "train"
    val_loader = _make_pfl_loader(cost_value=2.0)
    val_loader.tag = "val"
    trainer = PFLTrainer(
        pred_model=scalar_model,
        opt_model=SimpleNamespace(),
        optimizer=optimizer,
        loss_fn=nn.MSELoss(),
    )

    def fake_evaluate(self, loader):
        """Return the initial train and validation losses."""
        del self
        if loader.tag == "train":
            return 2.0, 0.0
        return 1.0, 0.0

    val_losses = iter([0.9, 0.8])

    def fake_evaluate_loss(self, loader):
        """Return an improving validation-loss trajectory."""
        del self
        assert loader.tag == "val", \
            "Validation-loss hook received the wrong loader."
        return next(val_losses)

    monkeypatch.setattr(PFLTrainer, "evaluate", fake_evaluate)
    monkeypatch.setattr(PFLTrainer, "train_epoch", lambda self, loader: 2.0)
    monkeypatch.setattr(PFLTrainer, "_evaluate_loss", fake_evaluate_loss)
    monkeypatch.setattr(PFLTrainer, "_compute_regret", lambda self, loader: 0.0)
    monkeypatch.setattr(
        PFLTrainer,
        "_print_epoch_metrics",
        lambda self, epoch, train_loss, train_regret, val_loss=None,
        val_regret=None: None,
    )

    # Act by fitting for two epochs with a persistent train/val loss gap.
    trainer.fit(train_loader, val_loader=val_loader, epochs=2, n_epochs=1)

    # Assert that LR stayed fixed because validation kept improving.
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.25), (
        "PFLTrainer reduced the LR even though validation loss improved."
    )
    pass


def test_pfl_trainer_fit_waits_for_patience_before_reducing_lr(
    scalar_model: "_ScalarModel",
    optimizer: torch.optim.Optimizer,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify that LR reduction waits for repeated validation deterioration."""
    # Arrange deterministic train and validation loaders.
    train_loader = _make_pfl_loader(cost_value=2.0)
    train_loader.tag = "train"
    val_loader = _make_pfl_loader(cost_value=2.0)
    val_loader.tag = "val"
    trainer = PFLTrainer(
        pred_model=scalar_model,
        opt_model=SimpleNamespace(),
        optimizer=optimizer,
        loss_fn=nn.MSELoss(),
    )

    def fake_evaluate(self, loader):
        """Return matching initial losses for both loaders."""
        del self, loader
        return 1.0, 0.0

    val_losses = iter([1.12, 1.14, 1.13])

    def fake_evaluate_loss(self, loader):
        """Return a sustained validation-loss deterioration."""
        del self
        assert loader.tag == "val", \
            "Validation-loss hook received the wrong loader."
        return next(val_losses)

    monkeypatch.setattr(PFLTrainer, "evaluate", fake_evaluate)
    monkeypatch.setattr(PFLTrainer, "train_epoch", lambda self, loader: 1.0)
    monkeypatch.setattr(PFLTrainer, "_evaluate_loss", fake_evaluate_loss)
    monkeypatch.setattr(PFLTrainer, "_compute_regret", lambda self, loader: 0.0)
    monkeypatch.setattr(
        PFLTrainer,
        "_print_epoch_metrics",
        lambda self, epoch, train_loss, train_regret, val_loss=None,
        val_regret=None: None,
    )

    # Act by fitting for exactly the patience horizon.
    trainer.fit(train_loader, val_loader=val_loader, epochs=3, n_epochs=1)

    # Assert that LR was reduced exactly once after the patience window.
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.125), (
        "PFLTrainer did not wait for patience before reducing LR."
    )
    pass


###############################
### test dfl_trainer_fit ###
###############################


def test_dfl_trainer_init_rejects_unknown_method_name(
    scalar_model: "_ScalarModel",
    optimizer: torch.optim.Optimizer,
):
    """Verify that DFLTrainer rejects unsupported training methods."""
    # Arrange the constructor inputs with an invalid method name.
    kwargs = dict(
        pred_model=scalar_model,
        opt_model=SimpleNamespace(),
        optimizer=optimizer,
        loss_fn=_ScalarSpoLoss(),
        method_name="not-a-method",
    )

    # Act and assert the constructor validation error.
    with pytest.raises(ValueError, match="Unknown method name"):
        DFLTrainer(**kwargs)
    pass


def test_dfl_trainer_init_requires_cfg_for_hybrid_method(
    scalar_model: "_ScalarModel",
    optimizer: torch.optim.Optimizer,
):
    """Verify that the hybrid DFL mode requires a configuration object."""
    # Arrange a valid method name but omit the configuration object.
    kwargs = dict(
        pred_model=scalar_model,
        opt_model=SimpleNamespace(),
        optimizer=optimizer,
        loss_fn=_ScalarSpoLoss(),
        method_name="hybrid",
        cfg=None,
    )

    # Act and assert the constructor validation error.
    with pytest.raises(ValueError, match="Configuration must be provided"):
        DFLTrainer(**kwargs)
    pass


@pytest.mark.parametrize(
    "dfl_variant, pred, costs, sols, objs, expected_pred, expected_costs, "
    "expected_sols, expected_objs",
    [
        (
            "a-dfl",
            torch.tensor(
                [
                    [[0.0], [1.0], [2.0]],
                    [[3.0], [4.0], [5.0]],
                ],
                dtype=torch.float,
            ),
            torch.tensor(
                [
                    [[10.0], [11.0], [12.0]],
                    [[13.0], [14.0], [15.0]],
                ],
                dtype=torch.float,
            ),
            torch.tensor(
                [
                    [[20.0], [21.0], [22.0]],
                    [[23.0], [24.0], [25.0]],
                ],
                dtype=torch.float,
            ),
            torch.tensor(
                [
                    [[30.0], [31.0], [32.0]],
                    [[33.0], [34.0], [35.0]],
                ],
                dtype=torch.float,
            ),
            torch.tensor([[1.0], [2.0], [4.0], [5.0]], dtype=torch.float),
            torch.tensor([[11.0], [12.0], [14.0], [15.0]], dtype=torch.float),
            torch.tensor([[21.0], [22.0], [24.0], [25.0]], dtype=torch.float),
            torch.tensor([[31.0], [32.0], [34.0], [35.0]], dtype=torch.float),
        ),
        (
            "mixed",
            torch.tensor(
                [
                    [[0.0], [1.0], [2.0]],
                    [[3.0], [4.0], [5.0]],
                ],
                dtype=torch.float,
            ),
            torch.tensor(
                [
                    [[10.0], [11.0], [12.0]],
                    [[13.0], [14.0], [15.0]],
                ],
                dtype=torch.float,
            ),
            torch.tensor(
                [
                    [[20.0], [21.0], [22.0]],
                    [[23.0], [24.0], [25.0]],
                ],
                dtype=torch.float,
            ),
            torch.tensor(
                [
                    [[30.0], [31.0], [32.0]],
                    [[33.0], [34.0], [35.0]],
                ],
                dtype=torch.float,
            ),
            torch.tensor(
                [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
                dtype=torch.float,
            ),
            torch.tensor(
                [[10.0], [11.0], [12.0], [13.0], [14.0], [15.0]],
                dtype=torch.float,
            ),
            torch.tensor(
                [[20.0], [21.0], [22.0], [23.0], [24.0], [25.0]],
                dtype=torch.float,
            ),
            torch.tensor(
                [[30.0], [31.0], [32.0], [33.0], [34.0], [35.0]],
                dtype=torch.float,
            ),
        ),
        (
            "a-dfl",
            torch.tensor([[[7.0]], [[8.0]]], dtype=torch.float),
            torch.tensor([[[17.0]], [[18.0]]], dtype=torch.float),
            torch.tensor([[[27.0]], [[28.0]]], dtype=torch.float),
            torch.tensor([[[37.0]], [[38.0]]], dtype=torch.float),
            torch.tensor([[7.0], [8.0]], dtype=torch.float),
            torch.tensor([[17.0], [18.0]], dtype=torch.float),
            torch.tensor([[27.0], [28.0]], dtype=torch.float),
            torch.tensor([[37.0], [38.0]], dtype=torch.float),
        ),
    ],
)
def test_dfl_trainer_flatten_scenarios_respects_variant_selection(
    dfl_variant: str,
    pred: torch.Tensor,
    costs: torch.Tensor,
    sols: torch.Tensor,
    objs: torch.Tensor,
    expected_pred: torch.Tensor,
    expected_costs: torch.Tensor,
    expected_sols: torch.Tensor,
    expected_objs: torch.Tensor,
):
    """Verify that scenario flattening follows the configured DFL variant."""
    # Act by flattening the scenario tensors.
    flat_pred, flat_costs, flat_sols, flat_objs, k_eff = DFLTrainer._flatten_scenarios(
        pred,
        costs,
        sols,
        objs,
        dfl_variant,
    )

    # Assert that the selected scenarios and flattening order are correct.
    assert k_eff == expected_pred.shape[0] // pred.shape[0], (
        "DFLTrainer returned an unexpected number of effective scenarios."
    )
    assert torch.equal(flat_pred, expected_pred), (
        "DFLTrainer flattened predicted costs incorrectly."
    )
    assert torch.equal(flat_costs, expected_costs), (
        "DFLTrainer flattened true costs incorrectly."
    )
    assert torch.equal(flat_sols, expected_sols), (
        "DFLTrainer flattened solutions incorrectly."
    )
    assert torch.equal(flat_objs, expected_objs), (
        "DFLTrainer flattened objectives incorrectly."
    )
    pass


@pytest.mark.parametrize(
    "method_name, expected_arity",
    [
        ("spo+", 4),
        ("hybrid", 4),
        ("ptb", 2),
        ("dbb", 3),
        ("pg", 2),
    ],
)
def test_dfl_trainer_compute_loss_dispatches_by_method_name(
    method_name: str,
    expected_arity: int,
):
    """Verify that DFLTrainer dispatches loss calls by method name."""
    # Arrange a recording loss that returns the dispatched arity.
    loss_fn = _RecordingLoss()
    costs_pred = torch.tensor([[1.0]], dtype=torch.float)
    costs = torch.tensor([[2.0]], dtype=torch.float)
    sols = torch.tensor([[3.0]], dtype=torch.float)
    objs = torch.tensor([[4.0]], dtype=torch.float)

    # Act by invoking the static dispatch helper.
    loss = DFLTrainer.compute_loss(
        loss_fn,
        costs_pred,
        costs,
        sols,
        objs,
        method_name=method_name,
    )

    # Assert that the correct branch was used.
    assert loss_fn.call_sizes == [expected_arity], (
        "DFLTrainer did not call the expected loss signature."
    )
    assert loss.item() == pytest.approx(float(expected_arity)), (
        "DFLTrainer returned the wrong loss for the selected method."
    )
    pass


def test_dfl_trainer_fit_restores_best_validation_state(
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify that best-model restoration is driven by validation loss."""
    # Arrange train and validation loaders that prefer different weights.
    train_loader = _BatchLoader(
        [
            (
                torch.ones((1, 1), dtype=torch.float),
                torch.tensor([[[2.0]]], dtype=torch.float),
                torch.zeros((1, 1, 1), dtype=torch.float),
                torch.zeros((1, 1, 1), dtype=torch.float),
                torch.zeros((1, 1, 1), dtype=torch.float),
            )
        ]
    )
    val_loader = _BatchLoader(
        [
            (
                torch.ones((1, 1), dtype=torch.float),
                torch.tensor([[[1.0]]], dtype=torch.float),
                torch.zeros((1, 1, 1), dtype=torch.float),
                torch.zeros((1, 1, 1), dtype=torch.float),
                torch.zeros((1, 1, 1), dtype=torch.float),
            )
        ]
    )
    monkeypatch.setattr(
        base_trainer_module.pyepo.metric,
        "regret",
        lambda pred_model, opt_model, loader: 0.0,
    )
    model = _ScalarModel(init_weight=0.0)
    trainer = DFLTrainer(
        pred_model=model,
        opt_model=SimpleNamespace(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.25),
        loss_fn=_ScalarSpoLoss(),
        method_name="spo+",
    )

    # Act by training for two epochs with validation enabled.
    train_loss_log, train_regret_log, val_loss_log, val_regret_log = trainer.fit(
        train_loader,
        val_loader=val_loader,
        epochs=2,
        n_epochs=1,
    )

    # Assert the restored model matches the best validation epoch, not train loss.
    assert model.weight.item() == pytest.approx(1.0), (
        "DFLTrainer restored the wrong model state; validation loss should "
        "control best-model selection."
    )
    assert len(train_loss_log) == 3, "DFLTrainer did not log the initial and two training losses."
    assert len(train_regret_log) == 3, "DFLTrainer did not log the initial and two training regrets."
    assert len(val_loss_log) == 3, "DFLTrainer did not log the initial and two validation losses."
    assert len(val_regret_log) == 3, "DFLTrainer did not log the initial and two validation regrets."
    pass


#################################
### test dfl_trainer_evaluate ###
#################################


def test_dfl_trainer_evaluate_restores_adverse_mode_after_regret(
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify that regret evaluation leaves DFL loaders in adverse mode."""
    # Arrange a loader whose mode changes are easy to inspect.
    eval_loader = _BatchLoader(
        [
            (
                torch.ones((1, 1), dtype=torch.float),
                torch.tensor([[[1.0]]], dtype=torch.float),
                torch.zeros((1, 1, 1), dtype=torch.float),
                torch.zeros((1, 1, 1), dtype=torch.float),
                torch.zeros((1, 1, 1), dtype=torch.float),
            )
        ]
    )
    regret_modes: list[str] = []

    def fake_regret(pred_model, opt_model, loader):
        """Record the mode observed during regret evaluation."""
        del pred_model, opt_model
        regret_modes.append(loader.mode)
        assert loader.mode == "normal", (
            "DFLTrainer should switch the loader to normal mode before calling regret."
        )
        return 0.0

    monkeypatch.setattr(base_trainer_module.pyepo.metric, "regret", fake_regret)
    model = _ScalarModel(init_weight=0.0)
    trainer = DFLTrainer(
        pred_model=model,
        opt_model=SimpleNamespace(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.25),
        loss_fn=_ScalarSpoLoss(),
        method_name="spo+",
    )

    # Act by evaluating and then fitting once on the same style of loader.
    trainer.evaluate(eval_loader)
    fit_loader = _BatchLoader(
        [
            (
                torch.ones((1, 1), dtype=torch.float),
                torch.tensor([[[1.0]]], dtype=torch.float),
                torch.zeros((1, 1, 1), dtype=torch.float),
                torch.zeros((1, 1, 1), dtype=torch.float),
                torch.zeros((1, 1, 1), dtype=torch.float),
            )
        ]
    )
    trainer.fit(fit_loader, val_loader=None, epochs=1, n_epochs=1)

    # Assert the loader returns to adverse mode after each regret evaluation.
    assert eval_loader.mode == "adverse", (
        "DFLTrainer.evaluate did not restore adverse mode after regret."
    )
    assert fit_loader.mode == "adverse", (
        "DFLTrainer.fit did not restore adverse mode after regret."
    )
    assert regret_modes == ["normal", "normal", "normal"], (
        "DFLTrainer did not evaluate regret while the loader was in normal mode."
    )
    pass
