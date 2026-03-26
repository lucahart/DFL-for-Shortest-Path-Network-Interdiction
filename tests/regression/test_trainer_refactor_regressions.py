from types import SimpleNamespace

import torch
from torch import nn

import pytest

import dflintdpy.utils.base_trainer as base_trainer_module
from dflintdpy.utils.pfl_trainer import PFLTrainer


pytestmark = [pytest.mark.regression, pytest.mark.torch, pytest.mark.pyepo]


############################
### Helper functionality ###
############################


class _ScalarModel(nn.Module):
    """Minimal predictor used to make the regression deterministic."""

    def __init__(self, init_weight: float):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[init_weight]], dtype=torch.float))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Broadcast the learned scalar to the batch shape."""
        return self.weight.expand(feats.shape[0], 1)


class _ModeCountingLoader:
    """Loader stub that records mode changes and identifies the loader."""

    def __init__(self, batches: list[tuple[torch.Tensor, ...]], tag: str):
        self._batches = batches
        self.tag = tag
        self.dataset = list(range(len(batches)))
        self.normal_mode_calls = 0
        self.mode = "adverse"

    def __iter__(self):
        """Yield the same deterministic batches on every pass."""
        return iter(self._batches)

    def normal_mode(self) -> None:
        """Record a transition into normal mode."""
        self.normal_mode_calls += 1
        self.mode = "normal"


class _RegressionOptModel:
    """Opt-model stub that is cheap to deepcopy."""

    def __init__(self):
        self.label = "opt"


###############################
### test trainer regression ###
###############################


def test_pfl_trainer_fit_only_evaluates_validation_regret_on_logging_cadence(
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify that validation regret stays tied to the logging cadence."""
    # Arrange a small training run where the legacy cadence would skip validation.
    train_loader = _ModeCountingLoader(
        [
            (
                torch.ones((1, 1), dtype=torch.float),
                torch.tensor([[2.0]], dtype=torch.float),
                torch.zeros((1, 1), dtype=torch.float),
                torch.zeros((1, 1), dtype=torch.float),
            )
        ],
        tag="train",
    )
    val_loader = _ModeCountingLoader(
        [
            (
                torch.ones((1, 1), dtype=torch.float),
                torch.tensor([[2.0]], dtype=torch.float),
                torch.zeros((1, 1), dtype=torch.float),
                torch.zeros((1, 1), dtype=torch.float),
            )
        ],
        tag="val",
    )
    model = _ScalarModel(init_weight=0.0)
    trainer = PFLTrainer(
        pred_model=model,
        opt_model=_RegressionOptModel(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.25),
        loss_fn=nn.MSELoss(),
    )
    regret_calls: list[str] = []
    monkeypatch.setattr(
        base_trainer_module.pyepo.metric,
        "regret",
        lambda pred_model, opt_model, loader: regret_calls.append(loader.tag)
        or 0.0,
    )

    # Act with epochs below the logging cadence.
    trainer.fit(train_loader, val_loader=val_loader, epochs=2, n_epochs=3)

    # Assert the intended cadence: validation regret should not run each epoch.
    assert regret_calls == ["train", "val", "train", "train"], (
        "Validation regret ran more often than the logging cadence allowed."
    )
    pass
