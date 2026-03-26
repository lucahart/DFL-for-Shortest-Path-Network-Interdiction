from __future__ import annotations

from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Optional

import matplotlib.pyplot as plt
import pyepo.metric
import torch
from matplotlib.axes import Axes
from numpy import arange
from numpy import ndarray
from torch.utils.data import DataLoader


class BaseTrainer(ABC):
    """Shared training loop for predictive and decision-focused trainers."""

    LOSS_INCREASE_THRESHOLD = 0.10

    device: torch.device
    pred_model: torch.nn.Module
    opt_model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    loss_criterion: torch.nn.Module
    n_epochs: int

    def __init__(
        self,
        pred_model: torch.nn.Module,
        opt_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
    ) -> None:
        """Store the common trainer state on the active torch device."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pred_model = pred_model.to(self.device)
        self.opt_model = deepcopy(opt_model)
        self.optimizer = optimizer
        self.loss_criterion = loss_fn
        self.n_epochs = 1

    def _prepare_loader_for_loss(self, loader: DataLoader) -> None:
        """Put a loader into the mode expected by this trainer."""
        return None

    def _before_epoch(self, epoch: int) -> None:
        """Hook for trainer-specific per-epoch updates."""
        return None

    @abstractmethod
    def _compute_batch_loss(
        self,
        batch: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, int]:
        """Return the batch loss and its base-instance count."""

    def _compute_regret(self, loader: DataLoader) -> float:
        """Compute regret for the current predictor on the given loader."""
        return pyepo.metric.regret(self.pred_model, self.opt_model, loader)

    def _snapshot_model_state(self) -> dict[str, torch.Tensor]:
        """Clone the current predictor weights for later restoration."""
        return deepcopy(self.pred_model.state_dict())

    @staticmethod
    def _dataset_size(loader: DataLoader) -> int:
        """Return the number of base samples represented by the loader."""
        return len(loader.dataset)

    def _print_epoch_metrics(
        self,
        epoch: int,
        train_loss: float,
        train_regret: float,
        val_loss: Optional[float] = None,
        val_regret: Optional[float] = None,
    ) -> None:
        """Emit one formatted progress line."""
        parts = [
            f"Epoch {epoch:02d}",
            f"Train Loss: {train_loss:.4f}",
            f"Train Regret: {train_regret:.4f}",
        ]
        if val_loss is not None and val_regret is not None:
            parts.append(f"Validation Loss: {val_loss:.4f}")
            parts.append(f"Validation Regret: {val_regret:.4f}")
        print(" | ".join(parts))

    def train_epoch(self, loader: DataLoader) -> float:
        """Train the predictor for one epoch on the given loader."""
        self._prepare_loader_for_loss(loader)
        self.pred_model.train()
        running_loss = 0.0

        for batch in loader:
            loss, batch_size = self._compute_batch_loss(batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * batch_size

        return running_loss / self._dataset_size(loader)

    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        """Evaluate loss and regret on the given loader."""
        total_loss = self._evaluate_loss(loader)
        regret = self._compute_regret(loader)
        return total_loss, regret

    def _evaluate_loss(self, loader: DataLoader) -> float:
        """Evaluate loss only on the given loader."""
        self._prepare_loader_for_loss(loader)
        self.pred_model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in loader:
                loss, batch_size = self._compute_batch_loss(batch)
                total_loss += loss.item() * batch_size

        return total_loss / self._dataset_size(loader)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        epochs: int = 10,
        n_epochs: int = -1,
    ) -> tuple[list[float], list[float], Optional[list[float]], Optional[list[float]]]:
        """Fit the predictor and return logged loss/regret trajectories."""
        self._prepare_loader_for_loss(train_loader)
        if val_loader is not None:
            self._prepare_loader_for_loss(val_loader)

        self.n_epochs = max(1, epochs // 10) if n_epochs < 0 else max(1, n_epochs)

        train_loss, train_regret = self.evaluate(train_loader)
        train_loss_vector = [train_loss]
        train_regret_vector = [train_regret]

        val_loss_vector: Optional[list[float]] = None
        val_regret_vector: Optional[list[float]] = None
        best_val_loss: Optional[float] = None
        best_model_state: Optional[dict[str, torch.Tensor]] = None

        if val_loader is not None:
            val_loss, val_regret = self.evaluate(val_loader)
            val_loss_vector = [val_loss]
            val_regret_vector = [val_regret]
            best_val_loss = val_loss
            best_model_state = self._snapshot_model_state()
            self._print_epoch_metrics(
                0,
                train_loss,
                train_regret,
                val_loss,
                val_regret,
            )
        else:
            self._print_epoch_metrics(0, train_loss, train_regret)

        for epoch in range(1, epochs + 1):
            self._before_epoch(epoch - 1)

            train_loss = self.train_epoch(train_loader)
            train_regret = self._compute_regret(train_loader)
            train_loss_vector.append(train_loss)
            train_regret_vector.append(train_regret)

            if val_loader is not None:
                val_loss = self._evaluate_loss(val_loader)

                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self._snapshot_model_state()

                if (
                    best_val_loss is not None
                    and train_loss - best_val_loss
                    > self.LOSS_INCREASE_THRESHOLD * best_val_loss
                ):
                    self.optimizer.param_groups[0]["lr"] *= 0.5
                    print(
                        f"Epoch {epoch:02d} | "
                        "Increase in training loss detected. "
                        "Reducing learning rate to "
                        f"{self.optimizer.param_groups[0]['lr']:.2e}"
                    )

            if epoch % self.n_epochs == 0:
                if val_loader is not None:
                    val_regret = self._compute_regret(val_loader)
                    val_loss_vector.append(val_loss)
                    val_regret_vector.append(val_regret)
                    self._print_epoch_metrics(
                        epoch,
                        train_loss,
                        train_regret,
                        val_loss,
                        val_regret,
                    )
                else:
                    self._print_epoch_metrics(epoch, train_loss, train_regret)

        if best_model_state is not None:
            self.pred_model.load_state_dict(best_model_state)

        return (
            train_loss_vector,
            train_regret_vector,
            val_loss_vector,
            val_regret_vector,
        )

    @staticmethod
    def vis_learning_curve(
        trainer: "BaseTrainer",
        train_loss_log: ndarray[float],
        train_regret_log: ndarray[float],
        test_loss_log: ndarray[float] = None,
        test_regret_log: ndarray[float] = None,
        ax: Optional[tuple[Axes, Axes]] = None,
        *,
        file_name: Optional[str] = None,
    ) -> None:
        """Visualize loss and regret traces for one training run."""
        if ax is None:
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
        else:
            ax1, ax2 = ax

        ax1.plot(train_regret_log, marker=".", label="Training Regret")
        if test_regret_log is not None:
            ax1.scatter(
                arange(len(test_regret_log)) * trainer.n_epochs,
                test_regret_log,
                marker="x",
                color="red",
                label="Testing Regret",
            )
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Regret")
        ax1.set_yscale("log")
        ax1.set_title("Regret Learning Curve")
        ax1.legend()

        ax2.plot(train_loss_log, marker=".", label="Training Loss")
        if test_loss_log is not None:
            ax2.scatter(
                arange(len(test_loss_log)) * trainer.n_epochs,
                test_loss_log,
                marker="x",
                color="red",
                label="Testing Loss",
            )
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.set_yscale("log")
        ax2.set_title("Loss Learning Curve")
        ax2.legend()

        plt.tight_layout()
        if file_name is not None:
            plt.savefig(file_name + ".png", dpi=300, bbox_inches="tight")
            plt.close()
