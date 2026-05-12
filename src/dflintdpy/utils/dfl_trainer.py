from __future__ import annotations

from collections.abc import Callable

import torch
from torch.utils.data import DataLoader

from dflintdpy.data.config import HP
from dflintdpy.utils.base_trainer import BaseTrainer


class DFLTrainer(BaseTrainer):
    """
    A class to handle the training and evaluation of a PyTorch model.
    """
    method_name: str
    diagnostics_callback: Callable[[dict[str, float | int]], None] | None

    def __init__(self,
                 pred_model: torch.nn.Module,
                 opt_model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: torch.nn.Module,
                 method_name: str = "spo+",
                 cfg: HP = None,
                 dfl_variant: str = "a-dfl",
                 diagnostics_callback: Callable[
                     [dict[str, float | int]], None
                 ] | None = None,
                 diagnostics_log_every_n_steps: int = 1,
                 diagnostics_max_batches_per_epoch: int | None = None,
                 surrogate_underprediction_penalty_weight: float = 0.0,
                 surrogate_underprediction_margin: float = 1e-6,
                 ) -> None:
        """
        Initializes the Trainer class.

        ------------
        Parameters
        ------------
        pred_model : torch.nn.Module
            The predictive model before the opt-layer to be trained.
        opt_model : torch.nn.Module
            The optimization model used as opt-layer.
        optimizer : torch.optim.Optimizer
            The optimizer to be used for training the model.
        loss_fn : torch.nn.Module
            The loss function to be used for training the model.
        device : torch.device, optional
            The device on which the model will be trained (default is 'cuda' if available,
            otherwise 'cpu').
        dfl_variant : str, optional
            The variant of DFL to train. Options are ``"a-dfl"`` (default) or ``"mixed"``.
            ``"mixed"`` uses both the original and interdicted scenarios for training.
        """

        super().__init__(pred_model, opt_model, optimizer, loss_fn)
        if method_name in type(self).VALID_METHODS():
            self.method_name = method_name
        else:
            raise ValueError(f"Unknown method name: {method_name}\n"
                             f"Valid methods are: {type(self).VALID_METHODS()}")
        self.cfg = cfg
        if cfg is None and method_name == "hybrid":
            raise ValueError("Configuration must be provided for hybrid method.")

        self.dfl_variant = dfl_variant
        self.diagnostics_callback = diagnostics_callback
        self.diagnostics_log_every_n_steps = max(
            1,
            int(diagnostics_log_every_n_steps),
        )
        if diagnostics_max_batches_per_epoch is not None:
            diagnostics_max_batches_per_epoch = int(
                diagnostics_max_batches_per_epoch
            )
            if diagnostics_max_batches_per_epoch < 1:
                raise ValueError(
                    "diagnostics_max_batches_per_epoch must be >= 1 when set."
                )
        self.diagnostics_max_batches_per_epoch = diagnostics_max_batches_per_epoch
        self.surrogate_underprediction_penalty_weight = float(
            surrogate_underprediction_penalty_weight
        )
        if self.surrogate_underprediction_penalty_weight < 0:
            raise ValueError(
                "surrogate_underprediction_penalty_weight must be >= 0."
            )
        self.surrogate_underprediction_margin = float(
            surrogate_underprediction_margin
        )
        if self.surrogate_underprediction_margin < 0:
            raise ValueError("surrogate_underprediction_margin must be >= 0.")
        self._active_epoch = 0
        self._global_step = 0

    @staticmethod
    def _select_scenarios(
        pred: torch.Tensor,
        costs: torch.Tensor,
        sols: torch.Tensor,
        objs: torch.Tensor,
        dfl_variant: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the scenario tensors used by the configured DFL variant."""
        if costs.shape[1] == 1 or dfl_variant == "mixed":
            return pred, costs, sols, objs
        return pred[:, 1:, ...], costs[:, 1:, ...], sols[:, 1:, ...], objs[:, 1:, ...]

    def _prepare_loader_for_loss(self, loader: DataLoader) -> None:
        """DFL optimizes on adverse scenarios."""
        if hasattr(loader, "adverse_mode"):
            loader.adverse_mode()

    def _compute_regret(self, loader: DataLoader) -> float:
        """Evaluate regret on the original scenarios, then restore DFL mode."""
        if hasattr(loader, "normal_mode"):
            loader.normal_mode()
        regret = super()._compute_regret(loader)
        if hasattr(loader, "adverse_mode"):
            loader.adverse_mode()
        return regret

    @staticmethod
    def _flatten_scenarios(
        pred: torch.Tensor,
        costs: torch.Tensor,
        sols: torch.Tensor,
        objs: torch.Tensor,
        dfl_variant: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Select scenarios according to the DFL variant and flatten batch/scenario
        axes for loss computation.
        """
        B = costs.shape[0]
        pred_sel, costs_sel, sols_sel, objs_sel = DFLTrainer._select_scenarios(
            pred,
            costs,
            sols,
            objs,
            dfl_variant,
        )
        K_eff = costs_sel.shape[1]

        p = pred_sel.reshape(B * K_eff, *pred_sel.shape[2:])
        c = costs_sel.reshape(B * K_eff, *costs_sel.shape[2:])
        s = sols_sel.reshape(B * K_eff, *sols_sel.shape[2:])
        o = objs_sel.reshape(B * K_eff, *objs_sel.shape[2:])
        return p, c, s, o, K_eff

    def _before_epoch(self, epoch: int) -> None:
        """Reserved for method-specific schedules such as hybrid lambda."""
        self._active_epoch = epoch + 1
        # if self.method_name == "hybrid":
        #     self.loss_criterion.lam = type(self).lambda_schedule(self.cfg, epoch)
        return None

    @staticmethod
    def _flatten_gradient_tensors(
        gradients: tuple[torch.Tensor | None, ...],
        parameters: list[torch.nn.Parameter],
    ) -> torch.Tensor:
        """Concatenate one gradient tuple into a single parameter vector."""
        if not parameters:
            return torch.empty(0)
        flattened = []
        for gradient, parameter in zip(gradients, parameters):
            if gradient is None:
                flattened.append(torch.zeros_like(parameter).reshape(-1))
            else:
                flattened.append(gradient.detach().reshape(-1))
        return torch.cat(flattened)

    @staticmethod
    def summarize_gradient_conflicts(
        gradient_vectors: list[torch.Tensor],
        *,
        eps: float = 1e-12,
    ) -> dict[str, float | int]:
        """Compute cosine and cancellation diagnostics from gradient vectors."""
        num_gradients = len(gradient_vectors)
        if num_gradients == 0:
            return {
                "mean_pairwise_cosine": float("nan"),
                "cancellation_ratio": float("nan"),
                "summed_gradient_norm": 0.0,
                "sum_individual_norms": 0.0,
                "pair_count": 0,
            }

        stacked = torch.stack(gradient_vectors)
        norms = torch.linalg.vector_norm(stacked, dim=1)
        sum_individual_norms = float(norms.sum().item())
        summed_gradient = stacked.sum(dim=0)
        summed_gradient_norm = float(torch.linalg.vector_norm(summed_gradient).item())

        if num_gradients < 2:
            mean_pairwise_cosine = float("nan")
            pair_count = 0
        else:
            gram = stacked @ stacked.T
            denom = norms[:, None] * norms[None, :]
            cosine_matrix = torch.full_like(gram, float("nan"))
            valid = denom > eps
            cosine_matrix[valid] = gram[valid] / denom[valid]
            upper_idx = torch.triu_indices(num_gradients, num_gradients, offset=1)
            upper = cosine_matrix[upper_idx[0], upper_idx[1]]
            valid_upper = upper[~torch.isnan(upper)]
            pair_count = int(valid_upper.numel())
            mean_pairwise_cosine = (
                float(valid_upper.mean().item()) if pair_count > 0 else float("nan")
            )

        cancellation_ratio = (
            summed_gradient_norm / sum_individual_norms
            if sum_individual_norms > eps
            else float("nan")
        )
        return {
            "mean_pairwise_cosine": mean_pairwise_cosine,
            "cancellation_ratio": float(cancellation_ratio),
            "summed_gradient_norm": summed_gradient_norm,
            "sum_individual_norms": sum_individual_norms,
            "pair_count": pair_count,
        }

    def _prepare_batch(
        self,
        batch: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Move one adverse batch onto the active device."""
        feats, costs, sols, objs, intds = batch
        return (
            feats.to(self.device),
            costs.to(self.device),
            sols.to(self.device),
            objs.to(self.device),
            intds.to(self.device),
        )

    def _forward_adverse_batch(
        self,
        batch: tuple[torch.Tensor, ...],
    ) -> tuple[
        torch.Tensor,
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return the joint training loss plus grouped scenario tensors."""
        feats, costs, sols, objs, intds = self._prepare_batch(batch)
        pred = self.pred_model(feats).unsqueeze(1) + intds
        batch_size = costs.shape[0]

        p, c, s, o, _ = type(self)._flatten_scenarios(
            pred,
            costs,
            sols,
            objs,
            self.dfl_variant,
        )
        loss = type(self).compute_loss(
            self.loss_criterion,
            p,
            c,
            s,
            o,
            method_name=self.method_name,
            surrogate_underprediction_penalty_weight=(
                self.surrogate_underprediction_penalty_weight
            ),
            surrogate_underprediction_margin=(
                self.surrogate_underprediction_margin
            ),
        )
        return loss, batch_size, pred, costs, sols, objs

    def _should_measure_gradient_conflicts(self, batch_idx: int) -> bool:
        """Return whether diagnostics should be collected for this batch."""
        if self.diagnostics_callback is None:
            return False
        if (
            self.diagnostics_max_batches_per_epoch is not None
            and batch_idx > self.diagnostics_max_batches_per_epoch
        ):
            return False
        return self._global_step % self.diagnostics_log_every_n_steps == 0

    def _build_gradient_diagnostics(
        self,
        pred: torch.Tensor,
        costs: torch.Tensor,
        sols: torch.Tensor,
        objs: torch.Tensor,
    ) -> dict[str, float | int]:
        """Compute gradient-conflict metrics for one selected-scenario batch."""
        pred_sel, costs_sel, sols_sel, objs_sel = type(self)._select_scenarios(
            pred,
            costs,
            sols,
            objs,
            self.dfl_variant,
        )
        effective_scenarios = int(costs_sel.shape[1])
        parameters = [
            parameter
            for parameter in self.pred_model.parameters()
            if parameter.requires_grad
        ]
        gradient_vectors = []
        for scenario_idx in range(effective_scenarios):
            scenario_loss = type(self).compute_loss(
                self.loss_criterion,
                pred_sel[:, scenario_idx, ...],
                costs_sel[:, scenario_idx, ...],
                sols_sel[:, scenario_idx, ...],
                objs_sel[:, scenario_idx, ...],
                method_name=self.method_name,
                surrogate_underprediction_penalty_weight=(
                    self.surrogate_underprediction_penalty_weight
                ),
                surrogate_underprediction_margin=(
                    self.surrogate_underprediction_margin
                ),
            )
            scenario_grads = torch.autograd.grad(
                scenario_loss,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            gradient_vectors.append(
                type(self)._flatten_gradient_tensors(scenario_grads, parameters)
            )

        diagnostics = type(self).summarize_gradient_conflicts(gradient_vectors)
        diagnostics.update(
            {
                "requested_scenarios": int(costs.shape[1]),
                "effective_scenarios": effective_scenarios,
                "parameter_count": int(
                    sum(parameter.numel() for parameter in parameters)
                ),
            }
        )
        return diagnostics

    def _compute_batch_loss(
        self,
        batch: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, int]:
        """Compute the decision-focused loss for one adverse batch."""
        loss, batch_size, _, _, _, _ = self._forward_adverse_batch(batch)
        return loss, batch_size

    def train_epoch(self, loader: DataLoader) -> float:
        """Train the predictor for one epoch and optionally record gradients."""
        self._prepare_loader_for_loss(loader)
        self.pred_model.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(loader, start=1):
            loss, batch_size, pred, costs, sols, objs = self._forward_adverse_batch(
                batch
            )
            diagnostics = None
            if self._should_measure_gradient_conflicts(batch_idx):
                diagnostics = self._build_gradient_diagnostics(
                    pred,
                    costs,
                    sols,
                    objs,
                )
                diagnostics.update(
                    {
                        "epoch": int(self._active_epoch),
                        "batch_index": int(batch_idx),
                        "global_step": int(self._global_step),
                        "batch_size": int(batch_size),
                    }
                )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if diagnostics is not None and self.diagnostics_callback is not None:
                self.diagnostics_callback(diagnostics)

            running_loss += loss.item() * batch_size
            self._global_step += 1

        return running_loss / self._dataset_size(loader)
    
    @staticmethod
    def VALID_METHODS():
        """
        Returns the list of valid method names for training.
        """
        return ["spo+", "hybrid", "ptb", "pfy", "imle", "aimle", "nce", "cmap",
                "dbb", "nid", "pg", "ltr"]

    @staticmethod
    def surrogate_underprediction_penalty(
            costs_pred: torch.Tensor,
            costs: torch.Tensor,
            margin: float = 1e-6,
        ) -> torch.Tensor:
        """Penalize predictions that make ``2 * pred - true`` negative."""
        threshold = 0.5 * costs + margin
        return torch.relu(threshold - costs_pred).mean()

    @staticmethod
    def safe_surrogate_costs(
            costs_pred: torch.Tensor,
            costs: torch.Tensor,
            margin: float = 1e-6,
        ) -> torch.Tensor:
        """Floor SPO-style surrogate costs so ``2 * pred - true`` is positive."""
        threshold = 0.5 * costs + margin
        return torch.maximum(costs_pred, threshold)

    @staticmethod
    def compute_loss(loss_criterion: torch.nn.Module,
                    costs_pred: torch.Tensor,
                    costs: torch.Tensor,
                    sols: torch.Tensor,
                    objs: torch.Tensor,
                    method_name: str,
                    surrogate_underprediction_penalty_weight: float = 0.0,
                    surrogate_underprediction_margin: float = 1e-6,
                    ) -> torch.Tensor:
        """
        Computes the loss for the given method name.

        ------------
        Parameters
        ------------
        loss_criterion : torch.nn.Module
            The loss criterion to use for computing the loss.
        costs_pred : torch.Tensor
            The predicted costs.
        costs : torch.Tensor
            The true costs.
        sols : torch.Tensor
            The solutions.
        objs : torch.Tensor
            The objectives.
        method_name : str
            The name of the method to use for computing the loss.

        ------------
        Returns
        ------------
        torch.Tensor
            The computed loss.
        """

        if method_name in ["spo+", "hybrid"]:
            surrogate_costs_pred = costs_pred
            if surrogate_underprediction_penalty_weight > 0:
                surrogate_costs_pred = DFLTrainer.safe_surrogate_costs(
                    costs_pred,
                    costs,
                    surrogate_underprediction_margin,
                )
            loss = loss_criterion(surrogate_costs_pred, costs, sols, objs)
            if surrogate_underprediction_penalty_weight > 0:
                loss = loss + surrogate_underprediction_penalty_weight * \
                    DFLTrainer.surrogate_underprediction_penalty(
                        costs_pred,
                        costs,
                        surrogate_underprediction_margin,
                    )
            return loss
        elif method_name in ["ptb", "pfy", "imle", "aimle", "nce", "cmap"]:
            return loss_criterion(costs_pred, sols)
        elif method_name in ["dbb", "nid"]:
            return loss_criterion(costs_pred, costs, objs)
        elif method_name in ["pg", "ltr"]:
            return loss_criterion(costs_pred, costs)
