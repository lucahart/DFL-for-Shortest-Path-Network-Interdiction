from torch.utils.data import DataLoader

import torch

from dflintdpy.utils.base_trainer import BaseTrainer


class PFLTrainer(BaseTrainer):
    """
    A class to handle the training and evaluation of a PyTorch model.
    """

    def __init__(self,
                 pred_model: torch.nn.Module,
                 opt_model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: torch.nn.Module
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
        """

        super().__init__(pred_model, opt_model, optimizer, loss_fn)

    def _prepare_loader_for_loss(self, loader: DataLoader) -> None:
        """PFL always optimizes on the original non-interdicted samples."""
        if hasattr(loader, "normal_mode"):
            loader.normal_mode()

    def _compute_batch_loss(
        self,
        batch: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, int]:
        """Compute the predictive loss for one batch."""
        feats, costs, _, _ = batch
        feats = feats.to(self.device)
        costs = costs.to(self.device)

        costs_pred = self.pred_model(feats)
        loss = self.loss_criterion(costs_pred, costs)
        return loss, feats.size(0)
