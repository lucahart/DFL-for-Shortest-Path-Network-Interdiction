import pyepo.metric
import torch
from torch.utils.data import DataLoader
from typing import Tuple

from .SPOTrainer import SPOTrainer


class CalibratedSPOTrainer(SPOTrainer):
    """Trainer specialized for :class:`CalibratedPredictor` models.

    This subclass unwraps the calibrated cost estimates returned by
    :class:`~src.models.CalibratedPredictor.CalibratedPredictor` and feeds the
    calibrated costs to the underlying ``SPOTrainer`` logic.
    """

    def train_epoch(self, loader: DataLoader) -> float:
        """Train for a single epoch using calibrated costs."""
        self.pred_model.train()
        running_loss = 0.0
        for feats, costs, sols, objs in loader:
            feats = feats.to(self.device)
            costs = costs.to(self.device)
            sols = sols.to(self.device)
            objs = objs.to(self.device)

            ctilde, *_ = self.pred_model(feats, return_all=True)
            loss = type(self).compute_loss(self.loss_criterion,
                                           ctilde, costs, sols, objs,
                                           method_name=self.method_name)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * feats.size(0)
        return running_loss / len(loader.dataset)

    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """Evaluate model returning average loss and regret."""
        self.pred_model.eval()
        total_loss = 0.0
        total_regret = 0.0
        with torch.no_grad():
            for feats, costs, sols, objs in loader:
                feats = feats.to(self.device)
                costs = costs.to(self.device)
                sols = sols.to(self.device)
                objs = objs.to(self.device)

                ctilde, *_ = self.pred_model(feats, return_all=True)
                loss = type(self).compute_loss(self.loss_criterion,
                                               ctilde, costs, sols, objs,
                                               method_name=self.method_name)
                regret = pyepo.metric.regret(ctilde, costs, sols)
                total_loss += loss.item() * feats.size(0)
                total_regret += regret.sum().item()
        n_samples = len(loader.dataset)
        return total_loss / n_samples, total_regret / n_samples
