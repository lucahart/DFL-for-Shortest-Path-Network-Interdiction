import torch
from torch.utils.data import DataLoader

from dflintdpy.data.config import HP
from dflintdpy.utils.base_trainer import BaseTrainer


class DFLTrainer(BaseTrainer):
    """
    A class to handle the training and evaluation of a PyTorch model.
    """
    method_name: str

    def __init__(self,
                 pred_model: torch.nn.Module,
                 opt_model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: torch.nn.Module,
                 method_name: str = "spo+",
                 cfg: HP = None,
                 dfl_variant: str = "a-dfl"
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
        B, K = costs.shape[:2]

        if K == 1 or dfl_variant == "mixed":
            pred_sel = pred
            costs_sel = costs
            sols_sel = sols
            objs_sel = objs
            K_eff = K
        else:
            # A-DFL uses only interdicted scenarios (drop scenario 0 per sample).
            pred_sel = pred[:, 1:, ...]
            costs_sel = costs[:, 1:, ...]
            sols_sel = sols[:, 1:, ...]
            objs_sel = objs[:, 1:, ...]
            K_eff = K - 1

        p = pred_sel.reshape(B * K_eff, *pred_sel.shape[2:])
        c = costs_sel.reshape(B * K_eff, *costs_sel.shape[2:])
        s = sols_sel.reshape(B * K_eff, *sols_sel.shape[2:])
        o = objs_sel.reshape(B * K_eff, *objs_sel.shape[2:])
        return p, c, s, o, K_eff

    def _before_epoch(self, epoch: int) -> None:
        """Reserved for method-specific schedules such as hybrid lambda."""
        # if self.method_name == "hybrid":
        #     self.loss_criterion.lam = type(self).lambda_schedule(self.cfg, epoch)
        return None

    def _compute_batch_loss(
        self,
        batch: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, int]:
        """Compute the decision-focused loss for one adverse batch."""
        feats, costs, sols, objs, intds = batch

        feats = feats.to(self.device)
        costs = costs.to(self.device)
        sols = sols.to(self.device)
        objs = objs.to(self.device)
        intds = intds.to(self.device)

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
        )
        return loss, batch_size
    
    @staticmethod
    def VALID_METHODS():
        """
        Returns the list of valid method names for training.
        """
        return ["spo+", "hybrid", "ptb", "pfy", "imle", "aimle", "nce", "cmap",
                "dbb", "nid", "pg", "ltr"]

    @staticmethod
    def compute_loss(loss_criterion: torch.nn.Module,
                    costs_pred: torch.Tensor,
                    costs: torch.Tensor,
                    sols: torch.Tensor,
                    objs: torch.Tensor,
                    method_name: str) -> torch.Tensor:
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

        if method_name == "spo+":
            return loss_criterion(costs_pred, costs, sols, objs)
        if method_name == "hybrid":
            return loss_criterion(costs_pred, costs, sols, objs)
        elif method_name in ["ptb", "pfy", "imle", "aimle", "nce", "cmap"]:
            return loss_criterion(costs_pred, sols)
        elif method_name in ["dbb", "nid"]:
            return loss_criterion(costs_pred, costs, objs)
        elif method_name in ["pg", "ltr"]:
            return loss_criterion(costs_pred, costs)
