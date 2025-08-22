import torch
import torch.nn as nn
import torch.nn.functional as F
from pyepo.func import SPOPlus


def mse_anchor(ctilde: torch.Tensor, ctrue: torch.Tensor) -> torch.Tensor:
    """Mean squared error between predicted and true costs."""
    return F.mse_loss(ctilde, ctrue)


def meanstd_anchor(ctilde: torch.Tensor,
                   ctrue: torch.Tensor,
                   eps: float = 1e-8) -> torch.Tensor:
    """Match mean and standard deviation of costs."""
    mu_hat, mu = ctilde.mean(1, True), ctrue.mean(1, True)
    sd_hat = ctilde.std(1, False, True) + eps
    sd = ctrue.std(1, False, True) + eps
    return ((mu_hat - mu) ** 2 + (sd_hat - sd) ** 2).mean()


class HybridSPOLoss(nn.Module):
    """Combine SPO+ loss with an anchoring term."""

    spo_plus: SPOPlus
    lam: float
    anchor: str

    def __init__(self,
                 opt_model,
                 lam: float = 0.1,
                 anchor: str = "mse") -> None:
        super().__init__()
        self.lam = lam
        self.anchor = anchor
        self.spo_plus = SPOPlus(opt_model, processes=1)

    def set_lambda(self, new_val: float) -> None:
        """Update mixing weight of anchor term."""
        self.lam = new_val

    def forward(self,
                c_pred: torch.Tensor,
                c_true: torch.Tensor,
                sols: torch.Tensor,
                objs: torch.Tensor) -> torch.Tensor:
        """Compute hybrid loss."""
        L_spo = self.spo_plus(c_pred, c_true, sols, objs)
        if self.anchor == "mse":
            L_anchor = mse_anchor(c_pred, c_true)
        else:
            L_anchor = meanstd_anchor(c_pred, c_true)
        return (1 - self.lam) * L_spo + self.lam * L_anchor
