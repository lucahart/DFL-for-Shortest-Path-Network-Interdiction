import torch
import torch.nn as nn
import torch.nn.functional as F

class CalibratedPredictor(nn.Module):
    """Wrap a predictive model with a global affine calibration."""

    base: nn.Module
    log_s: nn.Parameter
    b: nn.Parameter
    eps: float

    def __init__(self, base_model: nn.Module, eps: float = 1e-4) -> None:
        super().__init__()
        self.base = base_model
        self.log_s = nn.Parameter(torch.zeros(1))  # logarithm of scale
        self.b = nn.Parameter(torch.zeros(1))
        self.eps = eps

    def forward(self, feats: torch.Tensor, return_all: bool = False):
        """Forward pass applying calibration.

        Parameters
        ----------
        feats: torch.Tensor
            Input features passed to the base model.
        return_all: bool, optional
            If True, also returns the raw predictions and the positive scale
            value used for calibration.
        """
        chat = self.base(feats)
        s = F.softplus(self.log_s) + self.eps
        ctilde = s * chat + self.b
        if return_all:
            return ctilde, chat, s
        return ctilde

@torch.no_grad()
def scale_alignment_alpha(ctilde: torch.Tensor, ctrue: torch.Tensor) -> float:
    """Compute best scalar aligning predicted costs to true costs."""
    num = (ctilde * ctrue).sum(dim=1)
    den = (ctilde * ctilde).sum(dim=1).clamp_min(1e-12)
    return (num / den).mean().item()
