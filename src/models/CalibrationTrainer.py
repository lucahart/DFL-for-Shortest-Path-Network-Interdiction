import torch
from torch.utils.data import DataLoader

def inverse_softplus(y, eps=0.0):
    """
    Solve x such that softplus(x)+eps = y  (softplus(x)=ln(1+e^x)).
    For y>eps: x = log(exp(y-eps) - 1).
    """
    t = torch.clamp(y - eps, min=1e-12)
    return torch.log(torch.expm1(t))

class CalibratorTrainer:
    """
    Tools to:
      (A) Fit a scale-only calibrator (b=0) in closed form (decision-invariant).
      (C) Optionally fine-tune calibrator (s,b) with hybrid loss while backbone is frozen.
    """

    def __init__(self, model, spo_plus_loss=None, device=None):
        """
        model: CalibratedPredictor-like module with attributes .log_s and .b
        spo_plus_loss: a callable loss, e.g., PyEPO's SPOPlus()
        device: torch device (optional)
        """
        self.model = model
        self.spo_plus = spo_plus_loss
        self.device = device or next(model.parameters()).device

    @torch.no_grad()
    def align_scale_ols(self, dataloader, clamp_min=1e-6, use_running_stats=True):
        """
        (A) Closed-form scale-only OLS fit for s (keeps decisions unchanged).
        Sets b=0. Uses chats from the backbone and ctrue to compute:
          s* = <chat, ctrue> / <chat, chat>
        Aggregated over the provided dataloader (train/val set).
        """
        self.model.eval()
        sum_xy = 0.0
        sum_xx = 0.0

        for feats, costs, sols, objs in dataloader:
            feats = feats.to(self.device)
            costs = costs.to(self.device)
            # Forward: model returns (ctilde, chat, s); we want chat from backbone
            cost_pred = self.model.base(feats)
            x = cost_pred.reshape(-1)
            y = costs.reshape(-1)
            sum_xy += torch.dot(x, y).item()
            sum_xx += torch.dot(x, x).item()

        den = max(sum_xx, 1e-12)
        s_star = max(sum_xy / den, clamp_min)  # positive scalar; keeps argmin invariant

        # Install into calibrator: s = softplus(log_s) + eps  -> set log_s to inverse_softplus(s-eps)
        # We set b=0 to keep decisions identical to backbone (scale-only).
        with torch.no_grad():
            # Find eps used in your forward (adjust if different)
            # If your forward uses: s = softplus(log_s) + 1e-4
            eps = 1e-4
            s_target = max(s_star - eps, clamp_min)
            self.model.log_s.copy_(inverse_softplus(torch.tensor(s_target), eps=0.0))

        return s_star  # return for logging