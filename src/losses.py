import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """Multi-class Focal Loss.

    Args:
        gamma: focusing parameter (>= 0).
        alpha: balancing factor. Tensor of shape [C] for per-class weights or a scalar.
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = float(gamma)
        if isinstance(alpha, (int, float)):
            alpha = torch.tensor(float(alpha), dtype=torch.float)
        self.register_buffer("alpha", alpha if alpha is not None else None)
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of: 'mean', 'sum', 'none'")
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        target = target.view(-1)
        log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        # alpha per target class if provided
        if self.alpha is not None:
            if self.alpha.dim() == 0:
                at = self.alpha
            else:
                at = self.alpha.to(logits.device)[target]
        else:
            at = 1.0
        loss = -at * (1.0 - pt).pow(self.gamma) * log_pt
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

