# Normal-Inverse-Gamma negative log-likelihood for evidential regression (per coordinate).
import math

import torch
import torch.nn as nn


class EvidentialNLLLoss(nn.Module):
    """
    NIG NLL matching decoder outputs [..., 8] with slices γ(0:2), ν(2:4), α(4:6), β(6:8).
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean") -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        gamma = pred[..., :2]
        v = pred[..., 2:4]
        alpha = pred[..., 4:6]
        beta = pred[..., 6:8]
        two_blambda = 2.0 * beta * (1.0 + v)
        pi = pred.new_tensor(math.pi)
        nll = (
            0.5 * torch.log(pi / v.clamp(min=self.eps))
            - alpha * torch.log(two_blambda.clamp(min=self.eps))
            + (alpha + 0.5) * torch.log(v * (target - gamma) ** 2 + two_blambda.clamp(min=self.eps))
            + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
        )
        if self.reduction == "mean":
            return nll.mean()
        if self.reduction == "sum":
            return nll.sum()
        if self.reduction == "none":
            return nll
        raise ValueError(f"{self.reduction} is not a valid value for reduction")
