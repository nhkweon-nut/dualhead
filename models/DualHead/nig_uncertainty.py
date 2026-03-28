# Post-hoc NIG uncertainty from ``y_hat`` (does not modify EDLMLPDecoder / checkpoints).
from __future__ import annotations

from typing import Literal

import torch


def nig_epistemic_uncertainty(y_hat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Normal-Inverse-Gamma epistemic uncertainty per axis:
    :math:`u_{\\mathrm{epi}} = \\frac{\\beta}{\\nu(\\alpha - 1)}`.

    Args:
        y_hat: Last dim 8 — ``γ`` (0:2), ``ν`` (2:4), ``α`` (4:6), ``β`` (6:8), e.g.
            ``[num_modes, n_agents, future_steps, 8]`` from the trained decoder output.

    Returns:
        Same leading shape, last dim 2 (x, y).
    """
    v = y_hat[..., 2:4]
    alpha = y_hat[..., 4:6]
    beta = y_hat[..., 6:8]
    denom = v * (alpha - 1.0).clamp(min=0.1)
    return beta / denom.clamp(min=eps)


def nig_aleatoric_uncertainty(y_hat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    NIG aleatoric uncertainty per axis: :math:`u_{\\mathrm{ale}} = \\frac{\\beta}{\\alpha - 1}`.

    Args:
        y_hat: Same layout as :func:`nig_epistemic_uncertainty`.

    Returns:
        Same leading shape, last dim 2 (x, y).
    """
    alpha = y_hat[..., 4:6]
    beta = y_hat[..., 6:8]
    denom = (alpha - 1.0).clamp(min=0.1)
    return beta / denom.clamp(min=eps)


def combine_xy_uncertainty(
    u_xy: torch.Tensor,
    reduction: Literal["l2", "mean", "sum"] = "l2",
) -> torch.Tensor:
    """Collapse ``[..., 2]`` per-axis uncertainty to a scalar ``[...]``."""
    if u_xy.size(-1) != 2:
        raise ValueError("Expected last dim 2 (x, y), got %s" % (u_xy.shape,))
    if reduction == "l2":
        return torch.norm(u_xy, p=2, dim=-1)
    if reduction == "mean":
        return u_xy.mean(dim=-1)
    if reduction == "sum":
        return u_xy.sum(dim=-1)
    raise ValueError("reduction must be 'l2', 'mean', or 'sum'")
