# Evidential regression: NIG NLL + Inverse-Gamma KL regularizer, WTA by agent FDE.
from __future__ import annotations

import math
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


class EvidentialRegressionLoss(nn.Module):
    """
    Winner-Take-All (WTA) over modes using **agent final displacement error (FDE)** only,
    then NIG negative log-likelihood + KL regularizer on the selected mode.

    Total: ``nll + lambda_reg * kl``
    """

    def __init__(
        self,
        eps: float = 1e-6,
        alpha_prior: float = 2.0,
        beta_prior: float = 1.0,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior

    def nig_nll(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """NIG NLL, same as EvidentialNLLLoss (mean over last dimensions)."""
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
        return nll.mean()

    def kl_inverse_gamma(self, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        KL( IG(α, β) || IG(α0, β0) ) per channel (x, y), averaged.
        Encourages predicted evidence to stay close to a weakly-informative prior when not supported by data.
        """
        alpha = alpha.clamp(min=1.0 + self.eps)
        beta = beta.clamp(min=self.eps)
        a0 = alpha.new_tensor(self.alpha_prior)
        b0 = beta.new_tensor(self.beta_prior)
        q = D.InverseGamma(concentration=alpha, rate=beta, validate_args=False)
        p = D.InverseGamma(concentration=a0.expand_as(alpha), rate=b0.expand_as(beta), validate_args=False)
        kl = D.kl_divergence(q, p)
        return kl

    def forward(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        agent_index: Union[int, torch.Tensor],
        reg_mask: torch.Tensor,
        lambda_reg: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            y_hat: ``[F, N, H, 8]`` multi-mode predictions (γ, ν, α, β per axis).
            y: ``[N, H, 2]`` ground-truth offsets.
            agent_index: **global** agent node index: ``int`` (단일 그래프) 또는 ``[B]``
                (PyG ``Batch``는 ``agent_index``에 collate 시 노드 오프셋이 이미 적용됨).
            reg_mask: ``[N, H]`` bool, valid future steps (same as HiVT ``~padding_mask[:, 20:]``).
            lambda_reg: weight for KL term (schedule externally).

        Returns:
            ``(total_loss, nll, reg)``
        """
        if y_hat.dim() != 4 or y_hat.size(-1) != 8:
            raise ValueError(f"expected y_hat [F, N, H, 8], got {tuple(y_hat.shape)}")
        if isinstance(agent_index, int):
            ai = torch.tensor([agent_index], device=y_hat.device, dtype=torch.long)
        else:
            ai = agent_index.view(-1).long().to(y_hat.device)
        # WTA: per-agent best mode by FDE at last step (γ vs y).
        y_agent = y[ai]  # [B, H, 2]
        y_hat_agent = y_hat[:, ai, :, :]  # [F, B, H, 8]
        fde = torch.norm(
            y_hat_agent[:, :, -1, :2] - y_agent[:, -1].unsqueeze(0),
            p=2,
            dim=-1,
        )  # [F, B]
        best_f = fde.argmin(dim=0)  # [B]
        # y_hat: [F, N, H, 8]. 두 번째 축은 노드 N이지 배치 B가 아님.
        # [best_f, ai] 동시 인덱싱은 시나리오 b마다 (모드 best_f[b], 글로벌 노드 ai[b])를 고름.
        # gather와 동일: torch.arange(B)는 N축에 쓰이면 안 됨.
        pred = y_hat[best_f, ai, :, :]  # [B, H, 8]
        reg_m = reg_mask[ai]  # [B, H]
        pred_m = pred[reg_m]
        y_m = y_agent[reg_m]
        if pred_m.numel() == 0:
            z = y_hat.new_zeros(())
            return z, z, z

        nll = self.nig_nll(pred_m, y_m)
        alpha_m = pred_m[..., 4:6]
        beta_m = pred_m[..., 6:8]
        kl = self.kl_inverse_gamma(alpha_m, beta_m).mean()
        total = nll + float(lambda_reg) * kl
        return total, nll, kl
