# Evidential Deep Learning (EDL) decoder: NIG parameters per (x, y) from HiVT MLPDecoder layout.
# local_embed feeds both pi and aggr_embed (same as HiVT); KL from EDL loss backprops through both.
# Use edl_lambda_warmup_epochs on DualHead if local features become unstable early in training.
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.utils import init_weights


class EDLMLPDecoder(nn.Module):
    """
    Same interface as ``MLPDecoder`` for ``aggr_embed`` + ``pi``, but trajectory prediction uses
    a single ``evidential`` head: ``future_steps * 8`` raw outputs → γ, ν, α, β per axis.
    """

    def __init__(
        self,
        local_channels: int,
        global_channels: int,
        future_steps: int,
        num_modes: int,
    ) -> None:
        super().__init__()
        self.input_size = global_channels
        self.hidden_size = local_channels
        self.future_steps = future_steps
        self.num_modes = num_modes

        self.aggr_embed = nn.Sequential(
            nn.Linear(self.input_size + self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
        )
        self.evidential = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.future_steps * 8),
        )
        self.pi = nn.Sequential(
            nn.Linear(self.hidden_size + self.input_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 1),
        )
        self.apply(init_weights)

    def forward(
        self,
        local_embed: torch.Tensor,
        global_embed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pi = (
            self.pi(
                torch.cat(
                    (local_embed.expand(self.num_modes, *local_embed.shape), global_embed),
                    dim=-1,
                )
            )
            .squeeze(-1)
            .t()
        )
        out = self.aggr_embed(
            torch.cat((global_embed, local_embed.expand(self.num_modes, *local_embed.shape)), dim=-1)
        )
        raw = self.evidential(out).view(self.num_modes, -1, self.future_steps, 8)
        gamma = raw[..., 0:2]
        v_raw = raw[..., 2:4]
        alpha_raw = raw[..., 4:6]
        beta_raw = raw[..., 6:8]
        v = F.softplus(v_raw) + 1e-6
        alpha = F.softplus(alpha_raw) + 1.1
        beta = F.softplus(beta_raw) + 1e-6
        y_hat = torch.cat((gamma, v, alpha, beta), dim=-1)
        return y_hat, pi
