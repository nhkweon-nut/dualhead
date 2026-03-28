# Copyright (c) 2022, Zikang Zhou. All rights reserved.
# DualHead: HiVT with EgoOnlyInteraction (target-only global query).
from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import EvidentialRegressionLoss
from losses import SoftTargetCrossEntropyLoss
from metrics import ADE
from metrics import FDE
from metrics import MR

from datasets.utils import TemporalData

from .edl_mlp_decoder import EDLMLPDecoder
from .local_encoder import LocalEncoder
from .ego_only_interaction import EgoOnlyInteraction


class DualHead(pl.LightningModule):
    def __init__(
        self,
        historical_steps: int,
        future_steps: int,
        num_modes: int,
        rotate: bool,
        node_dim: int,
        edge_dim: int,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        num_temporal_layers: int,
        num_global_layers: int,
        local_radius: float,
        parallel: bool,
        lr: float,
        weight_decay: float,
        T_max: int,
        edl_lambda_reg: float = 0.1,
        edl_lambda_warmup_epochs: int = 8,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.historical_steps = historical_steps
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.rotate = rotate
        self.parallel = parallel
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max

        self.local_encoder = LocalEncoder(
            historical_steps=historical_steps,
            node_dim=node_dim,
            edge_dim=edge_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_temporal_layers=num_temporal_layers,
            local_radius=local_radius,
            parallel=parallel,
        )
        self.ego_only_interaction = EgoOnlyInteraction(
            historical_steps=historical_steps,
            embed_dim=embed_dim,
            edge_dim=edge_dim,
            num_modes=num_modes,
            num_heads=num_heads,
            num_layers=num_global_layers,
            dropout=dropout,
            rotate=rotate,
        )
        self.decoder = EDLMLPDecoder(
            local_channels=embed_dim,
            global_channels=embed_dim,
            future_steps=future_steps,
            num_modes=num_modes,
        )
        self.evidential_loss = EvidentialRegressionLoss()
        self.cls_loss = SoftTargetCrossEntropyLoss(reduction="mean")

        self.minADE = ADE()
        self.minFDE = FDE()
        self.minMR = MR()

    @staticmethod
    def absolute_agent_indices(data: TemporalData, device: torch.device) -> torch.Tensor:
        """
        에이전트 노드의 **글로벌** 인덱스 ``[B]``.

        PyG ``Batch`` 합치기 시 ``Data.__inc__`` 때문에 이름에 ``index``가 들어간 키(예: ``agent_index``)는
        그래프마다 ``num_nodes``만큼 이미 밀려 **글로벌 인덱스**가 됩니다. 여기서 ``ptr``를 다시 더하면 안 됩니다.
        단일 ``Data``는 로컬 인덱스가 곧 글로벌과 동일합니다.
        """
        ai = data["agent_index"]
        if not torch.is_tensor(ai):
            ai = torch.as_tensor(ai, dtype=torch.long, device=device)
        else:
            ai = ai.view(-1).long().to(device)
        return ai

    def forward(self, data: TemporalData):
        agent_index = data["agent_index"]
        if not torch.is_tensor(agent_index):
            agent_index = torch.as_tensor(agent_index, dtype=torch.long)
        agent_index = agent_index.to(self.device)

        if self.rotate:
            rotate_mat = torch.empty(data.num_nodes, 2, 2, device=self.device)
            sin_vals = torch.sin(data["rotate_angles"])
            cos_vals = torch.cos(data["rotate_angles"])
            rotate_mat[:, 0, 0] = cos_vals
            rotate_mat[:, 0, 1] = -sin_vals
            rotate_mat[:, 1, 0] = sin_vals
            rotate_mat[:, 1, 1] = cos_vals
            if data.y is not None:
                data.y = torch.bmm(data.y, rotate_mat)
            data["rotate_mat"] = rotate_mat
        else:
            data["rotate_mat"] = None

        local_embed = self.local_encoder(data=data)
        global_embed = self.ego_only_interaction(
            data=data,
            local_embed=local_embed,
            agent_index=agent_index,
        )
        y_hat, pi = self.decoder(local_embed=local_embed, global_embed=global_embed)
        return y_hat, pi

    def get_edl_lambda_reg(self) -> float:
        """
        KL weight for ``EvidentialRegressionLoss`` (NLL + λ * KL).

        If ``edl_lambda_warmup_epochs`` > 0: epochs ``0 .. W-1`` use λ=0 (NLL-only, fit x/y);
        from epoch ``W`` onward use full ``edl_lambda_reg``. If ``W`` is 0, always use full λ.
        """
        lam = float(self.hparams.edl_lambda_reg)
        w = int(self.hparams.edl_lambda_warmup_epochs)
        if w <= 0:
            return lam
        if int(self.current_epoch) < w:
            return 0.0
        return lam

    @staticmethod
    def get_uncertainty(y_hat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Epistemic uncertainty from NIG parameters (per axis): ``u = beta / (v * (alpha - 1))``.

        Args:
            y_hat: ``[..., 8]`` with γ(0:2), ν(2:4), α(4:6), β(6:8) (e.g. ``[F, N, H, 8]``).

        Returns:
            Tensor same layout as ``y_hat[..., 2:4]`` (uncertainty for x and y).
        """
        v = y_hat[..., 2:4]
        alpha = y_hat[..., 4:6]
        beta = y_hat[..., 6:8]
        # Floor (alpha-1) so u does not explode when alpha ≈ 1.1 early in training
        denom = v * (alpha - 1.0).clamp(min=0.1)
        return beta / denom.clamp(min=eps)

    def training_step(self, data, batch_idx):
        y_hat, pi = self(data)
        reg_mask = ~data["padding_mask"][:, self.historical_steps :]
        ai_abs = self.absolute_agent_indices(data, y_hat.device)
        lam = self.get_edl_lambda_reg()
        reg_total, reg_nll, reg_kl = self.evidential_loss(
            y_hat,
            data.y,
            ai_abs,
            reg_mask,
            lambda_reg=lam,
        )
        fde_modes = torch.norm(
            y_hat[:, ai_abs, -1, :2] - data.y[ai_abs, -1],
            p=2,
            dim=-1,
        )
        soft_target = F.softmax(-fde_modes, dim=0).transpose(0, 1).detach()
        cls_loss = self.cls_loss(pi[ai_abs], soft_target)
        loss = reg_total + cls_loss
        bs = int(ai_abs.size(0))
        self.log("train_reg_loss", reg_total, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
        self.log("train_edl_nll", reg_nll, on_step=True, on_epoch=True, batch_size=bs)
        self.log("train_edl_kl", reg_kl, on_step=True, on_epoch=True, batch_size=bs)
        self.log("train_cls_loss", cls_loss, on_step=True, on_epoch=True, batch_size=bs)
        self.log("edl_kl_weight", float(lam), on_step=False, on_epoch=True, batch_size=bs)
        return loss

    def validation_step(self, data, batch_idx):
        y_hat, pi = self(data)
        reg_mask = ~data["padding_mask"][:, self.historical_steps :]
        ai_abs = self.absolute_agent_indices(data, y_hat.device)
        lam = self.get_edl_lambda_reg()
        reg_total, reg_nll, reg_kl = self.evidential_loss(
            y_hat,
            data.y,
            ai_abs,
            reg_mask,
            lambda_reg=lam,
        )
        bs = int(ai_abs.size(0))
        self.log("val_reg_loss", reg_total, prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
        self.log("val_edl_nll", reg_nll, on_step=False, on_epoch=True, batch_size=bs)
        self.log("val_edl_kl", reg_kl, on_step=False, on_epoch=True, batch_size=bs)

        y_hat_agent = y_hat[:, ai_abs, :, :2]
        y_agent = data.y[ai_abs]
        fde_agent = torch.norm(y_hat_agent[:, :, -1] - y_agent[:, -1], p=2, dim=-1)
        best_mode_agent = fde_agent.argmin(dim=0)
        b_idx = torch.arange(bs, device=y_hat.device, dtype=torch.long)
        y_hat_best_agent = y_hat_agent[best_mode_agent, b_idx]
        self.minADE.update(y_hat_best_agent, y_agent)
        self.minFDE.update(y_hat_best_agent, y_agent)
        self.minMR.update(y_hat_best_agent, y_agent)
        self.log("val_minADE", self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
        self.log("val_minFDE", self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
        self.log("val_minMR", self.minMR, prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
        )
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = "%s.%s" % (module_name, param_name) if module_name else param_name
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params": [param_dict[param_name] for param_name in sorted(list(decay))],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DualHead")
        parser.add_argument("--historical_steps", type=int, default=20)
        parser.add_argument("--future_steps", type=int, default=30)
        parser.add_argument("--num_modes", type=int, default=6)
        parser.add_argument("--rotate", type=bool, default=True)
        parser.add_argument("--node_dim", type=int, default=2)
        parser.add_argument("--edge_dim", type=int, default=2)
        parser.add_argument(
            "--embed_dim",
            type=int,
            default=None,
            help="필수. CLI 또는 --config YAML/JSON으로 지정.",
        )
        parser.add_argument("--num_heads", type=int, default=8)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--num_temporal_layers", type=int, default=4)
        parser.add_argument("--num_global_layers", type=int, default=3)
        parser.add_argument("--local_radius", type=float, default=50)
        parser.add_argument("--parallel", type=bool, default=False)
        parser.add_argument("--lr", type=float, default=5e-4)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        parser.add_argument("--T_max", type=int, default=64)
        parser.add_argument(
            "--edl_lambda_reg",
            type=float,
            default=0.1,
            help=(
                "KL weight in NLL + lambda*KL. Typical 0.05-0.1; if ADE/FDE stall, try 0.01. "
                "Too large -> model may favor 'high uncertainty' over accuracy."
            ),
        )
        parser.add_argument(
            "--edl_lambda_warmup_epochs",
            type=int,
            default=8,
            help=(
                "Epochs 0..W-1: KL weight forced to 0 (fit trajectories first). "
                "From epoch W: use full edl_lambda_reg. Recommended 5-10; set 0 to disable."
            ),
        )
        return parent_parser
