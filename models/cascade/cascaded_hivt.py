# Cascaded HiVT: HiVT LocalEncoder + GlobalInteractor + Full MLPDecoder;
# Small path: proj_local/proj_global → SlimEDLDecoder.
# Adaptation: train only small_decoder + proj_*; encoder + full_decoder frozen (freeze_encoder_and_full).
# EDLMLP small 전용 모델: ``cascaded_hivt_edl_mlp.CascadedHiVTEDLMLP``.
from __future__ import annotations

import time

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.utils import TemporalData
from losses import EvidentialRegressionLoss
from losses import SoftTargetCrossEntropyLoss
from metrics import ADE
from metrics import FDE
from metrics import MR
from torchmetrics import Metric

from models.DualHead.edl_mlp_decoder import SlimEDLDecoder
from models.HiVT.decoder import MLPDecoder
from models.HiVT.global_interactor import GlobalInteractor
from models.HiVT.local_encoder import LocalEncoder


class CascadedHiVT(pl.LightningModule):
    """
    - ``embed_dim``: HiVT 인코더 / full 디코더 차원 (체크포인트와 동일, 보통 256).
    - ``small_embed_dim``: Small 경로 투영 차원. ``proj_local`` / ``proj_global`` 로 ``embed_dim`` → ``small_embed_dim``.
    - **Full path**: ``LocalEncoder → GlobalInteractor → MLPDecoder`` (동결 가능).
    - **Small path**: ``local_embed, global_embed`` 를 ``proj_*`` 한 뒤 ``SlimEDLDecoder``.
    - ``small_path_local_only=True``: GlobalInteractor 생략, ``proj_local`` 만 쓰고 global 은 0.
    """

    def __init__(
        self,
        historical_steps: int,
        future_steps: int,
        num_modes: int,
        rotate: bool,
        node_dim: int,
        edge_dim: int,
        embed_dim: int,
        small_embed_dim: int,
        num_heads: int,
        dropout: float,
        num_temporal_layers: int,
        num_global_layers: int,
        local_radius: float,
        parallel: bool,
        lr: float,
        weight_decay: float,
        T_max: int,
        edl_lambda_reg: float = 0.05,
        edl_lambda_warmup_epochs: int = 10,
        freeze_encoder_and_full: bool = True,
        proj_lr_mult: float = 1.0,
        proj_high_lr_epochs: int = 0,
        log_full_val_metrics: bool = True,
        eval_full_only: bool = False,
        small_path_local_only: bool = False,
        log_t_ps_each_epoch: bool = True,
        t_ps_benchmark_batches: int = 16,
        cascade_model: str = "cascaded_hivt",
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["kwargs"])
        self.historical_steps = historical_steps
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.rotate = rotate
        self.parallel = parallel
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max
        self.embed_dim = embed_dim
        self.small_embed_dim = small_embed_dim
        self.freeze_encoder_and_full = freeze_encoder_and_full

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
        self.global_interactor = GlobalInteractor(
            historical_steps=historical_steps,
            embed_dim=embed_dim,
            edge_dim=edge_dim,
            num_modes=num_modes,
            num_heads=num_heads,
            num_layers=num_global_layers,
            dropout=dropout,
            rotate=rotate,
        )
        self.full_decoder = MLPDecoder(
            local_channels=embed_dim,
            global_channels=embed_dim,
            future_steps=future_steps,
            num_modes=num_modes,
            uncertain=True,
        )
        self.proj_local = nn.Linear(embed_dim, small_embed_dim)
        self.proj_global = nn.Linear(embed_dim, small_embed_dim)
        self.small_decoder = self._build_small_decoder()
        self.evidential_loss = EvidentialRegressionLoss()
        self.cls_loss = SoftTargetCrossEntropyLoss(reduction="mean")

        self.minADE = ADE()
        self.minFDE = FDE()
        self.minMR = MR()
        self.full_minADE = ADE()
        self.full_minFDE = FDE()
        self.full_minMR = MR()

        nn.init.xavier_uniform_(self.proj_local.weight)
        nn.init.zeros_(self.proj_local.bias)
        nn.init.xavier_uniform_(self.proj_global.weight)
        nn.init.zeros_(self.proj_global.bias)

        # proj 는 embed_dim -> small_embed_dim 투영 역할.
        # embed_dim == small_embed_dim 인 경우(기본: 128==128)에는 디코더 초반 학습 안정화를 위해
        # 거의 identity 로 시작시키는 것이 유리함.
        if int(self.embed_dim) == int(self.small_embed_dim):
            with torch.no_grad():
                self.proj_local.weight.zero_()
                self.proj_global.weight.zero_()
                eye = torch.eye(
                    int(self.embed_dim),
                    device=self.proj_local.weight.device,
                    dtype=self.proj_local.weight.dtype,
                )
                self.proj_local.weight.copy_(eye)
                self.proj_global.weight.copy_(eye)

        if getattr(self.hparams, "small_path_local_only", False):
            for p in self.local_encoder.parameters():
                p.requires_grad = False
            for p in self.global_interactor.parameters():
                p.requires_grad = False
            for p in self.full_decoder.parameters():
                p.requires_grad = False
            for p in self.proj_global.parameters():
                p.requires_grad = False
            for p in self.proj_local.parameters():
                p.requires_grad = True
            for p in self.small_decoder.parameters():
                p.requires_grad = True
        elif freeze_encoder_and_full:
            for p in self.local_encoder.parameters():
                p.requires_grad = False
            for p in self.global_interactor.parameters():
                p.requires_grad = False
            for p in self.full_decoder.parameters():
                p.requires_grad = False

    def _build_small_decoder(self) -> nn.Module:
        return SlimEDLDecoder(
            local_channels=self.small_embed_dim,
            global_channels=self.small_embed_dim,
            future_steps=self.future_steps,
            num_modes=self.num_modes,
        )

    @staticmethod
    def absolute_agent_indices(data: TemporalData, device: torch.device) -> torch.Tensor:
        ai = data["agent_index"]
        if not torch.is_tensor(ai):
            ai = torch.as_tensor(ai, dtype=torch.long, device=device)
        else:
            ai = ai.view(-1).long().to(device)
        return ai

    def _apply_rotate(self, data: TemporalData) -> None:
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

    def encode(self, data: TemporalData) -> tuple[torch.Tensor, torch.Tensor]:
        local_embed = self.local_encoder(data=data)
        global_embed = self.global_interactor(data=data, local_embed=local_embed)
        return local_embed, global_embed

    def encode_local_only(self, data: TemporalData) -> torch.Tensor:
        """global_interaction 없이 LocalEncoder 출력만 (Small local-only fine-tuning용)."""
        return self.local_encoder(data=data)

    def _edl_global_zeros_like_proj_global(self, local_s: torch.Tensor) -> torch.Tensor:
        """
        EDL 디코더는 global이 ``proj_global(global_embed)`` 와 같은 rank 여야 함 — 보통 ``[num_modes, N, C]``.
        ``local_s`` 는 ``[N, C]`` 이므로 ``zeros_like(local_s)`` 는 2D만 되어 cat 에서 깨짐.

        ``proj_global`` 가중치는 읽지 않는다(동결 여부와 무관). ``torch.zeros(..., device=local_s.device)``
        만 사용하므로 할당/그래프 이슈 없음.
        """
        return torch.zeros(
            (self.num_modes,) + tuple(local_s.shape),
            dtype=local_s.dtype,
            device=local_s.device,
        )

    def forward_small_local_only(self, data: TemporalData) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Small 경로: local feature만 사용. global 쪽은 동일 차원의 0 텐서로 채워 EDL 디코더 입력 형태 유지.
        (global_interactor·proj_global 미사용)
        """
        self._apply_rotate(data)
        local_embed = self.encode_local_only(data)
        local_s = self.proj_local(local_embed)
        global_s = self._edl_global_zeros_like_proj_global(local_s)
        return self.small_decoder(local_embed=local_s, global_embed=global_s)

    def _forward_small_from_latents(
        self, data: TemporalData, local_embed: torch.Tensor, global_embed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_s = self.proj_local(local_embed)
        global_s = self.proj_global(global_embed)
        return self.small_decoder(local_embed=local_s, global_embed=global_s)

    def forward_small_from_latents(
        self, data: TemporalData, local_embed: torch.Tensor, global_embed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """인퍼런스 STEP 1: ``encode`` 로 얻은 latents에서 Small decoder만 (pred_s = y_hat)."""
        return self._forward_small_from_latents(data, local_embed, global_embed)

    def forward_small(self, data: TemporalData) -> tuple[torch.Tensor, torch.Tensor]:
        """EDL small 경로만 (적응 학습용)."""
        self._apply_rotate(data)
        local_embed, global_embed = self.encode(data)
        return self._forward_small_from_latents(data, local_embed, global_embed)

    def forward_full(self, data: TemporalData) -> tuple[torch.Tensor, torch.Tensor]:
        """Full HiVT 디코더 (추론/평가용; 학습 시에는 동결)."""
        self._apply_rotate(data)
        local_embed, global_embed = self.encode(data)
        return self.full_decoder(local_embed=local_embed, global_embed=global_embed)

    def forward_full_from_latents(
        self, data: TemporalData, local_embed: torch.Tensor, global_embed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """rotate+encode 한 번만 한 뒤 full 디코더만 호출할 때 사용."""
        return self.full_decoder(local_embed=local_embed, global_embed=global_embed)

    def full_interaction_and_decoder(
        self, data: TemporalData, local_embed: torch.Tensor, global_embed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        **Inference stack (Full)** — HiVT와 동일 차원에서의 global interaction 임베딩 위에 full MLP 디코딩.
        지연 측정 시 기준(100%)은 ``encode`` + 본 호출(Interaction 임베딩 위 full decode) 합으로 둔다.
        """
        return self.forward_full_from_latents(data, local_embed, global_embed)

    def small_interaction_and_decoder(
        self, data: TemporalData, local_embed: torch.Tensor, global_embed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        **Inference stack (Light)** — ``proj_*`` 로 ``small_embed_dim`` 투영 후 ``small_decoder`` 만 수행.
        게이팅에서 Full을 쓰지 않을 때의 가벼운 decode 경로(디코더 부분)에 해당.
        """
        return self.forward_small_from_latents(data, local_embed, global_embed)

    def forward(self, data: TemporalData) -> tuple[torch.Tensor, torch.Tensor]:
        if getattr(self.hparams, "small_path_local_only", False):
            return self.forward_small_local_only(data)
        return self.forward_small(data)

    def get_edl_lambda_reg(self) -> float:
        lam = float(self.hparams.edl_lambda_reg)
        w = int(self.hparams.edl_lambda_warmup_epochs)
        if w <= 0:
            return lam
        if int(self.current_epoch) < w:
            return 0.0
        return lam

    def training_step(self, data, batch_idx):
        y_hat, pi = (
            self.forward_small_local_only(data)
            if getattr(self.hparams, "small_path_local_only", False)
            else self.forward_small(data)
        )
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

    def _update_min_metrics_from_preds(
        self,
        y_hat: torch.Tensor,
        data: TemporalData,
        ai_abs: torch.Tensor,
        ade: Metric,
        fde: Metric,
        mr: Metric,
    ) -> int:
        bs = int(ai_abs.size(0))
        y_hat_agent = y_hat[:, ai_abs, :, :2]
        y_agent = data.y[ai_abs]
        fde_agent = torch.norm(y_hat_agent[:, :, -1] - y_agent[:, -1], p=2, dim=-1)
        best_mode_agent = fde_agent.argmin(dim=0)
        b_idx = torch.arange(bs, device=y_hat.device, dtype=torch.long)
        y_hat_best_agent = y_hat_agent[best_mode_agent, b_idx]
        ade.update(y_hat_best_agent, y_agent)
        fde.update(y_hat_best_agent, y_agent)
        mr.update(y_hat_best_agent, y_agent)
        return bs

    def validation_step(self, data, batch_idx):
        eval_full_only = bool(self.hparams.eval_full_only)
        log_full = bool(self.hparams.log_full_val_metrics) and not eval_full_only

        with torch.no_grad():
            self._apply_rotate(data)
            local_embed, global_embed = self.encode(data)
            device = local_embed.device

            if not eval_full_only:
                if getattr(self.hparams, "small_path_local_only", False):
                    local_s = self.proj_local(local_embed)
                    global_s = self._edl_global_zeros_like_proj_global(local_s)
                    y_hat, pi = self.small_decoder(local_embed=local_s, global_embed=global_s)
                else:
                    y_hat, pi = self._forward_small_from_latents(data, local_embed, global_embed)
                reg_mask = ~data["padding_mask"][:, self.historical_steps :]
                ai_abs = self.absolute_agent_indices(data, device)
                lam = self.get_edl_lambda_reg()
                reg_total, reg_nll, reg_kl = self.evidential_loss(
                    y_hat,
                    data.y,
                    ai_abs,
                    reg_mask,
                    lambda_reg=lam,
                )
                bs = self._update_min_metrics_from_preds(
                    y_hat, data, ai_abs, self.minADE, self.minFDE, self.minMR
                )
                self.log(
                    "val_reg_loss",
                    reg_total,
                    prog_bar=True,
                    on_step=False,
                    on_epoch=True,
                    batch_size=bs,
                )
                self.log("val_edl_nll", reg_nll, on_step=False, on_epoch=True, batch_size=bs)
                self.log("val_edl_kl", reg_kl, on_step=False, on_epoch=True, batch_size=bs)
                self.log("val_minADE", self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
                self.log("val_minFDE", self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
                self.log("val_minMR", self.minMR, prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)

            if log_full or eval_full_only:
                y_hat_f, _pi_f = self.forward_full_from_latents(data, local_embed, global_embed)
                ai_abs = self.absolute_agent_indices(data, device)
                bs = self._update_min_metrics_from_preds(
                    y_hat_f, data, ai_abs, self.full_minADE, self.full_minFDE, self.full_minMR
                )
                self.log(
                    "val_full_minADE",
                    self.full_minADE,
                    prog_bar=eval_full_only,
                    on_step=False,
                    on_epoch=True,
                    batch_size=bs,
                )
                self.log(
                    "val_full_minFDE",
                    self.full_minFDE,
                    prog_bar=eval_full_only,
                    on_step=False,
                    on_epoch=True,
                    batch_size=bs,
                )
                self.log(
                    "val_full_minMR",
                    self.full_minMR,
                    prog_bar=eval_full_only,
                    on_step=False,
                    on_epoch=True,
                    batch_size=bs,
                )

    def _set_frozen_backbone_eval_trainable_train(self) -> None:
        """
        ``freeze_encoder_and_full`` 일 때 Teacher 경로(local_encoder, global_interactor, full_decoder)는
        ``eval()`` 로 두어 Dropout 비활성·BN 추론 통계 사용. 학습 대상(small_decoder, proj 등)만 ``train()``.
        """
        if not getattr(self.hparams, "freeze_encoder_and_full", True):
            return
        self.local_encoder.eval()
        self.global_interactor.eval()
        self.full_decoder.eval()
        if any(p.requires_grad for p in self.small_decoder.parameters()):
            self.small_decoder.train()
        if any(p.requires_grad for p in self.proj_local.parameters()):
            self.proj_local.train()
        if any(p.requires_grad for p in self.proj_global.parameters()):
            self.proj_global.train()

    def on_train_epoch_start(self) -> None:
        self._set_frozen_backbone_eval_trainable_train()
        w = int(self.hparams.proj_high_lr_epochs)
        m = float(self.hparams.proj_lr_mult)
        if w <= 0 or m == 1.0:
            return
        if self.current_epoch != w:
            return
        opt = self.optimizers()
        if isinstance(opt, (list, tuple)):
            opt = opt[0]
        if len(opt.param_groups) < 2:
            return
        opt.param_groups[1]["lr"] = opt.param_groups[0]["lr"]
        scheds = self.lr_schedulers()
        if isinstance(scheds, (list, tuple)):
            sched = scheds[0] if scheds else None
        else:
            sched = scheds
        if sched is not None and hasattr(sched, "base_lrs") and len(sched.base_lrs) >= 2:
            sched.base_lrs[1] = sched.base_lrs[0]

    def configure_optimizers(self):
        small_params = [p for p in self.small_decoder.parameters() if p.requires_grad]
        proj_params = [p for p in self.proj_local.parameters() if p.requires_grad]
        if not getattr(self.hparams, "small_path_local_only", False):
            proj_params += [p for p in self.proj_global.parameters() if p.requires_grad]
        w = int(self.hparams.proj_high_lr_epochs)
        m = float(self.hparams.proj_lr_mult)
        if w <= 0 or m == 1.0 or len(proj_params) == 0:
            params = small_params + proj_params
            optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.AdamW(
                [
                    {"params": small_params, "lr": self.lr, "weight_decay": self.weight_decay},
                    {"params": proj_params, "lr": self.lr * m, "weight_decay": self.weight_decay},
                ]
            )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=self.T_max, eta_min=0.0
        )
        return [optimizer], [scheduler]

    def on_train_epoch_end(self) -> None:
        if not bool(getattr(self.hparams, "log_t_ps_each_epoch", False)):
            return
        dm = getattr(self.trainer, "datamodule", None) if self.trainer else None
        if dm is None:
            return
        loader = dm.val_dataloader()
        if loader is None:
            return
        nbench = max(1, int(getattr(self.hparams, "t_ps_benchmark_batches", 16)))
        dev = self.device
        self.eval()
        proj_times_ms: list[float] = []
        decoder_times_ms: list[float] = []
        with torch.no_grad():
            for bi, batch in enumerate(loader):
                if bi >= nbench:
                    break
                batch = batch.to(dev)
                self._apply_rotate(batch)
                if getattr(self.hparams, "small_path_local_only", False):
                    local_embed = self.encode_local_only(batch)
                    if dev.type == "cuda":
                        torch.cuda.synchronize(dev)
                    t_proj0 = time.perf_counter()
                    local_s = self.proj_local(local_embed)
                    if dev.type == "cuda":
                        torch.cuda.synchronize(dev)
                    t_proj1 = time.perf_counter()
                    global_s = self._edl_global_zeros_like_proj_global(local_s)
                    if dev.type == "cuda":
                        torch.cuda.synchronize(dev)
                    t_dec0 = time.perf_counter()
                    _ = self.small_decoder(local_embed=local_s, global_embed=global_s)
                    if dev.type == "cuda":
                        torch.cuda.synchronize(dev)
                    t_dec1 = time.perf_counter()
                    proj_times_ms.append((t_proj1 - t_proj0) * 1000.0)
                    decoder_times_ms.append((t_dec1 - t_dec0) * 1000.0)
                else:
                    local_embed, global_embed = self.encode(batch)
                    if dev.type == "cuda":
                        torch.cuda.synchronize(dev)
                    t_proj0 = time.perf_counter()
                    local_s = self.proj_local(local_embed)
                    global_s = self.proj_global(global_embed)
                    if dev.type == "cuda":
                        torch.cuda.synchronize(dev)
                    t_proj1 = time.perf_counter()
                    if dev.type == "cuda":
                        torch.cuda.synchronize(dev)
                    t_dec0 = time.perf_counter()
                    _ = self.small_decoder(local_embed=local_s, global_embed=global_s)
                    if dev.type == "cuda":
                        torch.cuda.synchronize(dev)
                    t_dec1 = time.perf_counter()
                    proj_times_ms.append((t_proj1 - t_proj0) * 1000.0)
                    decoder_times_ms.append((t_dec1 - t_dec0) * 1000.0)
        self.train()
        if proj_times_ms and decoder_times_ms:
            mean_proj_ms = float(sum(proj_times_ms) / len(proj_times_ms))
            mean_decoder_ms = float(sum(decoder_times_ms) / len(decoder_times_ms))
            print(
                f"[proj+decoder] epoch {int(self.current_epoch)} "
                f"mean proj(+proj_global if used): {mean_proj_ms:.4f} ms, "
                f"mean small_decoder: {mean_decoder_ms:.4f} ms (batches={len(proj_times_ms)})"
            )
            self.log(
                "train_epoch_proj_ms",
                mean_proj_ms,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                "train_epoch_small_decoder_ms",
                mean_decoder_ms,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

    @staticmethod
    def add_model_specific_args(parent_parser):
        p = parent_parser.add_argument_group("CascadedHiVT")
        p.add_argument("--historical_steps", type=int, default=20)
        p.add_argument("--future_steps", type=int, default=30)
        p.add_argument("--num_modes", type=int, default=6)
        p.add_argument("--rotate", type=bool, default=True)
        p.add_argument("--node_dim", type=int, default=2)
        p.add_argument("--edge_dim", type=int, default=2)
        p.add_argument("--embed_dim", type=int, default=256, help="HiVT 인코더/full 디코더 (체크포인트와 동일)")
        p.add_argument("--small_embed_dim", type=int, default=128, help="EDL small + proj 출력 차원")
        p.add_argument("--num_heads", type=int, default=8)
        p.add_argument("--dropout", type=float, default=0.1)
        p.add_argument("--num_temporal_layers", type=int, default=4)
        p.add_argument("--num_global_layers", type=int, default=3)
        p.add_argument("--local_radius", type=float, default=50)
        p.add_argument("--parallel", type=bool, default=False)
        p.add_argument(
            "--lr",
            type=float,
            default=1e-3,
            help="AdamW LR. CascadedHiVTMLPKD( full→small 복제 시작) 시 5e-4~1e-4 권장",
        )
        p.add_argument("--weight_decay", type=float, default=1e-4)
        p.add_argument("--T_max", type=int, default=20, help="Cosine 스케줄 (적응 학습 에폭에 맞춤)")
        p.add_argument("--edl_lambda_reg", type=float, default=0.05)
        p.add_argument("--edl_lambda_warmup_epochs", type=int, default=10)
        p.add_argument(
            "--freeze_encoder_and_full",
            type=bool,
            default=True,
            help="인코더·full_decoder 동결 (small+proj만 학습)",
        )
        p.add_argument(
            "--proj_lr_mult",
            type=float,
            default=1.0,
            help="proj_local/proj_global에만 적용하는 초기 LR 배수 (proj_high_lr_epochs 동안)",
        )
        p.add_argument(
            "--proj_high_lr_epochs",
            type=int,
            default=0,
            help="proj에 proj_lr_mult를 적용할 에폭 수(0이면 small과 동일 LR 한 그룹)",
        )
        p.add_argument(
            "--log_full_val_metrics",
            type=bool,
            default=True,
            help="검증 시 rotate+encode 한 번으로 small과 full FDE/ADE를 함께 기록",
        )
        p.add_argument(
            "--eval_full_only",
            type=bool,
            default=False,
            help="True면 검증에서 full 경로만 평가(HiVT 베이스라인 대조용)",
        )
        p.add_argument(
            "--small_path_local_only",
            type=bool,
            default=False,
            help="True면 Small 경로가 global_interaction 없이 local_encoder→proj_local→small_decoder만 사용",
        )
        p.add_argument(
            "--log_t_ps_each_epoch",
            type=bool,
            default=True,
            help="에폭 끝마다 val에서 소수 배치로 Small 디코더 구간 평균 시간(ms) 로깅",
        )
        p.add_argument(
            "--t_ps_benchmark_batches",
            type=int,
            default=16,
            help="t_ps 벤치마크에 쓸 val 배치 수 상한",
        )
        return parent_parser
