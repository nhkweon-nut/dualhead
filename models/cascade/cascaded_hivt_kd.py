# Teacher=full MLPDecoder(동결), Student=동일 MLPDecoder + global_embed=0. Laplace NLL + KD.
# ``CascadedHiVT``(Slim EDL small)와 분리된 클래스 — 학습/체크포인트는 ``CascadedHiVTMLPKD`` 로 구분.
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from datasets.utils import TemporalData
from losses import LaplaceNLLLoss
from models.HiVT.decoder import MLPDecoder

from .cascaded_hivt import CascadedHiVT
from .weight_utils import clone_full_decoder_to_small_mlp


class CascadedHiVTMLPKD(CascadedHiVT):
    """
    Small 경로: ``LocalEncoder → proj_local → MLPDecoder``, ``global_embed=0`` (Teacher는 Full 경로).
    ``mlp_kd_use_proj=False`` 이면 예전처럼 raw ``local_embed`` 를 디코더에 직접 넣음 (동일 차원일 때만).
    Teacher는 동결된 ``full_decoder``; 과제는 Laplace+cls, KD는 Teacher winner 모드 정렬.
    """

    def __init__(
        self,
        kd_alpha: float = 0.5,
        kd_temp: float = 1.0,
        kd_gt_boost: float = 0.5,
        kd_u_alpha: float = 0.25,
        strict_clone_small_from_full: bool = True,
        mlp_kd_use_proj: bool = True,
        **kwargs,
    ) -> None:
        kwargs.setdefault("cascade_model", "cascaded_hivt_mlp_kd")
        super().__init__(**kwargs)
        self.mlp_kd_use_proj = bool(mlp_kd_use_proj)
        # proj 경로: 디코더 입·출력 채널은 small_embed_dim (Full의 embed 와 같으면 clone 가능)
        _sl = int(self.small_embed_dim) if self.mlp_kd_use_proj else int(self.embed_dim)
        self.small_decoder = MLPDecoder(
            local_channels=_sl,
            global_channels=_sl,
            future_steps=self.future_steps,
            num_modes=self.num_modes,
            uncertain=True,
        )
        self.laplace_nll = LaplaceNLLLoss(reduction="mean")
        self.kd_alpha = float(kd_alpha)
        self.kd_temp = float(kd_temp)
        self.kd_gt_boost = float(kd_gt_boost)
        self.kd_u_alpha = float(kd_u_alpha)
        self.strict_clone_small_from_full = bool(strict_clone_small_from_full)
        self.save_hyperparameters(
            {
                "kd_alpha": self.kd_alpha,
                "kd_temp": self.kd_temp,
                "kd_gt_boost": self.kd_gt_boost,
                "kd_u_alpha": self.kd_u_alpha,
                "strict_clone_small_from_full": self.strict_clone_small_from_full,
                "mlp_kd_use_proj": self.mlp_kd_use_proj,
            },
            logger=False,
        )

    def _student_local(self, local_embed: torch.Tensor) -> torch.Tensor:
        if self.mlp_kd_use_proj:
            return self.proj_local(local_embed)
        return local_embed

    def _student_global_zeros(self, ref_global_embed: torch.Tensor) -> torch.Tensor:
        """Student MLPDecoder용 ``global_embed`` — ``[num_modes, N, C]`` (C = small 디코더 global 채널)."""
        c = int(self.small_decoder.input_size)
        f, n = int(ref_global_embed.shape[0]), int(ref_global_embed.shape[1])
        return torch.zeros(
            (f, n, c),
            dtype=ref_global_embed.dtype,
            device=ref_global_embed.device,
        )

    def on_train_start(self) -> None:
        """학습 시작 시 ``full_decoder`` → ``small_decoder`` strict copy (체크포인트 재개 시 생략)."""
        if not self.strict_clone_small_from_full:
            return
        tr = self.trainer
        if tr is None:
            return
        ckpt = getattr(tr, "ckpt_path", None)
        if ckpt is not None and str(ckpt).strip():
            return
        fd = self.full_decoder
        sd = self.small_decoder
        if (
            fd.hidden_size != sd.hidden_size
            or fd.input_size != sd.input_size
        ):
            print(
                "[CascadedHiVTMLPKD] full_decoder 와 small_decoder 채널이 달라 clone 생략 "
                f"(full {fd.hidden_size}/{fd.input_size}, small {sd.hidden_size}/{sd.input_size}). "
                "embed_dim==small_embed_dim 이고 mlp_kd_use_proj=True 이면 clone 가능.",
                flush=True,
            )
            return
        clone_full_decoder_to_small_mlp(self)

    def forward_small_local_only(self, data: TemporalData) -> tuple[torch.Tensor, torch.Tensor]:
        self._apply_rotate(data)
        local_embed = self.encode_local_only(data)
        local_s = self._student_local(local_embed)
        gz = torch.zeros(
            self.num_modes,
            local_embed.shape[0],
            int(self.small_decoder.input_size),
            device=local_embed.device,
            dtype=local_embed.dtype,
        )
        return self.small_decoder(local_embed=local_s, global_embed=gz)

    def _forward_small_from_latents(
        self, data: TemporalData, local_embed: torch.Tensor, global_embed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_s = self._student_local(local_embed)
        gz = self._student_global_zeros(global_embed)
        return self.small_decoder(local_embed=local_s, global_embed=gz)

    def small_interaction_and_decoder(
        self, data: TemporalData, local_embed: torch.Tensor, global_embed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        **Inference stack (Light)** — ``proj_local``(옵션) 후 MLP ``small_decoder``, ``global_embed=0``.
        """
        return self.forward_small_from_latents(data, local_embed, global_embed)

    def training_step(self, data, batch_idx):
        self._apply_rotate(data)
        if getattr(self.hparams, "small_path_local_only", False):
            local_embed = self.encode_local_only(data)
            global_embed = self.global_interactor(data=data, local_embed=local_embed)
        else:
            local_embed, global_embed = self.encode(data)

        local_s = self._student_local(local_embed)
        gz = self._student_global_zeros(global_embed)
        y_s, pi_s = self.small_decoder(local_embed=local_s, global_embed=gz)

        reg_mask = ~data["padding_mask"][:, self.historical_steps :]
        valid_steps = reg_mask.sum(dim=-1)
        cls_mask = valid_steps > 0
        num_nodes = int(data.num_nodes)
        node_idx = torch.arange(num_nodes, device=y_s.device, dtype=torch.long)

        # 과제: Student multimodal — GT에 가장 가까운 모드 (기존과 동일)
        l2_norm_s = (torch.norm(y_s[:, :, :, :2] - data.y, p=2, dim=-1) * reg_mask).sum(dim=-1)
        best_mode_s = l2_norm_s.argmin(dim=0)
        y_s_best = y_s[best_mode_s, node_idx]
        reg_loss = self.laplace_nll(y_s_best[reg_mask], data.y[reg_mask])
        soft_target = F.softmax(
            -l2_norm_s[:, cls_mask] / valid_steps[cls_mask].clamp(min=1.0), dim=0
        ).t().detach()
        cls_loss = self.cls_loss(pi_s[cls_mask], soft_target)
        loss_task = reg_loss + cls_loss

        ka = float(self.kd_alpha)
        bs = int(cls_mask.sum().item()) if cls_mask.any() else int(data.num_graphs)
        self.log("train_reg_loss", reg_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
        self.log("train_cls_loss", cls_loss, on_step=True, on_epoch=True, batch_size=bs)

        # kd_alpha<=0: GT 과제만 (Teacher 순전파·KD·task_mult 없음)
        if ka <= 0.0:
            self.log("train_kd_active", 0.0, on_step=True, on_epoch=True, batch_size=bs)
            return loss_task

        with torch.no_grad():
            y_t, _pi_t = self.full_decoder(local_embed=local_embed, global_embed=global_embed)

        # KD: Teacher winner-takes-all — GT 기준 최적 모드에서만 Student 정렬
        l2_norm_t = (torch.norm(y_t[:, :, :, :2] - data.y, p=2, dim=-1) * reg_mask).sum(dim=-1)
        mode_t = l2_norm_t.argmin(dim=0)
        y_t_w = y_t[mode_t, node_idx].detach()
        y_s_w = y_s[mode_t, node_idx]
        kd_mse_loc = ((y_s_w[..., :2] - y_t_w[..., :2]) ** 2).mean()
        kd_mse_u = ((y_s_w[..., 2:4] - y_t_w[..., 2:4]) ** 2).mean()
        kd_total = kd_mse_loc + float(self.kd_u_alpha) * kd_mse_u

        u_mean = y_t_w[..., 2:4].abs().mean()
        w_kd = torch.exp(-u_mean / float(self.kd_temp))
        task_mult = 1.0 + float(self.kd_gt_boost) * torch.tanh(u_mean)
        loss = (1.0 - ka) * loss_task * task_mult + ka * w_kd * kd_total

        self.log("train_kd_active", 1.0, on_step=True, on_epoch=True, batch_size=bs)
        self.log("train_kd_mse", kd_total, on_step=True, on_epoch=True, batch_size=bs)
        self.log("train_kd_mse_loc", kd_mse_loc, on_step=True, on_epoch=True, batch_size=bs)
        self.log("train_kd_mse_u", kd_mse_u, on_step=True, on_epoch=True, batch_size=bs)
        self.log("train_kd_w_scale", w_kd, on_step=True, on_epoch=True, batch_size=bs)
        self.log("train_kd_u_mean", u_mean, on_step=True, on_epoch=True, batch_size=bs)
        self.log("train_kd_task_mult", task_mult, on_step=True, on_epoch=True, batch_size=bs)
        return loss

    def validation_step(self, data, batch_idx):
        eval_full_only = bool(self.hparams.eval_full_only)
        log_full = bool(self.hparams.log_full_val_metrics) and not eval_full_only

        with torch.no_grad():
            self._apply_rotate(data)
            if getattr(self.hparams, "small_path_local_only", False):
                local_embed = self.encode_local_only(data)
                global_embed = self.global_interactor(data=data, local_embed=local_embed)
            else:
                local_embed, global_embed = self.encode(data)
            device = local_embed.device

            if not eval_full_only:
                local_s = self._student_local(local_embed)
                gz = self._student_global_zeros(global_embed)
                y_hat, pi = self.small_decoder(local_embed=local_s, global_embed=gz)
                reg_mask = ~data["padding_mask"][:, self.historical_steps :]
                l2_norm = (torch.norm(y_hat[:, :, :, :2] - data.y, p=2, dim=-1) * reg_mask).sum(dim=-1)
                valid_steps = reg_mask.sum(dim=-1)
                cls_mask = valid_steps > 0
                best_mode = l2_norm.argmin(dim=0)
                num_nodes = int(data.num_nodes)
                y_hat_best = y_hat[best_mode, torch.arange(num_nodes, device=y_hat.device)]
                reg_loss = self.laplace_nll(y_hat_best[reg_mask], data.y[reg_mask])
                soft_target = F.softmax(
                    -l2_norm[:, cls_mask] / valid_steps[cls_mask].clamp(min=1.0), dim=0
                ).t().detach()
                cls_loss = self.cls_loss(pi[cls_mask], soft_target)
                ai_abs = self.absolute_agent_indices(data, device)
                bs = self._update_min_metrics_from_preds(
                    y_hat, data, ai_abs, self.minADE, self.minFDE, self.minMR
                )
                self.log(
                    "val_reg_loss",
                    reg_loss,
                    prog_bar=True,
                    on_step=False,
                    on_epoch=True,
                    batch_size=bs,
                )
                self.log("val_cls_loss", cls_loss, on_step=False, on_epoch=True, batch_size=bs)
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

    def configure_optimizers(self):
        small_params = [p for p in self.small_decoder.parameters() if p.requires_grad]
        if self.mlp_kd_use_proj:
            small_params += [p for p in self.proj_local.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(small_params, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=self.T_max, eta_min=0.0
        )
        return [optimizer], [scheduler]

    def on_train_epoch_start(self) -> None:
        # 동결 Teacher eval + proj 스케줄(해당 시) — 부모와 동일
        super().on_train_epoch_start()

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
                    global_embed = self.global_interactor(data=batch, local_embed=local_embed)
                else:
                    local_embed, global_embed = self.encode(batch)
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                t_proj0 = time.perf_counter()
                local_s = self._student_local(local_embed)
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                t_proj1 = time.perf_counter()
                gz = self._student_global_zeros(global_embed)
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                t_dec0 = time.perf_counter()
                _ = self.small_decoder(local_embed=local_s, global_embed=gz)
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
                f"mean proj(local): {mean_proj_ms:.4f} ms, "
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
        parent_parser = CascadedHiVT.add_model_specific_args(parent_parser)
        g = parent_parser.add_argument_group("CascadedHiVTMLPKD (full→small MLP KD)")
        g.add_argument("--kd_alpha", type=float, default=0.5, help="KD MSE 항 가중 (1-α)=과제")
        g.add_argument("--kd_temp", type=float, default=1.0, help="Teacher 불확실성 스케일 (KD 가중)")
        g.add_argument(
            "--kd_gt_boost",
            type=float,
            default=0.5,
            help="불확실성에 따른 과제 손실 배수 (tanh)",
        )
        g.add_argument(
            "--kd_u_alpha",
            type=float,
            default=0.25,
            help="KD에서 Teacher winner 경로의 σ(불확실성) 채널 MSE 가중",
        )
        g.add_argument(
            "--strict_clone_small_from_full",
            type=bool,
            default=True,
            help="True면 학습 시작 시 full→small 복사(체크포인트 재개 시 자동 생략)",
        )
        g.add_argument(
            "--mlp_kd_use_proj",
            type=bool,
            default=True,
            help="True면 Small=proj_local(local)→MLPDecoder(small_embed_dim). False면 raw local→MLP(embed_dim)",
        )
        return parent_parser
