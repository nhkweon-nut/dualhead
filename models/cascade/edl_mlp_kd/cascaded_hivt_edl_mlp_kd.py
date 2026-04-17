"""
EDLMLP Small KD (Teacher=Full MLPDecoder, Student=EDLMLPDecoder).

Goal:
  - Task loss: EvidentialRegressionLoss (NIG NLL + lambda*KL) + SoftTargetCrossEntropyLoss
  - KD loss:
      * Location distillation: Student gamma (x,y) ~ Teacher loc mean (x,y)
      * Mode distillation: KL(student_pi || teacher_pi)
  - Total:
      loss = (1-kd_alpha) * task_loss + kd_alpha * (kd_loc_weight*kd_loc + kd_pi_weight*kd_pi)

Important implementation details:
  - _apply_rotate는 training_step에서 1회만 호출.
  - teacher(full_decoder)와 student(small_decoder)은 encode/decoder를 직접 호출해서
    중복 rotate 및 forward_* 경로 중복을 피함.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.cascade.cascaded_hivt_edl_mlp import CascadedHiVTEDLMLP


class CascadedHiVTEDLMLPKD(CascadedHiVTEDLMLP):
    def __init__(
        self,
        kd_alpha: float = 0.5,
        kd_loc_weight: float = 1.0,
        kd_pi_weight: float = 0.1,
        kd_pi_temperature: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.kd_alpha = float(kd_alpha)
        self.kd_loc_weight = float(kd_loc_weight)
        self.kd_pi_weight = float(kd_pi_weight)
        self.kd_pi_temperature = float(kd_pi_temperature)

        self.save_hyperparameters(
            {
                "kd_alpha": self.kd_alpha,
                "kd_loc_weight": self.kd_loc_weight,
                "kd_pi_weight": self.kd_pi_weight,
                "kd_pi_temperature": self.kd_pi_temperature,
            },
            logger=False,
        )

    def training_step(self, data, batch_idx):
        # 1) Rotate once (teacher/student 모두 같은 좌표계에서 계산)
        self._apply_rotate(data)

        # 2) Encode once for teacher; student도(작은 local-only 포함) 같은 local embed를 재사용
        local_embed = self.local_encoder(data=data)
        global_embed = self.global_interactor(data=data, local_embed=local_embed)

        # 3) Student forward (small path)
        reg_mask = ~data["padding_mask"][:, self.historical_steps :]
        ai_abs = self.absolute_agent_indices(data, device=local_embed.device)

        if getattr(self.hparams, "small_path_local_only", False):
            local_s = self.proj_local(local_embed)
            global_s = self._edl_global_zeros_like_proj_global(local_s)
            y_s, pi_s = self.small_decoder(local_embed=local_s, global_embed=global_s)
        else:
            y_s, pi_s = self._forward_small_from_latents(data, local_embed, global_embed)

        # 4) Task loss (EDL NIG + mode cls using student pi)
        lam = self.get_edl_lambda_reg()
        reg_total, reg_nll, reg_kl = self.evidential_loss(
            y_s,
            data.y,
            ai_abs,
            reg_mask,
            lambda_reg=lam,
        )

        fde_modes = torch.norm(
            y_s[:, ai_abs, -1, :2] - data.y[ai_abs, -1],
            p=2,
            dim=-1,
        )
        soft_target = F.softmax(-fde_modes, dim=0).transpose(0, 1).detach()
        cls_loss = self.cls_loss(pi_s[ai_abs], soft_target)
        loss_task = reg_total + cls_loss

        # 5) No KD case
        ka = float(self.kd_alpha)
        if ka <= 0.0:
            self.log("train_reg_loss", reg_total, prog_bar=True, on_step=True, on_epoch=True, batch_size=ai_abs.numel())
            self.log("train_edl_nll", reg_nll, on_step=True, on_epoch=True, batch_size=ai_abs.numel())
            self.log("train_edl_kl", reg_kl, on_step=True, on_epoch=True, batch_size=ai_abs.numel())
            self.log("train_cls_loss", cls_loss, on_step=True, on_epoch=True, batch_size=ai_abs.numel())
            self.log("train_kd_active", 0.0, on_step=True, on_epoch=True, batch_size=ai_abs.numel())
            return loss_task

        # 6) Teacher forward (full) - no grad
        with torch.no_grad():
            y_t, pi_t = self.full_decoder(local_embed=local_embed, global_embed=global_embed)

        # 7) Mode alignment for location distillation (teacher winner-takes-all by GT distance)
        valid_steps = reg_mask.sum(dim=-1)  # [N]
        cls_mask = valid_steps > 0

        # y_t: [F, N, H, 4] where loc mean is [:,:,:,:2]
        # data.y: [N, H, 2]
        l2_norm_t = (
            torch.norm(y_t[:, :, :, :2] - data.y, p=2, dim=-1) * reg_mask
        ).sum(dim=-1)  # [F, N]
        mode_t = l2_norm_t.argmin(dim=0)  # [N]

        num_nodes = int(data.num_nodes)
        node_idx = torch.arange(num_nodes, device=y_t.device, dtype=torch.long)

        y_t_w = y_t[mode_t, node_idx]  # [N, H, 4]
        y_s_w = y_s[mode_t, node_idx]  # [N, H, 8]

        # kd_loc: masked MSE between (gamma_xy) and (teacher loc_xy)
        student_gamma = y_s_w[..., :2]
        teacher_loc = y_t_w[..., :2]
        diff2 = (student_gamma - teacher_loc) ** 2  # [N, H, 2]
        valid_mask2 = reg_mask.unsqueeze(-1).expand_as(diff2)  # [N, H, 2]
        kd_loc = (diff2 * valid_mask2).sum() / valid_mask2.sum().clamp(min=1)

        # kd_pi: KL(student_pi || teacher_pi) over valid agents
        # pi_*: [N, F]
        t_temp = float(self.kd_pi_temperature)
        student_logp = F.log_softmax(pi_s[cls_mask] / t_temp, dim=-1)
        student_p = student_logp.exp()
        teacher_logp = F.log_softmax(pi_t[cls_mask] / t_temp, dim=-1)
        kd_pi = (student_p * (student_logp - teacher_logp)).sum(dim=-1).mean()

        kd_total = float(self.kd_loc_weight) * kd_loc + float(self.kd_pi_weight) * kd_pi
        loss = (1.0 - ka) * loss_task + ka * kd_total

        bs = int(cls_mask.sum().item()) if bool(cls_mask.any()) else int(data.num_graphs)

        # Logs
        self.log("train_reg_loss", reg_total, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
        self.log("train_edl_nll", reg_nll, on_step=True, on_epoch=True, batch_size=bs)
        self.log("train_edl_kl", reg_kl, on_step=True, on_epoch=True, batch_size=bs)
        self.log("train_cls_loss", cls_loss, on_step=True, on_epoch=True, batch_size=bs)

        self.log("train_kd_active", 1.0, on_step=True, on_epoch=True, batch_size=bs)
        self.log("train_kd_loc", kd_loc, on_step=True, on_epoch=True, batch_size=bs)
        self.log("train_kd_pi", kd_pi, on_step=True, on_epoch=True, batch_size=bs)
        self.log("train_kd_total", kd_total, on_step=True, on_epoch=True, batch_size=bs)
        self.log("kd_alpha", float(ka), on_step=False, on_epoch=True, batch_size=bs)

        return loss

