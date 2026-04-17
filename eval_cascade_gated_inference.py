# CascadedHiVT: 인퍼런스 — LocalEncoder → Small(128) → U_epi 계산 후,
#   U_epi > gate_tau 인 에이전트가 하나도 없으면 GlobalInteractor·FullDecoder 를 호출하지 않고 종료(Early exit).
#   필요 시에만 GlobalInteractor → FullDecoder 순차 실행. Latency 는 torch.cuda.Event 로 구간(ms) 합산
#   (Small: 로컬 인코더 시작~불확실도 산출까지 / Full: 글로벌 인터랙터~풀 디코더).
# all_results.jsonl: epi, fde_s, (fde_f 는 Full 실행 시만), latency_ms, is_full_path_activated 등.
# 사후 U_epi 다중 스윕(tau_min~tau_max)은 모든 배치에 fde_f 가 있을 때만(항상 Full 을 돈 옛 로그와 동일)
# counterfactual 로 의미가 있음. 조건부 분기로 Full 을 건너뛴 배치가 있으면 스윕 CSV/그래프는 생략하고
# conditional_eval_summary*.json 의 평균 지연·활성화율을 사용.
#
# minFDE(U_epi): Argoverse 등에서 흔한 표준 — 전체 예측 에이전트에 대한 총 FDE / 총 에이전트 수(에이전트 풀 평균).
# Relative Latency(U_epi): 배치(장면)별 상대 지연 %를 구한 뒤 배치 평균 — 장면별 개선 체감도에 적합하며,
#   t_fd가 장면마다 다를 때 “전체 시간 합” 방식보다 무거운 장면에 가중이 쏠리지 않음.
#
# Latency (Interaction 제외, 순수 디코더만):
#   분모 100% = t_fd  (HiVT full MLP decoder 단독)
#   분자     = t_ps  (small path) + (게이트 ON 시) t_fd
# Warm-up: 처음 N배치는 통계·지연·jsonl 기록에서 제외.
#   논문·리뷰어 대응: GPU 클럭이 초기 배치에서 불안정할 수 있으므로 **웜업 이후** 평균·요약만 보고하세요.
# CPU-only 경로: perf_counter()로 잰 ms는 GPU 비동기와 괴리될 수 있음 → **논문에는 CUDA Event 기준**만 권장.
# n_agents==0 배치: 기록·jsonl에서 제외(지연·평균이 비정상적으로 작아지는 오염 방지).
#
# 예 (epoch 26 ckpt, val 전체, 배치 1, gate τ 를 0..15 자동 스윕 — 기본 동작):
#   python eval_cascade_gated_inference.py --cuda_device 0
# 단일 τ 만 (예: 1.0):
#   python eval_cascade_gated_inference.py --cuda_device 0 --gate_tau 1
#   .venv/bin/python eval_cascade_gated_inference.py --cuda_device 3 --warmup_batches 10
# 데이터 샤딩 (GPU 4장, 동일 --out_dir, cuda_device 만 다르게):
#   python eval_cascade_gated_inference.py --cuda_device 0 --shard_idx 0 --num_shards 4 --num_workers 4
#   … --cuda_device 1 --shard_idx 1 …  /  shard_idx 2, 3
# 병합 후 후처리 (gate τ 스윕 시 shard jsonl·conditional_eval_sweep·timing_summary 가중합 →
#   CSV/PNG/PDF·activation·error_delta·고해상도 viz; --ckpt·--root·--cuda_device 권장):
#   python eval_cascade_gated_inference.py --merge_shards --out_dir ./your_output_path --ckpt ... --cuda_device 0
# 사후만 (이미 all_results.jsonl 또는 --jsonl_path):
#   .venv/bin/python eval_cascade_gated_inference.py --postprocess_only --out_dir results/paper_ksae
#
# 부가 산출: activation_by_scenario_tag.json, error_delta_gate_on.json, viz_high_delta/*.png (옵션)
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
import time
from pathlib import Path

from torch.utils.data import Subset

_ROOT = Path(__file__).resolve().parent
_ARGO = _ROOT / "datasets" / "argoverse-api-master"
sys.path.insert(0, str(_ARGO))
sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from argoverse.map_representation.map_api import ArgoverseMap  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages
from torch_geometric.loader import DataLoader

from datasets import ArgoverseV1Dataset  # noqa: E402
from models.DualHead.nig_uncertainty import (  # noqa: E402
    combine_xy_uncertainty,
    nig_epistemic_uncertainty,
)
from models.cascade import CascadedHiVT, load_cascade_from_checkpoint  # noqa: E402
from visualize import plot_hivt_scene  # noqa: E402

# 기본 U_epi 후처리 스윕(사후 재가중). 비어 있으면 tau_min..tau_max 연속 정수.
DEFAULT_TAU_LIST_STR = (
    "0,25,50,65,75,85,95,105,125,150,200,300,400,500"
)


def _parse_tau_list_csv(s: str) -> list[int]:
    """쉼표 구분 U_epi 목록 → 정수 리스트 (공백 허용)."""
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: list[int] = []
    for p in parts:
        v = float(p)
        out.append(int(round(v)))
    return out


def _resolve_u_epi_sweep(args: argparse.Namespace) -> tuple[list[int], int, int]:
    """
    후처리 U_epi 스윕: --tau_list 가 있으면 명시값만, 없으면 tau_min..tau_max 연속 정수.
    반환: (taus, sweep_min, sweep_max) — meta/CSV 축용 min·max는 리스트의 최소·최대.
    """
    tls = getattr(args, "tau_list", None)
    if tls is not None and str(tls).strip():
        taus = _parse_tau_list_csv(str(tls).strip())
        if not taus:
            raise ValueError("--tau_list 이 비어 있습니다.")
        return taus, min(taus), max(taus)
    tau_min = int(args.tau_min)
    tau_max = int(args.tau_max)
    if tau_min > tau_max:
        raise ValueError("tau_min > tau_max")
    return list(range(tau_min, tau_max + 1)), tau_min, tau_max


def _gather_best_mode(
    y_hat: torch.Tensor,
    y_agent: torch.Tensor,
    *,
    ch_loc: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    fde_modes = torch.norm(
        y_hat[:, :, -1, :ch_loc] - y_agent[:, -1].unsqueeze(0),
        p=2,
        dim=-1,
    )
    best = fde_modes.argmin(dim=0)
    b_idx = torch.arange(y_agent.size(0), device=y_hat.device, dtype=torch.long)
    pred = y_hat[best, b_idx, :, :ch_loc]
    return best, pred


def _sync_cuda(device: torch.device) -> None:
    """GPU 타이밍 전후에 반드시 호출 — 미동기화 시 Python 루프 시간만 잴 수 있음."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _cuda_elapsed_ms(ev_start: torch.cuda.Event, ev_end: torch.cuda.Event) -> float:
    """torch.cuda.Event 쌍의 경과 시간(ms). record() 직후·elapsed_time 전에 synchronize 권장."""
    return float(ev_start.elapsed_time(ev_end))


_TIMING_MS_KEYS: tuple[str, ...] = (
    "ms_local_encoder",
    "ms_proj_local",
    "ms_small_decoder",
    "ms_small_post",
    "ms_global_interactor",
    "ms_full_decoder",
)


def _mean_timing_breakdown(rows: list[dict[str, object]]) -> dict[str, float]:
    """JSONL 행들에서 구간별 ms 평균."""
    if not rows:
        return {}
    out: dict[str, float] = {}
    for k in _TIMING_MS_KEYS:
        xs = [float(r[k]) for r in rows if k in r]
        if xs:
            out[f"mean_{k}"] = sum(xs) / len(xs)
    return out


def _json_safe_float(x: float) -> float | None:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def _fde_f_list_for_json(
    fde_f_b: torch.Tensor | None,
    n: int,
) -> list[float | None]:
    if fde_f_b is None:
        return [None] * n
    return [_json_safe_float(float(x)) for x in fde_f_b.detach().cpu().tolist()]


def _records_allow_counterfactual_tau_sweep(records: list[dict[str, object]]) -> bool:
    """사후 U_epi 스윕은 모든 배치에서 Full FDE( fde_f )가 있어야 의미가 있음."""
    if not records:
        return False
    for r in records:
        if not bool(r.get("full_decoder_executed", True)):
            return False
        fde_f = r.get("fde_f_per_agent")
        if fde_f is None:
            return False
        for v in fde_f:
            if v is None:
                return False
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return False
    return True


def _focal_scenario_tag(data: object, focal_idx: int) -> str:
    """
    Argoverse v1에는 시나리오 ID가 없어, AGENT GT로 대략적 행동 태그를 씀.
    straight / left / right / unknown
    """
    if getattr(data, "y", None) is None:
        return "unknown"
    positions = data["positions"].detach().cpu().float().numpy()
    y = data.y[focal_idx, :, :2].detach().cpu().float().numpy()
    if positions.shape[0] < 20 or y.shape[0] < 2:
        return "unknown"
    v0 = positions[focal_idx, 19] - positions[focal_idx, 18]
    v_end = y[-1]
    n0 = float(np.linalg.norm(v0))
    n1 = float(np.linalg.norm(v_end))
    if n0 < 1e-6 or n1 < 1e-6:
        return "unknown"
    a0 = float(np.arctan2(v0[1], v0[0]))
    a1 = float(np.arctan2(v_end[1], v_end[0]))
    d = a1 - a0
    d = (d + math.pi) % (2 * math.pi) - math.pi
    deg = float(np.degrees(d))
    thr = 12.0
    if abs(deg) < thr:
        return "straight"
    return "left" if deg > 0 else "right"


def _seq_id_from_data(data: object) -> int:
    s = data["seq_id"]
    if isinstance(s, (int, np.integer)):
        return int(s)
    return int(s.view(-1)[0].item())


def _t_ps_pure_seconds(r: dict[str, object]) -> float:
    """
    Common cost(고정)인 Local Encoder를 제외한 가변 Small 구간 (초).
    proj_local + small_decoder + small_post(epi·FDE 등).
    """
    if all(k in r for k in ("ms_proj_local", "ms_small_decoder", "ms_small_post")):
        return (
            float(r["ms_proj_local"])
            + float(r["ms_small_decoder"])
            + float(r["ms_small_post"])
        ) / 1000.0
    if "t_ps" in r and "ms_local_encoder" in r:
        return max(float(r["t_ps"]) - float(r["ms_local_encoder"]) / 1000.0, 0.0)
    raise ValueError(
        "t_ps_pure 계산 불가: ms_proj_local·ms_small_decoder·ms_small_post "
        "또는 t_ps+ms_local_encoder 가 필요합니다."
    )


def _compute_tau_metrics_from_records(
    records: list[dict[str, object]],
    taus: list[int],
) -> tuple[
    list[float],
    list[float],
    list[float],
    list[float],
    float,
    float,
    int,
    int,
]:
    """
    all_results.jsonl 한 줄(배치)마다 저장된 epi, fde_s/a, fde_f/a, t_ps, t_fd로
    각 U_epi 임계값에 대한 평균 minFDE·상대 지연·Full 활성화율을 순수 파이썬/넘파이로 계산.

    Relative decoder latency (%), 레거시: 분모 = t_fd. 분자 = 게이트 ON이면 t_ps+t_fd, OFF면 t_ps.

    Relative variable latency (%), 권장 (Common vs. Variable):
    분모 = t_fd (가변 비용 기준 100% = Full branch만).
    분자 = t_ps_pure + (needs_full × t_fd), 여기서 t_ps_pure는 Local Encoder를 뺀 Small+epi 구간.

    t_ps/t_fd: 인퍼런스 루프에서 data.to(device)·rotate·encode 이후,
    각각 small_interaction_and_decoder / full_interaction_and_decoder 호출만 perf_counter로 재며,
    구간 끝에 cuda.synchronize()로 GPU 커널 완료를 보장함 (FDE·epi 계산은 타이밍 밖).
    """
    nb = len(records)
    if nb == 0:
        raise ValueError("records 비어 있음")
    if "fde_s_per_agent" not in records[0] or "t_ps" not in records[0]:
        raise ValueError(
            "all_results.jsonl 형식이 옛날 버전입니다. "
            "fde_s_per_agent, fde_f_per_agent, t_ps, t_fd, n_agents 필드가 필요합니다."
        )
    n_agents = sum(int(r["n_agents"]) for r in records)
    if n_agents < 1:
        raise ValueError("n_agents 합이 0")

    sum_s = 0.0
    sum_f = 0.0
    for r in records:
        sum_s += sum(float(x) for x in r["fde_s_per_agent"])
        sum_f += sum(float(x) for x in r["fde_f_per_agent"])
    mean_fde_small = sum_s / n_agents
    mean_fde_full = sum_f / n_agents

    n_t = len(taus)
    sum_fde = [0.0] * n_t
    sum_rel = [0.0] * n_t
    sum_rel_var = [0.0] * n_t
    n_full_act = [0] * n_t

    for r in records:
        epi = np.asarray(r["epi_per_agent"], dtype=np.float64)
        fde_s = np.asarray(r["fde_s_per_agent"], dtype=np.float64)
        fde_f = np.asarray(r["fde_f_per_agent"], dtype=np.float64)
        t_ps = float(r["t_ps"])
        t_fd = float(r["t_fd"])
        t_ps_pure = _t_ps_pure_seconds(r)
        denom = max(t_fd, 1e-12)
        for ti, tau in enumerate(taus):
            tf = float(tau)
            # U_epi=0: "불확실성>0이면 Full"이면 epi==0 인 에이전트는 small FDE가 섞여 HiVT(전원 Full)와 불일치.
            # τ=0은 baseline 정합을 위해 항상 Full FDE·(t_ps+t_fd) 지연으로 집계.
            if tf == 0.0:
                needs_full = True
                rel_pct = ((t_ps + t_fd) / denom) * 100.0
                rel_var_pct = ((t_ps_pure + t_fd) / denom) * 100.0
                fde_c = fde_f
            else:
                needs_full = bool(np.any(epi > tf))
                rel_pct = (
                    ((t_ps + t_fd) / denom) * 100.0 if needs_full else (t_ps / denom) * 100.0
                )
                num_v = t_ps_pure + (t_fd if needs_full else 0.0)
                rel_var_pct = (num_v / denom) * 100.0
                fde_c = np.where(epi > tf, fde_f, fde_s)
            sum_fde[ti] += float(fde_c.sum())
            sum_rel[ti] += rel_pct
            sum_rel_var[ti] += rel_var_pct
            if needs_full:
                n_full_act[ti] += 1

    mean_fde_list = [sum_fde[ti] / n_agents for ti in range(n_t)]
    rel_lat_list = [sum_rel[ti] / nb for ti in range(n_t)]
    rel_variable_lat_list = [sum_rel_var[ti] / nb for ti in range(n_t)]
    pct_list = [100.0 * n_full_act[ti] / nb for ti in range(n_t)]

    return (
        mean_fde_list,
        rel_lat_list,
        rel_variable_lat_list,
        pct_list,
        mean_fde_small,
        mean_fde_full,
        n_agents,
        nb,
    )


def _compute_per_batch_U_epi_detail(
    records: list[dict[str, object]],
    taus: list[int],
) -> list[dict[str, object]]:
    """
    배치(장면)마다 U_epi 임계값별 게이트 여부·합산 FDE·상대 지연을 저장 (_compute_tau_metrics_from_records 와 동일 규칙).
    """
    out: list[dict[str, object]] = []
    for r in records:
        epi = np.asarray(r["epi_per_agent"], dtype=np.float64)
        fde_s = np.asarray(r["fde_s_per_agent"], dtype=np.float64)
        fde_f = np.asarray(r["fde_f_per_agent"], dtype=np.float64)
        t_ps = float(r["t_ps"])
        t_fd = float(r["t_fd"])
        t_ps_pure = _t_ps_pure_seconds(r)
        denom = max(t_fd, 1e-12)
        n_agents = int(r["n_agents"])
        per_u: dict[str, object] = {}
        for tau in taus:
            tf = float(tau)
            if tf == 0.0:
                needs_full = True
                rel_pct = ((t_ps + t_fd) / denom) * 100.0
                rel_var_pct = ((t_ps_pure + t_fd) / denom) * 100.0
                fde_c = fde_f
            else:
                needs_full = bool(np.any(epi > tf))
                rel_pct = (
                    ((t_ps + t_fd) / denom) * 100.0
                    if needs_full
                    else (t_ps / denom) * 100.0
                )
                num_v = t_ps_pure + (t_fd if needs_full else 0.0)
                rel_var_pct = (num_v / denom) * 100.0
                fde_c = np.where(epi > tf, fde_f, fde_s)
            sum_fde = float(fde_c.sum())
            per_u[str(tau)] = {
                "any_full_path": needs_full,
                "sum_fde_m": sum_fde,
                "mean_fde_m": sum_fde / max(n_agents, 1),
                "relative_decoder_latency_pct_vs_hivt": rel_pct,
                "relative_variable_latency_pct_vs_fd": rel_var_pct,
            }
        out.append(
            {
                "seq_id": r["seq_id"],
                "batch_idx": r["batch_idx"],
                "n_agents": n_agents,
                "scenario_tag": r["scenario_tag"],
                "t_ps": t_ps,
                "t_fd": t_fd,
                "per_U_epi": per_u,
            }
        )
    return out


def _build_cascade_y_hat(
    y_s: torch.Tensor,
    y_f: torch.Tensor,
    ai: torch.Tensor,
    epi: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """게이트 마스크로 small/full 예측을 합친 [M, N, T, 2] (plot_hivt_scene용)."""
    n = int(y_s.size(1))
    gate = torch.zeros(n, device=y_s.device, dtype=torch.bool)
    gate[ai] = epi > float(tau)
    m = gate.view(1, n, 1, 1)
    return torch.where(m, y_f[..., :2], y_s[..., :2])


def _scene_plot_overrides(seq_id: int) -> dict[str, object]:
    if int(seq_id) == 21883:
        return {
            "title_scale": 1.0,
            "legend_axis_scale": 2.0,
            "actor_label_scale": 2.0,
            "agent_marker_scale": 2.0,
            "only_agent_av_paths": True,
            "force_xlim": (-30.0, 40.0),
            "force_ylim": (-30.0, 40.0),
        }
    if int(seq_id) == 1153:
        return {
            "title_scale": 1.0,
            "legend_axis_scale": 2.0,
            "actor_label_scale": 2.0,
            "agent_marker_scale": 2.0,
            "only_agent_av_paths": True,
            "force_xlim": (-10.0, 40.0),
            "force_ylim": (-20.0, 30.0),
        }
    return {}


@torch.no_grad()
def _viz_cascade_scene(
    model: CascadedHiVT,
    data: object,
    device: torch.device,
    tau: float,
    out_path: Path,
    am: ArgoverseMap,
    dataset_root: str,
) -> None:
    data = data.to(device)
    model._apply_rotate(data)
    loc, glo = model.encode(data)
    y_s, pi_s = model.small_interaction_and_decoder(data, loc, glo)
    y_f, _pi_f = model.full_interaction_and_decoder(data, loc, glo)
    ai = CascadedHiVT.absolute_agent_indices(data, device).view(-1)
    y_agent = data.y[ai]
    y_s_a = y_s.index_select(1, ai)
    best_s, _ = _gather_best_mode(y_s_a, y_agent, ch_loc=2)
    b_idx = torch.arange(ai.numel(), device=device, dtype=torch.long)
    y_last = y_s_a[best_s, b_idx, -1, :]
    u_xy = nig_epistemic_uncertainty(y_last)
    epi = combine_xy_uncertainty(u_xy, reduction="l2")
    y_hat = _build_cascade_y_hat(y_s, y_f, ai, epi, tau)
    sid = data["seq_id"]
    seq_id = int(sid) if isinstance(sid, (int, np.integer)) else int(sid.view(-1)[0].item())
    title = f"CascadedHiVT | seq {seq_id} | U_epi thr={tau}"
    scene_kw = _scene_plot_overrides(seq_id)
    d_cpu = data.cpu()
    for _k in ("seq_id", "agent_index", "av_index"):
        if _k in d_cpu and not torch.is_tensor(d_cpu[_k]):
            v = int(d_cpu[_k])
            d_cpu[_k] = torch.tensor([v], dtype=torch.long)
    plot_hivt_scene(
        am,
        d_cpu,
        y_hat.cpu(),
        out_path,
        title=title,
        pi=pi_s.cpu(),
        dataset_root=dataset_root,
        split="val",
        **scene_kw,
    )


@torch.no_grad()
def _viz_cascade_scene_small_only(
    model: CascadedHiVT,
    data: object,
    device: torch.device,
    out_path: Path,
    am: ArgoverseMap,
    dataset_root: str,
    *,
    u_epi_focal: float,
    minFDE_small_focal: float,
) -> None:
    """
    Small path만 시각화.
    title 포맷 요구: Small Path(U_epi=..., minFDE=...) 형태.
    """
    data = data.to(device)
    model._apply_rotate(data)
    loc, glo = model.encode(data)
    y_s, pi_s = model.small_interaction_and_decoder(data, loc, glo)

    # y_s: [M, N, T, 8] (NIG params 포함) -> plot용 xy mean만
    y_hat_small = y_s[..., :2]

    seq_id = _seq_id_from_data(data)
    title = (
        f"CascadedHiVT | seq {seq_id} | "
        f"Small Path(U_epi={u_epi_focal:.3f}, minFDE={minFDE_small_focal:.3f}m)"
    )
    scene_kw = _scene_plot_overrides(seq_id)

    d_cpu = data.cpu()
    for _k in ("seq_id", "agent_index", "av_index"):
        if _k in d_cpu and not torch.is_tensor(d_cpu[_k]):
            v = int(d_cpu[_k])
            d_cpu[_k] = torch.tensor([v], dtype=torch.long)

    plot_hivt_scene(
        am,
        d_cpu,
        y_hat_small.cpu(),
        out_path,
        title=title,
        pi=pi_s.cpu(),
        dataset_root=dataset_root,
        split="val",
        **scene_kw,
    )


@torch.no_grad()
def _viz_cascade_scene_small_and_gated_pair(
    model: CascadedHiVT,
    data: object,
    device: torch.device,
    tau: float,
    out_small_path: Path,
    out_gated_path: Path,
    am: ArgoverseMap,
    dataset_root: str,
    *,
    u_epi_focal: float,
    minFDE_small_focal: float,
    minFDE_full_focal: float,
    out_gated_extra_path: Path | None = None,
) -> None:
    """
    Small-only + gating 결과(Full이 선택된 상황)를 쌍으로 저장.

    title 포맷 요구:
      Small Path(U_epi=..., minFDE=...m) / Full Path(minFDE=...m)
    """
    data = data.to(device)
    model._apply_rotate(data)
    loc, glo = model.encode(data)
    y_s, pi_s = model.small_interaction_and_decoder(data, loc, glo)
    y_f, _pi_f = model.full_interaction_and_decoder(data, loc, glo)

    ai = CascadedHiVT.absolute_agent_indices(data, device).view(-1)
    y_agent = data.y[ai]
    y_s_a = y_s.index_select(1, ai)
    best_s, _ = _gather_best_mode(y_s_a, y_agent, ch_loc=2)
    b_idx = torch.arange(ai.numel(), device=device, dtype=torch.long)
    y_last = y_s_a[best_s, b_idx, -1, :]
    u_xy = nig_epistemic_uncertainty(y_last)
    epi = combine_xy_uncertainty(u_xy, reduction="l2")

    y_hat_small = y_s[..., :2]
    y_hat_gated = _build_cascade_y_hat(y_s, y_f, ai, epi, tau)

    seq_id = _seq_id_from_data(data)
    title = (
        f"CascadedHiVT | seq {seq_id}\n"
        f"Small Path(U_epi={u_epi_focal:.3f}, minFDE={minFDE_small_focal:.3f}m)\n"
        f"Full Path(minFDE={minFDE_full_focal:.3f}m)"
    )
    scene_kw = _scene_plot_overrides(seq_id)

    d_cpu = data.cpu()
    for _k in ("seq_id", "agent_index", "av_index"):
        if _k in d_cpu and not torch.is_tensor(d_cpu[_k]):
            v = int(d_cpu[_k])
            d_cpu[_k] = torch.tensor([v], dtype=torch.long)

    plot_hivt_scene(
        am,
        d_cpu,
        y_hat_small.cpu(),
        out_small_path,
        title=title,
        pi=pi_s.cpu(),
        dataset_root=dataset_root,
        split="val",
        **scene_kw,
    )
    plot_hivt_scene(
        am,
        d_cpu,
        y_hat_gated.cpu(),
        out_gated_path,
        title=title,
        pi=pi_s.cpu(),
        dataset_root=dataset_root,
        split="val",
        **scene_kw,
    )
    if out_gated_extra_path is not None:
        plot_hivt_scene(
            am,
            d_cpu,
            y_hat_gated.cpu(),
            out_gated_extra_path,
            title=title,
            pi=pi_s.cpu(),
            dataset_root=dataset_root,
            split="val",
            **scene_kw,
        )


def _shard_range(n_dataset: int, num_shards: int, shard_idx: int) -> tuple[int, int]:
    """전체 val 인덱스를 num_shards로 나눈 뒤, half-open 구간 [start, end) 반환."""
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not (0 <= shard_idx < num_shards):
        raise ValueError(f"shard_idx must be in [0, {num_shards}), got {shard_idx}")
    base = n_dataset // num_shards
    rem = n_dataset % num_shards
    if shard_idx < rem:
        start = shard_idx * (base + 1)
        end = start + (base + 1)
    else:
        start = rem * (base + 1) + (shard_idx - rem) * base
        end = start + base
    return start, end


def _shard_file_suffix(shard_idx: int, num_shards: int) -> str:
    """num_shards==1 이면 접미사 없음; 그 외에는 _shard_{idx}."""
    return "" if num_shards <= 1 else f"_shard_{shard_idx}"


def _merge_gate_tau_sharded_jsonl(out_dir: Path) -> Path | None:
    """
    all_results_gate_tau_{τ}_shard_{k}.jsonl 를 τ별로 모아 shard 순서대로 이어 붙여
    all_results_gate_tau_{τ}_merged.jsonl 생성.
    """
    pat = re.compile(r"^all_results_gate_tau_(\d+)_shard_(\d+)\.jsonl$")
    groups: dict[int, list[tuple[int, Path]]] = defaultdict(list)
    for p in out_dir.iterdir():
        if not p.is_file():
            continue
        m = pat.match(p.name)
        if m:
            tau = int(m.group(1))
            sk = int(m.group(2))
            groups[tau].append((sk, p))
    if not groups:
        return None
    for tau in sorted(groups):
        parts = sorted(groups[tau], key=lambda x: x[0])
        merged = out_dir / f"all_results_gate_tau_{tau}_merged.jsonl"
        with merged.open("w", encoding="utf-8") as out_f:
            for _sk, path in parts:
                with path.open(encoding="utf-8") as inf:
                    for line in inf:
                        line = line.strip()
                        if line:
                            out_f.write(line + "\n")
        print(f"[Merge gate τ={tau}] {len(parts)} shard(s) -> {merged.name}")
    primary = out_dir / "all_results_gate_tau_0_merged.jsonl"
    if primary.is_file():
        return primary
    return out_dir / f"all_results_gate_tau_{sorted(groups.keys())[0]}_merged.jsonl"


def _merge_conditional_eval_sweep_summaries_shards(out_dir: Path) -> None:
    """conditional_eval_sweep_summary_shard_*.json 을 n_samples 가중 평균으로 하나로 합침."""
    pat = re.compile(r"^conditional_eval_sweep_summary_shard_(\d+)\.json$")
    shard_data: dict[int, dict[str, object]] = {}
    for p in out_dir.iterdir():
        if not p.is_file():
            continue
        m = pat.match(p.name)
        if m:
            shard_data[int(m.group(1))] = json.loads(p.read_text(encoding="utf-8"))
    if len(shard_data) < 2:
        return
    by_tau: dict[float, list[dict[str, object]]] = defaultdict(list)
    for k in sorted(shard_data.keys()):
        for entry in shard_data[k]["per_gate_tau"]:  # type: ignore[index]
            e = entry  # type: ignore[assignment]
            by_tau[float(e["gate_tau"])].append(e)
    if not by_tau:
        return
    merged_entries: list[dict[str, object]] = []
    for gt in sorted(by_tau.keys()):
        chunks = by_tau[gt]
        n_tot = sum(int(c.get("n_samples", 0)) for c in chunks)
        if n_tot < 1:
            merged_entries.append(chunks[0])
            continue
        out_d: dict[str, object] = {"gate_tau": gt, "n_samples": n_tot}
        w_lat = 0.0
        w_lat_n = 0
        w_rate = 0.0
        w_rate_n = 0
        for c in chunks:
            ns = int(c.get("n_samples", 0))
            if ns <= 0:
                continue
            if "average_latency_ms" in c:
                w_lat += float(c["average_latency_ms"]) * ns
                w_lat_n += ns
            if "full_path_activation_rate" in c:
                w_rate += float(c["full_path_activation_rate"]) * ns
                w_rate_n += ns
        if w_lat_n > 0:
            out_d["average_latency_ms"] = w_lat / w_lat_n
        if w_rate_n > 0:
            out_d["full_path_activation_rate"] = w_rate / w_rate_n
        if chunks:
            out_d["samples_jsonl"] = chunks[0].get("samples_jsonl")
        merged_entries.append(out_d)
    g0 = shard_data[sorted(shard_data.keys())[0]]
    out_path = out_dir / "conditional_eval_sweep_summary_merged.json"
    out_path.write_text(
        json.dumps(
            {
                "merged_from_shards": sorted(shard_data.keys()),
                "gate_tau_min": g0.get("gate_tau_min"),
                "gate_tau_max": g0.get("gate_tau_max"),
                "per_gate_tau": merged_entries,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Merged conditional sweep summaries -> {out_path.name}")


def _merge_timing_summary_gate_tau_shards(out_dir: Path) -> None:
    """
    timing_summary_gate_tau_{τ}_shard_{k}.json 을 τ별로 n_batches_recorded 가중 평균해
    timing_summary_gate_tau_{τ}_merged.json 생성.
    """
    pat = re.compile(r"^timing_summary_gate_tau_(\d+)_shard_(\d+)\.json$")
    groups: dict[int, list[tuple[int, Path]]] = defaultdict(list)
    for p in out_dir.iterdir():
        if not p.is_file():
            continue
        m = pat.match(p.name)
        if m:
            tau = int(m.group(1))
            sk = int(m.group(2))
            groups[tau].append((sk, p))
    if not groups:
        return
    for tau in sorted(groups):
        parts = sorted(groups[tau], key=lambda x: x[0])
        total_n = 0
        wsum: dict[str, float] = {}
        warmup_excl = 0
        for _sk, path in parts:
            data = json.loads(path.read_text(encoding="utf-8"))
            n = int(data.get("n_batches_recorded", 0))
            if n <= 0:
                continue
            total_n += n
            warmup_excl = int(data.get("warmup_batches_excluded", warmup_excl))
            for k, v in data.items():
                if k.startswith("mean_") and isinstance(v, (int, float)):
                    wsum[k] = wsum.get(k, 0.0) + float(v) * n
        if total_n <= 0:
            continue
        out_d: dict[str, object] = {
            "gate_tau": float(tau),
            "n_batches_recorded": total_n,
            "warmup_batches_excluded": warmup_excl,
            "all_results_jsonl": f"all_results_gate_tau_{tau}_merged.jsonl",
            "merged_from_shards": [p[0] for p in parts],
            "note": "mean_* 는 샤드별 mean에 n_batches_recorded 가중 평균",
        }
        for k, s in wsum.items():
            out_d[k] = s / float(total_n)
        merged = out_dir / f"timing_summary_gate_tau_{tau}_merged.json"
        merged.write_text(json.dumps(out_d, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[Merge timing τ={tau}] {len(parts)} shard(s) -> {merged.name}")


def _merge_all_results_shards(out_dir: Path) -> Path:
    """out_dir 내 샤드 JSONL을 합침. gate τ 스윕 형식 우선, 없으면 all_results_shard_*.jsonl."""
    gt_path = _merge_gate_tau_sharded_jsonl(out_dir)
    if gt_path is not None:
        _merge_conditional_eval_sweep_summaries_shards(out_dir)
        _merge_timing_summary_gate_tau_shards(out_dir)
        return gt_path

    pat = re.compile(r"^all_results_shard_(\d+)\.jsonl$")
    found: list[tuple[int, Path]] = []
    for p in out_dir.iterdir():
        if p.is_file():
            m = pat.match(p.name)
            if m:
                found.append((int(m.group(1)), p))
    if not found:
        raise FileNotFoundError(
            f"{out_dir} 에 all_results_gate_tau_*_shard_*.jsonl 또는 all_results_shard_*.jsonl 이 없습니다."
        )
    found.sort(key=lambda x: x[0])
    merged = out_dir / "all_results_merged.jsonl"
    with merged.open("w", encoding="utf-8") as out_f:
        for _si, path in found:
            with path.open(encoding="utf-8") as inf:
                for line in inf:
                    line = line.strip()
                    if line:
                        out_f.write(line + "\n")
    print(f"Merged {len(found)} shard file(s) -> {merged.name}")
    return merged


def _load_jsonl_records(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(
        description="게이팅 캐스케이드 인퍼런스. Latency는 cuda.synchronize() 포함 구간에서만 합산."
    )
    p.add_argument(
        "--ckpt",
        type=str,
        default=str(
            _ROOT
            / "checkpoints"
            / "trash"
            / "CascadedHiVT"
            / "cascade-epoch=26-val_minFDE=val_minFDE=3.1726.ckpt"
        ),
        help="CascadedHiVT Lightning 체크포인트 (.ckpt)",
    )
    p.add_argument("--root", type=str, default=str(_ROOT / "datasets" / "argoverse_v1"))
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(_ROOT / "results" / "cascade_epoch26_val_full_tau0_15"),
        help="all_results.jsonl, τ 스윕 CSV/그래프, 배치×U_epi 상세 JSONL 등 저장 디렉터리",
    )
    p.add_argument("--cuda_device", type=int, default=None)
    p.add_argument(
        "--warmup_batches",
        type=int,
        default=10,
        help="처음 N개 배치는 통계·jsonl·timing_summary에서 제외 (GPU 클럭·커널 워밍업; 논문은 이후 구간만)",
    )
    p.add_argument(
        "--tau_min",
        type=int,
        default=0,
        help="U_epi 연속 스윕 하한 (--tau_list 를 비웠을 때만 사용)",
    )
    p.add_argument(
        "--tau_max",
        type=int,
        default=15,
        help="U_epi 연속 스윕 상한 (--tau_list 를 비웠을 때만 사용)",
    )
    p.add_argument(
        "--tau_list",
        type=str,
        default=DEFAULT_TAU_LIST_STR,
        help=(
            "U_epi 후처리 스윕 (쉼표 구분). 기본은 희소 목록. "
            "연속 정수 0..15만 쓰려면 --tau-list \"\" --tau_min 0 --tau_max 15. "
            "비어 있지 않으면 물리 인퍼런스는 gate τ=0 한 번(단, --gate_tau 직접 지정 시 예외)."
        ),
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="DataLoader 배치 크기 — 온디바이스 조건부 분기 평가는 1만 허용",
    )
    p.add_argument(
        "--gate_tau",
        type=float,
        default=None,
        help=(
            "이 값만 단일 실행. 미지정 시 --gate_tau_min ~ --gate_tau_max 정수를 하나씩 올리며 "
            "val 전체를 반복 실행 (물리적 분기·지연이 τ마다 다름)."
        ),
    )
    p.add_argument(
        "--gate_tau_min",
        type=int,
        default=0,
        help="--gate_tau 미지정일 때 스윕 하한(포함)",
    )
    p.add_argument(
        "--gate_tau_max",
        type=int,
        default=15,
        help="--gate_tau 미지정일 때 스윕 상한(포함)",
    )
    p.add_argument(
        "--limit_batches",
        type=int,
        default=None,
        help="디버그용 val 배치 상한 (미지정 시 val 전체)",
    )
    p.add_argument(
        "--no_save_per_batch_U_epi_detail",
        action="store_true",
        help="배치×U_epi 상세 JSONL(per_batch_U_epi_detail*.jsonl) 저장 생략",
    )
    p.add_argument(
        "--fde_vs_small_pct_max",
        type=float,
        default=1.0,
        help="요약 후보: Small-only 대비 FDE 상대 손실(%%)이 이 값 미만인 U_epi 임계값만",
    )
    p.add_argument(
        "--latency_ylim",
        type=str,
        default="0,130",
        help=(
            "오른쪽 Relative Variable Latency (%) 축 'ymin,ymax' (고정). "
            "곡선이 잘리면 값을 넓히세요."
        ),
    )
    p.add_argument(
        "--tau_xlim",
        type=str,
        default="0,150",
        help="그래프 $U_{epi}$ 축 xmin,xmax (세 축 공통).",
    )
    p.add_argument(
        "--analysis_U_epi",
        dest="analysis_u_epi",
        type=int,
        default=None,
        help="에피·태그·Error Delta 집계에 쓸 U_epi 임계값 (미지정 시 minFDE 최소인 U_epi)",
    )
    p.add_argument(
        "--max_viz_png",
        type=int,
        default=24,
        help="Error Delta 평균 이상인 게이트-온 장면 중 저장할 최대 PNG 수",
    )
    p.add_argument(
        "--no_viz",
        action="store_true",
        help="고해상도 시각화 PNG 저장 생략",
    )
    p.add_argument(
        "--viz_seq_ids_dir",
        type=str,
        default=None,
        help=(
            "고해상도 시각화에서 특정 seq_id들만 처리. "
            "지정한 폴더 하위 *.png 파일명에서 seq<id>를 스캔해서 사용 "
            "(예: viz_high_delta_shard_0/ 또는 viz_high_delta/)."
        ),
    )
    p.add_argument(
        "--postprocess_only",
        action="store_true",
        help="JSONL만 읽어 U_epi 스윕·CSV·그래프 재생성 (모델 인퍼런스 없음). 입력은 --jsonl_path 또는 out_dir/all_results.jsonl",
    )
    p.add_argument(
        "--jsonl_path",
        type=str,
        default=None,
        help="postprocess_only 또는 후처리 입력 JSONL 절대/상대 경로 (예: all_results_merged.jsonl)",
    )
    p.add_argument(
        "--shard_idx",
        type=int,
        default=0,
        help="데이터 샤딩: 현재 프로세스가 맡을 조각 번호 [0, num_shards)",
    )
    p.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="데이터 샤딩: 전체 프로세스/GPU 개수 (1이면 샤딩 없음)",
    )
    p.add_argument(
        "--merge_shards",
        action="store_true",
        help="out_dir 내 all_results_shard_*.jsonl 을 합쳐 all_results_merged.jsonl 생성 후 전체 후처리",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader num_workers (CPU 데이터 로딩 병렬)",
    )
    p.add_argument(
        "--print_decoder_timing",
        action="store_true",
        help="각 배치마다 구간별 ms 출력(le/pj/sd/post·gi/fd) — JSONL에는 항상 ms_* 필드로 저장",
    )
    args = p.parse_args()

    assert int(args.batch_size) == 1, (
        "온디바이스 조건부 분기·지연 측정은 batch_size=1 만 지원합니다."
    )

    # --- viz seq_id 제한(옵션) ---
    viz_seq_ids: set[int] | None = None
    if args.viz_seq_ids_dir:
        viz_dir = Path(args.viz_seq_ids_dir).expanduser().resolve()
        if not viz_dir.exists():
            raise FileNotFoundError(f"--viz_seq_ids_dir not found: {viz_dir}")
        pat = re.compile(r"seq(\d+)", re.IGNORECASE)
        ids: set[int] = set()
        for png in viz_dir.rglob("*.png"):
            # png.name만 보면 small_path.png처럼 seq가 없는 파일명은 누락될 수 있어,
            # 경로 전체에서 seq(\d+)를 찾는다.
            m = pat.search(str(png))
            if m:
                ids.add(int(m.group(1)))
        viz_seq_ids = ids
        print(f"[viz_seq_ids] loaded {len(viz_seq_ids)} seq_id(s) from {viz_dir}")

    # tau_list 로 사후 스윕할 때는 물리 인퍼런스를 τ=0 한 번만 (--gate_tau 미지정 시).
    if (
        getattr(args, "tau_list", None) is not None
        and str(args.tau_list).strip()
        and args.gate_tau is None
    ):
        args.gate_tau_min = 0
        args.gate_tau_max = 0

    if args.gate_tau is not None:
        gate_tau_sweep_list: list[float] = [float(args.gate_tau)]
    else:
        g0, g1 = int(args.gate_tau_min), int(args.gate_tau_max)
        if g0 > g1:
            raise ValueError("gate_tau_min > gate_tau_max")
        gate_tau_sweep_list = [float(t) for t in range(g0, g1 + 1)]

    ckpt = Path(args.ckpt).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    num_shards = int(args.num_shards)
    shard_idx = int(args.shard_idx)
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not (0 <= shard_idx < num_shards):
        raise ValueError(f"shard_idx must be in [0, {num_shards}), got {shard_idx}")

    shard_suffix = _shard_file_suffix(shard_idx, num_shards)
    # 병합 후 산출물은 접미사 없이 최종 파일명 사용
    output_suffix = "" if args.merge_shards else shard_suffix

    if args.merge_shards and args.postprocess_only:
        raise ValueError("--merge_shards 와 --postprocess_only 는 함께 쓰지 마세요.")

    if not args.merge_shards and not args.postprocess_only and not ckpt.is_file():
        raise FileNotFoundError(ckpt)

    per_batch_records: list[dict[str, object]] = []
    wall_s = 0.0
    model: CascadedHiVT | None = None
    val_ds: ArgoverseV1Dataset | None = None
    device: torch.device | None = None
    am: ArgoverseMap | None = None
    merged_from_shards = False

    if args.merge_shards:
        merged_path = _merge_all_results_shards(out_dir)
        per_batch_records = _load_jsonl_records(merged_path)
        all_results_path = merged_path
        merged_from_shards = True
        _tau_desc = (
            f"tau_list={args.tau_list!r}"
            if getattr(args, "tau_list", None) and str(args.tau_list).strip()
            else f"[{args.tau_min}, {args.tau_max}]"
        )
        print(
            f"[Merge] loaded {len(per_batch_records)} lines from {all_results_path.name}; "
            f"U_epi 후처리 {_tau_desc}"
        )
        # 샤드 병합만 한 경우에도 viz_high_delta PNG 를 쓰려면 ckpt·val_ds 필요
        if not args.no_viz and ckpt.is_file():
            if args.cuda_device is not None:
                device = torch.device(f"cuda:{int(args.cuda_device)}")
                torch.cuda.set_device(device)
            else:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = load_cascade_from_checkpoint(
                str(ckpt),
                map_location=torch.device("cpu"),
                weights_only=False,
            )
            model.eval()
            model.to(device)
            val_ds = ArgoverseV1Dataset(
                root=args.root,
                split="val",
                local_radius=float(model.hparams.local_radius),
            )
            am = ArgoverseMap()
            print(
                "[Merge] ckpt 로드 완료 — activation/error_delta/고해상도 viz 등 동일 후처리 진행"
            )
    elif args.postprocess_only:
        if args.jsonl_path:
            all_results_path = Path(args.jsonl_path).expanduser().resolve()
        else:
            all_results_path = out_dir / "all_results.jsonl"
        if not all_results_path.is_file():
            raise FileNotFoundError(
                f"없음: {all_results_path} — --jsonl_path 를 지정하거나 인퍼런스를 먼저 실행하세요."
            )
        per_batch_records = _load_jsonl_records(all_results_path)
        _tau_desc2 = (
            f"tau_list={args.tau_list!r}"
            if getattr(args, "tau_list", None) and str(args.tau_list).strip()
            else f"[{args.tau_min}, {args.tau_max}]"
        )
        print(
            f"[Post-process only] loaded {len(per_batch_records)} lines from {all_results_path.name}; "
            f"U_epi {_tau_desc2} 재계산"
        )
    elif not args.postprocess_only:
        if args.cuda_device is not None:
            device = torch.device(f"cuda:{int(args.cuda_device)}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = load_cascade_from_checkpoint(
            str(ckpt),
            map_location=torch.device("cpu"),
            weights_only=False,
        )
        model.eval()
        model.to(device)

        val_ds = ArgoverseV1Dataset(
            root=args.root,
            split="val",
            local_radius=float(model.hparams.local_radius),
        )
        n_dataset = len(val_ds)
        start, end = _shard_range(n_dataset, num_shards, shard_idx)
        if args.limit_batches is not None:
            lim = int(args.limit_batches)
            end = min(end, lim)
        if start >= end:
            raise RuntimeError(
                f"빈 샤드: indices [{start}, {end}) — limit_batches·shard 설정을 확인하세요."
            )
        print(
            f"Running Shard {shard_idx}/{num_shards}: Indices {start} to {end - 1} "
            f"(inclusive), n={end - start}"
        )

        shard_indices = list(range(start, end))
        subset_ds = Subset(val_ds, shard_indices)
        pin_mem = device.type == "cuda"
        nw = int(args.num_workers)
        bs = int(args.batch_size)
        _dl_kw: dict[str, object] = dict(
            batch_size=bs,
            shuffle=False,
            num_workers=nw,
            pin_memory=pin_mem,
        )
        if nw > 0:
            _dl_kw["persistent_workers"] = True
        loader = DataLoader(subset_ds, **_dl_kw)

        n_total_batches = end - start
        print(
            f"Val batches (this shard): {n_total_batches} / dataset {n_dataset} | "
            f"warmup {args.warmup_batches} excluded from metrics & jsonl"
        )
        if len(gate_tau_sweep_list) > 1:
            print(
                f"[Gate τ sweep] {len(gate_tau_sweep_list)}회 반복: "
                f"τ = {gate_tau_sweep_list[0]:.0f} … {gate_tau_sweep_list[-1]:.0f} "
                f"(각 τ마다 물리적 분기·지연 측정)"
            )
        else:
            print(f"[Gate τ] 단일 실행 τ = {gate_tau_sweep_list[0]:g}")

        am = ArgoverseMap()
        sweep_summaries: list[dict[str, object]] = []
        wall_s_total = 0.0
        all_results_path = out_dir / f"all_results{shard_suffix}.jsonl"

        # seq-only viz 시, 필요한 seq를 모두 찾으면 DataLoader 루프를 조기 종료
        found_viz_seqs: set[int] = set()

        def run_inference_batch(
            data: object,
            record: bool,
            batch_idx: int,
            gate_tau_val: float,
        ) -> None:
            nonlocal per_batch_records, conditional_eval_samples, found_viz_seqs
            ng = int(getattr(data, "num_graphs", 1))
            assert ng == 1, f"batch_size=1 그래프 1개만 지원 (num_graphs={ng})"

            ai_cpu = data["agent_index"]
            focal = (
                int(ai_cpu)
                if isinstance(ai_cpu, (int, np.integer))
                else int(ai_cpu.view(-1)[0].item())
            )
            scenario_tag = _focal_scenario_tag(data, focal)
            seq_id = _seq_id_from_data(data)
            viz_this_seq = (viz_seq_ids is None) or (seq_id in viz_seq_ids)
            record_effective = bool(record and viz_this_seq)

            # 요청: 시각화 대상 seq가 아니면 forward/timing/추가 기록을 완전히 스킵
            # (warmup 구간이라 record=False여도 동일하게 스킵)
            if not record_effective:
                return

            # record_effective=True 인 경우에만 시각화 대상으로 채택
            found_viz_seqs.add(seq_id)

            data = data.to(device)
            _sync_cuda(device)
            model._apply_rotate(data)
            _sync_cuda(device)

            use_cuda_timer = device.type == "cuda" and torch.cuda.is_available()
            small_ms = 0.0
            full_ms = 0.0
            ms_local_encoder = 0.0
            ms_proj_local = 0.0
            ms_small_decoder = 0.0
            ms_small_post = 0.0
            ms_global_interactor = 0.0
            ms_full_decoder = 0.0

            if use_cuda_timer:
                ev_s0 = torch.cuda.Event(enable_timing=True)
                ev_s1 = torch.cuda.Event(enable_timing=True)
                ev_s2 = torch.cuda.Event(enable_timing=True)
                ev_s3 = torch.cuda.Event(enable_timing=True)
                ev_s4 = torch.cuda.Event(enable_timing=True)
                ev_f0 = torch.cuda.Event(enable_timing=True)
                ev_f1 = torch.cuda.Event(enable_timing=True)
                ev_f2 = torch.cuda.Event(enable_timing=True)

                _sync_cuda(device)
                ev_s0.record()
                local_embed = model.local_encoder(data=data)
                ev_s1.record()
                local_s = model.proj_local(local_embed)
                ev_s2.record()
                global_s = model._edl_global_zeros_like_proj_global(local_s)
                y_s, _pi_s = model.small_decoder(local_embed=local_s, global_embed=global_s)
                ev_s3.record()

                ai = CascadedHiVT.absolute_agent_indices(data, device)
                n_agents_local = int(ai.numel())
                if n_agents_local == 0:
                    ev_s4.record()
                    _sync_cuda(device)
                    ms_local_encoder = _cuda_elapsed_ms(ev_s0, ev_s1)
                    ms_proj_local = _cuda_elapsed_ms(ev_s1, ev_s2)
                    ms_small_decoder = _cuda_elapsed_ms(ev_s2, ev_s3)
                    ms_small_post = _cuda_elapsed_ms(ev_s3, ev_s4)
                    small_ms = _cuda_elapsed_ms(ev_s0, ev_s4)
                    return

                y_agent = data.y[ai]
                y_s_a = y_s[:, ai, :, :]
                best_s, pred_s = _gather_best_mode(y_s_a, y_agent, ch_loc=2)
                b_idx = torch.arange(n_agents_local, device=device, dtype=torch.long)
                y_last = y_s_a[best_s, b_idx, -1, :]
                u_xy = nig_epistemic_uncertainty(y_last)
                epi = combine_xy_uncertainty(u_xy, reduction="l2")
                fde_s_b = torch.norm(pred_s[:, -1, :] - y_agent[:, -1], p=2, dim=-1)

                ev_s4.record()
                _sync_cuda(device)
                ms_local_encoder = _cuda_elapsed_ms(ev_s0, ev_s1)
                ms_proj_local = _cuda_elapsed_ms(ev_s1, ev_s2)
                ms_small_decoder = _cuda_elapsed_ms(ev_s2, ev_s3)
                ms_small_post = _cuda_elapsed_ms(ev_s3, ev_s4)
                small_ms = _cuda_elapsed_ms(ev_s0, ev_s4)

                needs_full = (
                    ((gate_tau_val <= 0.0) or bool((epi > gate_tau_val).any().item()))
                    and viz_this_seq
                )
                y_f = None
                fde_f_b: torch.Tensor | None = None

                if needs_full:
                    _sync_cuda(device)
                    ev_f0.record()
                    global_embed = model.global_interactor(data=data, local_embed=local_embed)
                    ev_f1.record()
                    y_f, _pi_f = model.full_decoder(
                        local_embed=local_embed, global_embed=global_embed
                    )
                    ev_f2.record()
                    _sync_cuda(device)
                    ms_global_interactor = _cuda_elapsed_ms(ev_f0, ev_f1)
                    ms_full_decoder = _cuda_elapsed_ms(ev_f1, ev_f2)
                    full_ms = _cuda_elapsed_ms(ev_f0, ev_f2)

                    y_f_a = y_f[:, ai, :, :]
                    _, pred_f = _gather_best_mode(y_f_a, y_agent, ch_loc=2)
                    fde_f_b = torch.norm(pred_f[:, -1, :] - y_agent[:, -1], p=2, dim=-1)
            else:
                t0 = time.perf_counter()
                local_embed = model.local_encoder(data=data)
                t1 = time.perf_counter()
                local_s = model.proj_local(local_embed)
                t2 = time.perf_counter()
                global_s = model._edl_global_zeros_like_proj_global(local_s)
                y_s, _pi_s = model.small_decoder(local_embed=local_s, global_embed=global_s)
                t3 = time.perf_counter()

                ai = CascadedHiVT.absolute_agent_indices(data, device)
                n_agents_local = int(ai.numel())
                if n_agents_local == 0:
                    t4 = time.perf_counter()
                    _sync_cuda(device)
                    ms_local_encoder = (t1 - t0) * 1000.0
                    ms_proj_local = (t2 - t1) * 1000.0
                    ms_small_decoder = (t3 - t2) * 1000.0
                    ms_small_post = (t4 - t3) * 1000.0
                    small_ms = (t4 - t0) * 1000.0
                    return

                y_agent = data.y[ai]
                y_s_a = y_s[:, ai, :, :]
                best_s, pred_s = _gather_best_mode(y_s_a, y_agent, ch_loc=2)
                b_idx = torch.arange(n_agents_local, device=device, dtype=torch.long)
                y_last = y_s_a[best_s, b_idx, -1, :]
                u_xy = nig_epistemic_uncertainty(y_last)
                epi = combine_xy_uncertainty(u_xy, reduction="l2")
                fde_s_b = torch.norm(pred_s[:, -1, :] - y_agent[:, -1], p=2, dim=-1)

                t4 = time.perf_counter()
                _sync_cuda(device)
                ms_local_encoder = (t1 - t0) * 1000.0
                ms_proj_local = (t2 - t1) * 1000.0
                ms_small_decoder = (t3 - t2) * 1000.0
                ms_small_post = (t4 - t3) * 1000.0
                small_ms = (t4 - t0) * 1000.0

                needs_full = (gate_tau_val <= 0.0) or bool((epi > gate_tau_val).any().item())
                y_f = None
                fde_f_b = None

                if needs_full:
                    tg0 = time.perf_counter()
                    global_embed = model.global_interactor(data=data, local_embed=local_embed)
                    tg1 = time.perf_counter()
                    y_f, _pi_f = model.full_decoder(
                        local_embed=local_embed, global_embed=global_embed
                    )
                    tg2 = time.perf_counter()
                    _sync_cuda(device)
                    ms_global_interactor = (tg1 - tg0) * 1000.0
                    ms_full_decoder = (tg2 - tg1) * 1000.0
                    full_ms = (tg2 - tg0) * 1000.0

                    y_f_a = y_f[:, ai, :, :]
                    _, pred_f = _gather_best_mode(y_f_a, y_agent, ch_loc=2)
                    fde_f_b = torch.norm(pred_f[:, -1, :] - y_agent[:, -1], p=2, dim=-1)

            latency_ms = float(small_ms + full_ms)
            t_ps = small_ms / 1000.0
            t_fd = full_ms / 1000.0

            if args.print_decoder_timing:
                print(
                    f"batch {batch_idx} "
                    f"le={ms_local_encoder:.3f} pj={ms_proj_local:.3f} sd={ms_small_decoder:.3f} "
                    f"post={ms_small_post:.3f} | small={small_ms:.3f} "
                    f"gi={ms_global_interactor:.3f} fd={ms_full_decoder:.3f} | full={full_ms:.3f} "
                    f"lat={latency_ms:.3f} needs_full={needs_full}"
                )

            if gate_tau_val <= 0.0:
                gate_mask = torch.ones_like(epi, dtype=torch.bool)
            else:
                gate_mask = epi > gate_tau_val
            if needs_full and fde_f_b is not None:
                final_fde_per_agent = torch.where(gate_mask, fde_f_b, fde_s_b)
            else:
                final_fde_per_agent = fde_s_b

            match = (ai == focal).nonzero(as_tuple=True)[0]
            if match.numel() > 0:
                j = int(match[0].item())
                epi_f = float(epi[j].item())
                fde_s_f = float(fde_s_b[j].item())
                fde_f_f = float(fde_f_b[j].item()) if fde_f_b is not None else float("nan")
                final_fde_focal = float(final_fde_per_agent[j].item())
            else:
                epi_f = float("nan")
                fde_s_f = float("nan")
                fde_f_f = float("nan")
                final_fde_focal = float("nan")

            if record_effective:
                fde_f_row = _fde_f_list_for_json(fde_f_b, n_agents_local)
                row = {
                    "seq_id": seq_id,
                    "batch_idx": batch_idx,
                    "focal_agent_index": focal,
                    "scenario_tag": scenario_tag,
                    "n_agents": n_agents_local,
                    "epi_per_agent": [float(x) for x in epi.detach().cpu().tolist()],
                    "fde_s_per_agent": [float(x) for x in fde_s_b.detach().cpu().tolist()],
                    "fde_f_per_agent": fde_f_row,
                    "t_ps": float(t_ps),
                    "t_fd": float(t_fd),
                    "epi_focal": epi_f,
                    "epi_max": float(epi.max().item()),
                    "fde_small_focal": fde_s_f,
                    "fde_full_focal": _json_safe_float(fde_f_f),
                    "gate_tau": gate_tau_val,
                    "full_decoder_executed": needs_full,
                    "latency_ms": latency_ms,
                    "small_region_ms": float(small_ms),
                    "full_region_ms": float(full_ms),
                    "ms_local_encoder": float(ms_local_encoder),
                    "ms_proj_local": float(ms_proj_local),
                    "ms_small_decoder": float(ms_small_decoder),
                    "ms_small_post": float(ms_small_post),
                    "ms_global_interactor": float(ms_global_interactor),
                    "ms_full_decoder": float(ms_full_decoder),
                    "is_full_path_activated": needs_full,
                    "final_fde_focal": _json_safe_float(final_fde_focal),
                }
                per_batch_records.append(row)
                conditional_eval_samples.append(
                    {
                        "seq_id": seq_id,
                        "batch_idx": batch_idx,
                        "latency_ms": latency_ms,
                        "is_full_path_activated": bool(needs_full),
                        "final_fde": _json_safe_float(final_fde_focal),
                        "gate_tau": gate_tau_val,
                    }
                )

        for gate_tau in gate_tau_sweep_list:
            per_batch_records = []
            conditional_eval_samples = []
            found_viz_seqs = set()
            gt_fn = int(gate_tau) if gate_tau == float(int(gate_tau)) else gate_tau
            if len(gate_tau_sweep_list) == 1:
                all_results_path = out_dir / f"all_results{shard_suffix}.jsonl"
                ce_name = f"conditional_eval_samples{shard_suffix}.jsonl"
            else:
                all_results_path = out_dir / f"all_results_gate_tau_{gt_fn}{shard_suffix}.jsonl"
                ce_name = f"conditional_eval_samples_gate_tau_{gt_fn}{shard_suffix}.jsonl"

            print(f"[gate_tau={gate_tau:g}] -> {all_results_path.name}")

            loop_t0 = time.perf_counter()
            batch_durations: list[float] = []
            for local_i, data in enumerate(loader):
                global_bi = start + local_i
                t_b = time.perf_counter()
                run_inference_batch(
                    data,
                    record=(local_i >= args.warmup_batches),
                    batch_idx=global_bi,
                    gate_tau_val=float(gate_tau),
                )

                if viz_seq_ids is not None and found_viz_seqs == viz_seq_ids:
                    print(
                        f"[viz_seq_ids] all {len(viz_seq_ids)} seq_id(s) found; stop early at local_i={local_i}"
                    )
                    break
                dt = time.perf_counter() - t_b
                if local_i >= args.warmup_batches:
                    batch_durations.append(dt)
                    if len(batch_durations) >= 8 and n_total_batches > local_i + 1:
                        avg = sum(batch_durations) / len(batch_durations)
                        rem = max(0, n_total_batches - local_i - 1) * avg
                        print(
                            f"[ETA τ={gate_tau:g}] ~{rem / 60.0:.1f} min left for "
                            f"{n_total_batches - local_i - 1} batches (~{avg * 1000.0:.0f} ms/batch, "
                            f"elapsed {time.perf_counter() - loop_t0:.0f}s)"
                        )
                        batch_durations.clear()

            wall_s_total += time.perf_counter() - loop_t0
            with all_results_path.open("w", encoding="utf-8") as jf:
                for row in per_batch_records:
                    jf.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"Saved {all_results_path.name} ({len(per_batch_records)} batches).")

            timing_means = _mean_timing_breakdown(per_batch_records)
            if timing_means:
                if len(gate_tau_sweep_list) == 1:
                    timing_summary_path = out_dir / f"timing_summary{shard_suffix}.json"
                else:
                    timing_summary_path = (
                        out_dir / f"timing_summary_gate_tau_{gt_fn}{shard_suffix}.json"
                    )
                timing_payload: dict[str, object] = {
                    "gate_tau": float(gate_tau),
                    "n_batches_recorded": len(per_batch_records),
                    "warmup_batches_excluded": int(args.warmup_batches),
                    "all_results_jsonl": all_results_path.name,
                    **timing_means,
                }
                with timing_summary_path.open("w", encoding="utf-8") as tf:
                    json.dump(timing_payload, tf, ensure_ascii=False, indent=2)
                print(
                    "[Timing 평균 ms] "
                    + " ".join(
                        f"{k.replace('mean_', '')}={timing_means[k]:.4f}"
                        for k in sorted(timing_means)
                    )
                )
                print(f"Saved {timing_summary_path.name}")

            ce_jsonl = out_dir / ce_name
            with ce_jsonl.open("w", encoding="utf-8") as jf:
                for row in conditional_eval_samples:
                    jf.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_ce = len(conditional_eval_samples)
            ce_summary: dict[str, object] = {
                "gate_tau": float(gate_tau),
                "n_samples": n_ce,
                "warmup_batches_excluded": int(args.warmup_batches),
                "samples_jsonl": ce_jsonl.name,
            }
            if n_ce > 0:
                avg_lat = sum(float(r["latency_ms"]) for r in conditional_eval_samples) / n_ce
                rate = (
                    sum(1 for r in conditional_eval_samples if r["is_full_path_activated"])
                    / n_ce
                )
                ce_summary["average_latency_ms"] = avg_lat
                ce_summary["full_path_activation_rate"] = rate
                print(
                    f"[Conditional branch τ={gate_tau:g}] 평균 지연: {avg_lat:.4f} ms | "
                    f"Full 경로 활성화: {rate * 100.0:.2f}% (n={n_ce})"
                )
            sweep_summaries.append(ce_summary)

            if len(gate_tau_sweep_list) == 1:
                (out_dir / f"conditional_eval_summary{shard_suffix}.json").write_text(
                    json.dumps(ce_summary, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                print(f"Saved {ce_jsonl.name} / conditional_eval_summary{shard_suffix}.json")

        wall_s = wall_s_total

        if len(gate_tau_sweep_list) > 1:
            sweep_path = out_dir / f"conditional_eval_sweep_summary{shard_suffix}.json"
            sweep_path.write_text(
                json.dumps(
                    {
                        "gate_tau_min": int(args.gate_tau_min),
                        "gate_tau_max": int(args.gate_tau_max),
                        "per_gate_tau": sweep_summaries,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            print(
                f"Saved {sweep_path.name} ({len(sweep_summaries)} gate τ 값, "
                f"평균 지연·Full 비율 요약)"
            )

        if len(gate_tau_sweep_list) > 1 and 0.0 in gate_tau_sweep_list:
            p0 = out_dir / f"all_results_gate_tau_0{shard_suffix}.jsonl"
            per_batch_records = _load_jsonl_records(p0)
            all_results_path = p0
        elif len(gate_tau_sweep_list) > 1:
            gt0 = int(gate_tau_sweep_list[0])
            pfirst = out_dir / f"all_results_gate_tau_{gt0}{shard_suffix}.jsonl"
            per_batch_records = _load_jsonl_records(pfirst)
            all_results_path = pfirst

    if len(per_batch_records) == 0:
        raise RuntimeError("기록된 배치가 없습니다. warmup_batches를 줄이거나 jsonl을 확인하세요.")

    csv_path = out_dir / f"cascade_gated_tau_sweep{output_suffix}.csv"
    summary_path = out_dir / f"cascade_gated_tau_sweep_summary{output_suffix}.json"
    meta_path = out_dir / f"cascade_gated_tau_sweep_meta{output_suffix}.json"
    act_json = out_dir / f"activation_by_scenario_tag{output_suffix}.json"
    err_delta_json = out_dir / f"error_delta_gate_on{output_suffix}.json"
    viz_dir = out_dir / f"viz_high_delta{output_suffix}"
    fig_path_png = out_dir / f"cascade_gated_tau_sweep{output_suffix}.png"
    fig_path_pdf = out_dir / f"cascade_gated_tau_sweep{output_suffix}.pdf"

    taus, tau_min, tau_max = _resolve_u_epi_sweep(args)
    n_t = len(taus)

    tau_sweep_ok = _records_allow_counterfactual_tau_sweep(per_batch_records)
    if not tau_sweep_ok:
        print(
            "[Info] 조건부 분기로 일부 배치에서 Full 디코더를 생략해, "
            "사후 다중 U_epi 스윕 CSV/그래프(per_batch_U_epi_detail 포함)는 생략합니다. "
            "런타임 통계는 conditional_eval_summary*.json 을 사용하세요."
        )

    if tau_sweep_ok:
        (
            mean_fde_list,
            rel_lat_list,
            rel_variable_lat_list,
            pct_list,
            mean_fde_small,
            mean_fde_full,
            total_agents,
            n_val_batches,
        ) = _compute_tau_metrics_from_records(per_batch_records, taus)
    else:
        total_agents = sum(int(r["n_agents"]) for r in per_batch_records)
        n_val_batches = len(per_batch_records)
        sum_s = 0.0
        for r in per_batch_records:
            sum_s += sum(float(x) for x in r["fde_s_per_agent"])
        mean_fde_small = sum_s / max(total_agents, 1)
        mean_fde_full = float("nan")
        mean_fde_list = []
        rel_lat_list = []
        rel_variable_lat_list = []
        pct_list = []

    n_a = max(total_agents, 1)
    nb = max(n_val_batches, 1)
    rel_hivt_decoder_ref = 100.0  # 분모 = t_fd (HiVT full decoder only)

    if tau_sweep_ok:
        print(
            f"[Post-process] U_epi 스윕 {n_t}개: minFDE·latency·activation — 메모리 집계 완료 "
            f"(agents={total_agents}, batches={n_val_batches})"
        )

    rows: list[dict[str, float | int]] = []
    loss_vs_small_pct: list[float] = []
    saving_vs_hivt_decoder_ref_pct: list[float] = []
    saving_vs_variable_fd_ref_pct: list[float] = []

    if tau_sweep_ok:
        for j, tau in enumerate(taus):
            mfde = mean_fde_list[j]
            rel_pct = rel_lat_list[j]
            rel_v = rel_variable_lat_list[j]
            pct_full = pct_list[j]
            lvs = (mfde - mean_fde_small) / mean_fde_small * 100.0 if mean_fde_small > 1e-12 else 0.0
            loss_vs_small_pct.append(lvs)
            sav = rel_hivt_decoder_ref - rel_pct
            saving_vs_hivt_decoder_ref_pct.append(sav)
            sav_v = rel_hivt_decoder_ref - rel_v
            saving_vs_variable_fd_ref_pct.append(sav_v)
            rows.append(
                {
                    "U_epi": tau,
                    "minFDE_m": round(mfde, 6),
                    "relative_variable_latency_pct_vs_fd": round(rel_v, 4),
                    "relative_decoder_latency_pct_vs_hivt": round(rel_pct, 4),
                    "full_path_activation_pct": round(pct_full, 4),
                    "fde_loss_vs_small_pct": round(lvs, 6),
                    "saving_vs_variable_fd_ref_pct": round(sav_v, 4),
                    "saving_vs_hivt_decoder_ref_pct": round(sav, 4),
                    "n_val_batches": n_val_batches,
                    "n_predicting_agents": total_agents,
                }
            )

    sweep_json_path = out_dir / f"cascade_gated_tau_sweep{output_suffix}.json"
    detail_path: Path | None = None
    if tau_sweep_ok and rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        sweep_json_path.write_text(
            json.dumps(
                {
                    "U_epi_range": [tau_min, tau_max],
                    "U_epi_values": taus,
                    "n_tau": len(taus),
                    "rows": rows,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        if not args.no_save_per_batch_U_epi_detail:
            detail_path = out_dir / f"per_batch_U_epi_detail{output_suffix}.jsonl"
            detail_rows = _compute_per_batch_U_epi_detail(per_batch_records, taus)
            with detail_path.open("w", encoding="utf-8") as jf:
                for row in detail_rows:
                    jf.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"Saved {detail_path.name} ({len(detail_rows)} batches × {len(taus)} U_epi)."
            )

    # --- 요약: Small 대비 FDE 손실 < fde_vs_small_pct_max 인 U_epi 중, Full 대비 Latency 감소 최대 ---
    thr = float(args.fde_vs_small_pct_max)
    candidates = (
        [i for i in range(n_t) if loss_vs_small_pct[i] < thr]
        if tau_sweep_ok and len(loss_vs_small_pct) == n_t
        else []
    )
    if candidates:
        # 동률: 절약량 최대 → Full 활성화 낮음 → U_epi 큼
        best_i = max(
            candidates,
            key=lambda i: (
                saving_vs_hivt_decoder_ref_pct[i],
                -pct_list[i],
                taus[i],
            ),
        )
        best_i_var = max(
            candidates,
            key=lambda i: (
                saving_vs_variable_fd_ref_pct[i],
                -pct_list[i],
                taus[i],
            ),
        )
        best_u_epi = taus[best_i]
        summary = {
            "rule": f"fde_loss_vs_small_pct < {thr} 인 U_epi 중 saving_vs_hivt_decoder_ref_pct 최대",
            "best_U_epi": best_u_epi,
            "minFDE_m": mean_fde_list[best_i],
            "relative_decoder_latency_pct_vs_hivt": rel_lat_list[best_i],
            "full_path_activation_pct": pct_list[best_i],
            "fde_loss_vs_small_pct": loss_vs_small_pct[best_i],
            "saving_vs_hivt_decoder_ref_pct": saving_vs_hivt_decoder_ref_pct[best_i],
            "summary_pick_variable_latency": {
                "rule": f"fde_loss_vs_small_pct < {thr} 인 U_epi 중 saving_vs_variable_fd_ref_pct 최대 (분모=t_fd, 분자=t_ps_pure+needs_full·t_fd)",
                "best_U_epi": taus[best_i_var],
                "relative_variable_latency_pct_vs_fd": rel_variable_lat_list[best_i_var],
                "saving_vs_variable_fd_ref_pct": saving_vs_variable_fd_ref_pct[best_i_var],
                "full_path_activation_pct": pct_list[best_i_var],
            },
        }
    else:
        summary = {
            "rule": f"fde_loss_vs_small_pct < {thr} 인 U_epi 없음",
            "best_U_epi": None,
        }

    if not tau_sweep_ok:
        summary = {
            "note": "조건부 분기로 fde_f 미산출 배치가 있어 다중 U_epi 스윕 요약을 생략함",
            "use_instead": "conditional_eval_summary*.json",
        }

    def _json_baseline_full(x: float) -> float | None:
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return x

    summary_path.write_text(
        json.dumps(
            {
                "baselines": {
                    "mean_minFDE_small_only": mean_fde_small,
                    "mean_minFDE_full_only": _json_baseline_full(mean_fde_full),
                    "hivt_full_decoder_latency_ref_pct": rel_hivt_decoder_ref,
                },
                "summary_pick": summary,
                "tau_sweep_postprocess": tau_sweep_ok,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --- 최적 U_epi: minFDE 최소 ---
    if tau_sweep_ok and mean_fde_list:
        opt_i = min(range(n_t), key=lambda i: mean_fde_list[i])
        opt_tau = float(taus[opt_i])
    else:
        opt_i = 0
        opt_tau = float(gate_tau_sweep_list[0])

    _default_analysis_tau = int(gate_tau_sweep_list[0])
    analysis_u_epi = (
        int(args.analysis_u_epi)
        if args.analysis_u_epi is not None
        else (int(taus[opt_i]) if tau_sweep_ok and taus else _default_analysis_tau)
    )

    activation_by_u_epi: dict[str, dict[str, dict[str, int]]] = {}
    for tau in taus:
        scenes_per_tag: dict[str, int] = {}
        full_on_per_tag: dict[str, int] = {}
        for r in per_batch_records:
            tag = str(r["scenario_tag"])
            scenes_per_tag[tag] = scenes_per_tag.get(tag, 0) + 1
            ef = float(r["epi_focal"])
            if not math.isnan(ef) and ef > float(tau):
                full_on_per_tag[tag] = full_on_per_tag.get(tag, 0) + 1
        activation_by_u_epi[str(tau)] = {
            "scenes_per_tag": scenes_per_tag,
            "full_path_on_focal_per_tag": full_on_per_tag,
        }
    act_json.write_text(
        json.dumps(
            {
                "note": "Focal agent 기준. epi_focal > U_epi 임계값이면 해당 샘플에서 Full 디코더 사용.",
                "per_U_epi": activation_by_u_epi,
                "analysis_U_epi": analysis_u_epi,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    gate_on_deltas: list[float] = []
    for r in per_batch_records:
        ef = float(r["epi_focal"])
        if math.isnan(ef) or ef <= float(analysis_u_epi):
            continue
        ff = r.get("fde_full_focal")
        if ff is None:
            continue
        gate_on_deltas.append(float(r["fde_small_focal"]) - float(ff))
    mean_delta = (
        sum(gate_on_deltas) / len(gate_on_deltas) if gate_on_deltas else float("nan")
    )
    err_delta_json.write_text(
        json.dumps(
            {
                "analysis_U_epi": analysis_u_epi,
                "definition": "focal에서 epi > U_epi 임계값(게이트 ON). delta_m = minFDE_small - minFDE_full (양수면 Full 우수).",
                "n_gate_on_focal": len(gate_on_deltas),
                "mean_delta_m": None
                if not gate_on_deltas or (isinstance(mean_delta, float) and math.isnan(mean_delta))
                else round(float(mean_delta), 6),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    viz_saved: list[str] = []
    if (
        not args.postprocess_only
        and not args.no_viz
        and model is not None
        and val_ds is not None
        and device is not None
        and am is not None
        and gate_on_deltas
        and not (isinstance(mean_delta, float) and math.isnan(mean_delta))
    ):
        viz_dir.mkdir(parents=True, exist_ok=True)

        # 요청: delta_m 상위 20개는 (small / gated-full) 쌍으로 저장
        viz_pair_dir = viz_dir / "pair_small_vs_gated_full_top20"
        viz_pair_dir.mkdir(parents=True, exist_ok=True)

        # 요청: small-only로도 minFDE가 낮은 상위 20개는 small path만 따로 저장
        viz_small_top_dir = viz_dir / "small_only_top20_low_minFDE"
        viz_small_top_dir.mkdir(parents=True, exist_ok=True)

        cand: list[tuple[float, float, float, float, int, int]] = []
        for r in per_batch_records:
            ef = float(r["epi_focal"])
            if math.isnan(ef) or ef <= float(analysis_u_epi):
                continue
            ff = r.get("fde_full_focal")
            if ff is None:
                continue
            fde_s_f = float(r["fde_small_focal"])
            fde_f_f = float(ff)
            ds = fde_s_f - fde_f_f
            # 요청: delta_m(ds) 기준 상위 rank를 보여주기 위해 mean_delta 필터 제거
            cand.append((ds, ef, fde_s_f, fde_f_f, int(r["batch_idx"]), int(r["seq_id"])))
        cand.sort(key=lambda x: -x[0])
        cand = cand[: max(0, int(args.max_viz_png))]

        # delta_m top20 pair 저장
        pair_k = 20
        for rank, (ds, ef, fde_s_f, fde_f_f, bidx, seq_id) in enumerate(cand):
            data_v = val_ds[bidx]
            out_png = viz_dir / f"rank{rank:02d}_seq{seq_id}_delta{ds:.3f}.png"

            if rank < pair_k:
                scene_dir = viz_pair_dir / f"rank{rank:02d}_seq{seq_id}_delta{ds:.3f}"
                scene_dir.mkdir(parents=True, exist_ok=True)
                out_small = scene_dir / "small_path.png"
                out_gated = scene_dir / "gated_full_path.png"

                _viz_cascade_scene_small_and_gated_pair(
                    model,
                    data_v,
                    device,
                    float(analysis_u_epi),
                    out_small,
                    out_gated,
                    am,
                    str(args.root),
                    u_epi_focal=ef,
                    minFDE_small_focal=fde_s_f,
                    minFDE_full_focal=fde_f_f,
                    out_gated_extra_path=out_png,
                )
                viz_saved.append(str(out_png.relative_to(out_dir)))
                viz_saved.append(str(out_small.relative_to(out_dir)))
                viz_saved.append(str(out_gated.relative_to(out_dir)))
            else:
                _viz_cascade_scene(
                    model,
                    data_v,
                    device,
                    float(analysis_u_epi),
                    out_png,
                    am,
                    str(args.root),
                )
                viz_saved.append(str(out_png.relative_to(out_dir)))

        # small-only top20 (low minFDE) 저장
        small_top_k = 20
        small_top_cand: list[tuple[float, float, int, int]] = []
        for r in per_batch_records:
            fde_s = r.get("fde_small_focal")
            if fde_s is None:
                continue
            fde_s_f = float(fde_s)
            if math.isnan(fde_s_f):
                continue
            ef = float(r["epi_focal"])
            bidx = int(r["batch_idx"])
            seq_id = int(r["seq_id"])
            small_top_cand.append((fde_s_f, ef, bidx, seq_id))
        small_top_cand.sort(key=lambda x: x[0])  # minFDE small 오름차순
        small_top_cand = small_top_cand[:small_top_k]
        for rank, (fde_s_f, ef, bidx, seq_id) in enumerate(small_top_cand):
            data_v = val_ds[bidx]
            scene_dir = viz_small_top_dir / f"rank{rank:02d}_seq{seq_id}_minFDE{fde_s_f:.3f}"
            scene_dir.mkdir(parents=True, exist_ok=True)
            out_small = scene_dir / "small_only.png"
            _viz_cascade_scene_small_only(
                model,
                data_v,
                device,
                out_small,
                am,
                str(args.root),
                u_epi_focal=ef,
                minFDE_small_focal=fde_s_f,
            )
            viz_saved.append(str(out_small.relative_to(out_dir)))

    if not args.postprocess_only and wall_s > 0:
        print(
            f"Wall time (inference loop): {wall_s:.1f}s "
            f"(~{wall_s / max(n_val_batches, 1):.2f}s / recorded batch)"
        )

    def _draw_figure() -> plt.Figure:
        fig, ax1 = plt.subplots(figsize=(10, 5.8))
        c_fde, c_lat, c_full = "#1f77b4", "#ff7f0e", "#2ca02c"
        # 요청: uepi=0도 포함해서 plot
        taus_plot = [float(t) for t in taus]
        mean_fde_plot = list(mean_fde_list)
        rel_variable_lat_plot = list(rel_variable_lat_list)
        pct_plot = list(pct_list)

        tau_lo, tau_hi = float(taus_plot[0]), float(taus_plot[-1])
        x_pad = max(0.55, (tau_hi - tau_lo) * 0.12 + 0.15)
        tx_lo, tx_hi = map(
            float, (args.tau_xlim or "0,150").replace(" ", "").split(",")
        )
        # 요청: optimal trade-off point을 U_epi=20으로 고정
        tradeoff_tau = 20.0
        idx20 = None
        for i, t in enumerate(taus):
            if float(t) == tradeoff_tau:
                idx20 = i
                break
        if idx20 is None:
            print(
                f"[warn] U_epi trade-off tau=20 not found in taus={taus}; vertical line/annotations skipped."
            )
            fde_at_20 = None
            rel_at_20 = None
            pct_at_20 = None
        else:
            fde_at_20 = mean_fde_list[idx20]
            rel_at_20 = rel_variable_lat_list[idx20]
            pct_at_20 = pct_list[idx20]

        ax1.set_xlabel(r"$U_{epi}$ (Epistemic Uncertainty Threshold)")
        # 요청: x축 라벨을 더 아래로
        ax1.xaxis.set_label_coords(0.5, -0.065)
        ax1.set_ylabel("minFDE (m)", color=c_fde)
        (ln1,) = ax1.plot(
            taus_plot,
            mean_fde_plot,
            color=c_fde,
            linewidth=2.2,
            label="minFDE(Proposed)",
        )
        ax1.axhline(mean_fde_full, color=c_fde, linestyle="--", linewidth=1.35, label="HiVT minFDE")
        ax1.tick_params(axis="y", labelcolor=c_fde)
        ax1.grid(True, alpha=0.35, linestyle="-", linewidth=0.6)
        ax1.set_axisbelow(True)

        # U_epi=20 세로선 + trade-off 표기
        if fde_at_20 is not None:
            ax1.axvline(
                tradeoff_tau,
                color="#555555",
                linestyle="--",
                linewidth=1.05,
                zorder=1,
                alpha=0.9,
            )

        y1a, y1b = ax1.get_ylim()
        y1span = max(y1b - y1a, 1e-6)
        dx_opt = max(0.12, (tau_hi - tau_lo) * 0.018)

        if fde_at_20 is not None:
            # 위쪽(그래프 위): minFDE(Proposed) 값만 표기
            ax1.text(
                tradeoff_tau + dx_opt,
                float(fde_at_20) + 0.015 * y1span,
                f"{float(fde_at_20):.3f}m",
                ha="left",
                va="bottom",
                fontsize=8,
                color=c_fde,
                zorder=5,
            )
            # 아래쪽(x축 아래): optimal trade-off point 라벨
            ax1.text(
                tradeoff_tau,
                -0.08,
                "Optimal trade-off point\n(U_epi=20)",
                transform=ax1.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
                color="#444444",
                zorder=10,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.75,
                ),
            )

        ax2 = ax1.twinx()
        # Matplotlib에서 r"\%" 이스케이프가 일부 환경에서 "/%"처럼 렌더링될 수 있어 plain "%"로 표기
        ax2.set_ylabel("Normalized Inference Cost(%)", color=c_lat)
        (ln2,) = ax2.plot(
            taus_plot,
            rel_variable_lat_plot,
            color=c_lat,
            linewidth=2.0,
            linestyle="-",
            label="Normalized Inference Cost(Proposed)",
        )
        ax2.axhline(
            100.0,
            color=c_lat,
            linestyle="--",
            linewidth=1.35,
            label="Normalized Inference Cost(HiVT)",
        )
        ax2.tick_params(axis="y", labelcolor=c_lat)
        y_lo, y_hi = map(float, (args.latency_ylim or "0,130").replace(" ", "").split(","))
        ax2.set_ylim(y_lo, y_hi)

        y2a, y2b = ax2.get_ylim()
        y2span = max(y2b - y2a, 1e-6)

        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.11))
        ax3.set_ylabel("Full Path Activation (%)", color=c_full)
        (ln3,) = ax3.plot(
            taus_plot,
            pct_plot,
            color=c_full,
            linewidth=2.0,
            linestyle="-",
            label="Full Path Activation (%)",
        )
        ax3.tick_params(axis="y", labelcolor=c_full)
        ax3.set_ylim(-2, 105)
        ax3.grid(False)

        ax1.set_xlim(tx_lo, tx_hi)
        # 요청: uepi=0은 제외된 x-tick만 보여주기
        ticks_show = [t for t in taus_plot if tx_lo <= float(t) <= tx_hi]
        if len(ticks_show) <= 24 and ticks_show:
            ax1.set_xticks(ticks_show)


        # 참조선 수치: 가시 U_epi 구간 [tx_lo, tx_hi] 안쪽 오른쪽
        x_ref = tx_hi - max(0.5, (tx_hi - tx_lo) * 0.035)
        dy_fde = 0.028 * y1span
        if mean_fde_full + dy_fde > y1b - 0.01 * y1span:
            dy_fde = -0.028 * y1span
        ax1.text(
            x_ref,
            mean_fde_full + dy_fde,
            f"{mean_fde_full:.3f} m",
            va="center",
            ha="left",
            fontsize=8,
            color=c_fde,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85),
        )
        dy_lat = 0.035 * y2span
        y_lab_100 = 100.0 + dy_lat
        if y_lab_100 > y2b - 0.02 * y2span:
            y_lab_100 = 100.0 - abs(dy_lat)
        ax2.text(
            x_ref,
            y_lab_100,
            "100%",
            va="center",
            ha="left",
            fontsize=8,
            color=c_lat,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85),
        )

        # 요청: U_epi=20에 해당하는 y축 값들을 모두 그래프 위에 표시
        if fde_at_20 is not None:
            ax2.text(
                tradeoff_tau + dx_opt,
                float(rel_at_20) + 0.015 * y2span,
                f"{float(rel_at_20):.2f}%",
                ha="left",
                va="center",
                fontsize=8,
                color=c_lat,
                zorder=6,
            )
            ax3.text(
                tradeoff_tau + dx_opt,
                float(pct_at_20) + 0.9,
                f"{float(pct_at_20):.2f}%",
                ha="left",
                va="center",
                fontsize=8,
                color=c_full,
                zorder=6,
            )

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        h3, l3 = ax3.get_legend_handles_labels()
        # 제목을 더 위로 올리기: 위쪽 공간 확보
        # 상단 여백 확보(제목과 legend 라벨 겹침 방지)
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        leg = ax1.legend(
            h1 + h2 + h3,
            l1 + l2 + l3,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.14),
            ncol=2,
            fontsize=7,
        )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bb = leg.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
        title_y = min(bb.y1 + 0.045, 0.998)
        fig.suptitle(
            "Performance and Efficiency Trade-off of Gated Cascade Inference",
            fontsize=12,
            # 요청: 라벨/제목을 위로 더 올리기
            y=min(bb.y1 + 0.40, 0.999),
        )
        return fig

    if tau_sweep_ok and mean_fde_list:
        fig = _draw_figure()
        fig.savefig(fig_path_png, dpi=180, bbox_inches="tight")
        with PdfPages(fig_path_pdf) as pdf:
            pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    meta = {
        "checkpoint": None if merged_from_shards else str(ckpt),
        "batch_size": int(args.batch_size),
        "warmup_batches": args.warmup_batches,
        "plot_tau_xlim": getattr(args, "tau_xlim", None) or "0,150",
        "plot_latency_ylim": getattr(args, "latency_ylim", None) or "0,130",
        "U_epi_sweep_range": [tau_min, tau_max],
        "U_epi_sweep_values": taus,
        "n_val_batches": n_val_batches,
        "n_predicting_agents": total_agents,
        "limit_batches": args.limit_batches,
        "analysis_U_epi": analysis_u_epi,
        "data_sharding": None
        if num_shards <= 1
        else {"num_shards": num_shards, "shard_idx": shard_idx, "output_suffix": output_suffix},
        "merged_from_shards": merged_from_shards,
        "aux_outputs": {
            "all_results_jsonl": str(all_results_path.name),
            "conditional_eval_summary_json": (
                f"conditional_eval_summary{output_suffix}.json"
                if not (args.postprocess_only or merged_from_shards)
                and len(gate_tau_sweep_list) == 1
                else None
            ),
            "conditional_eval_sweep_summary_json": (
                f"conditional_eval_sweep_summary{output_suffix}.json"
                if not (args.postprocess_only or merged_from_shards)
                and len(gate_tau_sweep_list) > 1
                else None
            ),
            "cascade_gated_tau_sweep_json": (
                str(sweep_json_path.name) if tau_sweep_ok and rows else None
            ),
            "per_batch_U_epi_detail_jsonl": (
                str(detail_path.name) if detail_path is not None else None
            ),
            "activation_by_scenario_tag_json": str(act_json.name),
            "error_delta_gate_on_json": str(err_delta_json.name),
            "viz_high_delta_pngs": viz_saved,
        },
        "runtime_conditional_branch": {
            "gate_tau_single": float(args.gate_tau) if args.gate_tau is not None else None,
            "gate_tau_sweep_list": gate_tau_sweep_list
            if len(gate_tau_sweep_list) > 1
            else None,
            "tau_sweep_postprocess_ok": tau_sweep_ok,
            "note": (
                "인퍼런스는 LocalEncoder→Small→U_epi 게이트 후 필요 시에만 "
                "GlobalInteractor+FullDecoder. Latency는 CUDA Event(ms) 구간 합."
            ),
        },
        "methodology_notes": {
            "minFDE": "에이전트 풀 전체 평균 — 총 FDE / 총 예측 에이전트 수 (Argoverse 등 모빌리티 논문에서 흔한 표준).",
            "relative_latency_legacy": "레거시: 분모=t_fd, 분자=t_ps+needs_full·t_fd (Local Encoder 포함).",
            "relative_variable_latency": (
                "권장: Common vs. Variable — 분모=t_fd(가변 Full branch 기준 100%). "
                "분자=t_ps_pure+needs_full·t_fd, t_ps_pure=proj+small_decoder+small_post(초), Local Encoder 제외."
            ),
        },
        "latency_note": (
            "가변 지연(플롯·relative_variable_latency_pct_vs_fd): 분모=t_fd; 분자=t_ps_pure+(needs_full×t_fd). "
            "레거시 컬럼 relative_decoder_latency_pct_vs_hivt는 분자에 전체 t_ps(로컬 인코더 포함)."
        ),
        "note_architecture": "플롯의 주 지연 곡선은 가변 비용만; 고정 Common cost(Local Encoder)는 분자에서 제외.",
        "scenario_tag_note": "Argoverse v1에 공식 시나리오 태그가 없어 AGENT GT 궤적로 straight/left/right 대략 분류.",
        "baselines": {
            "mean_minFDE_small_only": mean_fde_small,
            "mean_minFDE_full_only": _json_baseline_full(mean_fde_full),
            "hivt_full_decoder_latency_ref_pct": rel_hivt_decoder_ref,
        },
        "optimal_U_epi_at_minFDE": opt_tau if tau_sweep_ok else None,
        "optimal_minFDE_m": (mean_fde_list[opt_i] if tau_sweep_ok and mean_fde_list else None),
        "optimal_relative_decoder_latency_pct_vs_hivt": (
            rel_lat_list[opt_i] if tau_sweep_ok and rel_lat_list else None
        ),
        "optimal_relative_variable_latency_pct_vs_fd": (
            rel_variable_lat_list[opt_i] if tau_sweep_ok and rel_variable_lat_list else None
        ),
        "wall_time_s": None
        if (args.postprocess_only or merged_from_shards)
        else round(wall_s, 3),
        "postprocess_only": bool(args.postprocess_only),
        "merge_shards": bool(args.merge_shards),
        "num_workers_dataloader": int(args.num_workers),
    }
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if tau_sweep_ok and rows:
        print(f"Saved CSV:  {csv_path}")
        print(f"Saved τ JSON: {sweep_json_path}")
    if detail_path is not None:
        print(f"Saved batch×U_epi: {detail_path}")
    if tau_sweep_ok and mean_fde_list:
        print(f"Saved PNG:  {fig_path_png}")
        print(f"Saved PDF:  {fig_path_pdf}")
    print(f"Summary:    {summary_path}")
    print(f"Meta:       {meta_path}")
    print(f"All results: {all_results_path}")
    print(f"Tag act.:   {act_json}")
    print(f"Err delta:  {err_delta_json}")
    if viz_saved:
        print(f"Viz PNGs:   {viz_dir} ({len(viz_saved)} files)")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    mff_disp = (
        f"{mean_fde_full:.4f}"
        if isinstance(mean_fde_full, float) and not math.isnan(mean_fde_full)
        else "nan"
    )
    print(
        f"Baselines — minFDE small/full: {mean_fde_small:.4f} / {mff_disp} | "
        f"HiVT decoder ref (100%): {rel_hivt_decoder_ref:.1f}%"
    )


if __name__ == "__main__":
    main()
