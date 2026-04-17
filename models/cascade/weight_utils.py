# Load HiVT (encoder + full MLPDecoder) and DualHead EDL small decoder with 256→128 surgery.
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from models.DualHead.edl_mlp_decoder import EDLMLPDecoder, SlimEDLDecoder
from models.HiVT.decoder import MLPDecoder


def _strip_prefix(sd: dict[str, Any], prefix: str) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    plen = len(prefix)
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[plen:]] = v
    return out


# HiVT 체크포인트만 로드할 때 missing으로 남는 것이 정상인 키 접두사
HIVT_LOAD_EXPECTED_MISSING_PREFIXES: tuple[str, ...] = (
    "proj_local.",
    "proj_global.",
    "small_decoder.",
)


def filter_missing_keys_for_hivt_load(missing_keys: list[str]) -> tuple[list[str], list[str]]:
    """
    HiVT 단독 로드 후 ``missing_keys`` 중 의도된 것(``proj_*``, ``small_decoder.*``)과
    그 외(누락이면 안 되는 핵심 레이어 가능성)를 분리합니다.
    """
    expected: list[str] = []
    unexpected: list[str] = []
    for k in missing_keys:
        if any(k.startswith(p) for p in HIVT_LOAD_EXPECTED_MISSING_PREFIXES):
            expected.append(k)
        else:
            unexpected.append(k)
    return expected, unexpected


def load_hivt_encoder_and_full_decoder(
    model: nn.Module,
    ckpt_path: str | Path,
) -> tuple[list[str], list[str]]:
    """
    HiVT 체크포인트에서 ``local_encoder``, ``global_interactor``, ``decoder`` → ``full_decoder`` 로드.
    ``small_decoder`` / ``proj_*`` 는 체크포인트에 없어 missing_keys에 남는 것이 정상입니다.
    """
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    to_load: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if k.startswith("local_encoder."):
            to_load[k] = v
        elif k.startswith("global_interactor."):
            to_load[k] = v
        elif k.startswith("decoder."):
            to_load["full_decoder." + k[len("decoder.") :]] = v

    r = model.load_state_dict(to_load, strict=False)
    if r is None:
        return [], []
    return list(r.missing_keys), list(r.unexpected_keys)


def _copy_ln_256_to_128(dst: nn.LayerNorm, src_w: torch.Tensor, src_b: torch.Tensor | None) -> None:
    dst.weight.data.copy_(src_w[:128])
    if dst.bias is not None and src_b is not None:
        dst.bias.data.copy_(src_b[:128])


def load_dualhead_small_decoder_weights(
    small_decoder: nn.Module,
    ckpt_path: str | Path,
) -> dict[str, Any]:
    """
    DualHead ``decoder`` (EDL 256/256) → ``small_decoder`` (EDL 128/128).

    - ``aggr_embed.0`` (첫 Linear)는 shape 불일치로 **스킵** (랜덤 초기화 유지).
    - 나머지는 가능한 범위에서 **슬라이스** 복사.
    """
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    dh = _strip_prefix(state, "decoder.")
    report: dict[str, Any] = {"skipped": [], "copied": [], "sliced": []}

    def get(k: str) -> torch.Tensor | None:
        return dh.get(k)

    if isinstance(small_decoder, SlimEDLDecoder):
        # 다층→단층 이식은 비선형/LN이 빠져 초기 FDE가 튈 수 있음 → Small 전용 3~5 epoch 파인튜닝 권장.
        ew3, eb3 = get("evidential.3.weight"), get("evidential.3.bias")
        if ew3 is not None and isinstance(small_decoder.evidential, nn.Linear):
            m = small_decoder.evidential
            in_f = m.in_features
            if ew3.shape[1] >= in_f and ew3.shape[0] == m.out_features:
                m.weight.data.copy_(ew3[:, :in_f])
                if eb3 is not None and eb3.shape[0] == m.out_features:
                    m.bias.data.copy_(eb3)
                report["sliced"].append("evidential.3 -> SlimEDLDecoder.evidential")
        pw6, pb6 = get("pi.6.weight"), get("pi.6.bias")
        if pw6 is not None and isinstance(small_decoder.pi, nn.Linear):
            m = small_decoder.pi
            in_f = m.in_features
            if pw6.shape[1] >= in_f:
                w = pw6[:, :in_f]
                b = pb6
                if pw6.shape[0] == m.out_features:
                    m.weight.data.copy_(w)
                    if b is not None and b.shape[0] == m.out_features:
                        m.bias.data.copy_(b)
                    report["sliced"].append("pi.6 -> SlimEDLDecoder.pi")
                elif pw6.shape[0] == 1 and m.out_features > 1:
                    # 구 DualHead: 마지막 Linear 가 1 출력 → Slim 은 K개 로짓으로 반복 이식
                    m.weight.data.copy_(w.expand(m.out_features, -1).clone())
                    if b is not None:
                        m.bias.data.copy_(b.expand(m.out_features).clone())
                    report["sliced"].append(
                        f"pi.6 [1,{in_f}] -> SlimEDLDecoder.pi [{m.out_features},{in_f}] (행 반복)"
                    )
        return report

    # aggr_embed.0 — skip
    report["skipped"].append("aggr_embed.0 (512→256 vs 256→128; user: exclude first fusion layer)")

    # aggr_embed.1 LayerNorm
    w, b = get("aggr_embed.1.weight"), get("aggr_embed.1.bias")
    if w is not None and hasattr(small_decoder, "aggr_embed"):
        ln = small_decoder.aggr_embed[1]
        if isinstance(ln, nn.LayerNorm) and w.numel() >= 128:
            _copy_ln_256_to_128(ln, w, b)
            report["sliced"].append("aggr_embed.1 LayerNorm 256→128")

    # HiVT와 동일한 loc / scale (구버전 단일 evidential.* 는 행 순서 t*8+c 에 맞춰 분할)
    def _slice_edl_trunk_to_branch(
        branch_prefix: str,
        ew0: torch.Tensor | None,
        eb0: torch.Tensor | None,
        ew1: torch.Tensor | None,
        eb1: torch.Tensor | None,
        ew3: torch.Tensor | None,
        eb3: torch.Tensor | None,
    ) -> None:
        seq = getattr(small_decoder, branch_prefix, None)
        if seq is None or not isinstance(seq, nn.Sequential):
            return
        m0 = seq[0]
        if ew0 is not None and isinstance(m0, nn.Linear) and ew0.shape[0] >= 128 and ew0.shape[1] >= 128:
            m0.weight.data.copy_(ew0[:128, :128])
            if eb0 is not None:
                m0.bias.data.copy_(eb0[:128])
            report["sliced"].append(f"{branch_prefix}.0 Linear [:128,:128]")
        if ew1 is not None:
            ln = seq[1]
            if isinstance(ln, nn.LayerNorm):
                _copy_ln_256_to_128(ln, ew1, eb1)
                report["sliced"].append(f"{branch_prefix}.1 LayerNorm")
        m3 = seq[3]
        if ew3 is not None and isinstance(m3, nn.Linear):
            w = ew3[:, :128] if ew3.shape[1] >= 128 else ew3
            b = eb3
            if w.shape == m3.weight.shape:
                m3.weight.data.copy_(w)
            if b is not None and b.shape == m3.bias.shape:
                m3.bias.data.copy_(b)
            report["sliced"].append(f"{branch_prefix}.3 Linear (in-dim slice to 128)")

    ew0, eb0 = get("evidential.0.weight"), get("evidential.0.bias")
    ew1, eb1 = get("evidential.1.weight"), get("evidential.1.bias")
    ew3, eb3 = get("evidential.3.weight"), get("evidential.3.bias")
    if isinstance(small_decoder, EDLMLPDecoder) and ew0 is not None and ew3 is not None:
        H = small_decoder.future_steps
        if ew3.shape[0] >= 8 * H:
            loc_idx = [t * 8 + i for t in range(H) for i in range(4)]
            sc_idx = [t * 8 + 4 + i for t in range(H) for i in range(4)]
            ew3_loc = ew3[loc_idx]
            eb3_loc = eb3[loc_idx] if eb3 is not None else None
            ew3_sc = ew3[sc_idx]
            eb3_sc = eb3[sc_idx] if eb3 is not None else None
            _slice_edl_trunk_to_branch("loc", ew0, eb0, ew1, eb1, ew3_loc, eb3_loc)
            _slice_edl_trunk_to_branch("scale", ew0, eb0, ew1, eb1, ew3_sc, eb3_sc)
            report["sliced"].append("evidential → loc/scale (γν / αβ rows per timestep)")
        else:
            report["skipped"].append(
                f"evidential.3 out_dim {ew3.shape[0]} != 8*H ({8 * H}); loc/scale.3 skip, trunk only"
            )
            _slice_edl_trunk_to_branch("loc", ew0, eb0, ew1, eb1, None, None)
            _slice_edl_trunk_to_branch("scale", ew0, eb0, ew1, eb1, None, None)
    elif isinstance(small_decoder, EDLMLPDecoder):
        for pref in ("loc", "scale"):
            w0, b0 = get(f"{pref}.0.weight"), get(f"{pref}.0.bias")
            w1, b1 = get(f"{pref}.1.weight"), get(f"{pref}.1.bias")
            w3, b3 = get(f"{pref}.3.weight"), get(f"{pref}.3.bias")
            _slice_edl_trunk_to_branch(pref, w0, b0, w1, b1, w3, b3)

    # pi head
    pw0, pb0 = get("pi.0.weight"), get("pi.0.bias")
    if pw0 is not None:
        m = small_decoder.pi[0]
        if isinstance(m, nn.Linear):
            # old [256, 512] → new [128, 256]: 상반 입력(글로벌+로컬 각 128)에 대응해 앞 256열 사용
            if pw0.shape[0] >= 128 and pw0.shape[1] >= 256 and m.weight.shape == (128, 256):
                m.weight.data.copy_(pw0[:128, :256])
                if pb0 is not None:
                    m.bias.data.copy_(pb0[:128])
                report["sliced"].append("pi.0 Linear [:128,:256] (heuristic)")

    pw1, pb1 = get("pi.1.weight"), get("pi.1.bias")
    if pw1 is not None:
        ln = small_decoder.pi[1]
        if isinstance(ln, nn.LayerNorm):
            _copy_ln_256_to_128(ln, pw1, pb1)
            report["sliced"].append("pi.1 LayerNorm")

    pw3, pb3 = get("pi.3.weight"), get("pi.3.bias")
    if pw3 is not None:
        m = small_decoder.pi[3]
        if isinstance(m, nn.Linear) and pw3.shape[0] >= 128 and pw3.shape[1] >= 128:
            m.weight.data.copy_(pw3[:128, :128])
            if pb3 is not None:
                m.bias.data.copy_(pb3[:128])
            report["sliced"].append("pi.3 Linear [:128,:128]")

    pw4, pb4 = get("pi.4.weight"), get("pi.4.bias")
    if pw4 is not None:
        ln = small_decoder.pi[4]
        if isinstance(ln, nn.LayerNorm):
            _copy_ln_256_to_128(ln, pw4, pb4)
            report["sliced"].append("pi.4 LayerNorm")

    pw6, pb6 = get("pi.6.weight"), get("pi.6.bias")
    if pw6 is not None:
        m = small_decoder.pi[6]
        if isinstance(m, nn.Linear) and pw6.shape[1] >= 128:
            m.weight.data.copy_(pw6[:, :128])
            if pb6 is not None:
                m.bias.data.copy_(pb6)
            report["sliced"].append("pi.6 Linear (in 128)")

    return report


def clone_full_decoder_to_small_mlp(model: nn.Module) -> None:
    """
    ``full_decoder``(HiVT MLPDecoder)와 동일 구조의 ``small_decoder``에 가중치를 복사.
    Student는 forward 시 ``global_embed=0`` 만 사용 (global interaction 생략).
    """
    fd = getattr(model, "full_decoder", None)
    sd = getattr(model, "small_decoder", None)
    if not isinstance(fd, MLPDecoder) or not isinstance(sd, MLPDecoder):
        raise TypeError(
            "clone_full_decoder_to_small_mlp: full_decoder/small_decoder must both be MLPDecoder"
        )
    sd.load_state_dict(fd.state_dict())
