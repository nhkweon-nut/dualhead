# CascadedHiVT / CascadedHiVTMLPKD 체크포인트 로드 — hparams 의 cascade_model 로 클래스 자동 선택.
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def _hyper_parameters_dict(ckpt: dict[str, Any]) -> dict[str, Any]:
    hp = ckpt.get("hyper_parameters")
    if hp is None:
        return {}
    if isinstance(hp, dict):
        return hp
    if hasattr(hp, "items"):
        try:
            return dict(hp)
        except Exception:
            pass
    if hasattr(hp, "__dict__"):
        return vars(hp)
    return {}


def peek_cascade_hyper_parameters(ckpt_path: str | Path) -> dict[str, Any]:
    """체크포인트만 읽어 하이퍼파라미터 dict 반환 (모델 인스턴스 없음)."""
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return _hyper_parameters_dict(ckpt)


def cascade_model_from_checkpoint(ckpt_path: str | Path) -> str:
    """
    저장 시 기록된 ``cascade_model`` (없으면 ``cascaded_hivt``).
    ``cascaded_hivt_edl_mlp`` → ``CascadedHiVTEDLMLP``.
    """
    h = peek_cascade_hyper_parameters(ckpt_path)
    return str(h.get("cascade_model", "cascaded_hivt"))


def load_cascade_from_checkpoint(
    checkpoint_path: str | Path,
    map_location: Any = None,
    **kwargs: Any,
):
    """
    ``hyper_parameters.cascade_model`` 에 따라 ``CascadedHiVT`` (Slim 또는 EDLMLP small) 또는
    ``CascadedHiVTMLPKD`` 로
    ``load_from_checkpoint`` 호출.
    """
    from .cascaded_hivt import CascadedHiVT
    from .cascaded_hivt_edl_mlp import CascadedHiVTEDLMLP
    from .cascaded_hivt_kd import CascadedHiVTMLPKD

    path = str(Path(checkpoint_path).expanduser().resolve())
    mt = cascade_model_from_checkpoint(path)
    if map_location is not None:
        kwargs["map_location"] = map_location
    kwargs.setdefault("map_location", torch.device("cpu"))
    if mt == "cascaded_hivt_mlp_kd":
        return CascadedHiVTMLPKD.load_from_checkpoint(path, **kwargs)
    if mt == "cascaded_hivt_edl_mlp":
        return CascadedHiVTEDLMLP.load_from_checkpoint(path, **kwargs)
    return CascadedHiVT.load_from_checkpoint(path, **kwargs)
