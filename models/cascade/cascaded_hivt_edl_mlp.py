# Small 경로 전용: EDLMLPDecoder (HiVT MLPDecoder 와 동일 aggr + loc/scale + pi, NIG 8채널).
# Full path·동결·손실·최적화는 ``CascadedHiVT`` 와 동일.
from __future__ import annotations

import torch.nn as nn

from models.DualHead.edl_mlp_decoder import EDLMLPDecoder

from .cascaded_hivt import CascadedHiVT


class CascadedHiVTEDLMLP(CascadedHiVT):
    """
    - **Full (동결 가능)**: ``LocalEncoder → GlobalInteractor → MLPDecoder``
    - **Small (학습)**: ``local_embed, global_embed`` → ``proj_local`` / ``proj_global`` → ``EDLMLPDecoder``

    ``small_path_local_only=True`` 이면 global 은 0으로 두고 ``proj_local`` 만 사용 (부모와 동일).
    """

    def __init__(self, **kwargs) -> None:
        kw = dict(kwargs)
        kw.setdefault("cascade_model", "cascaded_hivt_edl_mlp")
        super().__init__(**kw)

    def _build_small_decoder(self) -> nn.Module:
        return EDLMLPDecoder(
            local_channels=self.small_embed_dim,
            global_channels=self.small_embed_dim,
            future_steps=self.future_steps,
            num_modes=self.num_modes,
        )
