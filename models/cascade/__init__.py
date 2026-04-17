# Cascaded HiVT (shared encoder, EDL small + Laplace full) — adaptation training utilities.
from .cascade_checkpoint import (
    cascade_model_from_checkpoint,
    load_cascade_from_checkpoint,
    peek_cascade_hyper_parameters,
)
from .cascaded_hivt import CascadedHiVT
from .cascaded_hivt_edl_mlp import CascadedHiVTEDLMLP
from .cascaded_hivt_kd import CascadedHiVTMLPKD
from .weight_utils import (
    clone_full_decoder_to_small_mlp,
    filter_missing_keys_for_hivt_load,
    load_dualhead_small_decoder_weights,
    load_hivt_encoder_and_full_decoder,
)

__all__ = [
    "CascadedHiVT",
    "CascadedHiVTEDLMLP",
    "CascadedHiVTMLPKD",
    "cascade_model_from_checkpoint",
    "load_cascade_from_checkpoint",
    "peek_cascade_hyper_parameters",
    "clone_full_decoder_to_small_mlp",
    "filter_missing_keys_for_hivt_load",
    "load_hivt_encoder_and_full_decoder",
    "load_dualhead_small_decoder_weights",
]
