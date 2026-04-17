# 적응 학습: CascadedHiVT(Slim EDL) 또는 CascadedHiVTMLPKD(동일 MLP + KD).
#
# 어디서 실행하든 프로젝트 루트로 맞춘 뒤, 기본 YAML을 자동 로드:
#   .venv/bin/python train_cascade_adapt.py
# 기본 설정: configs/cascade_edl_nll_kl.yaml (NIG NLL + KL, Slim EDL)
# YAML 없이 CLI만:
#   .venv/bin/python train_cascade_adapt.py --no-config
from __future__ import annotations

import json
import os
import sys
import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import TensorBoardLogger

_ROOT = Path(__file__).resolve().parent
_DEFAULT_CONFIG_CANDIDATES = [
    _ROOT / "configs" / "cascade_edl_nll_kl.yaml",
]
_ARGO = _ROOT / "datasets" / "argoverse-api-master"


def _default_config_path() -> Path | None:
    for p in _DEFAULT_CONFIG_CANDIDATES:
        if p.is_file():
            return p
    return None
sys.path.insert(0, str(_ARGO))
sys.path.insert(0, str(_ROOT))

from datamodules import ArgoverseV1DataModule  # noqa: E402
from models.cascade import (  # noqa: E402
    CascadedHiVT,
    CascadedHiVTEDLMLP,
    CascadedHiVTMLPKD,
    filter_missing_keys_for_hivt_load,
    load_dualhead_small_decoder_weights,
    load_hivt_encoder_and_full_decoder,
)


def _load_yaml_config(path: Path) -> dict[str, Any]:
    import yaml

    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    pl.seed_everything(2022)

    parser = ArgumentParser(
        description="CascadedHiVT 적응 학습 (Slim EDL / EDLMLP small / MLP KD)"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help="YAML 경로 (미지정 시 configs/cascade_edl_nll_kl.yaml 자동)",
    )
    parser.add_argument(
        "--no-config",
        action="store_true",
        help="YAML을 읽지 않고 아래 CLI 기본값만 사용",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(_ROOT / "datasets" / "argoverse_v1"),
    )
    parser.add_argument(
        "--hivt_ckpt",
        type=str,
        default=str(_ROOT / "checkpoints" / "HiVT" / "epoch=63-step=411903.ckpt"),
        help="HiVT 인코더 + full MLPDecoder",
    )
    parser.add_argument(
        "--cascade_model",
        type=str,
        default="cascaded_hivt",
        choices=("cascaded_hivt", "cascaded_hivt_edl_mlp", "cascaded_hivt_mlp_kd"),
        help=(
            "cascaded_hivt=SlimEDL small | cascaded_hivt_edl_mlp=EDLMLPDecoder(small) | "
            "cascaded_hivt_mlp_kd=Full 동일 MLP + KD"
        ),
    )
    parser.add_argument(
        "--dualhead_ckpt",
        type=str,
        default=str(_ROOT / "checkpoints" / "DualHead" / "epoch=61-step=12524.ckpt"),
        help="DualHead EDL 디코더 → small_decoder 이식 (MLP KD 모드에서는 사용 안 함)",
    )
    parser.add_argument("--skip_weight_load", action="store_true", help="가중치 로드 생략(구조만)")
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--persistent_workers", type=bool, default=True)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument(
        "--cuda_device",
        type=int,
        default=None,
        help="단일 물리 GPU만 사용 (예: 3 → Lightning devices=[3], DDP 비사용).",
    )
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--monitor", type=str, default="val_minFDE")
    parser.add_argument(
        "--save_top_k",
        type=int,
        default=-1,
        help="ModelCheckpoint: 상위 k개만 유지. -1이면 검증마다 매 에폭 체크포인트 전부 저장(디스크 사용량↑)",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="그래디언트 누적 (메모리 부족 시 배치↓·이 값↑)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        help="Lightning mixed precision (16-mixed 권장; 32=FP32)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/CascadedHiVT",
        help="ModelCheckpoint 저장 디렉터리 (프로젝트 루트 기준 상대 경로 또는 절대 경로)",
    )
    parser = CascadedHiVTMLPKD.add_model_specific_args(parser)

    args = parser.parse_args()
    os.chdir(_ROOT)

    if not args.no_config:
        cfg_path = args.config if args.config is not None else _default_config_path()
        if cfg_path is not None and cfg_path.is_file():
            cfg = _load_yaml_config(cfg_path)
            for k, v in cfg.items():
                if hasattr(args, k) and v is not None:
                    setattr(args, k, v)
            print(f"[cascade] 설정 파일 로드: {cfg_path.resolve()}")
        elif args.config is not None:
            raise FileNotFoundError(f"설정 파일 없음: {cfg_path}")
        elif cfg_path is None:
            print(
                "[cascade] 경고: configs/cascade_edl_nll_kl.yaml 없음 — CLI 기본값만 사용합니다."
            )

    # YAML이 --gpus 를 덮어쓰므로, run_train_then_tau_sweep 등에서 TRAIN_NUM_GPUS 로 최종 덮어쓰기 가능
    _ng = os.environ.get("TRAIN_NUM_GPUS")
    if _ng is not None and _ng.strip().isdigit():
        args.gpus = int(_ng)
        print(f"[cascade] TRAIN_NUM_GPUS={args.gpus} 로 GPU 개수 고정")

    prec = getattr(args, "precision", "16-mixed")
    if prec in (16, "16"):
        prec = "16-mixed"
    elif prec in (32, "32"):
        prec = "32"

    if not args.skip_weight_load:
        if not Path(args.hivt_ckpt).is_file():
            raise FileNotFoundError(
                f"HiVT 체크포인트 없음: {args.hivt_ckpt}\n"
                "파일을 복사하거나 --hivt_ckpt 로 경로를 지정하세요."
            )
        if args.cascade_model in ("cascaded_hivt", "cascaded_hivt_edl_mlp") and not Path(
            args.dualhead_ckpt
        ).is_file():
            raise FileNotFoundError(
                f"DualHead 체크포인트 없음: {args.dualhead_ckpt}\n"
                "파일을 복사하거나 --dualhead_ckpt 로 지정하세요."
            )

    base_keys = [
        "historical_steps",
        "future_steps",
        "num_modes",
        "rotate",
        "node_dim",
        "edge_dim",
        "embed_dim",
        "small_embed_dim",
        "num_heads",
        "dropout",
        "num_temporal_layers",
        "num_global_layers",
        "local_radius",
        "parallel",
        "lr",
        "weight_decay",
        "T_max",
        "edl_lambda_reg",
        "edl_lambda_warmup_epochs",
        "freeze_encoder_and_full",
        "proj_lr_mult",
        "proj_high_lr_epochs",
        "log_full_val_metrics",
        "eval_full_only",
        "small_path_local_only",
        "log_t_ps_each_epoch",
        "t_ps_benchmark_batches",
    ]
    model_kw = {k: getattr(args, k) for k in base_keys}
    model_kw["cascade_model"] = args.cascade_model
    if args.cascade_model == "cascaded_hivt_mlp_kd":
        model_kw["kd_alpha"] = getattr(args, "kd_alpha", 0.5)
        model_kw["kd_temp"] = getattr(args, "kd_temp", 1.0)
        model_kw["kd_gt_boost"] = getattr(args, "kd_gt_boost", 0.5)
        model_kw["kd_u_alpha"] = getattr(args, "kd_u_alpha", 0.25)
        model_kw["strict_clone_small_from_full"] = getattr(args, "strict_clone_small_from_full", True)
        model_kw["mlp_kd_use_proj"] = getattr(args, "mlp_kd_use_proj", True)
        model = CascadedHiVTMLPKD(**model_kw)
    elif args.cascade_model == "cascaded_hivt_edl_mlp":
        model = CascadedHiVTEDLMLP(**model_kw)
    else:
        model = CascadedHiVT(**model_kw)

    if not args.skip_weight_load:
        miss, unexp = load_hivt_encoder_and_full_decoder(model, args.hivt_ckpt)
        exp_miss, sus_miss = filter_missing_keys_for_hivt_load(miss)
        print(
            "[cascade] HiVT load: missing total",
            len(miss),
            "(의도 proj/small_decoder:",
            len(exp_miss),
            ") / unexpected ckpt keys:",
            len(unexp),
        )
        if sus_miss:
            warnings.warn(
                "[cascade] HiVT missing_keys에 proj/small 외 키가 있습니다 (인코더·full 이식 누락 가능): "
                + str(sus_miss[:20])
                + (" ..." if len(sus_miss) > 20 else ""),
                stacklevel=2,
            )
        if unexp:
            warnings.warn(f"HiVT unexpected keys (first 5): {unexp[:5]}", stacklevel=2)

        if args.cascade_model == "cascaded_hivt_mlp_kd":
            # strict copy 는 CascadedHiVTMLPKD.on_train_start 에서 수행(재개 시 자동 생략).
            print(
                "[cascade] MLP KD: 학습 루프 시작 시 on_train_start 에서 full_decoder→small_decoder 복사"
            )
        else:
            rep = load_dualhead_small_decoder_weights(model.small_decoder, args.dualhead_ckpt)
            print("[cascade] DualHead small_decoder:", json.dumps(rep, indent=2, ensure_ascii=False))

    ckpt_path = Path(args.checkpoint_dir)
    ckpt_dir = ckpt_path if ckpt_path.is_absolute() else (_ROOT / ckpt_path).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"[cascade] 체크포인트 저장: {ckpt_dir}")
    _mon = args.monitor
    # Lightning filename 플레이스홀더: monitor 이름과 일치해야 함 (예: val_minFDE={val_minFDE:.4f})
    _ckpt_filename = "cascade-{epoch:02d}-" + _mon + "={" + _mon + ":.4f}"
    ckpt_cb = ModelCheckpoint(
        monitor=args.monitor,
        save_top_k=args.save_top_k,
        mode="min",
        dirpath=str(ckpt_dir),
        filename=_ckpt_filename,
    )
    if args.save_top_k < 0:
        print("[cascade] save_top_k=-1 → 검증이 끝난 매 에폭마다 ckpt 전부 저장")

    trainer_kw: dict[str, Any] = {
        "max_epochs": args.max_epochs,
        "accumulate_grad_batches": int(getattr(args, "accumulate_grad_batches", 1)),
        "precision": prec,
        "callbacks": [ckpt_cb],
        "logger": [
            TensorBoardLogger(save_dir=str(_ROOT / "lightning_logs")),
            CSVLogger(save_dir=str(_ROOT / "lightning_logs")),
        ],
    }
    if args.gpus and torch.cuda.is_available():
        trainer_kw["accelerator"] = "gpu"
        if args.cuda_device is not None:
            trainer_kw["devices"] = [int(args.cuda_device)]
        else:
            trainer_kw["devices"] = args.gpus
            if args.gpus > 1:
                trainer_kw["strategy"] = "ddp"
    else:
        trainer_kw["accelerator"] = "cpu"
        trainer_kw["devices"] = 1

    trainer = pl.Trainer(**trainer_kw)
    datamodule = ArgoverseV1DataModule(
        root=args.root,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
    )
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
