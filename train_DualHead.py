# DualHead training. Run from project root (dualhead/).
# Uses Argoverse v1 under --root (e.g. datasets/argoverse_v1 with train/processed/*.pt).
from __future__ import annotations

import json
import logging
import sys
import warnings
from argparse import ArgumentParser

# Triton 미설치 시 torch.utils.flop_counter 경고 숨김 (Windows 등; FLOP 카운트만 비활성)
warnings.filterwarnings(
    "ignore",
    message=".*triton not found.*flop counting.*",
)
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

_ROOT = Path(__file__).resolve().parent
_ARGO = _ROOT / "datasets" / "argoverse-api-master"
sys.path.insert(0, str(_ARGO))
sys.path.insert(0, str(_ROOT))

from datamodules import ArgoverseV1DataModule  # noqa: E402
from models import DualHead  # noqa: E402

# 인자 없이 실행할 때 이 경로의 파일이 있으면 자동 로드
DEFAULT_CONFIG_PATH = _ROOT / "configs" / "dualhead_train.yaml"


def _parse_known_config_argv(
    argv: list[str],
) -> tuple[Path | None, list[str], bool]:
    pre = ArgumentParser(add_help=False)
    pre.add_argument("--config", "-c", type=Path, default=None)
    pre.add_argument(
        "--no-default-config",
        action="store_true",
        default=False,
        help="기본 configs/dualhead_train.yaml 자동 로드를 끕니다.",
    )
    known, rest = pre.parse_known_args(argv)
    return known.config, rest, known.no_default_config


def _resolve_config_path(
    explicit: Path | None, no_default: bool
) -> Path | None:
    if explicit is not None:
        return explicit
    if no_default:
        return None
    if DEFAULT_CONFIG_PATH.is_file():
        return DEFAULT_CONFIG_PATH
    return None


def _load_config_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "YAML 설정 파일을 쓰려면 PyYAML이 필요합니다: pip install pyyaml"
            ) from e
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
    elif suffix == ".json":
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError(f"지원하지 않는 설정 확장자: {suffix} (.yaml, .yml, .json)")
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("설정 파일 루트는 객체(키-값)여야 합니다.")
    return data


def _apply_config_to_parser(parser: ArgumentParser, cfg: dict[str, Any]) -> None:
    valid = {a.dest for a in parser._actions if a.dest and a.dest != "help"}
    skip = {"config"}
    unknown = set(cfg.keys()) - valid - skip
    if unknown:
        warnings.warn(
            "설정 파일에 알 수 없는 키가 있어 무시합니다: " + ", ".join(sorted(unknown)),
            stacklevel=2,
        )
    filtered = {k: v for k, v in cfg.items() if k in valid and k not in skip}
    parser.set_defaults(**filtered)


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default=str(_ROOT / "datasets" / "argoverse_v1"),
        help="Argoverse v1 root (contains train/, val/, ... with data/ and processed/)",
    )
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="If set, overrides both train_batch_size and val_batch_size to this value.",
    )
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--persistent_workers", type=bool, default=True)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=64)
    parser.add_argument(
        "--monitor",
        type=str,
        default="val_minFDE",
        choices=["val_minADE", "val_minFDE", "val_minMR"],
    )
    parser.add_argument("--save_top_k", type=int, default=5)
    parser.add_argument(
        "--precision",
        type=str,
        default="32-true",
        help="Trainer precision: e.g. 16-mixed, bf16-mixed, 32-true (see PyTorch Lightning Trainer).",
    )
    parser.add_argument(
        "--limit_train_batches",
        type=float,
        default=1.0,
        help=(
            "에폭당 학습 스텝 제한. 0~1이면 전체 배치 수의 비율(Lightning limit_train_batches), "
            "≥1 정수면 배치 개수."
        ),
    )
    parser.add_argument(
        "--matmul_precision",
        type=str,
        default="highest",
        choices=["highest", "high", "medium"],
        help="torch.set_float32_matmul_precision (Tensor Core 활용 등).",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help=(
            "Lightning accumulate_grad_batches. GPU당 배치가 작을 때 누적 스텝으로 유효 배치를 키움 "
            "(예: train_batch_size=32, 이 값=8 → GPU당 유효 256)."
        ),
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help=(
            "YAML/JSON 경로 (미지정 시 "
            f"{DEFAULT_CONFIG_PATH.relative_to(_ROOT)} 가 있으면 자동 사용). "
            "CLI 인자가 설정을 덮어씁니다."
        ),
    )
    parser.add_argument(
        "--no-default-config",
        action="store_true",
        default=False,
        help="기본 설정 파일 자동 로드를 끕니다.",
    )
    parser = DualHead.add_model_specific_args(parser)
    return parser


def _build_trainer(args: Any, callbacks: list) -> pl.Trainer:
    """Lightning 2.x: ``Trainer.from_argparse_args`` 제거 대응."""
    kw: dict[str, Any] = {
        "max_epochs": args.max_epochs,
        "precision": args.precision,
        "callbacks": callbacks,
        "limit_train_batches": args.limit_train_batches,
        "accumulate_grad_batches": getattr(args, "accumulate_grad_batches", 1),
    }
    if getattr(args, "gpus", 0) and torch.cuda.is_available():
        kw["accelerator"] = "gpu"
        kw["devices"] = args.gpus
    else:
        if getattr(args, "gpus", 0):
            warnings.warn(
                "CUDA를 사용할 수 없어 CPU로 학습합니다.",
                stacklevel=2,
            )
        kw["accelerator"] = "cpu"
        kw["devices"] = 1
    return pl.Trainer(**kw)


def main() -> None:
    pl.seed_everything(2022)

    explicit_config, argv_rest, no_default = _parse_known_config_argv(
        sys.argv[1:]
    )
    config_path = _resolve_config_path(explicit_config, no_default)
    parser = _build_parser()
    if config_path is not None:
        _apply_config_to_parser(parser, _load_config_file(config_path))
    parser.set_defaults(config=config_path)
    args = parser.parse_args(argv_rest)
    if args.embed_dim is None:
        parser.error(
            "embed_dim이 필요합니다. --embed_dim, "
            f"{DEFAULT_CONFIG_PATH.relative_to(_ROOT)} (자동 로드), "
            "또는 --config 경로에 embed_dim을 넣으세요."
        )

    if args.batch_size is not None:
        args.train_batch_size = args.batch_size
        args.val_batch_size = args.batch_size

    torch.set_float32_matmul_precision(args.matmul_precision)

    ckpt_dir = _ROOT / "checkpoints" / "DualHead"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_checkpoint = ModelCheckpoint(
        monitor=args.monitor,
        save_top_k=args.save_top_k,
        mode="min",
        dirpath=str(ckpt_dir),
    )
    trainer = _build_trainer(args, [model_checkpoint])
    model_kw = vars(args).copy()
    model_kw.pop("batch_size", None)
    model_kw.pop("precision", None)
    model_kw.pop("config", None)
    model_kw.pop("limit_train_batches", None)
    model_kw.pop("matmul_precision", None)
    model_kw.pop("accumulate_grad_batches", None)
    model = DualHead(**model_kw)
    datamodule = ArgoverseV1DataModule.from_argparse_args(args)
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
