# HiVT training. Run from project root (dualhead/).
from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

_ROOT = Path(__file__).resolve().parent
_ARGO = _ROOT / "datasets" / "argoverse-api-master"
sys.path.insert(0, str(_ARGO))
sys.path.insert(0, str(_ROOT))

from datamodules import ArgoverseV1DataModule  # noqa: E402
from models import HiVT  # noqa: E402


def main() -> None:
    pl.seed_everything(2022)

    parser = ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Argoverse v1 root (contains train/val/...)")
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=32)
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
    parser = HiVT.add_model_specific_args(parser)
    args = parser.parse_args()

    model_checkpoint = ModelCheckpoint(monitor=args.monitor, save_top_k=args.save_top_k, mode="min")
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[model_checkpoint])
    model = HiVT(**vars(args))
    datamodule = ArgoverseV1DataModule.from_argparse_args(args)
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
