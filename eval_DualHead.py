# Full validation set evaluation for DualHead (PyTorch Lightning). Run from project root.
from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch_geometric.loader import DataLoader

_ROOT = Path(__file__).resolve().parent
_ARGO = _ROOT / "datasets" / "argoverse-api-master"
sys.path.insert(0, str(_ARGO))
sys.path.insert(0, str(_ROOT))

from datasets import ArgoverseV1Dataset  # noqa: E402
from models import DualHead  # noqa: E402


def main() -> None:
    pl.seed_everything(2022)

    parser = ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Argoverse v1 root")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--persistent_workers", type=bool, default=True)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument(
        "--parallel",
        type=bool,
        default=False,
        help="Must match training (see dualhead_train.yaml parallel).",
    )
    args = parser.parse_args()

    if args.gpus and torch.cuda.is_available():
        trainer = pl.Trainer(accelerator="gpu", devices=args.gpus)
    else:
        trainer = pl.Trainer(accelerator="cpu", devices=1)

    model = DualHead.load_from_checkpoint(
        checkpoint_path=args.ckpt_path,
        parallel=args.parallel,
        weights_only=False,
    )
    val_dataset = ArgoverseV1Dataset(
        root=args.root,
        split="val",
        local_radius=model.hparams.local_radius,
    )
    persistent_workers = args.persistent_workers and args.num_workers > 0
    dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=persistent_workers,
    )
    trainer.validate(model, dataloader)


if __name__ == "__main__":
    main()
