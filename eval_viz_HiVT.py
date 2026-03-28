# Subset evaluation + Argoverse map visualization. Run from project root (dualhead/).
from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

_ROOT = Path(__file__).resolve().parent
_ARGO = _ROOT / "datasets" / "argoverse-api-master"
sys.path.insert(0, str(_ARGO))
sys.path.insert(0, str(_ROOT))

from argoverse.map_representation.map_api import ArgoverseMap  # noqa: E402

from datasets import ArgoverseV1Dataset  # noqa: E402
from metrics_io import save_metrics_json  # noqa: E402
from models import HiVT  # noqa: E402
from visualize import plot_hivt_scene  # noqa: E402


def main() -> None:
    parser = ArgumentParser(description="HiVT subset eval + map PNGs + metrics.json")
    parser.add_argument(
        "--root",
        type=str,
        default=str(_ROOT / "datasets" / "argoverse_v1"),
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=str(_ROOT / "checkpoints" / "HiVT" / "epoch=63-step=411903.ckpt"),
    )
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=("train", "val", "test", "sample"),
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = _ROOT / "results" / f"eval_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if args.gpus and torch.cuda.is_available() else "cpu")

    model = HiVT.load_from_checkpoint(
        checkpoint_path=args.ckpt_path,
        parallel=True,
        weights_only=False,
    )
    model = model.to(device)
    model.eval()

    full_ds = ArgoverseV1Dataset(
        root=args.root,
        split=args.split,
        local_radius=model.hparams.local_radius,
    )
    n = min(args.num_samples, len(full_ds))
    subset = Subset(full_ds, list(range(n)))
    loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    am = ArgoverseMap()

    per_sample = []
    sum_ade = 0.0
    sum_fde = 0.0
    valid = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            data = data.to(device)
            y_hat, pi = model(data)

            agent_idx = int(data["agent_index"].item())
            y_agent = data.y[agent_idx]
            reg_mask = ~data["padding_mask"][agent_idx, model.historical_steps :]
            y_hat_a = y_hat[:, agent_idx, :, :]

            diff = (
                torch.norm(y_hat_a[..., :2] - y_agent.unsqueeze(0), p=2, dim=-1)
                * reg_mask.unsqueeze(0)
            )
            den = reg_mask.sum().clamp(min=1)
            ade_modes = diff.sum(dim=-1) / den
            fde_modes = diff[..., -1]
            min_ade = ade_modes.min().item()
            min_fde = fde_modes.min().item()

            seq_id = int(data["seq_id"].item())
            per_sample.append(
                {
                    "index": batch_idx,
                    "seq_id": seq_id,
                    "minADE": min_ade,
                    "minFDE": min_fde,
                }
            )
            sum_ade += min_ade
            sum_fde += min_fde
            valid += 1

            title = f"seq {seq_id} | minADE={min_ade:.3f} minFDE={min_fde:.3f}"
            img_path = out_dir / f"sample_{batch_idx:03d}_seq_{seq_id}.png"
            plot_hivt_scene(
                am,
                data.cpu(),
                y_hat.cpu(),
                img_path,
                title=title,
                pi=pi.cpu(),
                dataset_root=args.root,
                split=args.split,
            )

    metrics = {
        "checkpoint": args.ckpt_path,
        "dataset_root": args.root,
        "num_samples": n,
        "mean_minADE": sum_ade / max(valid, 1),
        "mean_minFDE": sum_fde / max(valid, 1),
        "per_sample": per_sample,
    }
    save_metrics_json(out_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
