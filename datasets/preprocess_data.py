# Copyright (c) 2022, Zikang Zhou. All rights reserved.
# Preprocess Argoverse CSVs to .pt (same as ArgoverseV1Dataset.process).
#
# 이 파이프라인은 맵 조회·CSV·CPU 텐서 연산이 대부분이라 GPU 가속 이득이 거의 없고,
# 병렬화는 멀티프로세스(CPU 코어)로 하는 것이 맞다.
from __future__ import annotations

import os
import sys
from argparse import ArgumentParser
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_ARGO = _ROOT / "datasets" / "argoverse-api-master"
sys.path.insert(0, str(_ARGO))
sys.path.insert(0, str(_ROOT))

from datasets import ArgoverseV1Dataset  # noqa: E402


def _resolve_num_workers(cli_value: int) -> int:
    if cli_value > 0:
        return cli_value
    return os.cpu_count() or 1


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test", "sample"],
    )
    parser.add_argument("--local_radius", type=float, default=50.0)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="병렬 워커 수. 0이면 논리 CPU 개수(os.cpu_count()) 사용.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="이미 있는 .pt는 건너뛰기.",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="단일 프로세스만 사용(디버그). --num_workers 무시.",
    )
    args = parser.parse_args()

    nw = 1 if args.sequential else _resolve_num_workers(args.num_workers)
    # 워커마다 OpenMP/MKL이 코어를 나눠 쓰면 오히려 느려짐
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    ds = ArgoverseV1Dataset(
        root=args.root,
        split=args.split,
        local_radius=args.local_radius,
        defer_process=True,
        num_workers=nw,
        resume=args.resume,
    )
    print(f"Preprocessing finished: {len(ds)} sequences -> {ds.processed_dir}")
