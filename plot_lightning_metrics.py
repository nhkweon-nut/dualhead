#!/usr/bin/env python3
"""Lightning CSVLogger metrics.csv → PNG 그래프. TensorBoard 없이 로그 시각화용."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def main() -> None:
    p = argparse.ArgumentParser(description="Plot lightning_logs/.../metrics.csv")
    p.add_argument(
        "csv",
        type=Path,
        nargs="?",
        default=Path("lightning_logs/version_4/metrics.csv"),
        help="metrics.csv 경로 (기본: lightning_logs/version_4/metrics.csv)",
    )
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="출력 PNG (기본: CSV와 같은 폴더의 metrics_plots.png)",
    )
    args = p.parse_args()
    csv_path = args.csv.resolve()
    if not csv_path.is_file():
        raise SystemExit(f"파일 없음: {csv_path}")

    df = pd.read_csv(csv_path)
    for c in df.columns:
        if c not in ("epoch", "step"):
            df[c] = _numeric(df[c])

    out = args.out or (csv_path.parent / "metrics_plots.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    # --- 검증 (에폭 끝에 찍힌 행)
    val = df.dropna(subset=["val_minADE"]).copy()
    if val.empty:
        val = df.dropna(subset=["val_minFDE"])

    # --- 에폭 단위 학습 로스 (train_*_epoch 가 채워진 행)
    tr_ep = df.dropna(subset=["train_reg_loss_epoch"]).copy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(f"Lightning metrics — {csv_path.parent.name}", fontsize=12)

    # Val metrics
    ax = axes[0, 0]
    if not val.empty:
        x = val["epoch"].values
        ax.plot(x, val["val_minADE"], label="val_minADE", marker="o", ms=3)
        ax.plot(x, val["val_minFDE"], label="val_minFDE", marker="o", ms=3)
        ax.plot(x, val["val_minMR"], label="val_minMR", marker="o", ms=3)
        ax.set_xlabel("epoch")
        ax.set_ylabel("metric")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    ax.set_title("Validation (minADE / minFDE / MR)")

    ax = axes[0, 1]
    if not val.empty:
        ax.plot(val["epoch"], val["val_reg_loss"], label="val_reg_loss", color="C0")
        if val["val_edl_nll"].notna().any():
            ax.plot(val["epoch"], val["val_edl_nll"], label="val_edl_nll", color="C1")
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    ax.set_title("Validation loss")

    # Train epoch
    ax = axes[1, 0]
    if not tr_ep.empty:
        ax.plot(tr_ep["epoch"], tr_ep["train_reg_loss_epoch"], label="train_reg_loss", marker="o", ms=2)
        if tr_ep["train_cls_loss_epoch"].notna().any():
            ax.plot(tr_ep["epoch"], tr_ep["train_cls_loss_epoch"], label="train_cls_loss", marker="o", ms=2)
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    ax.set_title("Train (epoch aggregate)")

    # Train step (샘플링 — 너무 많으면 느려질 수 있어 최대 5000점)
    ax = axes[1, 1]
    ts = df.dropna(subset=["train_reg_loss_step"]).copy()
    if len(ts) > 5000:
        ts = ts.iloc[:: max(1, len(ts) // 5000)]
    if not ts.empty:
        ax.plot(ts["step"], ts["train_reg_loss_step"], lw=0.6, alpha=0.8, label="train_reg_loss_step")
        ax.set_xlabel("step")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    ax.set_title("Train (per-step, subsampled if long)")

    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
