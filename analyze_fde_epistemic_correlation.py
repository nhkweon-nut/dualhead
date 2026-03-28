# Val 전체: FDE vs Epistemic 상관·산점도 + 아웃라이어/일반 시나리오 맵 시각화 (한 스크립트).
# python analyze_fde_epistemic_correlation.py
# 시각화만(기존 per_sample_metrics.json): python analyze_fde_epistemic_correlation.py --only_scenario_viz --correlation_dir results/fde_epistemic_corr_*
from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

_ROOT = Path(__file__).resolve().parent
_ARGO = _ROOT / "datasets" / "argoverse-api-master"
sys.path.insert(0, str(_ARGO))
sys.path.insert(0, str(_ROOT))

from argoverse.map_representation.map_api import ArgoverseMap  # noqa: E402

from datasets import ArgoverseV1Dataset  # noqa: E402
from models import DualHead  # noqa: E402
from models.DualHead.nig_uncertainty import (  # noqa: E402
    combine_xy_uncertainty,
    nig_epistemic_uncertainty,
)
from visualize import load_actor_object_labels, plot_hivt_scene  # noqa: E402


def _scatter_fde_vs_epistemic(
    ax,
    fde: np.ndarray,
    epi: np.ndarray,
    fde_th: float,
    epi_th: float,
    *,
    xlim: tuple[float, float] | None = None,
) -> None:
    """FDE(y) vs Epistemic(x). 색: 인리어 / FDE만 / Epistemic만 / 둘 다."""
    m_in = (fde <= fde_th) & (epi <= epi_th)
    m_fde = (fde > fde_th) & (epi <= epi_th)
    m_epi = (epi > epi_th) & (fde <= fde_th)
    m_both = (fde > fde_th) & (epi > epi_th)

    n_in = int(np.count_nonzero(m_in))
    n_fde = int(np.count_nonzero(m_fde))
    n_epi = int(np.count_nonzero(m_epi))
    n_both = int(np.count_nonzero(m_both))

    ax.scatter(
        epi[m_in],
        fde[m_in],
        s=10,
        alpha=0.35,
        c="#1f77b4",
        edgecolors="none",
        label=f"Inlier (n={n_in})",
    )
    ax.scatter(
        epi[m_fde],
        fde[m_fde],
        s=14,
        alpha=0.7,
        c="#d62728",
        edgecolors="none",
        label=f"FDE outlier only (n={n_fde})",
    )
    ax.scatter(
        epi[m_epi],
        fde[m_epi],
        s=14,
        alpha=0.7,
        c="#2ca02c",
        edgecolors="none",
        label=f"Epistemic outlier only (n={n_epi})",
    )
    ax.scatter(
        epi[m_both],
        fde[m_both],
        s=16,
        alpha=0.85,
        c="#9467bd",
        edgecolors="none",
        label=f"Both (n={n_both})",
    )
    ax.set_xlabel("Epistemic uncertainty (L2), final step, best-FDE mode")
    ax.set_ylabel("FDE (m), min over modes at final step")
    ax.axhline(fde_th, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(epi_th, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])


def _scene_context(
    dataset_root: str,
    split: str,
    seq_id: int,
    num_nodes: int,
    data: Any,
    historical_steps: int,
) -> dict[str, Any]:
    ctx: dict[str, Any] = {
        "num_graph_nodes": int(num_nodes),
        "num_valid_agents_t19": int((~data["padding_mask"][:, historical_steps - 1]).sum().item()),
    }
    labels = load_actor_object_labels(dataset_root, split, seq_id)
    if labels:
        types_upper = [str(x).upper().strip() for x in labels]
        ctx["object_types_in_hist_order"] = types_upper
        ctx["n_actors_hist"] = len(types_upper)
        ctx["n_vehicle"] = sum(1 for t in types_upper if "VEHICLE" in t and "PED" not in t)
        ctx["n_pedestrian"] = sum(1 for t in types_upper if "PED" in t)
        ctx["n_bicycle"] = sum(1 for t in types_upper if "BICYCLE" in t or "BIKE" in t)
    else:
        ctx["n_actors_hist"] = None
    return ctx


def run_scenario_visualization(
    *,
    model: DualHead,
    val_ds: ArgoverseV1Dataset,
    root: Path,
    scenario_dir: Path,
    per_sample: list[dict],
    fde_th: float,
    epi_th: float,
    n_worst_fde: int,
    n_worst_epistemic: int,
    n_best_fde: int,
    device: torch.device,
) -> None:
    """out_dir/scenario_viz: FDE 최악 k, Epistemic 최악 k, FDE 최선 k(잘 예측)만 PNG 저장."""
    def _as_float(row: dict, key: str) -> float:
        return float(row[key])

    rows_fde_gt = [r for r in per_sample if _as_float(r, "fde") > fde_th]
    rows_epi_gt = [r for r in per_sample if _as_float(r, "epistemic_l2") > epi_th]

    epi_vals = [_as_float(r, "epistemic_l2") for r in per_sample]
    if epi_vals:
        print(
            f"[scenario_viz] epistemic_l2: min={min(epi_vals):.4f} max={max(epi_vals):.4f} "
            f"(>{epi_th} count={len(rows_epi_gt)} for scatter threshold)"
        )

    sorted_by_fde_desc = sorted(per_sample, key=lambda r: _as_float(r, "fde"), reverse=True)
    sorted_by_epi_desc = sorted(per_sample, key=lambda r: _as_float(r, "epistemic_l2"), reverse=True)
    sorted_by_fde_asc = sorted(per_sample, key=lambda r: _as_float(r, "fde"))

    kf = max(0, int(n_worst_fde))
    ke = max(0, int(n_worst_epistemic))
    kb = max(0, int(n_best_fde))

    fde_to_plot = sorted_by_fde_desc[:kf]
    epi_to_plot = sorted_by_epi_desc[:ke]
    typical_rows = sorted_by_fde_asc[:kb]

    (scenario_dir / "outlier_fde").mkdir(parents=True, exist_ok=True)
    (scenario_dir / "outlier_epistemic").mkdir(parents=True, exist_ok=True)
    (scenario_dir / "typical").mkdir(parents=True, exist_ok=True)

    am = ArgoverseMap()
    hs = int(model.historical_steps)
    root_str = str(root)

    def run_one(row: dict, kind: str, subfolder: str) -> dict:
        vi = int(row["val_index"])
        seq_id = int(row["seq_id"])
        loader = DataLoader(
            Subset(val_ds, [vi]),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        with torch.no_grad():
            batch = next(iter(loader)).to(device)
            y_hat, pi = model(batch)

        num_nodes = int(batch.num_nodes)
        b_cpu = batch.cpu()
        ctx = _scene_context(root_str, "val", seq_id, num_nodes, b_cpu, hs)

        fd = float(row["fde"])
        ep = float(row["epistemic_l2"])
        title = (
            f"{kind} | val_idx={vi} seq={seq_id} | FDE={fd:.2f} Epi={ep:.2f} | "
            f"nodes={num_nodes} valid@t19={ctx['num_valid_agents_t19']}"
        )
        if ctx.get("n_actors_hist") is not None:
            title += f" | Veh≈{ctx.get('n_vehicle')} Ped≈{ctx.get('n_pedestrian')}"

        fname = f"val{vi:05d}_seq_{seq_id}.png"
        out_path = scenario_dir / subfolder / fname
        plot_hivt_scene(
            am,
            b_cpu,
            y_hat.cpu(),
            out_path,
            title=title,
            pi=pi.cpu(),
            dataset_root=root_str,
            split="val",
        )

        try:
            png_rel = str(out_path.relative_to(_ROOT))
        except ValueError:
            png_rel = str(out_path)
        return {
            "kind": kind,
            "val_index": vi,
            "seq_id": seq_id,
            "fde": fd,
            "epistemic_l2": ep,
            "png": png_rel,
            **ctx,
        }

    summary: list[dict] = []
    plot_errors: list[str] = []

    def run_one_safe(row: dict, kind: str, subfolder: str) -> None:
        try:
            summary.append(run_one(row, kind, subfolder))
        except Exception as ex:
            vi = row.get("val_index", "?")
            plot_errors.append(f"{subfolder} val_index={vi}: {ex!r}")

    for row in fde_to_plot:
        run_one_safe(row, "WORST_FDE", "outlier_fde")
    for row in epi_to_plot:
        run_one_safe(row, "WORST_EPI", "outlier_epistemic")
    for row in typical_rows:
        run_one_safe(row, "BEST_FDE", "typical")

    if plot_errors:
        err_path = scenario_dir / "plot_failures.txt"
        err_path.write_text("\n".join(plot_errors), encoding="utf-8")
        print(f"[scenario_viz] WARNING: {len(plot_errors)} plot failures → {err_path}")

    (scenario_dir / "comparison_summary.json").write_text(
        json.dumps(
            {
                "selection": {
                    "outlier_fde": f"top {kf} worst FDE (highest fde)",
                    "outlier_epistemic": f"top {ke} worst epistemic_l2",
                    "typical": f"top {kb} best FDE (lowest fde)",
                },
                "n_outlier_fde_plotted": len(fde_to_plot),
                "n_above_fde_threshold": len(rows_fde_gt),
                "n_outlier_epistemic_plotted": len(epi_to_plot),
                "n_above_epistemic_threshold": len(rows_epi_gt),
                "n_typical_plotted": len(typical_rows),
                "thresholds": {"fde_gt": fde_th, "epistemic_gt": epi_th},
                "folders": {
                    "fde": "outlier_fde",
                    "epistemic": "outlier_epistemic",
                    "typical": "typical (best FDE)",
                },
                "items": summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"\n[scenario_viz] {scenario_dir / 'outlier_fde'}: {len(fde_to_plot)} PNG (worst FDE top {kf})")
    print(f"[scenario_viz] {scenario_dir / 'outlier_epistemic'}: {len(epi_to_plot)} PNG (worst Epistemic top {ke})")
    print(f"[scenario_viz] {scenario_dir / 'typical'}: {len(typical_rows)} PNG (best FDE top {kb})")
    print(f"[scenario_viz] {scenario_dir / 'comparison_summary.json'}")


def main() -> None:
    os.chdir(_ROOT)

    parser = ArgumentParser(
        description="Val: FDE–Epistemic 상관·산점도 + 아웃라이어/일반 맵 시각화(통합)."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(_ROOT / "datasets" / "argoverse_v1"),
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=str(_ROOT / "checkpoints" / "DualHead" / "epoch=61-step=12524.ckpt"),
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--limit_batches", type=int, default=None)
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="기본: results/fde_epistemic_corr_<timestamp>/",
    )
    parser.add_argument("--fde_outlier_threshold", type=float, default=5.0)
    parser.add_argument("--epistemic_outlier_threshold", type=float, default=10.0)
    parser.add_argument(
        "--no_scenario_viz",
        action="store_true",
        help="상관·산점도·JSON만 하고 맵 시각화는 생략.",
    )
    parser.add_argument(
        "--only_scenario_viz",
        action="store_true",
        help="이미 저장된 per_sample_metrics.json 만으로 시나리오 시각화만 수행.",
    )
    parser.add_argument(
        "--per_sample_json",
        type=str,
        default=None,
        help="--only_scenario_viz 일 때 메트릭 파일 (--correlation_dir 보다 우선).",
    )
    parser.add_argument(
        "--correlation_dir",
        type=str,
        default=None,
        help="--only_scenario_viz: 이 폴더의 per_sample_metrics.json 사용 (미지정 시 최신 fde_epistemic_corr_*).",
    )
    parser.add_argument(
        "--n_worst_fde",
        type=int,
        default=20,
        help="시나리오 PNG: FDE가 가장 나쁜 상위 N개 (outlier_fde/).",
    )
    parser.add_argument(
        "--n_worst_epistemic",
        type=int,
        default=20,
        help="시나리오 PNG: epistemic_l2가 가장 큰 상위 N개 (outlier_epistemic/).",
    )
    parser.add_argument(
        "--n_best_fde",
        type=int,
        default=20,
        help="시나리오 PNG: FDE가 가장 낮은(잘 예측) 상위 N개 (typical/).",
    )
    parser.add_argument(
        "--scatter_epi_xmax_zoom",
        type=float,
        default=300.0,
        help="추가 산점도에서 Epistemic(x) 축 범위 [0, 이 값]. 파일: fde_vs_epistemic_scatter_epi_zoom.png",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    ckpt_path = Path(args.ckpt_path).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {ckpt_path}")
    if not root.is_dir():
        raise FileNotFoundError(f"데이터 루트를 찾을 수 없습니다: {root}")

    device = torch.device("cuda" if args.gpus and torch.cuda.is_available() else "cpu")

    if args.only_scenario_viz:
        if args.per_sample_json:
            metrics_path = Path(args.per_sample_json).expanduser().resolve()
        elif args.correlation_dir:
            metrics_path = Path(args.correlation_dir).expanduser().resolve() / "per_sample_metrics.json"
        else:
            cand = sorted(_ROOT.glob("results/fde_epistemic_corr_*/per_sample_metrics.json"))
            if not cand:
                raise SystemExit(
                    "per_sample_metrics.json 을 찾을 수 없습니다. "
                    "--per_sample_json 또는 --correlation_dir 을 지정하세요."
                )
            metrics_path = cand[-1]
        if not metrics_path.is_file():
            raise FileNotFoundError(metrics_path)

        per_sample = json.loads(metrics_path.read_text(encoding="utf-8"))
        out_dir = metrics_path.parent
        scenario_dir = out_dir / "scenario_viz"

        model = DualHead.load_from_checkpoint(
            checkpoint_path=str(ckpt_path),
            parallel=args.parallel,
            weights_only=False,
        )
        model = model.to(device)
        model.eval()
        val_ds = ArgoverseV1Dataset(
            root=str(root),
            split="val",
            local_radius=model.hparams.local_radius,
        )
        run_scenario_visualization(
            model=model,
            val_ds=val_ds,
            root=root,
            scenario_dir=scenario_dir,
            per_sample=per_sample,
            fde_th=float(args.fde_outlier_threshold),
            epi_th=float(args.epistemic_outlier_threshold),
            n_worst_fde=args.n_worst_fde,
            n_worst_epistemic=args.n_worst_epistemic,
            n_best_fde=args.n_best_fde,
            device=device,
        )
        return

    try:
        from scipy.stats import pearsonr, spearmanr
    except ImportError as e:
        raise ImportError("pip install scipy") from e
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("pip install matplotlib") from e

    out_dir = Path(args.out_dir) if args.out_dir else _ROOT / "results" / f"fde_epistemic_corr_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = DualHead.load_from_checkpoint(
        checkpoint_path=str(ckpt_path),
        parallel=args.parallel,
        weights_only=False,
    )
    model = model.to(device)
    model.eval()

    val_ds = ArgoverseV1Dataset(
        root=str(root),
        split="val",
        local_radius=model.hparams.local_radius,
    )
    loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    fde_list: list[float] = []
    epi_list: list[float] = []
    per_sample: list[dict] = []
    val_offset = 0

    fde_th = float(args.fde_outlier_threshold)
    epi_th = float(args.epistemic_outlier_threshold)

    with torch.no_grad():
        for bi, data in enumerate(loader):
            if args.limit_batches is not None and bi >= args.limit_batches:
                break
            data = data.to(device)
            y_hat, _pi = model(data)
            ai_abs = DualHead.absolute_agent_indices(data, y_hat.device)
            bs = int(ai_abs.numel())
            seq_ids = data["seq_id"].view(-1).long().cpu().tolist()

            y_true_last = data.y[ai_abs, -1, :2]
            y_last = y_hat[:, ai_abs, -1, :]
            fde_modes = torch.norm(y_last[..., :2] - y_true_last.unsqueeze(0), p=2, dim=-1)
            best = fde_modes.argmin(dim=0)
            b_idx = torch.arange(bs, device=y_hat.device, dtype=torch.long)
            fde_min = fde_modes[best, b_idx]

            y_best = y_last[best, b_idx]
            u_epi_xy = nig_epistemic_uncertainty(y_best)
            u_epi = combine_xy_uncertainty(u_epi_xy, reduction="l2")

            fde_np = fde_min.detach().float().cpu().numpy()
            epi_np = u_epi.detach().float().cpu().numpy()
            fde_list.extend(fde_np.tolist())
            epi_list.extend(epi_np.tolist())

            for b in range(bs):
                vi = val_offset + b
                fd = float(fde_np[b])
                ep = float(epi_np[b])
                outlier = (fd > fde_th) or (ep > epi_th)
                per_sample.append(
                    {
                        "val_index": vi,
                        "seq_id": int(seq_ids[b]),
                        "fde": fd,
                        "epistemic_l2": ep,
                        "is_outlier": outlier,
                        "outlier_reason": (
                            []
                            if not outlier
                            else [
                                r
                                for r, cond in (
                                    ("fde_gt_threshold", fd > fde_th),
                                    ("epistemic_gt_threshold", ep > epi_th),
                                )
                                if cond
                            ]
                        ),
                    }
                )
            val_offset += bs

    fde = np.asarray(fde_list, dtype=np.float64)
    epi = np.asarray(epi_list, dtype=np.float64)
    n = int(fde.size)

    r_p, p_p = pearsonr(fde, epi)
    r_s, p_s = spearmanr(fde, epi)

    outlier_rows = [r for r in per_sample if r["is_outlier"]]
    outlier_indices = [r["val_index"] for r in outlier_rows]
    val_idx_fde_gt = [r["val_index"] for r in per_sample if r["fde"] > fde_th]
    val_idx_epi_gt = [r["val_index"] for r in per_sample if r["epistemic_l2"] > epi_th]

    stats = {
        "n_agents": n,
        "checkpoint": str(ckpt_path),
        "dataset_root": str(root),
        "pearson_r": float(r_p),
        "pearson_p": float(p_p),
        "spearman_rho": float(r_s),
        "spearman_p": float(p_s),
        "fde_mean": float(fde.mean()),
        "fde_std": float(fde.std()),
        "epistemic_l2_mean": float(epi.mean()),
        "epistemic_l2_std": float(epi.std()),
        "epistemic_l2_min": float(epi.min()),
        "epistemic_l2_max": float(epi.max()),
        "fde_outlier_threshold": fde_th,
        "epistemic_outlier_threshold": epi_th,
        "n_outliers": len(outlier_indices),
        "outlier_val_indices": outlier_indices,
        "n_fde_gt_threshold": len(val_idx_fde_gt),
        "val_indices_fde_gt_threshold": val_idx_fde_gt,
        "n_epistemic_gt_threshold": len(val_idx_epi_gt),
        "val_indices_epistemic_gt_threshold": val_idx_epi_gt,
        "note": "Per focal agent: FDE = min over modes of L2 error at final step; "
        "epistemic = L2(u_x, u_y) at same (best mode, final step) from beta/(nu*(alpha-1)).",
    }
    with open(out_dir / "correlation_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    with open(out_dir / "per_sample_metrics.json", "w", encoding="utf-8") as f:
        json.dump(per_sample, f, indent=2, ensure_ascii=False)

    outliers_payload = {
        "thresholds": {"fde_max_inlier": fde_th, "epistemic_max_inlier": epi_th},
        "rule": "outlier if fde > fde_max_inlier OR epistemic_l2 > epistemic_max_inlier",
        "val_indices": outlier_indices,
        "val_indices_fde_gt_threshold": val_idx_fde_gt,
        "val_indices_epistemic_gt_threshold": val_idx_epi_gt,
        "samples": outlier_rows,
    }
    with open(out_dir / "outliers.json", "w", encoding="utf-8") as f:
        json.dump(outliers_payload, f, indent=2, ensure_ascii=False)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    _scatter_fde_vs_epistemic(ax, fde, epi, fde_th, epi_th)
    ax.set_title(
        f"FDE vs Epistemic Uncertainty (n={n})\n"
        f"Pearson r={r_p:.4f} (p={p_p:.2e})  |  Spearman ρ={r_s:.4f} (p={p_s:.2e})"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "fde_vs_epistemic_scatter.png", dpi=160)
    plt.close(fig)

    xmx = float(args.scatter_epi_xmax_zoom)
    fig_z, ax_z = plt.subplots(figsize=(7.5, 6.5))
    _scatter_fde_vs_epistemic(ax_z, fde, epi, fde_th, epi_th, xlim=(0.0, xmx))
    ax_z.set_title(
        f"FDE vs Epistemic — x in [0, {xmx:g}] (zoom)\n"
        f"Pearson r={r_p:.4f} (p={p_p:.2e})  |  Spearman ρ={r_s:.4f} (p={p_s:.2e})"
    )
    fig_z.tight_layout()
    zoom_name = f"fde_vs_epistemic_scatter_epi_zoom_0_{int(xmx)}.png"
    fig_z.savefig(out_dir / zoom_name, dpi=160)
    plt.close(fig_z)

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nOutlier val_index count: {len(outlier_indices)}")
    print(f"Outlier val_indices: {outlier_indices}")
    print(f"\nSaved: {out_dir / 'fde_vs_epistemic_scatter.png'}")
    print(f"        {out_dir / zoom_name}")
    print(f"        {out_dir / 'correlation_stats.json'}")
    print(f"        {out_dir / 'per_sample_metrics.json'}")
    print(f"        {out_dir / 'outliers.json'}")

    if not args.no_scenario_viz:
        run_scenario_visualization(
            model=model,
            val_ds=val_ds,
            root=root,
            scenario_dir=out_dir / "scenario_viz",
            per_sample=per_sample,
            fde_th=fde_th,
            epi_th=epi_th,
            n_worst_fde=args.n_worst_fde,
            n_worst_epistemic=args.n_worst_epistemic,
            n_best_fde=args.n_best_fde,
            device=device,
        )


if __name__ == "__main__":
    main()
