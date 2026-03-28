# Argoverse scene visualization (project root). Used by eval_viz.py; model code lives in ./models/HiVT/.
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Same palette as the forecast agent (ego) trajectories.
COL_PAST = "#f9a825"
COL_GT_FUT = "#c62828"
COL_PRED = "#14532d"

# ---------------------------------------------------------------------------
# Coordinate transforms (HiVT AV local frame)
# ---------------------------------------------------------------------------


def world_xy_to_local(
    world_xy: np.ndarray,
    origin: np.ndarray,
    theta: Union[float, np.ndarray],
) -> np.ndarray:
    """Map world XY to AV-centered local frame used in HiVT."""
    xy = np.asarray(world_xy, dtype=np.float64)
    if xy.ndim == 1:
        xy = xy.reshape(1, -1)
    o = np.asarray(origin, dtype=np.float64).reshape(1, 2)
    t = float(theta)
    c, s = np.cos(t), np.sin(t)
    r = np.array([[c, -s], [s, c]], dtype=np.float64)
    return (xy[:, :2] - o) @ r


def local_xy_to_world(
    local_xy: np.ndarray,
    origin: np.ndarray,
    theta: float,
) -> np.ndarray:
    """Inverse of world_xy_to_local: world = origin + local @ R^T."""
    loc = np.asarray(local_xy, dtype=np.float64)
    if loc.ndim == 1:
        loc = loc.reshape(1, -1)
    o = np.asarray(origin, dtype=np.float64).reshape(1, 2)
    t = float(theta)
    c, s = np.cos(t), np.sin(t)
    r = np.array([[c, -s], [s, c]], dtype=np.float64)
    return loc[:, :2] @ r.T + o


def av_local_to_agent_up(
    local_xy: np.ndarray,
    p_agent: np.ndarray,
    psi: float,
) -> np.ndarray:
    """
    Translate so the target agent at t=19 is (0,0), then rotate so the agent's
    heading points along +Y (matplotlib "up").
    """
    pts = np.asarray(local_xy, dtype=np.float64).reshape(-1, 2)
    rel = pts - np.asarray(p_agent, dtype=np.float64).reshape(1, 2)
    ang = np.pi / 2.0 - float(psi)
    ca, sa = np.cos(ang), np.sin(ang)
    r = np.array([[ca, -sa], [sa, ca]], dtype=np.float64)
    return rel @ r.T


def hivt_heading_xy_to_plot_up(xy: np.ndarray) -> np.ndarray:
    """
    Heading-aligned frame (+X forward, ``rotate_mat``) → plot (+Y up): CCW 90°,
    ``(x, y) -> (-y, x)``.
    """
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    return np.stack([-pts[:, 1], pts[:, 0]], axis=1)


def av_local_rel_to_hivt_plot(rel: np.ndarray, psi: float) -> np.ndarray:
    """
    Relative points in AV-centered local frame (origin at agent t=19) → same plot frame
    as GT/pred: ``rel @ rotate_mat(psi)`` then ``hivt_heading_xy_to_plot_up`` (matches
    ``torch.bmm(data.y, rotate_mat)`` convention for row vectors).
    """
    rel = np.asarray(rel, dtype=np.float64).reshape(-1, 2)
    c, s = np.cos(psi), np.sin(psi)
    r = np.array([[c, -s], [s, c]], dtype=np.float64)
    return hivt_heading_xy_to_plot_up(rel @ r)


def rotated_offset_to_av_local(y_rot: np.ndarray, psi: float) -> np.ndarray:
    """Invert HiVT ``y_rot = y @ R`` per node: ``y = y_rot @ R.T`` (meters)."""
    y = np.asarray(y_rot, dtype=np.float64).reshape(-1, 2)
    c, s = np.cos(psi), np.sin(psi)
    r = np.array([[c, -s], [s, c]], dtype=np.float64)
    return y @ r.T


def _node_heading_psi(
    positions: np.ndarray,
    node_idx: int,
    rotate_angles: torch.Tensor,
) -> float:
    psi = float(rotate_angles[node_idx].detach().cpu().item())
    if np.isfinite(psi):
        return psi
    p = positions[node_idx]
    if p.shape[0] >= 2:
        d = p[19] - p[18]
        return float(np.arctan2(d[1], d[0]))
    return 0.0


def _agent_heading_psi(
    positions: np.ndarray,
    agent_idx: int,
    rotate_angles: torch.Tensor,
) -> float:
    return _node_heading_psi(positions, agent_idx, rotate_angles)


def _in_agent_window(
    xy: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> bool:
    """True if any point lies inside the plot axis box (agent-up frame, meters)."""
    if xy.size == 0:
        return False
    xmin, xmax = xlim
    ymin, ymax = ylim
    m = (
        (xy[:, 0] >= xmin - 1e-6)
        & (xy[:, 0] <= xmax + 1e-6)
        & (xy[:, 1] >= ymin - 1e-6)
        & (xy[:, 1] <= ymax + 1e-6)
    )
    return bool(np.any(m))


def _point_inside_plot(xy: np.ndarray, plot_xlim: tuple[float, float], plot_ylim: tuple[float, float]) -> bool:
    xmin, xmax = plot_xlim
    ymin, ymax = plot_ylim
    x, y = float(xy[0]), float(xy[1])
    return (xmin <= x <= xmax) and (ymin <= y <= ymax)


# Forecasting AGENT is always a vehicle-class actor; do not draw model preds for these as context.
_OTHER_ACTOR_SKIP_PRED_TYPES = frozenset({"PEDESTRIAN", "BICYCLE", "MOTORCYCLE", "OTHERS"})


def _skip_prediction_for_other_actor(actor_labels: Optional[List[str]], node_idx: int) -> bool:
    if actor_labels is None or node_idx >= len(actor_labels):
        return False
    return str(actor_labels[node_idx]).upper().strip() in _OTHER_ACTOR_SKIP_PRED_TYPES


def load_actor_object_labels(dataset_root: str, split: str, seq_id: int) -> Optional[List[str]]:
    """Match HiVT preprocessing: ``historical_df['TRACK_ID'].unique()`` order; raw ``OBJECT_TYPE`` strings."""
    p = Path(dataset_root) / split / "data" / f"{seq_id}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    ts = np.sort(df["TIMESTAMP"].unique())
    historical_timestamps = ts[:20]
    historical_df = df[df["TIMESTAMP"].isin(historical_timestamps)]
    actor_ids = list(historical_df["TRACK_ID"].unique())
    out: List[str] = []
    for aid in actor_ids:
        ot = df[df["TRACK_ID"] == aid]["OBJECT_TYPE"].iloc[0]
        out.append(str(ot).strip())
    return out


def _annotate_actor_class(
    ax: plt.Axes,
    xy_path: np.ndarray,
    pos_now_xy: np.ndarray,
    text: str,
    plot_xlim: tuple[float, float],
    plot_ylim: tuple[float, float],
    zorder: float,
) -> None:
    """Place label at current position, or first in-window point on path, else clamp to axes."""
    target = np.asarray(pos_now_xy, dtype=np.float64).reshape(2)
    if _point_inside_plot(target, plot_xlim, plot_ylim):
        lx, ly = float(target[0]), float(target[1])
    else:
        lx = ly = None
        path = np.asarray(xy_path, dtype=np.float64).reshape(-1, 2)
        for i in range(path.shape[0]):
            if _point_inside_plot(path[i], plot_xlim, plot_ylim):
                lx, ly = float(path[i, 0]), float(path[i, 1])
                break
        if lx is None:
            xmin, xmax = plot_xlim
            ymin, ymax = plot_ylim
            lx = float(np.clip(target[0], xmin, xmax))
            ly = float(np.clip(target[1], ymin, ymax))
    ax.annotate(
        text,
        xy=(lx, ly),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=6.2,
        color="#212121",
        zorder=zorder,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#bdbdbd", linewidth=0.4, alpha=0.9),
    )


def _plot_polyline_arrow(
    ax: plt.Axes,
    xs: np.ndarray,
    ys: np.ndarray,
    color: str,
    linestyle: str,
    linewidth: float,
    alpha: float,
    zorder: int,
    label: Optional[str] = None,
) -> None:
    ax.plot(
        xs,
        ys,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
        label=label,
    )
    if len(xs) >= 2:
        ax.annotate(
            "",
            xy=(float(xs[-1]), float(ys[-1])),
            xytext=(float(xs[-2]), float(ys[-2])),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=linewidth,
                alpha=alpha,
                linestyle=linestyle,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=zorder + 1,
        )


# ---------------------------------------------------------------------------
# Argoverse map (world → AV local → agent-up)
# ---------------------------------------------------------------------------


def _plot_driveable_agent_up(
    ax: plt.Axes,
    am: Any,
    city: str,
    origin: np.ndarray,
    theta: float,
    world_center_xy: np.ndarray,
    half_m: float,
    to_agent_up: Callable[[np.ndarray], np.ndarray],
) -> None:
    wx, wy = float(world_center_xy[0]), float(world_center_xy[1])
    bbox = (wx - half_m, wx + half_m, wy - half_m, wy + half_m)
    try:
        polys = am.find_local_driveable_areas(bbox, city)
    except Exception:
        return
    for poly in polys:
        arr = np.asarray(poly)
        if arr.size < 6:
            continue
        xy = arr[:, :2] if arr.shape[1] >= 2 else arr.reshape(-1, 2)
        loc = world_xy_to_local(xy, origin, theta)
        up = to_agent_up(loc)
        ax.plot(up[:, 0], up[:, 1], color="#9e9e9e", linewidth=0.55, alpha=0.5, zorder=0)


def _plot_lanes_agent_up(
    ax: plt.Axes,
    am: Any,
    city: str,
    origin: np.ndarray,
    theta: float,
    world_center_xy: np.ndarray,
    half_m: float,
    to_agent_up: Callable[[np.ndarray], np.ndarray],
) -> None:
    wx, wy = float(world_center_xy[0]), float(world_center_xy[1])
    manhattan_range = float(half_m * np.sqrt(2.0)) + 3.0
    try:
        lines = am.find_local_lane_centerlines(wx, wy, city, manhattan_range)
    except Exception:
        return
    if lines is None or len(lines) == 0:
        return
    for cl in lines:
        arr = np.asarray(cl)
        if arr.size == 0:
            continue
        xy = arr[:, :2]
        loc = world_xy_to_local(xy, origin, theta)
        up = to_agent_up(loc)
        ax.plot(up[:, 0], up[:, 1], color="#757575", linewidth=0.85, alpha=0.8, zorder=1)


def plot_hivt_scene(
    am: Any,
    data: Any,
    y_hat: torch.Tensor,
    out_path: Union[str, Path],
    title: Optional[str] = None,
    pi: Optional[torch.Tensor] = None,
    dataset_root: Optional[str] = None,
    split: str = "val",
) -> None:
    """
    Coordinate frames (val / train with labels):
      - **Past:** ``positions[agent, :20] - p_19`` in AV-local, then
        ``av_local_rel_to_hivt_plot`` (same rotate + 90° as GT/pred).
      - **GT future & pred:** HiVT ``y`` / ``y_hat`` are **offsets from t=19** at each future
        step (``x[:,20:] - x[:,19]`` in the dataset), **not** consecutive deltas — do **not**
        ``cumsum``. Prepend ``(0,0)``, then ``hivt_heading_xy_to_plot_up`` — same as minADE/minFDE.
      - Map: AV-local ``positions`` + ``to_agent_up`` (map view).
      - **Other actors:** same colors as ego — past ``COL_PAST``, GT future ``COL_GT_FUT``,
        pred ``COL_PRED`` (skipped for ``PEDESTRIAN`` / ``BICYCLE`` / ``MOTORCYCLE`` / ``OTHERS``);
        hero-relative transform. ``dataset_root`` + ``split`` load raw CSV ``OBJECT_TYPE`` strings.
        Label at t=19 or first in-window path point.

    View: X in [-50, 50] m; Y in [-30, 70] m. All figure text is English.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seq_id = int(data["seq_id"].view(-1)[0].item())
    actor_labels: Optional[List[str]] = None
    if dataset_root:
        actor_labels = load_actor_object_labels(dataset_root, split, seq_id)

    agent_idx = int(data["agent_index"].view(-1)[0].item())
    city = data["city"]
    if isinstance(city, (list, tuple)):
        city = city[0]
    if isinstance(city, bytes):
        city = city.decode()
    elif not isinstance(city, str):
        city = str(city)

    origin = data["origin"].detach().cpu().float().view(-1).numpy()[:2]
    theta = float(data["theta"].detach().cpu().item())
    rotate_angles = data["rotate_angles"]

    positions = data["positions"].detach().cpu().float().numpy()
    padding_mask = data["padding_mask"].detach().cpu().bool().numpy()
    num_nodes = positions.shape[0]

    p_agent = positions[agent_idx, 19].copy()
    psi = _agent_heading_psi(positions, agent_idx, rotate_angles)

    def to_agent_up(loc: np.ndarray) -> np.ndarray:
        return av_local_to_agent_up(loc, p_agent, psi)

    world_agent = local_xy_to_world(p_agent.reshape(1, 2), origin, theta).reshape(2)

    # Plot bounds (m, agent-up frame). Asymmetric Y: more forward (+Y) than back.
    extent_x_m = 50.0
    y_min_m, y_max_m = -30.0, 70.0
    plot_xlim = (-extent_x_m, extent_x_m)
    plot_ylim = (y_min_m, y_max_m)
    half_query = max(extent_x_m, abs(y_min_m), y_max_m) + 5.0

    y_hat = y_hat.detach().cpu().float()
    if y_hat.size(-1) >= 4:
        y_hat = y_hat[..., :2]
    # Per-timestep offset from t=19 (m), same as data.y — not incremental deltas.
    pred_offset = y_hat[:, agent_idx, :, :]
    num_modes = pred_offset.shape[0]
    if pi is not None:
        pi_row = pi[agent_idx].detach().float().view(-1)
        if pi_row.numel() != num_modes:
            pi_row = pi_row[:num_modes]
        prob = torch.softmax(pi_row, dim=0).numpy()
    else:
        prob = np.ones(num_modes, dtype=np.float64) / max(num_modes, 1)

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))

    _plot_driveable_agent_up(ax, am, city, origin, theta, world_agent, half_query, to_agent_up)
    _plot_lanes_agent_up(ax, am, city, origin, theta, world_agent, half_query, to_agent_up)

    if data.y is not None:
        rel_past = positions[agent_idx, :20] - p_agent.reshape(1, 2)
        past = av_local_rel_to_hivt_plot(rel_past, psi)
        y_gt = data.y[agent_idx, :, :2].detach().cpu().float().numpy()
        gt_traj = np.vstack([np.zeros((1, 2), dtype=np.float64), y_gt])
        gt_full = hivt_heading_xy_to_plot_up(gt_traj)
    else:
        past = to_agent_up(positions[agent_idx, :20])
        gt_full = to_agent_up(positions[agent_idx, 19:50])

    _plot_polyline_arrow(
        ax,
        past[:, 0],
        past[:, 1],
        color=COL_PAST,
        linestyle="-",
        linewidth=2.4,
        alpha=1.0,
        zorder=6,
        label="Past",
    )
    _plot_polyline_arrow(
        ax,
        gt_full[:, 0],
        gt_full[:, 1],
        color=COL_GT_FUT,
        linestyle="-",
        linewidth=2.2,
        alpha=1.0,
        zorder=7,
        label="GT future",
    )

    # Same green for all modes; alpha scales with softmax probability (darkest = highest p).
    # Draw low-probability modes first so the highest-p mode appears on top.
    pred_green = COL_PRED
    pmax = float(np.max(prob)) if prob.size else 1.0
    if pmax < 1e-9:
        pmax = 1.0
    draw_order = np.argsort(prob)
    for rank, m in enumerate(draw_order):
        m = int(m)
        off = pred_offset[m].numpy()
        traj_rel = np.vstack([np.zeros((1, 2), dtype=np.float64), off])
        pred_up = hivt_heading_xy_to_plot_up(traj_rel)
        alpha = float(0.12 + 0.88 * (prob[m] / pmax))
        z = 4 + rank
        _plot_polyline_arrow(
            ax,
            pred_up[:, 0],
            pred_up[:, 1],
            color=pred_green,
            linestyle="--",
            linewidth=1.85,
            alpha=alpha,
            zorder=z,
            label=f"Pred mode {m + 1} (p={prob[m]:.3f})",
        )

    # Other actors: same colors as ego (past / GT future / pred); hero-centered frame.
    z_other = 2.5
    lw_other_past, lw_other_gt, lw_other_pred = 1.75, 1.65, 1.45
    for n in range(num_nodes):
        if n == agent_idx:
            continue
        if bool(padding_mask[n, 19]):
            continue
        idx = np.where(~padding_mask[n])[0]
        if idx.size == 0:
            continue
        past_idx = idx[idx < 20]
        fut_idx = idx[idx >= 20]
        rel_gt = positions[n, idx] - p_agent.reshape(1, 2)
        traj_gt_other = av_local_rel_to_hivt_plot(rel_gt, psi)
        pos_now_plot = av_local_rel_to_hivt_plot(
            (positions[n, 19] - p_agent).reshape(1, 2), psi
        )[0]

        drew = False
        if past_idx.size > 0:
            traj_p = av_local_rel_to_hivt_plot(positions[n, past_idx] - p_agent.reshape(1, 2), psi)
            if _in_agent_window(traj_p, plot_xlim, plot_ylim):
                _plot_polyline_arrow(
                    ax,
                    traj_p[:, 0],
                    traj_p[:, 1],
                    color=COL_PAST,
                    linestyle="-",
                    linewidth=lw_other_past,
                    alpha=0.92,
                    zorder=z_other,
                    label=None,
                )
                drew = True
        if fut_idx.size > 0:
            traj_f = av_local_rel_to_hivt_plot(positions[n, fut_idx] - p_agent.reshape(1, 2), psi)
            if _in_agent_window(traj_f, plot_xlim, plot_ylim):
                _plot_polyline_arrow(
                    ax,
                    traj_f[:, 0],
                    traj_f[:, 1],
                    color=COL_GT_FUT,
                    linestyle="-",
                    linewidth=lw_other_gt,
                    alpha=0.92,
                    zorder=z_other + 0.05,
                    label=None,
                )
                drew = True

        fut_ok = ~padding_mask[n, 20:]
        psi_n = _node_heading_psi(positions, n, rotate_angles)
        if bool(fut_ok.any()) and pi is not None:
            pi_n = torch.softmax(pi[n].detach().float().view(-1), dim=0).numpy()
            if pi_n.size != num_modes:
                pi_n = pi_n[:num_modes]
        elif bool(fut_ok.any()):
            pi_n = np.ones(num_modes, dtype=np.float64) / max(num_modes, 1)
        else:
            pi_n = None

        if pi_n is not None and not _skip_prediction_for_other_actor(actor_labels, n):
            pmax_n = float(np.max(pi_n)) if pi_n.size else 1.0
            if pmax_n < 1e-9:
                pmax_n = 1.0
            for rank, m in enumerate(np.argsort(pi_n)):
                m = int(m)
                off_rot = y_hat[m, n, :, :].detach().cpu().numpy()
                off_av = rotated_offset_to_av_local(off_rot, psi_n)
                pts_h = [positions[n, 19] - p_agent]
                for k in range(off_av.shape[0]):
                    if bool(padding_mask[n, 20 + k]):
                        continue
                    pts_h.append(positions[n, 19] + off_av[k] - p_agent)
                pts_h_arr = np.asarray(pts_h, dtype=np.float64)
                if pts_h_arr.shape[0] < 2:
                    continue
                pred_other = av_local_rel_to_hivt_plot(pts_h_arr, psi)
                if not _in_agent_window(pred_other, plot_xlim, plot_ylim):
                    continue
                alpha = float(0.12 + 0.88 * (pi_n[m] / pmax_n))
                _plot_polyline_arrow(
                    ax,
                    pred_other[:, 0],
                    pred_other[:, 1],
                    color=COL_PRED,
                    linestyle="--",
                    linewidth=lw_other_pred,
                    alpha=alpha,
                    zorder=z_other + 0.3 + 0.01 * rank,
                    label=None,
                )
                drew = True

        if drew and actor_labels is not None and n < len(actor_labels):
            _annotate_actor_class(
                ax,
                traj_gt_other,
                pos_now_plot,
                actor_labels[n],
                plot_xlim,
                plot_ylim,
                zorder=9.5,
            )

    if actor_labels is not None and agent_idx < len(actor_labels):
        hero_path = np.vstack([past, gt_full])
        _annotate_actor_class(
            ax,
            hero_path,
            np.zeros(2, dtype=np.float64),
            actor_labels[agent_idx],
            plot_xlim,
            plot_ylim,
            zorder=10.5,
        )

    ax.set_xlim(*plot_xlim)
    ax.set_ylim(*plot_ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("Agent frame X (m)")
    ax.set_ylabel("Agent frame Y (m, forward)")
    if title:
        ax.set_title(title)
    ax.plot(0.0, 0.0, marker="o", color="#212121", markersize=5, zorder=10, label="Agent (now)")
    ax.legend(loc="upper left", fontsize=6.5, framealpha=0.92, ncol=1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
