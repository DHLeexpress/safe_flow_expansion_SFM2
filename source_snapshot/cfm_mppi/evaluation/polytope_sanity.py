"""Codex-only polytope sanity visualizer.

This file intentionally does not modify the existing polytope/verifier code.
It renders a cluttered UCY/SDD double-integrator rollout and overlays:

1. old baseline polytope.py corridor, orange dotted;
2. deterministic max-area local rectangle, blue;
3. trajectory-specific verifier rectangle witness, green.

Run from the repo root:
  python -m cfm_mppi.evaluation.polytope_sanity --dataset ucy --episode 110
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Polygon

from cfm_mppi.evaluation.eval_benchmark import BenchmarkPolicies, DEFAULTS, _dynamics_step
from cfm_mppi.evaluation.render_validation_comparison import (
    get_parser as _render_parser,
    _frame_obstacles,
    _frame_velocities,
    _make_scene,
)
from cfm_mppi.safegpc_adapter.polytope import build_nominal_polytope


@dataclass
class RectWitness:
    theta: float
    extents: np.ndarray  # [front, back, left, right]
    area: float
    min_level_slack: float
    feasible: bool
    reason: str


def _unit(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n > 1e-9:
        return (v / n).astype(np.float64)
    if fallback is None:
        fallback = np.array([1.0, 0.0], dtype=np.float64)
    return fallback.astype(np.float64)


def _rot(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _local(points: np.ndarray, anchor: np.ndarray, theta: float) -> np.ndarray:
    return (np.asarray(points, dtype=np.float64) - anchor[None]) @ _rot(theta)


def _rect_corners(anchor: np.ndarray, theta: float, ext: np.ndarray) -> np.ndarray:
    f, b, l, r = [float(x) for x in ext]
    local = np.array([[f, l], [f, -r], [-b, -r], [-b, l]], dtype=np.float64)
    return anchor[None] + local @ _rot(theta).T


def _rect_level_corners(anchor: np.ndarray, theta: float, ext: np.ndarray, q: float) -> np.ndarray:
    # H >= q shrinks every face toward the anchor by factor (1-q).
    scale = max(0.0, min(1.0, 1.0 - float(q)))
    return _rect_corners(anchor, theta, ext * scale)


def _rect_h(points: np.ndarray, anchor: np.ndarray, theta: float, ext: np.ndarray) -> np.ndarray:
    xy = _local(np.asarray(points, dtype=np.float64).reshape(-1, 2), anchor, theta)
    f, b, l, r = [max(float(x), 1e-9) for x in ext]
    vals = np.stack(
        [
            (f - xy[:, 0]) / f,
            (b + xy[:, 0]) / b,
            (l - xy[:, 1]) / l,
            (r + xy[:, 1]) / r,
        ],
        axis=1,
    )
    return vals.min(axis=1).reshape(np.asarray(points).shape[:-1])


def _ruler_levels(gamma: float, horizon: int) -> list[float]:
    levels = {0.0}
    for i in range(0, int(horizon) + 1):
        q = float((1.0 - gamma) ** i)
        # Avoid hundreds of visually identical near-zero contours for gamma=0.9.
        levels.add(round(q, 5))
    return sorted(x for x in levels if 0.0 <= x <= 1.0)


def _rect_h_grid(GX: np.ndarray, GY: np.ndarray, anchor: np.ndarray, witness: RectWitness) -> np.ndarray:
    grid = np.stack([GX.ravel(), GY.ravel()], axis=1)
    return _rect_h(grid, anchor, witness.theta, witness.extents).reshape(GX.shape)


def _poly_h(points: np.ndarray, A: np.ndarray, b: np.ndarray, ref: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    margin_ref = np.maximum(b - A @ ref, 1e-6)
    vals = (b[None, :] - pts @ A.T) / margin_ref[None, :]
    return vals.min(axis=1).reshape(np.asarray(points).shape[:-1])


def _poly_h_grid(GX: np.ndarray, GY: np.ndarray, A: np.ndarray, b: np.ndarray, ref: np.ndarray) -> np.ndarray:
    grid = np.stack([GX.ravel(), GY.ravel()], axis=1)
    return _poly_h(grid, A, b, ref).reshape(GX.shape)


def _rect_inside_circle(anchor: np.ndarray, theta: float, ext: np.ndarray, radius: float) -> bool:
    corners = _rect_corners(anchor, theta, ext)
    return bool(np.linalg.norm(corners - anchor[None], axis=1).max() <= radius + 1e-8)


def _rect_obstacle_free(
    anchor: np.ndarray,
    theta: float,
    ext: np.ndarray,
    obstacles: np.ndarray,
    safety_margin: float,
    robot_radius: float,
) -> bool:
    if obstacles.size == 0:
        return True
    centers = obstacles[:, :2].astype(np.float64)
    radii = obstacles[:, 2].astype(np.float64) + float(safety_margin) + float(robot_radius)
    cc = _local(centers, anchor, theta)
    f, b, l, r = [float(x) for x in ext]
    closest_x = np.clip(cc[:, 0], -b, f)
    closest_y = np.clip(cc[:, 1], -r, l)
    d = np.linalg.norm(cc - np.stack([closest_x, closest_y], axis=1), axis=1)
    return bool(np.all(d >= radii - 1e-8))


def _candidate_values(lo: float, hi: float, n: int) -> np.ndarray:
    lo = max(float(lo), 1e-4)
    hi = max(float(hi), lo)
    base = np.linspace(lo, hi, max(2, int(n)))
    return np.unique(np.clip(np.r_[lo, base, hi], lo, hi))


def _ray_upper_bounds(
    anchor: np.ndarray,
    theta: float,
    obstacles: np.ndarray,
    sensing_radius: float,
    safety_margin: float,
    robot_radius: float,
) -> np.ndarray:
    ub = np.full(4, float(sensing_radius), dtype=np.float64)
    if obstacles.size == 0:
        return ub
    centers = obstacles[:, :2].astype(np.float64)
    radii = obstacles[:, 2].astype(np.float64) + float(safety_margin) + float(robot_radius)
    cc = _local(centers, anchor, theta)
    for (u, v), rr in zip(cc, radii):
        rr = float(rr)
        if u > 0.0 and abs(v) < rr:
            ub[0] = min(ub[0], max(1e-3, u - math.sqrt(max(rr * rr - v * v, 0.0))))
        if u < 0.0 and abs(v) < rr:
            ub[1] = min(ub[1], max(1e-3, -u - math.sqrt(max(rr * rr - v * v, 0.0))))
        if v > 0.0 and abs(u) < rr:
            ub[2] = min(ub[2], max(1e-3, v - math.sqrt(max(rr * rr - u * u, 0.0))))
        if v < 0.0 and abs(u) < rr:
            ub[3] = min(ub[3], max(1e-3, -v - math.sqrt(max(rr * rr - u * u, 0.0))))
    return ub


def _fit_rectangle_for_theta(
    anchor: np.ndarray,
    obstacles: np.ndarray,
    theta: float,
    sensing_radius: float,
    safety_margin: float,
    robot_radius: float,
    min_ext: np.ndarray | None,
    grid: int,
    circle_mode: str,
) -> RectWitness:
    if circle_mode not in {"inside", "axis"}:
        raise ValueError(f"unknown circle_mode={circle_mode}")
    lo = np.asarray(min_ext if min_ext is not None else np.full(4, 0.05), dtype=np.float64)
    lo = np.maximum(lo, 0.05)
    ub = _ray_upper_bounds(anchor, theta, obstacles, sensing_radius, safety_margin, robot_radius)
    if np.any(lo > ub + 1e-9):
        return RectWitness(theta, lo, 0.0, -np.inf, False, "required_extents_exceed_ray_bounds")

    vals = [_candidate_values(lo[i], ub[i], grid) for i in range(4)]
    best: Optional[RectWitness] = None
    for f in vals[0]:
        for b in vals[1]:
            # In "inside" mode the whole rectangle must fit in the sensing circle.
            # In "axis" mode the circle only gates visible obstacles/prefix states.
            if circle_mode == "inside":
                if f * f + min(vals[2][0], vals[3][0]) ** 2 > sensing_radius * sensing_radius + 1e-8:
                    continue
                if b * b + min(vals[2][0], vals[3][0]) ** 2 > sensing_radius * sensing_radius + 1e-8:
                    continue
            for l in vals[2]:
                if circle_mode == "inside":
                    if f * f + l * l > sensing_radius * sensing_radius + 1e-8:
                        continue
                    if b * b + l * l > sensing_radius * sensing_radius + 1e-8:
                        continue
                for r in vals[3]:
                    ext = np.array([f, b, l, r], dtype=np.float64)
                    if circle_mode == "inside" and not _rect_inside_circle(anchor, theta, ext, sensing_radius):
                        continue
                    if not _rect_obstacle_free(anchor, theta, ext, obstacles, safety_margin, robot_radius):
                        continue
                    area = float((f + b) * (l + r))
                    if best is None or area > best.area:
                        best = RectWitness(theta, ext, area, 0.0, True, "ok")
    if best is not None:
        return best
    return RectWitness(theta, lo, 0.0, -np.inf, False, "no_obstacle_free_rectangle_on_grid")


def _theta_grid(state: np.ndarray, goal: np.ndarray, prev_theta: float | None, n: int) -> np.ndarray:
    seeds: list[float] = []
    vel = state[2:4] if state.shape[0] >= 4 else np.zeros(2, dtype=np.float64)
    for vec in (vel, goal[:2] - state[:2]):
        if np.linalg.norm(vec) > 1e-6:
            seeds.append(math.atan2(float(vec[1]), float(vec[0])))
    if prev_theta is not None and math.isfinite(prev_theta):
        seeds.append(float(prev_theta))
    coarse = np.linspace(-math.pi, math.pi, int(n), endpoint=False)
    fine = []
    for s in seeds:
        fine.extend([s + d for d in np.linspace(-0.35, 0.35, 9)])
    out = np.r_[coarse, np.asarray(fine, dtype=np.float64)]
    return ((out + math.pi) % (2.0 * math.pi)) - math.pi


def fit_deterministic_rectangle(
    state: np.ndarray,
    goal: np.ndarray,
    obstacles: np.ndarray,
    *,
    sensing_radius: float,
    safety_margin: float,
    robot_radius: float,
    prev_theta: float | None,
    n_angles: int,
    grid: int,
    circle_mode: str,
) -> RectWitness:
    anchor = state[:2].astype(np.float64)
    best: Optional[RectWitness] = None
    for theta in _theta_grid(state, goal, prev_theta, n_angles):
        cand = _fit_rectangle_for_theta(
            anchor, obstacles, float(theta), sensing_radius, safety_margin, robot_radius, None, grid, circle_mode
        )
        if cand.feasible and (best is None or cand.area > best.area):
            best = cand
    if best is not None:
        return best
    return RectWitness(0.0, np.full(4, 0.05), 0.0, -np.inf, False, "no_deterministic_rectangle")


def _required_extents_for_ruler(
    state: np.ndarray,
    future_xy: np.ndarray,
    theta: float,
    gamma: float,
    min_denom: float = 1e-4,
) -> tuple[np.ndarray, float]:
    anchor = state[:2].astype(np.float64)
    loc = _local(future_xy, anchor, theta)
    req = np.full(4, 0.05, dtype=np.float64)
    min_raw = np.inf
    for i, (x, y) in enumerate(loc):
        q = float((1.0 - gamma) ** i)
        denom = max(1.0 - q, min_denom)
        if x > 0.0:
            req[0] = max(req[0], x / denom)
        else:
            req[1] = max(req[1], -x / denom)
        if y > 0.0:
            req[2] = max(req[2], y / denom)
        else:
            req[3] = max(req[3], -y / denom)
        min_raw = min(min_raw, denom)
    return req, min_raw


def fit_verifier_rectangle(
    state: np.ndarray,
    goal: np.ndarray,
    obstacles: np.ndarray,
    future_xy: np.ndarray,
    *,
    gamma: float,
    sensing_radius: float,
    safety_margin: float,
    robot_radius: float,
    prev_theta: float | None,
    n_angles: int,
    grid: int,
    circle_mode: str,
) -> RectWitness:
    anchor = state[:2].astype(np.float64)
    best: Optional[RectWitness] = None
    for theta in _theta_grid(state, goal, prev_theta, n_angles):
        req, _ = _required_extents_for_ruler(state, future_xy, float(theta), gamma)
        cand = _fit_rectangle_for_theta(
            anchor, obstacles, float(theta), sensing_radius, safety_margin, robot_radius, req, grid, circle_mode
        )
        if not cand.feasible:
            continue
        h = _rect_h(future_xy, anchor, float(theta), cand.extents)
        q = np.asarray([(1.0 - gamma) ** i for i in range(len(future_xy))], dtype=np.float64)
        slack = float(np.min(h - q))
        cand.min_level_slack = slack
        if slack < -1e-6:
            continue
        # Prefer high slack, then high area. This is a verifier witness, not the area optimum.
        score = slack + 0.002 * math.log(max(cand.area, 1e-9))
        best_score = (
            best.min_level_slack + 0.002 * math.log(max(best.area, 1e-9))
            if best is not None
            else -np.inf
        )
        if best is None or score > best_score:
            best = cand
    if best is not None:
        return best
    return RectWitness(0.0, np.full(4, 0.05), 0.0, -np.inf, False, "no_rectangle_satisfies_ruler")


def _clip_polygon(poly: np.ndarray, a: np.ndarray, b: float) -> np.ndarray:
    if len(poly) == 0:
        return poly
    out = []
    prev = poly[-1]
    prev_in = float(a @ prev) <= b + 1e-9
    for cur in poly:
        cur_in = float(a @ cur) <= b + 1e-9
        if cur_in != prev_in:
            denom = float(a @ (cur - prev))
            if abs(denom) > 1e-12:
                t = (b - float(a @ prev)) / denom
                out.append(prev + t * (cur - prev))
        if cur_in:
            out.append(cur)
        prev, prev_in = cur, cur_in
    return np.asarray(out, dtype=np.float64)


def _polytope_polygon(A: np.ndarray, b: np.ndarray, limits: tuple[float, float, float, float]) -> np.ndarray:
    xmin, xmax, ymin, ymax = limits
    pad = 2.0
    poly = np.array(
        [[xmin - pad, ymin - pad], [xmax + pad, ymin - pad], [xmax + pad, ymax + pad], [xmin - pad, ymax + pad]],
        dtype=np.float64,
    )
    for ai, bi in zip(A, b):
        poly = _clip_polygon(poly, ai.astype(np.float64), float(bi))
        if len(poly) == 0:
            break
    return poly


def rollout_policy(args: argparse.Namespace):
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    base = _render_parser().parse_args([])
    base.dataset = args.dataset
    base.dynamics = "doubleintegrator"
    base.episode = args.episode
    base.steps = args.steps
    base.seed = args.seed
    base.pedestrian_source = args.pedestrian_source
    base.pedestrian_radius = args.pedestrian_radius
    base.smoke = bool(args.smoke)
    base.mizuta_checkpoint = args.mizuta_checkpoint
    base.safe_cfm_checkpoint = args.safe_cfm_checkpoint
    base.drifting_checkpoint = args.drifting_checkpoint
    base.safemppi_horizon = args.safemppi_horizon
    base.safemppi_samples = args.safemppi_samples
    base.debug_rollouts = 0
    base.collect_rollouts = False
    base.safemppi_running_goal_weight = 0.25
    base.safemppi_terminal_goal_weight = 80.0
    base.safemppi_control_weight = 0.03
    base.safemppi_smooth_weight = 0.12
    base.safemppi_soft_clearance_weight = 25.0
    base.safemppi_progress_weight = 2.0
    base.safemppi_use_sets_backup = False
    base.safemppi_sets_num_modes = 3
    base.safemppi_sets_branch_scale = 0.85
    base.safemppi_sets_include_cbf_backup = True
    base.safemppi_sets_cbf_push = 1.25
    base.safemppi_sets_reverse_speed = 0.75
    base.safemppi_sets_turn_rate = 1.4
    base.safe_cfm_num_candidates = 16

    state0, goal, obstacles_seq, velocities_seq, label = _make_scene(base)
    policies = BenchmarkPolicies(base, torch.device(args.device))
    if args.method in ("mizuta_cfm_mppi", "mizuta_safe"):
        policies._mizuta_episode = None
    state = state0.copy()
    states = [state.copy()]
    controls: list[np.ndarray] = []
    infos: list[dict] = []
    for step in range(int(args.steps)):
        obstacles = _frame_obstacles(obstacles_seq, step)
        velocities = _frame_velocities(velocities_seq, step)
        action, info = policies.action(
            args.method,
            state,
            goal,
            obstacles,
            controls,
            "doubleintegrator",
            float(args.gamma),
            int(args.steps),
            obstacle_velocities=velocities,
        )
        state = _dynamics_step(state, action, "doubleintegrator", float(DEFAULTS["dt"]))
        states.append(state.copy())
        controls.append(np.asarray(action, dtype=np.float32))
        infos.append(info)
        if np.linalg.norm(state[:2] - goal[:2]) < args.stop_radius:
            break
    return (
        np.asarray(states, dtype=np.float32),
        np.asarray(controls, dtype=np.float32),
        obstacles_seq,
        velocities_seq,
        goal.astype(np.float32),
        label,
        infos,
    )


def _limits(states: np.ndarray, goal: np.ndarray, obstacles_seq: np.ndarray, sensing_radius: float):
    pts = [states[:, :2], goal[None]]
    if obstacles_seq.size:
        pts.append(obstacles_seq[:, :, :2].reshape(-1, 2))
    allp = np.concatenate(pts, axis=0)
    finite = np.isfinite(allp).all(axis=1)
    allp = allp[finite]
    lo = allp.min(axis=0) - sensing_radius * 0.45
    hi = allp.max(axis=0) + sensing_radius * 0.45
    return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])


def _verification_slice(states: np.ndarray, t: int, horizon: int, sensing_radius: float, mode: str) -> np.ndarray:
    """Return the local rollout prefix that the current sensing-ball polytope can certify.

    A single rectangle inside the current robot-centered sensing ball cannot certify
    states after the rollout has moved outside that ball. For MPC, that is not a
    geometric failure: we will re-sense and build a new rectangle at the next
    inference step. The default therefore certifies the in-ball prefix.
    """
    end = min(states.shape[0], int(t) + int(horizon) + 1)
    future = states[int(t):end, :2]
    if mode == "full":
        return future
    rel = np.linalg.norm(future - states[int(t), :2][None], axis=1)
    inside = rel <= float(sensing_radius) + 1e-8
    if mode == "prefix":
        if not bool(inside.all()):
            first_out = int(np.argmax(~inside))
            return future[: max(1, first_out)]
        return future
    if mode == "inball":
        keep = future[inside]
        return keep if len(keep) else future[:1]
    raise ValueError(f"unknown verify_sensing_mode={mode}")


def _draw_rect(ax, anchor, witness: RectWitness, color: str, label: str, lw: float, ls: str, zorder: int):
    if not witness.feasible:
        return
    corners = _rect_corners(anchor, witness.theta, witness.extents)
    ax.add_patch(Polygon(corners, fill=False, closed=True, edgecolor=color, lw=lw, ls=ls, alpha=0.92, zorder=zorder))
    p = corners[0]
    ax.text(p[0], p[1], label, color=color, fontsize=6.4, ha="left", va="bottom", zorder=zorder + 1)


def _draw_rect_levels(ax, anchor, witness: RectWitness, gamma: float, horizon: int, color: str, stride: int, zorder: int):
    if not witness.feasible:
        return
    for i in range(0, horizon + 1, max(1, stride)):
        q = float((1.0 - gamma) ** i)
        corners = _rect_level_corners(anchor, witness.theta, witness.extents, q)
        alpha = 0.12 + 0.28 * (1.0 - i / max(horizon, 1))
        ax.add_patch(
            Polygon(corners, fill=False, closed=True, edgecolor=color, lw=0.55, alpha=alpha, zorder=zorder)
        )


def _draw_rect_contours(
    ax,
    GX: np.ndarray,
    GY: np.ndarray,
    anchor: np.ndarray,
    witness: RectWitness,
    gamma: float,
    horizon: int,
    *,
    cmap: str,
    color: str,
    alpha: float,
    zorder: int,
):
    if not witness.feasible:
        return
    H = _rect_h_grid(GX, GY, anchor, witness)
    levels = _ruler_levels(gamma, horizon)
    levels = [v for v in levels if float(np.nanmin(H)) < v < float(np.nanmax(H))]
    if len(levels) < 2:
        return
    ax.contourf(GX, GY, H, levels=levels + [1.0001], cmap=cmap, alpha=alpha, zorder=zorder)
    ax.contour(GX, GY, H, levels=levels[1:], colors=color, linewidths=0.5, alpha=0.72, zorder=zorder + 1)
    if float(np.nanmin(H)) < 0.0 < float(np.nanmax(H)):
        ax.contour(GX, GY, H, levels=[0.0], colors=color, linewidths=1.35, alpha=0.95, zorder=zorder + 2)


def _draw_poly_contours(
    ax,
    GX: np.ndarray,
    GY: np.ndarray,
    poly_params: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    gamma: float,
    horizon: int,
    *,
    cmap: str,
    color: str,
    alpha: float,
    zorder: int,
):
    if poly_params is None:
        return
    A, b, ref = poly_params
    H = _poly_h_grid(GX, GY, A, b, ref)
    levels = _ruler_levels(gamma, horizon)
    levels = [v for v in levels if float(np.nanmin(H)) < v < float(np.nanmax(H))]
    if len(levels) < 2:
        return
    ax.contourf(GX, GY, H, levels=levels + [1.0001], cmap=cmap, alpha=alpha, zorder=zorder)
    ax.contour(GX, GY, H, levels=levels[1:], colors=color, linewidths=0.45, alpha=0.72, zorder=zorder + 1)
    if float(np.nanmin(H)) < 0.0 < float(np.nanmax(H)):
        ax.contour(GX, GY, H, levels=[0.0], colors=color, linewidths=1.2, alpha=0.95, zorder=zorder + 2)


def _draw_obstacles(ax, obstacles: np.ndarray, safety_margin: float, robot_radius: float):
    for o in obstacles:
        if not np.isfinite(o).all():
            continue
        x, y, r = [float(v) for v in o]
        ax.add_patch(Circle((x, y), r, facecolor="#7b3294", alpha=0.25, edgecolor="#4d004b", lw=0.9, zorder=4))
        ax.add_patch(
            Circle(
                (x, y),
                r + safety_margin + robot_radius,
                facecolor="none",
                edgecolor="#7b3294",
                ls="--",
                lw=0.65,
                alpha=0.55,
                zorder=4,
            )
        )


def render(args: argparse.Namespace) -> dict:
    states, controls, obstacles_seq, velocities_seq, goal, label, infos = rollout_policy(args)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = min(len(states), int(args.steps) + 1)
    limits = _limits(states, goal, obstacles_seq[:frames], float(args.sensing_radius))
    horizon = min(int(args.verify_horizon), frames - 1)
    gx = np.linspace(limits[0], limits[1], int(args.grid_x))
    gy = np.linspace(limits[2], limits[3], int(args.grid_y))
    GX, GY = np.meshgrid(gx, gy)
    rects_det: list[RectWitness] = []
    rects_ver: list[RectWitness] = []
    verify_lengths: list[int] = []
    old_polys: list[np.ndarray] = []
    old_poly_params: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None] = []
    prev_det_theta: float | None = None
    prev_ver_theta: float | None = None
    for t in range(frames):
        state = states[t]
        obstacles = _frame_obstacles(obstacles_seq, t).copy()
        finite = np.isfinite(obstacles).all(axis=1) if obstacles.size else np.zeros((0,), dtype=bool)
        obstacles = obstacles[finite] if obstacles.size else obstacles
        det = fit_deterministic_rectangle(
            state,
            goal,
            obstacles,
            sensing_radius=float(args.sensing_radius),
            safety_margin=float(args.safety_margin),
            robot_radius=float(args.robot_radius),
            prev_theta=prev_det_theta,
            n_angles=int(args.n_angles),
            grid=int(args.grid),
            circle_mode=str(args.circle_mode),
        )
        if det.feasible:
            prev_det_theta = det.theta
        future_xy = _verification_slice(
            states, t, horizon, float(args.sensing_radius), str(args.verify_sensing_mode)
        )
        verify_lengths.append(int(len(future_xy)))
        ver = fit_verifier_rectangle(
            state,
            goal,
            obstacles,
            future_xy,
            gamma=float(args.gamma),
            sensing_radius=float(args.sensing_radius),
            safety_margin=float(args.safety_margin),
            robot_radius=float(args.robot_radius),
            prev_theta=prev_ver_theta,
            n_angles=int(args.n_angles),
            grid=int(args.grid),
            circle_mode=str(args.circle_mode),
        )
        if ver.feasible:
            prev_ver_theta = ver.theta
        rects_det.append(det)
        rects_ver.append(ver)

        heading = state[2:4] if np.linalg.norm(state[2:4]) > 0.1 else goal[:2] - state[:2]
        safe_obstacles = obstacles.copy()
        if safe_obstacles.size:
            safe_obstacles[:, 2] = safe_obstacles[:, 2] + float(args.safety_margin) + float(args.robot_radius)
        try:
            poly = build_nominal_polytope(
                torch.tensor(state[:2], dtype=torch.float32),
                torch.tensor(heading, dtype=torch.float32),
                torch.tensor(safe_obstacles, dtype=torch.float32),
                sensing_range=float(args.old_sensing_range),
                back_range=float(args.old_back_range),
                half_width=float(args.old_half_width),
                max_obstacles=int(args.old_max_obstacles),
            )
            old_A = poly.A.numpy()
            old_b = poly.b.numpy()
            old_ref = poly.ref.numpy()
            old_polys.append(_polytope_polygon(old_A, old_b, limits))
            old_poly_params.append((old_A, old_b, old_ref))
        except Exception:
            old_polys.append(np.zeros((0, 2), dtype=np.float64))
            old_poly_params.append(None)

    det_cert = []
    ver_cert = []
    old_cert = []
    for t in range(frames):
        future_xy = _verification_slice(
            states, t, horizon, float(args.sensing_radius), str(args.verify_sensing_mode)
        )
        q = np.asarray([(1.0 - float(args.gamma)) ** i for i in range(len(future_xy))], dtype=np.float64)
        det_h = (
            _rect_h(future_xy, states[t, :2], rects_det[t].theta, rects_det[t].extents)
            if rects_det[t].feasible
            else np.full(len(future_xy), -np.inf)
        )
        ver_h = (
            _rect_h(future_xy, states[t, :2], rects_ver[t].theta, rects_ver[t].extents)
            if rects_ver[t].feasible
            else np.full(len(future_xy), -np.inf)
        )
        old_h = (
            _poly_h(future_xy, *old_poly_params[t])
            if old_poly_params[t] is not None
            else np.full(len(future_xy), -np.inf)
        )
        det_cert.append(bool(np.all(det_h >= q - 1e-6)))
        ver_cert.append(bool(np.all(ver_h >= q - 1e-6)))
        old_cert.append(bool(np.all(old_h >= q - 1e-6)))

    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    fig.patch.set_facecolor("white")

    def draw(frame: int):
        ax.clear()
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.14, lw=0.55)
        ax.tick_params(labelsize=7)
        obstacles = _frame_obstacles(obstacles_seq, frame)
        _draw_obstacles(ax, obstacles, float(args.safety_margin), float(args.robot_radius))
        ax.scatter([goal[0]], [goal[1]], marker="*", s=160, c="gold", edgecolor="k", zorder=10)
        ax.plot(states[:, 0], states[:, 1], color="#bbbbbb", lw=1.0, alpha=0.45, zorder=2)
        ax.plot(states[: frame + 1, 0], states[: frame + 1, 1], color="#d73027", lw=2.25, zorder=8)
        ax.scatter([states[frame, 0]], [states[frame, 1]], s=70, c="#1a9850", edgecolor="white", lw=0.8, zorder=12)

        old = old_polys[frame]
        if len(old) >= 3:
            ax.add_patch(
                Polygon(old, closed=True, fill=False, edgecolor="#d95f02", lw=1.2, ls=":", alpha=0.62, zorder=5)
            )

        anchor = states[frame, :2]
        det = rects_det[frame]
        ver = rects_ver[frame]
        if args.contour == "deterministic":
            _draw_rect_contours(
                ax, GX, GY, anchor, det, float(args.gamma), horizon,
                cmap="Blues", color="#2166ac", alpha=0.38, zorder=1
            )
        elif args.contour == "verifier":
            _draw_rect_contours(
                ax, GX, GY, anchor, ver, float(args.gamma), horizon,
                cmap="Greens", color="#1a9850", alpha=0.38, zorder=1
            )
        elif args.contour == "both":
            _draw_rect_contours(
                ax, GX, GY, anchor, det, float(args.gamma), horizon,
                cmap="Blues", color="#2166ac", alpha=0.25, zorder=1
            )
            _draw_rect_contours(
                ax, GX, GY, anchor, ver, float(args.gamma), horizon,
                cmap="Greens", color="#1a9850", alpha=0.25, zorder=2
            )
        elif args.contour == "old":
            _draw_poly_contours(
                ax, GX, GY, old_poly_params[frame], float(args.gamma), horizon,
                cmap="Oranges", color="#d95f02", alpha=0.28, zorder=1
            )
        elif args.contour == "all":
            _draw_poly_contours(
                ax, GX, GY, old_poly_params[frame], float(args.gamma), horizon,
                cmap="Oranges", color="#d95f02", alpha=0.20, zorder=1
            )
            _draw_rect_contours(
                ax, GX, GY, anchor, det, float(args.gamma), horizon,
                cmap="Blues", color="#2166ac", alpha=0.18, zorder=2
            )
            _draw_rect_contours(
                ax, GX, GY, anchor, ver, float(args.gamma), horizon,
                cmap="Greens", color="#1a9850", alpha=0.18, zorder=3
            )
        _draw_rect_levels(ax, anchor, det, float(args.gamma), horizon, "#2166ac", int(args.level_stride), 3)
        _draw_rect(ax, anchor, det, "#2166ac", "det max-rect", 1.75, "-", 6)
        _draw_rect_levels(ax, anchor, ver, float(args.gamma), horizon, "#1a9850", int(args.level_stride), 4)
        _draw_rect(ax, anchor, ver, "#1a9850", "verifier witness", 1.7, "--", 7)

        future = _verification_slice(
            states, frame, horizon, float(args.sensing_radius), str(args.verify_sensing_mode)
        )
        ax.plot(future[:, 0], future[:, 1], color="#31a354", lw=1.2, alpha=0.75, zorder=9)
        ax.scatter(future[:, 0], future[:, 1], s=10, color="#31a354", alpha=0.75, zorder=10)
        ax.add_patch(
            Circle(tuple(anchor), float(args.sensing_radius), fill=False, edgecolor="#525252", lw=0.75, alpha=0.28, zorder=1)
        )

        msg = (
            f"{label}\n"
            f"method={args.method}  step={frame}/{frames - 1}  gamma={args.gamma:.2f}\n"
            f"det_cert={det_cert[frame]} area={det.area:.2f}  "
            f"ver_cert={ver_cert[frame]} area={ver.area:.2f} slack={ver.min_level_slack:.3f}\n"
            f"old_cert={old_cert[frame]}  verify_prefix={verify_lengths[frame]} states  "
            f"contour={args.contour} circle={args.circle_mode}\n"
            f"old polytope.py = orange dotted; blue = deterministic; green = verifier"
        )
        ax.text(
            0.02,
            0.98,
            msg,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7.1,
            color="#222222",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.88},
            zorder=20,
        )
        return []

    fig.tight_layout()
    anim = FuncAnimation(fig, draw, frames=frames, interval=1000.0 / max(1, int(args.fps)))
    anim.save(out_path, writer=PillowWriter(fps=int(args.fps)), dpi=int(args.dpi))
    plt.close(fig)

    summary = {
        "output": str(out_path),
        "dataset": args.dataset,
        "episode": int(args.episode),
        "method": args.method,
        "label": label,
        "frames": int(frames),
        "gamma": float(args.gamma),
        "det_cert_rate": float(np.mean(det_cert)),
        "verifier_cert_rate": float(np.mean(ver_cert)),
        "old_polytope_cert_rate": float(np.mean(old_cert)),
        "det_mean_area": float(np.mean([r.area for r in rects_det if r.feasible])) if any(r.feasible for r in rects_det) else 0.0,
        "verifier_mean_area": float(np.mean([r.area for r in rects_ver if r.feasible])) if any(r.feasible for r in rects_ver) else 0.0,
        "verify_sensing_mode": str(args.verify_sensing_mode),
        "circle_mode": str(args.circle_mode),
        "verify_prefix_mean_len": float(np.mean(verify_lengths)) if verify_lengths else 0.0,
        "final_goal_distance": float(np.linalg.norm(states[-1, :2] - goal[:2])),
        "num_controls": int(len(controls)),
        "policy_infos": [
            {k: v for k, v in info.items() if isinstance(v, (str, int, float, bool)) or v is None}
            for info in infos[:3]
        ],
    }
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render Codex polytope_v1 sanity GIF without touching existing files.")
    p.add_argument("--dataset", default="ucy", choices=["ucy", "sdd", "sfm"])
    p.add_argument("--pedestrian-source", default="auto", choices=["auto", "validation", "sfm-social-force", "synthetic-static"])
    p.add_argument("--episode", type=int, default=110)
    p.add_argument("--steps", type=int, default=36)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--method", default="mizuta_cfm_mppi", choices=["mizuta_cfm_mppi", "mizuta_safe", "safemppi_gamma"])
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--mizuta-checkpoint", default="output_dir/cfm_transformer/checkpoint.pth")
    p.add_argument("--safe-cfm-checkpoint", default="output_dir/safe_contextual_cfm/checkpoint_best.pth")
    p.add_argument("--drifting-checkpoint", default="output_dir/drifting_generator/checkpoint_best.pth")
    p.add_argument("--safemppi-horizon", type=int, default=32)
    p.add_argument("--safemppi-samples", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--verify-horizon", type=int, default=14)
    p.add_argument("--verify-sensing-mode", choices=["prefix", "full", "inball"], default="prefix")
    p.add_argument(
        "--circle-mode",
        choices=["axis", "inside"],
        default="axis",
        help="inside: whole rectangle inside sensing circle; axis: circle gates prefix/obstacles, not rectangle corners",
    )
    p.add_argument("--sensing-radius", type=float, default=6.0)
    p.add_argument("--safety-margin", type=float, default=float(DEFAULTS["safety_margin"]))
    p.add_argument("--robot-radius", type=float, default=0.0)
    p.add_argument("--pedestrian-radius", type=float, default=0.0)
    p.add_argument("--n-angles", type=int, default=44)
    p.add_argument("--grid", type=int, default=5)
    p.add_argument("--level-stride", type=int, default=2)
    p.add_argument("--contour", choices=["verifier", "deterministic", "both", "old", "all", "none"], default="verifier")
    p.add_argument("--grid-x", type=int, default=115)
    p.add_argument("--grid-y", type=int, default=95)
    p.add_argument("--old-sensing-range", type=float, default=6.0)
    p.add_argument("--old-back-range", type=float, default=2.0)
    p.add_argument("--old-half-width", type=float, default=4.0)
    p.add_argument("--old-max-obstacles", type=int, default=8)
    p.add_argument("--stop-radius", type=float, default=0.35)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--dpi", type=int, default=105)
    p.add_argument("--output", default="results/codex_polytope_sanity/ucy110_polytope_sanity.gif")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary = render(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
