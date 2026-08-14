from __future__ import annotations

import argparse
import csv
import math
import pickle
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.patches import Circle

from cfm_mppi.evaluation.eval_benchmark import (
    DEFAULTS,
    BenchmarkPolicies,
    _dynamics_step,
    _make_episode,
    _set_seed,
)
from cfm_mppi.safegpc_adapter import resolve_gamma_schedule
from cfm_mppi.utils import HumanAgent

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None


@dataclass
class RenderRun:
    label: str
    method: str
    gamma: Optional[float]
    states: np.ndarray
    controls: np.ndarray
    metrics: Dict[str, object]
    debug: Optional[List[Dict[str, np.ndarray]]] = None


@dataclass
class RenderScenario:
    runs: List[RenderRun]
    state0: np.ndarray
    goal: np.ndarray
    obstacles_seq: np.ndarray
    velocities_seq: np.ndarray
    label: str


def _iter_progress(items, **kwargs):
    if tqdm is None:
        return items
    return tqdm(items, **kwargs)


def _policy_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        smoke=args.smoke,
        seed=args.seed,
        mizuta_checkpoint=args.mizuta_checkpoint,
        safe_cfm_checkpoint=args.safe_cfm_checkpoint,
        drifting_checkpoint=args.drifting_checkpoint,
        collect_rollouts=True,
        debug_rollouts=args.debug_rollouts,
        safemppi_horizon=args.safemppi_horizon,
        safemppi_samples=args.safemppi_samples,
        safemppi_running_goal_weight=args.safemppi_running_goal_weight,
        safemppi_terminal_goal_weight=args.safemppi_terminal_goal_weight,
        safemppi_control_weight=args.safemppi_control_weight,
        safemppi_smooth_weight=args.safemppi_smooth_weight,
        safemppi_soft_clearance_weight=args.safemppi_soft_clearance_weight,
        safemppi_progress_weight=args.safemppi_progress_weight,
        safemppi_use_sets_backup=args.safemppi_use_sets_backup,
        safemppi_sets_num_modes=args.safemppi_sets_num_modes,
        safemppi_sets_branch_scale=args.safemppi_sets_branch_scale,
        safemppi_sets_include_cbf_backup=args.safemppi_sets_include_cbf_backup,
        safemppi_sets_cbf_push=args.safemppi_sets_cbf_push,
        safemppi_sets_reverse_speed=args.safemppi_sets_reverse_speed,
        safemppi_sets_turn_rate=args.safemppi_sets_turn_rate,
        safe_cfm_num_candidates=args.safe_cfm_num_candidates,
    )


def _method_label(method: str, gamma: Optional[float]) -> str:
    if method == "mizuta_cfm_mppi":
        return "Mizuta CFM-MPPI"
    if method == "safemppi_gamma":
        return f"SafeMPPI gamma={gamma:.3g}"
    if method == "safe_cfm":
        return f"Safe CFM gamma={gamma:.3g}"
    if method == "guided_safemppi":
        return f"Guided Safe MPPI gamma={gamma:.3g}"
    if method == "guided_adaptive":
        return "Guided Safe MPPI (adaptive gamma)"
    if method == "mirror_mppi":
        return f"Mirror-MPPI gamma={gamma:.3g}"
    return method


def _frame_obstacles(obstacles_seq: np.ndarray, frame: int) -> np.ndarray:
    if obstacles_seq.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return obstacles_seq[min(int(frame), obstacles_seq.shape[0] - 1)]


def _frame_velocities(velocities_seq: np.ndarray, frame: int) -> np.ndarray:
    if velocities_seq.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return velocities_seq[min(int(frame), velocities_seq.shape[0] - 1)]


def _pad_sequence(seq: np.ndarray, target_frames: int) -> np.ndarray:
    if seq.shape[0] >= target_frames:
        return seq
    if seq.shape[0] == 0:
        return seq
    pad = np.repeat(seq[-1:, ...], target_frames - seq.shape[0], axis=0)
    return np.concatenate([seq, pad], axis=0)


def _dynamic_clearances(states_xy: np.ndarray, obstacles_seq: np.ndarray, safety_margin: float) -> np.ndarray:
    if obstacles_seq.size == 0 or obstacles_seq.shape[1] == 0:
        return np.full((states_xy.shape[0],), np.inf, dtype=np.float32)
    values = np.empty((states_xy.shape[0],), dtype=np.float32)
    for i, pos in enumerate(states_xy):
        obstacles = _frame_obstacles(obstacles_seq, i)
        centers = obstacles[:, :2]
        radii = obstacles[:, 2] + float(safety_margin)
        finite = np.isfinite(centers).all(axis=1) & np.isfinite(radii)
        if not finite.any():
            values[i] = np.inf
            continue
        d = np.linalg.norm(pos[None, :] - centers[finite], axis=1) - radii[finite]
        values[i] = float(np.min(d))
    return values


def _compute_dynamic_episode_metrics(
    states: np.ndarray,
    controls: np.ndarray,
    obstacles_seq: np.ndarray,
    goal: np.ndarray,
    *,
    safety_margin: float,
    success_threshold: float,
    planning_times: List[float],
    min_barrier_h: float | None,
    num_barrier_violations: int,
) -> Dict[str, object]:
    xy = states[:, :2]
    c = _dynamic_clearances(xy, obstacles_seq, safety_margin)
    final_dist = float(np.linalg.norm(xy[-1] - goal[:2]))
    diffs = np.diff(xy, axis=0)
    path_length = float(np.linalg.norm(diffs, axis=1).sum()) if len(diffs) else 0.0
    effort = float(np.sum(controls**2)) if controls.size else 0.0
    smooth = float(np.sum(np.diff(controls, axis=0) ** 2)) if controls.shape[0] > 1 else 0.0
    finite_clearances = c[np.isfinite(c)]
    min_clearance = float(np.min(c)) if len(c) else float("inf")
    return {
        "success": bool(final_dist <= success_threshold and min_clearance >= 0.0),
        "collision": bool(min_clearance < 0.0),
        "goal_reached": bool(final_dist <= success_threshold),
        "min_clearance": min_clearance,
        "mean_clearance": float(np.mean(finite_clearances)) if finite_clearances.size else float("inf"),
        "min_barrier_h": min_barrier_h,
        "num_barrier_violations": int(num_barrier_violations),
        "final_goal_distance": final_dist,
        "path_length": path_length,
        "control_effort": effort,
        "control_smoothness": smooth,
        "episode_return": -final_dist - 100.0 * float(min_clearance < 0.0),
        "episode_cost": final_dist + effort,
        "planning_wall_time_mean": float(np.mean(planning_times)) if planning_times else 0.0,
        "planning_wall_time_median": float(np.median(planning_times)) if planning_times else 0.0,
        "planning_wall_time_p95": float(np.percentile(planning_times, 95)) if planning_times else 0.0,
        "total_wall_time": float(np.sum(planning_times)) if planning_times else 0.0,
    }


def _load_validation_scene(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    if args.dataset not in {"ucy", "sdd"}:
        raise ValueError("validation pedestrian files are available for --dataset ucy or --dataset sdd")
    suffix = f"_{args.dataset}"
    ego_path = Path(f"dataset/eval80_ego{suffix}.pt")
    obs_path = Path(f"dataset/eval80_obs{suffix}.pkl")
    if not ego_path.exists() or not obs_path.exists():
        raise FileNotFoundError(f"Missing validation files: {ego_path} / {obs_path}")

    batch_ego = torch.load(ego_path, map_location="cpu", weights_only=False)
    with obs_path.open("rb") as f:
        batch_obs = pickle.load(f)
    idx = int(args.episode) % len(batch_obs)
    state_obs = batch_obs[idx].detach().cpu()
    nan_mask = torch.isnan(state_obs).any(dim=(0, 2, 3))
    state_obs = state_obs[:, ~nan_mask]
    obs = state_obs[0].float().numpy()  # [N_ped, 6, T]
    ped_count, _, frames = obs.shape
    obstacles_seq = np.zeros((frames, ped_count, 3), dtype=np.float32)
    velocities_seq = np.zeros((frames, ped_count, 2), dtype=np.float32)
    obstacles_seq[:, :, 0] = obs[:, 0, :].T
    obstacles_seq[:, :, 1] = obs[:, 1, :].T
    obstacles_seq[:, :, 2] = float(args.pedestrian_radius)
    velocities_seq[:, :, 0] = obs[:, 2, :].T
    velocities_seq[:, :, 1] = obs[:, 3, :].T

    state_dim = 3 if args.dynamics == "unicycle" else 4
    state0 = np.zeros((state_dim,), dtype=np.float32)
    goal = batch_ego[idx, :2, -1].float().numpy().astype(np.float32)
    target_frames = int(args.steps) + 1
    return (
        state0,
        goal,
        _pad_sequence(obstacles_seq, target_frames),
        _pad_sequence(velocities_seq, target_frames),
        f"{args.dataset} validation episode {idx}",
    )


def _make_sfm_social_force_scene(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    state_dim = 3 if args.dynamics == "unicycle" else 4
    state0 = np.zeros((state_dim,), dtype=np.float32)
    goal = np.array([6.0, 6.0], dtype=np.float32)
    rng = np.random.RandomState(int(args.seed) + int(args.episode))
    humans = [HumanAgent(goal, random_generator=rng) for _ in range(int(args.num_pedestrians))]
    frames = int(args.steps) + 1
    obstacles_seq = np.zeros((frames, len(humans), 3), dtype=np.float32)
    velocities_seq = np.zeros((frames, len(humans), 2), dtype=np.float32)
    robot_state = state0[:2].copy()
    robot_control = np.zeros(2, dtype=np.float32)
    for t in range(frames):
        for i, human in enumerate(humans):
            obstacles_seq[t, i, :2] = human.state
            obstacles_seq[t, i, 2] = float(args.pedestrian_radius)
            velocities_seq[t, i] = human.control
        if t == frames - 1:
            break
        for i, human in enumerate(humans):
            others_states = np.vstack([obstacles_seq[t, :i, :2], obstacles_seq[t, i + 1 :, :2], robot_state[None, :]])
            others_controls = np.vstack([velocities_seq[t, :i], velocities_seq[t, i + 1 :], robot_control[None, :]])
            human.social_force_step(others_states, others_controls)
    return state0, goal, obstacles_seq, velocities_seq, f"sfm social-force episode {args.episode}"


def _make_static_scene(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    state0, goal, obstacles = _make_episode(args.seed + args.episode, args.dynamics, args.dataset)
    frames = int(args.steps) + 1
    obstacles_seq = np.repeat(obstacles[None, :, :], frames, axis=0).astype(np.float32)
    velocities_seq = np.zeros((frames, obstacles.shape[0], 2), dtype=np.float32)
    return state0, goal, obstacles_seq, velocities_seq, "synthetic static"


def _make_scene(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    source = args.pedestrian_source
    if source == "auto":
        source = "validation" if args.dataset in {"ucy", "sdd"} else "sfm-social-force"
    if source == "validation":
        return _load_validation_scene(args)
    if source == "sfm-social-force":
        return _make_sfm_social_force_scene(args)
    if source == "synthetic-static":
        return _make_static_scene(args)
    raise ValueError(f"Unknown pedestrian source: {args.pedestrian_source}")


def _rollout(
    args: argparse.Namespace,
    policies: BenchmarkPolicies,
    method: str,
    gamma: Optional[float],
    state0: np.ndarray,
    goal: np.ndarray,
    obstacles_seq: np.ndarray,
    velocities_seq: np.ndarray,
) -> RenderRun:
    if method in ("mizuta_cfm_mppi", "mizuta_safe"):
        policies._mizuta_episode = None

    horizon = int(args.steps)
    dt = float(DEFAULTS["dt"])
    gamma_value = float(gamma if gamma is not None else 0.5)
    state = state0.copy()
    states = [state.copy()]
    controls: List[np.ndarray] = []
    planning_times: List[float] = []
    debug_steps: List[Dict[str, np.ndarray]] = []
    min_barrier_h = None
    num_barrier_violations = 0
    checkpoint_path = None
    model_calls_per_step = 0
    nfe = 0
    backup_selected_count = 0
    backup_branch_count = 0

    for step in range(horizon):
        obstacles = _frame_obstacles(obstacles_seq, step)
        obstacle_velocities = _frame_velocities(velocities_seq, step)
        action, info = policies.action(
            method,
            state,
            goal,
            obstacles,
            controls,
            args.dynamics,
            gamma_value,
            horizon,
            obstacle_velocities=obstacle_velocities,
        )
        planning_times.append(float(info.get("planning_wall_time", 0.0)))
        checkpoint_path = info.get("checkpoint", checkpoint_path)
        model_calls_per_step = int(info.get("model_calls_per_step", model_calls_per_step))
        nfe = int(info.get("nfe", nfe))
        if info.get("min_barrier_h") is not None:
            value = float(info["min_barrier_h"])
            min_barrier_h = value if min_barrier_h is None else min(min_barrier_h, value)
        num_barrier_violations += int(info.get("num_barrier_violations", 0))
        if "debug_rollouts" in info:
            debug_steps.append(info["debug_rollouts"])
        elif "debug_sequences" in info:
            debug_steps.append(info["debug_sequences"])
        else:
            debug_steps.append({})
        backup_branch_count = max(backup_branch_count, int(info.get("num_backup_branches", 0) or 0))
        if info.get("selected_backup_branch"):
            backup_selected_count += 1

        state = _dynamics_step(state, action, args.dynamics, dt)
        states.append(state.copy())
        controls.append(action.copy())

    states_arr = np.asarray(states, dtype=np.float32)
    controls_arr = np.asarray(controls, dtype=np.float32) if controls else np.zeros((0, 2), dtype=np.float32)
    metrics = _compute_dynamic_episode_metrics(
        states_arr,
        controls_arr,
        obstacles_seq,
        goal,
        safety_margin=float(DEFAULTS["safety_margin"]),
        success_threshold=float(DEFAULTS["success_threshold"]),
        planning_times=planning_times,
        min_barrier_h=min_barrier_h,
        num_barrier_violations=num_barrier_violations,
    )
    metrics.update(
        {
            "method": method,
            "gamma": gamma_value if method in {"safemppi_gamma", "safe_cfm", "guided_safemppi", "mirror_mppi"} else None,
            "checkpoint_path": checkpoint_path,
            "model_calls_per_step": model_calls_per_step,
            "nfe": nfe,
            "backup_branch_count": backup_branch_count,
            "backup_selected_count": backup_selected_count,
        }
    )
    return RenderRun(
        label=_method_label(method, gamma if method != "mizuta_cfm_mppi" else None),
        method=method,
        gamma=gamma if method in {"safemppi_gamma", "safe_cfm", "guided_safemppi", "mirror_mppi"} else None,
        states=states_arr,
        controls=controls_arr,
        metrics=metrics,
        debug=debug_steps,
    )


def _make_runs(args: argparse.Namespace) -> Tuple[List[RenderRun], np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    _set_seed(args.seed)
    state0, goal, obstacles_seq, velocities_seq, scene_label = _make_scene(args)
    device = torch.device(args.device)
    policies = BenchmarkPolicies(_policy_args(args), device)
    gamma_values = resolve_gamma_schedule(args.gamma_grid, args.gamma_schedule)
    methods = ["mizuta_cfm_mppi"]
    if not args.no_safemppi:
        methods.append("safemppi_gamma")
    if getattr(args, "include_guided", False):
        methods.append("guided_safemppi")
    if getattr(args, "include_guided_adaptive", False):
        methods.append("guided_adaptive")
    if getattr(args, "include_mirror", False):
        methods.append("mirror_mppi")
    if not args.no_safe_cfm:
        methods.append("safe_cfm")

    variants: List[Tuple[str, Optional[float]]] = []
    for method in methods:
        if method in {"mizuta_cfm_mppi", "guided_adaptive"}:
            variants.append((method, None))
        else:
            variants.extend((method, gamma) for gamma in gamma_values)

    runs = [
        _rollout(args, policies, method, gamma, state0, goal, obstacles_seq, velocities_seq)
        for method, gamma in _iter_progress(variants, desc="rolling out validation variants")
    ]
    for run in runs:
        run.metrics["scene"] = scene_label
        run.metrics["episode"] = int(args.episode)
    return runs, state0, goal, obstacles_seq, velocities_seq, scene_label


def _make_scenario(args: argparse.Namespace, episode: int) -> RenderScenario:
    scenario_args = argparse.Namespace(**vars(args))
    scenario_args.episode = int(episode)
    runs, state0, goal, obstacles_seq, velocities_seq, scene_label = _make_runs(scenario_args)
    return RenderScenario(
        runs=runs,
        state0=state0,
        goal=goal,
        obstacles_seq=obstacles_seq,
        velocities_seq=velocities_seq,
        label=scene_label,
    )


def _axis_limits(runs: List[RenderRun], state0: np.ndarray, goal: np.ndarray, obstacles_seq: np.ndarray) -> Tuple[float, float, float, float]:
    points = [state0[:2].reshape(1, 2), goal[:2].reshape(1, 2)]
    points.extend(run.states[:, :2] for run in runs)
    if obstacles_seq.size:
        points.append(obstacles_seq[:, :, :2].reshape(-1, 2))
    xy = np.concatenate(points, axis=0)
    xmin, ymin = np.nanmin(xy, axis=0)
    xmax, ymax = np.nanmax(xy, axis=0)
    if obstacles_seq.size:
        radius = obstacles_seq[:, :, 2] + float(DEFAULTS["safety_margin"])
        xmin = min(xmin, float(np.nanmin(obstacles_seq[:, :, 0] - radius)))
        xmax = max(xmax, float(np.nanmax(obstacles_seq[:, :, 0] + radius)))
        ymin = min(ymin, float(np.nanmin(obstacles_seq[:, :, 1] - radius)))
        ymax = max(ymax, float(np.nanmax(obstacles_seq[:, :, 1] + radius)))
    span = max(xmax - xmin, ymax - ymin, 1.0)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    half = 0.55 * span + 0.5
    return cx - half, cx + half, cy - half, cy + half


def _draw_scene(ax, state0: np.ndarray, goal: np.ndarray, obstacles_seq: np.ndarray, frame: int, args: argparse.Namespace) -> None:
    margin = float(DEFAULTS["safety_margin"])
    obstacles = _frame_obstacles(obstacles_seq, frame)
    if args.show_pedestrian_trails and obstacles_seq.size:
        trail = obstacles_seq[: min(frame + 1, obstacles_seq.shape[0]), :, :2]
        for ped_idx in range(trail.shape[1]):
            xy = trail[:, ped_idx]
            finite = np.isfinite(xy).all(axis=1)
            if finite.sum() > 1:
                ax.plot(xy[finite, 0], xy[finite, 1], color="#7b3294", lw=1.0, alpha=0.16, zorder=1)
    for obs in obstacles:
        x, y, radius = float(obs[0]), float(obs[1]), float(obs[2])
        if not np.isfinite([x, y, radius]).all():
            continue
        ax.add_patch(
            Circle(
                (x, y),
                radius + margin,
                facecolor=(0.48, 0.24, 0.68, 0.12),
                edgecolor=(0.48, 0.24, 0.68, 0.35),
                lw=1.0,
            )
        )
        ax.add_patch(Circle((x, y), radius, facecolor="#7b3294", edgecolor="#4b1360", alpha=0.72, lw=0.8))
    ax.scatter(state0[0], state0[1], c="#1f77b4", s=82, marker="*", zorder=8)
    ax.scatter(goal[0], goal[1], c="#d62728", s=82, marker="*", zorder=8)


def _draw_safemppi_hyperplanes(
    ax,
    run: RenderRun,
    frame: int,
    obstacles: np.ndarray,
    limits: Tuple[float, float, float, float],
    args: argparse.Namespace,
) -> None:
    if run.method not in {"safemppi_gamma", "guided_safemppi"} or run.gamma is None or obstacles.size == 0 or not args.draw_hyperplanes:
        return

    idx = min(frame, run.states.shape[0] - 1)
    pos0 = run.states[idx, :2].astype(np.float64)
    centers = obstacles[:, :2].astype(np.float64)
    radii = obstacles[:, 2].astype(np.float64) + float(DEFAULTS["safety_margin"])
    clearances = np.linalg.norm(centers - pos0[None, :], axis=1) - radii
    obs_idx = int(np.argmin(clearances))
    center = centers[obs_idx]
    radius = float(radii[obs_idx])
    diff0 = pos0 - center
    dist0 = float(np.linalg.norm(diff0))
    if dist0 <= 1e-8:
        return

    nearest = center + diff0 / dist0 * radius
    d0b = max(dist0 - radius, 1e-6)
    normal = nearest - pos0
    n_norm = float(np.linalg.norm(normal))
    if n_norm <= 1e-8:
        return
    normal = normal / n_norm
    tangent = np.array([-normal[1], normal[0]], dtype=np.float64)
    xmin, xmax, ymin, ymax = limits
    span = 1.6 * max(xmax - xmin, ymax - ymin)
    horizon = max(1, int(args.hyperplane_horizon))
    stride = max(1, int(args.hyperplane_stride))

    ax.add_patch(
        Circle(
            tuple(center),
            radius,
            facecolor=(1.0, 0.55, 0.0, 0.10),
            edgecolor="#f46d43",
            lw=2.0,
            zorder=3,
        )
    )
    ax.scatter(center[0], center[1], s=22, c="#f46d43", edgecolor="white", lw=0.6, zorder=5)

    for i in range(0, horizon + 1, stride):
        threshold = float((1.0 - run.gamma) ** i)
        line_center = nearest - normal * threshold * d0b
        p0 = line_center - tangent * span
        p1 = line_center + tangent * span
        alpha = 0.08 + 0.34 * (1.0 - i / max(horizon, 1))
        lw = 0.65 if i else 1.1
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#f46d43", alpha=alpha, lw=lw, zorder=3)

    ax.text(
        0.98,
        0.98,
        r"$h_{aff}=(1-\gamma)^i$ planes",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        color="#a33b18",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#f4a582", "alpha": 0.78},
    )
    ax.text(
        0.98,
        0.90,
        "nearest safety disk highlighted",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
        color="#a33b18",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#f4a582", "alpha": 0.72},
    )


def _draw_debug_predictions(ax, run: RenderRun, frame: int) -> None:
    if not run.debug:
        return
    if frame >= len(run.debug):
        return
    debug = run.debug[frame]
    if not debug:
        return
    if run.method in {"safemppi_gamma", "guided_safemppi", "guided_adaptive", "mirror_mppi"} and "states" in debug and "feasible" in debug:
        states = np.asarray(debug["states"])
        feasible = np.asarray(debug["feasible"]).astype(bool)
        for rollout, ok in zip(states, feasible):
            color = "#2b8cbe" if ok else "#d73027"
            alpha = 0.20 if ok else 0.12
            lw = 0.65 if ok else 0.55
            ax.plot(rollout[:, 0], rollout[:, 1], color=color, alpha=alpha, lw=lw, zorder=2)
        if "best_state" in debug:
            best = np.asarray(debug["best_state"])
            ax.plot(best[:, 0], best[:, 1], color="#084081", alpha=0.75, lw=1.6, zorder=4)
        branch_states = np.asarray(debug.get("branch_states", []))
        branch_feasible = np.asarray(debug.get("branch_feasible", []), dtype=bool)
        branch_labels = list(debug.get("branch_labels", []))
        branch_kinds = list(debug.get("branch_kinds", []))
        branch_palette = [
            "#542788",
            "#b35806",
            "#01665e",
            "#5e3c99",
            "#e66101",
            "#1b7837",
            "#762a83",
            "#a6d854",
            "#3288bd",
            "#fdae61",
            "#66c2a5",
            "#d53e4f",
        ]
        if branch_states.size:
            for idx, seq in enumerate(branch_states):
                label = branch_labels[idx] if idx < len(branch_labels) else f"b{idx}"
                kind = branch_kinds[idx] if idx < len(branch_kinds) else "backup"
                ok = bool(branch_feasible[idx]) if idx < len(branch_feasible) else False
                color = branch_palette[idx % len(branch_palette)]
                linestyle = "-" if ok else "--"
                alpha = 0.86 if ok else 0.58
                lw = 1.55 if kind == "sets_mode" else 1.85
                ax.plot(seq[:, 0], seq[:, 1], color=color, alpha=alpha, lw=lw, ls=linestyle, zorder=4)
                if getattr(ax.figure, "_show_backup_labels", False):
                    end = seq[min(len(seq) - 1, max(1, len(seq) // 2)), :2]
                    ax.text(
                        end[0],
                        end[1],
                        label,
                        fontsize=5.4,
                        color=color,
                        ha="center",
                        va="center",
                        bbox={"boxstyle": "round,pad=0.08", "facecolor": "white", "edgecolor": color, "alpha": 0.58},
                        zorder=6,
                    )
        ax.text(
            0.98,
            0.82,
            f"rollouts: {int(feasible.sum())} accept / {int((~feasible).sum())} reject"
            + (f"\nbackup modes: {len(branch_states)}" if branch_states.size else ""),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.5,
            color="#444444",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.78},
        )
    elif run.method == "safe_cfm" and "states" in debug:
        states = np.asarray(debug["states"])
        for seq in states:
            ax.plot(seq[:, 0], seq[:, 1], color="#1a9850", alpha=0.18, lw=0.75, zorder=2)
        if "best_state" in debug:
            best = np.asarray(debug["best_state"])
            ax.plot(best[:, 0], best[:, 1], color="#006837", alpha=0.8, lw=1.7, zorder=4)
        ax.text(
            0.98,
            0.98,
            "SafeCFM generated sequences",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.5,
            color="#226b2f",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#a6dba0", "alpha": 0.78},
        )


def _draw_run_panel(
    ax,
    run: RenderRun,
    frame: int,
    state0: np.ndarray,
    goal: np.ndarray,
    obstacles_seq: np.ndarray,
    limits: Tuple[float, float, float, float],
    args: argparse.Namespace,
) -> None:
    xmin, xmax, ymin, ymax = limits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.14, lw=0.6)
    ax.tick_params(labelsize=7)
    _draw_scene(ax, state0, goal, obstacles_seq, frame, args)
    _draw_safemppi_hyperplanes(ax, run, frame, _frame_obstacles(obstacles_seq, frame), limits, args)
    _draw_debug_predictions(ax, run, frame)

    idx = min(frame, run.states.shape[0] - 1)
    path = run.states[: idx + 1, :2]
    full_path = run.states[:, :2]
    ax.plot(full_path[:, 0], full_path[:, 1], color="#bbbbbb", lw=1.1, alpha=0.45, zorder=2)
    ax.plot(path[:, 0], path[:, 1], color="#d73027", lw=2.4, zorder=5)
    ax.add_patch(Circle(tuple(path[-1]), 0.12, facecolor="#2ca25f", edgecolor="white", lw=0.8, zorder=7))

    status = "success" if run.metrics["success"] else ("collision" if run.metrics["collision"] else "incomplete")
    title_color = {"success": "#1a9850", "collision": "#d73027"}.get(status, "#444444")
    ax.set_title(run.label, fontsize=10, color=title_color, pad=5)
    stats = (
        f"{status} | final {run.metrics['final_goal_distance']:.2f}\n"
        f"min clearance {run.metrics['min_clearance']:.2f} | mean step {1000.0 * run.metrics['planning_wall_time_mean']:.1f} ms"
    )
    ax.text(
        0.02,
        0.02,
        stats,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.86},
    )


def _frame_image(
    runs: List[RenderRun],
    frame: int,
    state0: np.ndarray,
    goal: np.ndarray,
    obstacles_seq: np.ndarray,
    args: argparse.Namespace,
    limits: Tuple[float, float, float, float],
    scene_label: str,
) -> np.ndarray:
    cols = min(args.cols, len(runs))
    rows = int(math.ceil(len(runs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 4.1 * rows), squeeze=False)
    fig.patch.set_facecolor("white")
    fig._show_backup_labels = bool(args.show_backup_labels)
    for ax in axes.ravel()[len(runs) :]:
        ax.axis("off")
    for ax, run in zip(axes.ravel(), runs):
        _draw_run_panel(ax, run, frame, state0, goal, obstacles_seq, limits, args)
    fig.suptitle(
        f"Moving-pedestrian comparison ({scene_label}): Mizuta vs SafeMPPI gamma sweep vs Safe CFM gamma sweep | frame {frame:03d}",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    canvas = FigureCanvas(fig)
    canvas.draw()
    image = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return image


def _multi_scenario_frame_image(
    scenarios: List[RenderScenario],
    frame: int,
    args: argparse.Namespace,
    limits_by_scene: List[Tuple[float, float, float, float]],
) -> np.ndarray:
    rows = len(scenarios)
    cols = max(len(scenario.runs) for scenario in scenarios)
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.65 * rows), squeeze=False)
    fig.patch.set_facecolor("white")
    fig._show_backup_labels = bool(args.show_backup_labels)
    for row, (scenario, limits) in enumerate(zip(scenarios, limits_by_scene)):
        for col in range(cols):
            ax = axes[row, col]
            if col >= len(scenario.runs):
                ax.axis("off")
                continue
            _draw_run_panel(
                ax,
                scenario.runs[col],
                frame,
                scenario.state0,
                scenario.goal,
                scenario.obstacles_seq,
                limits,
                args,
            )
            if col == 0:
                ax.text(
                    -0.16,
                    0.50,
                    scenario.label,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#333333",
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85},
                )
    fig.suptitle(
        f"Moving-pedestrian SafeMPPI gamma comparison {args.figure_tag} | frame {frame:03d}",
        fontsize=13,
        y=0.997,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    canvas = FigureCanvas(fig)
    canvas.draw()
    image = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return image


def _save_video(frames: List[np.ndarray], output: Path, fps: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    try:
        import imageio.v2 as imageio

        if suffix == ".gif":
            imageio.mimsave(output, frames, duration=1.0 / float(fps), loop=0)
        else:
            imageio.mimsave(output, frames, fps=fps, macro_block_size=1)
        return
    except ModuleNotFoundError:
        pass

    if suffix != ".gif":
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("MP4 output requires imageio or ffmpeg. Use --output path.gif as a fallback.")
        from PIL import Image

        with tempfile.TemporaryDirectory(prefix="cfm_mppi_frames_") as tmp:
            tmp_path = Path(tmp)
            for i, frame in enumerate(frames):
                Image.fromarray(frame).save(tmp_path / f"frame_{i:05d}.png")
            cmd = [
                ffmpeg,
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(tmp_path / "frame_%05d.png"),
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-pix_fmt",
                "yuv420p",
                str(output),
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return

    from PIL import Image

    pil_frames = [Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(
        output,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / float(fps)),
        loop=0,
    )


def _write_metrics(runs: List[RenderRun], output: Path) -> Path:
    path = output.with_suffix(".csv")
    fields = [
        "method",
        "gamma",
        "success",
        "collision",
        "goal_reached",
        "final_goal_distance",
        "min_clearance",
        "path_length",
        "control_effort",
        "planning_wall_time_mean",
        "model_calls_per_step",
        "nfe",
        "backup_branch_count",
        "backup_selected_count",
        "checkpoint_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            writer.writerow({field: run.metrics.get(field) for field in fields})
    return path


def _write_scenario_metrics(scenarios: List[RenderScenario], output: Path) -> Path:
    path = output.with_suffix(".csv")
    fields = [
        "scene",
        "episode",
        "method",
        "gamma",
        "success",
        "collision",
        "goal_reached",
        "final_goal_distance",
        "min_clearance",
        "path_length",
        "control_effort",
        "planning_wall_time_mean",
        "model_calls_per_step",
        "nfe",
        "backup_branch_count",
        "backup_selected_count",
        "checkpoint_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for scenario in scenarios:
            for run in scenario.runs:
                writer.writerow({field: run.metrics.get(field) for field in fields})
    return path


def _write_experiment_notes(args: argparse.Namespace, output: Path, scenarios: List[RenderScenario]) -> Path:
    path = output.with_name(output.stem + "_notes.md")
    episode_list = [scenario.runs[0].metrics.get("episode") for scenario in scenarios if scenario.runs]
    tag = str(args.figure_tag)
    sets_flag = "--safemppi-use-sets-backup" if args.safemppi_use_sets_backup else "--no-safemppi-use-sets-backup"
    cbf_flag = "--safemppi-sets-include-cbf-backup" if args.safemppi_sets_include_cbf_backup else "--no-safemppi-sets-include-cbf-backup"
    text = f"""# Validation Comparison {tag} Notes

Output: `{output}`

## Command Settings

- dataset: `{args.dataset}`
- dynamics: `{args.dynamics}`
- pedestrian_source: `{args.pedestrian_source}`
- episodes: `{episode_list}`
- steps: `{args.steps}`
- gamma_grid: `{args.gamma_grid}`
- no_safe_cfm: `{args.no_safe_cfm}`
- safemppi_horizon: `{args.safemppi_horizon}`
- safemppi_samples: `{args.safemppi_samples}`
- safemppi_running_goal_weight: `{args.safemppi_running_goal_weight}`
- safemppi_terminal_goal_weight: `{args.safemppi_terminal_goal_weight}`
- safemppi_control_weight: `{args.safemppi_control_weight}`
- safemppi_smooth_weight: `{args.safemppi_smooth_weight}`
- safemppi_soft_clearance_weight: `{args.safemppi_soft_clearance_weight}`
- safemppi_progress_weight: `{args.safemppi_progress_weight}`
- safemppi_use_sets_backup: `{args.safemppi_use_sets_backup}`
- safemppi_sets_num_modes: `{args.safemppi_sets_num_modes}`
- safemppi_sets_branch_scale: `{args.safemppi_sets_branch_scale}`
- safemppi_sets_include_cbf_backup: `{args.safemppi_sets_include_cbf_backup}`
- safemppi_sets_cbf_push: `{args.safemppi_sets_cbf_push}`
- safemppi_sets_reverse_speed: `{args.safemppi_sets_reverse_speed}`
- safemppi_sets_turn_rate: `{args.safemppi_sets_turn_rate}`
- debug_rollouts: `{args.debug_rollouts}`
- draw_hyperplanes: `{args.draw_hyperplanes}`
- hyperplane_horizon: `{args.hyperplane_horizon}`
- hyperplane_stride: `{args.hyperplane_stride}`
- pedestrian_radius: `{args.pedestrian_radius}`
- safety_margin: `{DEFAULTS["safety_margin"]}`
- show_backup_labels: `{args.show_backup_labels}`

## {tag} Changes

- Hyperplanes are tangent to the effective safety disk, using `pedestrian_radius + r_safe`.
- The orange disk is the effective safety disk; the plotted affine CBF thresholds are parallel
  support planes between the robot-side plane and the tangent plane on that disk.
- The nearest safety disk for each SafeMPPI panel is highlighted in orange.
- SafeMPPI panels draw accepted rollouts in blue and rejected rollouts in red.
- This run uses a 4-scenario by 4-method grid: Mizuta, SafeMPPI gamma 0.1, 0.5, 1.0.
- Safe CFM is intentionally disabled here to isolate the SafeMPPI gamma tradeoff.
- SafeMPPI uses heading-aware unicycle nominal control, original-style input bounds `[-2, 2]`, unicycle noise `[0.3, 0.6]`, moving-pedestrian constant-velocity prediction, and tuned goal/smooth/safety costs.
- If enabled, SETS-style backup proposals are appended to the random MPPI samples. These
  branches linearize around the nominal trajectory, form `C C^T`, use `+-sqrt(lambda_i) q_i`
  terminal displacements, solve `C^+ delta_z`, clip normalized inputs to `[0,1]`, and add
  nearest-halfspace away/tangent/reverse backup controls.
- Backup modes are drawn with distinct colors; dashed backup branches were rejected by the
  affine barrier test and solid backup branches were accepted.

## Re-run Template

```bash
conda run --live-stream -n cfm_mppi python -m cfm_mppi.evaluation.render_validation_comparison \\
  --dataset {args.dataset} \\
  --dynamics {args.dynamics} \\
  --pedestrian-source {args.pedestrian_source} \\
  --episode-list {' '.join(str(x) for x in episode_list)} \\
  --steps {args.steps} \\
  --gamma-grid {' '.join(str(x) for x in args.gamma_grid)} \\
  --no-safe-cfm \\
  --device cuda \\
  --safemppi-horizon {args.safemppi_horizon} \\
  --safemppi-samples {args.safemppi_samples} \\
  --safemppi-running-goal-weight {args.safemppi_running_goal_weight} \\
  --safemppi-terminal-goal-weight {args.safemppi_terminal_goal_weight} \\
  --safemppi-control-weight {args.safemppi_control_weight} \\
  --safemppi-smooth-weight {args.safemppi_smooth_weight} \\
  --safemppi-soft-clearance-weight {args.safemppi_soft_clearance_weight} \\
  --safemppi-progress-weight {args.safemppi_progress_weight} \\
  {sets_flag} \\
  --safemppi-sets-num-modes {args.safemppi_sets_num_modes} \\
  --safemppi-sets-branch-scale {args.safemppi_sets_branch_scale} \\
  {cbf_flag} \\
  --safemppi-sets-cbf-push {args.safemppi_sets_cbf_push} \\
  --safemppi-sets-reverse-speed {args.safemppi_sets_reverse_speed} \\
  --safemppi-sets-turn-rate {args.safemppi_sets_turn_rate} \\
  --debug-rollouts {args.debug_rollouts} \\
  --hyperplane-horizon {args.hyperplane_horizon} \\
  --hyperplane-stride {args.hyperplane_stride} \\
  {'--show-backup-labels' if args.show_backup_labels else '--no-show-backup-labels'} \\
  --figure-tag {tag} \\
  --output results/benchmark_videos/YOUR_NAME.mp4 \\
  --gif-output results/benchmark_videos/YOUR_NAME.gif
```

## Episode Iteration Prompt

Pick four validation episodes that stress different tradeoffs, then only change `--episode-list`
and the output stem:

```bash
for stem in {tag}_episode_set_a {tag}_episode_set_b; do
  conda run --live-stream -n cfm_mppi python -m cfm_mppi.evaluation.render_validation_comparison \\
    --dataset {args.dataset} \\
    --dynamics {args.dynamics} \\
    --pedestrian-source {args.pedestrian_source} \\
    --episode-list EP0 EP1 EP2 EP3 \\
    --steps {args.steps} \\
    --gamma-grid {' '.join(str(x) for x in args.gamma_grid)} \\
    --no-safe-cfm \\
    --device cuda \\
    --safemppi-horizon {args.safemppi_horizon} \\
    --safemppi-samples {args.safemppi_samples} \\
    {sets_flag} \\
    --safemppi-sets-num-modes {args.safemppi_sets_num_modes} \\
    --safemppi-sets-branch-scale {args.safemppi_sets_branch_scale} \\
    --debug-rollouts {args.debug_rollouts} \\
    --hyperplane-horizon {args.hyperplane_horizon} \\
    --hyperplane-stride {args.hyperplane_stride} \\
    {'--show-backup-labels' if args.show_backup_labels else '--no-show-backup-labels'} \\
    --figure-tag {tag} \\
    --output results/benchmark_videos/${{stem}}.mp4 \\
    --gif-output results/benchmark_videos/${{stem}}.gif
done
```
"""
    path.write_text(text, encoding="utf-8")
    return path


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a Mizuta / SafeMPPI / Safe-CFM validation comparison video.")
    parser.add_argument("--dataset", default="sfm", choices=["sfm", "ucy", "sdd"])
    parser.add_argument("--dynamics", default="doubleintegrator", choices=["doubleintegrator", "unicycle"])
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--episode-list", nargs="*", type=int, default=None)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pedestrian-source", choices=["auto", "validation", "sfm-social-force", "synthetic-static"], default="auto")
    parser.add_argument("--pedestrian-radius", type=float, default=0.0)
    parser.add_argument("--num-pedestrians", type=int, default=20)
    parser.add_argument("--gamma-grid", nargs="*", type=float, default=[0.1, 0.5, 1.0])
    parser.add_argument("--gamma-schedule", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mizuta-checkpoint", default="output_dir/cfm_transformer/checkpoint.pth")
    parser.add_argument("--safe-cfm-checkpoint", default="output_dir/safe_contextual_cfm/checkpoint_best.pth")
    parser.add_argument("--drifting-checkpoint", default="output_dir/drifting_generator/checkpoint_best.pth")
    parser.add_argument("--output", type=Path, default=Path("results/benchmark_videos/validation_comparison.mp4"))
    parser.add_argument("--gif-output", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-safemppi", action="store_true")
    parser.add_argument("--no-safe-cfm", action="store_true")
    parser.add_argument("--include-guided", action="store_true", help="add Guided Safe MPPI (per-gamma) panels")
    parser.add_argument("--include-guided-adaptive", action="store_true", help="add Guided Safe MPPI adaptive-gamma panel")
    parser.add_argument("--include-mirror", action="store_true", help="add Mirror-MPPI (feasible-by-construction) panels")
    parser.add_argument("--safemppi-horizon", type=int, default=40)
    parser.add_argument("--safemppi-samples", type=int, default=512)
    parser.add_argument("--safemppi-running-goal-weight", type=float, default=0.25)
    parser.add_argument("--safemppi-terminal-goal-weight", type=float, default=80.0)
    parser.add_argument("--safemppi-control-weight", type=float, default=0.03)
    parser.add_argument("--safemppi-smooth-weight", type=float, default=0.12)
    parser.add_argument("--safemppi-soft-clearance-weight", type=float, default=25.0)
    parser.add_argument("--safemppi-progress-weight", type=float, default=2.0)
    parser.add_argument("--safemppi-use-sets-backup", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--safemppi-sets-num-modes", type=int, default=3)
    parser.add_argument("--safemppi-sets-branch-scale", type=float, default=0.85)
    parser.add_argument("--safemppi-sets-include-cbf-backup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--safemppi-sets-cbf-push", type=float, default=1.25)
    parser.add_argument("--safemppi-sets-reverse-speed", type=float, default=0.75)
    parser.add_argument("--safemppi-sets-turn-rate", type=float, default=1.4)
    parser.add_argument("--safe-cfm-num-candidates", type=int, default=16)
    parser.add_argument("--debug-rollouts", type=int, default=80)
    parser.add_argument("--draw-hyperplanes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hyperplane-horizon", type=int, default=20)
    parser.add_argument("--hyperplane-stride", type=int, default=1)
    parser.add_argument("--show-pedestrian-trails", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-backup-labels", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--figure-tag", default="v2")
    return parser


def main() -> None:
    args = get_parser().parse_args()
    args.cols = max(1, int(args.cols))
    if args.smoke:
        args.steps = min(args.steps, 8)
        if args.gamma_grid == [0.1, 0.5, 1.0]:
            args.gamma_grid = [0.1, 1.0]
        args.safemppi_horizon = min(args.safemppi_horizon, 8)
        args.safemppi_samples = min(args.safemppi_samples, 64)
        args.safe_cfm_num_candidates = min(args.safe_cfm_num_candidates, 4)
        args.debug_rollouts = min(args.debug_rollouts, 24)

    if args.episode_list:
        scenarios = [
            _make_scenario(args, episode)
            for episode in _iter_progress(args.episode_list, desc="rolling out scenarios")
        ]
        limits_by_scene = [
            _axis_limits(scenario.runs, scenario.state0, scenario.goal, scenario.obstacles_seq)
            for scenario in scenarios
        ]
        max_frames = max(run.states.shape[0] for scenario in scenarios for run in scenario.runs)
        frames = [
            _multi_scenario_frame_image(scenarios, frame, args, limits_by_scene)
            for frame in _iter_progress(range(max_frames), desc="rendering frames")
        ]
        metrics_path = _write_scenario_metrics(scenarios, args.output)
        notes_path = _write_experiment_notes(args, args.output, scenarios)
    else:
        runs, state0, goal, obstacles_seq, velocities_seq, scene_label = _make_runs(args)
        _ = velocities_seq
        limits = _axis_limits(runs, state0, goal, obstacles_seq)
        max_frames = max(run.states.shape[0] for run in runs)
        frames = [
            _frame_image(runs, frame, state0, goal, obstacles_seq, args, limits, scene_label)
            for frame in _iter_progress(range(max_frames), desc="rendering frames")
        ]
        metrics_path = _write_metrics(runs, args.output)
        notes_path = None
    _save_video(frames, args.output, args.fps)
    if args.gif_output is not None:
        _save_video(frames, args.gif_output, args.fps)
    print(f"saved video: {args.output}", flush=True)
    if args.gif_output is not None:
        print(f"saved gif: {args.gif_output}", flush=True)
    print(f"saved metrics: {metrics_path}", flush=True)
    if notes_path is not None:
        print(f"saved notes: {notes_path}", flush=True)


if __name__ == "__main__":
    main()
