from __future__ import annotations

import numpy as np


def clearances(states_xy: np.ndarray, obstacles: np.ndarray, safety_margin: float) -> np.ndarray:
    if obstacles.size == 0:
        return np.full((states_xy.shape[0],), np.inf, dtype=np.float32)
    centers = obstacles[:, :2]
    radii = obstacles[:, 2] + float(safety_margin)
    d = np.linalg.norm(states_xy[:, None, :] - centers[None, :, :], axis=2) - radii[None, :]
    return d.min(axis=1)


def compute_episode_metrics(
    states: np.ndarray,
    controls: np.ndarray,
    obstacles: np.ndarray,
    goal: np.ndarray,
    *,
    safety_margin: float,
    success_threshold: float = 0.5,
    planning_times: list[float] | None = None,
    min_barrier_h: float | None = None,
    num_barrier_violations: int = 0,
) -> dict:
    xy = states[:, :2]
    c = clearances(xy, obstacles, safety_margin)
    final_dist = float(np.linalg.norm(xy[-1] - goal[:2]))
    diffs = np.diff(xy, axis=0)
    path_length = float(np.linalg.norm(diffs, axis=1).sum()) if len(diffs) else 0.0
    effort = float(np.sum(controls**2)) if controls.size else 0.0
    smooth = float(np.sum(np.diff(controls, axis=0) ** 2)) if controls.shape[0] > 1 else 0.0
    planning_times = planning_times or []
    return {
        "success": bool(final_dist <= success_threshold and np.min(c) >= 0.0),
        "collision": bool(np.min(c) < 0.0),
        "goal_reached": bool(final_dist <= success_threshold),
        "min_clearance": float(np.min(c)),
        "mean_clearance": float(np.mean(c[np.isfinite(c)])) if np.isfinite(c).any() else float("inf"),
        "min_barrier_h": min_barrier_h,
        "num_barrier_violations": int(num_barrier_violations),
        "final_goal_distance": final_dist,
        "path_length": path_length,
        "control_effort": effort,
        "control_smoothness": smooth,
        "episode_return": -final_dist - 100.0 * float(np.min(c) < 0.0),
        "episode_cost": final_dist + effort,
        "planning_wall_time_mean": float(np.mean(planning_times)) if planning_times else 0.0,
        "planning_wall_time_median": float(np.median(planning_times)) if planning_times else 0.0,
        "planning_wall_time_p95": float(np.percentile(planning_times, 95)) if planning_times else 0.0,
        "total_wall_time": float(np.sum(planning_times)) if planning_times else 0.0,
    }
