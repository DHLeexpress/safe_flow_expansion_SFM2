#!/usr/bin/env python3
"""
Corrected Pillar-3 verifier-polytope experiment for SafeMPPI local DI rollouts.

Key correction relative to the earlier failed draft:
  * A trajectory is a 10-step local MPPI forward pass, not a global path.
  * One fixed robot-centered polytope is drawn/verified per queried trajectory.
  * The polytope is rendered with the same H_grid level-set convention used by
    overnight_run_2026-06-28/di_grid.py and polytope_explainer.py.
  * Data are read from the attached eval80_ego*.pt and eval80_obs*.pkl moving-pedestrian files.

Run examples:
  python pillar3_corrected_eval.py --dataset ucy --data-dir /mnt/data --out-dir /mnt/data/pillar3_corrected_outputs
  python pillar3_corrected_eval.py --dataset sdd --data-dir /mnt/data --out-dir /mnt/data/pillar3_corrected_outputs

Dependencies: numpy, scipy, torch, matplotlib, pillow.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection


DT = 0.1
R_PED = 0.5
R_ROBOT = 0.2


@dataclass
class Poly:
    A: np.ndarray       # [F,2]
    b: np.ndarray       # [F]
    c: np.ndarray       # [2]
    margins: np.ndarray # [F] = b - A@c; normalizes H_P
    meta: Dict

    def as_tuple(self):
        return (self.A, self.b, self.c, self.margins)


def h_grid(poly_tuple, GX, GY):
    """Same convention as overnight_run_2026-06-28/polytope_explainer.py::H_grid.
    H_P(x)=min_k (b_k-a_k^T x)/margin_k, with H_P(c)=1 and boundary H_P=0.
    """
    A, b, c, mr = poly_tuple
    pts = np.stack([GX.ravel(), GY.ravel()], axis=1)
    return (((b[None] - pts @ A.T) / np.clip(mr[None], 1e-9, None)).min(axis=1)).reshape(GX.shape)


def finite_goal(xy_T: np.ndarray) -> np.ndarray:
    """Last finite ego position in a [2,T] array."""
    ok = np.isfinite(xy_T).all(axis=0)
    if not ok.any():
        return np.zeros(2, dtype=np.float32)
    return xy_T[:, np.nonzero(ok)[0][-1]].astype(np.float32)


def load_eval80(dataset: str, data_dir: str):
    dd = Path(data_dir)
    if dataset.lower() in ("ucy", "default", "eval80"):
        ego_path = dd / "eval80_ego_ucy.pt"
        obs_path = dd / "eval80_obs_ucy.pkl"
        if not ego_path.exists():
            ego_path = dd / "eval80_ego.pt"
        if not obs_path.exists():
            obs_path = dd / "eval80_obs.pkl"
    elif dataset.lower() == "sdd":
        ego_path = dd / "eval80_ego_sdd.pt"
        obs_path = dd / "eval80_obs_sdd.pkl"
    else:
        raise ValueError(f"unknown dataset {dataset!r}; expected ucy/default/eval80/sdd")
    ego = torch.load(str(ego_path), map_location="cpu").float().numpy()  # [E,6,80]
    with open(obs_path, "rb") as f:
        obs_list = pickle.load(f)
    return ego, obs_list, str(ego_path), str(obs_path)


def get_scene_frame(ego: np.ndarray, obs_list: list, ep: int, t: int, ped_radius: float = R_PED):
    e = ego[ep]
    T = e.shape[1]
    t = int(np.clip(t, 0, T - 1))
    state = np.array([e[0, t], e[1, t], e[2, t], e[3, t]], dtype=np.float32)
    goal = finite_goal(e[:2])
    item = obs_list[ep]
    if torch.is_tensor(item):
        arr = item.detach().cpu().float().numpy()
    else:
        arr = np.asarray(item, dtype=np.float32)
    # attached files are [1,N,6,T]; tolerate [N,6,T].
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3 or arr.shape[1] < 4:
        raise ValueError(f"unexpected obstacle tensor shape for episode {ep}: {arr.shape}")
    tt = min(t, arr.shape[2] - 1)
    pos = arr[:, :2, tt]
    vel = arr[:, 2:4, tt]
    ok = np.isfinite(pos).all(axis=1)
    pos = pos[ok]
    vel = vel[ok]
    obs = np.column_stack([pos, np.full(len(pos), float(ped_radius), dtype=np.float32)]).astype(np.float32)
    vel = vel.astype(np.float32)
    return state, goal.astype(np.float32), obs, vel


def pick_cases(ego: np.ndarray, obs_list: list, n_cases: int = 25, sensing: float = 2.0, min_goal_dist: float = 1.0) -> List[Tuple[int, int, Dict]]:
    """Pick distinct episodes with clutter near the ego, one frame per episode."""
    candidates = []
    E, _, T = ego.shape
    for ep in range(E):
        best = None
        goal = finite_goal(ego[ep, :2])
        for t in range(0, min(T - 10, 70)):
            st, _, ob, _ = get_scene_frame(ego, obs_list, ep, t)
            if not np.isfinite(st).all():
                continue
            gd = float(np.linalg.norm(goal - st[:2]))
            if gd < min_goal_dist:
                continue
            if ob.shape[0] == 0:
                nnear = 0
                min_clear = np.inf
                score = -100.0
            else:
                clr = np.linalg.norm(ob[:, :2] - st[:2], axis=1) - ob[:, 2] - R_ROBOT
                min_clear = float(np.nanmin(clr))
                nnear = int(np.sum(clr <= sensing))
                # prefer non-colliding but near/cluttered frames; strong clutter beats distance.
                collision_pen = 30.0 if min_clear < -0.05 else 0.0
                score = 8.0 * nnear - min(abs(min_clear), sensing) + 0.05 * gd - collision_pen
            row = (score, ep, t, dict(nnear=nnear, min_clear=min_clear, goal_dist=gd))
            if best is None or row[0] > best[0]:
                best = row
        if best is not None:
            candidates.append(best)
    candidates.sort(reverse=True, key=lambda z: z[0])
    selected = [(ep, t, info | {"score": float(score)}) for score, ep, t, info in candidates[:n_cases]]
    if len(selected) < n_cases:
        raise RuntimeError(f"only found {len(selected)} cases")
    return selected


def base_normals(K: int):
    th = np.arange(K, dtype=float) * (2.0 * math.pi / K)
    return np.stack([np.cos(th), np.sin(th)], axis=1).astype(np.float64)


def polygon_centroid(A: np.ndarray, b: np.ndarray, interior: np.ndarray) -> np.ndarray:
    interior = np.asarray(interior, dtype=float).reshape(2)
    try:
        hs = np.hstack([A, -b[:, None]]).astype(float)  # a.x - b <= 0
        hsi = HalfspaceIntersection(hs, interior)
        V = hsi.intersections
        if V.shape[0] < 3:
            return interior.copy()
        m = V.mean(axis=0)
        V = V[np.argsort(np.arctan2(V[:, 1] - m[1], V[:, 0] - m[0]))]
        x, y = V[:, 0], V[:, 1]
        xs, ys = np.roll(x, -1), np.roll(y, -1)
        cr = x * ys - xs * y
        area = 0.5 * cr.sum()
        if abs(area) < 1e-9:
            return interior.copy()
        Cx = ((x + xs) * cr).sum() / (6 * area)
        Cy = ((y + ys) * cr).sum() / (6 * area)
        return np.array([Cx, Cy], dtype=float)
    except Exception:
        return interior.copy()


def build_nominal_polytope(
    c: np.ndarray,
    obs: np.ndarray,
    *,
    sensing: float = 2.0,
    n_base: int = 16,
    margin: float = 0.0,
    max_obstacles: int = 12,
    obstacle_velocities: Optional[np.ndarray] = None,
    robot_velocity: Optional[np.ndarray] = None,
    predict_gain: float = 0.4,
    predict_tau: float = 1.0,
) -> Poly:
    """Faithful local implementation of cfm_mppi/safegpc_adapter/polytope_v2.py.
    The first K faces are the inner K-gon approximation of the sensing disk; then
    one radial tangent face is added for each detected obstacle.
    """
    c = np.asarray(c, dtype=np.float64).reshape(2)
    R = float(sensing)
    K = max(4, int(n_base))
    A_rows = [n for n in base_normals(K)]
    base_off = R * math.cos(math.pi / K)
    b_rows = [float(n @ c + base_off) for n in A_rows]
    obs = np.asarray(obs, dtype=np.float64).reshape(-1, 3) if np.asarray(obs).size else np.zeros((0, 3), dtype=float)
    vrob = np.zeros(2, dtype=float) if robot_velocity is None else np.asarray(robot_velocity, dtype=float).reshape(2)
    vobs = None if obstacle_velocities is None else np.asarray(obstacle_velocities, dtype=float).reshape(-1, 2)
    detected = []
    if obs.shape[0]:
        d = np.linalg.norm(obs[:, :2] - c[None, :], axis=1)
        clr = d - (obs[:, 2] + margin)
        n_detected = 0
        for j in np.argsort(clr):
            if n_detected >= max_obstacles or clr[j] > R:
                break
            if d[j] < 1e-9:
                continue
            m = (obs[j, :2] - c) / d[j]
            off = float(d[j] - (obs[j, 2] + margin))
            pred = 0.0
            if predict_gain > 0 and vobs is not None and j < vobs.shape[0]:
                vclose = float(m @ (vrob - vobs[j]))
                pred = float(predict_gain * predict_tau * max(0.0, vclose))
                off -= pred
            A_rows.append(m)
            b_rows.append(float(m @ c + off))
            detected.append(dict(index=int(j), clearance=float(clr[j]), off=float(off), pred=float(pred)))
            n_detected += 1
    A = np.stack(A_rows).astype(np.float64)
    b = np.asarray(b_rows, dtype=np.float64)
    margins = b - A @ c
    margins = np.maximum(margins, 1e-6)
    return Poly(A=A, b=b, c=c, margins=margins, meta=dict(kind="nominal", n_base=K, sensing=R, detected=detected))


def poly_H_points(poly: Poly, pts: np.ndarray) -> np.ndarray:
    return ((poly.b[None] - pts @ poly.A.T) / np.clip(poly.margins[None], 1e-9, None)).min(axis=1)


def verify_traj_levelset(poly: Poly, traj_xy: np.ndarray, gamma: float) -> Dict:
    H = len(traj_xy) - 1
    vals = poly_H_points(poly, np.asarray(traj_xy, dtype=float))
    levels = np.array([(1.0 - float(gamma)) ** i for i in range(H + 1)], dtype=float)
    residual = vals - levels
    return dict(ok=bool(np.all(residual >= -1e-7)), min_residual=float(np.min(residual)), H_values=vals.tolist(), levels=levels.tolist())


def required_margin_for_face(a: np.ndarray, rel_traj: np.ndarray, gamma: float) -> float:
    """Lower bound on margin m for H_P(q_i)>=(1-gamma)^i.
    For a face a.(x-c)<=m, normalized H constraint is
       a.(q_i-c) <= (1-alpha_i)m, alpha_i=(1-gamma)^i.
    """
    H = rel_traj.shape[0] - 1
    req = 0.0
    for i in range(1, H + 1):
        alpha = (1.0 - float(gamma)) ** i
        denom = max(1.0 - alpha, 1e-9)
        proj = float(a @ rel_traj[i])
        if proj > 0:
            req = max(req, proj / denom)
    return float(req)


def choose_obstacle_normal(
    c: np.ndarray,
    traj_xy: np.ndarray,
    obs_j: np.ndarray,
    vel_j: Optional[np.ndarray],
    vrob: np.ndarray,
    gamma: float,
    predict_gain: float,
    predict_tau: float,
    n_angles: int = 121,
) -> Tuple[Optional[np.ndarray], float, Dict]:
    """Trajectory-specific support-normal search for one obstacle.

    Candidate normal n must leave the robot at positive margin and produce a
    tangent halfspace n.(x-c) <= n.(o-c)-r-pred. We select the feasible candidate
    with maximum slack upper_margin - required_margin.
    """
    rel_o = np.asarray(obs_j[:2], dtype=float) - c
    d = float(np.linalg.norm(rel_o))
    r = float(obs_j[2])
    if d <= r + 1e-6:
        return None, -np.inf, dict(reason="robot_inside_or_on_obstacle")
    theta0 = math.atan2(rel_o[1], rel_o[0])
    # n.rel_o must exceed r; this restricts the normal to a cone around rel_o.
    max_dev = max(0.0, math.acos(min(0.999999, r / max(d, 1e-9))) - 1e-4)
    # include radial, cone edges, and interior samples.
    deltas = np.linspace(-max_dev, max_dev, max(3, int(n_angles)))
    rel_traj = np.asarray(traj_xy, dtype=float) - c[None, :]
    best = None
    best_meta = None
    for delta in deltas:
        th = theta0 + float(delta)
        n = np.array([math.cos(th), math.sin(th)], dtype=float)
        pred = 0.0
        if predict_gain > 0.0 and vel_j is not None:
            vclose = float(n @ (vrob - vel_j))
            pred = float(predict_gain * predict_tau * max(0.0, vclose))
        upper = float(n @ rel_o - r - pred)
        if upper <= 1e-5:
            continue
        req = required_margin_for_face(n, rel_traj, gamma)
        slack = upper - req
        feasible = slack >= -1e-8
        # Prefer feasible normals with maximum slack; if none feasible, keep the least bad normal.
        key = (1 if feasible else 0, slack, upper)
        if best is None or key > best:
            best = key
            best_meta = dict(theta=float(th), delta=float(delta), upper=float(upper), required=float(req), slack=float(slack), feasible=bool(feasible), pred=float(pred))
            best_n = n
    if best is None:
        return None, -np.inf, dict(reason="no_positive_tangent_margin")
    if not best_meta["feasible"]:
        return best_n, best_meta["upper"], best_meta | {"reason": "trajectory_not_separable_by_candidate_halfspace"}
    return best_n, best_meta["upper"], best_meta


def solve_verifier_polytope_lp(
    c: np.ndarray,
    traj_xy: np.ndarray,
    obs: np.ndarray,
    obs_vel: Optional[np.ndarray],
    *,
    gamma: float = 0.5,
    sensing: float = 2.0,
    n_base: int = 16,
    max_obstacles: int = 12,
    robot_velocity: Optional[np.ndarray] = None,
    predict_gain: float = 0.4,
    predict_tau: float = 1.0,
    n_angle_candidates: int = 121,
) -> Tuple[Optional[Poly], Dict]:
    """Fit one trajectory-specific verifier polytope.

    Variables are margins m_k=b_k-a_k^T c.  Normals are fixed after a fast
    support-normal selection step: K sensing-disk faces plus one selected tangent
    support normal for every sensed obstacle.  LP:

      maximize 1^T m
      subject to m_min <= m_k <= u_k,
                 a_k^T(q_i-c) <= (1-alpha_i)m_k, alpha_i=(1-gamma)^i.

    This directly certifies H_P(q_i)>=alpha_i for the 10-step local rollout.
    """
    t0 = time.perf_counter()
    c = np.asarray(c, dtype=float).reshape(2)
    traj_xy = np.asarray(traj_xy, dtype=float).reshape(-1, 2)
    vrob = np.zeros(2, dtype=float) if robot_velocity is None else np.asarray(robot_velocity, dtype=float).reshape(2)
    obs = np.asarray(obs, dtype=float).reshape(-1, 3) if np.asarray(obs).size else np.zeros((0, 3), dtype=float)
    obs_vel = None if obs_vel is None else np.asarray(obs_vel, dtype=float).reshape(-1, 2)

    K = max(4, int(n_base))
    A_rows = []
    upper = []
    face_meta = []
    base_off = float(sensing) * math.cos(math.pi / K)
    for k, n in enumerate(base_normals(K)):
        A_rows.append(n)
        upper.append(base_off)
        face_meta.append(dict(type="base", index=k, upper=base_off))

    detected = []
    if obs.shape[0]:
        d = np.linalg.norm(obs[:, :2] - c[None, :], axis=1)
        clr = d - obs[:, 2]
        n_detected = 0
        for j in np.argsort(clr):
            if n_detected >= max_obstacles or clr[j] > float(sensing):
                break
            if not np.isfinite(clr[j]) or d[j] < 1e-9:
                continue
            vel_j = None if obs_vel is None or j >= obs_vel.shape[0] else obs_vel[j]
            n, ub, meta = choose_obstacle_normal(c, traj_xy, obs[j], vel_j, vrob, gamma, predict_gain, predict_tau, n_angles=n_angle_candidates)
            if n is None:
                elapsed = time.perf_counter() - t0
                return None, dict(ok=False, reason=meta.get("reason", "normal_selection_failed"), solve_time=elapsed,
                                  obstacle_index=int(j), face_meta=face_meta, detected=detected)
            A_rows.append(n)
            upper.append(float(ub))
            meta = meta | {"type": "obstacle", "index": int(j), "clearance": float(clr[j])}
            face_meta.append(meta)
            detected.append(meta)
            n_detected += 1

    A = np.stack(A_rows).astype(float)
    u = np.asarray(upper, dtype=float)
    F = A.shape[0]
    rel = traj_xy - c[None, :]
    H = traj_xy.shape[0] - 1
    A_ub = []
    b_ub = []
    # Level-set containment constraints: -coef*m_k <= -proj for positive proj.
    for i in range(1, H + 1):
        alpha = (1.0 - float(gamma)) ** i
        coef = max(1.0 - alpha, 1e-9)
        projs = A @ rel[i]
        for k in range(F):
            if projs[k] > 0.0:
                row = np.zeros(F, dtype=float)
                row[k] = -coef
                A_ub.append(row)
                b_ub.append(-float(projs[k]))
    # No extra coupling is needed: safety is captured by the upper support bounds.
    bounds = [(1e-5, float(max(1e-5, uk))) for uk in u]
    lp_t0 = time.perf_counter()
    res = linprog(c=-np.ones(F), A_ub=np.asarray(A_ub) if A_ub else None,
                  b_ub=np.asarray(b_ub) if b_ub else None, bounds=bounds, method="highs")
    lp_time = time.perf_counter() - lp_t0
    elapsed = time.perf_counter() - t0
    if not res.success:
        # Compute a useful minimal infeasibility diagnostic.
        reqs = np.array([required_margin_for_face(A[k], rel, gamma) for k in range(F)])
        deficit = reqs - u
        return None, dict(ok=False, reason="LP_infeasible", scipy_message=str(res.message), solve_time=elapsed, lp_time=lp_time,
                          n_faces=int(F), max_deficit=float(np.max(deficit)), n_detected=len(detected), face_meta=face_meta)
    m = np.asarray(res.x, dtype=float)
    b = A @ c + m
    poly = Poly(A=A, b=b, c=c.copy(), margins=np.maximum(m, 1e-6),
                meta=dict(kind="verifier", n_faces=int(F), n_base=K, sensing=float(sensing), n_detected=len(detected),
                          face_meta=face_meta, lp_objective=float(-res.fun)))
    ver = verify_traj_levelset(poly, traj_xy, gamma)
    ok = bool(ver["ok"])
    return poly, dict(ok=ok, reason="ok" if ok else "postcheck_failed", solve_time=elapsed, lp_time=lp_time,
                      n_faces=int(F), n_detected=len(detected), min_residual=ver["min_residual"], face_meta=face_meta)


class LocalDIMPPISampler:
    """Minimal local double-integrator MPPI sampler matching the relevant di_grid mechanics."""
    def __init__(self, horizon=10, dt=DT, num_samples=256, noise_sigma=(0.5, 0.5), u_min=(-2.0, -2.0), u_max=(2.0, 2.0),
                 gamma=0.5, temperature=0.3, sensing=2.0, n_base=16, predict_gain=0.4,
                 centroid_gain=0.2, centroid_smooth=0.0, centroid_eps=0.15,
                 sigma_volume_gain=0.5, sigma_aniso=2.0, sigma_max_mult=3.0,
                 control_weight=0.03, terminal_goal_weight=80.0, running_goal_weight=0.25,
                 soft_clearance_weight=25.0, smooth_weight=0.12, progress_weight=2.0,
                 debug_max_rollouts=80):
        self.horizon = int(horizon)
        self.dt = float(dt)
        self.num_samples = int(num_samples)
        self.noise_sigma = np.asarray(noise_sigma, dtype=np.float32)
        self.u_min = np.asarray(u_min, dtype=np.float32)
        self.u_max = np.asarray(u_max, dtype=np.float32)
        self.gamma = float(gamma)
        self.temperature = float(temperature)
        self.sensing = float(sensing)
        self.n_base = int(n_base)
        self.predict_gain = float(predict_gain)
        self.centroid_gain = float(centroid_gain)
        self.centroid_smooth = float(centroid_smooth)
        self.centroid_eps = float(centroid_eps)
        self.sigma_volume_gain = float(sigma_volume_gain)
        self.sigma_aniso = float(sigma_aniso)
        self.sigma_max_mult = float(sigma_max_mult)
        self.control_weight = float(control_weight)
        self.terminal_goal_weight = float(terminal_goal_weight)
        self.running_goal_weight = float(running_goal_weight)
        self.soft_clearance_weight = float(soft_clearance_weight)
        self.smooth_weight = float(smooth_weight)
        self.progress_weight = float(progress_weight)
        self.debug_max_rollouts = int(debug_max_rollouts)

    def step_np(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        nx = x.copy()
        dt = self.dt
        nx[:, 0] = x[:, 0] + dt * x[:, 2] + 0.5 * dt * dt * u[:, 0]
        nx[:, 1] = x[:, 1] + dt * x[:, 3] + 0.5 * dt * dt * u[:, 1]
        nx[:, 2] = x[:, 2] + dt * u[:, 0]
        nx[:, 3] = x[:, 3] + dt * u[:, 1]
        return nx

    def nominal_control(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        # Used only to score; sampling remains centered at zero plus optional centroid mode, as in di_grid's use_goal_nominal=False.
        to_goal = goal[:2] - state[:2]
        vel_err = -state[2:4]
        return np.clip(0.45 * to_goal + 0.8 * vel_err, self.u_min, self.u_max).astype(np.float32)

    def plan(self, state: np.ndarray, goal: np.ndarray, obs: np.ndarray, obs_vel: np.ndarray, seed: int = 0) -> Dict:
        t0 = time.perf_counter()
        rng = np.random.default_rng(int(seed))
        state = np.asarray(state, dtype=np.float32).reshape(4)
        goal = np.asarray(goal, dtype=np.float32).reshape(2)
        obs = np.asarray(obs, dtype=np.float32).reshape(-1, 3) if np.asarray(obs).size else np.zeros((0, 3), dtype=np.float32)
        obs_vel = np.asarray(obs_vel, dtype=np.float32).reshape(-1, 2) if np.asarray(obs_vel).size else np.zeros((0, 2), dtype=np.float32)

        poly = None
        sigma = self.noise_sigma.copy()
        mix_p = 0.0
        sample_mean = np.zeros(2, dtype=np.float32)
        centroid_pos = None
        centroid_dir = None
        size = None
        if obs.shape[0] > 0:
            poly = build_nominal_polytope(state[:2], obs, sensing=self.sensing, n_base=self.n_base,
                                          obstacle_velocities=obs_vel, robot_velocity=state[2:4],
                                          predict_gain=self.predict_gain, predict_tau=self.horizon * self.dt)
            size = float(np.min(poly.margins))
            trapped = max(0.0, (self.sensing - size) / (size + self.centroid_eps))
            if self.sigma_volume_gain > 0.0:
                sigma = sigma * min(1.0 + self.sigma_volume_gain * trapped, self.sigma_max_mult)
            centroid_pos = polygon_centroid(poly.A, poly.b, state[:2])
            d = centroid_pos - state[:2]
            dn = float(np.linalg.norm(d))
            if dn > 1e-6:
                centroid_dir = (d / dn).astype(np.float32)
                mix_p = min(self.centroid_gain * trapped, 1.0)
                sample_mean = (mix_p * float(self.u_max.max()) * centroid_dir).astype(np.float32)
            else:
                centroid_dir = np.zeros(2, dtype=np.float32)

        N, H = self.num_samples, self.horizon
        controls = rng.normal(loc=0.0, scale=sigma.reshape(1, 1, 2), size=(N, H, 2)).astype(np.float32)
        controls = controls + 0.0  # explicit cold nominal center
        nB = int(round(mix_p * N))
        if nB > 0 and centroid_dir is not None and np.linalg.norm(centroid_dir) > 1e-6:
            nA = N - nB
            u_target = float(self.u_max.max()) * centroid_dir.astype(np.float32)
            controls[nA:] += u_target.reshape(1, 1, 2)
            d_ctrl = centroid_dir.astype(np.float32)
            tangent = np.array([-d_ctrl[1], d_ctrl[0]], dtype=np.float32)
            noise = controls[nA:] - u_target.reshape(1, 1, 2)
            cn = (noise * d_ctrl.reshape(1, 1, 2)).sum(axis=-1, keepdims=True)
            ct = (noise * tangent.reshape(1, 1, 2)).sum(axis=-1, keepdims=True)
            controls[nA:] = u_target.reshape(1, 1, 2) + self.sigma_aniso * cn * d_ctrl.reshape(1, 1, 2) + ct * tangent.reshape(1, 1, 2)
        controls = np.clip(controls, self.u_min.reshape(1, 1, 2), self.u_max.reshape(1, 1, 2))

        x = np.repeat(state.reshape(1, 4), N, axis=0)
        state_seq = [x.copy()]
        infeasible = np.zeros(N, dtype=bool)
        min_h = np.full(N, np.inf, dtype=np.float64)
        costs = np.zeros(N, dtype=np.float64)
        raw_costs = np.zeros(N, dtype=np.float64)
        init_goal_dist = np.linalg.norm(x[:, :2] - goal.reshape(1, 2), axis=1)
        prev_action = np.zeros((N, 2), dtype=np.float32)
        for t in range(H):
            x_next = self.step_np(x, controls[:, t])
            if poly is not None:
                h_old = poly_H_points(poly, x[:, :2])
                h_new = poly_H_points(poly, x_next[:, :2])
                min_h = np.minimum(min_h, h_new)
                violation = h_new < (1.0 - self.gamma) * h_old - 1e-9
            else:
                violation = np.zeros(N, dtype=bool)
                min_h = np.minimum(min_h, np.ones(N))
            infeasible |= violation
            gd = np.linalg.norm(x_next[:, :2] - goal.reshape(1, 2), axis=1)
            goal_cost = self.running_goal_weight * gd**2
            effort = self.control_weight * np.sum(controls[:, t] ** 2, axis=1)
            smooth = self.smooth_weight * np.sum((controls[:, t] - prev_action) ** 2, axis=1)
            progress = -self.progress_weight * (init_goal_dist - gd)
            if obs.shape[0] > 0:
                obs_next = obs.copy()
                if obs_vel.shape[0] == obs.shape[0]:
                    obs_next[:, :2] += obs_vel * (self.dt * (t + 1))
                clearance = np.min(np.linalg.norm(obs_next[None, :, :2] - x_next[:, None, :2], axis=2) - obs_next[None, :, 2] - R_ROBOT, axis=1)
                soft_clear = self.soft_clearance_weight * np.maximum(0.0, -clearance) ** 2
            else:
                soft_clear = 0.0
            costs += goal_cost + effort + smooth + progress + soft_clear
            raw_costs = costs.copy()
            x = x_next
            prev_action = controls[:, t]
            state_seq.append(x.copy())
        terminal = np.linalg.norm(x[:, :2] - goal.reshape(1, 2), axis=1)
        costs += self.terminal_goal_weight * terminal**2
        raw_costs = costs.copy()
        masked_costs = costs.copy()
        masked_costs[infeasible] = np.inf
        if np.isinf(masked_costs).all():
            masked_costs = -min_h + 1e-3 * raw_costs
        best = int(np.argmin(masked_costs))
        temp = max(self.temperature, 1e-6)
        shifted = masked_costs - np.nanmin(masked_costs)
        w = np.exp(-shifted / temp)
        w[~np.isfinite(w)] = 0.0
        if w.sum() < 1e-12:
            action = controls[best, 0]
            u_avg_seq = controls[best]
        else:
            w = w / w.sum()
            action = np.sum(w[:, None] * controls[:, 0], axis=0)
            u_avg_seq = np.sum(w[:, None, None] * controls, axis=0)
        # Draw both accepted and rejected; keep up to debug_max_rollouts roughly balanced.
        all_states = np.stack(state_seq, axis=1)  # [N,H+1,4]
        feas = ~infeasible
        maxd = max(1, self.debug_max_rollouts)
        acc_idx = np.nonzero(feas)[0][:maxd // 2]
        rej_idx = np.nonzero(~feas)[0][:maxd - len(acc_idx)]
        draw_idx = np.concatenate([acc_idx, rej_idx])
        if draw_idx.size == 0:
            draw_idx = np.arange(min(maxd, N))
        # Choose selected trajectory: best feasible if any feasible, else safest best.
        selected_idx = best
        info = dict(
            gamma=self.gamma,
            solve_time=float(time.perf_counter() - t0),
            controls=controls,
            states=all_states,
            feasible=feas,
            best_idx=int(best),
            selected_idx=int(selected_idx),
            selected_feasible=bool(feas[selected_idx]),
            debug_rollouts=dict(states=all_states[draw_idx], feasible=feas[draw_idx]),
            num_barrier_violations=int(infeasible.sum()),
            infeasibility_rate=float(infeasible.mean()),
            n_acc=int(feas.sum()),
            n_rej=int(infeasible.sum()),
            first_controls=controls[:, 0].copy(),
            first_feasible=feas.copy(),
            mean_control=np.clip(action, self.u_min, self.u_max),
            sample_mean=sample_mean.copy(),
            sigma=sigma.copy(),
            mixture_p=float(mix_p),
            polytope=None if poly is None else poly.as_tuple(),
            poly_obj=poly,
            centroid_dir=centroid_dir,
            centroid_pos=centroid_pos,
            polytope_size=None if size is None else float(size),
        )
        return info


def draw_case_panel(ax, case: Dict, gamma: float, reveal: Optional[int] = None):
    state = case["state"]
    goal = case["goal"]
    obs = case["obs"]
    vel = case["obs_vel"]
    nominal = case["nominal_poly"]
    verifier = case["verifier_poly"]
    info = case["mppi"]
    selected = case["selected_traj"]
    p = state[:2]
    view_R = 2.6
    xl = (float(p[0] - view_R), float(p[0] + view_R))
    yl = (float(p[1] - view_R), float(p[1] + view_R))
    gx = np.linspace(*xl, 95)
    gy = np.linspace(*yl, 95)
    GX, GY = np.meshgrid(gx, gy)
    levels = sorted({round((1.0 - gamma) ** i, 4) for i in range(8)} | {0.0})
    if nominal is not None:
        Hh = h_grid(nominal.as_tuple(), GX, GY)
        ax.contourf(GX, GY, Hh, levels=levels + [1.0001], cmap="Blues", alpha=0.45, zorder=1)
        ax.contour(GX, GY, Hh, levels=[0.0], colors="#08306b", linewidths=1.0, zorder=3)
    if verifier is not None:
        Hv = h_grid(verifier.as_tuple(), GX, GY)
        # Green answer: one trajectory-specific verifier polytope.
        glv = [l for l in levels if l > 0.0]
        if glv:
            ax.contour(GX, GY, Hv, levels=glv, colors="#31a354", linewidths=0.35, alpha=0.55, zorder=4)
        ax.contour(GX, GY, Hv, levels=[0.0], colors="#006d2c", linewidths=1.6, zorder=5)
    for j, (ox, oy, rr) in enumerate(obs):
        ax.add_patch(Circle((ox, oy), rr, facecolor="#c8a2c8", alpha=0.55, edgecolor="#7b3294", lw=0.5, zorder=6))
        if j < len(vel) and np.linalg.norm(vel[j]) > 1e-5:
            ax.arrow(ox, oy, 0.25 * vel[j, 0], 0.25 * vel[j, 1], head_width=0.04, head_length=0.06,
                     fc="#7b3294", ec="#7b3294", alpha=0.6, lw=0.5, zorder=7, length_includes_head=True)
    dr = info["debug_rollouts"]
    trajs = dr["states"]
    feas = dr["feasible"]
    end = trajs.shape[1] if reveal is None else max(1, min(trajs.shape[1], int(reveal) + 1))
    # rejected under, accepted over, exactly di_grid ordering.
    for k in range(trajs.shape[0]):
        xy = trajs[k, :end, :2]
        if not feas[k]:
            ax.plot(xy[:, 0], xy[:, 1], "-", color="#d62728", lw=0.45, alpha=0.35, zorder=8)
            if end == trajs.shape[1]:
                ax.plot(xy[-1, 0], xy[-1, 1], "x", color="#d62728", ms=3, mew=0.7, zorder=9)
    for k in range(trajs.shape[0]):
        xy = trajs[k, :end, :2]
        if feas[k]:
            ax.plot(xy[:, 0], xy[:, 1], "-", color="#00a000", lw=0.65, alpha=0.75, zorder=10)
    sel_end = selected.shape[0] if reveal is None else max(1, min(selected.shape[0], int(reveal) + 1))
    xy = selected[:sel_end]
    ax.plot(xy[:, 0], xy[:, 1], "-", color="black", lw=1.8, alpha=0.95, zorder=12)
    ax.scatter([xy[-1, 0]], [xy[-1, 1]], s=14, c="black", zorder=13)
    if info.get("centroid_pos") is not None:
        cp = info["centroid_pos"]
        ax.annotate("", xy=(cp[0], cp[1]), xytext=(p[0], p[1]),
                    arrowprops=dict(arrowstyle="-|>", color="#ff7f00", lw=1.2), zorder=14)
    ax.scatter([p[0]], [p[1]], s=38, c="#00a000", edgecolor="k", zorder=15)
    if xl[0] <= goal[0] <= xl[1] and yl[0] <= goal[1] <= yl[1]:
        ax.scatter([goal[0]], [goal[1]], marker="*", s=90, c="gold", edgecolor="k", zorder=15)
    cert = case["cert"]
    title = f"{case['dataset']} ep{case['ep']} t{case['t']}  acc {info['n_acc']}/{info['n_rej']}\n"
    if cert["ok"]:
        title += f"cert {1000*cert['solve_time']:.1f} ms  res {cert.get('min_residual',0):+.2e}"
    else:
        title += f"FAIL {cert.get('reason','?')}"
    ax.set_title(title, fontsize=7.2)
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


def draw_grid(cases: List[Dict], out_png: str, gamma: float, title: str):
    fig, axes = plt.subplots(5, 5, figsize=(17, 17), squeeze=False)
    for ax, case in zip(axes.ravel(), cases):
        draw_case_panel(ax, case, gamma=gamma)
    fig.suptitle(title + "\nblue=current nominal SafeMPPI polytope, green=trajectory-specific Pillar-3 verifier polytope, black=queried 10-step local rollout", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(out_png, dpi=145)
    plt.close(fig)


def draw_gif(cases: List[Dict], out_gif: str, gamma: float, title: str):
    fig, axes = plt.subplots(5, 5, figsize=(15, 15), squeeze=False)
    def draw(frame):
        for ax, case in zip(axes.ravel(), cases):
            ax.clear()
            draw_case_panel(ax, case, gamma=gamma, reveal=frame)
        fig.suptitle(title + f" · revealing local horizon step {frame}/10", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return []
    anim = FuncAnimation(fig, draw, frames=11, interval=200, blit=False)
    anim.save(out_gif, writer=PillowWriter(fps=5), dpi=75)
    plt.close(fig)


def run_dataset(dataset: str, data_dir: str, out_dir: str, *, gamma: float, sensing: float, n_base: int, num_samples: int, make_gif: bool, seed_base: int) -> Dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ego, obs_list, ego_path, obs_path = load_eval80(dataset, data_dir)
    selected = pick_cases(ego, obs_list, n_cases=25, sensing=sensing)
    sampler = LocalDIMPPISampler(horizon=10, dt=DT, num_samples=num_samples, gamma=gamma,
                                 sensing=sensing, n_base=n_base, predict_gain=0.4,
                                 centroid_gain=0.2, sigma_volume_gain=0.5, sigma_aniso=2.0,
                                 temperature=0.3, debug_max_rollouts=80)
    cases = []
    for idx, (ep, t, pick_info) in enumerate(selected):
        st, goal, ob, vl = get_scene_frame(ego, obs_list, ep, t)
        mppi = sampler.plan(st, goal, ob, vl, seed=seed_base + ep * 1000 + t)
        # Query one actual 10-step local MPPI forward pass. Prefer the selected/best rollout.
        sel_idx = int(mppi["selected_idx"])
        selected_traj = mppi["states"][sel_idx, :, :2].copy()
        nominal_poly = mppi["poly_obj"]
        verifier_poly, cert = solve_verifier_polytope_lp(st[:2], selected_traj, ob, vl, gamma=gamma,
                                                          sensing=sensing, n_base=n_base,
                                                          robot_velocity=st[2:4], predict_gain=0.4,
                                                          predict_tau=10 * DT, n_angle_candidates=121)
        # Also check nominal certificate for reference.
        nominal_check = verify_traj_levelset(nominal_poly, selected_traj, gamma) if nominal_poly is not None else dict(ok=True, min_residual=np.nan)
        case = dict(dataset=dataset, ep=int(ep), t=int(t), pick_info=pick_info,
                    state=st, goal=goal, obs=ob, obs_vel=vl, mppi=mppi,
                    selected_traj=selected_traj, selected_feasible=bool(mppi["selected_feasible"]),
                    nominal_poly=nominal_poly, nominal_check=nominal_check,
                    verifier_poly=verifier_poly, cert=cert)
        cases.append(case)
        print(f"{dataset} {idx+1:02d}/25 ep={ep:03d} t={t:02d} obs={ob.shape[0]:02d} near={pick_info['nnear']} "
              f"acc={mppi['n_acc']}/{mppi['n_rej']} nominal_ok={nominal_check['ok']} cert={cert['ok']} "
              f"tver={1000*cert.get('solve_time',0):.2f}ms", flush=True)

    ok = [c for c in cases if c["cert"].get("ok")]
    fail = [c for c in cases if not c["cert"].get("ok")]
    times = np.array([c["cert"]["solve_time"] for c in ok], dtype=float)
    lp_times = np.array([c["cert"].get("lp_time", np.nan) for c in ok], dtype=float)
    mppi_times = np.array([c["mppi"]["solve_time"] for c in cases], dtype=float)
    n_acc = np.array([c["mppi"]["n_acc"] for c in cases], dtype=float)
    n_rej = np.array([c["mppi"]["n_rej"] for c in cases], dtype=float)
    nominal_ok = np.array([bool(c["nominal_check"].get("ok")) for c in cases], dtype=bool)
    summary = dict(dataset=dataset, ego_path=ego_path, obs_path=obs_path, n_cases=len(cases), gamma=float(gamma), sensing=float(sensing),
                   horizon=10, n_base=int(n_base), num_samples=int(num_samples),
                   verifier_success=int(len(ok)), verifier_failure=int(len(fail)),
                   verifier_success_rate=float(len(ok) / len(cases)),
                   verifier_mean_time_ms=float(1000 * times.mean()) if len(times) else None,
                   verifier_median_time_ms=float(1000 * np.median(times)) if len(times) else None,
                   verifier_max_time_ms=float(1000 * times.max()) if len(times) else None,
                   verifier_mean_lp_time_ms=float(1000 * np.nanmean(lp_times)) if len(lp_times) else None,
                   mppi_mean_time_ms=float(1000 * mppi_times.mean()),
                   mean_accepted=float(n_acc.mean()), mean_rejected=float(n_rej.mean()),
                   nominal_certified_count=int(nominal_ok.sum()),
                   failures=[dict(ep=c["ep"], t=c["t"], reason=c["cert"].get("reason")) for c in fail])

    # Save serializable case data.
    def ser_case(c):
        return dict(dataset=c["dataset"], ep=c["ep"], t=c["t"], pick_info=c["pick_info"],
                    state=np.asarray(c["state"]).tolist(), goal=np.asarray(c["goal"]).tolist(),
                    obs=np.asarray(c["obs"]).tolist(), obs_vel=np.asarray(c["obs_vel"]).tolist(),
                    selected_traj=np.asarray(c["selected_traj"]).tolist(),
                    selected_feasible=c["selected_feasible"],
                    mppi=dict(n_acc=c["mppi"]["n_acc"], n_rej=c["mppi"]["n_rej"], solve_time=c["mppi"]["solve_time"],
                              mixture_p=c["mppi"]["mixture_p"], polytope_size=c["mppi"]["polytope_size"],
                              sigma=np.asarray(c["mppi"]["sigma"]).tolist()),
                    nominal_check=c["nominal_check"], cert={k: v for k, v in c["cert"].items() if k != "face_meta"},
                    verifier_poly=None if c["verifier_poly"] is None else dict(A=c["verifier_poly"].A.tolist(), b=c["verifier_poly"].b.tolist(),
                                                                                 c=c["verifier_poly"].c.tolist(), margins=c["verifier_poly"].margins.tolist(),
                                                                                 meta=c["verifier_poly"].meta))
    result = dict(summary=summary, cases=[ser_case(c) for c in cases])
    json_path = out / f"pillar3_corrected_{dataset}_results.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    png_path = out / f"pillar3_corrected_{dataset}_5x5.png"
    draw_grid(cases, str(png_path), gamma=gamma,
              title=f"Corrected Pillar-3 verifier on eval80 {dataset.upper()} moving pedestrians · H=10, sensing={sensing}, K={n_base}")
    gif_path = None
    if make_gif:
        gif_path = out / f"pillar3_corrected_{dataset}_5x5.gif"
        draw_gif(cases, str(gif_path), gamma=gamma,
                 title=f"Corrected Pillar-3 verifier on eval80 {dataset.upper()} moving pedestrians")
    # Lightweight CSV.
    csv_path = out / f"pillar3_corrected_{dataset}_summary.csv"
    with open(csv_path, "w") as f:
        f.write("dataset,ep,t,n_obs,n_near,acc,rej,selected_feasible,nominal_ok,cert_ok,cert_time_ms,lp_time_ms,min_residual,reason\n")
        for c in cases:
            f.write(f"{dataset},{c['ep']},{c['t']},{len(c['obs'])},{c['pick_info']['nnear']},{c['mppi']['n_acc']},{c['mppi']['n_rej']},"
                    f"{int(c['selected_feasible'])},{int(c['nominal_check'].get('ok', False))},{int(c['cert'].get('ok', False))},"
                    f"{1000*c['cert'].get('solve_time', float('nan')):.6f},{1000*c['cert'].get('lp_time', float('nan')):.6f},"
                    f"{c['cert'].get('min_residual', float('nan'))},{c['cert'].get('reason','')}\n")

    md_path = out / f"pillar3_corrected_{dataset}_report.md"
    with open(md_path, "w") as f:
        f.write(f"# Corrected Pillar-3 verifier on eval80 {dataset.upper()}\n\n")
        f.write("## What was corrected\n\n")
        f.write("The local query trajectory is one 10-step double-integrator MPPI forward pass. The previous global-path interpretation was wrong. ")
        f.write("The plot now uses one fixed robot-centered verifier polytope per selected trajectory, rendered with the same normalized `H_grid` level-set convention as `di_grid.py`.\n\n")
        f.write("## Summary\n\n")
        for k, v in summary.items():
            f.write(f"- `{k}`: {v}\n")
        f.write("\n## Optimization problem\n\n")
        f.write("For a rollout `q_i`, `i=0..H`, with `q_0=c`, the verifier solves for one polytope `P={x: A x <= b}`. ")
        f.write("Write `m_k=b_k-a_k^T c`. The normalized level-set is `H_P(x)=min_k (b_k-a_k^T x)/m_k`. ")
        f.write("The certification condition is `H_P(q_i) >= alpha_i`, `alpha_i=(1-gamma)^i`, which becomes the linear constraints\n\n")
        f.write("```text\na_k^T(q_i-c) <= (1-alpha_i) m_k,  i=1..H, k=1..F.\n```\n\n")
        f.write("Base faces are bounded by the sensing inner K-gon: `m_k <= sensing*cos(pi/K)`. Obstacle faces use tangent support bounds ")
        f.write("`m_k <= n^T(o-c)-r-kappa*tau*max(0,n^T(v_robot-v_obs))`. For each sensed obstacle, a unit support normal `n` is selected by a fast 1-D search over the separating cone to maximize slack against the rollout. ")
        f.write("With normals fixed, the LP is\n\n")
        f.write("```text\nmaximize    sum_k m_k\nsubject to  1e-5 <= m_k <= upper_k\n            a_k^T(q_i-c) <= (1-alpha_i)m_k.\n```\n\n")
        f.write("This keeps the same `A,b,c,margins` representation used by the existing renderer, but the obstacle support normals are trajectory-specific, so it is less conservative than always using the radial nominal face.\n")
    return dict(summary=summary, json=str(json_path), png=str(png_path), gif=str(gif_path) if gif_path else None, csv=str(csv_path), md=str(md_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ucy", choices=["ucy", "sdd", "both"])
    ap.add_argument("--data-dir", default="/mnt/data")
    ap.add_argument("--out-dir", default="/mnt/data/pillar3_corrected_outputs")
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--sensing", type=float, default=2.0)
    ap.add_argument("--n-base", type=int, default=16)
    ap.add_argument("--num-samples", type=int, default=256)
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--seed-base", type=int, default=20260630)
    args = ap.parse_args()
    datasets = ["ucy", "sdd"] if args.dataset == "both" else [args.dataset]
    outputs = []
    for ds in datasets:
        outputs.append(run_dataset(ds, args.data_dir, args.out_dir, gamma=args.gamma, sensing=args.sensing,
                                   n_base=args.n_base, num_samples=args.num_samples, make_gif=not args.no_gif,
                                   seed_base=args.seed_base))
    # Combined summary and zip.
    out = Path(args.out_dir)
    combined = {"runs": outputs}
    comb_path = out / "pillar3_corrected_combined_summary.json"
    with open(comb_path, "w") as f:
        json.dump(combined, f, indent=2)
    zip_path = out / "pillar3_corrected_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(__file__, arcname="pillar3_corrected_eval.py")
        for run in outputs:
            for key in ("json", "png", "gif", "csv", "md"):
                p = run.get(key)
                if p and os.path.exists(p):
                    z.write(p, arcname=os.path.basename(p))
        z.write(comb_path, arcname=os.path.basename(comb_path))
    print(json.dumps({"outputs": outputs, "combined_summary": str(comb_path), "zip": str(zip_path)}, indent=2))


if __name__ == "__main__":
    main()
