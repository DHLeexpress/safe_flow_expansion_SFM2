"""Shared featurization for the 5x5 grid FM policy (WORLD / axis-aligned frame).

The grid task is axis-aligned and its coverage is defined by monotone right/up moves, so — unlike the
07-01 goal-aligned frame (which would make the relative-goal vector degenerate [dist,0]) — everything here
is expressed in the fixed WORLD frame:
  - relative-goal vector  (goal - pos) / R_GOAL          in R^2   (points up-right toward (5,5))
  - unitless velocity     v / V_SCALE                     in R^2
  - past control history  a_{t-K..t-1} / U_MAX            in R^{K x 2}   (fed to the GRU)
  - target window / output U                              raw world controls in R^{H_pred x 2}
An axis-aligned robot-centered polar occupancy/polytope grid [3,16,12] reuses the EXACT SafeMPPI nominal
polytope (`polytope_HP` -> `build_polytope_v2`), so the FM 'sees' what the expert saw.
"""
from __future__ import annotations

import numpy as np

import _paths  # noqa: F401
from polar_grid import polytope_HP           # exact SafeMPPI nominal polytope + H_P callable

GRID_M = 5.0
R_GOAL = 5.0          # relative-goal normalization (grid side) -> relgoal components ~[0,1]
V_SCALE = 2.0         # velocity normalization
U_MAX = 1.0           # control normalization (grid u_max)
K_HIST = 16           # GRU history length (past executed controls)
H_PRED = 10           # predicted window length
R_SENSE = 2.5         # polar-grid spatial extent
SENSING = 2.0         # polytope sensing range (matches SafeMPPI barrier_activation_radius=2.0)
N_THETA, N_R = 16, 12


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x, dtype=float)


def axis_polar_points(c, R=R_SENSE, n_theta=N_THETA, n_r=N_R):
    """Robot-centered, AXIS-ALIGNED polar grid points [n_theta, n_r, 2] (e_g=[1,0], e_lat=[0,1])."""
    c = np.asarray(c, dtype=float)[:2]
    theta = -np.pi + (np.arange(n_theta) + 0.5) * 2 * np.pi / n_theta
    r = (np.arange(n_r) + 0.5) * R / n_r
    dirs = np.stack([np.cos(theta), np.sin(theta)], axis=-1)                # [n_theta,2]
    pts = c[None, None, :] + r[None, :, None] * dirs[:, None, :]           # [n_theta,n_r,2]
    return pts


def axis_grid(c, obstacles, r_robot=0.0, R=R_SENSE, sensing=SENSING, n_theta=N_THETA, n_r=N_R):
    """[3,16,12] float32: ch0 occupancy, ch1 nominal-polytope mask, ch2 clipped H_P — axis-aligned."""
    pts = axis_polar_points(c, R, n_theta, n_r)
    flat = pts.reshape(-1, 2)
    obs = _np(obstacles)
    if obs.size:
        d = np.linalg.norm(flat[:, None, :] - obs[None, :, :2], axis=2) - (obs[None, :, 2] + r_robot)
        occ = (d.min(1) < 0).astype(np.float32)
    else:
        occ = np.zeros(len(flat), np.float32)
    HP, _ = polytope_HP(c, obs, sensing=sensing, n_base=n_theta)
    hp = HP(flat)
    mask = (hp >= 0).astype(np.float32)
    hclip = np.clip(hp, -1.0, 1.0).astype(np.float32)
    return np.stack([occ, mask, hclip], 0).reshape(3, n_theta, n_r).astype(np.float32)


def low5(state, goal, gamma):
    """[relgoal_x, relgoal_y, v_x/vs, v_y/vs, gamma] (world frame), float32."""
    s = np.asarray(state, dtype=float)
    p, v = s[:2], s[2:4]
    g = np.asarray(goal, dtype=float)[:2]
    rg = (g - p) / R_GOAL
    return np.array([rg[0], rg[1], v[0] / V_SCALE, v[1] / V_SCALE, float(gamma)], dtype=np.float32)


def hist_pad(ctrl_hist, K=K_HIST):
    """Past executed controls -> front-zero-padded, u_max-normalized [K,2] float32 (recent last)."""
    ch = np.asarray(ctrl_hist, dtype=float).reshape(-1, 2)
    ch = ch[-K:] / U_MAX
    if len(ch) < K:
        ch = np.concatenate([np.zeros((K - len(ch), 2)), ch], 0)
    return ch.astype(np.float32)


def featurize(state, goal, gamma, ctrl_hist, obstacles, r_robot=0.0, K=K_HIST):
    """One conditioning record: (grid[3,16,12], low5[5], hist[K,2]) all float32, world frame."""
    return (axis_grid(state[:2], obstacles, r_robot),
            low5(state, goal, gamma),
            hist_pad(ctrl_hist, K))


if __name__ == "__main__":
    import grid_scene as GS
    env = GS.make_grid()
    obs = env.obstacles.numpy()
    g, l, h = featurize([0.4, 0.3, 0.5, 0.2], env.goal.numpy(), 0.5, np.random.randn(20, 2) * 0.3, obs)
    print("grid", g.shape, "occ%", round(float(g[0].mean()), 3), "mask%", round(float(g[1].mean()), 3),
          "HP", round(float(g[2].min()), 2), round(float(g[2].max()), 2))
    print("low5", l.round(3), "| hist", h.shape, "recent", h[-1].round(3))
