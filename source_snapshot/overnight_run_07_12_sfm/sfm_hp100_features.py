"""Faithful current-tangent high-resolution :math:`H_P` features.

The raster resolution and the nominal-polytope geometry are deliberately
independent: the policy observes 32 angular samples, while SafeMPPI continues
to construct its nominal outer boundary with exactly 16 base faces.
"""
from __future__ import annotations

import numpy as np

import _paths  # noqa: F401
from cfm_mppi.safegpc_adapter.polytope_v2 import build_polytope_v2
from sfm_hp100_dynamics import DT, U_MAX, V_MAX

HP100_SHAPE = (32, 100)
R_SENSE = 2.0
POLYTOPE_N_BASE = 16
K_HIST = 16
R_GOAL = 5.0
PREDICT_GAIN = 0.0
PREDICT_TAU = 10 * DT


def _numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def hp100_frame(
    robot_xy,
    obstacles,
    *,
    sensing=R_SENSE,
    n_base=POLYTOPE_N_BASE,
    obstacle_velocities=None,
    robot_velocity=None,
    predict_gain=PREDICT_GAIN,
    predict_tau=PREDICT_TAU,
    return_geometry=False,
):
    """Return the clipped nominal-:math:`H_P` raster ``[32,100]``.

    The grid is axis-aligned and robot-centered.  Its radial cell centers are
    spaced every 0.02 m over the two-metre sensing disk.  ``n_base`` controls
    only the nominal-polytope outer boundary; it never changes the 32-angle
    observation raster.  Passing the live robot/pedestrian velocities exactly
    is accepted for interface compatibility, but the locked feature contract
    uses ``predict_gain=0``: pedestrian faces are tangent to their current
    positions and never retreat from relative velocity.
    """
    center = np.asarray(_numpy(robot_xy), dtype=np.float64).reshape(-1)[:2]
    obs = np.asarray(_numpy(obstacles), dtype=np.float64)
    obs = obs.reshape(-1, 3) if obs.size else np.zeros((0, 3), dtype=np.float64)
    sensing = float(sensing)
    n_base = int(n_base)
    if not np.isfinite(sensing) or sensing <= 0.0:
        raise ValueError(f"sensing must be positive, got {sensing}")
    if n_base < 4:
        raise ValueError(f"n_base must be at least four, got {n_base}")

    theta = -np.pi + (np.arange(HP100_SHAPE[0]) + 0.5) * (
        2.0 * np.pi / HP100_SHAPE[0]
    )
    radius = (np.arange(HP100_SHAPE[1]) + 0.5) * sensing / HP100_SHAPE[1]
    directions = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    points = center[None, None] + directions[:, None] * radius[None, :, None]

    polytope, _ = build_polytope_v2(
        center,
        obs,
        sensing_range=sensing,
        n_base=n_base,
        margin=0.0,
        obstacle_velocities=obstacle_velocities,
        robot_velocity=robot_velocity,
        predict_gain=float(predict_gain),
        predict_tau=float(predict_tau),
    )
    A = _numpy(polytope.A).astype(np.float64, copy=False)
    b = _numpy(polytope.b).astype(np.float64, copy=False)
    # The float32 geometry below is the canonical feature contract.  It can be
    # recomputed from the stored state/pedestrians and pinned source.  Build the
    # raster from exactly these values so even very narrow faces remain
    # bitwise reproducible; independently compare them with the planner tuple
    # in the dataset collector.
    stored_A = A.astype(np.float32, copy=False)
    stored_b = b.astype(np.float32, copy=False)
    stored_ref = np.asarray(_numpy(polytope.ref), dtype=np.float32)
    raw_margins = (
        stored_b.astype(np.float64)
        - stored_A.astype(np.float64) @ stored_ref.astype(np.float64)
    )
    if np.any(raw_margins <= 0.0):
        raise RuntimeError("nominal current-tangent polytope does not contain its robot reference")
    stored_margins = np.maximum(raw_margins, 1.0e-3).astype(np.float32)
    flat = points.reshape(-1, 2)
    hp = (
        (stored_b.astype(np.float64)[None]
         - flat @ stored_A.astype(np.float64).T)
        / stored_margins.astype(np.float64)[None]
    ).min(axis=1)
    frame = np.clip(hp, -1.0, 1.0).reshape(HP100_SHAPE).astype(np.float32)
    if not return_geometry:
        return frame
    geometry = dict(
        A=stored_A,
        b=stored_b,
        ref=stored_ref,
        margins=stored_margins,
    )
    return frame, geometry


def low5(state, goal, gamma, *, v_max=V_MAX, goal_scale=R_GOAL):
    """World-frame relative goal, clipped normalized velocity, and gamma."""
    state = np.asarray(_numpy(state), dtype=np.float64).reshape(-1)
    goal = np.asarray(_numpy(goal), dtype=np.float64).reshape(-1)
    v_max = float(v_max)
    goal_scale = float(goal_scale)
    if v_max <= 0.0 or goal_scale <= 0.0:
        raise ValueError("v_max and goal_scale must be positive")
    rel_goal = (goal[:2] - state[:2]) / goal_scale
    velocity = np.clip(state[2:4] / v_max, -1.0, 1.0)
    return np.asarray(
        [rel_goal[0], rel_goal[1], velocity[0], velocity[1], float(gamma)],
        dtype=np.float32,
    )


def hist_pad(control_history, K=K_HIST, *, u_max=U_MAX):
    """Front-zero-pad the latest controls and normalize them into ``[-1,1]``."""
    K = int(K)
    u_max = float(u_max)
    if K <= 0 or u_max <= 0.0:
        raise ValueError("K and u_max must be positive")
    history = np.asarray(_numpy(control_history), dtype=np.float64).reshape(-1, 2)
    history = np.clip(history[-K:] / u_max, -1.0, 1.0)
    if len(history) < K:
        history = np.concatenate([np.zeros((K - len(history), 2)), history], axis=0)
    return history.astype(np.float32)


def contract():
    """JSON-native observation contract pinned by data and checkpoints."""
    return dict(
        name="current_tangent_nominal_hp100_v2",
        temporal_frames=10,
        frame_shape=list(HP100_SHAPE),
        tensor_shape=[10, *HP100_SHAPE],
        angular_bins=HP100_SHAPE[0],
        radial_bins=HP100_SHAPE[1],
        sensing_radius=float(R_SENSE),
        radial_cell_width=float(R_SENSE / HP100_SHAPE[1]),
        raster_alignment="axis-aligned robot-centered cell centers",
        nominal_polytope_n_base=int(POLYTOPE_N_BASE),
        predict_gain=float(PREDICT_GAIN),
        predict_tau=float(PREDICT_TAU),
        pedestrian_velocity_in_geometry=False,
        face_definition="current-position tangent; no predictive retreat",
        value="clip(min_k (b_k-a_k^T x)/(b_k-a_k^T robot), -1, 1)",
        history_order="newest-to-oldest; pre-episode slots repeat first frame",
        radial_pooling="none",
        action_normalization=float(U_MAX),
        velocity_normalization=float(V_MAX),
    )
