"""Canonical paper-faithful full-H verifier for every B1 query.

GREEN certification uses the current-time pedestrian disks inside the fixed
finite sensing radius.  For each sensed disk and each of the 16 artificial
outer-boundary disks, the positive max-margin SOCP block from the paper is
solved exactly in two-dimensional angle space.  Pedestrian prediction is used
separately for the physical collision check; it does not enlarge or move the
GREEN verifier geometry.

Visualization imports this module, so rendered certificates and training
labels cannot silently use different solvers.
"""
from __future__ import annotations

import math
import numpy as np

import _paths  # noqa: F401
import verifier_polytope as VP
from demo_verifier_polytope import solve_face_interval as solve_paper_face_interval
import sfm_scene as SS


ARTIFICIAL_FACES = 16
ANGLE_TOL = 2.0e-10


def verifier_manifest():
    return dict(
        solver="paper_static_exact_2d_angular_interval_socp", angular_grid=False,
        K_artificial=ARTIFICIAL_FACES, horizon=10, rho_art=0.16,
        m_min=1.0e-4, sensing_radius=float(SS.R_SENSE),
        effective_radius="fixed sensing_radius",
        sensed_obstacles="current-time centers only",
        nominal_relation="nominal radial face is feasible whenever its contraction gate holds",
        pedestrian_prediction="constant velocity over t=0..H for collision check only",
    )


def predict_pedestrians(ped_xy, ped_vel, H=10, dt=SS.DT):
    xy = np.asarray(ped_xy, np.float32).reshape(-1, 2)
    velocity = np.asarray(ped_vel, np.float32).reshape(-1, 2)
    if xy.shape != velocity.shape:
        raise ValueError("pedestrian positions and velocities do not align")
    time = np.arange(int(H) + 1, dtype=np.float32)[:, None, None] * float(dt)
    return xy[None] + time * velocity[None]


def rollout_positions(state, controls, dt=SS.DT):
    current = np.asarray(state, np.float32).reshape(4).copy()
    positions = [current[:2].copy()]
    for action in np.asarray(controls, np.float32).reshape(-1, 2):
        current[:2] += float(dt) * current[2:4] + 0.5 * float(dt) ** 2 * action
        current[2:4] += float(dt) * action
        positions.append(current[:2].copy())
    return np.asarray(positions, np.float32)


def taskspace_ok(segment, lo=SS.TASK_LO, hi=SS.TASK_HI):
    value = np.asarray(segment, float)
    return bool(value.ndim == 2 and value.shape[1] == 2 and np.isfinite(value).all()
                and (value >= float(lo)).all() and (value <= float(hi)).all())


def collision_free_time_indexed(segment, pedestrians, radius=SS.R_PED):
    robot = np.asarray(segment, float)
    peds = np.asarray(pedestrians, float)
    if len(robot) != len(peds):
        raise ValueError("robot/pedestrian horizon mismatch")
    if peds.shape[1] == 0:
        return True
    distance = np.linalg.norm(robot[:, None, :] - peds, axis=2)
    return bool(float(distance.min()) >= float(radius) - 1.0e-9)


def _wrap(theta):
    return float(theta) % (2.0 * math.pi)


def _angular_constraint(vector, threshold):
    """Closed arc endpoints for ``unit(theta)^T vector >= threshold``."""
    vector = np.asarray(vector, dtype=float).reshape(2)
    norm = float(np.linalg.norm(vector))
    threshold = float(threshold)
    if norm <= ANGLE_TOL:
        return () if threshold <= ANGLE_TOL else None
    if threshold > norm + ANGLE_TOL:
        return None
    if threshold <= -norm + ANGLE_TOL:
        return ()
    halfwidth = math.acos(float(np.clip(threshold / norm, -1.0, 1.0)))
    center = math.atan2(float(vector[1]), float(vector[0]))
    return _wrap(center - halfwidth), _wrap(center + halfwidth)


def _is_feasible(theta, inequalities, tol=2.0e-9):
    normal = np.array([math.cos(theta), math.sin(theta)], dtype=float)
    return all(float(normal @ vector) >= float(threshold) - float(tol)
               for vector, threshold in inequalities)


def solve_static_face(robot_centered, obstacle_centered, radius, beta, label,
                      *, m_min=1.0e-4, kind="real"):
    """Solve one positive max-margin paper SOCP block for a static disk.

    The exact angular solution is equivalent to

    ``max m`` subject to ``a.T q_t <= beta_t m``,
    ``r ||a|| <= a.T d - m``, ``||a|| <= 1``, and ``m >= m_min``.
    With a positive objective an optimum has unit normal, so the disk
    constraint is tight and ``m = a.T d - r``.
    """
    robot = np.asarray(robot_centered, dtype=float).reshape(-1, 2)
    obstacle = np.asarray(obstacle_centered, dtype=float).reshape(2)
    beta = np.asarray(beta, dtype=float).reshape(-1)
    if len(robot) != len(beta):
        raise ValueError("face trajectory and beta horizons do not align")
    if len(robot) < 2 or np.any(beta[1:] <= 0.0):
        raise ValueError("face needs at least one positive-beta horizon")
    return solve_paper_face_interval(
        obstacle, float(radius), robot, beta, coefficient=1.0,
        kind=kind, label=label, m_min=float(m_min),
    )


def solve_moving_face(robot_centered, pedestrian_centered, radius, beta, label,
                      *, m_min=1.0e-4, kind="real"):
    """Compatibility wrapper using only the current-time obstacle center."""
    centers = np.asarray(pedestrian_centered, dtype=float).reshape(-1, 2)
    if not len(centers):
        raise ValueError("pedestrian_centered is empty")
    return solve_static_face(
        robot_centered, centers[0], radius, beta, label,
        m_min=m_min, kind=kind,
    )


def certify_moving_window(segment, pedestrians, gamma, *, K=ARTIFICIAL_FACES,
                          rho_art=0.16, m_min=1.0e-4):
    if int(K) != ARTIFICIAL_FACES:
        raise ValueError(f"faithful B1 verifier requires K={ARTIFICIAL_FACES}")
    robot = np.asarray(segment, float)
    peds = np.asarray(pedestrians, float)
    if robot.ndim != 2 or robot.shape[1] != 2 or peds.ndim != 3 or peds.shape[2] != 2:
        raise ValueError("invalid moving-window shapes")
    if len(robot) != len(peds):
        raise ValueError("moving-window horizons do not align")
    center = robot[0]
    robot_c = robot - center
    radius = float(SS.R_SENSE)
    if len(robot) == 1:
        # This generic helper also audits the zero-transition tail of an
        # already executed trajectory. Queried B1 plans never take this path:
        # verify_query below requires and certifies all H=10 transitions.
        return True, [], dict(
            solver="paper_static_exact_2d_angular_interval_socp", angular_grid=False,
            slack=float("inf"), worst_t=0, R_eff=float(radius),
            n_real=0, n_real_feasible=0, n_artificial=0,
            n_artificial_feasible=0, K_artificial=ARTIFICIAL_FACES,
            empty_executed_tail=True,
        )
    alpha = (1.0 - float(gamma)) ** np.arange(len(robot), dtype=float)
    beta = 1.0 - alpha
    faces = []
    for index in range(peds.shape[1]):
        ped_current_c = peds[0, index] - center
        if float(np.linalg.norm(ped_current_c) - SS.R_PED) <= radius:
            faces.append(solve_static_face(
                robot_c, ped_current_c, SS.R_PED, beta, f"ped{index}",
                m_min=m_min,
            ))
    for index, (x, y, obstacle_radius) in enumerate(
            VP.artificial_obstacles(radius, ARTIFICIAL_FACES, float(rho_art))):
        faces.append(solve_static_face(
            robot_c, np.array([x, y]), obstacle_radius, beta, f"art{index}",
            m_min=m_min, kind="artificial",
        ))
    ok, slack, worst_t = VP.check_certificate(faces, robot_c, alpha, include_start=False)
    real = [face for face in faces if face.kind == "real"]
    artificial = [face for face in faces if face.kind == "artificial"]
    return bool(ok), faces, dict(
        solver="paper_static_exact_2d_angular_interval_socp", angular_grid=False,
        slack=float(slack), worst_t=int(worst_t), R_eff=float(radius),
        sensing_radius=float(radius), rho_art=float(rho_art),
        sensed_obstacles="current-time centers only",
        n_real=len(real), n_real_feasible=sum(bool(face.feasible) for face in real),
        n_artificial=len(artificial),
        n_artificial_feasible=sum(bool(face.feasible) for face in artificial),
        K_artificial=ARTIFICIAL_FACES,
    )


def _verify_window(state, controls, ped_xy, ped_vel, gamma):
    robot = rollout_positions(state, controls)
    pedestrian = predict_pedestrians(ped_xy, ped_vel, H=len(controls))
    task = taskspace_ok(robot)
    collision = collision_free_time_indexed(robot, pedestrian)
    certificate, faces, diagnostics = certify_moving_window(
        robot, pedestrian, gamma,
    )
    y = bool(task and collision and certificate)
    return dict(
        resolved=True, error=None, y=int(y), taskspace=bool(task),
        collision_free=bool(collision), certificate=bool(certificate),
        segment=robot, pedestrian_prediction=pedestrian, faces=faces,
        diagnostics=diagnostics,
    )


def verify_executed_window(state, controls, ped_xy, ped_vel, gamma):
    """Certify one terminal-truncated executed window of length 1 through 10.

    This API is for offline trajectory evaluation.  A short window is complete
    relative to the executed trajectory; it is not a B1 terminal-prefix query
    and therefore deliberately has no ``full_h`` or ``train_eligible`` field.
    """
    try:
        controls = np.asarray(controls, np.float32).reshape(-1, 2)
        if not 1 <= len(controls) <= 10:
            raise ValueError("executed-window verifier requires 1 <= H_t <= 10")
        result = _verify_window(state, controls, ped_xy, ped_vel, gamma)
        result.update(window_horizon=len(controls))
        return result
    except Exception as error:
        return dict(resolved=False, error=f"{type(error).__name__}: {error}")


def verify_query(state, controls, ped_xy, ped_vel, gamma):
    """Certify every queried plan over all H=10 transitions.

    Goal reach is a closed-loop episode trigger after the selected first action;
    it never truncates a candidate window or changes its verifier label.
    """
    try:
        controls = np.asarray(controls, np.float32).reshape(-1, 2)
        if len(controls) != 10:
            raise ValueError("B1 verifier requires H=10")
        result = _verify_window(state, controls, ped_xy, ped_vel, gamma)
        result.update(
            full_h=True, terminal_step=len(controls),
            train_eligible=bool(result["y"]),
        )
        return result
    except Exception as error:  # worker boundary: a failed solver/query enters no store.
        return dict(resolved=False, error=f"{type(error).__name__}: {error}")


def verify_in_worker(payload):
    context_id, candidate_id, state, controls, ped_xy, ped_vel, gamma = payload
    result = verify_query(state, controls, ped_xy, ped_vel, gamma)
    return int(context_id), int(candidate_id), result
