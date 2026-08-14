"""Swappable VALIDITY module (a trajectory is 'valid' if it passes a composed list of checks).

Currently: collision-free ∧ goal-reaching ∧ SOCP-verifier-certified (per γ). Add performant / smoothness /
comfort checks later by appending to `CHECKS` — the expansion loop and metrics only call `is_valid`.
Each check is `fn(path[T+1,2], env, gamma, **kw) -> bool`.
"""
from __future__ import annotations

import numpy as np

import _paths  # noqa: F401
import verifier_polytope as VP
import config as C


def min_clearance(path, env):
    obs = env.obstacles.detach().cpu().numpy()
    p = np.asarray(path, float)
    return float((np.linalg.norm(p[:, None, :] - obs[None, :, :2], axis=2) - obs[None, :, 2]
                  - float(env.r_robot)).min())


def collision_free(path, env, gamma, **kw):
    return min_clearance(path, env) >= 0.0


def reaches_goal(path, env, gamma, reach_radius=0.5, **kw):
    g = env.goal.detach().cpu().numpy()
    return bool(np.linalg.norm(np.asarray(path, float) - g, axis=1).min() < reach_radius)


def verifier_certified(path, env, gamma, **kw):
    v = dict(C.VERIFIER)
    v.pop("gamma_max", None)
    return VP.certify_trajectory(np.asarray(path, float), env.obstacles.detach().cpu().numpy(),
                                 float(env.r_robot), float(gamma), **v)


# the composed check-list (swap/extend here)
CHECKS = [collision_free, reaches_goal, verifier_certified]


def is_valid(path, env, gamma, checks=None, reach_radius=0.5):
    checks = checks or CHECKS
    for fn in checks:
        if not fn(path, env, gamma, reach_radius=reach_radius):
            return False
    return True


def labels(path, env, gamma, reach_radius=0.5):
    """Detailed per-check dict (diagnostics)."""
    return {
        "collision_free": collision_free(path, env, gamma),
        "reaches_goal": reaches_goal(path, env, gamma, reach_radius=reach_radius),
        "verifier": verifier_certified(path, env, gamma),
        "min_clearance": min_clearance(path, env),
    }
