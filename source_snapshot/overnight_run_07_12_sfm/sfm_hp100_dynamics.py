"""Shared capped double-integrator dynamics for the additive Hp100 pipeline.

The old Hp10 study intentionally remains untouched.  Every new Hp100 caller
must use this module so expert data, raw evaluation, and later expansion agree
on both componentwise physical limits.
"""
from __future__ import annotations

import numpy as np
import torch


DT = 0.1
U_MAX = 2.0
V_MAX = 2.0


def _check_last_dim(value, expected: int, name: str) -> None:
    if value.ndim == 0 or int(value.shape[-1]) != int(expected):
        raise ValueError(f"{name} must have last dimension {expected}, got {tuple(value.shape)}")


def clip_action_numpy(action, *, u_max: float = U_MAX) -> np.ndarray:
    value = np.asarray(action)
    _check_last_dim(value, 2, "action")
    return np.clip(value, -float(u_max), float(u_max))


def clip_velocity_numpy(velocity, *, v_max: float = V_MAX) -> np.ndarray:
    value = np.asarray(velocity)
    _check_last_dim(value, 2, "velocity")
    return np.clip(value, -float(v_max), float(v_max))


def step_numpy(
    state,
    action,
    *,
    dt: float = DT,
    u_max: float = U_MAX,
    v_max: float = V_MAX,
) -> np.ndarray:
    """Advance ``[...,4]`` states with capped acceleration and velocity.

    The position update retains the study's double-integrator discretization,
    while both the incoming and stored outgoing velocity are clipped.  This
    exact ordering is part of the Hp100 data manifest.
    """
    if float(dt) <= 0.0:
        raise ValueError("dt must be positive")
    value = np.asarray(state)
    control = np.asarray(action)
    _check_last_dim(value, 4, "state")
    _check_last_dim(control, 2, "action")
    velocity = clip_velocity_numpy(value[..., 2:4], v_max=v_max)
    control = clip_action_numpy(control, u_max=u_max)
    position = value[..., :2] + float(dt) * velocity + 0.5 * float(dt) ** 2 * control
    next_velocity = clip_velocity_numpy(
        velocity + float(dt) * control, v_max=v_max
    )
    return np.concatenate((position, next_velocity), axis=-1).astype(
        np.result_type(value.dtype, control.dtype, np.float32), copy=False
    )


def clip_action_torch(action: torch.Tensor, *, u_max: float = U_MAX) -> torch.Tensor:
    _check_last_dim(action, 2, "action")
    return torch.clamp(action, -float(u_max), float(u_max))


def clip_velocity_torch(velocity: torch.Tensor, *, v_max: float = V_MAX) -> torch.Tensor:
    _check_last_dim(velocity, 2, "velocity")
    return torch.clamp(velocity, -float(v_max), float(v_max))


def step_torch(
    state: torch.Tensor,
    action: torch.Tensor,
    *,
    dt: float = DT,
    u_max: float = U_MAX,
    v_max: float = V_MAX,
) -> torch.Tensor:
    """Torch equivalent of :func:`step_numpy`, preserving autograd."""
    if float(dt) <= 0.0:
        raise ValueError("dt must be positive")
    _check_last_dim(state, 4, "state")
    _check_last_dim(action, 2, "action")
    velocity = clip_velocity_torch(state[..., 2:4], v_max=v_max)
    control = clip_action_torch(action, u_max=u_max)
    position = state[..., :2] + float(dt) * velocity + 0.5 * float(dt) ** 2 * control
    next_velocity = clip_velocity_torch(
        velocity + float(dt) * control, v_max=v_max
    )
    return torch.cat((position, next_velocity), dim=-1)


def contract() -> dict:
    return dict(
        name="componentwise_capped_double_integrator_v1",
        dt=float(DT),
        action_cap=dict(kind="componentwise", minimum=-float(U_MAX), maximum=float(U_MAX)),
        velocity_cap=dict(kind="componentwise", minimum=-float(V_MAX), maximum=float(V_MAX)),
        ordering=(
            "clip incoming velocity; clip action; update position as "
            "p+dt*v+0.5*dt^2*u; update and clip outgoing velocity"
        ),
    )
