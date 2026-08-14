"""Faithful SafeMPPI demonstration expert used for Hp10 data generation.

This comparator is separate from raw policy evaluation.  Its MPPI temperature
is 0.1; the learned flow evaluator remains temperature 1.0.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import torch

import _paths  # noqa: F401
import grid_scene as GS
from cfm_mppi.safegpc_adapter.safemppi import SafeMPPIAdapter, SafeMPPIConfig
import sfm_scene as SS


EXPERT_NAME = "r2_n2048_nv3_pg025"
DATASET_MANIFEST_SHA256 = "fd6fac0140411b20d2f496609029c1fa02eabfc5464c3f41889119e7d31b9585"


def demonstration_config():
    values = GS.mode1_config(range_m=SS.R_SENSE, u_max=SS.U_MAX, noise_var_mult=3.0)
    values.update(
        num_samples=2048, temperature=0.1, centroid_gain=0.2,
        predict_gain=0.25, polytope_nbase=16,
        use_goal_nominal=False, warm_start=True,
    )
    config = SafeMPPIConfig(**values)
    expected = dict(
        horizon=10, centroid_smooth=.25, centroid_eps=.15,
        smooth_weight=.12, predict_gain=.25, polytope_nbase=16,
    )
    for key, value in expected.items():
        if getattr(config, key) != value:
            raise RuntimeError(f"demonstration expert drifted at {key}")
    return config


def _step(state, action):
    value = np.asarray(state, np.float32).copy()
    action = np.asarray(action, np.float32).reshape(2)
    value[:2] += SS.DT * value[2:4] + 0.5 * SS.DT ** 2 * action
    value[2:4] += SS.DT * action
    return value


def _plan_states(state, controls):
    rows = [np.asarray(state, np.float32).copy()]
    for action in np.asarray(controls, np.float32).reshape(-1, 2):
        rows.append(_step(rows[-1], action))
    return np.asarray(rows, np.float32)


def rollout(episode, gamma, *, n_ped, ped_speed_range, T=180, reach=0.5,
            device="cpu", collect_trace=False):
    """Run the exact velocity-aware expert that generated the ID demonstrations."""
    config = demonstration_config()
    adapter = SafeMPPIAdapter(**asdict(config))
    humans = SS.make_humans(episode, 0, n_ped, ped_speed_range)
    state = np.zeros(4, np.float32)
    states = [state.copy()]
    controls, pedestrian_rows, pedestrian_velocities, trace = [], [], [], []
    minimum_clearance = float("inf")
    collision = reached = False
    goal = torch.as_tensor(SS.GOAL, dtype=torch.float32, device=device)
    for step in range(int(T)):
        ped_xy, ped_vel = SS.collect_humans(humans)
        clearance = float(np.linalg.norm(ped_xy - state[:2][None], axis=1).min() - SS.R_PED)
        minimum_clearance = min(minimum_clearance, clearance)
        if clearance < 0.0:
            collision = True
            break
        if float(np.linalg.norm(state[:2] - SS.GOAL)) < float(reach):
            reached = True
            break
        obstacles = np.concatenate([
            ped_xy, np.full((len(ped_xy), 1), SS.R_PED, np.float32),
        ], axis=1)
        before = state.copy()
        action, info = adapter.plan(
            torch.as_tensor(state, dtype=torch.float32, device=device), goal,
            torch.as_tensor(obstacles, dtype=torch.float32, device=device),
            gamma=float(gamma),
            obstacle_velocities=torch.as_tensor(ped_vel, dtype=torch.float32, device=device),
            seed=int(episode) * 200 + int(step), return_rollouts=False,
        )
        action = action.detach().cpu().numpy().astype(np.float32).reshape(2)
        mean_sequence = np.asarray(info["mean_sequence"], np.float32)
        state = _step(state, action)
        controls.append(action); states.append(state.copy())
        pedestrian_rows.append(ped_xy.copy()); pedestrian_velocities.append(ped_vel.copy())
        if collect_trace:
            polytope = info.get("polytope")
            trace.append(dict(
                step=int(step), state=before, action=action.copy(), controls=mean_sequence,
                planned_states=_plan_states(before, mean_sequence), ped_xy=ped_xy.copy(),
                ped_vel=ped_vel.copy(), gamma=float(gamma),
                sequence_kind="reward_weighted_mean",
                nominal_polytope=(None if polytope is None else dict(
                    A=np.asarray(polytope[0]), b=np.asarray(polytope[1]),
                    margins=np.asarray(polytope[3]),
                    n_base=16, velocity_used=True,
                )),
            ))
        SS.advance_humans(humans, state)
    if not collision and not reached:
        terminal_xy, _ = SS.collect_humans(humans)
        terminal_clearance = float(np.linalg.norm(terminal_xy - state[:2][None], axis=1).min() - SS.R_PED)
        minimum_clearance = min(minimum_clearance, terminal_clearance)
        collision = terminal_clearance < 0.0
        reached = bool(not collision and np.linalg.norm(state[:2] - SS.GOAL) < float(reach))
    success = bool(reached and not collision)
    return dict(
        episode=int(episode), gamma=float(gamma), success=success, collision=bool(collision),
        reached=bool(reached), timeout=bool(not reached and not collision), steps=len(controls),
        time_to_goal=(len(controls) * SS.DT if success else None),
        min_clear=float(minimum_clearance), min_clearance=float(minimum_clearance),
        successful_clearance=(float(minimum_clearance) if success else None),
        states=np.asarray(states, np.float32), controls=np.asarray(controls, np.float32),
        peds=np.asarray(pedestrian_rows, np.float32),
        ped_vels=np.asarray(pedestrian_velocities, np.float32),
        n_ped=int(n_ped), ped_speed_range=tuple(map(float, ped_speed_range)),
        trace=(trace if collect_trace else None),
        expert_semantics=("SafeMPPI demonstration expert: 2048 rollouts, MPPI temperature=.1, "
                          "centroid_gain=.2, predict_gain=.25, pedestrian velocity supplied"),
    )


def manifest():
    return dict(
        name=EXPERT_NAME, config=asdict(demonstration_config()),
        pass_pedestrian_velocity=True,
        dataset_manifest_sha256=DATASET_MANIFEST_SHA256,
        provenance=("reconstructed from the declared dataset expert configuration and seed convention; "
                    "not a stored rollout replay"),
        role="demonstration expert comparator; never a raw flow evaluation",
    )
