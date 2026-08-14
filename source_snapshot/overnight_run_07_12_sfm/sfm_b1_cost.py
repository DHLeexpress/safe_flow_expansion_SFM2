"""Pure frozen SafeMPPI proposal scoring and B1 execution gating."""
from __future__ import annotations

from dataclasses import asdict
import numpy as np
import torch

import _paths  # noqa: F401
import grid_scene as GS
from polar_grid import polytope_HP
from cfm_mppi.safegpc_adapter.barrier import barrier_clearance
from cfm_mppi.safegpc_adapter.safemppi import SafeMPPIConfig
import sfm_scene as SS


def frozen_expert_config():
    """The exact mode-1 adapter config whose cost terms define arms B--D."""
    values = GS.mode1_config(range_m=SS.R_SENSE, u_max=SS.U_MAX, noise_var_mult=3.0)
    return SafeMPPIConfig(**values)


def rollout_states(state, controls, dt=SS.DT):
    controls = torch.as_tensor(controls)
    current = torch.as_tensor(state, dtype=controls.dtype, device=controls.device).reshape(1, 4)
    current = current.expand(controls.shape[0], -1).clone()
    states = [current.clone()]
    for step in range(controls.shape[1]):
        action = controls[:, step]
        nxt = current.clone()
        nxt[:, :2] = current[:, :2] + float(dt) * current[:, 2:4] + 0.5 * float(dt) ** 2 * action
        nxt[:, 2:4] = current[:, 2:4] + float(dt) * action
        states.append(nxt)
        current = nxt
    return torch.stack(states, dim=1)


def safemppi_proposal_cost(state, controls, goal, ped_xy, ped_vel, config=None):
    """Adapter cost terms only: no rejection, refinement, proposal generation, or fallback."""
    config = frozen_expert_config() if config is None else config
    controls = torch.as_tensor(controls, dtype=torch.float32)
    if controls.ndim == 2:
        controls = controls.unsqueeze(0)
    if controls.shape[1:] != (int(config.horizon), 2):
        raise ValueError(f"expected [N,{config.horizon},2], got {tuple(controls.shape)}")
    states = rollout_states(state, controls, dt=config.dt)
    goal = torch.as_tensor(goal, dtype=controls.dtype, device=controls.device)[:2]
    ped_xy = torch.as_tensor(ped_xy, dtype=controls.dtype, device=controls.device).reshape(-1, 2)
    ped_vel = torch.as_tensor(ped_vel, dtype=controls.dtype, device=controls.device).reshape(-1, 2)
    safe_radius = SS.R_PED + float(config.safety_margin) + float(config.barrier_extra_margin)
    initial_distance = torch.linalg.vector_norm(states[:, 0, :2] - goal, dim=1)
    previous_distance = initial_distance
    previous_action = torch.zeros_like(controls[:, 0])
    costs = torch.zeros(len(controls), dtype=controls.dtype, device=controls.device)
    for step in range(controls.shape[1]):
        next_state = states[:, step + 1]
        distance = torch.linalg.vector_norm(next_state[:, :2] - goal, dim=1)
        goal_cost = float(config.running_goal_weight) * distance.square()
        effort = float(config.control_weight) * controls[:, step].square().sum(dim=1)
        smooth = float(config.smooth_weight) * (controls[:, step] - previous_action).square().sum(dim=1)
        progress = -float(config.progress_weight) * (initial_distance - distance)
        if float(config.goal_retreat_exp_weight) > 0.0:
            scale = max(float(config.goal_retreat_exp_scale), torch.finfo(controls.dtype).eps)
            normalized = torch.clamp(
                torch.relu(distance - previous_distance) / scale,
                max=max(float(config.goal_retreat_exp_cap), 0.0),
            )
            retreat = float(config.goal_retreat_exp_weight) * torch.expm1(normalized)
        else:
            retreat = 0.0
        center = ped_xy + ped_vel * (float(config.dt) * (step + 1))
        obstacles = torch.cat([
            center, torch.full((len(center), 1), safe_radius, device=center.device, dtype=center.dtype)
        ], dim=1)
        batched = obstacles.unsqueeze(0).expand(len(controls), -1, -1)
        clearance = barrier_clearance(next_state[:, :2], batched)
        soft_clearance = float(config.soft_clearance_weight) * torch.relu(-clearance).square()
        costs = costs + goal_cost + effort + smooth + soft_clearance + progress + retreat
        previous_distance = distance
        previous_action = controls[:, step]
    terminal = torch.linalg.vector_norm(states[:, -1, :2] - goal, dim=1)
    return costs + float(config.terminal_goal_weight) * terminal.square()


def nominal_hp_margin(state, first_action, ped_xy, gamma):
    state = np.asarray(state, np.float32).reshape(4)
    action = np.asarray(first_action, np.float32).reshape(2)
    obstacles = np.concatenate([
        np.asarray(ped_xy, np.float32).reshape(-1, 2),
        np.full((len(ped_xy), 1), SS.R_PED, np.float32),
    ], axis=1)
    hp, _ = polytope_HP(state[:2], obstacles, sensing=SS.R_SENSE, n_base=16)
    next_position = state[:2] + SS.DT * state[2:4] + 0.5 * SS.DT ** 2 * action
    old = float(hp(state[:2][None])[0])
    new = float(hp(next_position[None])[0])
    return new - (1.0 - float(gamma)) * old, old, new


def select_admissible(query_rows, *, selector, state, ped_xy, ped_vel, gamma):
    """Gate full-H positives by one-step nominal Hp before either selector."""
    admissible = []
    state = np.asarray(state, np.float32).reshape(4)
    start_goal_distance = float(np.linalg.norm(state[:2] - SS.GOAL))
    for row in query_rows:
        result = row["result"]
        if not result.get("resolved"):
            continue
        if not bool(result.get("full_h")) or int(result.get("terminal_step", -1)) != 10:
            raise ValueError("B1 execution requires every resolved query to be full H=10")
        if int(result.get("y", 0)) != 1:
            continue
        first_action = np.asarray(row["controls"][0], np.float32)
        margin, hp_old, hp_new = nominal_hp_margin(state, first_action, ped_xy, gamma)
        next_position = state[:2] + SS.DT * state[2:4] + 0.5 * SS.DT ** 2 * first_action
        step_progress = start_goal_distance - float(np.linalg.norm(next_position - SS.GOAL))
        row["hp_margin"] = float(margin)
        row["hp_old"] = float(hp_old)
        row["hp_new"] = float(hp_new)
        row["step_progress"] = float(step_progress)
        if margin >= -1.0e-9:
            admissible.append(row)
    if not admissible:
        return None
    if selector == "margin":
        return max(admissible, key=lambda row: (
            row["hp_margin"], row["step_progress"], -int(row["candidate_id"]),
        ))
    if selector not in ("safemppi_cost", "balanced_rank"):
        raise ValueError(f"unknown selector: {selector}")
    controls = torch.as_tensor(np.stack([row["controls"] for row in admissible]), dtype=torch.float32)
    costs = safemppi_proposal_cost(state, controls, SS.GOAL, ped_xy, ped_vel).cpu().numpy()
    for row, cost in zip(admissible, costs):
        row["expert_cost"] = float(cost)
    if selector == "safemppi_cost":
        return min(
            admissible,
            key=lambda row: (row["expert_cost"], int(row["candidate_id"])),
        )
    safety_order = sorted(
        admissible,
        key=lambda row: (-row["hp_margin"], int(row["candidate_id"])),
    )
    performance_order = sorted(
        admissible,
        key=lambda row: (row["expert_cost"], int(row["candidate_id"])),
    )
    for rank, row in enumerate(safety_order, start=1):
        row["safety_rank"] = rank
    for rank, row in enumerate(performance_order, start=1):
        row["performance_rank"] = rank
    for row in admissible:
        row["rank_sum"] = row["safety_rank"] + row["performance_rank"]
    return min(
        admissible,
        key=lambda row: (
            row["rank_sum"], row["safety_rank"], -row["hp_margin"],
            row["expert_cost"], int(row["candidate_id"]),
        ),
    )


def scorer_manifest():
    return dict(config=asdict(frozen_expert_config()), semantics="frozen SafeMPPI raw proposal cost terms")
