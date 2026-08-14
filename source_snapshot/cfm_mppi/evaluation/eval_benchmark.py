from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from cfm_mppi.evaluation.metrics import compute_episode_metrics
from cfm_mppi.evaluation.result_writer import JSONLResultWriter, write_summary
from cfm_mppi.evaluation.eval_utils import CFMConfig, synthesize_control
from cfm_mppi.evaluation.run_drifting import load_drifting, sample_drifting_controls
from cfm_mppi.evaluation.run_safe_cfm import load_safe_cfm, sample_safe_cfm_controls
from cfm_mppi.models.transformer import TransformerModel
from cfm_mppi.mppi.flowmppi import FlowMPPI
from cfm_mppi.mppi.utils import doubleintegrator_dynamics, stage_cost, terminal_cost, unicycle_dynamics
from cfm_mppi.safegpc_adapter import SafeMPPIAdapter, resolve_gamma_schedule
from cfm_mppi.safegpc_adapter.mirror_sampler import mirror_mppi_action
from cfm_mppi.evaluation.run_tilted import load_tilted_flow, tilted_sample


DEFAULTS = {
    "horizon": 80,
    "dt": 0.1,
    "safety_margin": 0.5,
    "success_threshold": 0.5,
    "u_min": (-2.0, -2.0),
    "u_max": (2.0, 2.0),
}


class _AgentHistory:
    def __init__(self, max_length: int):
        self.max_length = int(max_length)
        self.data = None

    def update(self, new_data: torch.Tensor):
        new_data = new_data.unsqueeze(-1)
        if self.data is None:
            self.data = new_data
        elif len(self) < self.max_length:
            self.data = torch.cat([self.data, new_data], dim=-1)
        else:
            self.data = torch.cat([self.data[..., 1:], new_data], dim=-1)

    def get(self):
        return self.data

    def __len__(self):
        return self.data.shape[-1] if self.data is not None else 0


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _dynamics_step(state: np.ndarray, action: np.ndarray, dynamics: str, dt: float) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    if dynamics == "doubleintegrator":
        x = state.copy()
        x[0] = state[0] + dt * state[2] + 0.5 * dt * dt * action[0]
        x[1] = state[1] + dt * state[3] + 0.5 * dt * dt * action[1]
        x[2] = state[2] + dt * action[0]
        x[3] = state[3] + dt * action[1]
        return x.astype(np.float32)
    if dynamics == "unicycle":
        x = state.copy()
        x[0] = state[0] + dt * action[0] * np.cos(state[2])
        x[1] = state[1] + dt * action[0] * np.sin(state[2])
        x[2] = np.arctan2(np.sin(state[2] + dt * action[1]), np.cos(state[2] + dt * action[1]))
        return x.astype(np.float32)
    x = state.copy()
    x[:2] = state[:2] + dt * action
    return x.astype(np.float32)


def _goal_action(state: np.ndarray, goal: np.ndarray, dynamics: str) -> np.ndarray:
    direction = goal[:2] - state[:2]
    if dynamics == "unicycle":
        desired = np.arctan2(direction[1], direction[0])
        angle_err = np.arctan2(np.sin(desired - state[2]), np.cos(desired - state[2]))
        return np.array([np.clip(np.linalg.norm(direction), -2.0, 2.0), np.clip(2.0 * angle_err, -2.0, 2.0)], np.float32)
    if dynamics == "doubleintegrator":
        vel_err = -state[2:4] if state.shape[0] >= 4 else np.zeros(2, dtype=np.float32)
        return np.clip(0.6 * direction + 0.8 * vel_err, -2.0, 2.0).astype(np.float32)
    return np.clip(direction, -2.0, 2.0).astype(np.float32)


def _make_episode(seed: int, dynamics: str, dataset: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    if dynamics == "unicycle":
        start = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    else:
        start = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    goal = np.array([6.0, 6.0], dtype=np.float32)
    n_obs = 5 if dataset == "sfm" else 8
    obs = []
    attempts = 0
    while len(obs) < n_obs and attempts < 500:
        attempts += 1
        c = rng.uniform(1.0, 5.5, size=2)
        r = rng.uniform(0.25, 0.55)
        if np.linalg.norm(c - start[:2]) < 1.2 or np.linalg.norm(c - goal) < 1.2:
            continue
        obs.append([c[0], c[1], r])
    return start, goal, np.asarray(obs, dtype=np.float32)


def _context_batch(
    state: np.ndarray,
    goal: np.ndarray,
    controls: List[np.ndarray],
    obstacles: np.ndarray,
    gamma: float,
    safety_margin: float,
    horizon: int,
    obstacle_velocities: np.ndarray | None = None,
) -> Dict[str, torch.Tensor]:
    state4 = np.zeros(4, dtype=np.float32)
    state4[: min(len(state), 4)] = state[: min(len(state), 4)]
    action_hist = np.zeros((10, 2), dtype=np.float32)
    if controls:
        recent = np.asarray(controls[-10:], dtype=np.float32)
        action_hist[-len(recent) :] = recent
    ego_hist = np.tile(state4[None, :], (10, 1)).astype(np.float32)
    rel = np.zeros(4, dtype=np.float32)
    if obstacles.size:
        d = np.linalg.norm(obstacles[:, :2] - state[:2][None, :], axis=1) - obstacles[:, 2]
        idx = int(np.argmin(d))
        rel[:2] = obstacles[idx, :2] - state[:2]
        if obstacle_velocities is not None and obstacle_velocities.size:
            rel[2:4] = obstacle_velocities[idx, :2]
    obs_hist = np.tile(rel[None, :], (10, 1)).astype(np.float32)
    states = np.tile(state4[None, None, :], (1, horizon + 1, 1)).astype(np.float32)
    return {
        "states": torch.from_numpy(states),
        "controls_si": torch.zeros(1, horizon, 2),
        "controls_dyn": torch.zeros(1, horizon, 2),
        "start": torch.from_numpy(state[:2].reshape(1, 2).astype(np.float32)),
        "goal": torch.from_numpy(goal.reshape(1, 2).astype(np.float32)),
        "ego_history": torch.from_numpy(ego_hist.reshape(1, 10, 4)),
        "action_history": torch.from_numpy(action_hist.reshape(1, 10, 2)),
        "nearest_obstacle_history": torch.from_numpy(obs_hist.reshape(1, 10, 4)),
        "gamma": torch.tensor([gamma], dtype=torch.float32),
        "safety_margin": torch.tensor([safety_margin], dtype=torch.float32),
    }


def _repeat_context_batch(batch: Dict[str, torch.Tensor], count: int) -> Dict[str, torch.Tensor]:
    if count <= 1:
        return batch
    out: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.shape[0] == 1:
            out[key] = value.repeat((count,) + (1,) * (value.ndim - 1))
        else:
            out[key] = value
    return out


def _rollout_sequences_np(state: np.ndarray, controls: np.ndarray, dynamics: str, dt: float) -> np.ndarray:
    trajectories = []
    for seq in controls:
        x = state.astype(np.float32).copy()
        xs = [x.copy()]
        for action in seq.T:
            x = _dynamics_step(x, action, dynamics, dt)
            xs.append(x.copy())
        trajectories.append(xs)
    return np.asarray(trajectories, dtype=np.float32)


def _sequence_costs(trajectories: np.ndarray, controls: np.ndarray, goal: np.ndarray, obstacles: np.ndarray, safety_margin: float) -> np.ndarray:
    final = np.linalg.norm(trajectories[:, -1, :2] - goal[:2][None, :], axis=1) ** 2
    effort = 0.03 * np.sum(controls.transpose(0, 2, 1) ** 2, axis=(1, 2))
    if obstacles.size:
        centers = obstacles[:, :2]
        radii = obstacles[:, 2] + safety_margin
        d = np.linalg.norm(trajectories[:, :, None, :2] - centers[None, None, :, :], axis=3) - radii[None, None, :]
        clearance_penalty = 25.0 * np.sum(np.maximum(-d.min(axis=2), 0.0) ** 2, axis=1)
    else:
        clearance_penalty = 0.0
    return 80.0 * final + effort + clearance_penalty


class BenchmarkPolicies:
    def __init__(self, args, device: torch.device):
        self.args = args
        self.device = device
        self._mizuta = None
        self._mizuta_episode = None
        self._safe_cfm = None
        self._drifting = None
        self._guided_filter = None
        self._proposal_adapter = None
        self._tilted = None

    def _load_mizuta(self):
        if self._mizuta is not None:
            return self._mizuta
        ckpt_path = Path(self.args.mizuta_checkpoint)
        if not ckpt_path.exists():
            if self.args.smoke:
                self._mizuta = False
                return self._mizuta
            raise FileNotFoundError(f"Missing Mizuta checkpoint: {ckpt_path}")
        model = TransformerModel().to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.eval()
        self._mizuta = model
        return model

    def _reset_mizuta_episode(self, state: np.ndarray, goal: np.ndarray, dynamics: str, horizon: int):
        n_sample = 20 if self.args.smoke else 200
        if n_sample % 5 != 0:
            n_sample = 5 * max(1, n_sample // 5)
        dim_state = 3 if dynamics == "unicycle" else 4
        dyn_fn = unicycle_dynamics if dynamics == "unicycle" else doubleintegrator_dynamics
        sigma = torch.tensor([0.3, 0.6] if dynamics == "unicycle" else [0.4, 0.4], dtype=torch.float32)
        solver = FlowMPPI(
            num_samples=n_sample,
            dim_state=dim_state,
            dim_control=2,
            dynamics=dyn_fn,
            stage_cost=stage_cost,
            terminal_cost=terminal_cost,
            u_min=torch.tensor(DEFAULTS["u_min"], dtype=torch.float32),
            u_max=torch.tensor(DEFAULTS["u_max"], dtype=torch.float32),
            sigmas=sigma,
            lambda_=0.1,
            goal=torch.tensor(goal, dtype=torch.float32),
            horizon=horizon,
            dt=DEFAULTS["dt"],
            device=self.device,
            dynamics_type=dynamics,
        )
        self._mizuta_episode = {
            "solver": solver,
            "x_t": torch.randn(n_sample, 2, horizon, dtype=torch.float32, device=self.device),
            "histories": {
                "ego_state": _AgentHistory(max_length=10),
                "ego_control_sin": _AgentHistory(max_length=10),
                "obs_state": _AgentHistory(max_length=10),
                "obs_control": _AgentHistory(max_length=10),
            },
            "last_controls_sin": None,
            "n_sample": n_sample,
            "horizon": horizon,
            "step": 0,
        }

    def _mizuta_action(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        obstacles: np.ndarray,
        controls: List[np.ndarray],
        dynamics: str,
        horizon: int,
        obstacle_velocities: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, Dict]:
        model = self._load_mizuta()
        if model is False:
            return _goal_action(state, goal, dynamics), {
                "checkpoint": None,
                "model_calls_per_step": 0,
                "nfe": 0,
                "smoke_fallback": "goal_controller_missing_checkpoint",
            }
        if not controls or self._mizuta_episode is None:
            self._reset_mizuta_episode(state, goal, dynamics, horizon)
        ep = self._mizuta_episode
        histories = ep["histories"]
        n_sample = ep["n_sample"]
        pos_obs = torch.tensor(obstacles[:, :2], dtype=torch.float32, device=self.device).view(1, -1, 2)
        if obstacle_velocities is not None and obstacle_velocities.size:
            vel_obs = torch.tensor(obstacle_velocities[:, :2], dtype=torch.float32, device=self.device).view(1, -1, 2)
        else:
            vel_obs = torch.zeros_like(pos_obs)

        if ep["last_controls_sin"] is not None:
            control_history_len = len(histories["ego_control_sin"])
            noise_level = torch.tensor([0.8], device=self.device)
            noise = torch.randn(n_sample, ep["last_controls_sin"].shape[1], ep["last_controls_sin"].shape[2], device=self.device)
            x_t = noise_level * ep["last_controls_sin"] / 10.0 + (1.0 - noise_level) * noise
            x_t = x_t[:, :, (control_history_len + 1) :]
            histories["ego_control_sin"].update(ep["last_controls_sin"][:, :, control_history_len])
            histories["ego_state"].update(torch.tensor(state, dtype=torch.float32, device=self.device).view(1, -1))
            histories["obs_state"].update(pos_obs)
            histories["obs_control"].update(vel_obs)
            control_history_sin = histories["ego_control_sin"].get()
            ep["x_t"] = torch.cat([control_history_sin.expand(n_sample, -1, -1) / 10.0, x_t], dim=-1)

        first_step = ep["step"] == 0
        ode_times = [0.5, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0] if first_step else [0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]
        t_curr = torch.tensor([0.0], device=self.device) if first_step else torch.tensor([0.8], device=self.device)
        config = CFMConfig(
            ode_times=ode_times,
            dt=DEFAULTS["dt"],
            agent_radius=DEFAULTS["safety_margin"],
            space_scale=10.0,
            safe_margin_coefs=[0.1, 0.3, 0.5, 0.7, 0.9],
            goal_margin_coef=0.1,
            device=str(self.device),
        )
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).view(1, -1)
        goal_t = torch.tensor(goal, dtype=torch.float32, device=self.device).view(1, 2)
        with torch.no_grad():
            controls_dyn, controls_sin = synthesize_control(
                model,
                ep["solver"],
                config,
                state_t,
                goal_t,
                ep["x_t"],
                t_curr,
                pos_obs,
                vel_obs,
                ep["x_t"].shape[-1],
                histories=histories,
                d=0.1,
                k_p=3.0,
            )
        ep["last_controls_sin"] = controls_sin.detach()
        ep["step"] += 1
        return controls_dyn[0, :, 0].detach().cpu().numpy(), {
            "checkpoint": str(self.args.mizuta_checkpoint),
            "model_calls_per_step": len(ode_times),
            "nfe": len(ode_times),
        }

    def action(
        self,
        method: str,
        state: np.ndarray,
        goal: np.ndarray,
        obstacles: np.ndarray,
        controls: List[np.ndarray],
        dynamics: str,
        gamma: float,
        horizon: int,
        obstacle_velocities: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, Dict]:
        t0 = time.perf_counter()
        info: Dict = {"model_calls_per_step": 0, "nfe": 0}
        if method == "mizuta_cfm_mppi":
            action, step_info = self._mizuta_action(
                state,
                goal,
                obstacles,
                controls,
                dynamics,
                horizon,
                obstacle_velocities=obstacle_velocities,
            )
            info.update(step_info)
        elif method == "safemppi_gamma":
            sigma_default = (0.3, 0.6) if dynamics == "unicycle" else (0.4, 0.4)
            adapter = SafeMPPIAdapter(
                horizon=min(int(getattr(self.args, "safemppi_horizon", horizon)), horizon),
                dt=DEFAULTS["dt"],
                num_samples=int(getattr(self.args, "safemppi_samples", 64 if self.args.smoke else 512)),
                gamma=gamma,
                temperature=float(getattr(self.args, "safemppi_temperature", 0.1)),
                noise_sigma=getattr(self.args, "safemppi_noise_sigma", sigma_default),
                dynamics_type=dynamics,
                u_min=DEFAULTS["u_min"],
                u_max=DEFAULTS["u_max"],
                safety_margin=DEFAULTS["safety_margin"],
                running_goal_weight=float(getattr(self.args, "safemppi_running_goal_weight", 0.25)),
                terminal_goal_weight=float(getattr(self.args, "safemppi_terminal_goal_weight", 80.0)),
                control_weight=float(getattr(self.args, "safemppi_control_weight", 0.03)),
                smooth_weight=float(getattr(self.args, "safemppi_smooth_weight", 0.12)),
                soft_clearance_weight=float(getattr(self.args, "safemppi_soft_clearance_weight", 25.0)),
                progress_weight=float(getattr(self.args, "safemppi_progress_weight", 2.0)),
                debug_max_rollouts=int(getattr(self.args, "debug_rollouts", 80)),
                use_sets_backup=bool(getattr(self.args, "safemppi_use_sets_backup", False)),
                sets_num_modes=int(getattr(self.args, "safemppi_sets_num_modes", 3)),
                sets_branch_scale=float(getattr(self.args, "safemppi_sets_branch_scale", 0.85)),
                sets_include_cbf_backup=bool(getattr(self.args, "safemppi_sets_include_cbf_backup", True)),
                sets_cbf_push=float(getattr(self.args, "safemppi_sets_cbf_push", 1.25)),
                sets_reverse_speed=float(getattr(self.args, "safemppi_sets_reverse_speed", 0.75)),
                sets_turn_rate=float(getattr(self.args, "safemppi_sets_turn_rate", 1.4)),
            )
            action_t, step_info = adapter.plan(
                torch.tensor(state, dtype=torch.float32, device=self.device),
                torch.tensor(goal, dtype=torch.float32, device=self.device),
                torch.tensor(obstacles, dtype=torch.float32, device=self.device),
                gamma=gamma,
                obstacle_velocities=torch.tensor(obstacle_velocities, dtype=torch.float32, device=self.device)
                if obstacle_velocities is not None
                else None,
                seed=self.args.seed + len(controls),
                return_rollouts=bool(getattr(self.args, "collect_rollouts", False)),
            )
            action = action_t.detach().cpu().numpy()
            info.update(step_info)
            info.update({"model_calls_per_step": 0, "nfe": 0})
        elif method == "mizuta_safe":
            # Flavor A: wrap the SOTA policy (Mizuta) in our certificate => add a
            # hard per-step safety guarantee + gamma knob to any policy (Props 3-4).
            raw_action, step_info = self._mizuta_action(
                state, goal, obstacles, controls, dynamics, horizon,
                obstacle_velocities=obstacle_velocities,
            )
            info.update(step_info)
            if self._guided_filter is None:
                self._guided_filter = SafeMPPIAdapter(
                    horizon=1, dt=DEFAULTS["dt"], num_samples=1, dynamics_type=dynamics,
                    u_min=DEFAULTS["u_min"], u_max=DEFAULTS["u_max"], safety_margin=DEFAULTS["safety_margin"],
                    use_ho_barrier=True, eta=float(getattr(self.args, "guided_eta", 0.6)),
                    barrier_extra_margin=float(getattr(self.args, "guided_extra_margin", 0.25)),
                    barrier_activation_radius=float(getattr(self.args, "guided_activation_radius", 3.5)),
                )
            safe_a, finfo = self._guided_filter.safety_filter_action(
                torch.tensor(state, dtype=torch.float32, device=self.device),
                torch.tensor(obstacles, dtype=torch.float32, device=self.device),
                torch.tensor(raw_action, dtype=torch.float32, device=self.device),
                gamma=gamma,
                obstacle_velocities=torch.tensor(obstacle_velocities, dtype=torch.float32, device=self.device)
                if obstacle_velocities is not None else None,
            )
            action = safe_a.detach().cpu().numpy()
            info["filter_feasible"] = finfo["filter_feasible"]
        elif method in ("guided_safemppi", "guided_adaptive"):
            sigma_default = (0.3, 0.6) if dynamics == "unicycle" else (0.4, 0.4)
            adapter = SafeMPPIAdapter(
                horizon=min(int(getattr(self.args, "safemppi_horizon", horizon)), horizon),
                dt=DEFAULTS["dt"],
                num_samples=int(getattr(self.args, "safemppi_samples", 64 if self.args.smoke else 512)),
                gamma=gamma,
                temperature=float(getattr(self.args, "safemppi_temperature", 0.1)),
                noise_sigma=getattr(self.args, "safemppi_noise_sigma", sigma_default),
                dynamics_type=dynamics,
                u_min=DEFAULTS["u_min"],
                u_max=DEFAULTS["u_max"],
                safety_margin=DEFAULTS["safety_margin"],
                running_goal_weight=float(getattr(self.args, "guided_running_goal_weight", 0.6)),
                terminal_goal_weight=float(getattr(self.args, "guided_terminal_goal_weight", 120.0)),
                control_weight=float(getattr(self.args, "safemppi_control_weight", 0.03)),
                smooth_weight=float(getattr(self.args, "safemppi_smooth_weight", 0.12)),
                soft_clearance_weight=float(getattr(self.args, "safemppi_soft_clearance_weight", 25.0)),
                progress_weight=float(getattr(self.args, "guided_progress_weight", 5.0)),
                debug_max_rollouts=int(getattr(self.args, "debug_rollouts", 80)),
                guidance_horizon=int(getattr(self.args, "guided_guidance_horizon", 12)),
                use_ho_barrier=True,
                filter_output=True,
                eta=float(getattr(self.args, "guided_eta", 0.6)),
                use_guidance=True,
                guidance_relax=float(getattr(self.args, "guided_relax", 1.0)),
                use_aniso_cov=bool(getattr(self.args, "guided_aniso", True)),
                aniso_normal_scale=float(getattr(self.args, "guided_normal_scale", 0.5)),
                aniso_tangent_scale=float(getattr(self.args, "guided_tangent_scale", 1.7)),
                barrier_extra_margin=float(getattr(self.args, "guided_extra_margin", 0.2)),
                barrier_activation_radius=float(getattr(self.args, "guided_activation_radius", 3.5)),
                adaptive_gamma=(method == "guided_adaptive"),
                gamma_min=float(getattr(self.args, "guided_gamma_min", 0.1)),
                gamma_max=float(getattr(self.args, "guided_gamma_max", 1.0)),
            )
            action_t, step_info = adapter.plan(
                torch.tensor(state, dtype=torch.float32, device=self.device),
                torch.tensor(goal, dtype=torch.float32, device=self.device),
                torch.tensor(obstacles, dtype=torch.float32, device=self.device),
                gamma=gamma,
                obstacle_velocities=torch.tensor(obstacle_velocities, dtype=torch.float32, device=self.device)
                if obstacle_velocities is not None
                else None,
                seed=self.args.seed + len(controls),
                return_rollouts=bool(getattr(self.args, "collect_rollouts", False)),
            )
            action = action_t.detach().cpu().numpy()
            info.update(step_info)
            info.update({"model_calls_per_step": 0, "nfe": 0})
        elif method == "cfm_proposal_mppi":
            # Learned-proposal Safe MPPI (THEORY §10 / IDEA_learned_proposal):
            # sample the MPPI proposal from the gamma-CFM, then apply the hard
            # DCBF rejection + averaging + output projection as the certificate.
            ckpt_path = Path(self.args.safe_cfm_checkpoint)
            M = int(getattr(self.args, "proposal_samples", 256))
            if ckpt_path.exists():
                if self._safe_cfm is None:
                    self._safe_cfm = load_safe_cfm(ckpt_path, self.device)
                batch = _context_batch(state, goal, controls, obstacles, gamma,
                                       DEFAULTS["safety_margin"], horizon,
                                       obstacle_velocities=obstacle_velocities)
                batch = _repeat_context_batch(batch, M)
                seq = sample_safe_cfm_controls(self._safe_cfm, batch, horizon=horizon, nfe=8, device=self.device)
                proposal = seq.transpose(1, 2).contiguous()  # [M, H, 2]
                ncalls = 8
            else:
                proposal = None
                ncalls = 0
            if self._proposal_adapter is None:
                self._proposal_adapter = SafeMPPIAdapter(
                    horizon=min(int(getattr(self.args, "safemppi_horizon", horizon)), horizon),
                    dt=DEFAULTS["dt"], num_samples=M, dynamics_type=dynamics,
                    u_min=DEFAULTS["u_min"], u_max=DEFAULTS["u_max"], safety_margin=DEFAULTS["safety_margin"],
                    use_ho_barrier=True, eta=float(getattr(self.args, "guided_eta", 0.6)),
                    barrier_extra_margin=float(getattr(self.args, "guided_extra_margin", 0.2)),
                    barrier_activation_radius=float(getattr(self.args, "guided_activation_radius", 3.5)),
                    filter_output=True, progress_weight=5.0, terminal_goal_weight=120.0, running_goal_weight=0.6,
                )
            action_t, step_info = self._proposal_adapter.plan(
                torch.tensor(state, dtype=torch.float32, device=self.device),
                torch.tensor(goal, dtype=torch.float32, device=self.device),
                torch.tensor(obstacles, dtype=torch.float32, device=self.device),
                gamma=gamma,
                obstacle_velocities=torch.tensor(obstacle_velocities, dtype=torch.float32, device=self.device)
                if obstacle_velocities is not None else None,
                seed=self.args.seed + len(controls),
                proposal_controls=proposal,
            )
            action = action_t.detach().cpu().numpy()
            info.update(step_info)
            info.update({"model_calls_per_step": ncalls, "nfe": ncalls, "checkpoint": str(ckpt_path)})
        elif method == "safe_cfm":
            ckpt_path = Path(self.args.safe_cfm_checkpoint)
            if not ckpt_path.exists() and not self.args.smoke:
                raise FileNotFoundError(f"Missing safe CFM checkpoint: {ckpt_path}")
            if ckpt_path.exists():
                if self._safe_cfm is None:
                    self._safe_cfm = load_safe_cfm(ckpt_path, self.device)
                batch = _context_batch(
                    state,
                    goal,
                    controls,
                    obstacles,
                    gamma,
                    DEFAULTS["safety_margin"],
                    horizon,
                    obstacle_velocities=obstacle_velocities,
                )
                n_candidates = int(getattr(self.args, "safe_cfm_num_candidates", 1))
                batch = _repeat_context_batch(batch, n_candidates)
                seq = sample_safe_cfm_controls(self._safe_cfm, batch, horizon=horizon, nfe=8, device=self.device)
                seq_np = seq.detach().cpu().numpy()
                trajectories = _rollout_sequences_np(state, seq_np, dynamics, DEFAULTS["dt"])
                costs = _sequence_costs(trajectories, seq_np, goal, obstacles, DEFAULTS["safety_margin"])
                best = int(np.argmin(costs))
                action = seq_np[best, :, 0]
                info.update({"model_calls_per_step": 8, "nfe": 8, "checkpoint": str(ckpt_path)})
                if bool(getattr(self.args, "collect_rollouts", False)):
                    max_seq = int(getattr(self.args, "debug_rollouts", 80))
                    info["debug_sequences"] = {
                        "states": trajectories[:max_seq],
                        "best_state": trajectories[best],
                    }
            else:
                action = _goal_action(state, goal, dynamics)
                info.update({"smoke_fallback": "goal_controller_missing_checkpoint", "checkpoint": None})
        elif method == "drifting":
            ckpt_path = Path(self.args.drifting_checkpoint)
            if not ckpt_path.exists() and not self.args.smoke:
                raise FileNotFoundError(f"Missing Drifting checkpoint: {ckpt_path}")
            if ckpt_path.exists():
                if self._drifting is None:
                    self._drifting = load_drifting(ckpt_path, self.device)
                batch = _context_batch(
                    state,
                    goal,
                    controls,
                    obstacles,
                    gamma,
                    DEFAULTS["safety_margin"],
                    horizon,
                    obstacle_velocities=obstacle_velocities,
                )
                seq = sample_drifting_controls(self._drifting, batch, horizon=horizon, device=self.device)
                action = seq[0, :, 0].detach().cpu().numpy()
                info.update({"model_calls_per_step": 1, "nfe": 1, "checkpoint": str(ckpt_path)})
            else:
                action = _goal_action(state, goal, dynamics)
                info.update({"smoke_fallback": "goal_controller_missing_checkpoint", "checkpoint": None, "nfe": 1, "model_calls_per_step": 1})
        elif method == "guided_drifting":
            ckpt_path = Path(self.args.drifting_checkpoint)
            if ckpt_path.exists():
                if self._drifting is None:
                    self._drifting = load_drifting(ckpt_path, self.device)
                batch = _context_batch(state, goal, controls, obstacles, gamma,
                                       DEFAULTS["safety_margin"], horizon,
                                       obstacle_velocities=obstacle_velocities)
                seq = sample_drifting_controls(self._drifting, batch, horizon=horizon, device=self.device)
                raw_action = seq[0, :, 0]
            else:
                raw_action = torch.tensor(_goal_action(state, goal, dynamics), dtype=torch.float32, device=self.device)
            # runtime affine safety filter => hard per-step certificate (THEORY §7)
            if self._guided_filter is None:
                self._guided_filter = SafeMPPIAdapter(
                    horizon=1, dt=DEFAULTS["dt"], num_samples=1, dynamics_type=dynamics,
                    u_min=DEFAULTS["u_min"], u_max=DEFAULTS["u_max"], safety_margin=DEFAULTS["safety_margin"],
                    use_ho_barrier=True, eta=float(getattr(self.args, "guided_eta", 0.6)),
                    barrier_extra_margin=float(getattr(self.args, "guided_extra_margin", 0.2)),
                    barrier_activation_radius=float(getattr(self.args, "guided_activation_radius", 3.5)),
                )
            safe_a, finfo = self._guided_filter.safety_filter_action(
                torch.tensor(state, dtype=torch.float32, device=self.device),
                torch.tensor(obstacles, dtype=torch.float32, device=self.device),
                raw_action if torch.is_tensor(raw_action) else torch.tensor(raw_action, device=self.device),
                gamma=gamma,
                obstacle_velocities=torch.tensor(obstacle_velocities, dtype=torch.float32, device=self.device)
                if obstacle_velocities is not None else None,
            )
            action = safe_a.detach().cpu().numpy()
            info.update({"model_calls_per_step": 1, "nfe": 1, "filter_iters": finfo["filter_iters"],
                         "checkpoint": str(ckpt_path)})
        elif method == "mirror_mppi":
            # Mirror-map proposal: samples feasible-by-construction in the per-step
            # polytope (accept rate ~1.0 vs ~0.01 Gaussian); MPPI averages them.
            ov = (torch.tensor(obstacle_velocities, dtype=torch.float32, device=self.device)
                  if obstacle_velocities is not None else None)
            a_t, minfo = mirror_mppi_action(
                torch.tensor(state, dtype=torch.float32, device=self.device),
                torch.tensor(goal, dtype=torch.float32, device=self.device),
                torch.tensor(obstacles, dtype=torch.float32, device=self.device), ov,
                horizon=min(int(getattr(self.args, "mirror_horizon", 25)), horizon),
                num_samples=int(getattr(self.args, "mirror_samples", 320)),
                dt=DEFAULTS["dt"], u_min=DEFAULTS["u_min"], u_max=DEFAULTS["u_max"],
                safety_margin=DEFAULTS["safety_margin"],
                eta=float(getattr(self.args, "mirror_eta", 1.0)),
                dual_sigma=float(getattr(self.args, "mirror_dual_sigma", 1.2)),
                temperature=float(getattr(self.args, "mirror_temperature", 0.3)),
                clear_w=float(getattr(self.args, "mirror_clear_w", 40.0)),
                terminal_w=float(getattr(self.args, "mirror_terminal_w", 15.0)),
                margin_gain=float(getattr(self.args, "mirror_margin_gain", 0.2)),
                sensing_range=float(getattr(self.args, "mirror_sensing_range", 5.0)),
                gamma=gamma, seed=self.args.seed + len(controls),
                return_rollouts=bool(getattr(self.args, "collect_rollouts", False)),
                device=self.device)
            action = a_t.detach().cpu().numpy()
            info.update(minfo)
            info.update({"model_calls_per_step": 0, "nfe": 0})
        else:
            raise ValueError(f"Unknown method: {method}")
        action = np.clip(action, np.asarray(DEFAULTS["u_min"]), np.asarray(DEFAULTS["u_max"]))
        info["planning_wall_time"] = time.perf_counter() - t0
        return action.astype(np.float32), info


def _run_episode(args, policies, method: str, episode_idx: int, gamma: float | None) -> Dict:
    horizon = 20 if args.smoke else DEFAULTS["horizon"]
    dt = DEFAULTS["dt"]
    state, goal, obstacles = _make_episode(args.seed + episode_idx, args.dynamics, args.dataset)
    states = [state.copy()]
    controls: List[np.ndarray] = []
    planning_times = []
    min_barrier_h = None
    num_barrier_violations = 0
    checkpoint_path = None
    model_calls_per_step = 0
    nfe = 0
    gamma_value = float(gamma if gamma is not None else (args.gamma_grid[0] if args.gamma_grid else 0.5))
    for _ in range(horizon):
        action, info = policies.action(method, state, goal, obstacles, controls, args.dynamics, gamma_value, horizon)
        planning_times.append(info.get("planning_wall_time", 0.0))
        checkpoint_path = info.get("checkpoint", checkpoint_path)
        model_calls_per_step = int(info.get("model_calls_per_step", model_calls_per_step))
        nfe = int(info.get("nfe", nfe))
        if info.get("min_barrier_h") is not None:
            min_barrier_h = info["min_barrier_h"] if min_barrier_h is None else min(min_barrier_h, info["min_barrier_h"])
        num_barrier_violations += int(info.get("num_barrier_violations", 0))
        state = _dynamics_step(state, action, args.dynamics, dt)
        states.append(state.copy())
        controls.append(action.copy())
        if np.linalg.norm(state[:2] - goal[:2]) <= DEFAULTS["success_threshold"]:
            break
    states_arr = np.asarray(states, dtype=np.float32)
    controls_arr = np.asarray(controls, dtype=np.float32) if controls else np.zeros((0, 2), dtype=np.float32)
    metrics = compute_episode_metrics(
        states_arr,
        controls_arr,
        obstacles,
        goal,
        safety_margin=DEFAULTS["safety_margin"],
        success_threshold=DEFAULTS["success_threshold"],
        planning_times=planning_times,
        min_barrier_h=min_barrier_h,
        num_barrier_violations=num_barrier_violations,
    )
    scope = "linear_system_theorem_relevant" if args.dynamics == "doubleintegrator" else "empirical_only_unicycle"
    metrics.update(
        {
            "episode": episode_idx,
            "seed": args.seed,
            "dataset": args.dataset,
            "dynamics": args.dynamics,
            "method": method,
            "gamma": gamma_value if method == "safemppi_gamma" else None,
            "safe_coef": None,
            "safety_margin": DEFAULTS["safety_margin"],
            "safety_guarantee_scope": scope,
            "checkpoint_path": checkpoint_path,
            "config_path": None,
            "model_calls_per_step": model_calls_per_step,
            "nfe": nfe,
        }
    )
    return metrics


def get_parser():
    p = argparse.ArgumentParser(description="Unified CFM-MPPI / safeGPC benchmark harness.")
    p.add_argument("--dataset", default="sfm", choices=["sfm", "ucy", "sdd"])
    p.add_argument("--dynamics", default="doubleintegrator", choices=["doubleintegrator", "unicycle"])
    p.add_argument("--methods", nargs="+", default=["mizuta_cfm_mppi"])
    p.add_argument("--num-episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-root", default="results/benchmark")
    p.add_argument("--gamma-grid", nargs="*", type=float, default=None)
    p.add_argument("--gamma-schedule", default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--mizuta-checkpoint", default="output_dir/cfm_transformer/checkpoint.pth")
    p.add_argument("--safe-cfm-checkpoint", default="output_dir/safe_contextual_cfm/checkpoint_best.pth")
    p.add_argument("--drifting-checkpoint", default="output_dir/drifting_generator/checkpoint_best.pth")
    p.add_argument("--safemppi-horizon", type=int, default=40)
    p.add_argument("--safemppi-samples", type=int, default=512)
    p.add_argument("--safemppi-running-goal-weight", type=float, default=0.25)
    p.add_argument("--safemppi-terminal-goal-weight", type=float, default=80.0)
    p.add_argument("--safemppi-control-weight", type=float, default=0.03)
    p.add_argument("--safemppi-smooth-weight", type=float, default=0.12)
    p.add_argument("--safemppi-soft-clearance-weight", type=float, default=25.0)
    p.add_argument("--safemppi-progress-weight", type=float, default=2.0)
    p.add_argument("--safemppi-use-sets-backup", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--safemppi-sets-num-modes", type=int, default=3)
    p.add_argument("--safemppi-sets-branch-scale", type=float, default=0.85)
    p.add_argument("--safemppi-sets-include-cbf-backup", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--safemppi-sets-cbf-push", type=float, default=1.25)
    p.add_argument("--safemppi-sets-reverse-speed", type=float, default=0.75)
    p.add_argument("--safemppi-sets-turn-rate", type=float, default=1.4)
    p.add_argument("--debug-rollouts", type=int, default=80)
    return p


def main() -> None:
    args = get_parser().parse_args()
    if args.gamma_grid is None:
        args.gamma_grid = []
    _set_seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp_root = Path(args.output_root) / timestamp
    run_root = timestamp_root / args.dataset / args.dynamics
    device = torch.device(args.device)
    policies = BenchmarkPolicies(args, device)
    all_records: List[Dict] = []
    gamma_values = resolve_gamma_schedule(args.gamma_grid, args.gamma_schedule)
    writers: Dict[str, JSONLResultWriter] = {}
    try:
        for method in args.methods:
            writers[method] = JSONLResultWriter(run_root / f"{method}.jsonl")
            variants = gamma_values if method == "safemppi_gamma" else [None]
            for gamma in variants:
                for ep in range(args.num_episodes):
                    rec = _run_episode(args, policies, method, ep, gamma)
                    writers[method].write(rec)
                    all_records.append(rec)
                    gtxt = "" if gamma is None else f" gamma={float(gamma):.3g}"
                    print(
                        f"{method}{gtxt} ep={ep+1}/{args.num_episodes} "
                        f"success={int(rec['success'])} collision={int(rec['collision'])} "
                        f"min_clearance={rec['min_clearance']:.3f}",
                        flush=True,
                    )
    finally:
        for writer in writers.values():
            writer.close()
    summary_paths = write_summary(timestamp_root, all_records)
    print(json.dumps({k: str(v) for k, v in summary_paths.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
