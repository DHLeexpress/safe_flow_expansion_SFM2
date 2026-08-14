"""HP100 SFM adapter for the frozen 3-D-ball expansion loop.

Only the task adapter is SFM-specific.  Acquisition, successful-lineage
retry/commit, RBF replay, and optimization are provided by the byte-authenticated
vendored ball core in :mod:`sfm_hp100_ball_core.expansion`.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

import _paths  # noqa: F401
import grid_policy_sfm_hp100 as GPS
import sfm_hp100_dynamics as DYN
import sfm_hp100_features as HPF
import sfm_hp100_history as HPH
import sfm_metrics2 as VERIFY
import sfm_scene as SS
import stage2_hp100_data as HP100_DATA
from sfm_hp100_ball_core.expansion import Verification


VERSION = "sfm_hp100_ball_adapter_v1"
H = 10
NFE = 8
K = 16
B = 4
REACH = 0.5
DEFAULT_SCENE_PROFILE = "double_density_velocity_ood"
DEFAULT_SCENARIO_START = 300_000
DECLARED_EVAL_RANGES = ((150_000, 150_050), (250_000, 250_050))


@dataclass(frozen=True)
class SFMStateSnapshot:
    robot: np.ndarray
    steps: int
    status: str | None
    scenario_id: int


@dataclass
class SFMState:
    robot: np.ndarray
    humans: list
    hp_history: HPH.Hp100History
    controls: list[np.ndarray]
    steps: int
    status: str | None
    scenario_id: int
    core_episode: int

    def __deepcopy__(self, memo):
        del memo
        return SFMStateSnapshot(
            robot=self.robot.copy(), steps=int(self.steps),
            status=self.status, scenario_id=int(self.scenario_id),
        )


def _face_dict(face) -> dict:
    return dict(
        a=np.asarray(face.a, np.float32).tolist(), m=float(face.m),
        kind=str(face.kind), label=str(face.label),
        coefficient=float(face.coefficient), feasible=bool(face.feasible),
        interval=(None if face.interval is None else list(map(float, face.interval))),
    )


def _obstacles(ped_xy: np.ndarray) -> np.ndarray:
    positions = np.asarray(ped_xy, np.float32).reshape(-1, 2)
    return np.concatenate(
        (positions, np.full((len(positions), 1), SS.R_PED, np.float32)), axis=1,
    )


def clipped_plan_states(state, controls) -> np.ndarray:
    current = np.asarray(state, np.float32).reshape(4).copy()
    rows = [current.copy()]
    for action in np.asarray(controls, np.float32).reshape(-1, 2):
        current = DYN.step_numpy(current, action).astype(np.float32, copy=False)
        rows.append(current.copy())
    return np.asarray(rows, np.float32)


class HP100ExpansionPolicy(nn.Module):
    """Translate compact archived conditioning tokens to the HP100 policy API.

    The task computes and archives the frozen 176-D conditioning token.  This
    is exact for scopes that freeze every conditioning encoder (head-only and
    last-trunk-block-plus-head); it avoids duplicating each 10x32x100 raster in
    every replay row.
    """

    def __init__(self, policy: GPS.GridSFMHP100FlowPolicy, *, nfe: int = NFE):
        super().__init__()
        if not isinstance(policy, GPS.GridSFMHP100FlowPolicy):
            raise TypeError("HP100 expansion requires GridSFMHP100FlowPolicy")
        self.policy = policy
        self.context_dim = int(policy.ctx_dim)
        self.plan_shape = (H, 2)
        self.nfe = int(nfe)
        if self.nfe <= 0:
            raise ValueError("nfe must be positive")

    def _policy_context(self, context: torch.Tensor) -> torch.Tensor:
        if context.shape[-1] < self.context_dim:
            raise ValueError("packed SFM context is shorter than the HP100 token")
        return context[..., :self.context_dim]

    def sample_with_base(self, context, count, generator, base_std=1.0):
        if not math.isfinite(float(base_std)) or float(base_std) < 0.0:
            raise ValueError("base_std must be finite and nonnegative")
        device = self.policy.head.weight.device
        dtype = self.policy.head.weight.dtype
        base = torch.randn(
            int(count), self.policy.d, generator=generator,
            device=device, dtype=dtype,
        ) * float(base_std)
        token = self._policy_context(context).to(device=device, dtype=dtype)
        plans = self.policy.sample(
            int(count), token, nfe=self.nfe, temp=1.0, initial_noise=base,
        )
        return plans, base.reshape(int(count), H, 2)

    def sample(self, context, count, generator, base_std=1.0):
        plans, _ = self.sample_with_base(
            context, count, generator, base_std=base_std,
        )
        return plans

    def embed(self, context, candidates, flow_time=0.9, base=None):
        token = self._policy_context(context).to(candidates.device)
        if base is None:
            return self.policy.phi_s(candidates, token, s=float(flow_time))
        flat_base = base.reshape(len(candidates), self.policy.d)
        return self.policy.phi_s_from_x0(
            candidates, token, flat_base, s=float(flow_time),
        )

    def cfm_loss(self, contexts, candidates, reduction="none", loss_mask=None):
        if reduction not in {"none", "mean"}:
            raise ValueError("reduction must be none or mean")
        count = len(candidates)
        token = self._policy_context(contexts).to(candidates.device)
        x1 = (candidates / float(self.policy.u_max)).reshape(count, self.policy.d)
        x0 = torch.randn_like(x1)
        tau = torch.rand(count, device=x1.device).clamp(1.0e-4, 1.0)
        x_tau = (1.0 - tau)[:, None] * x0 + tau[:, None] * x1
        target = x1 - x0
        predicted = self.policy(x_tau, tau, token)
        squared = (predicted - target).square().reshape(count, H, 2)
        if loss_mask is None:
            per_sample = squared.mean(dim=(1, 2))
        else:
            mask = loss_mask.to(device=squared.device, dtype=squared.dtype)
            if mask.shape != squared.shape:
                raise ValueError("loss mask must match [B,10,2] candidates")
            per_sample = (squared * mask).sum(dim=(1, 2)) / mask.sum(
                dim=(1, 2)
            ).clamp_min(1.0)
        return per_sample if reduction == "none" else per_sample.mean()

    def freeze_visual_encoder_for_expansion(self) -> int:
        modules = (self.policy.grid_conv, self.policy.grid_projection)
        parameters = [parameter for module in modules for parameter in module.parameters()]
        for parameter in parameters:
            parameter.requires_grad_(False)
        return sum(parameter.numel() for parameter in parameters)

    def expansion_optimizer_parameters(
        self, scope: str,
    ) -> list[torch.nn.Parameter]:
        """Apply the declared HP100 optimizer scope and return its parameters."""
        if scope not in {
            "last_block_and_head", "last_two_blocks_and_head", "trunk_and_head",
        }:
            raise ValueError(
                "HP100 adapter optimizer scope must be last_block_and_head, "
                "last_two_blocks_and_head, or trunk_and_head"
            )
        blocks = self.policy.trunk.blocks
        if len(blocks) != 2:
            raise RuntimeError(
                "HP100 expansion scopes require the declared two-block trunk"
            )
        for parameter in self.policy.parameters():
            parameter.requires_grad_(False)
        selected_blocks = blocks[1:] if scope == "last_block_and_head" else blocks
        parameters = [
            *(
                self.policy.trunk.inp.parameters()
                if scope == "trunk_and_head" else ()
            ),
            *(parameter for block in selected_blocks for parameter in block.parameters()),
            *self.policy.head.parameters(),
        ]
        for parameter in parameters:
            parameter.requires_grad_(True)
        return parameters

    @property
    def head(self):
        return self.policy.head

    def state_dict(self, *args, **kwargs):
        return self.policy.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return self.policy.load_state_dict(state_dict, strict=strict, assign=assign)


class SFMHP100ExpansionTask:
    """Moving-pedestrian OOD task with exact-positive-only execution gating."""

    def __init__(
        self,
        *,
        scene_profile: str = DEFAULT_SCENE_PROFILE,
        scenario_start: int = DEFAULT_SCENARIO_START,
        fixed_scenario_id: int | None = None,
        trace_sidecars: bool = False,
    ):
        profile = SS.scene_profile(scene_profile)
        if profile["role"] == "pretraining_data_distribution":
            raise ValueError("expansion gathering must use an explicit OOD profile")
        self.scene_profile = str(scene_profile)
        self.profile = profile
        self.n_ped = int(profile["n_ped"])
        self.scenario_start = int(scenario_start)
        if self.scenario_start < DEFAULT_SCENARIO_START:
            raise ValueError(
                f"production expansion scenario_start must be >= {DEFAULT_SCENARIO_START}"
            )
        self.fixed_scenario_id = (
            None if fixed_scenario_id is None else int(fixed_scenario_id)
        )
        self._context_encoder: GPS.GridSFMHP100FlowPolicy | None = None
        self.trace_sidecars = bool(trace_sidecars)
        self._trace_sidecars: list[list[dict]] = []
        self.scene_ledger: list[dict] = []
        self._scenario_sources: dict[int, int] = {}

    def __getstate__(self):
        values = self.__dict__.copy()
        # Verifier workers only decode packed tokens; do not replicate a CUDA model.
        values["_context_encoder"] = None
        values["_trace_sidecars"] = []
        values["scene_ledger"] = []
        values["_scenario_sources"] = {}
        return values

    def attach_context_encoder(self, policy: GPS.GridSFMHP100FlowPolicy):
        if not isinstance(policy, GPS.GridSFMHP100FlowPolicy):
            raise TypeError("context encoder must be the strict HP100 policy")
        self._context_encoder = policy
        return self

    @property
    def context_dim(self) -> int:
        return int(GPS.LOW_TOKEN + GPS.VISUAL_TOKEN + 4 + 4 * self.n_ped)

    def reset(self, gamma: float, episode: int, seed: int) -> SFMState:
        # The vendored core deliberately gives identical reset_seed coordinates
        # to matching (round,retry,replica) lineages across gamma.  This produces
        # paired scenes across gamma while keeping replicas/retries distinct.
        scenario_id = (
            self.fixed_scenario_id
            if self.fixed_scenario_id is not None
            else self.scenario_start + int(seed) % 1_000_000_000
        )
        if self.fixed_scenario_id is None:
            previous_seed = self._scenario_sources.get(int(scenario_id))
            if previous_seed is not None and previous_seed != int(seed):
                raise RuntimeError("production scenario hash collision")
            self._scenario_sources[int(scenario_id)] = int(seed)
        humans = SS.make_humans(
            scenario_id, seed=0, n_ped=self.n_ped,
            speed_range=tuple(self.profile["ped_speed_range"]),
        )
        self.scene_ledger.append(dict(
            core_episode=int(episode), scenario_id=int(scenario_id),
            gamma=float(gamma), reset_seed=int(seed),
            fixed_scenario_audit=bool(self.fixed_scenario_id is not None),
        ))
        return SFMState(
            robot=np.zeros(4, np.float32), humans=humans,
            hp_history=HPH.Hp100History(), controls=[], steps=0,
            status=None, scenario_id=int(scenario_id), core_episode=int(episode),
        )

    @torch.no_grad()
    def context(self, state: SFMState, gamma: float) -> torch.Tensor:
        if self._context_encoder is None:
            raise RuntimeError("attach the frozen HP100 context encoder before gathering")
        ped_xy, ped_vel = SS.collect_humans(state.humans)
        ped_xy = np.asarray(ped_xy, np.float32)
        ped_vel = np.asarray(ped_vel, np.float32)
        frame = HPF.hp100_frame(
            state.robot[:2], _obstacles(ped_xy), sensing=SS.R_SENSE,
            n_base=HPF.POLYTOPE_N_BASE, obstacle_velocities=ped_vel,
            robot_velocity=state.robot[2:4], predict_gain=HPF.PREDICT_GAIN,
            predict_tau=HPF.PREDICT_TAU,
        )
        grid = state.hp_history.append(frame).unsqueeze(0)
        low5 = torch.from_numpy(HPF.low5(state.robot, SS.GOAL, gamma)).unsqueeze(0)
        history = torch.from_numpy(HPF.hist_pad(state.controls)).unsqueeze(0)
        device = next(self._context_encoder.parameters()).device
        token = self._context_encoder.ctx_from(
            grid.to(device), low5.to(device), history.to(device),
        )[0].detach().cpu().to(torch.float32)
        suffix = torch.from_numpy(np.concatenate([
            state.robot.astype(np.float32, copy=False), ped_xy.reshape(-1),
            ped_vel.reshape(-1),
        ]))
        packed = torch.cat([token, suffix])
        if len(packed) != self.context_dim:
            raise RuntimeError("packed HP100 SFM context length drifted")
        return packed

    def decode_context(self, context: torch.Tensor):
        values = context.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        expected = self.context_dim
        if len(values) != expected:
            raise ValueError(f"expected packed SFM context length {expected}, got {len(values)}")
        start = GPS.LOW_TOKEN + GPS.VISUAL_TOKEN
        robot = values[start:start + 4].copy()
        ped_xy = values[start + 4:start + 4 + 2 * self.n_ped].reshape(self.n_ped, 2).copy()
        ped_vel = values[start + 4 + 2 * self.n_ped:].reshape(self.n_ped, 2).copy()
        return robot, ped_xy, ped_vel

    def _nominal_step_margin(self, robot, action, ped_xy, ped_vel, gamma):
        _, geometry = HPF.hp100_frame(
            robot[:2], _obstacles(ped_xy), sensing=SS.R_SENSE,
            n_base=HPF.POLYTOPE_N_BASE, obstacle_velocities=ped_vel,
            robot_velocity=robot[2:4], predict_gain=HPF.PREDICT_GAIN,
            predict_tau=HPF.PREDICT_TAU, return_geometry=True,
        )
        after = DYN.step_numpy(robot, action)
        A = np.asarray(geometry["A"], np.float64)
        b = np.asarray(geometry["b"], np.float64)
        margins = np.asarray(geometry["margins"], np.float64)
        old = float(np.min((b - A @ robot[:2]) / margins))
        new = float(np.min((b - A @ after[:2]) / margins))
        return float(new - (1.0 - float(gamma)) * old), old, new

    def _native_cost(self, robot, controls, ped_xy, ped_vel) -> float:
        config = HP100_DATA.locked_expert_config()
        states = clipped_plan_states(robot, controls)
        goal = np.asarray(SS.GOAL, np.float64)
        initial = float(np.linalg.norm(states[0, :2] - goal))
        previous_distance = initial
        previous_action = np.zeros(2, np.float64)
        cost = 0.0
        safe_radius = (
            SS.R_PED + float(config["safety_margin"])
            + float(config["barrier_extra_margin"])
        )
        for index, action in enumerate(np.asarray(controls, np.float64)):
            position = states[index + 1, :2].astype(np.float64)
            distance = float(np.linalg.norm(position - goal))
            cost += float(config["running_goal_weight"]) * distance**2
            cost += float(config["control_weight"]) * float(action @ action)
            delta = action - previous_action
            cost += float(config["smooth_weight"]) * float(delta @ delta)
            cost -= float(config["progress_weight"]) * (initial - distance)
            if float(config["goal_retreat_exp_weight"]) > 0.0:
                scale = max(float(config["goal_retreat_exp_scale"]), np.finfo(float).eps)
                retreat = max(distance - previous_distance, 0.0) / scale
                retreat = min(retreat, float(config["goal_retreat_exp_cap"]))
                cost += float(config["goal_retreat_exp_weight"]) * math.expm1(retreat)
            predicted = ped_xy + ped_vel * (DYN.DT * (index + 1))
            clearance = float(np.linalg.norm(predicted - position[None], axis=1).min() - safe_radius)
            cost += float(config["soft_clearance_weight"]) * max(-clearance, 0.0) ** 2
            previous_distance = distance
            previous_action = action
        terminal = float(np.linalg.norm(states[-1, :2] - goal))
        return float(cost + float(config["terminal_goal_weight"]) * terminal**2)

    def _verify_one(self, context, candidate, gamma):
        robot, ped_xy, ped_vel = self.decode_context(context)
        controls = np.asarray(candidate.detach().cpu(), np.float32).reshape(-1, 2)
        if not 1 <= len(controls) <= H:
            raise ValueError("SFM verifier horizon must lie in [1,10]")
        segment4 = clipped_plan_states(robot, controls)
        segment = segment4[:, :2]
        pedestrians = VERIFY.predict_pedestrians(ped_xy, ped_vel, H=len(controls))
        taskspace = VERIFY.taskspace_ok(segment)
        collision_free = VERIFY.collision_free_time_indexed(segment, pedestrians)
        certified, faces, diagnostics = VERIFY.certify_moving_window(
            segment, pedestrians, float(gamma),
        )
        # Online proposals are always H=10.  The vendored success archive also
        # asks the task to reverify terminal suffixes shorter than H before it
        # pads them for fixed-shape CFM.  For SFM those prefixes must not be
        # promoted: the declared GREEN label is a full-H certificate, and goal
        # termination is not permission to weaken that label.
        full_h = bool(len(controls) == H)
        valid = bool(full_h and taskspace and collision_free and certified)
        step_margin, hp_old, hp_new = self._nominal_step_margin(
            robot, controls[0], ped_xy, ped_vel, gamma,
        )
        progress = float(
            np.linalg.norm(robot[:2] - SS.GOAL)
            - np.linalg.norm(segment[-1] - SS.GOAL)
        )
        result = Verification(
            valid=valid, hp_eligible=bool(step_margin >= -1.0e-9),
            margin=float(diagnostics["slack"]),
            execution_cost=self._native_cost(robot, controls, ped_xy, ped_vel),
            progress=progress,
            # No extra progress or nominal-Hp gate: NVP iff B has no exact y=1.
            progress_eligible=True, error=False, target_eligible=True,
            step_margin=float(step_margin),
        )
        sidecar = dict(
            result=dict(
                resolved=True, y=int(valid), full_h=full_h,
                terminal_step=int(len(controls)), taskspace=bool(taskspace),
                collision_free=bool(collision_free), certificate=bool(certified),
                segment=segment.astype(np.float32), faces=[_face_dict(face) for face in faces],
                diagnostics=diagnostics, hp_old=hp_old, hp_new=hp_new,
            ),
            execution_eligible=bool(valid),
        )
        return result, sidecar

    def verify(self, context, candidates, gamma) -> Sequence[Verification]:
        results, sidecars = [], []
        for candidate_id, candidate in enumerate(candidates):
            try:
                result, sidecar = self._verify_one(context, candidate, gamma)
            except Exception as error:
                result = Verification(
                    valid=False, hp_eligible=False, margin=-float("inf"),
                    execution_cost=float("inf"), progress=0.0,
                    progress_eligible=True, error=True, target_eligible=True,
                    step_margin=None,
                )
                sidecar = dict(
                    result=dict(resolved=False, y=0, error=f"{type(error).__name__}: {error}"),
                    execution_eligible=False,
                )
            sidecar["candidate_id"] = int(candidate_id)
            results.append(result); sidecars.append(sidecar)
        if self.trace_sidecars:
            self._trace_sidecars.append(sidecars)
        return results

    def pop_trace_sidecars(self) -> list[dict]:
        if not self._trace_sidecars:
            raise RuntimeError("no serial-verifier sidecar is available")
        return self._trace_sidecars.pop(0)

    def advance(self, state: SFMState, candidate: torch.Tensor) -> SFMState:
        action = DYN.clip_action_numpy(
            np.asarray(candidate.detach().cpu(), np.float32).reshape(-1, 2)[0]
        ).astype(np.float32, copy=False)
        robot = DYN.step_numpy(state.robot, action).astype(np.float32, copy=False)
        SS.advance_humans(state.humans, robot)
        ped_xy, _ = SS.collect_humans(state.humans)
        clearance = float(np.linalg.norm(ped_xy - robot[:2][None], axis=1).min() - SS.R_PED)
        if clearance < 0.0:
            status = "COLLISION"
        elif not VERIFY.taskspace_ok(robot[:2][None]):
            status = "OOB"
        elif float(np.linalg.norm(robot[:2] - SS.GOAL)) < REACH:
            status = "SUCCESS"
        else:
            status = None
        return SFMState(
            robot=robot, humans=state.humans, hp_history=state.hp_history,
            controls=[*state.controls, action.copy()], steps=state.steps + 1,
            status=status, scenario_id=state.scenario_id,
            core_episode=state.core_episode,
        )

    def terminal(self, state: SFMState) -> str | None:
        return state.status

    @staticmethod
    def successful_trajectory_above_fraction(executed_states) -> float:
        del executed_states
        return 0.0

    @staticmethod
    def successful_trajectory_mean_z(executed_states) -> float:
        del executed_states
        return 0.0


def load_head_only(checkpoint: str, *, device: str):
    policy, payload = GPS.load_sfm_hp100_policy(checkpoint, device=device)
    GPS.configure_head_only_expansion(policy)
    adapter = HP100ExpansionPolicy(policy)
    assert_head_only(adapter)
    task = SFMHP100ExpansionTask().attach_context_encoder(policy)
    return adapter, task, payload


def assert_head_only(adapter: HP100ExpansionPolicy) -> list[str]:
    trainable = sorted(name for name, value in adapter.named_parameters() if value.requires_grad)
    expected = sorted(f"policy.head.{name}" for name, _ in adapter.policy.head.named_parameters())
    if trainable != expected:
        raise RuntimeError(f"head-only contract violated: {trainable} != {expected}")
    return trainable
