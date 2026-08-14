"""Generate fresh Hp100 ID demonstrations with one shared capped dynamics.

This is an additive replacement for the unavailable Hp10 stage-2 generator;
it never reads, interpolates, or upsamples the old ``16 x 12`` grid files.
The canonical run fills exactly 500 successful trajectories for each of seven
gammas by advancing deterministic episode IDs from zero.  It uses the locked
SafeMPPI expert in the matched training environment (20 pedestrians,
0.5--1.0 m/s).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import multiprocessing as mp
import sys

import numpy as np
import torch

import _paths  # noqa: F401
from cfm_mppi.safegpc_adapter.safemppi import SafeMPPIAdapter
import cfm_mppi.safegpc_adapter.barrier as SAFETY_BARRIER
import cfm_mppi.safegpc_adapter.polytope_v2 as NOMINAL_POLYTOPE
import sfm_b1_expert as EXPERT
import sfm_hp100_dynamics as DYN
import sfm_hp100_features as HPF
import sfm_scene as SS


SCHEMA_VERSION = "sfm_hp100_id_demonstrations_v3_certified_weighted_plan"
HP100_EXPERT_NAME = "hp100_current_tangent_r3_n2048_certified_weighted_h10"
HORIZON = 10
N_BASE = 16
N_PED = 20
PED_SPEED_RANGE = (0.5, 1.0)
EPISODE_START = 0
SUCCESSES_PER_GAMMA = 500
MAX_ATTEMPTS_PER_GAMMA = 5000
T = 180
REACH = 0.5
TARGET_ELIGIBLE = 0
TARGET_ALL_REJECTED = 1
TARGET_WEIGHTED_H10_FAILED = 2
SUPERVISED_TARGET = (
    "current SafeMPPI accepted-set weighted H=10 plan, admitted only when at "
    "least one candidate survives and the weighted plan independently passes "
    "the same frozen nominal-polytope H=10 contraction recheck"
)


def target_contract() -> dict:
    return dict(
        stored_rows="every context from each retained task-successful lineage",
        target_tensor="U = current planner mean_sequence [10,2]",
        eligible_identity=(
            "target_eligible == (plan_accepted_count > 0 and "
            "plan_first_violation == -1)"
        ),
        recheck=(
            "shared capped double-integrator rollout against the single nominal "
            "A,b,margins tuple persisted with the current Hp100 context after "
            "its planner-equivalence assertion"
        ),
        excluded_rows=(
            "all-rejected fallback means and accepted-set weighted means that fail "
            "the frozen-set H10 recheck remain provenance/history rows but never "
            "enter the CFM objective"
        ),
    )


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CappedSafeMPPIAdapter(SafeMPPIAdapter):
    """Locked expert whose internal rollouts use the Hp100 dynamics contract."""

    def _step(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        if self.config.dynamics_type != "doubleintegrator":
            raise ValueError("Hp100 demonstrations require double-integrator dynamics")
        return DYN.step_torch(
            state,
            control,
            dt=float(self.config.dt),
            u_max=DYN.U_MAX,
            v_max=DYN.V_MAX,
        )


def locked_expert_config() -> dict:
    config = asdict(EXPERT.demonstration_config())
    # Preserve the historical B1 expert module as a comparator, but the fresh
    # Hp100 demonstrations use the faithful current-position tangent geometry.
    config["predict_gain"] = HPF.PREDICT_GAIN
    required = dict(
        horizon=HORIZON,
        dt=DYN.DT,
        polytope_nbase=N_BASE,
        barrier_activation_radius=SS.R_SENSE,
        predict_gain=HPF.PREDICT_GAIN,
        dynamics_type="doubleintegrator",
    )
    for key, expected in required.items():
        if config[key] != expected:
            raise RuntimeError(
                f"locked SafeMPPI expert drifted at {key}: {config[key]} != {expected}"
            )
    if tuple(map(float, config["u_min"])) != (-DYN.U_MAX, -DYN.U_MAX):
        raise RuntimeError("locked expert action lower bound disagrees with Hp100 dynamics")
    if tuple(map(float, config["u_max"])) != (DYN.U_MAX, DYN.U_MAX):
        raise RuntimeError("locked expert action upper bound disagrees with Hp100 dynamics")
    return config


def _obstacles(ped_xy) -> np.ndarray:
    positions = np.asarray(ped_xy, dtype=np.float32).reshape(-1, 2)
    return np.concatenate(
        (positions, np.full((len(positions), 1), SS.R_PED, np.float32)), axis=1
    )


def _assert_feature_matches_planner(
    hp, feature_geometry, planner_polytope, robot_xy, *, provenance="unknown"
):
    """Fail closed unless the stored Hp raster is the planner's exact geometry."""
    if planner_polytope is None:
        raise RuntimeError("locked SafeMPPI expert did not expose its nominal polytope")
    planner_geometry = dict(zip(("A", "b", "ref", "margins"), planner_polytope))
    for key in ("A", "b", "ref"):
        expected = np.asarray(planner_geometry[key], np.float32)
        actual = np.asarray(feature_geometry[key], np.float32)
        if expected.shape != actual.shape or not np.array_equal(actual, expected):
            delta = (
                float(np.max(np.abs(actual - expected)))
                if actual.shape == expected.shape else float("inf")
            )
            raise RuntimeError(
                "Hp100/planner nominal polytope mismatch "
                f"at {key} ({provenance}): max_delta={delta}"
            )

    A32 = np.asarray(feature_geometry["A"], np.float32)
    b32 = np.asarray(feature_geometry["b"], np.float32)
    ref32 = np.asarray(feature_geometry["ref"], np.float32)
    canonical_m64 = np.maximum(
        b32.astype(np.float64)
        - A32.astype(np.float64) @ ref32.astype(np.float64),
        1.0e-3,
    )
    feature_m32 = np.asarray(feature_geometry["margins"], np.float32)
    if not np.array_equal(feature_m32, canonical_m64.astype(np.float32)):
        raise RuntimeError(
            "Hp100 feature margins are not canonical float32 geometry "
            f"({provenance})"
        )
    planner_m64 = np.asarray(planner_geometry["margins"], np.float32).astype(np.float64)
    if planner_m64.shape != canonical_m64.shape:
        raise RuntimeError(f"Hp100/planner margin shape mismatch ({provenance})")
    unit_roundoff = 2.0 ** -24
    gamma3 = (3.0 * unit_roundoff) / (1.0 - 3.0 * unit_roundoff)
    scale = np.abs(b32.astype(np.float64)) + np.sum(
        np.abs(A32.astype(np.float64) * ref32.astype(np.float64)[None]), axis=1
    )
    floor_error = abs(float(np.float32(1.0e-3)) - 1.0e-3)
    bound = np.nextafter(gamma3 * scale + floor_error, np.inf)
    margin_delta = np.abs(planner_m64 - canonical_m64)
    if np.any(margin_delta > bound):
        face = int(np.argmax(margin_delta - bound))
        condition = scale[face] / max(abs(canonical_m64[face]), np.finfo(float).tiny)
        raise RuntimeError(
            "Hp100/planner margin exceeds IEEE-float32 roundoff envelope "
            f"({provenance}): face={face}, planner={planner_m64[face]}, "
            f"canonical={canonical_m64[face]}, delta={margin_delta[face]}, "
            f"bound={bound[face]}, min_margin={canonical_m64.min()}, "
            f"condition={condition}"
        )

    # Raster provenance and planner provenance are separate checks.  The
    # planner tuple and the canonical feature geometry are independently
    # rounded to float32; normalizing by a very narrow face margin can amplify
    # their sub-micrometre difference.  Reconstruct the raster from the exact
    # recomputed geometry, after proving above that this geometry matches
    # the planner before normalization.
    center = np.asarray(robot_xy, np.float64).reshape(2)
    n_theta, n_r = HPF.HP100_SHAPE
    theta = -np.pi + (np.arange(n_theta) + 0.5) * 2.0 * np.pi / n_theta
    radius = (np.arange(n_r) + 0.5) * SS.R_SENSE / n_r
    directions = np.stack((np.cos(theta), np.sin(theta)), axis=1)
    points = center[None, None] + directions[:, None] * radius[None, :, None]
    A = np.asarray(feature_geometry["A"], np.float64)
    b = np.asarray(feature_geometry["b"], np.float64)
    margins = np.asarray(feature_geometry["margins"], np.float64)
    expected_hp = ((b[None] - points.reshape(-1, 2) @ A.T) / margins[None]).min(axis=1)
    expected_hp = np.clip(expected_hp, -1.0, 1.0).reshape(n_theta, n_r).astype(np.float32)
    actual_hp = np.asarray(hp, np.float32)
    if not np.array_equal(actual_hp, expected_hp):
        delta = float(np.max(np.abs(actual_hp - expected_hp)))
        raise RuntimeError(
            "stored Hp100 raster disagrees with its persisted geometry "
            f"({provenance}): max_delta={delta}"
        )


def audit_weighted_plan(state, controls, planner_polytope, gamma: float) -> dict:
    """Recheck the MPPI weighted H10 under the planner's frozen nominal set."""
    controls = np.asarray(controls, np.float32)
    if controls.shape != (HORIZON, 2):
        raise ValueError(f"weighted plan must be [{HORIZON},2], got {controls.shape}")
    if planner_polytope is None or len(planner_polytope) < 4:
        raise ValueError("weighted-plan audit requires planner A,b,ref,margins")
    A = np.asarray(planner_polytope[0], np.float32)
    b = np.asarray(planner_polytope[1], np.float32)
    margins = np.asarray(planner_polytope[3], np.float32)
    state_t = torch.as_tensor(state, dtype=torch.float32).reshape(1, 4)
    states_t = [state_t[0]]
    for action in torch.as_tensor(controls, dtype=torch.float32):
        state_t = DYN.step_torch(
            state_t, action.reshape(1, 2), dt=DYN.DT,
            u_max=DYN.U_MAX, v_max=DYN.V_MAX,
        )
        states_t.append(state_t[0])
    states_t = torch.stack(states_t)
    h_t = CappedSafeMPPIAdapter._polytope_H(
        states_t[:, :2], torch.as_tensor(A), torch.as_tensor(b),
        torch.as_tensor(margins),
    )
    states = states_t.numpy()
    h = h_t.numpy()
    violations = h[1:] < np.float32(1.0 - float(gamma)) * h[:-1]
    first = None if not bool(violations.any()) else int(np.flatnonzero(violations)[0] + 1)
    return dict(
        states=states,
        h=h.astype(np.float32, copy=False),
        violations=violations,
        full_h_pass=bool(not violations.any()),
        first_violation_step=first,
    )


def rollout_episode(
    episode: int,
    gamma: float,
    *,
    device: str = "cpu",
    planner=None,
    T_max: int = T,
) -> tuple[list[dict], dict]:
    """Collect current weighted H10 plans and their same-polytope eligibility."""
    expert_config = locked_expert_config()
    if planner is None:
        planner = CappedSafeMPPIAdapter(**expert_config)
    humans = SS.make_humans(
        int(episode), seed=0, n_ped=N_PED, speed_range=PED_SPEED_RANGE
    )
    state = np.zeros(4, np.float32)
    control_history: list[np.ndarray] = []
    records: list[dict] = []
    collision = reached = False
    minimum_clearance = float("inf")
    goal = torch.as_tensor(SS.GOAL, dtype=torch.float32, device=device)
    for step in range(int(T_max)):
        ped_xy, ped_vel = SS.collect_humans(humans)
        ped_xy = np.asarray(ped_xy, np.float32)
        ped_vel = np.asarray(ped_vel, np.float32)
        if ped_xy.shape != (N_PED, 2) or ped_vel.shape != (N_PED, 2):
            raise ValueError(
                f"expected {N_PED} pedestrian states, got {ped_xy.shape}/{ped_vel.shape}"
            )
        clearance = float(
            np.linalg.norm(ped_xy - state[:2][None], axis=1).min() - SS.R_PED
        )
        minimum_clearance = min(minimum_clearance, clearance)
        if clearance < 0.0:
            collision = True
            break
        if float(np.linalg.norm(state[:2] - SS.GOAL)) < REACH:
            reached = True
            break

        obstacles = _obstacles(ped_xy)
        hp, feature_geometry = HPF.hp100_frame(
            state[:2],
            obstacles,
            sensing=SS.R_SENSE,
            n_base=N_BASE,
            obstacle_velocities=ped_vel,
            robot_velocity=state[2:4],
            predict_gain=float(expert_config["predict_gain"]),
            predict_tau=HORIZON * DYN.DT,
            return_geometry=True,
        )
        hp = np.asarray(hp, dtype=np.float32)
        if hp.shape != (32, 100):
            raise ValueError(f"fresh Hp100 feature must be [32,100], got {hp.shape}")
        low5 = np.asarray(HPF.low5(state, SS.GOAL, float(gamma)), np.float32)
        hist = np.asarray(HPF.hist_pad(control_history), np.float32)

        action, info = planner.plan(
            torch.as_tensor(state, dtype=torch.float32, device=device),
            goal,
            torch.as_tensor(obstacles, dtype=torch.float32, device=device),
            gamma=float(gamma),
            obstacle_velocities=torch.as_tensor(
                ped_vel, dtype=torch.float32, device=device
            ),
            seed=int(episode) * 200 + int(step),
            return_rollouts=False,
        )
        _assert_feature_matches_planner(
            hp,
            feature_geometry,
            info.get("polytope"),
            state[:2],
            provenance=f"gamma={float(gamma):g},episode={int(episode)},step={int(step)}",
        )
        mean_sequence = np.asarray(info["mean_sequence"], np.float32)
        persisted_polytope = tuple(
            feature_geometry[key] for key in ("A", "b", "ref", "margins")
        )
        plan_audit = audit_weighted_plan(
            state, mean_sequence, persisted_polytope, float(gamma)
        )
        accepted_count = int(info["num_accepted"])
        rejected_count = int(info["num_rejected"])
        candidate_count = int(info["num_candidates"])
        if candidate_count != int(expert_config["num_samples"]):
            raise RuntimeError("planner candidate count differs from locked N")
        if accepted_count + rejected_count != candidate_count:
            raise RuntimeError("planner acceptance accounting differs from N")
        if accepted_count == 0:
            reason_code = TARGET_ALL_REJECTED
        elif not plan_audit["full_h_pass"]:
            reason_code = TARGET_WEIGHTED_H10_FAILED
        else:
            reason_code = TARGET_ELIGIBLE
        action = DYN.clip_action_numpy(
            action.detach().cpu().numpy().astype(np.float32).reshape(2)
        ).astype(np.float32, copy=False)
        action_mean_error = float(np.max(np.abs(action - mean_sequence[0])))
        if action_mean_error > 2.0e-6:
            raise RuntimeError(
                "executed first action differs from the current MPPI weighted plan"
            )
        records.append(dict(
            hp=hp.copy(),
            low5=low5.copy(),
            hist=hist.copy(),
            episode=np.int64(episode),
            step=np.int64(step),
            state=state.copy(),
            ped_xy=ped_xy.copy(),
            ped_vel=ped_vel.copy(),
            executed_action=action.copy(),
            U=mean_sequence.copy(),
            target_eligible=bool(reason_code == TARGET_ELIGIBLE),
            target_reason_code=np.int8(reason_code),
            plan_candidate_count=np.int32(candidate_count),
            plan_accepted_count=np.int32(accepted_count),
            plan_rejected_count=np.int32(rejected_count),
            plan_weighted_h=plan_audit["h"].copy(),
            plan_first_violation=np.int16(
                -1 if plan_audit["first_violation_step"] is None
                else plan_audit["first_violation_step"]
            ),
            action_mean_max_abs_error=np.float32(action_mean_error),
        ))
        state = DYN.step_numpy(state, action).astype(np.float32, copy=False)
        control_history.append(action.copy())
        SS.advance_humans(humans, state)

    if not collision and not reached:
        ped_xy, _ = SS.collect_humans(humans)
        clearance = float(
            np.linalg.norm(np.asarray(ped_xy) - state[:2][None], axis=1).min()
            - SS.R_PED
        )
        minimum_clearance = min(minimum_clearance, clearance)
        collision = clearance < 0.0
        reached = bool(
            not collision and float(np.linalg.norm(state[:2] - SS.GOAL)) < REACH
        )
    eligible = sum(bool(row["target_eligible"]) for row in records)
    all_rejected = sum(
        int(row["target_reason_code"]) == TARGET_ALL_REJECTED for row in records
    )
    weighted_failed = sum(
        int(row["target_reason_code"]) == TARGET_WEIGHTED_H10_FAILED
        for row in records
    )
    return records, dict(
        episode=int(episode),
        gamma=float(gamma),
        success=bool(reached and not collision),
        collision=bool(collision),
        timeout=bool(not reached and not collision),
        steps=len(records),
        min_clearance=float(minimum_clearance),
        eligible_weighted_h10_contexts=int(eligible),
        excluded_all_rejected_contexts=int(all_rejected),
        excluded_weighted_h10_failed_contexts=int(weighted_failed),
    )


def pack_records(records: list[dict]) -> dict[str, torch.Tensor]:
    if not records:
        raise ValueError("cannot pack an empty Hp100 demonstration set")
    array_keys = (
        "hp", "low5", "hist", "U", "state", "ped_xy", "ped_vel",
        "executed_action", "plan_weighted_h",
    )
    payload = {
        key: torch.from_numpy(np.stack([row[key] for row in records])).to(torch.float32)
        for key in array_keys
    }
    payload["episode"] = torch.as_tensor(
        [row["episode"] for row in records], dtype=torch.int64
    )
    payload["step"] = torch.as_tensor(
        [row["step"] for row in records], dtype=torch.int64
    )
    payload["target_eligible"] = torch.as_tensor(
        [bool(row["target_eligible"]) for row in records], dtype=torch.bool
    )
    payload["target_reason_code"] = torch.as_tensor(
        [row["target_reason_code"] for row in records], dtype=torch.int8
    )
    payload["plan_accepted_count"] = torch.as_tensor(
        [row["plan_accepted_count"] for row in records], dtype=torch.int32
    )
    payload["plan_candidate_count"] = torch.as_tensor(
        [row["plan_candidate_count"] for row in records], dtype=torch.int32
    )
    payload["plan_rejected_count"] = torch.as_tensor(
        [row["plan_rejected_count"] for row in records], dtype=torch.int32
    )
    payload["plan_first_violation"] = torch.as_tensor(
        [row["plan_first_violation"] for row in records], dtype=torch.int16
    )
    payload["action_mean_max_abs_error"] = torch.as_tensor(
        [row["action_mean_max_abs_error"] for row in records], dtype=torch.float32
    )
    expected = dict(
        hp=(32, 100), low5=(5,), hist=(16, 2), U=(HORIZON, 2),
        state=(4,), ped_xy=(N_PED, 2), ped_vel=(N_PED, 2),
        executed_action=(2,), plan_weighted_h=(HORIZON + 1,),
    )
    for key, trailing in expected.items():
        if tuple(payload[key].shape[1:]) != trailing:
            raise ValueError(f"packed {key} has shape {tuple(payload[key].shape)}")
    if not torch.equal(payload["target_eligible"], payload["target_reason_code"] == TARGET_ELIGIBLE):
        raise RuntimeError("target eligibility differs from its reason code")
    if torch.any(payload["plan_candidate_count"] != 2048):
        raise RuntimeError("packed planner candidate count differs from 2048")
    if torch.any(
        payload["plan_accepted_count"] + payload["plan_rejected_count"]
        != payload["plan_candidate_count"]
    ):
        raise RuntimeError("packed planner candidate counts differ from 2048")
    logical_eligible = (
        (payload["plan_accepted_count"] > 0)
        & (payload["plan_first_violation"] == -1)
    )
    if not torch.equal(payload["target_eligible"], logical_eligible):
        raise RuntimeError("target mask does not match the declared logical identity")
    return payload


def _atomic_torch_save(payload, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _source_hashes() -> dict[str, dict[str, str]]:
    paths = {
        "generator": Path(__file__).resolve(),
        "dynamics": Path(inspect.getsourcefile(DYN)).resolve(),
        "features": Path(inspect.getsourcefile(HPF)).resolve(),
        "expert_config": Path(inspect.getsourcefile(EXPERT)).resolve(),
        "safemppi_adapter": Path(inspect.getsourcefile(SafeMPPIAdapter)).resolve(),
        "safemppi_barrier": Path(inspect.getsourcefile(SAFETY_BARRIER)).resolve(),
        "nominal_polytope": Path(inspect.getsourcefile(NOMINAL_POLYTOPE)).resolve(),
        "scene": Path(inspect.getsourcefile(SS)).resolve(),
        "human_agent": Path(inspect.getsourcefile(SS.HumanAgent)).resolve(),
        "human_advance": Path(inspect.getsourcefile(SS._advance_humans)).resolve(),
    }
    return {
        name: dict(path=str(path), sha256=sha256_file(path))
        for name, path in paths.items()
    }


def _git_provenance() -> dict:
    root = Path(__file__).resolve().parents[1]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    return dict(root=str(root), head=head, clean=not bool(status), status=status)


def _runtime_provenance(device: str) -> dict:
    payload = dict(
        requested_device=str(device), python=sys.version,
        numpy=np.__version__, torch=torch.__version__,
        torch_cuda=torch.version.cuda,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        cuda_device_order=os.environ.get("CUDA_DEVICE_ORDER"),
    )
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA collection requested but torch.cuda is unavailable")
        index = torch.device(device).index
        index = torch.cuda.current_device() if index is None else int(index)
        properties = torch.cuda.get_device_properties(index)
        inventory = subprocess.run(
            [
                "nvidia-smi", "--query-gpu=index,uuid,name,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True, capture_output=True, text=True,
        ).stdout.strip().splitlines()
        payload["cuda"] = dict(
            logical_index=int(index), name=str(properties.name),
            total_memory_bytes=int(properties.total_memory),
            capability=list(torch.cuda.get_device_capability(index)),
            nvidia_smi_inventory=inventory,
        )
    return payload


def _assert_provenance_unchanged(initial_git: dict, initial_hashes: dict) -> dict:
    final_git = _git_provenance()
    final_hashes = _source_hashes()
    if final_git != initial_git:
        raise RuntimeError(
            "source Git provenance changed during HP100 collection; "
            "refusing to publish a manifest"
        )
    if final_hashes != initial_hashes:
        raise RuntimeError(
            "source file hashes changed during HP100 collection; "
            "refusing to publish a manifest"
        )
    return final_git


def _collect_gamma(payload, rollout_fn=rollout_episode) -> tuple[dict, list[dict]]:
    """Worker-safe collection for one gamma; writes only its own tensor file."""
    (
        output_dir, gamma, episode_start, successes_per_gamma,
        max_attempts_per_gamma, device, T_max,
    ) = payload
    output = Path(output_dir).resolve()
    worker_runtime = _runtime_provenance(device)
    path = output / f"sfm_hp100_windows_g{float(gamma)}.pt"
    progress_path = output / f"collection_progress_g{float(gamma)}.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing data: {path}")
    if progress_path.exists():
        raise FileExistsError(f"refusing to overwrite existing progress: {progress_path}")
    accepted, summaries, successful_ids = [], [], []
    episode = int(episode_start)
    while (
        len(successful_ids) < int(successes_per_gamma)
        and len(summaries) < int(max_attempts_per_gamma)
    ):
        records, summary = rollout_fn(
            int(episode), float(gamma), device=device, T_max=int(T_max)
        )
        summary = dict(summary)
        eligible_count = int(summary.get(
            "eligible_weighted_h10_contexts",
            sum(bool(row["target_eligible"]) for row in records),
        ))
        all_rejected_count = int(summary.get(
            "excluded_all_rejected_contexts",
            sum(
                int(row["target_reason_code"]) == TARGET_ALL_REJECTED
                for row in records
            ),
        ))
        weighted_failed_count = int(summary.get(
            "excluded_weighted_h10_failed_contexts",
            sum(
                int(row["target_reason_code"]) == TARGET_WEIGHTED_H10_FAILED
                for row in records
            ),
        ))
        if eligible_count + all_rejected_count + weighted_failed_count != len(records):
            raise RuntimeError("episode target accounting differs from its context count")
        accepted_for_dataset = bool(summary["success"] and eligible_count > 0)
        summary.update(
            eligible_weighted_h10_contexts=eligible_count,
            excluded_all_rejected_contexts=all_rejected_count,
            excluded_weighted_h10_failed_contexts=weighted_failed_count,
            accepted_for_dataset=accepted_for_dataset,
        )
        summaries.append(summary)
        if accepted_for_dataset:
            accepted.extend(records)
            successful_ids.append(int(episode))
        _atomic_json_save(dict(
            status="HP100_GAMMA_COLLECTION_IN_PROGRESS",
            gamma=float(gamma), accepted_successes=len(successful_ids),
            target_successes=int(successes_per_gamma), attempted_episodes=len(summaries),
            max_attempts=int(max_attempts_per_gamma), latest_episode=int(episode),
            latest_outcome={
                key: summary[key]
                for key in (
                    "success", "collision", "timeout", "steps", "min_clearance",
                    "accepted_for_dataset", "eligible_weighted_h10_contexts",
                    "excluded_all_rejected_contexts",
                    "excluded_weighted_h10_failed_contexts",
                )
            },
        ), progress_path)
        episode += 1
    if len(successful_ids) != int(successes_per_gamma):
        raise RuntimeError(
            f"gamma {gamma} obtained {len(successful_ids)}/{successes_per_gamma} "
            f"eligible successful trajectories in {max_attempts_per_gamma} attempts"
        )
    tensors = pack_records(accepted)
    successful_episodes = sorted({int(value) for value in tensors["episode"].tolist()})
    if successful_episodes != successful_ids:
        raise RuntimeError(f"gamma {gamma} packed lineage IDs disagree with success ledger")
    eligible_windows = int(tensors["target_eligible"].sum().item())
    all_rejected_windows = int(
        (tensors["target_reason_code"] == TARGET_ALL_REJECTED).sum().item()
    )
    weighted_failed_windows = int(
        (tensors["target_reason_code"] == TARGET_WEIGHTED_H10_FAILED).sum().item()
    )
    support = {
        episode_id: int(tensors["target_eligible"][
            tensors["episode"] == int(episode_id)
        ].sum().item())
        for episode_id in successful_episodes
    }
    if any(value <= 0 for value in support.values()):
        raise RuntimeError(f"gamma {gamma} retained a lineage without eligible targets")
    packed = dict(
        schema_version=SCHEMA_VERSION, success_only=True, gamma=float(gamma),
        n_traj=len(successful_episodes), n_seeds=len(summaries),
        episode_start=int(episode_start), episode_stop_exclusive=int(episode),
        target_contract=target_contract(),
        eligible_windows=eligible_windows,
        excluded_all_rejected_windows=all_rejected_windows,
        excluded_weighted_h10_failed_windows=weighted_failed_windows,
        dynamics=DYN.contract(), **tensors,
    )
    _atomic_torch_save(packed, path)
    _atomic_json_save(dict(
        status="HP100_GAMMA_COLLECTION_COMPLETE",
        gamma=float(gamma), accepted_successes=len(successful_ids),
        target_successes=int(successes_per_gamma), attempted_episodes=len(summaries),
        episode_range=[int(episode_start), int(episode)], data_file=path.name,
        data_sha256=sha256_file(path),
        windows=len(tensors["episode"]), eligible_windows=eligible_windows,
        excluded_all_rejected_windows=all_rejected_windows,
        excluded_weighted_h10_failed_windows=weighted_failed_windows,
    ), progress_path)
    row = dict(
        gamma=float(gamma), file=path.name, sha256=sha256_file(path),
        bytes=path.stat().st_size, windows=len(tensors["episode"]),
        eligible_windows=eligible_windows,
        excluded_all_rejected_windows=all_rejected_windows,
        excluded_weighted_h10_failed_windows=weighted_failed_windows,
        n_traj=len(successful_episodes), successful_episodes=successful_episodes,
        attempted_episodes=len(summaries),
        rejected_episodes=[
            int(item["episode"])
            for item in summaries if not bool(item["accepted_for_dataset"])
        ],
        rejected_outcomes=[
            dict(
                episode=int(item["episode"]),
                success=bool(item["success"]),
                collision=bool(item["collision"]),
                timeout=bool(item["timeout"]),
                eligible_weighted_h10_contexts=int(
                    item["eligible_weighted_h10_contexts"]
                ),
            )
            for item in summaries if not bool(item["accepted_for_dataset"])
        ],
        episode_range=[int(episode_start), int(episode)],
        progress_file=progress_path.name,
        progress_sha256=sha256_file(progress_path),
        runtime=worker_runtime,
    )
    return row, summaries


def generate_dataset(
    output_dir,
    *,
    episode_start: int = EPISODE_START,
    successes_per_gamma: int = SUCCESSES_PER_GAMMA,
    max_attempts_per_gamma: int = MAX_ATTEMPTS_PER_GAMMA,
    gammas=SS.GAMMAS,
    device: str = "cpu",
    devices=None,
    T_max: int = T,
    rollout_fn=rollout_episode,
    expected_source_commit: str | None = None,
    jobs: int = 1,
) -> dict:
    """Generate per-gamma successful-only files and an authenticated manifest."""
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {manifest_path}")
    if int(successes_per_gamma) <= 0:
        raise ValueError("successes_per_gamma must be positive")
    if int(max_attempts_per_gamma) < int(successes_per_gamma):
        raise ValueError("max_attempts_per_gamma must be at least successes_per_gamma")
    git = _git_provenance()
    source_hashes = _source_hashes()
    device_values = tuple(map(str, devices)) if devices is not None else (str(device),)
    if not device_values:
        raise ValueError("at least one collection device is required")
    runtime_devices = [_runtime_provenance(value) for value in device_values]
    runtime = dict(
        requested_devices=list(device_values), devices=runtime_devices,
        requested_device=(device_values[0] if len(device_values) == 1 else None),
    )
    if expected_source_commit is not None and git["head"] != str(expected_source_commit):
        raise RuntimeError(
            f"source commit {git['head']} != expected {expected_source_commit}"
        )
    canonical_request = bool(
        int(episode_start) == EPISODE_START
        and int(successes_per_gamma) == SUCCESSES_PER_GAMMA
        and tuple(map(float, gammas)) == tuple(map(float, SS.GAMMAS))
        and int(T_max) == T
    )
    if canonical_request and not git["clean"]:
        raise RuntimeError("canonical HP100 collection requires a clean frozen worktree")
    if canonical_request and expected_source_commit is None:
        raise RuntimeError("canonical HP100 collection requires --expected-source-commit")
    gamma_values = tuple(map(float, gammas))
    if int(jobs) < 1:
        raise ValueError("jobs must be positive")
    gamma_device_map = {
        str(gamma): device_values[index % len(device_values)]
        for index, gamma in enumerate(gamma_values)
    }
    worker_payloads = [(
        str(output), gamma, int(episode_start), int(successes_per_gamma),
        int(max_attempts_per_gamma), gamma_device_map[str(gamma)], int(T_max),
    ) for gamma in gamma_values]
    if int(jobs) > 1:
        if rollout_fn is not rollout_episode:
            raise ValueError("parallel collection requires the canonical rollout function")
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(int(jobs), len(worker_payloads)), mp_context=context,
        ) as executor:
            results = list(executor.map(_collect_gamma, worker_payloads))
    else:
        results = [
            _collect_gamma(worker_payload, rollout_fn=rollout_fn)
            for worker_payload in worker_payloads
        ]
    results.sort(key=lambda result: result[0]["gamma"])
    final_git = _assert_provenance_unchanged(git, source_hashes)
    file_rows = [result[0] for result in results]
    rollout_summaries = {
        str(result[0]["gamma"]): result[1] for result in results
    }
    manifest = dict(
        status="HP100_ID_DATASET_COMPLETE",
        schema_version=SCHEMA_VERSION,
        canonical_full_run=bool(
            canonical_request
            and sum(row["n_traj"] for row in file_rows)
            == SUCCESSES_PER_GAMMA * len(SS.GAMMAS)
        ),
        role="successful SafeMPPI ID demonstrations for fresh Hp100 pretraining",
        total_successful_lineages=sum(row["n_traj"] for row in file_rows),
        total_context_rows=sum(row["windows"] for row in file_rows),
        total_eligible_windows=sum(row["eligible_windows"] for row in file_rows),
        target_contract=target_contract(),
        episode_allocation=dict(
            start=int(episode_start),
            successful_trajectories_per_gamma=int(successes_per_gamma),
            max_attempts_per_gamma=int(max_attempts_per_gamma),
            terminal_ranges={
                str(row["gamma"]): row["episode_range"] for row in file_rows
            },
        ),
        environment=dict(
            n_ped=N_PED,
            ped_speed_range=list(PED_SPEED_RANGE),
            gammas=list(map(float, gammas)),
            horizon=HORIZON,
            T=int(T_max),
            goal=np.asarray(SS.GOAL, float).tolist(),
            task_bounds=[float(SS.TASK_LO), float(SS.TASK_HI)],
            pedestrian_radius=float(SS.R_PED),
            sensing_radius=float(SS.R_SENSE),
        ),
        expert=dict(
            name=HP100_EXPERT_NAME,
            config=locked_expert_config(),
            execution=(
                "CappedSafeMPPIAdapter using sfm_hp100_dynamics for internal and real "
                "steps; current-position tangent nominal geometry (predict_gain=0)"
            ),
            supervised_target=SUPERVISED_TARGET,
        ),
        feature=dict(
            shape=[32, 100],
            dtype="float32",
            temporal_storage="current Hp frame; Hp10 history is built trajectory-locally by the loader",
            radial_bins=100,
            angular_bins=32,
            radial_pooling="none",
            nominal_polytope_n_base=N_BASE,
            velocity_aware=False,
            current_position_tangent=True,
            predict_gain=float(locked_expert_config()["predict_gain"]),
            predict_tau=float(HORIZON * DYN.DT),
            planner_geometry_runtime_assertion="A,b,ref,margins and raster checked at every context",
            construction="fresh from raw state and pedestrian geometry; no interpolation or old-grid upsample",
            contract=(HPF.contract() if hasattr(HPF, "contract") else None),
        ),
        dynamics=DYN.contract(),
        files=file_rows,
        source_hashes=source_hashes,
        source_git=git,
        source_completion_audit=dict(
            git=final_git,
            source_hashes_equal=True,
            manifest_published_only_after_all_workers_completed=True,
        ),
        runtime=runtime,
        parallelism=dict(
            jobs=int(jobs), start_method=("spawn" if int(jobs) > 1 else "none"),
            devices=list(device_values), gamma_device_map=gamma_device_map,
            gamma_workers_are_independent=True,
        ),
        rollout_summaries=rollout_summaries,
    )
    temporary = manifest_path.with_suffix(".json.tmp")
    with open(temporary, "w") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
    os.replace(temporary, manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episode-start", type=int, default=EPISODE_START)
    parser.add_argument(
        "--successes-per-gamma", type=int, default=SUCCESSES_PER_GAMMA
    )
    parser.add_argument(
        "--max-attempts-per-gamma", type=int, default=MAX_ATTEMPTS_PER_GAMMA
    )
    parser.add_argument("--gammas", type=float, nargs="+", default=SS.GAMMAS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--devices", nargs="+", default=None,
        help="round-robin gamma workers over these logical devices",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--expected-source-commit", default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="fill one successful gamma=0.5 trajectory; still uses T=180 and the locked expert",
    )
    args = parser.parse_args()
    if args.smoke:
        args.episode_start = EPISODE_START
        args.successes_per_gamma = 1
        args.max_attempts_per_gamma = 20
        args.gammas = [0.5]
    manifest = generate_dataset(
        args.output_dir,
        episode_start=args.episode_start,
        successes_per_gamma=args.successes_per_gamma,
        max_attempts_per_gamma=args.max_attempts_per_gamma,
        gammas=args.gammas,
        device=args.device,
        devices=args.devices,
        expected_source_commit=args.expected_source_commit,
        jobs=args.jobs,
    )
    print(json.dumps({
        "status": manifest["status"],
        "manifest": str(Path(args.output_dir).resolve() / "manifest.json"),
        "canonical_full_run": manifest["canonical_full_run"],
    }), flush=True)


if __name__ == "__main__":
    main()
