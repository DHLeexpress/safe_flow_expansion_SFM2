"""Deterministic moving-pedestrian SFM scene used by Hp10/B1."""
from __future__ import annotations

import numpy as np

import _paths  # noqa: F401
from cfm_mppi.utils import HumanAgent
from cfm_mppi.evaluation.render_sfm_kazuki_policy import _advance_humans, _collect_humans

GOAL = np.array([6.0, 6.0], dtype=np.float32)
R_PED = 0.2
U_MAX = 2.0
DT = 0.1
R_SENSE = 2.0
TASK_LO = -0.5
TASK_HI = 6.5
GAMMAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
LEGACY_PED_SPEED_RANGE = (0.8, 1.3)
ID_PED_SPEED_RANGE = (0.5, 1.0)
OOD_PED_SPEED_RANGE = (1.0, 1.5)
DOUBLE_OOD_PED_SPEED_RANGE = (1.0, 2.0)

# These names are part of the evaluation contract written into every result.
# The requested ``id`` benchmark matches the training speed range, but uses
# fewer pedestrians than the training data; that distinction is kept explicit
# rather than silently treating it as the full training distribution.
SCENE_PROFILES = {
    "training": dict(
        n_ped=20,
        ped_speed_range=ID_PED_SPEED_RANGE,
        role="pretraining_data_distribution",
        shift_from_training="none",
    ),
    "matched_id": dict(
        n_ped=20,
        ped_speed_range=ID_PED_SPEED_RANGE,
        role="matched_in_distribution_benchmark",
        shift_from_training="none; evaluation-only alias with the training density and speed range",
    ),
    "id": dict(
        n_ped=10,
        ped_speed_range=ID_PED_SPEED_RANGE,
        role="requested_in_distribution_benchmark",
        shift_from_training="lower pedestrian count (10 versus 20); speed range unchanged",
    ),
    "requested_ood": dict(
        n_ped=30,
        ped_speed_range=OOD_PED_SPEED_RANGE,
        role="requested_density_and_velocity_ood_benchmark",
        shift_from_training="higher pedestrian count (30 versus 20) and speed range 1.0-1.5 versus 0.5-1.0 m/s",
    ),
    "double_density_velocity_ood": dict(
        n_ped=40,
        ped_speed_range=DOUBLE_OOD_PED_SPEED_RANGE,
        role="twofold_density_and_velocity_ood_benchmark",
        shift_from_training=("pedestrian count 40 versus 20 and speed range 1.0-2.0 "
                             "versus 0.5-1.0 m/s"),
    ),
    "density_ood": dict(
        n_ped=50,
        ped_speed_range=ID_PED_SPEED_RANGE,
        role="requested_density_only_ood_benchmark",
        shift_from_training="higher pedestrian count (50 versus 20); speed range unchanged",
    ),
    "legacy_velocity_ood": dict(
        n_ped=20,
        ped_speed_range=OOD_PED_SPEED_RANGE,
        role="legacy_103476d_velocity_ood_benchmark",
        shift_from_training="pedestrian count unchanged; speed range 1.0-1.5 versus 0.5-1.0 m/s",
    ),
}
SCIENTIFIC_EVAL_PROFILES = (
    "matched_id", "id", "density_ood", "requested_ood", "double_density_velocity_ood",
    "legacy_velocity_ood",
)


def scene_profile(name):
    """Return a JSON-native, complete environment contract for ``name``."""
    if name not in SCENE_PROFILES:
        raise ValueError(f"unknown scene profile {name!r}; choose from {tuple(SCENE_PROFILES)}")
    profile = SCENE_PROFILES[name]
    return dict(
        scene_profile=str(name),
        n_ped=int(profile["n_ped"]),
        ped_speed_range=list(map(float, profile["ped_speed_range"])),
        role=str(profile["role"]),
        shift_from_training=str(profile["shift_from_training"]),
        training_reference=dict(n_ped=20, ped_speed_range=list(map(float, ID_PED_SPEED_RANGE))),
        goal=GOAL.astype(float).tolist(),
        pedestrian_radius=float(R_PED),
        dt=float(DT),
        sensing_radius=float(R_SENSE),
        task_bounds=[float(TASK_LO), float(TASK_HI)],
    )


def _validate_speed_range(speed_range):
    lo, hi = map(float, speed_range)
    if not (0.0 <= lo < hi):
        raise ValueError(f"invalid pedestrian speed range: {speed_range}")
    return lo, hi


def _remap_human_speed(human, speed_range):
    lo, hi = _validate_speed_range(speed_range)
    q = (float(human.sfm_des_speed) - LEGACY_PED_SPEED_RANGE[0]) / (
        LEGACY_PED_SPEED_RANGE[1] - LEGACY_PED_SPEED_RANGE[0]
    )
    human.sfm_des_speed = float(lo + np.clip(q, 0.0, 1.0) * (hi - lo))
    return human


def make_humans(scenario_id, seed=0, n_ped=20, speed_range=OOD_PED_SPEED_RANGE):
    """A scenario ID fixes starts, goals, and pedestrian speed quantiles."""
    rng = np.random.RandomState(int(seed) + int(scenario_id))
    return [_remap_human_speed(HumanAgent(GOAL, random_generator=rng), speed_range) for _ in range(n_ped)]


def collect_humans(humans):
    return _collect_humans(humans)


def advance_humans(humans, robot_state):
    _advance_humans(
        humans,
        robot_xy=np.asarray(robot_state[:2], np.float32).copy(),
        robot_control_si=np.asarray(robot_state[2:4], np.float32).copy(),
    )


def scenario_snapshot(scenario_ids, *, n_ped=20, seed=0, speed_range=OOD_PED_SPEED_RANGE):
    rows = []
    for scenario_id in map(int, scenario_ids):
        humans = make_humans(scenario_id, seed, n_ped, speed_range)
        xy, vel = collect_humans(humans)
        rows.append(dict(scenario_id=scenario_id, ped_xy=xy.tolist(), ped_vel=vel.tolist()))
    return dict(seed=int(seed), n_ped=int(n_ped), speed_range=list(map(float, speed_range)), rows=rows)
