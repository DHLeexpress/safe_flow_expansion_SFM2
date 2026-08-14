"""Exact raw-policy branch visualization for the promoted SFM HP100 model.

This is a read-only presentation pipeline.  It consumes two completed raw-M50
evaluations and their declared common-random-number bank, deterministically
selects the first success and first collision at gamma=0.5 in each scene, and
replays only those four cells.  A raw H=10 proposal is blue exactly when the
canonical full-H moving-pedestrian verifier accepts it and red otherwise.

The proposal label is deliberately distinct from the trajectory-level
``Validity`` stored by :mod:`sfm_hp100_eval`: the former labels the current raw
H=10 branch, while the latter averages terminal-truncated windows along the
closed-loop trajectory.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

import _paths  # noqa: F401
import grid_policy_sfm_hp100 as GPS
import sfm_b1_viz as BV
import sfm_hp100_dynamics as DYN
import sfm_hp100_eval as RAW
import sfm_hp100_features as HPF
import sfm_metrics2 as VERIFY
import sfm_scene as SS


STATUS = "SFM_HP100_ID_OOD_BRANCH_VIZ_COMPLETE"
TRACE_STATUS = "SFM_HP100_ID_OOD_BRANCH_TRACE_COMPLETE"
DISPLAY_GAMMA = 0.5
M_PER_GAMMA = 50
SCENES = {
    "id": dict(profile="matched_id", ep0=150_000),
    "ood": dict(profile="double_density_velocity_ood", ep0=250_000),
}
CASE_ORDER = ("id_success", "id_failure", "ood_success", "ood_failure")
BLUE = "#0072B2"
RED = "#D62728"
GREEN = "#009E73"
GRAY = "#8C8C8C"
BLACK = "#111111"


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return RAW.array_sha256(value)


def _read_json(path) -> dict:
    with open(path) as stream:
        return json.load(stream)


def _noise_seed(payload: dict) -> int:
    bank = payload.get("noise_bank", {})
    value = bank.get("seed", payload.get("noise_seed"))
    if value is None:
        raise ValueError(
            "raw M50 artifact does not declare its CRN noise seed; exact replay "
            "would be unauthenticated"
        )
    return int(value)


def validate_evaluation(
    payload: dict,
    *,
    scene_profile: str,
    ep0: int,
    checkpoint_sha256: str,
    policy_d: int,
) -> np.ndarray:
    """Validate one fixed raw-M50 artifact and reconstruct its exact CRN bank."""
    if payload.get("status") != "SFM_HP100_RAW_EVAL_COMPLETE":
        raise ValueError("not a completed SFM HP100 raw evaluation")
    if payload.get("version") != RAW.VERSION:
        raise ValueError("raw evaluation version differs from the renderer")
    if payload.get("evaluator_sha256") != sha256_file(inspect.getsourcefile(RAW)):
        raise ValueError("raw evaluation source hash differs from the renderer")
    if payload.get("checkpoint_sha256") != str(checkpoint_sha256):
        raise ValueError("raw evaluation and promoted checkpoint digests differ")
    scene = payload.get("scene", {})
    if scene != SS.scene_profile(scene_profile):
        raise ValueError("raw evaluation scene contract differs from the requested case")
    if payload.get("dynamics") != DYN.contract():
        raise ValueError("raw evaluation dynamics contract differs from the renderer")
    if payload.get("observation") != HPF.contract():
        raise ValueError("raw evaluation observation contract differs from the renderer")
    verifier = payload.get("verifier", {})
    if verifier.get("contract") != VERIFY.verifier_manifest():
        raise ValueError("raw evaluation verifier contract differs from the renderer")
    if verifier.get("evaluator_sha256") != sha256_file(inspect.getsourcefile(VERIFY)):
        raise ValueError("raw evaluation verifier source hash differs")
    if verifier.get("polytope_sha256") != sha256_file(
        inspect.getsourcefile(VERIFY.VP)
    ):
        raise ValueError("raw evaluation polytope source hash differs")
    if int(payload.get("ep0", -1)) != int(ep0):
        raise ValueError("raw evaluation uses the wrong fixed episode bank")
    if int(payload.get("M_per_gamma", -1)) != M_PER_GAMMA:
        raise ValueError("branch selection requires the fixed M=50/gamma bank")
    if float(payload.get("temperature", float("nan"))) != 1.0:
        raise ValueError("branch selection requires raw temperature-one evaluation")
    if int(payload.get("NFE", -1)) != RAW.NFE:
        raise ValueError("raw evaluation NFE differs from the replay integrator")

    rows = list(payload.get("rows", ()))
    expected_rows = len(SS.GAMMAS) * M_PER_GAMMA
    if len(rows) != expected_rows:
        raise ValueError(f"raw M50 artifact has {len(rows)} rows, expected {expected_rows}")
    keys = [(round(float(row["gamma"]), 8), int(row["episode"])) for row in rows]
    if len(set(keys)) != expected_rows:
        raise ValueError("raw M50 artifact has duplicate gamma/episode cells")
    expected_keys = {
        (round(float(gamma), 8), int(ep0) + rollout)
        for gamma in SS.GAMMAS for rollout in range(M_PER_GAMMA)
    }
    if set(keys) != expected_keys:
        raise ValueError("raw M50 artifact does not cover the declared CRN cells")
    for row in rows:
        status = str(row.get("status"))
        if status not in ("success", "collision", "timeout"):
            raise ValueError(f"invalid raw outcome {status!r}")

    seed = _noise_seed(payload)
    if "noise_seed" in payload and "seed" in payload.get("noise_bank", {}):
        if int(payload["noise_seed"]) != int(payload["noise_bank"]["seed"]):
            raise ValueError("raw evaluation declares conflicting CRN seeds")
    noise = RAW.noise_bank(M=M_PER_GAMMA, d=int(policy_d), seed=seed)
    bank = payload.get("noise_bank", {})
    if bank:
        if "shape" in bank and list(map(int, bank["shape"])) != list(noise.shape):
            raise ValueError("declared raw noise-bank shape is inconsistent")
        if "dtype" in bank and str(bank["dtype"]) != str(noise.dtype):
            raise ValueError("declared raw noise-bank dtype is inconsistent")
        if "sha256" in bank and str(bank["sha256"]) != array_sha256(noise):
            raise ValueError("reconstructed raw noise bank fails its SHA-256 contract")
    return noise


def select_cases(payload: dict, *, gamma: float = DISPLAY_GAMMA) -> dict:
    """Choose the first success and first collision; timeout is fallback only."""
    rows = sorted(
        (
            row for row in payload["rows"]
            if abs(float(row["gamma"]) - float(gamma)) < 1.0e-8
        ),
        key=lambda row: int(row["episode"]),
    )
    if len(rows) != M_PER_GAMMA:
        raise ValueError("gamma=0.5 raw cell does not contain exactly M=50 rows")
    successes = [row for row in rows if row["status"] == "success"]
    collisions = [row for row in rows if row["status"] == "collision"]
    timeouts = [row for row in rows if row["status"] == "timeout"]
    if not successes:
        raise ValueError("fixed M50 gamma=0.5 bank contains no success case")
    if collisions:
        failure = collisions[0]
        failure_selection = "first_collision"
    elif timeouts:
        failure = timeouts[0]
        failure_selection = "timeout_fallback_no_collision_in_fixed_bank"
    else:
        raise ValueError("fixed M50 gamma=0.5 bank contains no failure case")
    return dict(
        success=successes[0], failure=failure,
        success_selection="first_success",
        failure_selection=failure_selection,
    )


def load_promoted_policy(path, *, device: str):
    """Authenticate a canonical ID-promoted from-scratch checkpoint."""
    path = Path(path).resolve()
    before = sha256_file(path)
    policy, checkpoint = GPS.load_sfm_hp100_policy(path, device=device)
    after = sha256_file(path)
    if before != after:
        raise RuntimeError("promoted checkpoint bytes changed during load")
    if checkpoint.get("scientific_status") != "canonical_ID_promoted":
        raise ValueError("branch renderer requires a canonical ID-promoted checkpoint")
    if checkpoint.get("pretrained_from_scratch") is not True:
        raise ValueError("branch renderer requires the declared from-scratch HP100 model")
    gate = checkpoint.get("selected_gate", {})
    if gate.get("distribution") != "ID" or float(
        gate.get("temperature", float("nan"))
    ) != 1.0:
        raise ValueError("promoted checkpoint lacks its ID raw-temperature-one gate")
    return policy.eval(), checkpoint, before


def verify_raw_proposal(state, controls, ped_xy, ped_vel, gamma) -> dict:
    """Use the raw evaluator label and retain the exact candidate geometry."""
    controls = DYN.clip_action_numpy(
        np.asarray(controls, np.float32).reshape(RAW.H, 2)
    ).astype(np.float32, copy=False)
    canonical = RAW.verify_executed_window(
        state, controls, ped_xy, ped_vel, gamma,
    )
    if not canonical.get("resolved", False):
        return canonical
    segment = RAW.clipped_rollout_positions(state, controls)
    pedestrians = VERIFY.predict_pedestrians(ped_xy, ped_vel, H=RAW.H)
    certificate, faces, diagnostics = VERIFY.certify_moving_window(
        segment, pedestrians, gamma,
    )
    taskspace = VERIFY.taskspace_ok(segment)
    collision_free = VERIFY.collision_free_time_indexed(segment, pedestrians)
    label = int(taskspace and collision_free and certificate)
    if label != int(canonical["y"]):
        raise RuntimeError("raw evaluator label and retained exact geometry disagree")
    result = dict(canonical)
    result.update(
        y=label, taskspace=bool(taskspace),
        collision_free=bool(collision_free), certificate=bool(certificate),
        segment=segment, pedestrian_prediction=pedestrians, faces=faces,
        diagnostics=diagnostics, full_h=True, terminal_step=RAW.H,
    )
    return result


def _same_metric(actual, expected, *, name: str) -> None:
    if expected is None:
        if actual is not None:
            raise RuntimeError(f"batched replay changed null metric {name}")
        return
    if not np.isclose(float(actual), float(expected), rtol=0.0, atol=2.0e-7):
        raise RuntimeError(
            f"batched replay changed {name}: {actual} != {expected}"
        )


def trace_case(
    replay_row: dict,
    *,
    scene_profile: str,
    rollout_index: int,
    expected_row: dict,
) -> dict:
    """Convert one cell from a replay of the complete original M50 batch.

    Keeping the original active-batch composition matters: replaying a selected
    cell at batch size one changes floating-point reduction order and therefore
    is not an exact reproduction of the official raw evaluation.
    """
    steps = int(replay_row["steps"])
    if str(replay_row["status"]) != str(expected_row["status"]):
        raise RuntimeError("full-batch replay outcome differs from the M50 row")
    if steps != int(expected_row["steps"]):
        raise RuntimeError("full-batch replay step count differs from the M50 row")
    for name in (
        "min_clearance", "successful_clearance", "time_to_goal",
    ):
        _same_metric(replay_row.get(name), expected_row.get(name), name=name)
    required_shapes = dict(
        states=(steps + 1, 4), controls=(steps, 2),
        proposals=(steps, RAW.H, 2), ped_xy=(steps, -1, 2),
        ped_vel=(steps, -1, 2),
    )
    for name, shape in required_shapes.items():
        value = np.asarray(replay_row[name])
        if len(shape) != value.ndim or any(
            expected != -1 and actual != expected
            for actual, expected in zip(value.shape, shape)
        ):
            raise RuntimeError(
                f"full-batch replay {name} shape {value.shape} != {shape}"
            )

    verified_episode = RAW._verify_executed_episode(replay_row)
    _same_metric(
        verified_episode["validity"], expected_row["validity"], name="validity",
    )
    if int(verified_episode["valid_windows"]) != int(expected_row["valid_windows"]):
        raise RuntimeError("full-batch replay changed the trajectory-valid window count")

    traces = []
    for step in range(steps):
        state = np.asarray(replay_row["states"][step], np.float32)
        next_state = np.asarray(replay_row["states"][step + 1], np.float32)
        proposal = np.asarray(replay_row["proposals"][step], np.float32)
        action = DYN.clip_action_numpy(proposal[0]).astype(np.float32, copy=False)
        if not np.array_equal(action, np.asarray(replay_row["controls"][step])):
            raise RuntimeError("retained raw proposal does not reproduce its executed action")
        reproduced = DYN.step_numpy(state, action).astype(np.float32, copy=False)
        if not np.array_equal(reproduced, next_state):
            raise RuntimeError("retained raw proposal does not reproduce its next state")
        ped_xy = np.asarray(replay_row["ped_xy"][step], np.float32)
        ped_vel = np.asarray(replay_row["ped_vel"][step], np.float32)
        result = verify_raw_proposal(
            state, proposal, ped_xy, ped_vel, float(replay_row["gamma"]),
        )
        if not result.get("resolved", False):
            raise RuntimeError(result.get("error", "raw proposal verification failed"))
        traces.append(dict(
            step=int(step), state=state.copy(), next_state=next_state.copy(),
            gamma=float(replay_row["gamma"]), ped_xy=ped_xy.copy(),
            ped_vel=ped_vel.copy(), proposal_controls=proposal.copy(),
            proposal_result=result,
            proposal_label=(
                "full_h_positive" if int(result["y"]) else "full_h_negative"
            ),
        ))
    terminal_snapshot = None
    if not steps:
        environment = SS.scene_profile(scene_profile)
        humans = SS.make_humans(
            int(replay_row["episode"]), seed=0,
            n_ped=int(environment["n_ped"]),
            speed_range=tuple(environment["ped_speed_range"]),
        )
        ped_xy, ped_vel = SS.collect_humans(humans)
        terminal_snapshot = dict(
            state=np.asarray(replay_row["states"][0], np.float32),
            ped_xy=np.asarray(ped_xy, np.float32),
            ped_vel=np.asarray(ped_vel, np.float32),
            gamma=float(replay_row["gamma"]),
        )
    if traces:
        traces[-1]["episode_outcome"] = str(replay_row["status"])
    positives = sum(row["proposal_label"] == "full_h_positive" for row in traces)
    return dict(
        scene_profile=str(scene_profile), environment=SS.scene_profile(scene_profile),
        episode=int(replay_row["episode"]), rollout_index=int(rollout_index),
        gamma=float(replay_row["gamma"]), outcome=str(replay_row["status"]),
        traces=traces, terminal_snapshot=terminal_snapshot,
        proposal_full_h_positive=int(positives),
        proposal_full_h_negative=int(len(traces) - positives),
        proposal_positive_fraction=(positives / len(traces) if traces else None),
        trajectory_validity=float(verified_episode["validity"]),
        metrics_row=dict(expected_row),
    )


def verifier_geometry(trace: dict) -> dict:
    """Build the candidate-specific H1..H10 GREEN geometry with face audit."""
    result = trace["proposal_result"]
    if int(result.get("y", 0)) != 1 or not result.get("resolved", False):
        raise ValueError("GREEN geometry is defined only for a positive raw proposal")
    diagnostics = result["diagnostics"]
    if diagnostics.get("solver") != "paper_static_exact_2d_angular_interval_socp":
        raise ValueError("GREEN geometry requires the exact analytic solver")
    if int(diagnostics.get("K_artificial", -1)) != VERIFY.ARTIFICIAL_FACES:
        raise ValueError("GREEN geometry requires exactly 16 artificial faces")
    faces = [face for face in result["faces"] if bool(face.feasible)]
    artificial = [face for face in faces if face.kind == "artificial"]
    if len(artificial) != VERIFY.ARTIFICIAL_FACES:
        raise ValueError("positive certificate does not retain 16 artificial faces")
    segment = np.asarray(result["segment"], float)
    pedestrians = np.asarray(result["pedestrian_prediction"], float)
    radius = float(diagnostics["R_eff"])
    expected_real = {
        f"ped{index}" for index in range(pedestrians.shape[1])
        if float(np.linalg.norm(
            pedestrians[0, index] - segment[0],
        ) - SS.R_PED) <= radius
    }
    actual_real = {face.label for face in faces if face.kind == "real"}
    if expected_real != actual_real:
        raise RuntimeError("GREEN face set differs from the current-sensed pedestrians")
    A = np.stack([np.asarray(face.a, float) for face in faces])
    margins = np.asarray([float(face.m) for face in faces])
    center = segment[0]
    outer = BV.halfspace_polygon(A, A @ center + margins)
    if outer is None:
        raise RuntimeError("positive GREEN faces do not define a bounded polytope")
    return dict(
        levels=BV._level_polygons(
            A, margins, center, float(trace["gamma"]), H=RAW.H,
        ),
        outer=outer, real_face_labels=sorted(expected_real),
        artificial_faces=len(artificial), radius=radius,
    )


def _draw_scene(axis, trace):
    BV._draw_pedestrians(axis, trace["ped_xy"], alpha=.72)
    ped_xy = np.asarray(trace["ped_xy"], float)
    ped_vel = np.asarray(trace["ped_vel"], float)
    for index in range(len(ped_xy)):
        prediction = ped_xy[index][None] + (
            np.arange(RAW.H + 1)[:, None] * SS.DT * ped_vel[index][None]
        )
        axis.plot(
            prediction[:, 0], prediction[:, 1], ".--", color=GRAY,
            ms=1.7, lw=.5, alpha=.52, zorder=2,
        )
    axis.plot(SS.GOAL[0], SS.GOAL[1], marker="*", ms=10,
              color="#F0A202", mec="#7A4E00", mew=.5, zorder=12)
    BV._set_world_frame(axis)
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.tick_params(labelbottom=False, labelleft=False, length=0)


def _draw_green(axis, geometry):
    BV._draw_level_polygons(
        axis, geometry["levels"], color=GREEN,
        linewidth=.48, alpha=.65, zorder=2.7,
    )
    outer = geometry["outer"]
    axis.plot(
        np.r_[outer[:, 0], outer[0, 0]],
        np.r_[outer[:, 1], outer[0, 1]],
        color=GREEN, lw=1.2, alpha=.95, zorder=3,
    )


def draw_case(axis, case: dict, step: int):
    traces = case["traces"]
    if not traces:
        snapshot = case.get("terminal_snapshot")
        if snapshot is None:
            raise ValueError("zero-step case lacks its terminal scene snapshot")
        _draw_scene(axis, snapshot)
        position = np.asarray(snapshot["state"], float)[:2]
        axis.plot(position[0], position[1], "o", color=BLACK, ms=4, zorder=10)
        axis.text(
            .5, .04, "initial terminal event · no raw proposal",
            transform=axis.transAxes, ha="center", va="bottom", fontsize=7,
        )
        axis.set_title("")
        return None
    current_index = min(max(int(step), 0), len(traces) - 1)
    current = traces[current_index]
    _draw_scene(axis, current)
    for index, trace in enumerate(traces[:current_index + 1]):
        path = np.asarray(trace["proposal_result"]["segment"], float)
        positive = trace["proposal_label"] == "full_h_positive"
        color = BLUE if positive else RED
        is_current = index == current_index
        axis.plot(
            path[:, 0], path[:, 1], color=color,
            lw=1.35 if is_current else .48,
            marker=".", ms=1.7 if is_current else .7,
            alpha=.95 if is_current else .24, zorder=6 if is_current else 3,
        )
        if not positive:
            # The analytic solver can reject because an entire face is
            # infeasible, in which case there is no meaningful temporal
            # ``worst_t``.  Keep the visual convention honest and stable by
            # placing every rejection marker at the branch endpoint.
            axis.plot(path[-1, 0], path[-1, 1], "x", color=RED,
                      ms=4.2 if is_current else 2.2, mew=1.0, zorder=8)
    states = [np.asarray(row["state"], float)[:2]
              for row in traces[:current_index + 1]]
    states.append(np.asarray(current["next_state"], float)[:2])
    states = np.asarray(states)
    axis.plot(states[:, 0], states[:, 1], color=BLACK, lw=2.2,
              marker=".", ms=1.7, alpha=.98, zorder=10)
    if current["proposal_label"] == "full_h_positive":
        _draw_green(axis, verifier_geometry(current))
    axis.set_title("")
    return current_index


def _case_label(key: str, case: dict) -> str:
    distribution, role = key.split("_", 1)
    return (
        f"{distribution.upper()} · {role} · ep {case['episode']}\n"
        f"outcome={case['outcome']} · trajectory Validity="
        f"{case['trajectory_validity']:.3f}"
    )


def render_bundle(
    bundle: dict,
    *,
    output_mp4: str,
    output_png: str,
    output_dir: str,
    fps: int = 5,
    frame_stride: int = 2,
) -> dict:
    if int(fps) <= 0 or int(frame_stride) <= 0:
        raise ValueError("fps and frame_stride must be positive")
    cases = bundle["cases"]
    if tuple(cases) != CASE_ORDER:
        raise ValueError("branch bundle case order changed")
    maximum = max(max(0, len(cases[key]["traces"]) - 1) for key in CASE_ORDER)
    frames = list(range(0, maximum + 1, int(frame_stride)))
    if frames[-1] != maximum:
        frames.append(maximum)
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 10.2))
    figure.subplots_adjust(left=.04, right=.78, bottom=.04, top=.95,
                           wspace=.035, hspace=.10)
    positions = {
        "id_success": (0, 0), "id_failure": (0, 1),
        "ood_success": (1, 0), "ood_failure": (1, 1),
    }
    for key, (row, column) in positions.items():
        axes[row, column].text(
            .5, 1.015, _case_label(key, cases[key]),
            transform=axes[row, column].transAxes, ha="center", va="bottom",
            fontsize=8,
        )
    figure.legend(
        handles=[
            Line2D([], [], color=BLUE, lw=1.5, label="raw H10: exact full-H positive"),
            Line2D([], [], color=RED, lw=1.5, marker="x", label="raw H10: exact full-H negative"),
            Line2D([], [], color=GREEN, lw=.8, label="candidate-specific GREEN h=1..10"),
            Line2D([], [], color=GREEN, lw=1.4, label="GREEN outer set · exact K=16"),
            Line2D([], [], color=BLACK, lw=2.3, label="closed-loop first-action trajectory"),
            Line2D([], [], color=GRAY, lw=.6, ls="--", label="pedestrian constant-velocity prediction"),
        ],
        loc="center left", bbox_to_anchor=(.79, .63), frameon=False, fontsize=8,
    )
    figure.text(
        .79, .40,
        "Fixed gamma = 0.5\n"
        "First success/collision in each fixed M50 bank.\n"
        "Timeout appears only if no collision exists.\n\n"
        "Branch color: current raw proposal H10 label.\n"
        "Validity in titles: trajectory window-average.\n"
        "These are different quantities.",
        ha="left", va="top", fontsize=8,
    )

    def update(step):
        for key, (row, column) in positions.items():
            axis = axes[row, column]
            axis.clear()
            draw_case(axis, cases[key], int(step))
            axis.text(
                .5, 1.015, _case_label(key, cases[key]),
                transform=axis.transAxes, ha="center", va="bottom", fontsize=8,
            )
        figure.suptitle(f"raw HP100 branch audit · frame step {int(step)}", fontsize=11)
        return []

    for path in (output_mp4, output_png):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    movie = animation.FuncAnimation(
        figure, update, frames=frames, interval=1000 / int(fps), blit=False,
    )
    movie.save(
        output_mp4, writer=animation.FFMpegWriter(fps=int(fps), bitrate=4200),
        dpi=105,
    )
    update(maximum)
    figure.savefig(output_png, dpi=170, bbox_inches="tight")
    plt.close(figure)

    os.makedirs(output_dir, exist_ok=True)
    snapshots = {}
    for key in CASE_ORDER:
        figure, axis = plt.subplots(figsize=(5.4, 5.2))
        final_step = max(0, len(cases[key]["traces"]) - 1)
        draw_case(axis, cases[key], final_step)
        axis.set_title(_case_label(key, cases[key]), fontsize=9)
        path = os.path.join(output_dir, f"{key}_final_branch.png")
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        snapshots[key] = os.path.abspath(path)
    return dict(
        frames=frames, mp4=os.path.abspath(output_mp4),
        png=os.path.abspath(output_png), snapshots=snapshots,
    )


def build_bundle(
    checkpoint_path: str,
    id_eval_path: str,
    ood_eval_path: str,
    *,
    device: str,
) -> dict:
    policy, checkpoint, checkpoint_sha = load_promoted_policy(
        checkpoint_path, device=device,
    )
    payloads = {
        "id": _read_json(id_eval_path), "ood": _read_json(ood_eval_path),
    }
    banks = {}
    selections = {}
    replay_rows = {}
    for distribution, contract in SCENES.items():
        payload = payloads[distribution]
        banks[distribution] = validate_evaluation(
            payload, scene_profile=contract["profile"], ep0=contract["ep0"],
            checkpoint_sha256=checkpoint_sha, policy_d=policy.d,
        )
        selections[distribution] = select_cases(payload)
        full_batch = RAW.run_batched_raw(
            policy, scene_profile=contract["profile"], ep0=contract["ep0"],
            M=M_PER_GAMMA, noise=banks[distribution], device=device,
            retain_proposals=True,
        )
        expected_by_key = {
            (round(float(row["gamma"]), 8), int(row["episode"])): row
            for row in payload["rows"]
        }
        replay_rows[distribution] = {}
        for row in full_batch:
            key = (round(float(row["gamma"]), 8), int(row["episode"]))
            if key not in expected_by_key or key in replay_rows[distribution]:
                raise RuntimeError("full-batch replay cell identity differs from M50")
            expected = expected_by_key[key]
            if str(row["status"]) != str(expected["status"]):
                raise RuntimeError("full-batch replay changed an M50 outcome")
            if int(row["steps"]) != int(expected["steps"]):
                raise RuntimeError("full-batch replay changed an M50 step count")
            for name in (
                "min_clearance", "successful_clearance", "time_to_goal",
            ):
                _same_metric(row.get(name), expected.get(name), name=name)
            replay_rows[distribution][key] = row
        if set(replay_rows[distribution]) != set(expected_by_key):
            raise RuntimeError("full-batch replay does not cover the complete M50 bank")
    if _noise_seed(payloads["id"]) != _noise_seed(payloads["ood"]):
        raise ValueError("ID and OOD M50 evaluations do not share one CRN seed")
    if array_sha256(banks["id"]) != array_sha256(banks["ood"]):
        raise ValueError("ID and OOD M50 evaluations do not share one CRN bank")

    cases = {}
    selection_report = {}
    for distribution, role in (
        ("id", "success"), ("id", "failure"),
        ("ood", "success"), ("ood", "failure"),
    ):
        contract = SCENES[distribution]
        row = selections[distribution][role]
        rollout_index = int(row["episode"]) - int(contract["ep0"])
        key = f"{distribution}_{role}"
        replay_key = (round(DISPLAY_GAMMA, 8), int(row["episode"]))
        cases[key] = trace_case(
            replay_rows[distribution][replay_key],
            scene_profile=contract["profile"], rollout_index=rollout_index,
            expected_row=row,
        )
        selection_report[key] = dict(
            episode=int(row["episode"]), rollout_index=rollout_index,
            outcome=row["status"],
            selection=selections[distribution][f"{role}_selection"],
        )
    return dict(
        status=TRACE_STATUS, checkpoint=os.path.abspath(checkpoint_path),
        checkpoint_sha256=checkpoint_sha,
        checkpoint_promotion=dict(
            scientific_status=checkpoint["scientific_status"],
            selected_gate=checkpoint["selected_gate"],
        ),
        gamma=DISPLAY_GAMMA, cases=cases, selection=selection_report,
        evaluation=dict(
            id=os.path.abspath(id_eval_path), ood=os.path.abspath(ood_eval_path),
            id_sha256=sha256_file(id_eval_path),
            ood_sha256=sha256_file(ood_eval_path),
            M_per_gamma=M_PER_GAMMA,
            noise_seed=_noise_seed(payloads["id"]),
            noise_bank_sha256=array_sha256(banks["id"]),
        ),
        semantics=dict(
            branch_label=(
                "current raw H10 proposal, independent exact full-H moving-pedestrian "
                "certificate under clipped dynamics"
            ),
            trajectory_validity=(
                "mean terminal-truncated executed-window label along the full trajectory; "
                "not the current proposal label"
            ),
            green_geometry=(
                "candidate-specific analytic verifier faces for every sensed or "
                "current-sensed pedestrian plus exactly 16 artificial outer faces"
            ),
        ),
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--id-eval", required=True)
    parser.add_argument("--ood-eval", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--frame-stride", type=int, default=2)
    args = parser.parse_args(argv)
    outdir = os.path.abspath(args.outdir)
    if os.path.exists(outdir):
        raise FileExistsError(f"restart-only branch renderer refuses existing outdir: {outdir}")
    os.makedirs(outdir, exist_ok=False)
    bundle = build_bundle(
        args.checkpoint, args.id_eval, args.ood_eval, device=args.device,
    )
    trace_path = os.path.join(outdir, "hp100_id_ood_branch_trace.pt")
    torch.save(bundle, trace_path)
    render = render_bundle(
        bundle,
        output_mp4=os.path.join(outdir, "hp100_id_ood_branch.mp4"),
        output_png=os.path.join(outdir, "hp100_id_ood_branch_final.png"),
        output_dir=outdir, fps=args.fps, frame_stride=args.frame_stride,
    )
    artifact_paths = dict(
        trace=os.path.abspath(trace_path), mp4=render["mp4"],
        final_png=render["png"], **{
            f"{key}_png": path for key, path in render["snapshots"].items()
        },
    )
    artifacts = {
        name: dict(
            path=path, bytes=os.path.getsize(path), sha256=sha256_file(path),
        )
        for name, path in artifact_paths.items()
    }
    report = dict(
        status=STATUS, trace=os.path.abspath(trace_path),
        trace_sha256=sha256_file(trace_path),
        checkpoint=bundle["checkpoint"],
        checkpoint_sha256=bundle["checkpoint_sha256"],
        selection=bundle["selection"], evaluation=bundle["evaluation"],
        semantics=bundle["semantics"], render=render, artifacts=artifacts,
    )
    report_path = os.path.join(outdir, "BRANCH_VIZ_COMPLETE.json")
    with open(report_path + ".tmp", "w") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    os.replace(report_path + ".tmp", report_path)
    print(json.dumps({
        "status": STATUS, "report": report_path,
        "selection": report["selection"],
    }), flush=True)


if __name__ == "__main__":
    main()
