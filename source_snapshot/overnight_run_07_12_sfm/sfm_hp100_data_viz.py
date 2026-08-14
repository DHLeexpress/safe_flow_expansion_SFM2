"""Render an audited SafeMPPI Hp100 demonstration after data collection.

The renderer is deliberately data-only: it loads one successful trajectory
from a completed ``stage2_hp100_data`` directory and never reruns the expert.
For every stored context it reconstructs the declared nominal polytope
from the stored robot/pedestrian state, then requires the resulting float32
``[32,100]`` raster to match the stored training feature bit for bit.

The left panel distinguishes the *16 artificial outer faces* used by the
nominal SafeMPPI geometry from the right panel's *32 angular observation
rays*.  Those are independent discretizations and must not be conflated.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon
import numpy as np
import torch

import _paths  # noqa: F401
import sfm_b1_viz as BV
import sfm_hp100_dynamics as DYN
import sfm_hp100_features as HPF
import sfm_hp100_history as HPH
import sfm_scene as SS
import stage2_hp100_data as DATA


STATUS = "SFM_HP100_DATA_PROVENANCE_VIZ_COMPLETE"
COUNTERFACTUAL_STATUS = "SFM_HP100_COUNTERFACTUAL_NO_RETREAT_VIZ_COMPLETE"
EXPECTED_DATA_STATUS = "HP100_ID_DATASET_COMPLETE"
EXPECTED_SCHEMA = DATA.SCHEMA_VERSION
BASE_FACES = 16
ANGULAR_RAYS = 32
RADIAL_BINS = 100
BLUE = "#0072B2"
ORANGE = "#E69F00"
GRAY = "#777777"


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:  # pragma: no cover - only for older supported torch builds
        return torch.load(path, map_location="cpu", weights_only=False)


def _assert_manifest_contract(manifest: dict) -> None:
    if manifest.get("dynamics") != DYN.contract():
        raise RuntimeError("dataset dynamics contract differs from the renderer")
    if manifest.get("target_contract") != DATA.target_contract():
        raise RuntimeError("dataset weighted-plan target contract differs")
    feature = manifest.get("feature", {})
    if feature.get("contract") != HPF.contract():
        raise RuntimeError("dataset Hp100 feature contract differs from the renderer")
    recorded_hashes = manifest.get("source_hashes")
    current_hashes = DATA._source_hashes()
    if not isinstance(recorded_hashes, dict):
        raise RuntimeError("dataset manifest is missing authenticated source hashes")
    for name, current in current_hashes.items():
        recorded = recorded_hashes.get(name)
        if not isinstance(recorded, dict) or recorded.get("sha256") != current["sha256"]:
            raise RuntimeError(f"dataset source hash differs at {name}")
    environment = manifest.get("environment", {})
    required = (
        "n_ped", "pedestrian_radius", "sensing_radius", "goal", "task_bounds",
    )
    missing = [name for name in required if name not in environment]
    if missing:
        raise RuntimeError(f"dataset manifest is missing scene constants: {missing}")


def _file_row(manifest: dict, gamma: float) -> dict:
    matches = [
        row for row in manifest.get("files", [])
        if math.isclose(float(row["gamma"]), float(gamma), rel_tol=0.0, abs_tol=1.0e-8)
    ]
    if len(matches) != 1:
        raise ValueError(f"gamma {gamma:g} appears {len(matches)} times in the manifest")
    return matches[0]


def _episode_arrays(payload: dict, episode: int) -> dict[str, np.ndarray]:
    required = (
        "hp", "low5", "hist", "U", "state", "ped_xy", "ped_vel",
        "executed_action", "episode", "step", "target_eligible",
        "target_reason_code", "plan_candidate_count", "plan_accepted_count",
        "plan_rejected_count", "plan_weighted_h", "plan_first_violation",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"dataset payload is missing fields: {missing}")
    episode_tensor = torch.as_tensor(payload["episode"], dtype=torch.int64)
    indices = torch.nonzero(episode_tensor == int(episode), as_tuple=False).reshape(-1)
    if not len(indices):
        raise ValueError(f"episode {episode} is not a stored successful trajectory")
    steps = torch.as_tensor(payload["step"], dtype=torch.int64).index_select(0, indices)
    order = torch.argsort(steps)
    indices = indices.index_select(0, order)
    output = {}
    for key in required:
        value = torch.as_tensor(payload[key]).index_select(0, indices)
        output[key] = value.detach().cpu().numpy()
    expected_steps = np.arange(len(output["step"]), dtype=np.int64)
    if not np.array_equal(output["step"], expected_steps):
        raise ValueError(
            f"episode {episode} must contain contiguous steps 0..T-1; "
            f"got {output['step'].tolist()}"
        )
    return output


def load_episode(dataset_dir, gamma: float, episode: int) -> tuple[dict, dict, Path]:
    """Load one authenticated successful trajectory from a stage-2 dataset."""
    root = Path(dataset_dir).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing completed dataset manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != EXPECTED_DATA_STATUS:
        raise ValueError("input directory is not a completed Hp100 demonstration dataset")
    if manifest.get("schema_version") != EXPECTED_SCHEMA:
        raise ValueError(f"unexpected dataset schema: {manifest.get('schema_version')}")
    _assert_manifest_contract(manifest)
    feature = manifest.get("feature", {})
    if feature.get("shape") != [ANGULAR_RAYS, RADIAL_BINS]:
        raise ValueError("manifest does not declare the exact [32,100] Hp raster")
    if int(feature.get("nominal_polytope_n_base", -1)) != BASE_FACES:
        raise ValueError("manifest does not declare exactly 16 nominal outer faces")
    expected_velocity_aware = bool(HPF.PREDICT_GAIN > 0.0)
    if feature.get("velocity_aware") is not expected_velocity_aware:
        raise ValueError("manifest Hp feature has the wrong predictive-retreat contract")
    row = _file_row(manifest, float(gamma))
    data_path = root / row["file"]
    if not data_path.is_file():
        raise FileNotFoundError(f"missing manifest-declared data file: {data_path}")
    if int(row["bytes"]) != data_path.stat().st_size:
        raise RuntimeError("manifest-declared dataset byte count does not match")
    if row["sha256"] != _sha256(data_path):
        raise RuntimeError("manifest-declared dataset SHA-256 does not match")
    payload = _torch_load(data_path)
    if payload.get("schema_version") != EXPECTED_SCHEMA or not payload.get("success_only"):
        raise ValueError("dataset file is not a successful-only Hp100 payload")
    if not math.isclose(float(payload["gamma"]), float(gamma), rel_tol=0.0, abs_tol=1.0e-8):
        raise ValueError("dataset payload gamma does not match the request")
    return manifest, _episode_arrays(payload, int(episode)), data_path


def _base_geometry(center: np.ndarray, sensing: float) -> tuple[np.ndarray, np.ndarray]:
    theta = np.arange(BASE_FACES, dtype=np.float64) * 2.0 * np.pi / BASE_FACES
    A = np.stack((np.cos(theta), np.sin(theta)), axis=1)
    b = A @ np.asarray(center, np.float64) + float(sensing) * np.cos(np.pi / BASE_FACES)
    return A, b


def _assert_base_faces(geometry: dict, center: np.ndarray, sensing: float) -> None:
    expected_A, expected_b = _base_geometry(center, sensing)
    A = np.asarray(geometry["A"], np.float64)
    b = np.asarray(geometry["b"], np.float64)
    if len(A) < BASE_FACES:
        raise RuntimeError(f"nominal geometry has only {len(A)} faces")
    if not np.allclose(A[:BASE_FACES], expected_A, rtol=0.0, atol=1.0e-6):
        raise RuntimeError("nominal geometry's first 16 rows are not the locked outer faces")
    if not np.allclose(b[:BASE_FACES], expected_b, rtol=0.0, atol=1.0e-6):
        raise RuntimeError("nominal geometry's first 16 offsets are not the locked outer faces")


def validate_episode(manifest: dict, rows: dict[str, np.ndarray]) -> dict:
    """Recompute every feature and fail closed on any provenance mismatch."""
    count = len(rows["step"])
    expected_shapes = {
        "hp": (count, ANGULAR_RAYS, RADIAL_BINS),
        "U": (count, 10, 2),
        "state": (count, 4),
        "ped_xy": (count, int(manifest["environment"]["n_ped"]), 2),
        "ped_vel": (count, int(manifest["environment"]["n_ped"]), 2),
        "executed_action": (count, 2),
    }
    for key, expected in expected_shapes.items():
        if rows[key].shape != expected:
            raise ValueError(f"stored {key} has shape {rows[key].shape}, expected {expected}")

    feature = manifest["feature"]
    environment = manifest["environment"]
    sensing = float(environment["sensing_radius"])
    pedestrian_radius = float(environment["pedestrian_radius"])
    predict_gain = float(feature["predict_gain"])
    predict_tau = float(feature["predict_tau"])
    geometries = []
    max_abs_error = 0.0
    bitwise_matches = 0
    for index in range(count):
        ped_xy = np.asarray(rows["ped_xy"][index], np.float32)
        obstacles = np.concatenate(
            (ped_xy, np.full((len(ped_xy), 1), pedestrian_radius, np.float32)),
            axis=1,
        )
        reconstructed, geometry = HPF.hp100_frame(
            rows["state"][index, :2], obstacles,
            sensing=sensing, n_base=BASE_FACES,
            obstacle_velocities=rows["ped_vel"][index],
            robot_velocity=rows["state"][index, 2:4],
            predict_gain=predict_gain, predict_tau=predict_tau,
            return_geometry=True,
        )
        stored = np.asarray(rows["hp"][index], np.float32)
        error = float(np.max(np.abs(reconstructed - stored)))
        max_abs_error = max(max_abs_error, error)
        if not np.array_equal(reconstructed, stored):
            raise RuntimeError(
                f"stored Hp100 provenance mismatch at step {index}: "
                f"not bitwise equal, max_abs_error={error:.9g}"
            )
        bitwise_matches += 1
        _assert_base_faces(geometry, rows["state"][index, :2], sensing)
        geometries.append(geometry)

    target_eligible = np.asarray(rows["target_eligible"], bool)
    for index, geometry in enumerate(geometries):
        planner_polytope = tuple(
            np.asarray(geometry[key], np.float32)
            for key in ("A", "b", "ref", "margins")
        )
        plan_audit = DATA.audit_weighted_plan(
            rows["state"][index], rows["U"][index], planner_polytope,
            float(rows["low5"][index, -1]),
        )
        stored_first = int(rows["plan_first_violation"][index])
        actual_first = (
            -1 if plan_audit["first_violation_step"] is None
            else int(plan_audit["first_violation_step"])
        )
        if actual_first != stored_first:
            raise RuntimeError(
                f"stored weighted-plan audit differs at step {index}: "
                f"{stored_first} != {actual_first}"
            )
        stored_h = np.asarray(rows["plan_weighted_h"][index], np.float32)
        if not np.array_equal(stored_h, np.asarray(plan_audit["h"], np.float32)):
            delta = float(np.max(np.abs(stored_h - plan_audit["h"])))
            raise RuntimeError(
                f"stored weighted-plan H differs at step {index}: max_delta={delta}"
            )
        logical = bool(
            int(rows["plan_accepted_count"][index]) > 0 and actual_first == -1
        )
        if logical != bool(target_eligible[index]):
            raise RuntimeError(f"stored target eligibility differs at step {index}")
    rollout_max_abs_error = 0.0
    rollout_bitwise = True
    if count > 1:
        reconstructed_next = DYN.step_numpy(
            rows["state"][:-1], rows["executed_action"][:-1]
        ).astype(np.float32, copy=False)
        expected_next = np.asarray(rows["state"][1:], np.float32)
        rollout_max_abs_error = float(np.max(np.abs(reconstructed_next - expected_next)))
        rollout_bitwise = bool(np.array_equal(reconstructed_next, expected_next))
        if not rollout_bitwise:
            raise RuntimeError(
                "stored expert state sequence does not reproduce under the shared dynamics: "
                f"max_abs_error={rollout_max_abs_error:.9g}"
            )
    histories = HPH.build_hp100(
        torch.from_numpy(np.asarray(rows["hp"], np.float32)),
        torch.from_numpy(np.asarray(rows["episode"], np.int64)),
        torch.from_numpy(np.asarray(rows["step"], np.int64)),
    ).numpy()
    return dict(
        geometries=geometries,
        histories=histories,
        hp_frames=np.asarray(rows["hp"], np.float32),
        geometry_label=(
            "velocity-aware nominal polytope"
            if float(feature["predict_gain"]) > 0.0
            else "current-position tangent nominal polytope"
        ),
        raster_label="Exact stored $H_P$ raster",
        counterfactual=None,
        sensing_radius=float(sensing),
        pedestrian_radius=pedestrian_radius,
        goal=np.asarray(environment["goal"], np.float64),
        task_bounds=tuple(map(float, environment["task_bounds"])),
        audit=dict(
            contexts=count,
            hp_bitwise_matches=bitwise_matches,
            all_hp_bitwise_equal=bool(bitwise_matches == count),
            hp_max_abs_error=float(max_abs_error),
            rollout_bitwise_equal=bool(rollout_bitwise),
            rollout_max_abs_error=float(rollout_max_abs_error),
            artificial_outer_faces=BASE_FACES,
            observation_angular_rays=ANGULAR_RAYS,
            radial_bins=RADIAL_BINS,
            radial_cell_width=float(sensing / RADIAL_BINS),
            eligible_weighted_plan_contexts=int(target_eligible.sum()),
            excluded_weighted_plan_contexts=int((~target_eligible).sum()),
            detected_obstacle_faces_min=int(min(len(row["A"]) - BASE_FACES for row in geometries)),
            detected_obstacle_faces_max=int(max(len(row["A"]) - BASE_FACES for row in geometries)),
        ),
    )


def counterfactual_no_retreat(manifest: dict, rows: dict[str, np.ndarray]) -> dict:
    """Re-render stored states with current-position tangent faces.

    The original feature and rollout are authenticated first.  This function
    changes neither the stored controls nor the expert trajectory, so its
    output is a geometry counterfactual rather than replacement training data.
    """
    feature = manifest["feature"]
    if float(feature["predict_gain"]) == 0.0:
        raise ValueError("source dataset already uses current-position tangent geometry")
    validated = validate_episode(manifest, rows)
    environment = manifest["environment"]
    sensing = float(environment["sensing_radius"])
    pedestrian_radius = float(environment["pedestrian_radius"])
    predict_tau = float(feature["predict_tau"])
    frames, geometries = [], []
    for index in range(len(rows["step"])):
        ped_xy = np.asarray(rows["ped_xy"][index], np.float32)
        obstacles = np.concatenate(
            (ped_xy, np.full((len(ped_xy), 1), pedestrian_radius, np.float32)),
            axis=1,
        )
        frame, geometry = HPF.hp100_frame(
            rows["state"][index, :2], obstacles,
            sensing=sensing, n_base=BASE_FACES,
            obstacle_velocities=rows["ped_vel"][index],
            robot_velocity=rows["state"][index, 2:4],
            predict_gain=0.0, predict_tau=predict_tau,
            return_geometry=True,
        )
        raw_margins = (
            np.asarray(geometry["b"], np.float64)
            - np.asarray(geometry["A"], np.float64)
            @ np.asarray(geometry["ref"], np.float64)
        )
        if np.any(raw_margins <= 0.0):
            raise RuntimeError("current-tangent counterfactual does not contain its robot reference")
        _assert_base_faces(geometry, rows["state"][index, :2], sensing)
        frames.append(np.asarray(frame, np.float32))
        geometries.append(geometry)
    frames = np.stack(frames)
    histories = HPH.build_hp100(
        torch.from_numpy(frames),
        torch.from_numpy(np.asarray(rows["episode"], np.int64)),
        torch.from_numpy(np.asarray(rows["step"], np.int64)),
    ).numpy()
    validated.update(
        geometries=geometries,
        histories=histories,
        hp_frames=frames,
        geometry_label="current-position tangent nominal polytope",
        raster_label="Counterfactual current-tangent $H_P$ raster",
        counterfactual=dict(
            enabled=True,
            predict_gain=0.0,
            source_predict_gain=float(feature["predict_gain"]),
            stored_trajectory_reused=True,
            expert_rerun=False,
            training_data=False,
        ),
    )
    return validated


def _level_polygons(geometry: dict, center: np.ndarray, gamma: float):
    return BV._level_polygons(
        geometry["A"], geometry["margins"], center, float(gamma), H=10
    )


def _target_segment(state: np.ndarray, controls: np.ndarray) -> np.ndarray:
    states = [np.asarray(state, np.float32)]
    for action in np.asarray(controls, np.float32):
        states.append(DYN.step_numpy(states[-1], action).astype(np.float32, copy=False))
    return np.asarray(states, np.float32)


def _geometry_condition(geometry: dict) -> dict:
    A = np.asarray(geometry["A"], np.float64)
    b = np.asarray(geometry["b"], np.float64)
    ref = np.asarray(geometry["ref"], np.float64)
    margins = np.asarray(geometry["margins"], np.float64)
    scale = np.abs(b) + np.sum(np.abs(A * ref[None]), axis=1)
    face = int(np.argmin(margins))
    condition = scale[face] / max(abs(margins[face]), np.finfo(float).tiny)
    return dict(
        minimum_margin_face=face,
        minimum_margin=float(margins[face]),
        cancellation_scale=float(scale[face]),
        cancellation_condition=float(condition),
    )


def _full_rollout(rows: dict[str, np.ndarray]) -> np.ndarray:
    terminal = DYN.step_numpy(rows["state"][-1], rows["executed_action"][-1])
    return np.concatenate((rows["state"], terminal[None]), axis=0)


def _draw_world(axis, rows: dict, validated: dict, index: int, gamma: float) -> None:
    state = np.asarray(rows["state"][index], float)
    ped_xy = np.asarray(rows["ped_xy"][index], float)
    ped_vel = np.asarray(rows["ped_vel"][index], float)
    geometry = validated["geometries"][index]
    full = _full_rollout(rows)
    axis.plot(full[:, 0], full[:, 1], color="#B7B7B7", lw=1.0, alpha=.75, zorder=1)
    axis.plot(full[:index + 1, 0], full[:index + 1, 1], color="black", lw=1.8, zorder=5)
    target = _target_segment(state, rows["U"][index])
    target_color = BLUE if bool(rows["target_eligible"][index]) else "#D62728"
    axis.plot(
        target[:, 0], target[:, 1], color=target_color,
        lw=1.6, marker=".", ms=2.6, zorder=7,
    )

    sensing = float(validated["sensing_radius"])
    ray_theta = -np.pi + (np.arange(ANGULAR_RAYS) + .5) * (
        2.0 * np.pi / ANGULAR_RAYS
    )
    ray_ends = state[:2][None] + sensing * np.stack(
        (np.cos(ray_theta), np.sin(ray_theta)), axis=1
    )
    for endpoint in ray_ends:
        axis.plot(
            [state[0], endpoint[0]], [state[1], endpoint[1]],
            color="#56B4E9", lw=.28, alpha=.14, zorder=1.8,
        )
    base_A, base_b = _base_geometry(state[:2], sensing)
    base_polygon = BV.halfspace_polygon(base_A, base_b)
    nominal_polygon = BV.halfspace_polygon(geometry["A"], geometry["b"])
    if base_polygon is None or nominal_polygon is None:
        raise RuntimeError("locked K16 geometry did not form a bounded polygon")
    axis.add_patch(Polygon(
        base_polygon, closed=True, fill=False, edgecolor=GRAY, linewidth=1.0,
        linestyle="--", alpha=.9, zorder=2,
    ))
    for horizon, polygon in _level_polygons(geometry, state[:2], gamma):
        axis.add_patch(Polygon(
            polygon, closed=True, fill=False, edgecolor=BLUE, linewidth=.46,
            alpha=.18 + .045 * horizon, zorder=2.5,
        ))
    axis.add_patch(Polygon(
        nominal_polygon, closed=True, fill=False, edgecolor=BLUE,
        linewidth=1.6, alpha=.98, zorder=3,
    ))
    for position, velocity in zip(ped_xy, ped_vel):
        axis.add_patch(Circle(
            position, validated["pedestrian_radius"], facecolor="#777777", edgecolor="#333333",
            linewidth=.5, alpha=.72, zorder=6,
        ))
        predicted = position[None] + np.arange(11)[:, None] * DYN.DT * velocity[None]
        axis.plot(predicted[:, 0], predicted[:, 1], ".--", color=GRAY,
                  lw=.45, ms=1.6, alpha=.55, zorder=4)
    goal = validated["goal"]
    axis.plot(goal[0], goal[1], marker="*", color="#F0C419", ms=13,
              mec="black", mew=.45, zorder=10)
    axis.plot(state[0], state[1], "o", color=BLUE, ms=7, zorder=10)
    # Keep the complete two-metre outer support visible even at the start and
    # goal corners; clipping it to the nominal task box would obscure the
    # exact K=16 provenance this figure is meant to audit.
    task_lo, task_hi = validated["task_bounds"]
    axis.set_xlim(
        min(task_lo, float(full[:, 0].min() - sensing - .1)),
        max(task_hi, float(full[:, 0].max() + sensing + .1)),
    )
    axis.set_ylim(
        min(task_lo, float(full[:, 1].min() - sensing - .1)),
        max(task_hi, float(full[:, 1].max() + sensing + .1)),
    )
    axis.set_aspect("equal")
    axis.grid(alpha=.13)
    axis.set_xlabel("world x [m]")
    axis.set_ylabel("world y [m]")
    detected = len(geometry["A"]) - BASE_FACES
    axis.set_title(
        f"Stored SafeMPPI rollout · step {int(rows['step'][index])}\n"
        f"nominal geometry = 16 outer faces + {detected} pedestrian faces"
    )
    axis.legend(handles=[
        Line2D([], [], color="black", lw=1.8, label="executed rollout to current state"),
        Line2D([], [], color=BLUE, lw=1.6, label="eligible current weighted H=10 target"),
        Line2D([], [], color="#D62728", lw=1.6, label="excluded weighted/fallback plan"),
        Line2D([], [], color=BLUE, lw=1.6, label=validated["geometry_label"]),
        Line2D([], [], color=BLUE, lw=.6, label="nominal level sets h=1..10"),
        Line2D([], [], color=GRAY, lw=1.0, ls="--", label="K=16 artificial outer support"),
        Line2D([], [], color="#56B4E9", lw=.7, label="32 observation rays (not faces)"),
    ], loc="lower right", fontsize=6.6, framealpha=.92)


def _draw_rasters(current_axis, history_axis, rows: dict, validated: dict, index: int) -> None:
    current = np.asarray(validated["hp_frames"][index], float)
    current_axis.imshow(
        current, origin="lower", aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0,
        extent=(.0, validated["sensing_radius"], -np.pi, np.pi), interpolation="nearest",
    )
    cell_width = float(validated["sensing_radius"]) / RADIAL_BINS
    current_axis.set_xlabel(
        f"radius [m] · 100 bins ({cell_width:.2f} m)"
    )
    current_axis.set_ylabel(r"observation angle $\theta$ [rad]")
    current_axis.set_title(f"{validated['raster_label']} · 32 observation rays")
    current_axis.set_yticks((-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi),
                           (r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"))

    history = np.asarray(validated["histories"][index], float)
    diagnostic = history.min(axis=1)
    history_axis.imshow(
        diagnostic, origin="upper", aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0,
        extent=(.0, validated["sensing_radius"], 9.5, -.5), interpolation="nearest",
    )
    history_axis.set_xlabel("radius [m]")
    history_axis.set_ylabel("history lag")
    history_axis.set_yticks((0, 3, 6, 9))
    history_axis.set_title(
        "Hp10 diagnostic: angular minimum per radial bin\n"
        "(model input remains the full 10×32×100 tensor)"
    )


def _new_figure():
    figure = plt.figure(figsize=(13.4, 7.5))
    grid = figure.add_gridspec(
        2, 2, width_ratios=(1.15, 1.0), height_ratios=(1.0, .62),
        left=.06, right=.91, bottom=.09, top=.80, wspace=.25, hspace=.72,
    )
    world = figure.add_subplot(grid[:, 0])
    current = figure.add_subplot(grid[0, 1])
    history = figure.add_subplot(grid[1, 1])
    color_axis = figure.add_axes((.925, .15, .014, .68))
    figure.colorbar(
        ScalarMappable(norm=Normalize(-1.0, 1.0), cmap="coolwarm"),
        cax=color_axis, label="clipped nominal $H_P$",
    )
    return figure, (world, current, history)


def _draw_frame(figure, axes, rows, validated, index: int, gamma: float, episode: int):
    for axis in axes:
        axis.clear()
    _draw_world(axes[0], rows, validated, int(index), float(gamma))
    _draw_rasters(axes[1], axes[2], rows, validated, int(index))
    condition = _geometry_condition(validated["geometries"][int(index)])
    role = (
        "Hp100 current-tangent counterfactual"
        if validated["counterfactual"] else "Hp100 provenance audit"
    )
    figure.suptitle(
        f"{role} | successful episode {episode} | gamma={gamma:g}\n"
        "K=16 nominal outer faces are independent of 32 observation rays | "
        f"min face margin={condition['minimum_margin']:.4g} m, "
        f"cancellation κ={condition['cancellation_condition']:.1f}",
        fontsize=12, y=.975,
    )
    return axes


def render(dataset_dir, gamma: float, episode: int, output_dir, *,
           selected_step: int | None = None, frame_stride: int = 1,
           fps: int = 8, dpi: int = 115,
           current_tangent_counterfactual: bool = False) -> dict:
    """Render MP4, PNG/PDF, geometry sidecar, and an authenticated contract."""
    if int(frame_stride) <= 0 or int(fps) <= 0 or int(dpi) <= 0:
        raise ValueError("frame_stride, fps, and dpi must be positive")
    manifest, rows, data_path = load_episode(dataset_dir, float(gamma), int(episode))
    validated = (
        counterfactual_no_retreat(manifest, rows)
        if current_tangent_counterfactual else validate_episode(manifest, rows)
    )
    available_steps = rows["step"].astype(int).tolist()
    if selected_step is None:
        selected_step = available_steps[len(available_steps) // 2]
    if int(selected_step) not in available_steps:
        raise ValueError(f"selected step {selected_step} is not in {available_steps}")
    selected_index = available_steps.index(int(selected_step))
    indices = list(range(0, len(available_steps), int(frame_stride)))
    if indices[-1] != len(available_steps) - 1:
        indices.append(len(available_steps) - 1)
    if selected_index not in indices:
        indices.append(selected_index)
        indices.sort()

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    gamma_tag = f"{float(gamma):g}".replace(".", "p")
    suffix = "_current_tangent_counterfactual" if current_tangent_counterfactual else ""
    stem = f"hp100_expert_g{gamma_tag}_ep{int(episode)}{suffix}"
    mp4 = output / f"{stem}.mp4"
    png = output / f"{stem}_step{int(selected_step):03d}.png"
    pdf = output / f"{stem}_step{int(selected_step):03d}.pdf"
    geometry_npz = output / f"{stem}_step{int(selected_step):03d}_geometry.npz"
    contract_path = output / f"{stem}.json"
    for path in (mp4, png, pdf, geometry_npz, contract_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite output: {path}")

    figure, axes = _new_figure()
    movie = animation.FuncAnimation(
        figure,
        lambda frame: _draw_frame(
            figure, axes, rows, validated, frame, float(gamma), int(episode)
        ),
        frames=indices, interval=1000.0 / int(fps), blit=False,
    )
    movie.save(
        mp4, writer=animation.FFMpegWriter(fps=int(fps), bitrate=4200), dpi=int(dpi)
    )
    plt.close(figure)

    figure, axes = _new_figure()
    _draw_frame(
        figure, axes, rows, validated, selected_index, float(gamma), int(episode)
    )
    figure.savefig(png, dpi=int(dpi) + 35)
    figure.savefig(pdf)
    plt.close(figure)

    selected_geometry = validated["geometries"][selected_index]
    selected_condition = _geometry_condition(selected_geometry)
    np.savez_compressed(
        geometry_npz,
        A=np.asarray(selected_geometry["A"], np.float32),
        b=np.asarray(selected_geometry["b"], np.float32),
        ref=np.asarray(selected_geometry["ref"], np.float32),
        margins=np.asarray(selected_geometry["margins"], np.float32),
        state=np.asarray(rows["state"][selected_index], np.float32),
        pedestrian_positions=np.asarray(rows["ped_xy"][selected_index], np.float32),
        pedestrian_velocities=np.asarray(rows["ped_vel"][selected_index], np.float32),
        stored_hp100=np.asarray(rows["hp"][selected_index], np.float32),
        rendered_hp100=np.asarray(validated["hp_frames"][selected_index], np.float32),
        weighted_plan_H10=np.asarray(rows["U"][selected_index], np.float32),
        full_executed_rollout=np.asarray(_full_rollout(rows), np.float32),
    )

    file_row = _file_row(manifest, float(gamma))
    source = Path(inspect.getsourcefile(HPF)).resolve()
    contract = dict(
        status=(COUNTERFACTUAL_STATUS if current_tangent_counterfactual else STATUS),
        role=(
            "counterfactual current-tangent geometry on an authenticated stored "
            "trajectory; no expert rerun and not replacement training data"
            if current_tangent_counterfactual else
            "post-collection provenance visualization; no expert rerun, no data "
            "selection, and no approximate geometry"
        ),
        dataset=dict(
            root=str(Path(dataset_dir).resolve()),
            manifest=str(Path(dataset_dir).resolve() / "manifest.json"),
            manifest_sha256=_sha256(Path(dataset_dir).resolve() / "manifest.json"),
            file=str(data_path), file_sha256=file_row["sha256"],
            gamma=float(gamma), successful_episode=int(episode),
            stored_contexts=len(available_steps),
        ),
        selected_frame=dict(
            selected_step=int(selected_step), stored_row_index=int(selected_index),
            rendered_frame_index=int(indices.index(selected_index)),
            geometry_sidecar=str(geometry_npz),
            geometry_condition=selected_condition,
        ),
        provenance_distinction=dict(
            stored=(
                "Hp100 raster, robot state, pedestrian positions/velocities, "
                "executed action, current weighted H10 plan, acceptance counts, "
                "and target eligibility"
            ),
            recomputed=(
                "canonical float32 A,b,ref,margins from the stored state/pedestrians "
                "under the manifest-pinned feature source"
            ),
            equality_check="recomputed Hp100 is bitwise equal to the stored raster",
        ),
        geometry_observation_separation=dict(
            nominal_artificial_outer_faces=BASE_FACES,
            observation_angular_rays=ANGULAR_RAYS,
            equal=False,
            statement="K=16 nominal support geometry is independent of 32 raster rays",
        ),
        provenance_audit=validated["audit"],
        counterfactual=validated["counterfactual"],
        render=dict(
            frame_stride=int(frame_stride), fps=int(fps), rendered_frames=len(indices),
            level_sets=10, pedestrian_radius=float(validated["pedestrian_radius"]),
            history_diagnostic=(
                "min over 32 angles for each of the exact ten stored/reconstructed "
                "history frames; diagnostic only"
            ),
        ),
        outputs={
            "mp4": dict(path=str(mp4), sha256=_sha256(mp4), bytes=mp4.stat().st_size),
            "png": dict(path=str(png), sha256=_sha256(png), bytes=png.stat().st_size),
            "pdf": dict(path=str(pdf), sha256=_sha256(pdf), bytes=pdf.stat().st_size),
            "geometry_npz": dict(
                path=str(geometry_npz), sha256=_sha256(geometry_npz),
                bytes=geometry_npz.stat().st_size,
            ),
        },
        source=dict(path=str(Path(__file__).resolve()), sha256=_sha256(Path(__file__).resolve())),
        feature_source=dict(path=str(source), sha256=_sha256(source)),
    )
    _write_json(contract_path, contract)
    return dict(contract, contract_path=str(contract_path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--gamma", required=True, type=float)
    parser.add_argument("--episode", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selected-step", type=int)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=115)
    parser.add_argument("--current-tangent-counterfactual", action="store_true")
    args = parser.parse_args()
    report = render(
        args.dataset_dir, args.gamma, args.episode, args.output_dir,
        selected_step=args.selected_step, frame_stride=args.frame_stride,
        fps=args.fps, dpi=args.dpi,
        current_tangent_counterfactual=args.current_tangent_counterfactual,
    )
    print(json.dumps({
        "status": report["status"], "contract": report["contract_path"],
        "mp4": report["outputs"]["mp4"]["path"],
        "png": report["outputs"]["png"]["path"],
        "pdf": report["outputs"]["pdf"]["path"],
    }), flush=True)


if __name__ == "__main__":
    main()
