"""Render the actual candidate/rejection/weighting mechanism of the Hp100 expert.

Unlike :mod:`sfm_hp100_data_viz`, this module reruns the locked SafeMPPI
expert and requests its current candidate population.  It never presents the
post-hoc window of future receding-horizon actions as a plan from the current
context.  The current expert generates 2,048 candidates; a deterministic,
acceptance-rate-preserving subset is drawn only to keep the world panel legible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon
import numpy as np
import torch

import _paths  # noqa: F401
import sfm_b1_viz as BV
import sfm_hp100_data_viz as PV
import sfm_hp100_dynamics as DYN
import sfm_scene as SS
import stage2_hp100_data as DATA


STATUS = "SFM_HP100_EXPERT_MECHANISM_VIZ_COMPLETE"
GREEN = "#009E73"
RED = "#D62728"
BLUE = "#0072B2"
ORANGE = "#E69F00"
PURPLE = "#7B2CBF"
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


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def polytope_h(states, polytope: dict) -> np.ndarray:
    """Evaluate the planner's normalized min-face barrier on ``[...,4]`` states."""
    positions = np.asarray(states, np.float32)[..., :2]
    A = np.asarray(polytope["A"], np.float32)
    b = np.asarray(polytope["b"], np.float32)
    margins = np.asarray(polytope["margins"], np.float32)
    values = (b - positions @ A.T) / margins
    return values.min(axis=-1)


def contraction_audit(states, polytope: dict, gamma: float) -> dict:
    """Return exact internal-H rejection locations for one or many plans."""
    h = polytope_h(states, polytope)
    violations = h[..., 1:] < np.float32(1.0 - float(gamma)) * h[..., :-1]
    any_violation = violations.any(axis=-1)
    first = np.where(any_violation, violations.argmax(axis=-1) + 1, -1).astype(np.int16)
    return dict(h=h, violations=violations, feasible=~any_violation, first_violation=first)


def proportional_subset(feasible, cap: int) -> np.ndarray:
    """Choose a deterministic subset while preserving the population ratio."""
    feasible = np.asarray(feasible, bool).reshape(-1)
    cap = min(max(int(cap), 1), len(feasible))
    accepted = np.flatnonzero(feasible)
    rejected = np.flatnonzero(~feasible)
    if cap == len(feasible):
        return np.arange(len(feasible), dtype=np.int64)
    n_accepted = int(round(cap * len(accepted) / len(feasible)))
    if len(accepted) and len(rejected) and cap >= 2:
        n_accepted = min(max(n_accepted, 1), cap - 1)
    else:
        n_accepted = cap if len(accepted) else 0
    n_rejected = cap - n_accepted

    def spread(indices, count):
        if count <= 0:
            return np.empty(0, np.int64)
        if count >= len(indices):
            return indices.astype(np.int64, copy=False)
        locations = np.floor(np.linspace(0, len(indices), count, endpoint=False)).astype(int)
        return indices[locations].astype(np.int64, copy=False)

    return np.sort(np.concatenate((spread(accepted, n_accepted), spread(rejected, n_rejected))))


def _plan_states(state, controls) -> np.ndarray:
    rows = [np.asarray(state, np.float32)]
    for action in np.asarray(controls, np.float32):
        rows.append(DYN.step_numpy(rows[-1], action).astype(np.float32, copy=False))
    return np.asarray(rows, np.float32)


def _polytope_dict(raw) -> dict:
    if raw is None:
        raise RuntimeError("locked expert did not return its nominal polytope")
    return dict(
        A=np.asarray(raw[0], np.float32),
        b=np.asarray(raw[1], np.float32),
        ref=np.asarray(raw[2], np.float32),
        margins=np.asarray(raw[3], np.float32),
    )


def collect_episode(episode: int, gamma: float, *, device="cpu", T=DATA.T) -> dict:
    """Rerun one matched-ID episode with full, read-only planner diagnostics."""
    config = DATA.locked_expert_config()
    config["debug_max_rollouts"] = int(config["num_samples"])
    planner = DATA.CappedSafeMPPIAdapter(**config)
    humans = SS.make_humans(
        int(episode), seed=0, n_ped=DATA.N_PED, speed_range=DATA.PED_SPEED_RANGE,
    )
    torch_device = torch.device(device)
    goal = torch.as_tensor(SS.GOAL, dtype=torch.float32, device=torch_device)
    state = np.zeros(4, np.float32)
    executed_states = [state.copy()]
    frames = []
    collision = reached = False
    minimum_clearance = float("inf")
    with torch.no_grad():
        for step in range(int(T)):
            ped_xy, ped_vel = SS.collect_humans(humans)
            ped_xy = np.asarray(ped_xy, np.float32)
            ped_vel = np.asarray(ped_vel, np.float32)
            clearance = float(
                np.linalg.norm(ped_xy - state[:2][None], axis=1).min() - SS.R_PED
            )
            minimum_clearance = min(minimum_clearance, clearance)
            if clearance < 0.0:
                collision = True
                break
            if float(np.linalg.norm(state[:2] - SS.GOAL)) < DATA.REACH:
                reached = True
                break
            obstacles = DATA._obstacles(ped_xy)
            action, info = planner.plan(
                torch.as_tensor(state, dtype=torch.float32, device=torch_device), goal,
                torch.as_tensor(obstacles, dtype=torch.float32, device=torch_device),
                gamma=float(gamma),
                obstacle_velocities=torch.as_tensor(
                    ped_vel, dtype=torch.float32, device=torch_device,
                ),
                seed=int(episode) * 200 + int(step), return_rollouts=True,
            )
            debug = info["debug_rollouts"]
            sample_indices = np.asarray(debug["sample_indices"], np.int64)
            candidate_states = np.asarray(debug["states"], np.float32)
            candidate_controls = np.asarray(debug["controls"], np.float32)
            candidate_feasible = np.asarray(debug["feasible"], bool)
            candidate_weights = np.asarray(debug["weights"], np.float32)
            first_violation = np.asarray(debug["first_violation_step"], np.int16)
            expected_indices = np.arange(int(config["num_samples"]), dtype=np.int64)
            if not np.array_equal(sample_indices, expected_indices):
                raise RuntimeError("full candidate trace is not in canonical sample order")
            polytope = _polytope_dict(info.get("polytope"))
            if not np.array_equal(first_violation < 0, candidate_feasible):
                raise RuntimeError("planner rejection locations differ from its acceptance mask")
            accepted = int(candidate_feasible.sum())
            if accepted != int(info["num_accepted"]):
                raise RuntimeError("planner acceptance count differs from full trace")
            if accepted and not np.all(candidate_weights[~candidate_feasible] == 0.0):
                raise RuntimeError("rejected candidates received nonzero accepted-set weights")
            if not math.isclose(float(candidate_weights.sum()), 1.0, abs_tol=2.0e-6):
                raise RuntimeError("candidate MPPI weights do not sum to one")
            mean_controls = np.asarray(info["mean_sequence"], np.float32)
            mean_states = _plan_states(state, mean_controls)
            mean_audit = contraction_audit(mean_states[None], polytope, float(gamma))
            action_np = DYN.clip_action_numpy(
                action.detach().cpu().numpy().astype(np.float32).reshape(2)
            ).astype(np.float32, copy=False)
            action_mean_error = float(np.max(np.abs(action_np - mean_controls[0])))
            if action_mean_error > 2.0e-6:
                raise RuntimeError("applied first action differs from the MPPI weighted sequence")
            frames.append(dict(
                step=int(step), state=state.copy(), ped_xy=ped_xy.copy(),
                ped_vel=ped_vel.copy(), polytope=polytope,
                candidate_states=candidate_states, candidate_controls=candidate_controls,
                candidate_feasible=candidate_feasible,
                candidate_weights=candidate_weights,
                first_violation=first_violation,
                mean_controls=mean_controls, mean_states=mean_states,
                mean_h=np.asarray(mean_audit["h"][0], np.float32),
                mean_violations=np.asarray(mean_audit["violations"][0], bool),
                mean_feasible=bool(mean_audit["feasible"][0]),
                action=action_np.copy(), action_mean_max_abs_error=action_mean_error,
                accepted=accepted,
                rejected=int(len(candidate_feasible) - accepted),
                selection_semantics=str(info["selection_semantics"]),
                best_candidate_index=int(info["best_candidate_index"]),
            ))
            state = DYN.step_numpy(state, action_np).astype(np.float32, copy=False)
            executed_states.append(state.copy())
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
            not collision and float(np.linalg.norm(state[:2] - SS.GOAL)) < DATA.REACH
        )
    return dict(
        episode=int(episode), gamma=float(gamma), frames=frames,
        executed_states=np.asarray(executed_states, np.float32),
        success=bool(reached and not collision), collision=bool(collision),
        timeout=bool(not reached and not collision),
        min_clearance=float(minimum_clearance), config=config,
    )


def _add_world_geometry(axis, frame: dict, gamma: float, executed_states) -> None:
    state = frame["state"]
    polytope = frame["polytope"]
    base_A, base_b = PV._base_geometry(state[:2], SS.R_SENSE)
    base_polygon = BV.halfspace_polygon(base_A, base_b)
    nominal_polygon = BV.halfspace_polygon(polytope["A"], polytope["b"])
    if base_polygon is None or nominal_polygon is None:
        raise RuntimeError("nominal geometry is not a bounded polygon")
    axis.add_patch(Polygon(
        base_polygon, closed=True, fill=False, edgecolor=GRAY, linewidth=.75,
        linestyle="--", alpha=.7, zorder=1,
    ))
    for _, polygon in BV._level_polygons(
        polytope["A"], polytope["margins"], state[:2], float(gamma), H=10,
    ):
        axis.add_patch(Polygon(
            polygon, closed=True, fill=False, edgecolor=BLUE, linewidth=.42,
            alpha=.14, zorder=1.5,
        ))
    axis.add_patch(Polygon(
        nominal_polygon, closed=True, fill=False, edgecolor=BLUE,
        linewidth=1.55, alpha=.96, zorder=2,
    ))
    for position, velocity in zip(frame["ped_xy"], frame["ped_vel"]):
        axis.add_patch(Circle(
            position, SS.R_PED, facecolor="#888888", edgecolor="#333333",
            linewidth=.45, alpha=.72, zorder=5,
        ))
        prediction = position[None] + np.arange(11)[:, None] * DYN.DT * velocity[None]
        axis.plot(prediction[:, 0], prediction[:, 1], ".--", color=GRAY,
                  lw=.4, ms=1.4, alpha=.42, zorder=3)
    history = np.asarray(executed_states[:frame["step"] + 1], float)
    axis.plot(history[:, 0], history[:, 1], color="black", lw=1.25, zorder=7)
    axis.plot(state[0], state[1], "o", color="black", ms=5.5, zorder=9)
    axis.plot(SS.GOAL[0], SS.GOAL[1], "*", color="#F0C419", mec="black",
              mew=.4, ms=12, zorder=9)


def _draw_world(axis, frame: dict, gamma: float, executed_states, display_cap: int) -> None:
    _add_world_geometry(axis, frame, gamma, executed_states)
    draw_indices = proportional_subset(frame["candidate_feasible"], int(display_cap))
    states = frame["candidate_states"][draw_indices, :, :2]
    feasible = frame["candidate_feasible"][draw_indices]
    accepted_paths = [path for path, ok in zip(states, feasible) if ok]
    rejected_paths = [path for path, ok in zip(states, feasible) if not ok]
    if rejected_paths:
        axis.add_collection(LineCollection(
            rejected_paths, colors=RED, linewidths=.34, alpha=.055, zorder=2.6,
        ))
        rejected_indices = draw_indices[~feasible]
        violation_steps = frame["first_violation"][rejected_indices]
        violation_points = frame["candidate_states"][
            rejected_indices, violation_steps, :2,
        ]
        axis.scatter(
            violation_points[:, 0], violation_points[:, 1], marker="x", s=7,
            linewidths=.45, c=RED, alpha=.48, zorder=4,
        )
    if accepted_paths:
        axis.add_collection(LineCollection(
            accepted_paths, colors=GREEN, linewidths=.42, alpha=.11, zorder=3,
        ))
    mean_color = PURPLE if frame["accepted"] else ORANGE
    mean_states = frame["mean_states"]
    axis.plot(mean_states[:, 0], mean_states[:, 1], color=mean_color,
              lw=2.25, marker=".", ms=3.0, zorder=8)
    axis.plot(mean_states[:2, 0], mean_states[:2, 1], color="black", lw=3.0, zorder=9)
    lo = min(SS.TASK_LO, float(frame["state"][0] - SS.R_SENSE - .2))
    hi = max(SS.TASK_HI, float(frame["state"][0] + SS.R_SENSE + .2))
    bottom = min(SS.TASK_LO, float(frame["state"][1] - SS.R_SENSE - .2))
    top = max(SS.TASK_HI, float(frame["state"][1] + SS.R_SENSE + .2))
    axis.set_xlim(lo, hi); axis.set_ylim(bottom, top)
    axis.set_aspect("equal"); axis.grid(alpha=.12)
    axis.set_xlabel("world x [m]"); axis.set_ylabel("world y [m]")


def _draw_control_space(axis, frame: dict) -> None:
    controls = frame["candidate_controls"][:, 0]
    feasible = frame["candidate_feasible"]
    weights = frame["candidate_weights"]
    max_weight = max(float(weights.max()), np.finfo(float).tiny)
    marker_sizes = 4.0 + 65.0 * np.sqrt(weights / max_weight)
    axis.scatter(controls[~feasible, 0], controls[~feasible, 1], marker="x",
                 s=marker_sizes[~feasible], linewidths=.35, c=RED,
                 alpha=.24, label="rejected")
    if feasible.any():
        axis.scatter(controls[feasible, 0], controls[feasible, 1], marker="o",
                     s=marker_sizes[feasible], linewidths=0, c=GREEN,
                     alpha=.38, label="accepted")
    mean = np.asarray(frame["action"])
    axis.scatter(mean[0], mean[1], marker="*", s=135, c=ORANGE,
                 edgecolors="black", linewidths=.45, zorder=8, label="weighted output")
    axis.set_xlim(-DYN.U_MAX - .1, DYN.U_MAX + .1)
    axis.set_ylim(-DYN.U_MAX - .1, DYN.U_MAX + .1)
    axis.set_aspect("equal"); axis.grid(alpha=.15)
    axis.set_xlabel(r"first control $u_x$"); axis.set_ylabel(r"first control $u_y$")
    axis.set_title("All candidate first controls · marker size reflects MPPI weight")


def _draw_h_audit(axis, frame: dict, gamma: float) -> None:
    h = np.asarray(frame["mean_h"], float)
    transitions = np.arange(1, len(h))
    threshold = (1.0 - float(gamma)) * h[:-1]
    axis.plot(np.arange(len(h)), h, "o-", color=(PURPLE if frame["accepted"] else ORANGE),
              lw=1.7, ms=3.5, label=r"weighted-plan $H_P(x_h)$")
    axis.plot(transitions, threshold, ".--", color=BLUE, lw=1.0, ms=3,
              label=r"required $(1-\gamma)H_P(x_{h-1})$")
    failures = np.flatnonzero(frame["mean_violations"]) + 1
    if len(failures):
        axis.scatter(failures, h[failures], marker="x", s=35, linewidths=1.2,
                     c=RED, zorder=7, label="weighted-plan violation")
    axis.axhline(0.0, color="#555555", lw=.7, alpha=.6)
    axis.set_xticks(np.arange(0, 11, 2)); axis.set_xlim(0, 10)
    axis.grid(alpha=.15); axis.set_xlabel("plan horizon h"); axis.set_ylabel(r"normalized $H_P$")
    axis.set_title("Weighted output · same-polytope H10 audit")
    axis.legend(loc="best", fontsize=7, framealpha=.88)


def _new_figure():
    figure = plt.figure(figsize=(13.4, 7.5))
    grid = figure.add_gridspec(
        2, 2, width_ratios=(1.35, .9), left=.055, right=.965,
        bottom=.08, top=.82, wspace=.24, hspace=.42,
    )
    return figure, (
        figure.add_subplot(grid[:, 0]),
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[1, 1]),
    )


def _draw_frame(figure, axes, run: dict, frame_index: int, display_cap: int):
    for axis in axes:
        axis.clear()
    frame = run["frames"][int(frame_index)]
    _draw_world(axes[0], frame, run["gamma"], run["executed_states"], display_cap)
    _draw_control_space(axes[1], frame)
    _draw_h_audit(axes[2], frame, run["gamma"])
    output_label = (
        "accepted-set weighted mean"
        if frame["accepted"] else "ALL-REJECTED safest-fallback weighted mean"
    )
    figure.suptitle(
        f"SafeMPPI candidate mechanism · matched-ID episode {run['episode']} · "
        f"gamma={run['gamma']:g} · frame {frame['step']}\n"
        f"accepted/rejected = {frame['accepted']}/{frame['rejected']} of "
        f"{run['config']['num_samples']} · displayed {min(display_cap, run['config']['num_samples'])} · "
        f"output: {output_label} · weighted H10 "
        f"{'PASS' if frame['mean_feasible'] else 'FAIL'}",
        fontsize=11.5, y=.975,
    )
    axes[0].legend(handles=[
        Line2D([], [], color=GREEN, lw=.8, label="accepted candidate plan"),
        Line2D([], [], color=RED, lw=.7, marker="x", label="rejected · first violating state"),
        Line2D([], [], color=PURPLE, lw=2.2, label="accepted-set weighted mean"),
        Line2D([], [], color=ORANGE, lw=2.2, label="all-rejected fallback / uncertified mean"),
        Line2D([], [], color="black", lw=3.0, label="applied first action only"),
        Line2D([], [], color=BLUE, lw=1.4, label="current nominal polytope + H levels"),
    ], loc="lower right", fontsize=6.7, framealpha=.92)
    return axes


def render(episode: int, gamma: float, output_dir, *, selected_step=26,
           display_cap=512, frame_stride=2, fps=7, dpi=112, device="cpu") -> dict:
    if int(display_cap) <= 0 or int(frame_stride) <= 0 or int(fps) <= 0:
        raise ValueError("display_cap, frame_stride, and fps must be positive")
    run = collect_episode(int(episode), float(gamma), device=device)
    if not run["frames"]:
        raise RuntimeError("expert episode produced no planning contexts")
    steps = [int(row["step"]) for row in run["frames"]]
    if int(selected_step) not in steps:
        raise ValueError(f"selected step {selected_step} is not in {steps}")
    selected_index = steps.index(int(selected_step))
    indices = list(range(0, len(steps), int(frame_stride)))
    if indices[-1] != len(steps) - 1:
        indices.append(len(steps) - 1)
    if selected_index not in indices:
        indices.append(selected_index); indices.sort()

    output = Path(output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    gamma_tag = f"{float(gamma):g}".replace(".", "p")
    stem = f"hp100_safemppi_mechanism_g{gamma_tag}_ep{int(episode)}"
    mp4 = output / f"{stem}.mp4"
    png = output / f"{stem}_step{int(selected_step):03d}.png"
    pdf = output / f"{stem}_step{int(selected_step):03d}.pdf"
    npz = output / f"{stem}_step{int(selected_step):03d}_trace.npz"
    contract_path = output / f"{stem}.json"
    for path in (mp4, png, pdf, npz, contract_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite output: {path}")

    figure, axes = _new_figure()
    movie = animation.FuncAnimation(
        figure, lambda index: _draw_frame(figure, axes, run, index, int(display_cap)),
        frames=indices, interval=1000.0 / int(fps), blit=False,
    )
    movie.save(mp4, writer=animation.FFMpegWriter(fps=int(fps), bitrate=4800), dpi=int(dpi))
    plt.close(figure)
    figure, axes = _new_figure()
    _draw_frame(figure, axes, run, selected_index, int(display_cap))
    figure.savefig(png, dpi=int(dpi) + 35); figure.savefig(pdf); plt.close(figure)

    selected = run["frames"][selected_index]
    draw_indices = proportional_subset(selected["candidate_feasible"], int(display_cap))
    np.savez_compressed(
        npz,
        candidate_states=selected["candidate_states"],
        candidate_controls=selected["candidate_controls"],
        candidate_accepted=selected["candidate_feasible"],
        candidate_weights=selected["candidate_weights"],
        first_violation_step=selected["first_violation"],
        displayed_indices=draw_indices,
        weighted_controls=selected["mean_controls"],
        weighted_states=selected["mean_states"], weighted_h=selected["mean_h"],
        weighted_violations=selected["mean_violations"],
        applied_action=selected["action"], A=selected["polytope"]["A"],
        b=selected["polytope"]["b"], ref=selected["polytope"]["ref"],
        margins=selected["polytope"]["margins"],
    )
    frame_rows = [dict(
        frame_index=index, step=int(row["step"]), accepted=int(row["accepted"]),
        rejected=int(row["rejected"]),
        selection_semantics=row["selection_semantics"],
        weighted_h10_pass=bool(row["mean_feasible"]),
        weighted_first_violation=(
            None if not row["mean_violations"].any()
            else int(np.flatnonzero(row["mean_violations"])[0] + 1)
        ),
    ) for index, row in enumerate(run["frames"])]
    contract = dict(
        status=STATUS, role=(
            "read-only rerun of the locked current-tangent Hp100 SafeMPPI expert; "
            "shows current candidate acceptance, MPPI weighting, and all-rejected fallback; "
            "never uses the post-hoc future-executed H10 training window"
        ),
        source_commit=_git_head(), episode=int(episode), gamma=float(gamma),
        scene=SS.scene_profile("matched_id"), outcome=dict(
            success=run["success"], collision=run["collision"], timeout=run["timeout"],
            steps=len(run["frames"]), min_clearance=run["min_clearance"],
        ),
        planner_config=run["config"],
        candidate_contract=dict(
            generated_per_replan=int(run["config"]["num_samples"]),
            displayed_per_frame=min(int(display_cap), int(run["config"]["num_samples"])),
            display_selection="deterministic proportional accepted/rejected subset",
            acceptance=(
                "all H=10 transitions satisfy H_P(x_next) >= (1-gamma) H_P(x_current) "
                "under one frozen current nominal polytope"
            ),
            rejected_marker="small red x at the first violating predicted state",
            output=(
                "temperature-weighted accepted candidates when any exist; when none exist, "
                "the planner's declared safest-fallback weights all rejected candidates"
            ),
        ),
        dataset_target_contract=DATA.target_contract(),
        selected_frame=dict(
            step=int(selected_step), frame_index=int(selected_index),
            accepted=int(selected["accepted"]), rejected=int(selected["rejected"]),
            selection_semantics=selected["selection_semantics"],
            weighted_h10_pass=bool(selected["mean_feasible"]),
            action_mean_max_abs_error=float(selected["action_mean_max_abs_error"]),
            trace_npz=str(npz),
        ),
        frames=frame_rows,
        render=dict(frame_stride=int(frame_stride), fps=int(fps), rendered_frames=len(indices)),
        outputs={},
    )
    for name, path in (("mp4", mp4), ("png", png), ("pdf", pdf), ("trace_npz", npz)):
        contract["outputs"][name] = dict(
            path=str(path), sha256=_sha256(path), bytes=path.stat().st_size,
        )
    _write_json(contract_path, contract)
    contract["contract_path"] = str(contract_path)
    return contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selected-step", type=int, default=26)
    parser.add_argument("--display-cap", type=int, default=512)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--fps", type=int, default=7)
    parser.add_argument("--dpi", type=int, default=112)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = render(
        args.episode, args.gamma, args.output_dir,
        selected_step=args.selected_step, display_cap=args.display_cap,
        frame_stride=args.frame_stride, fps=args.fps, dpi=args.dpi, device=args.device,
    )
    print(json.dumps({
        "status": report["status"], "contract": report["contract_path"],
        "mp4": report["outputs"]["mp4"]["path"],
        "png": report["outputs"]["png"]["path"],
    }), flush=True)


if __name__ == "__main__":
    main()
