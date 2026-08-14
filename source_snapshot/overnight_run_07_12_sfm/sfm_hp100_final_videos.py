"""Trace-faithful single-scene renderers for final SFM paper videos.

The module consumes already authenticated traces.  It does not change or
rerun the scientific algorithms.  One invocation renders one scene and one
gamma with a fixed camera; method labels live in a separate label sheet.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Polygon
from matplotlib import patheffects
from matplotlib.colors import Normalize
import numpy as np
import torch

import sfm_b1_viz as BV
import sfm_hp100_branch_viz as RAW_VIZ
import sfm_hp100_expert_mechanism_viz as EXPERT
import sfm_hp100_paper_video_style as STYLE
import sfm_scene as SS


STATUS = "SFM_HP100_FINAL_PAPER_VIDEO_COMPLETE"


def _gamma_tag(gamma: float) -> str:
    return f"{float(gamma):g}".replace(".", "p")


def _writer(fps: int, bitrate: int = 4800):
    return animation.FFMpegWriter(
        fps=int(fps), codec="libx264", bitrate=int(bitrate),
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )


def _draw_pedestrians(axis, ped_xy, ped_vel=None) -> None:
    for index, position in enumerate(np.asarray(ped_xy, float)):
        axis.add_patch(Circle(
            position, SS.R_PED, facecolor=STYLE.PEDESTRIAN_GRAY,
            edgecolor="#333333", linewidth=0.45, alpha=0.72, zorder=5,
        ))
        if ped_vel is not None:
            prediction = position[None] + (
                np.arange(11)[:, None] * SS.DT
                * np.asarray(ped_vel, float)[index][None]
            )
            axis.plot(
                prediction[:, 0], prediction[:, 1], ".--",
                color=STYLE.PEDESTRIAN_GRAY, lw=0.42, ms=1.25,
                alpha=0.38, zorder=2,
            )


def _draw_robot_goal(axis, state) -> None:
    axis.plot(float(state[0]), float(state[1]), "o",
              color=STYLE.EXECUTED_BLACK, ms=5.8, zorder=30)
    axis.plot(float(SS.GOAL[0]), float(SS.GOAL[1]), "*",
              color=STYLE.GOAL_GOLD, mec=STYLE.EXECUTED_BLACK,
              mew=0.5, ms=12.0, zorder=30)


def _draw_verifier(axis, gamma: float, sidecar: dict) -> None:
    sidecar = dict(sidecar)
    sidecar["faces"] = [
        face if hasattr(face, "feasible") else SimpleNamespace(**face)
        for face in sidecar["faces"]
    ]
    geometry = RAW_VIZ.verifier_geometry({
        "gamma": float(gamma), "proposal_result": sidecar,
    })
    BV._draw_level_polygons(
        axis, geometry["levels"], color=STYLE.VERIFIER_GREEN,
        linewidth=0.52, alpha=0.62, zorder=3,
    )
    outer = np.asarray(geometry["outer"], float)
    axis.plot(
        np.r_[outer[:, 0], outer[0, 0]],
        np.r_[outer[:, 1], outer[0, 1]],
        color=STYLE.VERIFIER_GREEN, lw=STYLE.POLYTOPE_LW,
        alpha=0.95, zorder=3.2,
    )


def _finalize(output: Path, metadata: dict) -> dict:
    payload = {
        "status": STATUS, "style_version": STYLE.VERSION,
        **metadata,
        "mp4": str(output.resolve()),
        "mp4_sha256": STYLE.sha256_file(output),
        "bytes": output.stat().st_size,
    }
    sidecar = output.with_suffix(".json")
    STYLE.write_json(sidecar, payload)
    payload["sidecar"] = str(sidecar.resolve())
    return payload


def build_manifest(root: str | Path) -> dict:
    """Hash and validate one completed FINAL_VIDEOS delivery tree."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    manifest_path = root / "MANIFEST.json"
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path == manifest_path:
            continue
        files.append({
            "path": str(path.relative_to(root)),
            "bytes": int(path.stat().st_size),
            "sha256": STYLE.sha256_file(path),
        })
    mp4s = [row for row in files if row["path"].endswith(".mp4")]
    for row in mp4s:
        sidecar_path = root / Path(row["path"]).with_suffix(".json")
        if not sidecar_path.is_file():
            raise FileNotFoundError(f"MP4 lacks JSON sidecar: {row['path']}")
        sidecar = json.loads(sidecar_path.read_text())
        if sidecar.get("mp4_sha256") != row["sha256"]:
            raise RuntimeError(f"MP4 sidecar digest mismatch: {row['path']}")
    payload = {
        "status": "SFM_HP100_FINAL_VIDEOS_MANIFEST_COMPLETE",
        "style_version": STYLE.VERSION,
        "file_count": len(files),
        "video_count": len(mp4s),
        "files": files,
    }
    STYLE.write_json(manifest_path, payload)
    return payload


def render_expert_run(
    run: dict, output: str | Path, *, fps: int = 7,
    frame_stride: int = 2, display_cap: int = 512,
) -> dict:
    """Render the locked ID SafeMPPI candidate mechanism as one world panel."""
    STYLE.apply_computer_modern_style()
    if not run.get("frames"):
        raise ValueError("expert trace contains no replanning frames")
    indices = list(range(0, len(run["frames"]), int(frame_stride)))
    if indices[-1] != len(run["frames"]) - 1:
        indices.append(len(run["frames"]) - 1)
    output = Path(output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    figure, axis = plt.subplots(figsize=(6.8, 6.8))

    def update(index):
        axis.clear()
        frame = run["frames"][int(index)]
        state = np.asarray(frame["state"], float)
        _draw_pedestrians(axis, frame["ped_xy"], frame["ped_vel"])
        polytope = frame["polytope"]
        for _, polygon in BV._level_polygons(
            polytope["A"], polytope["margins"], state[:2],
            float(run["gamma"]), H=10,
        ):
            axis.add_patch(Polygon(
                polygon, closed=True, fill=False,
                edgecolor=STYLE.POSITIVE_BLUE, linewidth=0.48,
                alpha=0.24, zorder=1.5,
            ))
        nominal = BV.halfspace_polygon(polytope["A"], polytope["b"])
        if nominal is None:
            raise RuntimeError("expert nominal polytope is unbounded")
        axis.add_patch(Polygon(
            nominal, closed=True, fill=False,
            edgecolor=STYLE.POSITIVE_BLUE, linewidth=STYLE.POLYTOPE_LW,
            alpha=0.92, zorder=2,
        ))
        draw_indices = EXPERT.proportional_subset(
            frame["candidate_feasible"], int(display_cap),
        )
        states = np.asarray(frame["candidate_states"], float)[draw_indices, :, :2]
        feasible = np.asarray(frame["candidate_feasible"], bool)[draw_indices]
        for accepted, color in (
            (False, STYLE.NEGATIVE_RED), (True, STYLE.POSITIVE_BLUE),
        ):
            paths = states[feasible == accepted]
            if len(paths):
                axis.add_collection(LineCollection(
                    paths, colors=color, linewidths=STYLE.ROLLOUT_LW,
                    alpha=0.11 if accepted else 0.08, zorder=4,
                ))
        rejected_indices = draw_indices[~feasible]
        if len(rejected_indices):
            violation_steps = np.asarray(frame["first_violation"])[rejected_indices]
            points = np.asarray(frame["candidate_states"])[
                rejected_indices, violation_steps, :2
            ]
            axis.scatter(points[:, 0], points[:, 1], marker="x", s=12,
                         linewidths=0.75, c=STYLE.NEGATIVE_RED,
                         alpha=0.62, zorder=7)
        mean = np.asarray(frame["mean_states"], float)
        axis.plot(mean[:, 0], mean[:, 1], color=STYLE.WEIGHTED_GOLD,
                  lw=STYLE.ROLLOUT_LW, alpha=0.98, zorder=15)
        history = np.asarray(run["executed_states"][:int(index) + 1], float)
        axis.plot(history[:, 0], history[:, 1], color=STYLE.EXECUTED_BLACK,
                  lw=STYLE.EXECUTED_LW, zorder=20)
        _draw_robot_goal(axis, state)
        STYLE.fixed_world_frame(axis)
        return []

    movie = animation.FuncAnimation(
        figure, update, frames=indices, interval=1000 / int(fps), blit=False,
    )
    movie.save(output, writer=_writer(fps), dpi=130)
    plt.close(figure)
    return _finalize(output, {
        "kind": "matched_id_safemppi_candidate_mechanism",
        "episode": int(run["episode"]), "gamma": float(run["gamma"]),
        "outcome": {
            "success": bool(run["success"]),
            "collision": bool(run["collision"]),
            "timeout": bool(run["timeout"]),
        },
        "candidate_population": int(run["config"]["num_samples"]),
        "displayed_candidate_cap": int(display_cap),
        "fixed_camera": [float(SS.TASK_LO), float(SS.TASK_HI)],
        "embedded_labels": [],
    })


def render_raw_case(
    case: dict, output: str | Path, *, fps: int = 7,
    frame_stride: int = 1, show_safety_badge: bool = False,
    bounds=None,
) -> dict:
    """Render one authenticated raw generative-policy rollout."""
    STYLE.apply_computer_modern_style()
    traces = list(case.get("traces", ()))
    if not traces:
        raise ValueError("raw case has no proposal trace")
    indices = list(range(0, len(traces), int(frame_stride)))
    if indices[-1] != len(traces) - 1:
        indices.append(len(traces) - 1)
    output = Path(output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    figure, axis = plt.subplots(figsize=(6.8, 6.8))

    def update(index):
        axis.clear()
        current = traces[int(index)]
        _draw_pedestrians(axis, current["ped_xy"], current["ped_vel"])
        for row in traces[:int(index) + 1]:
            segment = np.asarray(row["proposal_result"]["segment"], float)
            positive = row["proposal_label"] == "full_h_positive"
            axis.plot(
                segment[:, 0], segment[:, 1],
                color=(STYLE.POSITIVE_BLUE if positive else STYLE.NEGATIVE_RED),
                lw=STYLE.SAMPLE_LW, alpha=0.18 if row is not current else 0.96,
                zorder=8,
            )
            if not positive:
                axis.plot(segment[-1, 0], segment[-1, 1], "x",
                          color=STYLE.NEGATIVE_RED, ms=5.5, mew=1.2, zorder=12)
        states = [np.asarray(row["state"], float)[:2]
                  for row in traces[:int(index) + 1]]
        states.append(np.asarray(current["next_state"], float)[:2])
        states = np.asarray(states)
        axis.plot(states[:, 0], states[:, 1], color=STYLE.EXECUTED_BLACK,
                  lw=STYLE.EXECUTED_LW, zorder=20)
        positive = current["proposal_label"] == "full_h_positive"
        if positive:
            _draw_verifier(axis, float(case["gamma"]), current["proposal_result"])
        _draw_robot_goal(axis, current["state"])
        STYLE.fixed_world_frame(axis, bounds=bounds)
        STYLE.safety_badge(axis, positive, enabled=show_safety_badge)
        return []

    movie = animation.FuncAnimation(
        figure, update, frames=indices, interval=1000 / int(fps), blit=False,
    )
    movie.save(output, writer=_writer(fps), dpi=130)
    plt.close(figure)
    return _finalize(output, {
        "kind": "raw_generative_policy_branch",
        "scene_profile": str(case["scene_profile"]),
        "episode": int(case["episode"]), "gamma": float(case["gamma"]),
        "outcome": str(case["outcome"]),
        "trajectory_validity": float(case["trajectory_validity"]),
        "show_safety_badge": bool(show_safety_badge),
        "fixed_camera": (list(map(float, bounds)) if bounds is not None
                         else [float(SS.TASK_LO), float(SS.TASK_HI)]),
        "embedded_labels": (["multi_step_safety"] if show_safety_badge else []),
    })


def _visible_uncertainty_indices(
    attempt: dict, count: int | None = None,
) -> np.ndarray:
    sigma = np.asarray(attempt["selected_sigma"], float)
    if count is None:
        count = len(sigma)
    return np.argsort(sigma)[::-1][:min(int(count), len(sigma))]


def _acquisition_phases(rows: list[dict]) -> list[tuple[dict, str, int | None]]:
    """Show only the final repair family and never preview an executed state."""
    phases = []
    for row in rows:
        phases.append((row, "raw", None))
        attempts = list(row.get("repair_attempts", ()))
        final_attempt = len(attempts) - 1 if attempts else None
        if final_attempt is not None:
            phases.extend([
                (row, "clear", None),
                (row, "uncertainty", final_attempt),
            ])
            if attempts[final_attempt].get("selected_local") is not None:
                phases.append((row, "selected", final_attempt))
        if row.get("executed_group") in ("P1", "P2"):
            phases.append((row, "execute", final_attempt))
    return phases


def render_acquisition_clip(
    events: list[dict], output: str | Path, *, center_step: int,
    n_before: int = 3, n_after: int = 3, fps: int = 3,
    show_safety_badge: bool = True, bounds=None,
) -> dict:
    """Render a slow, faithful P1-then-conditional-repair acquisition clip.

    The stored trace contains the actual uncertainty-selected B=16 paths, not
    the full K=64 population.  All B paths are shown with the same viridis
    uncertainty mapping as the 3-D evaluator.
    """
    STYLE.apply_computer_modern_style()
    rows = sorted(events, key=lambda row: int(row["step"]))
    rows = [row for row in rows
            if int(center_step) - int(n_before) <= int(row["step"])
            <= int(center_step) + int(n_after)]
    if not rows:
        raise ValueError("requested acquisition window has no events")
    phases = _acquisition_phases(rows)
    output = Path(output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    figure, axis = plt.subplots(figsize=(6.8, 6.8))

    def update(payload):
        row, phase, attempt_index = payload
        axis.clear()
        _draw_pedestrians(axis, row["ped_xy"], row["ped_vel"])
        prior = [item for item in rows if int(item["step"]) <= int(row["step"])]
        states = [np.asarray(item["state_before"], float)[:2] for item in prior]
        executing = phase == "execute" and row.get("executed_group") in ("P1", "P2")
        if executing:
            states.append(np.asarray(row["state_after"], float)[:2])
        states = np.asarray(states)
        axis.plot(states[:, 0], states[:, 1], color=STYLE.EXECUTED_BLACK,
                  lw=STYLE.EXECUTED_LW, zorder=20)
        past_states = states[:-1]
        if len(past_states):
            axis.scatter(
                past_states[:, 0], past_states[:, 1], s=7.0,
                color=STYLE.EXECUTED_BLACK, edgecolors="none", zorder=21,
            )
        raw = np.asarray(row["raw_segment"], float)
        raw_positive = bool(row["raw_verification"]["valid"])
        show_raw = phase == "raw" or (phase == "execute" and raw_positive)
        safe = raw_positive if show_raw else None
        sidecar = row.get("raw_sidecar") if show_raw and raw_positive else None
        if show_raw:
            axis.plot(
                raw[:, 0], raw[:, 1],
                color=(STYLE.POSITIVE_BLUE if raw_positive else STYLE.NEGATIVE_RED),
                lw=STYLE.SAMPLE_LW, alpha=0.95, zorder=9,
            )
        if show_raw and not raw_positive:
            axis.plot(raw[-1, 0], raw[-1, 1], "x",
                      color=STYLE.NEGATIVE_RED, ms=6, mew=1.25, zorder=14)
        if attempt_index is not None:
            attempt = row["repair_attempts"][int(attempt_index)]
            segments = np.asarray(attempt["segments"], float)
            sigma = np.asarray(attempt["selected_sigma"], float)
            emphasized = set(map(int, _visible_uncertainty_indices(attempt)))
            norm = Normalize(float(sigma.min()), float(sigma.max()) + 1e-12)
            cmap = plt.get_cmap("viridis")
            for local, segment in enumerate(segments):
                color = cmap(norm(sigma[local])) if local in emphasized \
                    else STYLE.UNCERTAINTY_GRAY
                axis.plot(segment[:, 0], segment[:, 1], color=color,
                          lw=STYLE.ROLLOUT_LW, alpha=0.34,
                          zorder=6)
            chosen = attempt.get("selected_local")
            if phase in ("selected", "execute") and chosen is not None:
                chosen = int(chosen)
                segment = segments[chosen]
                valid = bool(attempt["verification"][chosen]["valid"])
                line, = axis.plot(
                    segment[:, 0], segment[:, 1],
                    color=(STYLE.POSITIVE_BLUE if valid else STYLE.NEGATIVE_RED),
                    lw=STYLE.SAMPLE_LW, alpha=1.0, zorder=18,
                )
                line.set_path_effects([
                    patheffects.Stroke(linewidth=6.0, foreground="white", alpha=0.90),
                    patheffects.Stroke(linewidth=4.2, foreground=cmap(norm(sigma[chosen])),
                                       alpha=0.58),
                    patheffects.Normal(),
                ])
                safe = valid
                sidecar = attempt.get("selected_sidecar") if valid else None
        if safe and sidecar is not None:
            _draw_verifier(axis, float(row["gamma"]), sidecar)
        robot_state = row["state_after"] if executing else row["state_before"]
        _draw_robot_goal(axis, robot_state)
        STYLE.fixed_world_frame(axis, bounds=bounds)
        if safe is not None:
            STYLE.safety_badge(axis, safe, enabled=show_safety_badge)
        return []

    movie = animation.FuncAnimation(
        figure, update, frames=phases, interval=1000 / int(fps), blit=False,
    )
    movie.save(output, writer=_writer(fps, bitrate=5200), dpi=130)
    plt.close(figure)
    first = rows[0]
    return _finalize(output, {
        "kind": "conditional_uncertainty_repair_acquisition",
        "scene_profile": "double_density_velocity_ood",
        "episode": int(first["scenario_id"]),
        "lineage": str(first["lineage"]),
        "gamma": float(first["gamma"]),
        "center_step": int(center_step),
        "step_window": [int(rows[0]["step"]), int(rows[-1]["step"])],
        "scientific_K": 64, "scientific_B": 16,
        "displayed_B": 16, "uncertainty_emphasized_B": 16,
        "uncertainty_normalization": "per-attempt B min-max",
        "repair_family_display": "final attempted B only",
        "executed_state_preview": False,
        "past_executed_state_markers": True,
        "full_K_paths_available_in_trace": False,
        "faithful_order": "raw P1 first; uncertainty repair only after exact-negative raw proposal",
        "show_safety_badge": bool(show_safety_badge),
        "fixed_camera": (list(map(float, bounds)) if bounds is not None else None),
        "embedded_labels": (["multi_step_safety"] if show_safety_badge else []),
    })


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    sub = value.add_subparsers(dest="command", required=True)
    labels = sub.add_parser("labels")
    labels.add_argument("--output-stem", required=True)
    manifest = sub.add_parser("manifest")
    manifest.add_argument("--root", required=True)
    expert = sub.add_parser("expert")
    expert.add_argument("--trace", required=True)
    expert.add_argument("--output", required=True)
    expert.add_argument("--fps", type=int, default=7)
    expert.add_argument("--frame-stride", type=int, default=2)
    expert.add_argument("--display-cap", type=int, default=512)
    raw = sub.add_parser("raw")
    raw.add_argument("--trace", required=True)
    raw.add_argument("--case-key", required=True)
    raw.add_argument("--output", required=True)
    raw.add_argument("--fps", type=int, default=7)
    raw.add_argument("--frame-stride", type=int, default=1)
    raw.add_argument("--show-safety-badge", action="store_true")
    raw.add_argument("--bounds", type=float, nargs="+")
    acquisition = sub.add_parser("acquisition")
    acquisition.add_argument("--trace", required=True)
    acquisition.add_argument("--lineage", required=True)
    acquisition.add_argument("--center-step", type=int, required=True)
    acquisition.add_argument("--output", required=True)
    acquisition.add_argument("--n-before", type=int, default=3)
    acquisition.add_argument("--n-after", type=int, default=3)
    acquisition.add_argument("--fps", type=int, default=3)
    acquisition.add_argument("--show-safety-badge", action="store_true")
    acquisition.add_argument("--bounds", type=float, nargs="+")
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.command == "labels":
        result = STYLE.render_label_sheet(args.output_stem)
    elif args.command == "manifest":
        result = build_manifest(args.root)
    elif args.command == "expert":
        result = render_expert_run(
            _load(args.trace), args.output, fps=args.fps,
            frame_stride=args.frame_stride, display_cap=args.display_cap,
        )
    elif args.command == "raw":
        bundle = _load(args.trace)
        cases = bundle.get("cases", bundle)
        result = render_raw_case(
            cases[args.case_key], args.output, fps=args.fps,
            frame_stride=args.frame_stride,
            show_safety_badge=args.show_safety_badge,
            bounds=args.bounds,
        )
    else:
        trace = _load(args.trace)
        events = [row for row in trace["events"]
                  if str(row["lineage"]) == str(args.lineage)]
        result = render_acquisition_clip(
            events, args.output, center_step=args.center_step,
            n_before=args.n_before, n_after=args.n_after, fps=args.fps,
            show_safety_badge=args.show_safety_badge, bounds=args.bounds,
        )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
