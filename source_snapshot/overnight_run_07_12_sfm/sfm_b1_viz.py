"""B1 query/certificate animation and four candidate-specific zoom panels."""
from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon
import numpy as np
import torch

import _paths  # noqa: F401
from polar_grid import polytope_HP
import sfm_scene as SS


GRAY = "#8c8c8c"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D62728"
BLUE = "#0057FF"


def halfspace_polygon(A, b, tol=1.0e-7):
    """Return the bounded 2-D polygon for ``A x <= b`` from face intersections."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2 or A.shape[1] != 2 or b.shape != (len(A),):
        raise ValueError("expected A[m,2] and b[m]")
    vertices = []
    for first in range(len(A)):
        for second in range(first + 1, len(A)):
            matrix = np.stack([A[first], A[second]])
            if abs(float(np.linalg.det(matrix))) < 1.0e-10:
                continue
            point = np.linalg.solve(matrix, np.array([b[first], b[second]]))
            if np.all(A @ point <= b + float(tol)):
                vertices.append(point)
    if len(vertices) < 3:
        return None
    vertices = np.unique(np.round(np.asarray(vertices), decimals=9), axis=0)
    if len(vertices) < 3:
        return None
    center = vertices.mean(axis=0)
    order = np.argsort(np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0]))
    return vertices[order]


def _level_polygons(A, margins, center, gamma, H=10):
    """Build the H DTCBF level sets ``A x <= A c + beta_h m``."""
    A = np.asarray(A, dtype=float).reshape(-1, 2)
    margins = np.asarray(margins, dtype=float).reshape(-1)
    center = np.asarray(center, dtype=float).reshape(2)
    if len(A) != len(margins) or not len(A):
        raise ValueError("level-set faces and margins do not align")
    if not (0.0 < float(gamma) <= 1.0):
        raise ValueError("gamma must lie in (0,1]")
    polygons = []
    for horizon in range(1, int(H) + 1):
        beta = 1.0 - (1.0 - float(gamma)) ** horizon
        polygon = halfspace_polygon(A, A @ center + beta * margins)
        if polygon is None:
            raise ValueError(f"unbounded or empty level set at h={horizon}")
        polygons.append((horizon, polygon))
    return polygons


def nominal_level_polygons(state, ped_xy, gamma, H=10):
    """Exact nominal SafeMPPI H_P level sets for horizons one through H."""
    obstacles = np.concatenate([
        np.asarray(ped_xy), np.full((len(ped_xy), 1), SS.R_PED)
    ], axis=1)
    _, (A, _, margins) = polytope_HP(
        np.asarray(state)[:2], obstacles, sensing=SS.R_SENSE, n_base=16
    )
    return _level_polygons(A, margins, np.asarray(state)[:2], gamma, H=H)


def verifier_level_polygons(trace, query_row, H=10):
    """Candidate-fitted verifier sets; all feasible real and artificial faces bound them."""
    result = query_row["result"]
    if not result.get("resolved") or int(result.get("y", 0)) != 1:
        raise ValueError("verifier level sets require a resolved verifier-positive query")
    if not bool(result.get("full_h", False)) or int(result.get("terminal_step", H)) != int(H):
        raise ValueError("verifier visualization requires a full-H query")
    faces = [face for face in result.get("faces", []) if bool(face.feasible)]
    if not faces:
        raise ValueError("full-H positive query has no feasible verifier faces")
    A = np.stack([np.asarray(face.a, dtype=float) for face in faces])
    margins = np.asarray([float(face.m) for face in faces])
    center = np.asarray(result["segment"], dtype=float)[0]
    return _level_polygons(A, margins, center, float(trace["gamma"]), H=H)


def _draw_level_polygons(axis, polygons, *, color, linewidth, alpha, zorder):
    for horizon, polygon in polygons:
        shade = 0.35 + 0.55 * horizon / len(polygons)
        axis.add_patch(Polygon(
            polygon, closed=True, fill=False, edgecolor=color, linewidth=linewidth,
            alpha=alpha * shade, zorder=zorder,
        ))


def _draw_pedestrians(axis, ped_xy, *, alpha=.72):
    for position in np.asarray(ped_xy):
        axis.add_patch(Circle(
            position, SS.R_PED, facecolor="#555555", edgecolor="#222222",
            linewidth=.45, alpha=alpha, zorder=5,
        ))


def _set_world_frame(axis):
    axis.set_aspect("equal")
    axis.set_xlim(SS.TASK_LO, SS.TASK_HI)
    axis.set_ylim(SS.TASK_LO, SS.TASK_HI)
    axis.grid(alpha=.15)


def _candidate_status(trace, candidate_id):
    for row in trace["query_rows"]:
        if int(row["candidate_id"]) == int(candidate_id):
            if not row["result"].get("resolved"):
                return "error", row
            if (not bool(row["result"].get("full_h"))
                    or int(row["result"].get("terminal_step", -1)) != 10):
                raise ValueError("visualization received a legacy partial B1 query")
            return ("positive" if int(row["result"]["y"]) else "negative"), row
    return "unqueried", None


def _trace_candidate(trace, candidate_id):
    matches = [row for row in trace["all_K"] if int(row["candidate_id"]) == int(candidate_id)]
    if len(matches) != 1:
        raise ValueError(f"candidate {candidate_id} appears {len(matches)} times")
    return matches[0]


def _draw_common(axis, trace, *, nominal_levels=True):
    state = np.asarray(trace["state"])
    ped_xy = np.asarray(trace["ped_xy"])
    ped_vel = np.asarray(trace["ped_vel"])
    if nominal_levels:
        _draw_level_polygons(
            axis, nominal_level_polygons(state, ped_xy, trace["gamma"]),
            color=BLUE, linewidth=.75, alpha=.72, zorder=2,
        )
    _draw_pedestrians(axis, ped_xy)
    for index in range(len(ped_xy)):
        prediction = ped_xy[index][None] + np.arange(11)[:, None] * SS.DT * ped_vel[index][None]
        axis.plot(prediction[:, 0], prediction[:, 1], ".--", color=GRAY, ms=2.2, lw=.65, alpha=.65)
    axis.plot(state[0], state[1], "o", color=BLUE, ms=7, zorder=8)
    _set_world_frame(axis)


def _query_legend(axis):
    handles = [
        Line2D([], [], color=GRAY, lw=.8, label="K generated"),
        Line2D([], [], color=ORANGE, lw=1.5, marker=".", label="B queried"),
        Line2D([], [], color=GREEN, lw=2.0, marker=".", label="verifier positive"),
        Line2D([], [], color=RED, lw=2.0, marker="x", label="rejected / worst h"),
        Line2D([], [], color=BLUE, lw=3.2, label="executed first action"),
        Line2D([], [], color=BLUE, lw=.9, label="nominal H_P levels h=1..10"),
        Line2D([], [], color=GREEN, lw=.9, label="verifier levels h=1..10"),
    ]
    axis.legend(handles=handles, loc="upper left", fontsize=6.5, framealpha=.92)


def draw_query_frame(axis, trace, *, show_legend=True, show_executed_levels=True):
    _draw_common(axis, trace)
    selected = set(map(int, trace["selected_ids"]))
    executed = trace.get("executed_id")
    # K is always visible in gray; B is then overlaid in orange before its label color.
    for row in trace["all_K"]:
        path = np.asarray(row["segment"])
        axis.plot(path[:, 0], path[:, 1], color=GRAY, lw=.65, alpha=.35, zorder=3)
    for candidate in sorted(selected):
        path = np.asarray(_trace_candidate(trace, candidate)["segment"])
        axis.plot(path[:, 0], path[:, 1], color=ORANGE, lw=1.35, marker=".", ms=2.0,
                  alpha=.95, zorder=4)
    for candidate in sorted(selected):
        status, query = _candidate_status(trace, candidate)
        if status not in ("positive", "negative"):
            continue
        path = np.asarray(_trace_candidate(trace, candidate)["segment"])
        color = GREEN if status == "positive" else RED
        axis.plot(path[:, 0], path[:, 1], color=color, lw=1.9, marker=".", ms=2.2,
                  alpha=.96, zorder=5)
        if status == "negative":
            worst = int(query["result"].get("diagnostics", {}).get("worst_t", len(path) - 1))
            worst = min(max(worst, 1), len(path) - 1)
            axis.plot(path[worst, 0], path[worst, 1], marker="x", color=RED,
                      ms=5.5, mew=1.3, zorder=7)
    if executed is not None:
        path = np.asarray(_trace_candidate(trace, executed)["segment"])
        axis.plot(path[:2, 0], path[:2, 1], color=BLUE, lw=3.2, zorder=9)
        axis.annotate("", xy=path[1], xytext=path[0],
                      arrowprops=dict(arrowstyle="->", color=BLUE, lw=2.8))
        if show_executed_levels:
            status, query = _candidate_status(trace, executed)
            if status != "positive":
                raise ValueError("executed candidate is not a resolved verifier-positive query")
            _draw_level_polygons(
                axis, verifier_level_polygons(trace, query),
                color=GREEN, linewidth=.85, alpha=.78, zorder=2.5,
            )
    axis.set_title(
        f"r{trace['round']} s{trace['scenario_id']} gamma={trace['gamma']} t={trace['step']}"
        + (" | NVP" if executed is None else "")
    )
    if show_legend:
        _query_legend(axis)
    if abs(float(trace["gamma"]) - 1.0) < 1.0e-9:
        axis.text(.99, .99, "gamma=1: h=1..10 levels coincide",
                  transform=axis.transAxes, ha="right", va="top", fontsize=6, color="#333333")


def _draw_time_indexed_sets(axis, trace, query_row):
    result = query_row["result"]
    segment = np.asarray(result["segment"])
    _draw_level_polygons(
        axis, verifier_level_polygons(trace, query_row),
        color=GREEN, linewidth=.85, alpha=.78, zorder=2.5,
    )
    for horizon, position in enumerate(segment[1:11], start=1):
        axis.plot(position[0], position[1], "o", ms=2.8,
                  color=plt.cm.Greens(.35 + .6 * horizon / 10), zorder=7)


def render_zoom_panels(trace, output):
    selected = list(map(int, trace["selected_ids"]))[:4]
    figure, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    for axis, candidate in zip(axes.flat, selected):
        _draw_common(axis, trace)
        status, query = _candidate_status(trace, candidate)
        path = np.asarray(_trace_candidate(trace, candidate)["segment"])
        color = GREEN if status == "positive" else RED if status == "negative" else ORANGE
        axis.plot(path[:, 0], path[:, 1], color=color, lw=2.4, zorder=8)
        if status == "positive":
            _draw_time_indexed_sets(axis, trace, query)
        axis.set_title(f"queried candidate {candidate}: {status}; each h uses its h-set")
    for axis in axes.flat[len(selected):]:
        axis.set_visible(False)
    if selected:
        _query_legend(axes.flat[0])
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def _trace_index(traces):
    index = {}
    for trace in traces:
        key = (round(float(trace["gamma"]), 6), int(trace["scenario_id"]), int(trace["step"]))
        if key in index:
            raise ValueError(f"duplicate trace key {key}")
        index[key] = trace
    return index


def render_selector_comparison(
        margin_traces, cost_traces, case_keys, margin_output, cost_output,
        *, gammas=(.1, .5, 1.0)):
    """Render paired 3x3 snapshots without choosing or replacing any requested case.

    ``case_keys`` is exactly three explicit ``(scenario_id, step)`` pairs.  Every
    pair must exist for every gamma and in both selector traces, so the two
    figures compare identical scenario/step cells rather than curated examples.
    """
    case_keys = [(int(scenario_id), int(step)) for scenario_id, step in case_keys]
    if len(case_keys) != 3 or len(set(case_keys)) != 3:
        raise ValueError("selector comparison requires three distinct (scenario_id, step) case keys")
    if tuple(round(float(value), 6) for value in gammas) != (.1, .5, 1.0):
        raise ValueError("selector comparison rows must be gamma=(0.1,0.5,1.0)")
    indices = {
        "max one-step margin": _trace_index(margin_traces),
        "exact SafeMPPI proposal cost": _trace_index(cost_traces),
    }
    outputs = {
        "max one-step margin": os.path.abspath(margin_output),
        "exact SafeMPPI proposal cost": os.path.abspath(cost_output),
    }
    used = {}
    for selector, index in indices.items():
        figure, axes = plt.subplots(3, 3, figsize=(13.5, 12.5), constrained_layout=True)
        selector_used = []
        for row_index, gamma in enumerate(gammas):
            for column_index, (scenario_id, step) in enumerate(case_keys):
                key = (round(float(gamma), 6), scenario_id, step)
                if key not in index:
                    raise KeyError(f"{selector} trace is missing requested key {key}")
                trace = index[key]
                status = "NVP"
                if trace.get("executed_id") is not None:
                    status, query = _candidate_status(trace, trace["executed_id"])
                    if status != "positive":
                        raise ValueError(f"{selector} requested key {key} executed a non-positive query")
                    # Validate that every executed comparison cell has exactly ten bounded verifier levels.
                    verifier_level_polygons(trace, query, H=10)
                draw_query_frame(
                    axes[row_index, column_index], trace,
                    show_legend=(row_index == 0 and column_index == 0),
                    show_executed_levels=True,
                )
                axes[row_index, column_index].set_title(
                    f"gamma={gamma:g} | episode={scenario_id} | step={step}"
                )
                selector_used.append(dict(
                    gamma=float(gamma), scenario_id=scenario_id, step=step, status=status,
                    executed_id=(None if trace.get("executed_id") is None else int(trace["executed_id"])),
                ))
        figure.suptitle(f"OOD queried-plan snapshots: {selector}", fontsize=14)
        os.makedirs(os.path.dirname(outputs[selector]), exist_ok=True)
        figure.savefig(outputs[selector], dpi=180)
        plt.close(figure)
        used[selector] = selector_used
    return dict(outputs=outputs, case_keys=[list(value) for value in case_keys], used=used)


def render_query_animation(trace_path, output_mp4, output_panels, *, fps=5, max_frames=120):
    started = time.perf_counter()
    traces = torch.load(trace_path, map_location="cpu", weights_only=False)
    if not traces:
        raise ValueError("query trace is empty")
    render_zoom_panels(next(trace for trace in traces if len(trace["selected_ids"]) >= 4), output_panels)
    frames = traces[:int(max_frames)]
    figure, axis = plt.subplots(figsize=(7.5, 7.0))
    def update(index):
        axis.clear()
        draw_query_frame(axis, frames[index])
        return []
    movie = animation.FuncAnimation(figure, update, frames=len(frames), interval=1000 / fps, blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(output_mp4)), exist_ok=True)
    movie.save(output_mp4, writer=animation.FFMpegWriter(fps=fps, bitrate=2400), dpi=130)
    plt.close(figure)
    return dict(
        trace=os.path.abspath(trace_path), mp4=os.path.abspath(output_mp4),
        panels=os.path.abspath(output_panels), frames=len(frames),
        rendering_wall_seconds=time.perf_counter() - started,
    )


def render_raw_gallery(
        r0_checkpoint, selected_checkpoint, output_png, output_mp4,
        *, scene_profile, episode=None, device="cuda"):
    """Fixed raw rows: Hp10 r0, selected raw expansion, default Kazuki generate-refine."""
    import grid_policy_sfm as GPS
    import sfm_b1_eval as BE
    import sfm_kazuki as KZ
    import sfm_protocol as SP
    r0, _ = GPS.load_sfm_policy(r0_checkpoint, device=device)
    selected, _ = GPS.load_sfm_policy(selected_checkpoint, device=device)
    environment = SS.scene_profile(scene_profile)
    n_ped = int(environment["n_ped"])
    ped_speed_range = tuple(environment["ped_speed_range"])
    methods = ("Hp10 r0 raw", "selected raw", "default Kazuki generate-refine")
    rows = {name: [] for name in methods}
    config = KZ.KazukiConfig(safe_coefs=(0.3,), goal_coef=0.5).validate()
    episode = SP.CONFIRM_EP0 if episode is None else int(episode)
    for gamma in SP.GAMMAS:
        rows[methods[0]].append(BE.raw_rollout(
            r0, episode, gamma, device=device, n_ped=n_ped,
            ped_speed_range=ped_speed_range, collect_trace=True,
        ))
        rows[methods[1]].append(BE.raw_rollout(
            selected, episode, gamma, device=device, n_ped=n_ped,
            ped_speed_range=ped_speed_range, collect_trace=True,
        ))
        value = KZ.kazuki_sfm_deploy(
            r0, episode, gamma, cfg=config, T=SP.T, n_ped=n_ped, device=device,
            ped_speed_range=ped_speed_range, collect_diagnostics=False,
        )
        rows[methods[2]].append(value)
    figure, axes = plt.subplots(3, len(SP.GAMMAS), figsize=(19, 8.2), constrained_layout=True)
    colors = ("#666666", "#0072B2", "#D55E00")
    for row_index, method in enumerate(methods):
        for gamma_index, gamma in enumerate(SP.GAMMAS):
            axis = axes[row_index, gamma_index]
            value = rows[method][gamma_index]
            states = np.asarray(value["states"])
            peds = np.asarray(value["peds"])
            if len(peds):
                _draw_pedestrians(axis, peds[0], alpha=.55)
            axis.plot(states[:, 0], states[:, 1], color=colors[row_index], lw=1.5,
                      marker=".", ms=1.8)
            axis.plot(SS.GOAL[0], SS.GOAL[1], "*", color="#009E73", ms=8)
            status = "success" if value["success"] else "collision" if value["collision"] else "timeout"
            clearance = float(value.get("min_clearance", value.get("min_clear")))
            seconds = value.get("time_to_goal")
            if seconds is None and status == "success":
                seconds = float(value["steps"]) * SS.DT
            axis.text(
                .02, .02,
                f"{status} | clr={clearance:.2f} m" + (f" | t={float(seconds):.1f} s" if seconds is not None else ""),
                transform=axis.transAxes, fontsize=6.5, va="bottom",
                color=GREEN if status == "success" else RED,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.72, pad=1.2),
            )
            _set_world_frame(axis)
            if row_index == 0:
                axis.set_title(f"gamma={gamma}")
            if gamma_index == 0:
                axis.set_ylabel(method)
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    figure.suptitle(
        f"Fixed raw rollouts | {environment['scene_profile']} | "
        f"n_ped={n_ped}, speed={ped_speed_range[0]:g}-{ped_speed_range[1]:g} m/s"
    )
    figure.savefig(output_png, dpi=170)
    plt.close(figure)
    figure, axes = plt.subplots(3, len(SP.GAMMAS), figsize=(19, 8.2), constrained_layout=True)
    max_steps = max(len(np.asarray(value["states"])) for values in rows.values() for value in values)
    def update(frame):
        for row_index, method in enumerate(methods):
            for gamma_index, gamma in enumerate(SP.GAMMAS):
                axis = axes[row_index, gamma_index]; axis.clear()
                value = rows[method][gamma_index]
                states = np.asarray(value["states"]); peds = np.asarray(value["peds"])
                stop = min(frame + 1, len(states))
                axis.plot(states[:stop, 0], states[:stop, 1], color=colors[row_index], lw=1.6,
                          marker=".", ms=1.8)
                if len(peds):
                    ped_index = min(frame, len(peds) - 1)
                    _draw_pedestrians(axis, peds[ped_index])
                axis.plot(SS.GOAL[0], SS.GOAL[1], "*", color="#009E73", ms=8)
                _set_world_frame(axis)
                if row_index == 0: axis.set_title(f"gamma={gamma}")
                if gamma_index == 0: axis.set_ylabel(method)
        figure.suptitle(
            f"Fixed raw rollouts | {environment['scene_profile']} | "
            f"n_ped={n_ped}, speed={ped_speed_range[0]:g}-{ped_speed_range[1]:g} m/s"
        )
        return []
    movie = animation.FuncAnimation(figure, update, frames=max_steps, interval=100, blit=False)
    movie.save(output_mp4, writer=animation.FFMpegWriter(fps=10, bitrate=3000), dpi=95)
    plt.close(figure)
    return dict(
        png=os.path.abspath(output_png), mp4=os.path.abspath(output_mp4), methods=methods,
        environment=environment, episode=int(episode),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace")
    parser.add_argument("--mp4")
    parser.add_argument("--panels")
    parser.add_argument("--r0")
    parser.add_argument("--selected")
    parser.add_argument("--gallery")
    parser.add_argument("--gallery-episode", type=int)
    parser.add_argument("--scene-profile", choices=SS.SCIENTIFIC_EVAL_PROFILES)
    parser.add_argument("--margin-trace")
    parser.add_argument("--cost-trace")
    parser.add_argument("--case", action="append", help="explicit selector-comparison scenario_id:step")
    parser.add_argument("--margin-comparison")
    parser.add_argument("--cost-comparison")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    if args.margin_trace or args.cost_trace:
        if not (args.margin_trace and args.cost_trace and args.margin_comparison
                and args.cost_comparison and args.case and len(args.case) == 3):
            parser.error("selector comparison requires both traces, both outputs, and exactly three --case values")
        try:
            case_keys = [tuple(map(int, value.split(":"))) for value in args.case]
        except (TypeError, ValueError):
            parser.error("each --case must be scenario_id:step")
        report = render_selector_comparison(
            torch.load(args.margin_trace, map_location="cpu", weights_only=False),
            torch.load(args.cost_trace, map_location="cpu", weights_only=False),
            case_keys, args.margin_comparison, args.cost_comparison,
        )
    elif args.r0 or args.selected:
        if not (args.r0 and args.selected and args.gallery and args.mp4 and args.scene_profile):
            parser.error("raw gallery requires --r0, --selected, --gallery, --mp4, and --scene-profile")
        report = render_raw_gallery(
            args.r0, args.selected, args.gallery, args.mp4,
            scene_profile=args.scene_profile, episode=args.gallery_episode, device=args.device,
        )
    else:
        if not (args.trace and args.panels and args.mp4):
            parser.error("query animation requires --trace, --panels, and --mp4")
        report = render_query_animation(args.trace, args.mp4, args.panels)
    with open(args.report, "w") as stream:
        json.dump(report, stream, indent=2)


if __name__ == "__main__":
    main()
