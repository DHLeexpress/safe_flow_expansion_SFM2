"""Render the one planned H=10 D sample attached to every executed context.

Each thin blue/red branch is the exact H=10 window whose first action advanced
the offline collector at that context.  The thick black path joins those first
actions.  Current K/B queries and the current executed-window verifier geometry
remain visible so the branch origin can be audited.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

import _paths  # noqa: F401
import sfm_b1_density_viz as DV
import sfm_b1_full_episode_viz as FV
import sfm_b1_viz as BV
import sfm_metrics2 as SM
import sfm_scene as SS


def _branch(trace):
    result = trace["executed_result"]
    path = np.asarray(result.get("segment", ()), float)
    if path.shape != (11, 2):
        path = SM.rollout_positions(
            np.asarray(trace["state"], float),
            np.asarray(trace["executed_controls"], float),
        )
    return path


def _draw_D_branches(axis, rows, step, *, line_scale=1.0):
    available = sorted(value for value in rows if value <= int(step))
    for context_step in available:
        trace = rows[context_step]
        path = _branch(trace)
        color = FV._executed_color(trace)
        is_current = context_step == available[-1]
        axis.plot(
            path[:, 0], path[:, 1],
            color=color,
            lw=(1.25 if is_current else .55) * float(line_scale),
            marker=".",
            ms=(1.3 if is_current else .75) * np.sqrt(float(line_scale)),
            alpha=.9 if is_current else .32,
            zorder=7 if is_current else 3,
        )
        axis.plot(
            path[0, 0], path[0, 1], marker=".", color=color,
            ms=(2.7 if is_current else 1.5) * np.sqrt(float(line_scale)),
            zorder=8,
        )


def _draw_executed_trajectory(
        axis, rows, step, *, linewidth=2.8, marker_size=2.1,
        draw_direction_arrow=True,
):
    available = sorted(value for value in rows if value <= int(step))
    if not available:
        return
    states = [np.asarray(rows[value]["state"], float)[:2] for value in available]
    states.append(np.asarray(rows[available[-1]]["next_state"], float)[:2])
    states = np.asarray(states)
    axis.plot(
        states[:, 0], states[:, 1], color="#111111", lw=float(linewidth),
        marker=".", ms=float(marker_size), alpha=.97, zorder=11,
    )
    if draw_direction_arrow:
        axis.annotate(
            "", xy=states[-1], xytext=states[-2],
            arrowprops=dict(
                arrowstyle="->", color="#111111",
                lw=max(.8, .78 * float(linewidth)),
            ),
            zorder=12,
        )


def _robot_frame(path, trace):
    position = np.asarray(trace["state"], float)[:2]
    direction = np.asarray(trace["state"], float)[2:4]
    if np.linalg.norm(direction) < 1.0e-6:
        direction = np.asarray(SS.GOAL, float)[:2] - position
    direction = direction / max(np.linalg.norm(direction), 1.0e-12)
    normal = np.array([-direction[1], direction[0]])
    delta = np.asarray(path, float) - position
    return np.stack([delta @ direction, delta @ normal], axis=1)


def _draw_candidate_inset(axis, trace):
    for child in list(axis.child_axes):
        if getattr(child, "_sfm_candidate_inset", False):
            child.remove()
    inset = axis.inset_axes((.035, .675, .30, .29), zorder=30)
    inset._sfm_candidate_inset = True
    inset.set_facecolor((1., 1., 1., .68))
    selected_id = trace.get("executed_id")
    local_paths = []
    for query_index, row in enumerate(trace["query_rows"], start=1):
        path = np.asarray(
            BV._trace_candidate(trace, int(row["candidate_id"]))["segment"],
            float,
        )
        local = _robot_frame(path, trace)
        local_paths.append(local)
        status, _ = BV._candidate_status(trace, int(row["candidate_id"]))
        color = (
            BV.GREEN if status == "positive"
            else BV.RED if status == "negative"
            else BV.GRAY
        )
        is_selected = (
            selected_id is not None
            and int(row["candidate_id"]) == int(selected_id)
        )
        if is_selected:
            inset.plot(
                local[:, 0], local[:, 1], color="#111111",
                lw=3.3, alpha=.82, zorder=3,
            )
        inset.plot(
            local[:, 0], local[:, 1], color=color,
            lw=2.25 if is_selected else 1.05,
            alpha=.98 if is_selected else .72,
            zorder=4 if is_selected else 2,
        )
        inset.text(
            local[-1, 0], local[-1, 1], str(query_index),
            fontsize=4.8, color=color, ha="center", va="center", zorder=5,
        )
    inset.plot(0., 0., marker=">", color="#111111", ms=3.2, zorder=6)
    if local_paths:
        joined = np.concatenate(local_paths)
        span = max(.18, 1.08 * float(np.max(np.abs(joined))))
        inset.set_xlim(-.08 * span, span)
        inset.set_ylim(-span, span)
    inset.set_aspect("equal")
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title(
        "robot-frame B=4" if selected_id is not None else "robot-frame B=4 · NVP",
        fontsize=5.1, pad=1.2,
    )
    for spine in inset.spines.values():
        spine.set_alpha(.34)
        spine.set_linewidth(.55)


def draw_cell(
        axis, rows, step, *, branch_line_scale=1.0,
        trajectory_linewidth=2.8, trajectory_marker_size=2.1,
        candidate_inset=False, draw_direction_arrow=True,
):
    available = [value for value in rows if value <= int(step)]
    current_step = max(available) if available else min(rows)
    trace = rows[current_step]
    BV._draw_common(axis, trace, nominal_levels=False)
    _draw_D_branches(
        axis, rows, current_step, line_scale=branch_line_scale,
    )
    FV._draw_candidates(axis, trace)
    FV._draw_executed(
        axis, trace, draw_direction_arrow=draw_direction_arrow,
    )
    _draw_executed_trajectory(
        axis, rows, current_step, linewidth=trajectory_linewidth,
        marker_size=trajectory_marker_size,
        draw_direction_arrow=draw_direction_arrow,
    )
    if candidate_inset:
        _draw_candidate_inset(axis, trace)
    DV._set_clean_axis(axis)
    return trace


def _legend():
    return [
        Line2D([], [], color=BV.BLUE, lw=1.2, label=r"$D^+$ planned H10 branch"),
        Line2D([], [], color=BV.RED, lw=1.2, label=r"$D^-$ planned H10 branch"),
        Line2D([], [], color="#111111", lw=2.8, label="executed first-action trajectory"),
        Line2D([], [], color=BV.GRAY, lw=.7, label="current K=16 generated"),
        Line2D([], [], color=BV.ORANGE, lw=1.1, label="current B=4 RBF queried"),
        Line2D([], [], color=BV.GREEN, lw=1.2, label="current B full-H positive"),
        Line2D([], [], color=BV.RED, lw=1.2, marker="x", label="current B full-H rejected"),
        Line2D([], [], color=BV.GREEN, lw=.7, label="executed verifier levels h=1..10"),
    ]


def render(trace_path, output_mp4, output_png, output_json, *, fps=5, frame_stride=2):
    bundle = torch.load(trace_path, map_location="cpu", weights_only=False)
    if bundle.get("status") != "SFM_B1_FULL_EPISODE_LABEL_AUDIT_COMPLETE":
        raise ValueError("input is not a completed full-episode audit")
    scenarios = tuple(map(int, bundle["scenarios"]))
    gammas = tuple(map(float, bundle["gammas"]))
    if len(scenarios) != 3 or gammas != tuple(map(float, SS.GAMMAS)):
        raise ValueError("renderer requires three scenarios and all seven gammas")
    index = FV._index(bundle["traces"])
    maximum = max(max(rows) for rows in index.values())
    frames = list(range(0, maximum + 1, int(frame_stride)))
    if frames[-1] != maximum:
        frames.append(maximum)

    figure, axes = plt.subplots(3, 7, figsize=(23.2, 10.1))
    figure.subplots_adjust(
        left=.035, right=.815, bottom=.025, top=.94, wspace=.025, hspace=.04,
    )
    for column, gamma in enumerate(gammas):
        figure.text(
            .035 + (.78 / 7) * (column + .5), .965, f"$\\gamma={gamma:g}$",
            ha="center", va="center", fontsize=10,
        )
    for row, scenario in enumerate(scenarios):
        figure.text(
            .012, .94 - (.915 / 3) * (row + .5), f"episode\n{scenario}",
            ha="center", va="center", rotation=90, fontsize=9,
        )
    figure.legend(
        handles=_legend(), loc="center left", bbox_to_anchor=(.825, .59),
        frameon=False, fontsize=8,
    )
    figure.text(
        .825, .26,
        "One D sample per context.\n"
        "Each branch is the complete planned H10 window;\n"
        "only its first action advances the black trajectory.\n"
        "Finite-B NVP does not stop this offline collector.",
        ha="left", va="top", fontsize=8,
    )

    def update(step):
        for row, scenario in enumerate(scenarios):
            for column, gamma in enumerate(gammas):
                axis = axes[row, column]
                axis.clear()
                draw_cell(axis, index[(scenario, round(gamma, 8))], int(step))
        return []

    for path in (output_mp4, output_png, output_json):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    movie = animation.FuncAnimation(
        figure, update, frames=frames, interval=1000 / int(fps), blit=False,
    )
    movie.save(
        output_mp4, writer=animation.FFMpegWriter(fps=int(fps), bitrate=4600),
        dpi=105,
    )
    update(maximum)
    figure.savefig(output_png, dpi=165, bbox_inches="tight")
    plt.close(figure)

    report = dict(
        status="SFM_B1_D_BRANCH_VIZ_COMPLETE",
        trace_path=os.path.abspath(trace_path),
        scenarios=list(scenarios),
        gammas=list(gammas),
        D_semantics=(
            "one exact full-H10 planned window per executed context; blue y=1, "
            "red y=0; thick black joins executed first actions"
        ),
        frames=frames,
        mp4=os.path.abspath(output_mp4),
        png=os.path.abspath(output_png),
    )
    with open(output_json + ".tmp", "w") as stream:
        json.dump(report, stream, indent=2)
    os.replace(output_json + ".tmp", output_json)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output-mp4", required=True)
    parser.add_argument("--output-png", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--frame-stride", type=int, default=2)
    args = parser.parse_args(argv)
    render(
        args.trace, args.output_mp4, args.output_png, args.output_json,
        fps=args.fps, frame_stride=args.frame_stride,
    )


if __name__ == "__main__":
    main()
