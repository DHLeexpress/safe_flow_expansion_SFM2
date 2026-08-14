"""Render the diagnostic full-episode B1 label audit.

Blue/red trajectory segments are exact full-H verifier labels of the action
window that was actually executed.  A red NVP ring is a separate finite-B
context event.  Green H=10 levels are drawn only for an executed verifier
positive; post-NVP uncertified raw continuations never acquire green geometry.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
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
import sfm_b1_viz as BV
import sfm_scene as SS


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = os.fspath(path) + ".tmp"
    with open(temporary, "w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _index(traces):
    output = defaultdict(dict)
    for trace in traces:
        key = (int(trace["scenario_id"]), round(float(trace["gamma"]), 8))
        step = int(trace["step"])
        if step in output[key]:
            raise ValueError(f"duplicate trace key {key + (step,)}")
        output[key][step] = trace
    return output


def _executed_color(trace):
    if trace["executed_label"] == "verifier_positive":
        return BV.BLUE
    if trace["executed_label"] == "verifier_negative":
        return BV.RED
    return BV.GRAY


def _draw_history(axis, rows, step):
    for index in sorted(value for value in rows if value <= int(step)):
        trace = rows[index]
        before = np.asarray(trace["state"], float)[:2]
        after = np.asarray(trace["next_state"], float)[:2]
        color = _executed_color(trace)
        axis.plot(
            [before[0], after[0]], [before[1], after[1]],
            color=color, lw=2.2, marker=".", ms=2.0, alpha=.94, zorder=6,
        )


def _draw_candidates(axis, trace):
    selected = set(map(int, trace["selected_ids"]))
    for row in trace["all_K"]:
        path = np.asarray(row["segment"], float)
        axis.plot(
            path[:, 0], path[:, 1], color=BV.GRAY, lw=.38,
            marker=".", ms=1.0, alpha=.22, zorder=3,
        )
    for candidate_id in sorted(selected):
        row = BV._trace_candidate(trace, candidate_id)
        path = np.asarray(row["segment"], float)
        axis.plot(
            path[:, 0], path[:, 1], color=BV.ORANGE, lw=.72,
            marker=".", ms=1.3, alpha=.92, zorder=4,
        )
    for candidate_id in sorted(selected):
        status, query = BV._candidate_status(trace, candidate_id)
        if status not in ("positive", "negative"):
            continue
        path = np.asarray(BV._trace_candidate(trace, candidate_id)["segment"], float)
        color = BV.GREEN if status == "positive" else BV.RED
        axis.plot(
            path[:, 0], path[:, 1], color=color, lw=1.0,
            marker=".", ms=1.45, alpha=.96, zorder=5,
        )
        if status == "negative":
            axis.plot(
                path[-1, 0], path[-1, 1], "x", color=BV.RED,
                ms=3.5, mew=.9, zorder=8,
            )


def _draw_executed(axis, trace, *, draw_direction_arrow=True):
    result = trace["executed_result"]
    path = np.asarray(result.get(
        "segment",
        trace["executed_controls"],
    ), float)
    if path.shape != (11, 2):
        path = np.asarray([
            np.asarray(trace["state"], float)[:2],
            np.asarray(trace["next_state"], float)[:2],
        ])
    color = _executed_color(trace)
    axis.plot(path[:, 0], path[:, 1], color=color, lw=.75, alpha=.62, zorder=6)
    axis.plot(path[:2, 0], path[:2, 1], color=color, lw=3.0, zorder=9)
    if draw_direction_arrow:
        axis.annotate(
            "", xy=path[1], xytext=path[0],
            arrowprops=dict(arrowstyle="->", color=color, lw=2.3),
        )
    if trace["executed_label"] == "verifier_positive":
        query = dict(result=result)
        audit = DV.checked_verifier_levels(trace, query, H=10)
        DV._draw_verifier_geometry(axis, audit)


def draw_cell(axis, rows, step):
    available = [value for value in rows if value <= int(step)]
    current_step = max(available) if available else min(rows)
    trace = rows[current_step]
    BV._draw_common(axis, trace, nominal_levels=False)
    _draw_history(axis, rows, current_step)
    _draw_candidates(axis, trace)
    _draw_executed(axis, trace)
    position = np.asarray(trace["state"], float)[:2]
    if trace["nvp_context"]:
        axis.plot(
            position[0], position[1], marker="o", ms=10, mfc="none",
            mec=BV.RED, mew=1.6, zorder=12,
        )
    if trace["trap_entry"]:
        axis.plot(position[0], position[1], marker="s", ms=6,
                  mfc="none", mec=BV.RED, mew=1.2, zorder=12)
    if trace["collision_after_action"]:
        after = np.asarray(trace["next_state"], float)[:2]
        axis.plot(after[0], after[1], marker="x", ms=8,
                  color=BV.RED, mew=1.8, zorder=13)
    DV._set_clean_axis(axis)
    return trace


def _legend():
    return [
        Line2D([], [], color=BV.GRAY, lw=.7, label="K=16 generated"),
        Line2D([], [], color=BV.ORANGE, lw=1.1, label="B=4 RBF queried"),
        Line2D([], [], color=BV.GREEN, lw=1.4, label="B full-H positive"),
        Line2D([], [], color=BV.RED, lw=1.4, marker="x", label="B full-H rejected"),
        Line2D([], [], color=BV.BLUE, lw=2.7, label="executed window: full-H positive"),
        Line2D([], [], color=BV.RED, lw=2.7, label="executed window: full-H rejected"),
        Line2D([], [], color=BV.GREEN, lw=.7, label="executed verifier levels h=1..10"),
        Line2D([], [], marker="o", ms=8, mfc="none", mec=BV.RED, lw=0,
               label="finite-B NVP context"),
        Line2D([], [], marker="s", ms=6, mfc="none", mec=BV.RED, lw=0,
               label="first entry: 10-step progress < 0.2 m"),
    ]


def render(trace_path, output_mp4, output_png, output_json, *, fps=5, frame_stride=2):
    if int(fps) <= 0 or int(frame_stride) <= 0:
        raise ValueError("fps and frame_stride must be positive")
    bundle = torch.load(trace_path, map_location="cpu", weights_only=False)
    if bundle.get("status") != "SFM_B1_FULL_EPISODE_LABEL_AUDIT_COMPLETE":
        raise ValueError("input is not a completed full-episode audit")
    scenarios = tuple(map(int, bundle["scenarios"]))
    gammas = tuple(map(float, bundle["gammas"]))
    if len(scenarios) != 3 or gammas != tuple(map(float, SS.GAMMAS)):
        raise ValueError("renderer requires three scenarios and all seven gammas")
    index = _index(bundle["traces"])
    missing = [
        (scenario, gamma) for scenario in scenarios for gamma in gammas
        if (scenario, round(gamma, 8)) not in index
    ]
    if missing:
        raise ValueError(f"missing audit cells: {missing}")

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
        handles=_legend(), loc="center left", bbox_to_anchor=(.825, .58),
        frameon=False, fontsize=8,
    )
    figure.text(
        .825, .25,
        "Offline diagnostic only\n"
        "NVP does not stop this simulator trace.\n"
        "After NVP, a separately sampled raw\n"
        "temperature-1 first action advances it.\n"
        "Red continuation is not certified safety.",
        ha="left", va="top", fontsize=8,
    )

    final_cells = {}

    def update(step):
        final_cells.clear()
        for row, scenario in enumerate(scenarios):
            for column, gamma in enumerate(gammas):
                axis = axes[row, column]
                axis.clear()
                trace = draw_cell(
                    axis, index[(scenario, round(gamma, 8))], int(step)
                )
                final_cells[f"{scenario}:{gamma:g}"] = dict(
                    rendered_step=int(trace["step"]),
                    execution_source=trace["execution_source"],
                    executed_label=trace["executed_label"],
                    nvp_context=bool(trace["nvp_context"]),
                )
        return []

    for path in (output_mp4, output_png, output_json):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    movie = animation.FuncAnimation(
        figure, update, frames=frames, interval=1000 / int(fps), blit=False,
    )
    movie.save(
        output_mp4, writer=animation.FFMpegWriter(fps=int(fps), bitrate=4200),
        dpi=105,
    )
    update(maximum)
    figure.savefig(output_png, dpi=165, bbox_inches="tight")
    plt.close(figure)

    report = dict(
        status="SFM_B1_FULL_EPISODE_LABEL_VIZ_COMPLETE",
        diagnostic_only=True, trace_path=os.path.abspath(trace_path),
        trace_sha256=_sha256(trace_path), scenarios=list(scenarios),
        gammas=list(gammas), frame_stride=int(frame_stride), fps=int(fps),
        frames=frames, mp4=os.path.abspath(output_mp4),
        mp4_sha256=_sha256(output_mp4), png=os.path.abspath(output_png),
        png_sha256=_sha256(output_png),
        color_semantics=(
            "blue/red trail is exact full-H label of the actually executed "
            "window; NVP/trap/collision remain separate event markers"
        ),
        final_cells=dict(final_cells),
    )
    _write_json(output_json, report)
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
