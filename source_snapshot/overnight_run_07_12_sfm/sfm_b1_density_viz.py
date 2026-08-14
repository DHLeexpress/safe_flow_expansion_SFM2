"""Reusable, audit-first visualizations for the Hp10 density-OOD study.

This module consumes already generated rollout/query traces.  Rendering never
changes the replay store, the GP, an expansion checkpoint, or an execution
decision.  Every green verifier set is tied to exactly one full-H positive
window; the renderer rejects legacy partial-query results instead of silently
presenting them as part of the current protocol.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np
import torch

import _paths  # noqa: F401
from polar_grid import polytope_HP
import sfm_b1_viz as BV
import sfm_b1_viz_socp as VS
import sfm_scene as SS


DISPLAY_GAMMAS = (0.1, 0.5, 1.0)
METHOD_KEYS = ("expert", "selected", "kazuki")
METHOD_LABELS = {
    "expert": "SafeMPPI demonstration expert",
    "selected": "Arm-A r10 learned raw",
    "kazuki": "Kazuki generate-guide-refine",
}
MAGENTA = "#CC79A7"
CYAN = "#00A6D6"
METHOD_ALIASES = {
    "expert": ("expert", "safemppi_expert"),
    "selected": ("selected", "arm_a_r10_raw"),
    "kazuki": ("kazuki", "default_kazuki"),
}


def _write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump(payload, stream, indent=2)
    os.replace(temporary, path)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _query_handles(*, include_nominal=False, include_guidance=False):
    handles = [
        Line2D([], [], color=BV.GRAY, lw=.8, label="K generated"),
        Line2D([], [], color=BV.ORANGE, lw=1.5, marker=".", label="B queried"),
        Line2D([], [], color=BV.GREEN, lw=2.0, marker=".", label="full-H verifier positive"),
        Line2D([], [], color=BV.RED, lw=2.0, marker="x", label="rejected"),
        Line2D([], [], color=BV.BLUE, lw=3.2, label="executed first action"),
        Line2D([], [], color=BV.GREEN, lw=.55, label="exact SOCP h=1..10"),
        Line2D([], [], color=BV.GREEN, lw=1.25, label="exact SOCP outer set (K=16)"),
    ]
    if include_nominal:
        handles.extend([
            Line2D([], [], color=BV.BLUE, lw=.5, label="nominal h=1..10"),
            Line2D([], [], color=BV.BLUE, lw=1.25, label="nominal outer set (16 faces)"),
        ])
    if include_guidance:
        handles.append(Line2D([], [], color=MAGENTA, lw=2.5, label="Kazuki net guidance"))
    return handles


def _set_clean_axis(axis):
    BV._set_world_frame(axis)
    axis.set_title("")
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.tick_params(axis="both", which="both", labelbottom=False, labelleft=False, length=0)


def _query_result(trace, candidate_id):
    """Classify a B query under the mandatory full-H runtime contract."""
    status, row = BV._candidate_status(trace, candidate_id)
    if row is not None and row["result"].get("resolved"):
        result = row["result"]
        if not bool(result.get("full_h")) or int(result.get("terminal_step", -1)) != 10:
            raise ValueError("visualization received a legacy partial B1 query; reverify it over H=10")
    if status != "positive":
        return status, row
    return "positive_full_h", row


def _disk_is_separated(A, b, center, radius=SS.R_PED, tol=2.0e-6):
    """Whether at least one half-space excludes the complete pedestrian disk."""
    value = np.asarray(A, float) @ np.asarray(center, float) - np.asarray(b, float)
    return bool(np.any(value - float(radius) >= -float(tol)))


def checked_verifier_levels(trace, query_row, *, H=10):
    """Return H candidate-specific levels and a current-disk geometry audit.

    The compact verifier is local to the candidate trajectory.  If a current
    pedestrian omitted by the local face construction visibly overlaps the
    resulting set, the set is withheld instead of being painted as globally
    safe.
    """
    polygons = BV.verifier_level_polygons(trace, query_row, H=H)
    result = query_row["result"]
    faces = [face for face in result["faces"] if bool(face.feasible)]
    diagnostics = result.get("diagnostics", {})
    if diagnostics.get("solver") != "paper_static_exact_2d_angular_interval_socp":
        raise ValueError("green verifier geometry requires the exact 2-D SOCP solver")
    if int(diagnostics.get("K_artificial", -1)) != VS.ARTIFICIAL_FACES:
        raise ValueError("green verifier geometry requires exactly 16 artificial outer faces")
    if sum(face.kind == "artificial" for face in faces) != VS.ARTIFICIAL_FACES:
        raise ValueError("resolved verifier face list does not contain 16 feasible artificial faces")
    A = np.stack([np.asarray(face.a, float) for face in faces])
    margins = np.asarray([float(face.m) for face in faces])
    center = np.asarray(result["segment"], float)[0]
    b = A @ center + margins
    outer = BV.halfspace_polygon(A, b)
    if outer is None:
        raise ValueError("faithful verifier faces do not form a bounded outer polytope")
    overlaps = [
        index for index, pedestrian in enumerate(np.asarray(trace["ped_xy"], float))
        if not _disk_is_separated(A, b, pedestrian)
    ]
    return dict(
        polygons=([] if overlaps else polygons), requested_levels=int(H),
        rendered_levels=(0 if overlaps else len(polygons)),
        outer_polygon=(None if overlaps else outer),
        current_disk_overlap_indices=overlaps,
        locally_clear=not overlaps,
        artificial_faces=sum(face.kind == "artificial" for face in faces),
        solver=result.get("diagnostics", {}).get("solver"),
    )


def nominal_safemppi_levels(trace, *, gamma=None, H=10):
    """The trace-owning nominal SafeMPPI geometry and 16-sided outer set."""
    state = np.asarray(trace["state"], np.float32)
    stored = trace.get("nominal_polytope")
    if stored is None:
        ped_xy = np.asarray(trace["ped_xy"], np.float32)
        obstacles = np.concatenate([
            ped_xy, np.full((len(ped_xy), 1), SS.R_PED, np.float32),
        ], axis=1)
        _, (A, b, margins) = polytope_HP(
            state[:2], obstacles, sensing=SS.R_SENSE, n_base=16,
        )
        velocity_used = False
    else:
        if int(stored.get("n_base", -1)) != 16:
            raise ValueError("stored SafeMPPI nominal polytope is not K=16")
        if stored.get("velocity_used") is not True:
            raise ValueError("stored SafeMPPI nominal polytope did not use pedestrian velocity")
        A, b, margins = stored["A"], stored["b"], stored["margins"]
        velocity_used = True
    A = np.asarray(A, float); b = np.asarray(b, float)
    margins = np.asarray(margins, float)
    outer = BV.halfspace_polygon(A, b)
    if outer is None:
        raise ValueError("nominal K=16 faces do not form a bounded outer polytope")
    value = float(trace.get("gamma", gamma) if gamma is None else gamma)
    return dict(
        polygons=BV._level_polygons(A, margins, state[:2], value, H=H),
        outer_polygon=outer, base_faces=16,
        detected_faces=max(0, int(len(A)) - 16),
        contains_robot=bool(np.all(A @ state[:2] <= b + 1.0e-7)),
        velocity_used=velocity_used,
    )


def _draw_outer_polygon(axis, polygon, *, color, linewidth=1.25, alpha=.95, zorder=2.7):
    axis.plot(
        np.r_[polygon[:, 0], polygon[0, 0]],
        np.r_[polygon[:, 1], polygon[0, 1]],
        color=color, lw=linewidth, alpha=alpha, zorder=zorder,
    )


def _draw_nominal_geometry(axis, trace):
    nominal = nominal_safemppi_levels(trace, H=10)
    BV._draw_level_polygons(
        axis, nominal["polygons"], color=BV.BLUE,
        linewidth=.48, alpha=.62, zorder=2,
    )
    _draw_outer_polygon(axis, nominal["outer_polygon"], color=BV.BLUE)
    return nominal


def _draw_verifier_geometry(axis, audit):
    if not audit or not audit["polygons"]:
        return
    BV._draw_level_polygons(
        axis, audit["polygons"], color=BV.GREEN,
        linewidth=.52, alpha=.68, zorder=2.5,
    )
    _draw_outer_polygon(axis, audit["outer_polygon"], color=BV.GREEN)


def draw_margin_query_frame(axis, trace, *, executed_levels=True, nominal_levels=False):
    """Draw one exact-K16 max-margin acquisition context."""
    BV._draw_common(axis, trace, nominal_levels=False)
    nominal = _draw_nominal_geometry(axis, trace) if nominal_levels else None
    selected = list(map(int, trace["selected_ids"]))
    for row in trace["all_K"]:
        path = np.asarray(row["segment"])
        axis.plot(path[:, 0], path[:, 1], color=BV.GRAY, lw=.48,
                  marker=".", ms=1.5, alpha=.27, zorder=3)
    for candidate_id in selected:
        path = np.asarray(BV._trace_candidate(trace, candidate_id)["segment"])
        axis.plot(path[:, 0], path[:, 1], color=BV.ORANGE, lw=.72,
                  marker="o", ms=1.9, alpha=.9, zorder=4)
    rejected = []
    positives = []
    for candidate_id in selected:
        status, query = _query_result(trace, candidate_id)
        if status not in ("positive_full_h", "negative"):
            continue
        path = np.asarray(BV._trace_candidate(trace, candidate_id)["segment"])
        color = BV.GREEN if status == "positive_full_h" else BV.RED
        axis.plot(path[:, 0], path[:, 1], color=color, lw=.92,
                  marker="o", ms=2.1, alpha=.96, zorder=5)
        if status == "positive_full_h":
            positives.append(candidate_id)
        else:
            axis.plot(path[-1, 0], path[-1, 1], "x", color=BV.RED,
                      ms=6, mew=1.4, zorder=8)
            rejected.append(candidate_id)
    level_audit = None
    executed = trace.get("executed_id")
    if executed is not None:
        path = np.asarray(BV._trace_candidate(trace, executed)["segment"])
        axis.plot(path[:2, 0], path[:2, 1], color=BV.BLUE, lw=2.6, zorder=9)
        axis.annotate("", xy=path[1], xytext=path[0],
                      arrowprops=dict(arrowstyle="->", color=BV.BLUE, lw=2.8))
        status, query = _query_result(trace, executed)
        if status != "positive_full_h":
            raise ValueError("the executed candidate must be a resolved full-H positive query")
        if executed_levels:
            level_audit = checked_verifier_levels(trace, query, H=10)
            _draw_verifier_geometry(axis, level_audit)
    _set_clean_axis(axis)
    return dict(
        nominal_drawn=bool(nominal is not None), selected_ids=selected, positive_ids=positives,
        nominal=(None if nominal is None else {
            key: value for key, value in nominal.items()
            if key not in ("polygons", "outer_polygon")
        }),
        rejected_ids=rejected,
        executed_id=(None if executed is None else int(executed)),
        executed_verifier=(None if level_audit is None else {
            key: value for key, value in level_audit.items()
            if key not in ("polygons", "outer_polygon")
        }),
    )


def _trace_index(traces, scenario_id, gammas):
    output = defaultdict(dict)
    for trace in traces:
        if int(trace["scenario_id"]) != int(scenario_id):
            continue
        gamma = round(float(trace["gamma"]), 6)
        if gamma in tuple(round(float(value), 6) for value in gammas):
            output[gamma][int(trace["step"])] = trace
    missing = [gamma for gamma in gammas if not output[round(float(gamma), 6)]]
    if missing:
        raise ValueError(f"scenario {scenario_id} has no trace for gammas {missing}")
    return output


def render_margin_gathering_video(
        traces, scenario_id, output_mp4, *, output_snapshot=None,
        snapshot_step=None, gammas=DISPLAY_GAMMAS, fps=6, frame_stride=1):
    """Render one explicit scenario as three gamma columns.

    Columns freeze at their last available context after independent NVP or
    termination.  The scenario and optional snapshot step are caller-supplied;
    this function performs no outcome-based curation.
    """
    gammas = tuple(map(float, gammas))
    if gammas != DISPLAY_GAMMAS:
        raise ValueError(f"the density-OOD gathering view requires gammas={DISPLAY_GAMMAS}")
    index = _trace_index(traces, scenario_id, gammas)
    maximum = max(max(rows) for rows in index.values())
    frame_steps = list(range(0, maximum + 1, int(frame_stride)))
    if frame_steps[-1] != maximum:
        frame_steps.append(maximum)
    figure, axes = plt.subplots(1, 3, figsize=(13.8, 4.6))
    figure.subplots_adjust(left=.02, right=.78, bottom=.03, top=.97, wspace=.025)
    figure.legend(handles=_query_handles(), loc="center left", bbox_to_anchor=(.79, .55),
                  fontsize=8, frameon=False)
    figure.text(.79, .30, f"scenario {int(scenario_id)}\ncolumns: gamma 0.1 | 0.5 | 1.0\n"
                "green set: executed exact-SOCP candidate only\nK=16 artificial outer faces",
                ha="left", va="top", fontsize=8)

    def trace_at(gamma, step):
        rows = index[round(float(gamma), 6)]
        available = [value for value in rows if value <= int(step)]
        return rows[max(available) if available else min(rows)]

    rendered = {}
    def update(step):
        rendered.clear()
        for column, gamma in enumerate(gammas):
            axes[column].clear()
            trace = trace_at(gamma, step)
            rendered[str(gamma)] = draw_margin_query_frame(axes[column], trace)
        return []

    movie = animation.FuncAnimation(
        figure, update, frames=frame_steps, interval=1000 / int(fps), blit=False,
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_mp4)), exist_ok=True)
    movie.save(output_mp4, writer=animation.FFMpegWriter(fps=fps, bitrate=2800), dpi=120)
    snapshot_metadata = None
    if output_snapshot is not None:
        if snapshot_step is None:
            raise ValueError("output_snapshot requires an explicit snapshot_step")
        rendered.clear()
        for column, gamma in enumerate(gammas):
            axes[column].clear()
            trace = trace_at(gamma, int(snapshot_step))
            rendered[str(gamma)] = draw_margin_query_frame(
                axes[column], trace, nominal_levels=True,
            )
        figure.legends.clear()
        figure.legend(handles=_query_handles(include_nominal=True), loc="center left",
                      bbox_to_anchor=(.79, .55), fontsize=8, frameon=False)
        os.makedirs(os.path.dirname(os.path.abspath(output_snapshot)), exist_ok=True)
        figure.savefig(output_snapshot, dpi=180, bbox_inches="tight")
        snapshot_metadata = dict(step=int(snapshot_step), cells=dict(rendered))
    plt.close(figure)
    return dict(
        semantics=("diagnostic-only max-margin gather rerun with exact K16 SOCP; animation has no "
                   "nominal overlay; static snapshot adds position-only nominal h=1..10 and outer set; "
                   "only an executed full-H positive may own green levels"),
        scenario_id=int(scenario_id), gammas=list(gammas), frame_steps=frame_steps,
        mp4=os.path.abspath(output_mp4),
        snapshot=(None if output_snapshot is None else os.path.abspath(output_snapshot)),
        snapshot_metadata=snapshot_metadata,
    )


def _positive_control_spread(trace):
    controls = []
    endpoints = []
    for candidate_id in trace["selected_ids"]:
        status, query = _query_result(trace, candidate_id)
        if status == "positive_full_h":
            controls.append(np.asarray(query["controls"], float).reshape(-1))
            endpoints.append(np.asarray(query["result"]["segment"], float)[-1])
    if len(controls) < 2:
        return 0.0, 0.0
    horizon = int(np.asarray(controls[0]).size // 2)
    denominator = 2.0 * float(SS.U_MAX) * math.sqrt(2.0 * horizon)
    control = [np.linalg.norm(controls[i] - controls[j]) / denominator
               for i in range(len(controls)) for j in range(i + 1, len(controls))]
    endpoint = [np.linalg.norm(endpoints[i] - endpoints[j])
                for i in range(len(endpoints)) for j in range(i + 1, len(endpoints))]
    return float(np.clip(max(control), 0.0, 1.0)), float(max(endpoint))


def _snapshot_row(trace):
    statuses = [_query_result(trace, candidate_id)[0] for candidate_id in trace["selected_ids"]]
    positive = sum(status == "positive_full_h" for status in statuses)
    negative = sum(status == "negative" for status in statuses)
    control_spread, endpoint_spread = _positive_control_spread(trace)
    return dict(
        scenario_id=int(trace["scenario_id"]), gamma=float(trace["gamma"]),
        step=int(trace["step"]), positive_full_h=positive,
        rejected=negative,
        control_spread=control_spread, endpoint_spread=endpoint_spread,
        control_spread_normalization="||Ui-Uj||/(2*u_max*sqrt(2H)), clipped to [0,1]",
        eligible=positive >= 2 and negative >= 1,
    )


def rank_query_snapshots(traces):
    """Exhaustively rank mixed B contexts by few rejections, then control spread."""
    rows = [_snapshot_row(trace) for trace in traces]
    eligible_rows = [row for row in rows if row["eligible"]]
    eligible_rows.sort(key=lambda row: (
        row["rejected"], -row["control_spread"], -row["endpoint_spread"],
        row["scenario_id"], row["gamma"], row["step"],
    ))
    if not eligible_rows:
        raise RuntimeError("no context has at least two full-H positives and at least one rejected B query")
    chosen = eligible_rows[0]
    return dict(
        rule=("exhaustive over supplied traces: require >=2 full-H positives and >=1 rejection; "
              "minimize rejection count, maximize normalized pairwise control spread, then endpoint spread; "
              "tie-break by scenario, gamma, step"),
        chosen=chosen, candidates=rows,
    )


def _find_trace(traces, key):
    matched = [trace for trace in traces if (
        int(trace["scenario_id"]) == int(key["scenario_id"])
        and abs(float(trace["gamma"]) - float(key["gamma"])) <= 1.0e-8
        and int(trace["step"]) == int(key["step"])
    )]
    if len(matched) != 1:
        raise ValueError(f"query snapshot key matched {len(matched)} traces")
    return matched[0]


def render_candidate_query_snapshot(traces, output_png, *, selection=None, report_path=None):
    """Render one candidate-specific panel per queried B plan."""
    selection = rank_query_snapshots(traces) if selection is None else selection
    trace = _find_trace(traces, selection["chosen"])
    candidate_ids = list(map(int, trace["selected_ids"]))
    columns = 2
    rows = int(math.ceil(len(candidate_ids) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(9.6, 4.5 * rows), squeeze=False)
    figure.subplots_adjust(left=.02, right=.76, bottom=.03, top=.98, wspace=.03, hspace=.03)
    candidate_metadata = []
    for axis, candidate_id in zip(axes.flat, candidate_ids):
        BV._draw_common(axis, trace, nominal_levels=False)
        nominal = _draw_nominal_geometry(axis, trace)
        status, query = _query_result(trace, candidate_id)
        path = np.asarray(BV._trace_candidate(trace, candidate_id)["segment"])
        color = BV.GREEN if status == "positive_full_h" else BV.RED if status == "negative" else BV.ORANGE
        axis.plot(path[:, 0], path[:, 1], color=color, lw=.92,
                  marker="o", ms=2.3, zorder=7)
        level_audit = None
        rejected_x = None
        if status == "positive_full_h":
            level_audit = checked_verifier_levels(trace, query, H=10)
            _draw_verifier_geometry(axis, level_audit)
        elif status == "negative":
            rejected_x = path[-1].astype(float).tolist()
            axis.plot(*path[-1], "x", color=BV.RED, ms=7, mew=1.6, zorder=9)
        if trace.get("executed_id") is not None and int(trace["executed_id"]) == candidate_id:
            axis.plot(path[:2, 0], path[:2, 1], color=BV.BLUE, lw=3.2, zorder=10)
        _set_clean_axis(axis)
        candidate_metadata.append(dict(
            candidate_id=candidate_id, status=status,
            nominal_outer_faces=nominal["base_faces"],
            full_h=bool(query and query["result"].get("full_h", False)),
            verifier=(None if level_audit is None else {
                key: value for key, value in level_audit.items()
                if key not in ("polygons", "outer_polygon")
            }),
            rejected_x=rejected_x,
            executed=bool(trace.get("executed_id") is not None
                          and int(trace["executed_id"]) == candidate_id),
        ))
    for axis in axes.flat[len(candidate_ids):]:
        axis.set_visible(False)
    figure.legend(handles=_query_handles(include_nominal=True), loc="center left", bbox_to_anchor=(.77, .58),
                  fontsize=8, frameon=False)
    chosen = selection["chosen"]
    figure.text(
        .77, .36,
        f"scenario {chosen['scenario_id']}\ngamma {chosen['gamma']:g}\nstep {chosen['step']}\n"
        f"full-H positive {chosen['positive_full_h']}\nrejected {chosen['rejected']}\n"
        f"control spread {chosen['control_spread']:.3f}",
        ha="left", va="top", fontsize=8,
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    figure.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(figure)
    report = dict(
        semantics=("each panel is one B query rerun with exact moving-disk SOCP and K=16 artificial "
                   "outer faces; green levels are candidate-specific and full-H only; blue is the "
                   "position-only nominal polytope"),
        output=os.path.abspath(output_png), selection=selection,
        candidates=candidate_metadata,
    )
    if report_path is not None:
        _write_json(report_path, report)
    return report


def _run_trace(run, step):
    traces = list(run.get("trace") or [])
    if not traces:
        raise ValueError("mechanism comparison requires collect_trace/collect_diagnostics")
    return traces[min(max(int(step), 0), len(traces) - 1)]


def _draw_robot_and_pedestrians(axis, run, trace, step, color):
    states = np.asarray(run["states"])
    state_index = min(max(int(step), 0), len(states) - 1)
    BV._draw_pedestrians(axis, trace["ped_xy"], alpha=.62)
    for pedestrian, velocity in zip(np.asarray(trace["ped_xy"]), np.asarray(trace["ped_vel"])):
        future = pedestrian[None] + np.arange(11)[:, None] * SS.DT * velocity[None]
        axis.plot(future[:, 0], future[:, 1], ".--", color=BV.GRAY, ms=1.8, lw=.55, alpha=.5)
    axis.plot(states[:state_index + 1, 0], states[:state_index + 1, 1],
              color=color, lw=1.25, marker=".", ms=1.6)
    axis.plot(states[state_index, 0], states[state_index, 1], "o", color=color, ms=6, zorder=9)
    axis.plot(SS.GOAL[0], SS.GOAL[1], "*", color="#F0E442", mec="#333333", ms=8)


def draw_method_panel(axis, method, run, gamma, step, *, verifier_result=None,
                      guidance_scale=3.0, guidance_cap=1.8):
    """Draw one cell of the method-by-gamma comparison without titles/labels."""
    if method not in METHOD_KEYS:
        raise ValueError(f"unknown method {method!r}")
    trace = _run_trace(run, step)
    colors = {"expert": BV.BLUE, "selected": "#333333", "kazuki": "#7F3C8D"}
    _draw_robot_and_pedestrians(axis, run, trace, step, colors[method])
    metadata = dict(method=method, gamma=float(gamma), step=int(trace["step"]))
    if method == "expert":
        controls = np.asarray(trace["controls"], float)
        if controls.shape != (10, 2):
            raise ValueError("SafeMPPI expert trace must carry one H=10 reward-weighted sequence")
        if trace.get("sequence_kind") != "reward_weighted_mean":
            raise ValueError("SafeMPPI expert trace does not identify the executed MPPI mean sequence")
        # ``action`` and ``mean_sequence[0]`` are formed by two mathematically
        # equivalent float32 reductions.  They may differ by one float32 ULP
        # even though they are the same reward-weighted MPPI control.
        action_atol = 8.0 * np.finfo(np.float32).eps
        if not np.allclose(np.asarray(trace["action"], float), controls[0],
                           atol=action_atol, rtol=0.0):
            raise ValueError("SafeMPPI plotted sequence does not begin with the executed action")
        plan = np.asarray(trace["planned_states"], float)[:, :2]
        axis.plot(plan[:, 0], plan[:, 1], color=BV.BLUE, lw=.82, marker="o", ms=2.2)
        nominal = nominal_safemppi_levels(trace, gamma=gamma, H=10)
        BV._draw_level_polygons(axis, nominal["polygons"], color=BV.BLUE,
                                linewidth=.48, alpha=.62, zorder=2)
        _draw_outer_polygon(axis, nominal["outer_polygon"], color=BV.BLUE)
        metadata.update(nominal_levels=len(nominal["polygons"]),
                        nominal_contains_robot=nominal["contains_robot"],
                        nominal_outer_faces=nominal["base_faces"],
                        nominal_detected_faces=nominal["detected_faces"],
                        nominal_velocity_used=nominal["velocity_used"],
                        expert_sequence_kind=trace["sequence_kind"])
    elif method == "selected":
        plan = np.asarray(trace["planned_states"], float)[:, :2]
        result = verifier_result if verifier_result is not None else VS.verify_query(
            trace["state"], trace["controls"], trace["ped_xy"], trace["ped_vel"], gamma,
        )
        resolved = bool(result.get("resolved"))
        if resolved and (
                not bool(result.get("full_h"))
                or int(result.get("terminal_step", -1)) != 10):
            raise ValueError("learned-plan visualization requires a full H=10 verifier result")
        positive = bool(resolved and int(result.get("y", 0)) == 1)
        full_h_positive = positive
        rejected = bool(resolved and int(result.get("y", 0)) == 0)
        terminal_step = int(result.get("terminal_step", 10)) if resolved else None

        # Green is reserved for a complete H=10 certificate, not method identity.
        axis.plot(plan[:, 0], plan[:, 1], color="#333333", lw=.82, marker="o", ms=2.2)
        query = dict(candidate_id=0, controls=np.asarray(trace["controls"]), result=result)
        level_audit = None
        if full_h_positive:
            level_audit = checked_verifier_levels(dict(trace, gamma=float(gamma)), query, H=10)
            _draw_verifier_geometry(axis, level_audit)
        elif rejected:
            axis.plot(*plan[-1], "x", color=BV.RED, ms=7, mew=1.5, zorder=9)
        metadata.update(
            verifier_positive=positive,
            verifier_full_h_positive=full_h_positive,
            rejected=rejected,
            terminal_step=terminal_step,
            verifier_levels=(0 if level_audit is None else level_audit["rendered_levels"]),
            verifier_local_overlap=([] if level_audit is None else level_audit["current_disk_overlap_indices"]),
            verifier_runtime_authority=False,
            verifier_solver=result.get("diagnostics", {}).get("solver"),
            verifier_artificial_faces=result.get("diagnostics", {}).get("K_artificial"),
        )
    else:
        plan = np.asarray(trace["selected_plan_positions"], float)
        axis.plot(plan[:, 0], plan[:, 1], color="#7F3C8D", lw=.82, marker="o", ms=2.2)
        guidance = trace.get("accumulated_guidance")
        if guidance:
            start = np.asarray(trace["state"], float)[:2]
            component_keys = (
                ("goal_guidance_action", CYAN, "goal"),
                ("safety_guidance_action", MAGENTA, "safety"),
            )
            if all(key in guidance for key, _, _ in component_keys):
                component_metadata = {}
                for key, color, label in component_keys:
                    raw = np.asarray(guidance[key], float)
                    vector = float(guidance_scale) * raw
                    norm = float(np.linalg.norm(vector))
                    if norm > float(guidance_cap):
                        vector *= float(guidance_cap) / norm
                    axis.add_patch(FancyArrowPatch(
                        tuple(start), tuple(start + vector), arrowstyle="-|>",
                        mutation_scale=15, lw=2.6, color=color,
                        shrinkA=0, shrinkB=0, zorder=11,
                    ))
                    component_metadata[label] = dict(
                        action=raw.tolist(), norm=float(np.linalg.norm(raw)),
                    )
                metadata.update(
                    guidance_present=True,
                    guidance_components=component_metadata,
                    guidance_semantics=guidance.get("component_semantics"),
                    display_scale=float(guidance_scale),
                    display_cap=float(guidance_cap),
                )
            else:
                vector = float(guidance_scale) * np.asarray(
                    guidance["net_guidance_action"], float
                )
                norm = float(np.linalg.norm(vector))
                if norm > float(guidance_cap):
                    vector *= float(guidance_cap) / norm
                axis.add_patch(FancyArrowPatch(
                    tuple(start), tuple(start + vector), arrowstyle="-|>",
                    mutation_scale=15, lw=2.6, color=MAGENTA,
                    shrinkA=0, shrinkB=0, zorder=11,
                ))
                metadata.update(
                    guidance_present=True,
                    legacy_net_guidance=True,
                    net_guidance_action=np.asarray(
                        guidance["net_guidance_action"], float
                    ).tolist(),
                    net_guidance_norm=float(guidance["net_guidance_norm"]),
                    display_scale=float(guidance_scale),
                    display_cap=float(guidance_cap),
                )
        else:
            metadata["guidance_present"] = False
    _set_clean_axis(axis)
    return metadata


def _validate_method_runs(runs_by_method, gammas):
    if set(runs_by_method) != set(METHOD_KEYS):
        raise ValueError(f"method mapping keys must be exactly {METHOD_KEYS}")
    episodes = set()
    for method in METHOD_KEYS:
        for gamma in gammas:
            if float(gamma) not in runs_by_method[method]:
                raise KeyError(f"{method} is missing gamma={gamma}")
            episodes.add(int(runs_by_method[method][float(gamma)]["episode"]))
    if len(episodes) != 1:
        raise ValueError("all 3x3 cells must use one explicit episode ID")
    return episodes.pop()


def render_method_gamma_comparison(
        runs_by_method, output_png, *, snapshot_step, output_mp4=None,
        gammas=DISPLAY_GAMMAS, fps=8, frame_stride=2, report_path=None):
    """Render rows=(SafeMPPI expert, selected, Kazuki), columns=(gamma .1,.5,1)."""
    gammas = tuple(map(float, gammas))
    if gammas != DISPLAY_GAMMAS:
        raise ValueError(f"comparison requires gammas={DISPLAY_GAMMAS}")
    episode = _validate_method_runs(runs_by_method, gammas)
    verifier_cache = {}
    kazuki_guidance = runs_by_method["kazuki"][gammas[0]].get("effective_guidance", {})
    for gamma in gammas[1:]:
        if runs_by_method["kazuki"][gamma].get("effective_guidance", {}) != kazuki_guidance:
            raise ValueError("3x3 comparison requires one declared Kazuki coefficient pair")
    safe_values = tuple(kazuki_guidance.get("safe_coefs", ()))
    safe_text = ",".join(f"{float(value):g}" for value in safe_values) or "n/a"
    goal_text = ("n/a" if kazuki_guidance.get("goal_coef") is None
                 else f"{float(kazuki_guidance['goal_coef']):g}")
    reference_run = runs_by_method["selected"][gammas[0]]
    reference_trace = _run_trace(reference_run, 0)
    n_ped = int(len(np.asarray(reference_trace["ped_xy"])))
    speed_range = tuple(map(float, reference_run.get("ped_speed_range", ())))
    environment_text = f"n_ped={n_ped}"
    if len(speed_range) == 2:
        environment_text += f"\nspeed={speed_range[0]:g}--{speed_range[1]:g} m/s"

    def selected_verifier(gamma, step):
        trace = _run_trace(runs_by_method["selected"][gamma], step)
        key = (float(gamma), int(trace["step"]))
        if key not in verifier_cache:
            verifier_cache[key] = VS.verify_query(
                trace["state"], trace["controls"], trace["ped_xy"], trace["ped_vel"], gamma,
            )
        return verifier_cache[key]

    def make_figure():
        figure, axes = plt.subplots(3, 3, figsize=(13.7, 12.0))
        figure.subplots_adjust(left=.015, right=.78, bottom=.015, top=.985, wspace=.02, hspace=.02)
        figure.legend(
            handles=[
                Line2D([], [], color=BV.BLUE, lw=1.5, label="SafeMPPI expert + nominal levels"),
                Line2D([], [], color="#333333", lw=1.2, label="Arm-A r10 learned raw / planned window"),
                Line2D([], [], color=BV.GREEN, lw=.55, label="offline exact-SOCP h=1..10 (learned row)"),
                Line2D([], [], color=BV.GREEN, lw=1.25, label="offline exact-SOCP outer set (K=16)"),
                Line2D([], [], color="#7F3C8D", lw=1.5, label="Kazuki refined plan"),
                Line2D([], [], color=MAGENTA, lw=2.5, label="Kazuki net guidance"),
            ],
            loc="center left", bbox_to_anchor=(.79, .58), fontsize=8, frameon=False,
        )
        figure.text(
            .79, .38,
            "columns\ngamma 0.1 | 0.5 | 1.0\n\nrows\nSafeMPPI expert\nArm-A r10 learned raw\n"
            f"Kazuki generate-guide-refine\n  safe={safe_text}, goal={goal_text}\n\n{environment_text}\n\n"
            "green appears only when the learned\n"
            "raw H=10 window passes exact SOCP",
            ha="left", va="top", fontsize=8,
        )
        return figure, axes

    def draw(figure, axes, step):
        cells = []
        for row, method in enumerate(METHOD_KEYS):
            for column, gamma in enumerate(gammas):
                axes[row, column].clear()
                cells.append(draw_method_panel(
                    axes[row, column], method, runs_by_method[method][gamma], gamma, step,
                    verifier_result=(selected_verifier(gamma, step) if method == "selected" else None),
                ))
        return cells

    figure, axes = make_figure()
    cells = draw(figure, axes, int(snapshot_step))
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    figure.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(figure)
    video_frames = None
    if output_mp4 is not None:
        figure, axes = make_figure()
        maximum = max(
            len(list(run.get("trace") or [])) for values in runs_by_method.values() for run in values.values()
        ) - 1
        video_frames = list(range(0, maximum + 1, int(frame_stride)))
        if video_frames[-1] != maximum:
            video_frames.append(maximum)
        movie = animation.FuncAnimation(
            figure, lambda step: (draw(figure, axes, step) and []),
            frames=video_frames, interval=1000 / int(fps), blit=False,
        )
        movie.save(output_mp4, writer=animation.FFMpegWriter(fps=fps, bitrate=3200), dpi=95)
        plt.close(figure)
    report = dict(
        semantics={
            "expert": ("SafeMPPI demonstration expert; blue levels are its actual velocity-aware "
                       "nominal proposal polytope"),
            "selected": ("Arm-A r10 learned raw temp=1 rollout; green exact-K16 SOCP is an "
                         "offline audit and did not select the action"),
            "kazuki": "generate-guide-refine comparator; magenta is guided minus unguided first action for one latent",
        },
        explicit_episode=int(episode), explicit_snapshot_step=int(snapshot_step),
        gammas=list(gammas), rows=list(METHOD_KEYS), cells=cells,
        environment=dict(n_ped=n_ped, ped_speed_range=list(speed_range)),
        kazuki_guidance=dict(safe_coefs=list(safe_values), goal_coef=kazuki_guidance.get("goal_coef")),
        verifier_cache_entries=len(verifier_cache),
        png=os.path.abspath(output_png),
        mp4=(None if output_mp4 is None else os.path.abspath(output_mp4)),
        video_frames=video_frames,
    )
    if report_path is not None:
        _write_json(report_path, report)
    return report


def select_comparison_episode(episode_runs, *, gammas=DISPLAY_GAMMAS):
    """Predeclared exhaustive rule for a selected-success/baseline-failure case.

    ``episode_runs`` maps every inspected episode to the same method/gamma run
    mapping accepted by :func:`render_method_gamma_comparison`.  The complete
    score table is returned so a publication case is reproducible rather than
    hand-picked after looking at trajectories.
    """
    rows = []
    for episode in sorted(map(int, episode_runs)):
        runs = episode_runs[episode]
        _validate_method_runs(runs, gammas)
        selected_success = sum(bool(runs["selected"][float(gamma)]["success"]) for gamma in gammas)
        contrasts = sum(
            bool(runs["selected"][float(gamma)]["success"])
            and (not bool(runs[baseline][float(gamma)]["success"]))
            for baseline in ("expert", "kazuki") for gamma in gammas
        )
        baseline_failures = sum(
            not bool(runs[baseline][float(gamma)]["success"])
            for baseline in ("expert", "kazuki") for gamma in gammas
        )
        selected_clearances = [
            float(runs["selected"][float(gamma)].get("min_clearance",
                  runs["selected"][float(gamma)].get("min_clear", -float("inf"))))
            for gamma in gammas
        ]
        rows.append(dict(
            episode=episode, selected_successes=selected_success,
            selected_all_gammas=selected_success == len(gammas), contrasts=contrasts,
            baseline_failures=baseline_failures,
            selected_min_clearance=min(selected_clearances),
        ))
    eligible = [row for row in rows if row["selected_all_gammas"] and row["contrasts"] >= 1]
    if not eligible:
        raise RuntimeError("no episode has selected success at all gammas and at least one expert/Kazuki failure")
    eligible.sort(key=lambda row: (
        -row["contrasts"], -row["baseline_failures"],
        -row["selected_min_clearance"], row["episode"],
    ))
    return dict(
        rule=("exhaustive fixed bank: require selected success at all displayed gammas and >=1 paired expert/Kazuki "
              "failure; maximize selected-vs-baseline contrasts, then baseline failures, then selected minimum "
              "clearance; tie-break by episode ID"),
        chosen=eligible[0], candidates=rows,
    )


def _mapping_value(mapping, gamma):
    candidates = (float(gamma), str(float(gamma)), f"{float(gamma):g}")
    for key in candidates:
        if key in mapping:
            return mapping[key]
    raise KeyError(f"method bundle is missing gamma={gamma}")


def normalize_method_bundle(bundle, *, scenario_id, gammas=DISPLAY_GAMMAS):
    """Normalize one explicit, already-run controller bundle for rendering.

    Accepted payload: either the method mapping itself, or a dictionary with a
    ``runs`` method mapping.  Method aliases match the density diagnostic's
    persisted names.  No controller is invoked here.
    """
    if not isinstance(bundle, dict):
        raise TypeError("method-runs bundle must be a dictionary")
    if bundle.get("scenario_id") is not None and int(bundle["scenario_id"]) != int(scenario_id):
        raise ValueError("method-runs bundle scenario does not match --scenario")
    source = bundle.get("runs", bundle)
    normalized = {}
    for method in METHOD_KEYS:
        aliases = [alias for alias in METHOD_ALIASES[method] if alias in source]
        if len(aliases) != 1:
            raise KeyError(f"method-runs bundle needs exactly one of {METHOD_ALIASES[method]}")
        values = source[aliases[0]]
        if not isinstance(values, dict):
            raise TypeError(f"{aliases[0]} runs must be keyed by gamma")
        normalized[method] = {}
        for gamma in gammas:
            run = _mapping_value(values, gamma)
            if int(run["episode"]) != int(scenario_id):
                raise ValueError(f"{aliases[0]} gamma={gamma} has the wrong episode")
            normalized[method][float(gamma)] = run
    return normalized


def _snapshot_from_metadata(traces, path):
    with open(path) as stream:
        payload = json.load(stream)
    selected = payload.get("selected_snapshot", payload.get("chosen"))
    if not isinstance(selected, dict):
        raise ValueError("snapshot metadata needs selected_snapshot or chosen")
    key = dict(
        scenario_id=int(selected["scenario_id"]), gamma=float(selected["gamma"]),
        step=int(selected["step"]),
    )
    return _find_trace(traces, key), dict(source="metadata", path=os.path.abspath(path), key=key)


def _explicit_snapshot_selection(trace, source):
    row = _snapshot_row(trace)
    return dict(
        rule=f"explicit snapshot supplied by {source}; no renderer-side outcome curation",
        chosen=row, candidates=[row],
    )


def _validate_exact_query_traces(traces):
    """Fail closed instead of painting legacy K12/angular traces as faithful."""
    resolved = 0
    for trace in traces:
        for row in trace.get("query_rows", []):
            result = row.get("result", {})
            if not result.get("resolved"):
                continue
            resolved += 1
            diagnostics = result.get("diagnostics", {})
            if diagnostics.get("solver") != "paper_static_exact_2d_angular_interval_socp":
                raise ValueError("query traces were not collected with the exact 2-D SOCP")
            if int(diagnostics.get("K_artificial", -1)) != VS.ARTIFICIAL_FACES:
                raise ValueError("query traces do not use 16 artificial outer faces")
    if resolved == 0:
        raise ValueError("query traces contain no resolved exact-SOCP queries")
    return resolved


def render_bundle(
        method_runs_path, query_traces_path, scenario_id, output_dir, *,
        snapshot_trace_path=None, snapshot_metadata_path=None,
        snapshot_step=None, snapshot_gamma=None,
        success_method_runs_path=None, success_scenario_id=None):
    """Render the complete density-OOD handoff without running controllers."""
    sources = [snapshot_trace_path is not None, snapshot_metadata_path is not None, snapshot_step is not None]
    if sum(sources) != 1:
        raise ValueError("choose exactly one snapshot source: trace, metadata, or explicit step")
    if snapshot_step is not None and snapshot_gamma is None:
        raise ValueError("--snapshot-step requires --snapshot-gamma")
    method_bundle = torch.load(method_runs_path, map_location="cpu", weights_only=False)
    method_snapshot = method_bundle.get("shared_snapshot") if isinstance(method_bundle, dict) else None
    if not isinstance(method_snapshot, dict) or "step" not in method_snapshot:
        raise ValueError("method-runs bundle requires driver-declared shared_snapshot.step")
    method_snapshot_step = int(method_snapshot["step"])
    method_runs = normalize_method_bundle(method_bundle, scenario_id=scenario_id)
    if (success_method_runs_path is None) != (success_scenario_id is None):
        raise ValueError("success method runs and scenario must be supplied together")
    traces = torch.load(query_traces_path, map_location="cpu", weights_only=False)
    if not isinstance(traces, list) or not traces:
        raise ValueError("query-traces must contain a nonempty trace list")
    exact_resolved_queries = _validate_exact_query_traces(traces)
    if snapshot_trace_path is not None:
        snapshot = torch.load(snapshot_trace_path, map_location="cpu", weights_only=False)
        if not isinstance(snapshot, dict):
            raise TypeError("snapshot-trace must contain one trace dictionary")
        source = dict(source="trace", path=os.path.abspath(snapshot_trace_path), key={
            "scenario_id": int(snapshot["scenario_id"]), "gamma": float(snapshot["gamma"]),
            "step": int(snapshot["step"]),
        })
    elif snapshot_metadata_path is not None:
        snapshot, source = _snapshot_from_metadata(traces, snapshot_metadata_path)
    else:
        key = dict(scenario_id=int(scenario_id), gamma=float(snapshot_gamma), step=int(snapshot_step))
        snapshot = _find_trace(traces, key)
        source = dict(source="explicit", key=key)
    if int(snapshot["scenario_id"]) != int(scenario_id):
        raise ValueError("snapshot scenario does not match --scenario")
    _validate_exact_query_traces([snapshot])

    os.makedirs(output_dir, exist_ok=False)
    paths = {
        "comparison_png": os.path.join(output_dir, "method_gamma_3x3.png"),
        "comparison_mp4": os.path.join(output_dir, "method_gamma_3x3.mp4"),
        "gathering_mp4": os.path.join(output_dir, "max_margin_gathering.mp4"),
        "gathering_snapshot": os.path.join(output_dir, "max_margin_gathering_snapshot.png"),
        "candidate_snapshot": os.path.join(output_dir, "candidate_specific_B.png"),
    }
    if success_method_runs_path is not None:
        paths.update(
            success_comparison_png=os.path.join(output_dir, "method_gamma_3x3_all_success.png"),
            success_comparison_mp4=os.path.join(output_dir, "method_gamma_3x3_all_success.mp4"),
        )
    comparison = render_method_gamma_comparison(
        method_runs, paths["comparison_png"], snapshot_step=method_snapshot_step,
        output_mp4=paths["comparison_mp4"],
    )
    gathering = render_margin_gathering_video(
        traces, int(scenario_id), paths["gathering_mp4"],
        output_snapshot=paths["gathering_snapshot"], snapshot_step=int(snapshot["step"]),
    )
    candidate = render_candidate_query_snapshot(
        [snapshot], paths["candidate_snapshot"],
        selection=_explicit_snapshot_selection(snapshot, source["source"]),
    )
    success_comparison = None
    success_source = None
    if success_method_runs_path is not None:
        success_bundle = torch.load(success_method_runs_path, map_location="cpu", weights_only=False)
        success_snapshot = success_bundle.get("shared_snapshot") if isinstance(success_bundle, dict) else None
        if not isinstance(success_snapshot, dict) or "step" not in success_snapshot:
            raise ValueError("success method-runs bundle requires shared_snapshot.step")
        success_runs = normalize_method_bundle(
            success_bundle, scenario_id=int(success_scenario_id),
        )
        if not all(bool(success_runs[method][gamma]["success"])
                   for method in METHOD_KEYS for gamma in DISPLAY_GAMMAS):
            raise ValueError("the declared all-success episode does not have nine successes")
        success_comparison = render_method_gamma_comparison(
            success_runs, paths["success_comparison_png"],
            snapshot_step=int(success_snapshot["step"]),
            output_mp4=paths["success_comparison_mp4"],
        )
        success_source = dict(
            path=os.path.abspath(success_method_runs_path),
            sha256=_sha256(success_method_runs_path), scenario_id=int(success_scenario_id),
            selection_rule="lowest scenario ID in the fixed finite bank with all 3 methods x 3 gammas successful",
            shared_snapshot=success_snapshot,
        )
    artifacts = {
        name: dict(path=os.path.abspath(path), bytes=os.path.getsize(path), sha256=_sha256(path))
        for name, path in paths.items()
    }
    for trace in traces:
        for candidate_id in trace["selected_ids"]:
            _query_result(trace, candidate_id)
    manifest = dict(
        status="DENSITY_OOD_VISUALIZATION_BUNDLE_COMPLETE",
        diagnostic_only=True, controllers_rerun_by_renderer=False,
        exact_socp_resolved_queries=int(exact_resolved_queries),
        scenario_id=int(scenario_id), gammas=list(DISPLAY_GAMMAS),
        source_files={
            "method_runs": dict(path=os.path.abspath(method_runs_path), sha256=_sha256(method_runs_path),
                                shared_snapshot=method_snapshot),
            "query_traces": dict(path=os.path.abspath(query_traces_path), sha256=_sha256(query_traces_path)),
            "snapshot": source,
            "all_success_method_runs": success_source,
        },
        query_horizon_contract="every resolved B query is reverified over all H=10 transitions",
        snapshot_steps=dict(
            method_comparison=dict(
                step=method_snapshot_step,
                source="method-runs shared_snapshot.step",
                metadata=method_snapshot,
            ),
            query_gathering_and_candidates=dict(
                step=int(snapshot["step"]), gamma=float(snapshot["gamma"]),
                source=source["source"],
            ),
        ),
        comparison=comparison, all_success_comparison=success_comparison,
        gathering=gathering, candidate_snapshot=candidate,
        artifacts=artifacts,
    )
    manifest_path = os.path.join(output_dir, "render_manifest.json")
    _write_json(manifest_path, manifest)
    return manifest


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-runs", required=True,
                        help="torch bundle of already-run expert/selected/Kazuki traces keyed by gamma")
    parser.add_argument("--query-traces", required=True,
                        help="torch list from diagnostic-only max-margin gathering")
    parser.add_argument("--scenario", required=True, type=int)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--snapshot-trace")
    source.add_argument("--snapshot-metadata")
    source.add_argument("--snapshot-step", type=int)
    parser.add_argument("--snapshot-gamma", type=float)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--success-method-runs")
    parser.add_argument("--success-scenario", type=int)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    render_bundle(
        args.method_runs, args.query_traces, args.scenario, args.output_dir,
        snapshot_trace_path=args.snapshot_trace,
        snapshot_metadata_path=args.snapshot_metadata,
        snapshot_step=args.snapshot_step, snapshot_gamma=args.snapshot_gamma,
        success_method_runs_path=args.success_method_runs,
        success_scenario_id=args.success_scenario,
    )


if __name__ == "__main__":
    main()
