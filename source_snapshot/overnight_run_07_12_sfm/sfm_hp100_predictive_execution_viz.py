"""Trace-only videos for the always-on predictive execution diagnostic.

This module consumes ``predictive_trace.pt`` produced by
``sfm_hp100_predictive_execution.py``.  It never samples, verifies, advances a
scene, or edits a checkpoint.  The historical final-video renderer is imported
only for its authenticated drawing primitives and presentation style.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import torch

import sfm_hp100_final_videos as FINAL
import sfm_hp100_paper_video_style as STYLE
import sfm_scene as SS


STATUS = "SFM_HP100_PREDICTIVE_EXECUTION_VIZ_COMPLETE"
VERSION = "sfm_hp100_predictive_execution_viz_v1"


def _finalize(output: Path, metadata: dict) -> dict:
    payload = {
        "status": STATUS,
        "viz_version": VERSION,
        "style_version": STYLE.VERSION,
        **metadata,
        "mp4": str(output.resolve()),
        "mp4_sha256": STYLE.sha256_file(output),
        "bytes": int(output.stat().st_size),
    }
    sidecar = output.with_suffix(".json")
    STYLE.write_json(sidecar, payload)
    payload["sidecar"] = str(sidecar.resolve())
    return payload


def _load(path: str | Path) -> dict:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict) or not isinstance(value.get("events"), list):
        raise ValueError("predictive trace must be a mapping with an events list")
    return value


def _events_for_lineage(trace: dict, lineage: str) -> list[dict]:
    rows = sorted(
        (row for row in trace["events"] if str(row["lineage"]) == str(lineage)),
        key=lambda row: int(row["step"]),
    )
    if not rows:
        raise KeyError(f"trace has no lineage {lineage!r}")
    steps = [int(row["step"]) for row in rows]
    if steps != list(range(len(steps))):
        raise ValueError("lineage event steps must be contiguous from zero")
    return rows


def _final_attempt(event: dict) -> dict:
    attempts = list(event.get("attempts", ()))
    if not attempts:
        raise ValueError("always-on predictive event contains no acquisition attempt")
    return attempts[-1]


def _selected_segment(event: dict) -> tuple[np.ndarray, bool, dict | None]:
    attempt = _final_attempt(event)
    local = attempt.get("predictive_local")
    if local is not None:
        local = int(local)
        if not bool(attempt["verification"][local]["valid"]):
            raise ValueError("predictive selection is not exact-positive")
        replay_positive = event.get("executed_role") == "positive"
        return (
            np.asarray(attempt["B_segments"][local], float),
            bool(replay_positive),
            attempt.get("predictive_sidecar"),
        )
    local = attempt.get("negative_counterfactual_local")
    if local is None:
        raise ValueError("unexecuted event lacks its terminal counterfactual")
    local = int(local)
    if bool(attempt["verification"][local]["valid"]):
        raise ValueError("terminal counterfactual is not exact-negative")
    return np.asarray(attempt["B_segments"][local], float), False, None


def _history(rows: list[dict], index: int, *, reveal_current: bool) -> np.ndarray:
    states = [np.asarray(row["state_before"], float)[:2] for row in rows[:index + 1]]
    current = rows[index]
    if reveal_current and current.get("executed_role") is not None:
        states.append(np.asarray(current["state_after"], float)[:2])
    return np.asarray(states, float)


def _rollout_stats(rows: list[dict]) -> dict:
    attempts = [_final_attempt(row) for row in rows]
    positive = [
        int(attempt["positive_B32"])
        for attempt in attempts
    ]
    comparable = [
        attempt for attempt in attempts
        if attempt.get("predictive_local") is not None
        and attempt.get("max_margin_reference_local") is not None
    ]
    changed = sum(
        int(attempt["predictive_local"])
        != int(attempt["max_margin_reference_local"])
        for attempt in comparable
    )
    chosen_progress = []
    chosen_clearance = []
    for attempt in comparable:
        local = int(attempt["predictive_local"])
        chosen_progress.append(float(attempt["verification"][local]["H10_progress"]))
        chosen_clearance.append(float(
            attempt["prediction_audits"][local]["predicted_min_clearance"]
        ))
    return {
        "contexts": len(rows),
        "executed_positive_contexts": sum(
            row.get("executed_role") == "positive" for row in rows
        ),
        "executed_realized_failure_contexts": sum(
            str(row.get("executed_role", "")).startswith("realized_")
            for row in rows
        ),
        "terminal": rows[-1].get("terminal"),
        "mean_exact_positive_B32": float(np.mean(positive)),
        "median_exact_positive_B32": float(np.median(positive)),
        "zero_positive_B32_fraction": float(np.mean(np.asarray(positive) == 0)),
        "selector_change_fraction": (
            None if not comparable else float(changed / len(comparable))
        ),
        "mean_selected_H10_progress": (
            None if not chosen_progress else float(np.mean(chosen_progress))
        ),
        "mean_selected_predicted_clearance": (
            None if not chosen_clearance else float(np.mean(chosen_clearance))
        ),
    }


def acquisition_grid_lineages(
    trace: dict,
    *,
    gammas: tuple[float, ...] = (0.1, 0.5, 1.0),
    replicas: tuple[int, ...] = (0, 1),
) -> list[list[tuple[str, list[dict]]]]:
    """Resolve an exact replica-by-gamma lineage grid from trace outcomes."""
    outcomes = trace.get("outcomes")
    if not isinstance(outcomes, dict):
        raise ValueError("predictive trace must contain an outcomes mapping")
    grid = []
    for replica in replicas:
        row = []
        scenario_ids = set()
        for gamma in gammas:
            matches = [
                (lineage, outcome)
                for lineage, outcome in outcomes.items()
                if int(outcome["replica"]) == int(replica)
                and np.isclose(float(outcome["gamma"]), float(gamma))
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"expected one outcome for replica={replica}, gamma={gamma}; "
                    f"found {len(matches)}"
                )
            lineage, outcome = matches[0]
            rows = _events_for_lineage(trace, lineage)
            scenario_ids.add(int(outcome["scenario_id"]))
            row.append((str(lineage), rows))
        if len(scenario_ids) != 1:
            raise ValueError(
                f"replica {replica} is not paired across gamma: {scenario_ids}"
            )
        grid.append(row)
    return grid


def acquisition_grid_outcomes(
    trace: dict,
    grid: list[list[tuple[str, list[dict]]]],
) -> dict:
    """Summarize terminal outcomes for an acquisition lineage grid."""
    statuses = ("success", "collision", "nvp", "timeout")
    lineages = []
    per_gamma: dict[str, dict] = {}
    pooled = {status: 0 for status in statuses}
    for row in grid:
        for lineage, events in row:
            outcome = trace["outcomes"][lineage]
            status = str(outcome["status"]).lower()
            if status not in pooled:
                raise ValueError(f"unsupported terminal status {status!r}")
            gamma = float(outcome["gamma"])
            key = f"{gamma:g}"
            cell = per_gamma.setdefault(
                key, {"gamma": gamma, "total": 0, **{name: 0 for name in statuses}},
            )
            cell[status] += 1
            cell["total"] += 1
            pooled[status] += 1
            lineages.append({
                "lineage": str(lineage),
                "gamma": gamma,
                "replica": int(outcome["replica"]),
                "scenario_id": int(outcome["scenario_id"]),
                "status": status,
                "executed_steps": int(outcome["executed_steps"]),
                "contexts": len(events),
            })
    total = len(lineages)
    for cell in per_gamma.values():
        cell["rates"] = {
            name: float(cell[name] / cell["total"]) for name in statuses
        }
    return {
        "total": total,
        **pooled,
        "rates": {name: float(pooled[name] / total) for name in statuses},
        "per_gamma": per_gamma,
        "lineages": lineages,
    }


def _draw_acquisition_grid_axis(
    axis,
    rows: list[dict],
    index: int,
    *,
    terminal_status: str,
    held_terminal: bool,
    bounds,
) -> None:
    axis.clear()
    event = rows[index]
    FINAL._draw_pedestrians(axis, event["ped_xy"], event["ped_vel"])
    _draw_candidate_population(axis, event, phase="B")
    segment, replay_positive, sidecar = _selected_segment(event)
    axis.plot(
        segment[:, 0], segment[:, 1],
        color=(STYLE.POSITIVE_BLUE if replay_positive else STYLE.NEGATIVE_RED),
        lw=STYLE.SAMPLE_LW + 0.9, alpha=1.0, zorder=18,
    )
    if not replay_positive:
        axis.plot(
            segment[-1, 0], segment[-1, 1], "x",
            color=STYLE.NEGATIVE_RED, ms=7.2, mew=1.5, zorder=19,
        )
    if sidecar is not None:
        FINAL._draw_verifier(axis, float(event["gamma"]), sidecar)
    history = _history(rows, index, reveal_current=True)
    axis.plot(
        history[:, 0], history[:, 1], color=STYLE.EXECUTED_BLACK,
        lw=max(0.9, STYLE.EXECUTED_LW - 0.35), zorder=20,
    )
    if len(history) > 1:
        axis.scatter(
            history[:-1, 0], history[:-1, 1], s=5.0,
            color=STYLE.EXECUTED_BLACK, edgecolors="none", zorder=21,
        )
    robot = event["state_after"] if event.get("executed_role") is not None \
        else event["state_before"]
    FINAL._draw_robot_goal(axis, robot)
    STYLE.fixed_world_frame(axis, bounds=bounds)
    attempt = _final_attempt(event)
    axis.text(
        0.025, 0.025,
        (
            rf"$t={int(event['step'])}$" "\n"
            rf"$B^+={int(attempt['positive_B32'])}/32$" "\n"
            rf"$a={int(attempt['attempt']) + 1}$"
        ),
        transform=axis.transAxes, ha="left", va="bottom", fontsize=7.3,
        color=STYLE.EXECUTED_BLACK, zorder=100,
        bbox={
            "boxstyle": "round,pad=0.20", "facecolor": "white",
            "edgecolor": "none", "alpha": 0.74,
        },
    )
    if held_terminal or event.get("terminal") is not None:
        good = terminal_status == "success"
        axis.text(
            0.975, 0.975, terminal_status.upper(),
            transform=axis.transAxes, ha="right", va="top", fontsize=8.2,
            weight="bold", color=("#138A36" if good else STYLE.NEGATIVE_RED),
            zorder=100,
            bbox={
                "boxstyle": "round,pad=0.22", "facecolor": "white",
                "edgecolor": "none", "alpha": 0.84,
            },
        )


def render_acquisition_grid(
    trace: dict,
    output: str | Path,
    *,
    gammas: tuple[float, ...] = (0.1, 0.5, 1.0),
    replicas: tuple[int, ...] = (0, 1),
    fps: int = 5,
    frame_stride: int = 1,
    bounds=None,
) -> dict:
    """Render two paired episodes across gamma for their complete acquisition."""
    STYLE.apply_computer_modern_style()
    gammas = tuple(float(value) for value in gammas)
    replicas = tuple(int(value) for value in replicas)
    grid = acquisition_grid_lineages(trace, gammas=gammas, replicas=replicas)
    outcomes = acquisition_grid_outcomes(trace, grid)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    maximum = max(len(rows) for row in grid for _, rows in row)
    indices = list(range(0, maximum, int(frame_stride)))
    if indices[-1] != maximum - 1:
        indices.append(maximum - 1)
    figure, axes = plt.subplots(
        len(replicas), len(gammas),
        figsize=(5.25 * len(gammas), 5.25 * len(replicas)),
        squeeze=False,
    )
    for column, gamma in enumerate(gammas):
        axes[0, column].set_title(rf"$\gamma={gamma:g}$", fontsize=11.0)
    for row_index, replica in enumerate(replicas):
        outcome = trace["outcomes"][grid[row_index][0][0]]
        axes[row_index, 0].set_ylabel(
            f"episode {int(outcome['scenario_id'])}", fontsize=10.0,
        )
    figure.legend(
        handles=[
            Line2D([], [], color=STYLE.UNCERTAINTY_GRAY, lw=1.1,
                   label="K=64 flow proposals"),
            Line2D([], [], color="#2A788E", lw=STYLE.ROLLOUT_LW,
                   label="B=32 uncertainty-acquired"),
            Line2D([], [], color=STYLE.POSITIVE_BLUE, lw=STYLE.SAMPLE_LW,
                   label="prediction-selected exact positive"),
            Line2D([], [], color=STYLE.NEGATIVE_RED, marker="x", lw=0,
                   label="exact negative / terminal counterfactual"),
            Line2D([], [], color=STYLE.EXECUTED_BLACK, lw=STYLE.EXECUTED_LW,
                   label="executed trajectory"),
        ],
        loc="lower center", ncol=5, frameon=False, fontsize=8.5,
    )

    def update(global_index):
        for row_index, row in enumerate(grid):
            for column, (lineage, rows) in enumerate(row):
                index = min(int(global_index), len(rows) - 1)
                terminal_status = str(trace["outcomes"][lineage]["status"]).lower()
                _draw_acquisition_grid_axis(
                    axes[row_index, column], rows, index,
                    terminal_status=terminal_status,
                    held_terminal=int(global_index) >= len(rows) - 1,
                    bounds=bounds,
                )
                if row_index == 0:
                    axes[row_index, column].set_title(
                        rf"$\gamma={gammas[column]:g}$", fontsize=11.0,
                    )
                if column == 0:
                    scenario = trace["outcomes"][lineage]["scenario_id"]
                    axes[row_index, column].set_ylabel(
                        f"episode {int(scenario)}", fontsize=10.0,
                    )
        figure.tight_layout(rect=(0.0, 0.055, 1.0, 1.0), pad=0.35)
        return []

    movie = animation.FuncAnimation(
        figure, update, frames=indices, interval=1000 / int(fps), blit=False,
    )
    movie.save(output, writer=FINAL._writer(fps, bitrate=7600), dpi=120)
    plt.close(figure)
    return _finalize(output, {
        "kind": "always_on_predictive_acquisition_grid",
        "scene_profile": str(trace.get("preflight", {}).get(
            "scene", {}).get("name", "double_density_velocity_ood"
        )),
        "gammas": list(gammas),
        "replicas": list(replicas),
        "scientific_K": 64,
        "scientific_B": 32,
        "execution_rule": "exact-positive max H10 progress; clearance/sigma/index tie-break",
        "fixed_camera": (
            list(map(float, bounds)) if bounds is not None
            else [float(SS.TASK_LO), float(SS.TASK_HI)]
        ),
        "frame_stride": int(frame_stride),
        "outcomes": outcomes,
        "scientific_scope": (
            "acquisition-controller lineages; not raw-policy evaluation and "
            "not an expanded checkpoint"
        ),
    })


def render_rollout(
    trace: dict,
    lineage: str,
    output: str | Path,
    *,
    fps: int = 7,
    frame_stride: int = 1,
    show_safety_badge: bool = True,
    bounds=None,
) -> dict:
    """Render the executed predictive trajectory in the raw-after style."""
    STYLE.apply_computer_modern_style()
    rows = _events_for_lineage(trace, lineage)
    indices = list(range(0, len(rows), int(frame_stride)))
    if indices[-1] != len(rows) - 1:
        indices.append(len(rows) - 1)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    figure, axis = plt.subplots(figsize=(6.8, 6.8))

    def update(index):
        index = int(index)
        axis.clear()
        current = rows[index]
        FINAL._draw_pedestrians(axis, current["ped_xy"], current["ped_vel"])
        for offset, row in enumerate(rows[:index + 1]):
            segment, valid, _ = _selected_segment(row)
            axis.plot(
                segment[:, 0], segment[:, 1],
                color=(STYLE.POSITIVE_BLUE if valid else STYLE.NEGATIVE_RED),
                lw=STYLE.SAMPLE_LW,
                alpha=0.18 if offset != index else 0.96,
                zorder=8,
            )
            if not valid:
                axis.plot(
                    segment[-1, 0], segment[-1, 1], "x",
                    color=STYLE.NEGATIVE_RED, ms=6.0, mew=1.25, zorder=14,
                )
        segment, valid, sidecar = _selected_segment(current)
        history = _history(rows, index, reveal_current=True)
        axis.plot(
            history[:, 0], history[:, 1], color=STYLE.EXECUTED_BLACK,
            lw=STYLE.EXECUTED_LW, zorder=20,
        )
        if len(history) > 1:
            axis.scatter(
                history[:-1, 0], history[:-1, 1], s=7.0,
                color=STYLE.EXECUTED_BLACK, edgecolors="none", zorder=21,
            )
        if sidecar is not None:
            FINAL._draw_verifier(axis, float(current["gamma"]), sidecar)
        robot = current["state_after"] if current.get("executed_role") is not None \
            else current["state_before"]
        FINAL._draw_robot_goal(axis, robot)
        STYLE.fixed_world_frame(axis, bounds=bounds)
        STYLE.safety_badge(axis, valid, enabled=show_safety_badge)
        return []

    movie = animation.FuncAnimation(
        figure, update, frames=indices, interval=1000 / int(fps), blit=False,
    )
    movie.save(output, writer=FINAL._writer(fps), dpi=130)
    plt.close(figure)
    first = rows[0]
    return _finalize(output, {
        "kind": "always_on_predictive_execution_rollout",
        "scene_profile": str(trace.get("preflight", {}).get(
            "scene", {}).get("name", "double_density_velocity_ood"
        )),
        "episode": int(first["scenario_id"]),
        "lineage": str(lineage),
        "gamma": float(first["gamma"]),
        "scientific_K": 64,
        "scientific_B": 32,
        "execution_rule": "exact-positive max H10 progress; clearance/sigma/index tie-break",
        "fixed_camera": (
            list(map(float, bounds)) if bounds is not None
            else [float(SS.TASK_LO), float(SS.TASK_HI)]
        ),
        "show_safety_badge": bool(show_safety_badge),
        "rollout_stats": _rollout_stats(rows),
    })


def event_case_stats(event: dict) -> dict:
    """Return selector-comparison statistics for one trace event."""
    attempt = _final_attempt(event)
    predictive = attempt.get("predictive_local")
    reference = attempt.get("max_margin_reference_local")
    ped_xy = np.asarray(event["ped_xy"], float).reshape(-1, 2)
    robot_xy = np.asarray(event["state_before"], float)[:2]
    nearest = (
        float("inf") if not len(ped_xy) else
        float(np.linalg.norm(ped_xy - robot_xy[None], axis=1).min() - SS.R_PED)
    )
    value = {
        "event_id": (
            f"{event['lineage']}__scenario{int(event['scenario_id'])}"
            f"__step{int(event['step']):03d}"
        ),
        "lineage": str(event["lineage"]),
        "gamma": float(event["gamma"]),
        "scenario_id": int(event["scenario_id"]),
        "step": int(event["step"]),
        "attempt": int(attempt["attempt"]),
        "base_std": float(attempt["base_std"]),
        "current_nearest_pedestrian_clearance": nearest,
        "exact_positive_B32": int(attempt["positive_B32"]),
        "predictive_local": None if predictive is None else int(predictive),
        "max_margin_reference_local": (
            None if reference is None else int(reference)
        ),
    }
    if predictive is None or reference is None:
        return {
            **value,
            "selector_disagreement": False,
            "predictive_collision_free": None,
            "predictive_H10_progress": None,
            "max_margin_H10_progress": None,
            "H10_progress_gain": None,
            "predictive_min_clearance": None,
            "max_margin_min_clearance": None,
            "predictive_step_margin": None,
            "max_margin_step_margin": None,
            "predictive_sigma_rank": None,
        }
    predictive = int(predictive)
    reference = int(reference)
    verification = attempt["verification"]
    audits = attempt["prediction_audits"]
    sigma = np.asarray(attempt["selected_sigma"], float)
    descending = np.argsort(-sigma, kind="stable")
    sigma_rank = int(np.where(descending == predictive)[0][0]) + 1
    return {
        **value,
        "selector_disagreement": predictive != reference,
        "predictive_collision_free": bool(
            audits[predictive]["predicted_collision_free"]
        ),
        "predictive_H10_progress": float(
            verification[predictive]["H10_progress"]
        ),
        "max_margin_H10_progress": float(
            verification[reference]["H10_progress"]
        ),
        "H10_progress_gain": float(
            verification[predictive]["H10_progress"]
            - verification[reference]["H10_progress"]
        ),
        "predictive_min_clearance": float(
            audits[predictive]["predicted_min_clearance"]
        ),
        "max_margin_min_clearance": float(
            audits[reference]["predicted_min_clearance"]
        ),
        "predictive_step_margin": float(
            verification[predictive]["step_margin"]
        ),
        "max_margin_step_margin": float(
            verification[reference]["step_margin"]
        ),
        "predictive_sigma_rank": sigma_rank,
    }


def select_case_events(
    events: list[dict],
    *,
    count: int = 5,
    max_nearest_clearance: float = 0.5,
    min_exact_positive: int = 4,
    min_progress_gain: float = 0.1,
) -> list[tuple[dict, dict]]:
    """Select deterministic, near-pedestrian selector-disagreement cases.

    A first pass takes the strongest eligible event from every represented
    gamma.  Remaining slots use the global ranking.  This keeps gamma coverage
    without hiding a stronger fifth event.
    """
    if count not in (4, 5):
        raise ValueError("mechanism delivery requires four or five cases")
    candidates = []
    for event in events:
        stats = event_case_stats(event)
        if not stats["selector_disagreement"]:
            continue
        if stats["current_nearest_pedestrian_clearance"] > max_nearest_clearance:
            continue
        if stats["exact_positive_B32"] < int(min_exact_positive):
            continue
        if stats["H10_progress_gain"] < float(min_progress_gain):
            continue
        if not stats["predictive_collision_free"]:
            continue
        if stats["predictive_min_clearance"] < -1.0e-9:
            continue
        candidates.append((event, stats))
    key = lambda pair: (
        -float(pair[1]["H10_progress_gain"]),
        float(pair[1]["current_nearest_pedestrian_clearance"]),
        float(pair[1]["gamma"]), int(pair[1]["scenario_id"]),
        int(pair[1]["step"]), str(pair[1]["lineage"]),
    )
    candidates.sort(key=key)
    selected = []
    for gamma in sorted({pair[1]["gamma"] for pair in candidates}):
        selected.append(next(pair for pair in candidates if pair[1]["gamma"] == gamma))
        if len(selected) == count:
            return selected
    selected_ids = {pair[1]["event_id"] for pair in selected}
    for pair in candidates:
        if pair[1]["event_id"] in selected_ids:
            continue
        selected.append(pair)
        selected_ids.add(pair[1]["event_id"])
        if len(selected) == count:
            break
    if len(selected) < count:
        raise RuntimeError(
            f"only {len(selected)} events satisfy the declared case screen; "
            f"requested {count}"
        )
    return selected


def _draw_candidate_population(axis, event: dict, *, phase: str) -> None:
    attempt = _final_attempt(event)
    K_segments = np.asarray(attempt["K_segments"], float)
    axis.add_collection(LineCollection(
        K_segments, colors=STYLE.UNCERTAINTY_GRAY, linewidths=0.42,
        alpha=0.085, zorder=3,
    ))
    if phase == "K":
        return
    segments = np.asarray(attempt["B_segments"], float)
    sigma = np.asarray(attempt["selected_sigma"], float)
    lo, hi = float(sigma.min()), float(sigma.max())
    norm = Normalize(lo, hi + 1.0e-12)
    cmap = plt.get_cmap("viridis")
    for index, segment in enumerate(segments):
        axis.plot(
            segment[:, 0], segment[:, 1], color=cmap(norm(sigma[index])),
            lw=STYLE.ROLLOUT_LW, alpha=0.38, zorder=6,
        )
        if not bool(attempt["verification"][index]["valid"]):
            axis.plot(
                segment[-1, 0], segment[-1, 1], "x",
                color=STYLE.NEGATIVE_RED, ms=5.0, mew=1.0, alpha=0.90,
                zorder=11,
            )


def _draw_case_axis(
    axis,
    event: dict,
    stats: dict,
    *,
    selector: str,
    phase: str,
    bounds,
    history: np.ndarray,
) -> None:
    axis.clear()
    FINAL._draw_pedestrians(axis, event["ped_xy"], event["ped_vel"])
    _draw_candidate_population(axis, event, phase=phase)
    history = np.asarray(history, float).reshape(-1, 2)
    axis.plot(
        history[:, 0], history[:, 1], color=STYLE.EXECUTED_BLACK,
        lw=STYLE.EXECUTED_LW, zorder=20,
    )
    if phase in ("selected", "execute"):
        attempt = _final_attempt(event)
        field = (
            "max_margin_reference_local" if selector == "max_margin"
            else "predictive_local"
        )
        local = int(attempt[field])
        segment = np.asarray(attempt["B_segments"][local], float)
        color = STYLE.WEIGHTED_GOLD if selector == "max_margin" \
            else STYLE.POSITIVE_BLUE
        axis.plot(
            segment[:, 0], segment[:, 1], color=color,
            lw=STYLE.SAMPLE_LW + 0.55, alpha=1.0, zorder=18,
        )
        if selector == "predictive":
            FINAL._draw_verifier(
                axis, float(event["gamma"]), attempt["predictive_sidecar"],
            )
            if phase == "execute":
                state_after = np.asarray(event["state_after"], float)[:2]
                axis.plot(
                    [history[-1, 0], state_after[0]],
                    [history[-1, 1], state_after[1]],
                    color=STYLE.EXECUTED_BLACK, lw=STYLE.EXECUTED_LW,
                    zorder=20,
                )
    FINAL._draw_robot_goal(axis, event["state_before"])
    STYLE.fixed_world_frame(axis, bounds=bounds)
    if selector == "max_margin":
        progress = stats["max_margin_H10_progress"]
        clearance = stats["max_margin_min_clearance"]
        step_margin = stats["max_margin_step_margin"]
        extra = ""
    else:
        progress = stats["predictive_H10_progress"]
        clearance = stats["predictive_min_clearance"]
        step_margin = stats["predictive_step_margin"]
        extra = rf"\n$\sigma\text{{-rank}}={stats['predictive_sigma_rank']}/32$"
    axis.text(
        0.975, 0.025,
        (
            rf"$B^+={stats['exact_positive_B32']}/32$" "\n"
            rf"$p_{{10}}={progress:.2f}\,\mathrm{{m}}$" "\n"
            rf"$c_{{\min}}={clearance:.2f}\,\mathrm{{m}}$" "\n"
            rf"$m_{{\rm step}}={step_margin:.2f}$"
            + extra
        ),
        transform=axis.transAxes, ha="right", va="bottom", fontsize=8.3,
        color=STYLE.EXECUTED_BLACK, zorder=100,
        bbox={
            "boxstyle": "round,pad=0.24", "facecolor": "white",
            "edgecolor": "none", "alpha": 0.78,
        },
    )


def _montage(
    selected: list[tuple[dict, dict]],
    output: Path,
    *,
    bounds,
    histories: dict[str, np.ndarray],
) -> None:
    figure, axes = plt.subplots(
        len(selected), 2, figsize=(11.8, 5.45 * len(selected)), squeeze=False,
    )
    for row_index, (event, stats) in enumerate(selected):
        _draw_case_axis(
            axes[row_index, 0], event, stats, selector="max_margin",
            phase="selected", bounds=bounds,
            history=histories[stats["event_id"]],
        )
        _draw_case_axis(
            axes[row_index, 1], event, stats, selector="predictive",
            phase="selected", bounds=bounds,
            history=histories[stats["event_id"]],
        )
    figure.legend(
        handles=[
            Line2D([], [], color=STYLE.WEIGHTED_GOLD,
                   lw=STYLE.SAMPLE_LW, label="max-step-margin reference"),
            Line2D([], [], color=STYLE.POSITIVE_BLUE,
                   lw=STYLE.SAMPLE_LW, label="predictive-progress selection"),
        ],
        loc="lower center", ncol=2, frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.025, 1.0, 1.0), pad=0.25)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_cases(
    trace: dict,
    output: str | Path,
    *,
    count: int = 5,
    fps: int = 2,
    max_nearest_clearance: float = 0.5,
    min_exact_positive: int = 4,
    min_progress_gain: float = 0.1,
    bounds=None,
) -> dict:
    """Render declared near-pedestrian selector-disagreement cases."""
    STYLE.apply_computer_modern_style()
    selected = select_case_events(
        trace["events"], count=count,
        max_nearest_clearance=max_nearest_clearance,
        min_exact_positive=min_exact_positive,
        min_progress_gain=min_progress_gain,
    )
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    montage = output.with_name(f"{output.stem}_montage.png")
    if montage.exists():
        raise FileExistsError(montage)
    histories = {}
    for event, stats in selected:
        prior = sorted(
            (
                row for row in trace["events"]
                if str(row["lineage"]) == str(event["lineage"])
                and int(row["step"]) <= int(event["step"])
            ),
            key=lambda row: int(row["step"]),
        )
        histories[stats["event_id"]] = np.asarray([
            np.asarray(row["state_before"], float)[:2] for row in prior
        ])
    _montage(selected, montage, bounds=bounds, histories=histories)

    phases = ["K", "B", "selected", "execute"]
    frames = [
        (event, stats, phase)
        for event, stats in selected
        for phase in phases
    ]
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 5.8), squeeze=False)
    axes = axes[0]
    figure.legend(
        handles=[
            Line2D([], [], color=STYLE.WEIGHTED_GOLD,
                   lw=STYLE.SAMPLE_LW, label="max-step-margin reference"),
            Line2D([], [], color=STYLE.POSITIVE_BLUE,
                   lw=STYLE.SAMPLE_LW, label="predictive-progress selection"),
        ],
        loc="lower center", ncol=2, frameon=False,
    )

    def update(payload):
        event, stats, phase = payload
        _draw_case_axis(
            axes[0], event, stats, selector="max_margin",
            phase=phase, bounds=bounds,
            history=histories[stats["event_id"]],
        )
        _draw_case_axis(
            axes[1], event, stats, selector="predictive",
            phase=phase, bounds=bounds,
            history=histories[stats["event_id"]],
        )
        figure.tight_layout(rect=(0.0, 0.055, 1.0, 1.0), pad=0.25)
        return []

    movie = animation.FuncAnimation(
        figure, update, frames=frames, interval=1000 / int(fps), blit=False,
    )
    movie.save(output, writer=FINAL._writer(fps, bitrate=6200), dpi=130)
    plt.close(figure)
    criteria = {
        "count": int(count),
        "max_nearest_pedestrian_clearance": float(max_nearest_clearance),
        "min_exact_positive_B32": int(min_exact_positive),
        "min_predictive_minus_max_margin_H10_progress": float(min_progress_gain),
        "requires_selector_disagreement": True,
        "requires_nonnegative_predictive_min_clearance": True,
        "diversity": "strongest eligible event per represented gamma, then global rank",
    }
    return _finalize(output, {
        "kind": "predictive_vs_max_margin_mechanism_cases",
        "scientific_K": 64,
        "scientific_B": 32,
        "selection_criteria": criteria,
        "selected_cases": [stats for _, stats in selected],
        "phases_per_case": phases,
        "fixed_camera": (
            list(map(float, bounds)) if bounds is not None
            else [float(SS.TASK_LO), float(SS.TASK_HI)]
        ),
        "montage": str(montage),
        "montage_sha256": STYLE.sha256_file(montage),
    })


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    sub = value.add_subparsers(dest="command", required=True)
    rollout = sub.add_parser("rollout")
    rollout.add_argument("--trace", required=True)
    rollout.add_argument("--lineage", required=True)
    rollout.add_argument("--output", required=True)
    rollout.add_argument("--fps", type=int, default=7)
    rollout.add_argument("--frame-stride", type=int, default=1)
    rollout.add_argument("--show-safety-badge", action="store_true")
    rollout.add_argument("--bounds", type=float, nargs="+")
    cases = sub.add_parser("cases")
    cases.add_argument("--trace", required=True)
    cases.add_argument("--output", required=True)
    cases.add_argument("--count", type=int, choices=(4, 5), default=5)
    cases.add_argument("--fps", type=int, default=2)
    cases.add_argument("--max-nearest-clearance", type=float, default=0.5)
    cases.add_argument("--min-exact-positive", type=int, default=4)
    cases.add_argument("--min-progress-gain", type=float, default=0.1)
    cases.add_argument("--bounds", type=float, nargs="+")
    grid = sub.add_parser("grid")
    grid.add_argument("--trace", required=True)
    grid.add_argument("--output", required=True)
    grid.add_argument("--gammas", type=float, nargs="+", default=(0.1, 0.5, 1.0))
    grid.add_argument("--replicas", type=int, nargs="+", default=(0, 1))
    grid.add_argument("--fps", type=int, default=5)
    grid.add_argument("--frame-stride", type=int, default=1)
    grid.add_argument("--bounds", type=float, nargs="+")
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    trace = _load(args.trace)
    if args.command == "rollout":
        result = render_rollout(
            trace, args.lineage, args.output, fps=args.fps,
            frame_stride=args.frame_stride,
            show_safety_badge=args.show_safety_badge, bounds=args.bounds,
        )
    elif args.command == "cases":
        result = render_cases(
            trace, args.output, count=args.count, fps=args.fps,
            max_nearest_clearance=args.max_nearest_clearance,
            min_exact_positive=args.min_exact_positive,
            min_progress_gain=args.min_progress_gain, bounds=args.bounds,
        )
    else:
        result = render_acquisition_grid(
            trace, args.output, gammas=tuple(args.gammas),
            replicas=tuple(args.replicas), fps=args.fps,
            frame_stride=args.frame_stride, bounds=args.bounds,
        )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
