"""Render a 4-gamma x 4-lineage exhaustive-hybrid qualification trace."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

import _paths  # noqa: F401
import sfm_b1_viz as BV
import sfm_hp100_branch_viz as RAW_BRANCH
import sfm_hp100_exhaustive_hybrid as HYBRID
import sfm_scene as SS


VERSION = "sfm_hp100_exhaustive_hybrid_viz_v1"
STATUS = "SFM_HP100_EXHAUSTIVE_HYBRID_VIZ_COMPLETE"
P1_BLUE = "#0057FF"
DMINUS_RED = "#FF1F1F"
QUERY_GREEN = "#00A651"
P2_CYAN = "#00B7C7"
NCAUSAL_MAGENTA = "#D000C8"
BLACK = "#111111"
GRAY = "#777777"
CLEAR_GREEN = "#148A2A"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_trace(path: str | Path) -> dict:
    trace = torch.load(path, map_location="cpu", weights_only=False)
    validate_trace(trace)
    return trace


def validate_trace(trace: dict) -> None:
    if trace.get("version") != HYBRID.TRACE_VERSION:
        raise ValueError("not an exhaustive-hybrid trace")
    config = trace.get("config", {})
    if tuple(map(float, config.get("gammas", ()))) != HYBRID.DEFAULT_GAMMAS:
        raise ValueError("the paper 4x4 renderer requires gamma .1/.3/.5/1")
    if int(config.get("lineages_per_gamma", -1)) != 4:
        raise ValueError("the paper 4x4 renderer requires four lineages/gamma")
    if (int(config.get("K", -1)), int(config.get("B", -1))) != (64, 16):
        raise ValueError("trace K/B contract differs from exhaustive-hybrid")
    events = trace.get("events", ())
    if not events:
        raise ValueError("trace has no gathering events")
    valid_labels = {
        f"g{gamma:.9g}:rep{replica:02d}"
        for gamma in HYBRID.DEFAULT_GAMMAS for replica in range(4)
    }
    for event in events:
        if event.get("lineage") not in valid_labels:
            raise ValueError("trace contains an undeclared 4x4 lineage")
        if np.asarray(event["raw_segment"]).shape != (11, 2):
            raise ValueError("raw branch must retain H10 positions")
        raw_valid = bool(event["raw_verification"]["valid"])
        repair = list(event.get("repair_attempts", ()))
        group = event.get("executed_group")
        if raw_valid and (repair or group != "P1"):
            raise ValueError("exact-positive raw proposal must be direct P1")
        if not raw_valid and group == "P1":
            raise ValueError("raw exact-negative cannot be P1")
        if group == "P2":
            if not repair or repair[-1].get("selected_local") is None:
                raise ValueError("P2 lacks its selected repair B branch")
        for attempt, row in enumerate(repair):
            expected_std = 1.0 + 0.1 * attempt
            if not math.isclose(float(row["base_std"]), expected_std, abs_tol=1e-9):
                raise ValueError("repair temperature/base-std schedule drifted")
            if np.asarray(row["segments"]).shape != (16, 11, 2):
                raise ValueError("repair attempt does not retain B=16 branches")


def _group_events(trace: dict) -> dict[tuple[int, str], list[dict]]:
    grouped = {}
    for event in trace["events"]:
        key = (int(event["microcycle"]), str(event["lineage"]))
        grouped.setdefault(key, []).append(event)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["step"]))
    return grouped


def _recheck_status(trace: dict) -> dict[tuple[int, str], bool]:
    output = {}
    for cycle in trace.get("cycles", ()):
        recheck = cycle.get("recheck")
        if not recheck:
            continue
        microcycle = int(cycle["microcycle"])
        for label, row in recheck.get("outcomes", {}).items():
            output[(microcycle, str(label))] = bool(row.get("clear", False))
    return output


def _frames(trace: dict) -> list[tuple[int, int, int | None, bool]]:
    grouped = _group_events(trace)
    rechecked_cycles = {
        int(cycle["microcycle"])
        for cycle in trace.get("cycles", ()) if cycle.get("recheck") is not None
    }
    frames = []
    for microcycle in sorted({key[0] for key in grouped}):
        maximum = max(
            (int(row[-1]["step"]) for key, row in grouped.items()
             if key[0] == microcycle),
            default=0,
        )
        for step in range(maximum + 1):
            # Raw proposal is shown before any conditional repair is revealed.
            frames.append((microcycle, step, None, False))
            repair_count = max((
                len(event.get("repair_attempts", ()))
                for (cycle, _), rows in grouped.items() if cycle == microcycle
                for event in rows if int(event["step"]) == step
            ), default=0)
            frames.extend(
                (microcycle, step, attempt, False)
                for attempt in range(repair_count)
            )
        # One explicit post-update/recheck frame keeps the CLEAR badge
        # temporally honest: it cannot appear during the gathering that
        # precedes the fixed raw-only development gate.
        if microcycle in rechecked_cycles:
            frames.append((microcycle, maximum, None, True))
    return frames


def _draw_scene(axis, event: dict) -> None:
    ped_xy = np.asarray(event["ped_xy"], float)
    ped_vel = np.asarray(event["ped_vel"], float)
    BV._draw_pedestrians(axis, ped_xy, alpha=.67)
    for index in range(len(ped_xy)):
        prediction = ped_xy[index][None] + (
            np.arange(11)[:, None] * SS.DT * ped_vel[index][None]
        )
        axis.plot(
            prediction[:, 0], prediction[:, 1], ".--", color=GRAY,
            lw=.28, ms=.75, alpha=.24, zorder=1,
        )
    axis.plot(
        SS.GOAL[0], SS.GOAL[1], marker="*", ms=7,
        color="#F0A202", mec="#7A4E00", mew=.4, zorder=12,
    )
    BV._set_world_frame(axis)
    axis.set_xlabel(""); axis.set_ylabel("")
    axis.tick_params(labelbottom=False, labelleft=False, length=0)


def _green_geometry(gamma: float, sidecar: dict) -> dict:
    result = dict(sidecar)
    result["faces"] = [SimpleNamespace(**row) for row in result["faces"]]
    return RAW_BRANCH.verifier_geometry({
        "gamma": float(gamma), "proposal_result": result,
    })


def _draw_green(axis, gamma: float, sidecar: dict) -> None:
    geometry = _green_geometry(gamma, sidecar)
    BV._draw_level_polygons(
        axis, geometry["levels"], color=QUERY_GREEN,
        linewidth=.22, alpha=.30, zorder=2,
    )
    outer = np.asarray(geometry["outer"], float)
    axis.plot(
        np.r_[outer[:, 0], outer[0, 0]],
        np.r_[outer[:, 1], outer[0, 1]],
        color=QUERY_GREEN, lw=.55, alpha=.55, zorder=2.2,
    )


def _latest_event(
    grouped: dict[tuple[int, str], list[dict]],
    lineage: str,
    microcycle: int,
    step: int,
) -> tuple[int, list[dict], dict] | None:
    available = sorted(
        cycle for cycle, label in grouped if label == lineage and cycle <= microcycle
    )
    if not available:
        return None
    selected_cycle = available[-1]
    rows = grouped[(selected_cycle, lineage)]
    if selected_cycle == microcycle:
        visible = [row for row in rows if int(row["step"]) <= step]
        if not visible:
            return None
    else:
        visible = rows
    return selected_cycle, visible, visible[-1]


def draw_cell(
    axis,
    grouped: dict[tuple[int, str], list[dict]],
    recheck_status: dict[tuple[int, str], bool],
    *,
    gamma: float,
    replica: int,
    microcycle: int,
    step: int,
    repair_attempt_index: int | None,
    post_recheck: bool,
    show_causal_negative: bool = False,
) -> None:
    lineage = f"g{gamma:.9g}:rep{replica:02d}"
    selected = _latest_event(grouped, lineage, microcycle, step)
    if selected is None:
        BV._set_world_frame(axis)
        axis.tick_params(labelbottom=False, labelleft=False, length=0)
        axis.text(.5, .5, "waiting", transform=axis.transAxes,
                  ha="center", va="center", color=GRAY, fontsize=6)
        return
    selected_cycle, visible, current = selected
    _draw_scene(axis, current)
    positions = [np.asarray(row["state_before"], float)[:2] for row in visible]
    positions.append(np.asarray(current["state_after"], float)[:2])
    positions = np.asarray(positions)
    axis.plot(
        positions[:, 0], positions[:, 1], color=BLACK,
        lw=1.05, marker=".", ms=.8, zorder=8,
    )
    raw = np.asarray(current["raw_segment"], float)
    if current["raw_verification"]["valid"]:
        axis.plot(raw[:, 0], raw[:, 1], color=P1_BLUE,
                  lw=2.0, marker=".", ms=1.7, zorder=9)
        _draw_green(axis, gamma, current["raw_sidecar"])
    else:
        axis.plot(raw[:, 0], raw[:, 1], color=DMINUS_RED,
                  lw=1.8, marker=".", ms=1.4, zorder=8.5)
        axis.plot(raw[-1, 0], raw[-1, 1], "x", color=DMINUS_RED,
                  ms=5, mew=1.3, zorder=10)
        attempts = current.get("repair_attempts", ())
        if attempts and repair_attempt_index is not None:
            attempt = attempts[min(int(repair_attempt_index), len(attempts) - 1)]
            for segment in np.asarray(attempt["segments"], float):
                axis.plot(segment[:, 0], segment[:, 1], color=QUERY_GREEN,
                          lw=.55, alpha=.48, zorder=4)
            chosen = attempt.get("selected_local")
            if chosen is not None:
                segment = np.asarray(attempt["segments"], float)[int(chosen)]
                axis.plot(segment[:, 0], segment[:, 1], color=P2_CYAN,
                          lw=2.2, marker=".", ms=1.8, zorder=10)
                _draw_green(axis, gamma, attempt["selected_sidecar"])
            axis.text(
                .02, .02,
                f"repair q={int(attempt['attempt'])} · T={float(attempt['base_std']):.1f}",
                transform=axis.transAxes, ha="left", va="bottom",
                fontsize=5.4, color="#075F65",
            )
    if show_causal_negative:
        for event in visible:
            if not bool(event.get("causal_negative", False)):
                continue
            group = event.get("executed_group")
            if group == "P1":
                segment = np.asarray(event["raw_segment"], float)
            elif group == "P2":
                selected_attempts = [
                    attempt for attempt in event.get("repair_attempts", ())
                    if attempt.get("selected_local") is not None
                ]
                if not selected_attempts:
                    raise ValueError("Ncausal P2 event lacks its selected branch")
                attempt = selected_attempts[-1]
                segment = np.asarray(attempt["segments"], float)[
                    int(attempt["selected_local"])
                ]
            else:
                raise ValueError("Ncausal overlay requires executed P1/P2")
            axis.plot(
                segment[:, 0], segment[:, 1], color=NCAUSAL_MAGENTA,
                lw=2.7, marker=".", ms=1.9, alpha=.92, zorder=11,
            )
    status_cycle = microcycle if post_recheck else microcycle - 1
    is_clear = recheck_status.get((status_cycle, lineage), False)
    if is_clear:
        axis.text(
            .98, .97, f"CLEAR μ={status_cycle}", transform=axis.transAxes,
            ha="right", va="top", fontsize=6.2, fontweight="bold",
            color="white", bbox=dict(
                boxstyle="round,pad=.24", facecolor=CLEAR_GREEN,
                edgecolor="none", alpha=.93,
            ), zorder=20,
        )
    terminal = current.get("terminal")
    if terminal:
        terminal_color = CLEAR_GREEN if str(terminal).lower() == "success" else DMINUS_RED
        axis.text(
            .02, .97, str(terminal).upper(), transform=axis.transAxes,
            ha="left", va="top", fontsize=5.5, color=terminal_color,
        )
    axis.text(
        .5, 1.01, f"γ={gamma:g} · lineage {replica + 1}",
        transform=axis.transAxes, ha="center", va="bottom", fontsize=6.2,
    )


def render(
    trace: dict,
    *,
    output_mp4: str | Path,
    output_png: str | Path,
    fps: int = 10,
    show_causal_negative: bool = False,
) -> dict:
    if int(fps) <= 0:
        raise ValueError("fps must be positive")
    grouped = _group_events(trace)
    recheck_status = _recheck_status(trace)
    frames = _frames(trace)
    figure, axes = plt.subplots(4, 4, figsize=(13.1, 12.8))
    figure.subplots_adjust(
        left=.025, right=.87, bottom=.025, top=.94,
        wspace=.02, hspace=.08,
    )
    legend_handles = [
            Line2D([], [], color=P1_BLUE, lw=2.2,
                   label="P1 · raw temp-1 exact-positive (executed)"),
            Line2D([], [], color=DMINUS_RED, lw=2.0, marker="x",
                   label="D− · raw temp-1 exact-negative (not executed)"),
            Line2D([], [], color=QUERY_GREEN, lw=.8,
                   label="repair B=16 queried branches"),
            Line2D([], [], color=P2_CYAN, lw=2.3,
                   label="P2 · max-step-margin exact-positive repair"),
            Line2D([], [], color=BLACK, lw=1.2,
                   label="closed-loop first-action trajectory"),
    ]
    if show_causal_negative:
        legend_handles.insert(
            4,
            Line2D([], [], color=NCAUSAL_MAGENTA, lw=2.7,
                   label="Ncausal · executed prefix before repair exhaustion"),
        )
    figure.legend(
        handles=legend_handles,
        loc="center left", bbox_to_anchor=(.875, .69), frameon=False,
        fontsize=7.2,
    )
    figure.text(
        .875, .46,
        "Raw first. Repair only on red.\n"
        "Retry keeps context fixed.\n"
        "T(q)=1+0.1q.\n\n"
        "CLEAR: fixed raw-only\n"
        "temperature-1 success with\n"
        "every executed H10 certified.\n\n"
        "Round-1 GP reference:\n"
        "gamma-matched certified\n"
        "pretraining embeddings.\n"
        "New rows commit only on CLEAR.",
        ha="left", va="top", fontsize=7.1,
    )

    def update(frame):
        microcycle, step, repair_attempt_index, post_recheck = frame
        for row, gamma in enumerate(HYBRID.DEFAULT_GAMMAS):
            for replica in range(4):
                axis = axes[row, replica]
                axis.clear()
                draw_cell(
                    axis, grouped, recheck_status, gamma=gamma,
                    replica=replica, microcycle=microcycle, step=step,
                    repair_attempt_index=repair_attempt_index,
                    post_recheck=post_recheck,
                    show_causal_negative=show_causal_negative,
                )
        phase = (
            "fixed raw development gate" if post_recheck
            else (
                f"repair q={repair_attempt_index}"
                if repair_attempt_index is not None else f"raw gather step {step}"
            )
        )
        figure.suptitle(
            "HP100 exhaustive hybrid · "
            f"microcycle {microcycle} · {phase}",
            fontsize=11,
        )
        return []

    output_mp4 = Path(output_mp4).resolve()
    output_png = Path(output_png).resolve()
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    movie = animation.FuncAnimation(
        figure, update, frames=frames, interval=1000 / int(fps), blit=False,
    )
    movie.save(
        output_mp4, writer=animation.FFMpegWriter(fps=int(fps), bitrate=5000),
        dpi=105,
    )
    update(frames[-1])
    figure.savefig(output_png, dpi=170, bbox_inches="tight")
    plt.close(figure)
    result = {
        "frames": len(frames), "fps": int(fps),
        "mp4": str(output_mp4), "mp4_sha256": _sha256(output_mp4),
        "png": str(output_png), "png_sha256": _sha256(output_png),
        "recheck_status": {
            f"{cycle}:{lineage}": clear
            for (cycle, lineage), clear in sorted(recheck_status.items())
        },
    }
    if show_causal_negative:
        result["causal_negative_overlay"] = True
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--trace", required=True)
    value.add_argument("--output-dir", required=True)
    value.add_argument("--fps", type=int, default=10)
    value.add_argument("--show-causal-negative", action="store_true")
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing viz output: {output}")
    output.mkdir(parents=True)
    trace = load_trace(args.trace)
    result = render(
        trace,
        output_mp4=output / "exhaustive_hybrid_4x4.mp4",
        output_png=output / "exhaustive_hybrid_4x4_final.png",
        fps=args.fps,
        show_causal_negative=bool(args.show_causal_negative),
    )
    marker = {
        "status": STATUS, "version": VERSION,
        "trace": str(Path(args.trace).resolve()),
        "trace_sha256": _sha256(args.trace), "render": result,
    }
    marker_path = output / "VIZ_COMPLETE.json"
    marker_path.write_text(json.dumps(marker, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": STATUS, "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
