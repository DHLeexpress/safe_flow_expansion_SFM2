"""Render the compact HP100 early-acquisition branch trace without rerunning.

The trace is intentionally narrow: one paired scenario, replica zero, and
gamma 0.1/0.5/1.0.  Green denotes all and only the queried B branches.  Blue
is the exact-positive plan whose first action was executed.  At terminal NVP,
red is the resolved-negative counterfactual archived for Dminus; it was not
executed.  The thin black polyline is the actual closed-loop state path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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
import sfm_scene as SS


VERSION = "sfm_hp100_early_branch_viz_v3"
STATUS = "SFM_HP100_EARLY_BRANCH_VIZ_COMPLETE"
TRACE_STATUS = "SFM_HP100_EARLY_BRANCH_TRACE_COMPLETE"
TRACE_VERSION = "sfm_hp100_early_branch_trace_v2"
GAMMAS = (0.1, 0.5, 1.0)
GREEN = "#00A651"
BLUE = "#0000FF"
RED = "#FF0000"
BLACK = "#111111"
GRAY = "#777777"
REFERENCE = "#7A5195"


def sha256_file(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trace(path: str | os.PathLike) -> dict:
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    validate_trace(bundle)
    return bundle


def _same(value: float, expected: float) -> bool:
    return math.isclose(float(value), float(expected), rel_tol=0.0, abs_tol=1.0e-8)


def validate_trace(bundle: dict) -> None:
    if bundle.get("status") != TRACE_STATUS or bundle.get("version") != TRACE_VERSION:
        raise ValueError("not a completed HP100 early-branch trace")
    config = bundle.get("config", {})
    expected = {
        "K": 64, "B": 16,
        "execution_rule": "weighted_cost", "parallel_episodes_per_gamma": 16,
    }
    for key, value in expected.items():
        actual = config.get(key)
        if isinstance(value, float):
            matches = actual is not None and _same(actual, value)
        else:
            matches = actual == value
        if not matches:
            raise ValueError(f"trace {key}={actual!r}, expected canonical {value!r}")
    max_steps = int(config.get("max_steps", 0))
    if max_steps < 30:
        raise ValueError("trace max_steps must retain at least the first 30 steps")
    if not math.isfinite(float(config.get("flow_base_std", float("nan")))) \
            or float(config["flow_base_std"]) <= 0.0:
        raise ValueError("trace flow_base_std must be finite and positive")
    if not math.isfinite(float(config.get(
        "execution_step_margin_weight", float("nan")
    ))) or float(config["execution_step_margin_weight"]) < 0.0:
        raise ValueError("trace step-margin weight must be finite and nonnegative")
    declared = bundle.get("lineage_filter", {})
    if tuple(map(float, declared.get("gammas", ()))) != GAMMAS:
        raise ValueError("trace must contain gamma 0.1/0.5/1.0")
    declared_replica = int(declared.get("replica", -1))
    parallel = int(config.get("parallel_episodes_per_gamma", 0))
    if not 0 <= declared_replica < parallel:
        raise ValueError("trace declares an out-of-range paired replica")

    events = bundle.get("events", [])
    grouped = {gamma: [] for gamma in GAMMAS}
    for event in events:
        gamma = next((value for value in GAMMAS if _same(event["gamma"], value)), None)
        if gamma is None:
            raise ValueError("trace contains an undeclared gamma")
        if int(event["replica"]) != declared_replica:
            raise ValueError("trace event differs from its declared paired replica")
        segments = np.asarray(event["queried_segments"])
        controls = np.asarray(event["queried_controls"])
        if segments.shape != (16, 11, 2) or controls.shape != (16, 10, 2):
            raise ValueError("trace does not retain exactly B=16 H10 branches")
        if len(event["verification"]) != 16:
            raise ValueError("trace verifier rows differ from B=16")
        chosen = event.get("chosen_local")
        reference = event.get("performance_reference_local")
        negative = event.get("archived_negative_local")
        if reference is not None:
            reference_row = event["verification"][int(reference)]
            if reference_row["error"] or not reference_row["valid"]:
                raise ValueError("performance reference is not an exact-positive query")
        if chosen is not None:
            sidecar = event.get("chosen_verifier_sidecar")
            if negative is not None or not event["verification"][int(chosen)]["valid"]:
                raise ValueError("blue branch is not the sole exact-positive execution")
            if sidecar is None:
                raise ValueError("blue branch lacks its exact GREEN verifier sidecar")
            rank = event.get("chosen_H10_progress_rank")
            eligible = event.get("eligible_progress_candidates")
            if rank is not None and not 1 <= int(rank) <= int(eligible):
                raise ValueError("blue branch has an invalid declared progress rank")
            if (
                not sidecar.get("resolved") or int(sidecar.get("y", 0)) != 1
                or not sidecar.get("full_h")
                or sidecar.get("diagnostics", {}).get("solver")
                != "paper_static_exact_2d_angular_interval_socp"
                or int(sidecar.get("diagnostics", {}).get("K_artificial", -1)) != 16
            ):
                raise ValueError("blue GREEN sidecar does not use the exact paper solver")
            if np.asarray(sidecar.get("pedestrian_prediction")).ndim != 3:
                raise ValueError("blue GREEN sidecar lacks its pedestrian prediction")
            artificial = [
                face for face in sidecar.get("faces", [])
                if face.get("kind") == "artificial" and bool(face.get("feasible"))
            ]
            if len(artificial) != 16:
                raise ValueError("blue GREEN sidecar does not retain 16 artificial faces")
            if not np.array_equal(
                np.asarray(sidecar["segment"], np.float32),
                segments[int(chosen)].astype(np.float32, copy=False),
            ):
                raise ValueError("blue GREEN sidecar segment differs from the query")
        else:
            if event.get("status") != "nvp" or negative is None:
                raise ValueError("NVP event lacks its archived counterfactual Dminus branch")
            if event.get("chosen_verifier_sidecar") is not None:
                raise ValueError("red NVP counterfactual must not carry a fake GREEN polytope")
            row = event["verification"][int(negative)]
            if row["error"] or row["valid"]:
                raise ValueError("red branch is not a resolved exact negative")
        grouped[gamma].append(event)

    scenario_ids = set()
    for gamma, rows in grouped.items():
        if not rows:
            raise ValueError(f"trace has no events for gamma {gamma:g}")
        rows.sort(key=lambda row: int(row["step"]))
        if [int(row["step"]) for row in rows] != list(range(len(rows))):
            raise ValueError("trace steps are not contiguous from zero")
        for previous, current in zip(rows, rows[1:]):
            if not np.array_equal(previous["state_after"], current["state_before"]):
                raise ValueError("black closed-loop path is discontinuous")
        terminal = rows[-1].get("status")
        if terminal not in {"nvp", "success", "collision", "oob", "EARLY_CUTOFF"}:
            raise ValueError(f"unexpected final trace status {terminal!r}")
        scenario_ids.add(int(rows[0]["scenario_id"]))
    if len(scenario_ids) != 1:
        raise ValueError("three gamma columns do not share one paired scenario")


def grouped_events(bundle: dict) -> dict[float, list[dict]]:
    output = {gamma: [] for gamma in GAMMAS}
    for event in bundle["events"]:
        gamma = next(value for value in GAMMAS if _same(event["gamma"], value))
        output[gamma].append(event)
    for rows in output.values():
        rows.sort(key=lambda row: int(row["step"]))
    return output


def _draw_scene(axis, event: dict) -> None:
    ped_xy = np.asarray(event["ped_xy"], float)
    ped_vel = np.asarray(event["ped_vel"], float)
    BV._draw_pedestrians(axis, ped_xy, alpha=.74)
    for index in range(len(ped_xy)):
        prediction = ped_xy[index][None] + (
            np.arange(11)[:, None] * SS.DT * ped_vel[index][None]
        )
        axis.plot(
            prediction[:, 0], prediction[:, 1], ".--", color=GRAY,
            lw=.42, ms=1.25, alpha=.38, zorder=1,
        )
    axis.plot(
        SS.GOAL[0], SS.GOAL[1], marker="*", ms=10,
        color="#F0A202", mec="#7A4E00", mew=.5, zorder=12,
    )
    BV._set_world_frame(axis)
    axis.set_xlabel(""); axis.set_ylabel("")
    axis.tick_params(labelbottom=False, labelleft=False, length=0)


def chosen_green_geometry(event: dict) -> dict:
    """Adapt an authenticated serialized sidecar to the existing GREEN renderer."""
    sidecar = event.get("chosen_verifier_sidecar")
    if event.get("chosen_local") is None or sidecar is None:
        raise ValueError("GREEN geometry exists only for a chosen exact-positive branch")
    proposal_result = dict(sidecar)
    proposal_result["faces"] = [
        SimpleNamespace(**face) for face in sidecar["faces"]
    ]
    return RAW_BRANCH.verifier_geometry({
        "gamma": float(event["gamma"]), "proposal_result": proposal_result,
    })


def draw_lineage(axis, rows: list[dict], frame: int) -> dict:
    current_index = min(max(int(frame), 0), len(rows) - 1)
    current = rows[current_index]
    _draw_scene(axis, current)

    if current.get("chosen_local") is not None:
        # Use the same exact K=16 outer set and h=1..10 drawing contract as the
        # established raw HP100 branch audit.  NVP/red gets no surrogate set.
        RAW_BRANCH._draw_green(axis, chosen_green_geometry(current))

    # Green is the query mechanism, not the verifier label.  Plot all B first,
    # then overlay exactly one blue execution or one terminal red Dminus row.
    for segment in np.asarray(current["queried_segments"], float):
        axis.plot(
            segment[:, 0], segment[:, 1], color=GREEN, lw=1.7,
            marker=".", ms=1.8, alpha=.32, zorder=4,
        )
    chosen = current.get("chosen_local")
    reference = current.get("performance_reference_local")
    archived_negative = current.get("archived_negative_local")
    if reference is not None:
        segment = np.asarray(current["queried_segments"][int(reference)], float)
        axis.plot(
            segment[:, 0], segment[:, 1], color=REFERENCE, lw=2.2,
            ls="--", marker=".", ms=2.0, alpha=.88, zorder=7,
        )
    if chosen is not None:
        segment = np.asarray(current["queried_segments"][int(chosen)], float)
        axis.plot(
            segment[:, 0], segment[:, 1], color=BLUE, lw=3.4,
            marker=".", ms=2.5, alpha=.98, zorder=8,
        )
    elif archived_negative is not None:
        segment = np.asarray(
            current["queried_segments"][int(archived_negative)], float,
        )
        axis.plot(
            segment[:, 0], segment[:, 1], color=RED, lw=3.4,
            marker=".", ms=2.5, alpha=.98, zorder=8,
        )
        axis.plot(
            segment[-1, 0], segment[-1, 1], marker="x", color=RED,
            ms=7.0, mew=1.8, zorder=10,
        )

    path = [np.asarray(row["state_before"], float)[:2]
            for row in rows[:current_index + 1]]
    if chosen is not None:
        path.append(np.asarray(current["state_after"], float)[:2])
    path = np.asarray(path)
    axis.plot(
        path[:, 0], path[:, 1], color=BLACK, lw=1.05,
        marker=".", ms=1.8, alpha=.98, zorder=11,
    )
    if chosen is not None:
        chosen_row = current["verification"][int(chosen)]
        reference_row = (
            None if reference is None
            else current["verification"][int(reference)]
        )
        one_step = float(
            np.linalg.norm(np.asarray(current["state_before"], float)[:2] - SS.GOAL)
            - np.linalg.norm(np.asarray(current["state_after"], float)[:2] - SS.GOAL)
        )
        net = float(
            np.linalg.norm(np.asarray(rows[0]["state_before"], float)[:2] - SS.GOAL)
            - np.linalg.norm(np.asarray(current["state_after"], float)[:2] - SS.GOAL)
        )
        delta = (
            None if reference_row is None
            else float(chosen_row["progress"] - reference_row["progress"])
        )
        annotation = (
            f"H10={float(chosen_row['progress']):+.2f} m"
            + ("" if delta is None else f"  Δvs λ0={delta:+.2f} m")
            + f"\n1-step={one_step:+.3f} m  net={net:+.2f} m"
        )
        rank = current.get("chosen_H10_progress_rank")
        eligible = current.get("eligible_progress_candidates")
        if rank is not None and eligible is not None:
            annotation += f"  rank={int(rank)}/{int(eligible)}"
        axis.text(
            .02, .02, annotation, transform=axis.transAxes,
            ha="left", va="bottom", fontsize=7.2,
            bbox=dict(facecolor="white", edgecolor="none", alpha=.72, pad=2.0),
            zorder=20,
        )
    axis.set_title("")
    return current


def render_trace(
    bundle: dict,
    *,
    output_mp4: str,
    output_png: str,
    fps: int = 5,
    frame_stride: int = 1,
    model_label: str = "",
) -> dict:
    validate_trace(bundle)
    if int(fps) < 1 or int(frame_stride) < 1:
        raise ValueError("fps and frame_stride must be positive")
    rows = grouped_events(bundle)
    final_frame = max(len(value) for value in rows.values()) - 1
    frames = list(range(0, final_frame + 1, int(frame_stride)))
    if frames[-1] != final_frame:
        frames.append(final_frame)
    figure, axes = plt.subplots(1, 3, figsize=(14.4, 5.1))
    figure.subplots_adjust(left=.025, right=.80, bottom=.05, top=.90, wspace=.025)
    figure.legend(
        handles=[
            Line2D([], [], color=GREEN, lw=2.0, label="queried B=16 branches"),
            Line2D([], [], color=REFERENCE, lw=2.2, ls="--",
                   label="λ=0 performance-reference branch"),
            Line2D([], [], color=RAW_BRANCH.GREEN, lw=1.2,
                   label="chosen branch: exact GREEN outer set + h=1..10"),
            Line2D([], [], color=BLUE, lw=3.4,
                   label="chosen exact-positive plan; first action executed"),
            Line2D([], [], color=RED, lw=3.4, marker="x",
                   label="terminal resolved-negative D− counterfactual"),
            Line2D([], [], color=BLACK, lw=1.1,
                   label="actual closed-loop executed-state path"),
        ],
        loc="center left", bbox_to_anchor=(.805, .61), frameon=False, fontsize=8,
    )
    pooled = bundle.get("summary", {}).get("pooled", {})
    aggregate = ""
    if pooled:
        progress = pooled.get("goal_progress", {})
        aggregate = (
            f"\n\nS30={float(pooled['S30']):.3f} · "
            f"RMST@40={float(pooled['early_gather_RMST_to_max_steps']):.2f}\n"
            f"net@30={float(pooled['lineage_macro_net_goal_progress_at_30']):.2f} m · "
            f"mean rank={float(progress['chosen_H10_progress_rank_mean']):.2f}"
        )
    figure.text(
        .805, .36,
        "Red is never executed.\n"
        "It is archived only after NVP, when all B queries are negative.\n\n"
        "K=64 · B=16 · "
        f"σbase={float(bundle['config']['flow_base_std']):g} · "
        f"T_diagnostic={int(bundle['config']['max_steps'])} "
        "(selection prefix=30)\n"
        "selector: J_SafeMPPI − "
        f"{float(bundle['config']['execution_step_margin_weight']):g} · m_step"
        + aggregate,
        ha="left", va="top", fontsize=8,
    )

    def update(frame):
        for axis, gamma in zip(axes, GAMMAS):
            axis.clear()
            event = draw_lineage(axis, rows[gamma], int(frame))
            status = event.get("status") or "active"
            status = {"EARLY_CUTOFF": "cutoff", "nvp": "NVP"}.get(
                status, status,
            )
            axis.text(
                .5, 1.01,
                f"γ={gamma:g} | t={event['step']} | {status}",
                transform=axis.transAxes, ha="center", va="bottom", fontsize=9,
            )
        title = "HP100 early-acquisition executed-window branch audit"
        if model_label:
            title += f" — {model_label}"
        figure.suptitle(title, fontsize=11)
        return []

    for path in (output_mp4, output_png):
        Path(path).resolve().parent.mkdir(parents=True, exist_ok=True)
    movie = animation.FuncAnimation(
        figure, update, frames=frames, interval=1000 / int(fps), blit=False,
    )
    movie.save(
        output_mp4, writer=animation.FFMpegWriter(fps=int(fps), bitrate=4800),
        dpi=110,
    )
    update(final_frame)
    figure.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return {
        "frames": frames, "mp4": str(Path(output_mp4).resolve()),
        "png": str(Path(output_png).resolve()),
        "model_label": model_label,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--model-label", default="")
    args = parser.parse_args(argv)
    outdir = Path(args.outdir).resolve()
    if outdir.exists():
        raise FileExistsError(f"restart-only renderer refuses existing outdir: {outdir}")
    outdir.mkdir(parents=True)
    bundle = load_trace(args.trace)
    render = render_trace(
        bundle,
        output_mp4=str(outdir / "early_acquisition_branches.mp4"),
        output_png=str(outdir / "early_acquisition_branches_final.png"),
        fps=args.fps, frame_stride=args.frame_stride,
        model_label=args.model_label,
    )
    artifacts = {
        key: {
            "path": path, "bytes": os.path.getsize(path),
            "sha256": sha256_file(path),
        }
        for key, path in (("mp4", render["mp4"]), ("png", render["png"]))
    }
    marker = {
        "status": STATUS, "version": VERSION,
        "trace": str(Path(args.trace).resolve()),
        "trace_sha256": sha256_file(args.trace),
        "renderer_sha256": sha256_file(__file__),
        "checkpoint_sha256": bundle["checkpoint_sha256"],
        "semantics": bundle["semantics"],
        "render": render, "artifacts": artifacts,
    }
    marker_path = outdir / "EARLY_BRANCH_VIZ_COMPLETE.json"
    temporary = marker_path.with_name(f".{marker_path.name}.tmp")
    temporary.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    temporary.replace(marker_path)
    print(json.dumps({"status": STATUS, "marker": str(marker_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
