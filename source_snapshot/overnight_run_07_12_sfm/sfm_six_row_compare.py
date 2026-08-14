"""Collect and render the six-row SFM mechanism comparison.

The trace bundle is the authority.  MP4 and PDF rendering never reruns a
controller, so a requested frame index can later be reproduced as a vector
snapshot.  Rows are:

1. matched-ID SafeMPPI demonstration expert, stopped at first full-H NVP;
2. double-shift-OOD SafeMPPI demonstration expert, stopped at first NVP;
3. pretrained B1 collection with max one-step nominal-Hp margin execution;
4. the same collection with native SafeMPPI-cost execution;
5. the same collection with balanced safety/performance rank execution;
6. Kazuki guidance applied to the same pretrained policy in OOD.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

import _paths  # noqa: F401
import grid_policy_sfm as GPS
import sfm_b1_d_branch_viz as DB
import sfm_b1_density_viz as DV
import sfm_b1_expert as EX
import sfm_b1_full_episode_audit as FA
import sfm_b1_full_episode_viz as FV
import sfm_b1_viz as BV
import sfm_kazuki as KZ
import sfm_metrics2 as SM
import sfm_protocol as SP
import sfm_scene as SS


STATUS = "SFM_SIX_ROW_TRACE_BUNDLE_COMPLETE"
RENDER_STATUS = "SFM_SIX_ROW_RENDER_COMPLETE"
SELECTORS = ("margin", "safemppi_cost", "balanced_rank")
TRUE_BLUE = "#0057FF"
TRUE_RED = "#D62728"
KAZUKI_PURPLE = "#7F3C8D"
YELLOW = "#F0E442"
DEFAULT_CHECKPOINT_SHA256 = (
    "1b5179c935d3eeff8824967d707d64cc9bab273949ee1f0e4f190172bab1b215"
)
EPISODE_BANKS = {
    "demonstrations": "0–7,999",
    "pretrain_gate": "12,000 onward (declared gate bank)",
    "expansion": "20,000–20,159 for the frozen 20-round, 8-scenario protocol",
    "screen": "50,000 onward",
    "confirmation": "80,000 onward",
    "kazuki_confirmation": "90,000 onward",
    "matched_id_deploy": "150,000 onward",
    "legacy_ood_deploy": "170,000 onward",
    "query_diagnostic": "190,000 onward",
    "density_ood_deploy": "210,000 onward",
    "double_shift_ood_deploy": "250,000 onward",
    "current_raw_m50_eval": "260,000–260,049 per gamma",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, payload):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
    os.replace(temporary, path)


def _save_torch(path, payload):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def resolve_device(requested):
    requested = str(requested)
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def validate_gammas(values):
    values = tuple(map(float, values))
    declared = tuple(map(float, SS.GAMMAS))
    if (
        not values
        or len(set(values)) != len(values)
        or any(value not in declared for value in values)
    ):
        raise ValueError(f"gammas must be a distinct nonempty subset of {declared}")
    return values


def _gate_expert_run(run, gamma):
    """Stop an expert trace before executing its first verifier-negative H10 plan."""
    traces = []
    positive_steps = 0
    nvp_step = None
    for row in run.get("trace") or ():
        row = dict(row)
        result = SM.verify_query(
            row["state"], row["controls"], row["ped_xy"], row["ped_vel"],
            float(gamma),
        )
        row["verifier_result"] = result
        positive = bool(
            result.get("resolved")
            and int(result.get("y", 0)) == 1
            and bool(result.get("full_h"))
        )
        row["full_h_positive"] = positive
        traces.append(row)
        if not positive:
            nvp_step = int(row["step"])
            break
        positive_steps += 1

    gated = dict(run)
    gated["trace"] = traces
    gated["states"] = np.asarray(run["states"])[: positive_steps + 1]
    gated["controls"] = np.asarray(run["controls"])[:positive_steps]
    gated["peds"] = np.asarray(run["peds"])[:positive_steps]
    gated["ped_vels"] = np.asarray(run["ped_vels"])[:positive_steps]
    gated["steps"] = int(positive_steps)
    gated["nvp"] = nvp_step is not None
    gated["nvp_step"] = nvp_step
    if nvp_step is not None:
        gated.update(
            success=False, reached=False, collision=False, timeout=False,
            status="nvp",
        )
    else:
        gated["status"] = (
            "success" if bool(run["success"])
            else "collision" if bool(run["collision"])
            else "timeout"
        )
    gated["gate_semantics"] = (
        "the exact full-H=10 moving-pedestrian verifier is applied to the "
        "SafeMPPI reward-weighted plan before its first action; first negative "
        "plan is displayed but not executed"
    )
    return gated


def _collect_expert(episode, gammas, environment, device, T):
    output = {}
    for gamma in gammas:
        print(
            f"[collect] SafeMPPI expert {environment['scene_profile']} "
            f"episode={episode} gamma={gamma:g}",
            flush=True,
        )
        run = EX.rollout(
            int(episode), float(gamma), device=device, T=int(T),
            n_ped=int(environment["n_ped"]),
            ped_speed_range=tuple(environment["ped_speed_range"]),
            collect_trace=True,
        )
        output[float(gamma)] = _gate_expert_run(run, gamma)
    return output


def _collect_kazuki(
        checkpoint, episode, gammas, environment, device, T, sample_seed,
        safe_coef, goal_coef,
):
    policy, _ = GPS.load_sfm_policy(checkpoint, device=device)
    policy.eval()
    output = {}
    config = KZ.KazukiConfig(
        safe_coefs=(float(safe_coef),), goal_coef=float(goal_coef),
    ).validate()
    for gamma in gammas:
        print(
            f"[collect] Kazuki OOD episode={episode} gamma={gamma:g}",
            flush=True,
        )
        output[float(gamma)] = KZ.kazuki_sfm_deploy(
            policy, episode=int(episode), gamma=float(gamma), device=device,
            T=int(T), n_ped=int(environment["n_ped"]),
            ped_speed_range=tuple(environment["ped_speed_range"]),
            sample_seed=int(sample_seed), cfg=config,
            collect_diagnostics=True,
        )
    return output


def collect(
        checkpoint, output_dir, *, id_episode=150_000, ood_episode=250_000,
        gammas=(0.1, 0.5, 1.0), id_profile="matched_id",
        ood_profile="double_density_velocity_ood", device="auto",
        verifier_workers=8, sample_seed=700_000, audit_seed=20260723,
        ell=0.24210826720721101, T=SP.T, expected_checkpoint_sha256=None,
        kazuki_safe_coef=.3, kazuki_goal_coef=.5,
):
    output_dir = os.path.abspath(output_dir)
    if os.path.exists(output_dir):
        raise FileExistsError(f"refusing to reuse output directory: {output_dir}")
    gammas = validate_gammas(gammas)
    device = resolve_device(device)
    checkpoint = os.path.abspath(checkpoint)
    checkpoint_sha256 = sha256_file(checkpoint)
    expected = expected_checkpoint_sha256
    if expected is None and os.path.basename(checkpoint) == "hp10_pretrained_r0.pt":
        expected = DEFAULT_CHECKPOINT_SHA256
    if expected and checkpoint_sha256 != expected:
        raise RuntimeError(
            f"checkpoint SHA mismatch: expected {expected}, observed "
            f"{checkpoint_sha256}"
        )
    id_environment = SS.scene_profile(id_profile)
    ood_environment = SS.scene_profile(ood_profile)
    os.makedirs(output_dir)

    expert_id = _collect_expert(
        id_episode, gammas, id_environment, device, T,
    )
    expert_ood = _collect_expert(
        ood_episode, gammas, ood_environment, device, T,
    )

    selectors = {}
    for selector in SELECTORS:
        selector_dir = os.path.join(output_dir, f"selector_{selector}")
        print(
            f"[collect] B1 selector={selector} OOD episode={ood_episode}",
            flush=True,
        )
        FA.collect(
            checkpoint, scenarios=(int(ood_episode),), gammas=gammas,
            scene_profile=ood_profile, device=device,
            verifier_workers=int(verifier_workers),
            sample_seed=int(sample_seed), audit_seed=int(audit_seed),
            ell=float(ell), T=int(T), selector=selector,
            outdir=selector_dir,
        )
        selectors[selector] = torch.load(
            os.path.join(selector_dir, "full_episode_label_audit.pt"),
            map_location="cpu", weights_only=False,
        )

    kazuki = _collect_kazuki(
        checkpoint, ood_episode, gammas, ood_environment, device, T,
        sample_seed, kazuki_safe_coef, kazuki_goal_coef,
    )
    bundle = dict(
        version=1, status=STATUS,
        checkpoint=os.path.abspath(checkpoint),
        checkpoint_sha256=checkpoint_sha256,
        id_episode=int(id_episode), ood_episode=int(ood_episode),
        gammas=list(gammas), T=int(T), sample_seed=int(sample_seed),
        audit_seed=int(audit_seed), device=str(device),
        id_environment=id_environment, ood_environment=ood_environment,
        expert_id=expert_id, expert_ood=expert_ood,
        selectors=selectors, kazuki=kazuki,
        selector_semantics=dict(
            shared=(
                "same pretrained checkpoint, OOD episode, gamma, keyed K=16 "
                "proposal/noise contract, B=4 RBF acquisition, exact H10 "
                "moving-pedestrian verifier, and post-NVP offline continuation"
            ),
            margin="max one-step nominal-Hp margin, then one-step progress",
            safemppi_cost="minimum frozen native SafeMPPI proposal cost",
            balanced_rank=(
                "minimum sum of ordinal safety/performance ranks with "
                "safety-first tie-breaking"
            ),
        ),
        expert_gate=(
            "SafeMPPI expert is stopped before executing the first plan that "
            "fails the exact full-H moving-pedestrian verifier"
        ),
        kazuki_config=dict(
            safe_coef=float(kazuki_safe_coef),
            goal_coef=float(kazuki_goal_coef),
            prior="same pretrained checkpoint",
            arrows=(
                "cyan: integrated goal guidance; magenta: integrated safety "
                "guidance along the guided ODE path"
            ),
        ),
        episode_banks=EPISODE_BANKS,
    )
    trace_path = os.path.join(output_dir, "six_row_traces.pt")
    _save_torch(trace_path, bundle)
    manifest = {
        key: value for key, value in bundle.items()
        if key not in ("expert_id", "expert_ood", "selectors", "kazuki")
    }
    manifest.update(
        trace_path=os.path.abspath(trace_path),
        trace_sha256=sha256_file(trace_path),
    )
    _write_json(os.path.join(output_dir, "TRACE_COMPLETE.json"), manifest)
    return trace_path


def _latest_trace(rows, step):
    if not rows:
        raise ValueError("empty trace")
    return rows[min(max(int(step), 0), len(rows) - 1)]


def _goal(axis):
    axis.plot(
        SS.GOAL[0], SS.GOAL[1], "*", color=YELLOW,
        mec="#222222", mew=.7, ms=8.5, zorder=20,
    )


def _draw_expert(axis, run, gamma, step):
    trace = _latest_trace(run["trace"], step)
    available = min(int(step), int(run["steps"]))
    states = np.asarray(run["states"], float)
    if len(states):
        axis.plot(
            states[: available + 1, 0], states[: available + 1, 1],
            color="#111111", lw=1.05, marker=".", ms=1.2, zorder=10,
        )
    BV._draw_pedestrians(axis, trace["ped_xy"], alpha=.62)
    for position, velocity in zip(trace["ped_xy"], trace["ped_vel"]):
        future = np.asarray(position)[None] + (
            np.arange(11)[:, None] * SS.DT * np.asarray(velocity)[None]
        )
        axis.plot(
            future[:, 0], future[:, 1], ".--", color=BV.GRAY,
            ms=1.6, lw=.45, alpha=.42,
        )
    nominal = DV.nominal_safemppi_levels(trace, gamma=gamma, H=10)
    BV._draw_level_polygons(
        axis, nominal["polygons"], color=TRUE_BLUE,
        linewidth=.48, alpha=.62, zorder=2,
    )
    DV._draw_outer_polygon(axis, nominal["outer_polygon"], color=TRUE_BLUE)
    plan = np.asarray(trace["planned_states"], float)[:, :2]
    positive = bool(trace["full_h_positive"])
    color = TRUE_BLUE if positive else TRUE_RED
    axis.plot(
        plan[:, 0], plan[:, 1], color=color, lw=1.5,
        marker=".", ms=2.1, alpha=.96, zorder=8,
    )
    if not positive:
        axis.plot(
            plan[-1, 0], plan[-1, 1], "x", color=TRUE_RED,
            ms=6.5, mew=1.5, zorder=12,
        )
    position = np.asarray(trace["state"], float)[:2]
    axis.plot(position[0], position[1], "o", color="#111111", ms=4.7, zorder=13)
    _goal(axis)
    DV._set_clean_axis(axis)
    return trace


def _draw_selector(axis, bundle, gamma, step):
    index = FV._index(bundle["traces"])
    scenario = int(bundle["scenarios"][0])
    rows = index[(scenario, round(float(gamma), 8))]
    trace = DB.draw_cell(
        axis, rows, int(step), branch_line_scale=1.9,
        trajectory_linewidth=.9, trajectory_marker_size=1.0,
        candidate_inset=False, draw_direction_arrow=False,
    )
    _goal(axis)
    return trace


def _draw_kazuki(axis, run, gamma, step):
    trace = DV.draw_method_panel(
        axis, "kazuki", run, gamma, int(step),
        guidance_scale=3.0, guidance_cap=1.8,
    )
    _goal(axis)
    return trace


def _trace_maximum(bundle):
    lengths = []
    for key in ("expert_id", "expert_ood", "kazuki"):
        lengths.extend(
            len(bundle[key][float(gamma)].get("trace") or ())
            for gamma in bundle["gammas"]
        )
    for selector in SELECTORS:
        traces = bundle["selectors"][selector]["traces"]
        lengths.extend(int(row["step"]) + 1 for row in traces)
    maximum = max(lengths, default=1)
    return max(0, maximum - 1)


def _layout(bundle):
    gammas = tuple(map(float, bundle["gammas"]))
    figure = plt.figure(figsize=(3.05 * len(gammas) + 4.1, 16.6))
    grid = figure.add_gridspec(
        6, len(gammas) + 1,
        width_ratios=[1.0] * len(gammas) + [1.18],
        left=.025, right=.985, bottom=.025, top=.985,
        wspace=.025, hspace=.035,
    )
    axes = np.empty((6, len(gammas)), dtype=object)
    side_axes = []
    for row in range(6):
        for column in range(len(gammas)):
            axes[row, column] = figure.add_subplot(grid[row, column])
        side = figure.add_subplot(grid[row, -1])
        side.set_axis_off()
        side_axes.append(side)
    return figure, axes, side_axes, gammas


def _side_text(bundle, row, frame_index, frame_count, simulator_step):
    labels = (
        (
            "SafeMPPI expert · ID",
            f"episode {bundle['id_episode']}",
            "nominal polytope + H=10 levels",
            "stop before first exact-verifier NVP",
        ),
        (
            "SafeMPPI expert · OOD",
            f"episode {bundle['ood_episode']}",
            "same demonstration controller",
            "stop before first exact-verifier NVP",
        ),
        (
            "B1 gather · max margin",
            "K=16 → RBF B=4",
            "exact H10 labels; no inset/arrow",
            "post-NVP: offline raw continuation",
        ),
        (
            "B1 gather · SafeMPPI cost",
            "same K/B/verifier/proposals",
            "only execution ranking changes",
            "post-NVP: offline raw continuation",
        ),
        (
            "B1 gather · balanced rank",
            "safety rank + performance rank",
            "safety-first tie-breaking",
            "post-NVP: offline raw continuation",
        ),
        (
            "Kazuki guidance · OOD",
            "same pretrained flow prior",
            "cyan: ∇ goal reward",
            "magenta: ∇ safety reward",
        ),
    )
    columns = ", ".join(f"{value:g}" for value in bundle["gammas"])
    lines = list(labels[row])
    if row == 0:
        lines.extend((
            "",
            f"columns γ (left→right): {columns}",
            f"frame {frame_index}/{frame_count - 1}",
            f"simulator step {simulator_step}",
        ))
    return "\n".join(lines)


def draw_frame(bundle, frame_index, frames, *, figure=None):
    if bundle.get("status") != STATUS:
        raise ValueError("not a completed six-row trace bundle")
    if not 0 <= int(frame_index) < len(frames):
        raise IndexError(
            f"frame index {frame_index} outside [0,{len(frames) - 1}]"
        )
    if figure is None:
        figure, axes, side_axes, gammas = _layout(bundle)
    else:
        figure, axes, side_axes, gammas = figure
    step = int(frames[int(frame_index)])
    for row in range(6):
        side_axes[row].clear()
        side_axes[row].set_axis_off()
        side_axes[row].text(
            .02, .96,
            _side_text(bundle, row, int(frame_index), len(frames), step),
            ha="left", va="top", fontsize=8.2, linespacing=1.45,
        )
    for column, gamma in enumerate(gammas):
        for row in range(6):
            axes[row, column].clear()
        _draw_expert(
            axes[0, column], bundle["expert_id"][gamma], gamma, step,
        )
        _draw_expert(
            axes[1, column], bundle["expert_ood"][gamma], gamma, step,
        )
        _draw_selector(
            axes[2, column], bundle["selectors"]["margin"], gamma, step,
        )
        _draw_selector(
            axes[3, column], bundle["selectors"]["safemppi_cost"], gamma, step,
        )
        _draw_selector(
            axes[4, column], bundle["selectors"]["balanced_rank"], gamma, step,
        )
        _draw_kazuki(
            axes[5, column], bundle["kazuki"][gamma], gamma, step,
        )
    return figure, axes, side_axes, gammas


def _legend():
    return [
        Line2D([], [], color=TRUE_BLUE, lw=1.6, label="positive planned H10 / nominal geometry"),
        Line2D([], [], color=TRUE_RED, lw=1.6, marker="x", label="negative planned H10"),
        Line2D([], [], color=BV.GRAY, lw=.7, label="K=16 generated"),
        Line2D([], [], color=BV.ORANGE, lw=1.0, label="B=4 RBF queried"),
        Line2D([], [], color=BV.GREEN, lw=1.2, label="B exact full-H positive / verifier levels"),
        Line2D([], [], color="#111111", lw=1.0, label="executed first-action trajectory"),
        Line2D([], [], color=DV.CYAN, lw=2.1, label=r"Kazuki $\nabla$ goal reward"),
        Line2D([], [], color=DV.MAGENTA, lw=2.1, label=r"Kazuki $\nabla$ safety reward"),
    ]


def render(
        trace_path, output_dir, *, fps=5, frame_stride=2, dpi=105,
):
    trace_path = os.path.abspath(trace_path)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    bundle = torch.load(trace_path, map_location="cpu", weights_only=False)
    maximum = _trace_maximum(bundle)
    frames = list(range(0, maximum + 1, int(frame_stride)))
    if frames[-1] != maximum:
        frames.append(maximum)
    layout = _layout(bundle)
    figure = layout[0]
    figure.legend(
        handles=_legend(), loc="lower right",
        bbox_to_anchor=(.985, .012), frameon=False, fontsize=7.1,
    )
    draw_frame(bundle, len(frames) - 1, frames, figure=layout)
    last_png = os.path.join(output_dir, "six_row_last_frame.png")
    figure.savefig(last_png, dpi=170)

    def update(frame_index):
        draw_frame(bundle, int(frame_index), frames, figure=layout)
        return []

    movie = animation.FuncAnimation(
        figure, update, frames=range(len(frames)),
        interval=1000 / int(fps), blit=False,
    )
    mp4 = os.path.join(output_dir, "six_row_comparison.mp4")
    movie.save(
        mp4, writer=animation.FFMpegWriter(
            fps=int(fps), bitrate=5200,
        ), dpi=int(dpi),
    )
    plt.close(figure)
    report = dict(
        status=RENDER_STATUS,
        trace_path=trace_path, trace_sha256=sha256_file(trace_path),
        frames=frames, frame_count=len(frames),
        frame_index_semantics=(
            "frame index addresses the frames list; each entry is the "
            "simulator step shown in the right-side label"
        ),
        fps=int(fps), frame_stride=int(frame_stride),
        mp4=os.path.abspath(mp4),
        last_frame_png=os.path.abspath(last_png),
        checkpoint_sha256=bundle["checkpoint_sha256"],
        episodes=dict(
            id=int(bundle["id_episode"]), ood=int(bundle["ood_episode"])
        ),
        gammas=list(map(float, bundle["gammas"])),
    )
    report_path = os.path.join(output_dir, "RENDER_COMPLETE.json")
    _write_json(report_path, report)
    return report


def snapshot(trace_path, render_json, frame_index, output_pdf):
    with open(render_json) as stream:
        report = json.load(stream)
    if report.get("status") != RENDER_STATUS:
        raise ValueError("not a completed six-row render manifest")
    trace_path = os.path.abspath(trace_path)
    if sha256_file(trace_path) != report["trace_sha256"]:
        raise RuntimeError("snapshot trace differs from rendered video trace")
    frames = list(map(int, report["frames"]))
    bundle = torch.load(trace_path, map_location="cpu", weights_only=False)
    layout = _layout(bundle)
    figure = layout[0]
    figure.legend(
        handles=_legend(), loc="lower right",
        bbox_to_anchor=(.985, .012), frameon=False, fontsize=7.1,
    )
    draw_frame(bundle, int(frame_index), frames, figure=layout)
    output_pdf = os.path.abspath(output_pdf)
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    figure.savefig(output_pdf, format="pdf")
    plt.close(figure)
    metadata = dict(
        status="SFM_SIX_ROW_VECTOR_SNAPSHOT_COMPLETE",
        frame_index=int(frame_index),
        simulator_step=int(frames[int(frame_index)]),
        trace_sha256=report["trace_sha256"],
        output_pdf=output_pdf,
        output_sha256=sha256_file(output_pdf),
    )
    _write_json(os.path.splitext(output_pdf)[0] + ".json", metadata)
    return metadata


def build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--checkpoint", required=True)
    collect_parser.add_argument("--output-dir", required=True)
    collect_parser.add_argument("--id-episode", type=int, default=150_000)
    collect_parser.add_argument("--ood-episode", type=int, default=250_000)
    collect_parser.add_argument(
        "--gammas", type=float, nargs="+", default=(.1, .5, 1.0),
    )
    collect_parser.add_argument("--id-profile", default="matched_id")
    collect_parser.add_argument(
        "--ood-profile", default="double_density_velocity_ood",
    )
    collect_parser.add_argument("--device", default="auto")
    collect_parser.add_argument("--verifier-workers", type=int, default=8)
    collect_parser.add_argument("--sample-seed", type=int, default=700_000)
    collect_parser.add_argument("--audit-seed", type=int, default=20260723)
    collect_parser.add_argument("--ell", type=float, default=.24210826720721101)
    collect_parser.add_argument("--T", type=int, default=SP.T)
    collect_parser.add_argument("--expected-checkpoint-sha256")
    collect_parser.add_argument("--kazuki-safe-coef", type=float, default=.3)
    collect_parser.add_argument("--kazuki-goal-coef", type=float, default=.5)

    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--trace-bundle", required=True)
    render_parser.add_argument("--output-dir", required=True)
    render_parser.add_argument("--fps", type=int, default=5)
    render_parser.add_argument("--frame-stride", type=int, default=2)
    render_parser.add_argument("--dpi", type=int, default=105)

    all_parser = subparsers.add_parser("all")
    for action in collect_parser._actions[1:]:
        if action.dest in ("help",):
            continue
        option_strings = list(action.option_strings)
        kwargs = dict(
            dest=action.dest, default=action.default, required=action.required,
            type=action.type, nargs=action.nargs, choices=action.choices,
            help=action.help,
        )
        all_parser.add_argument(*option_strings, **{
            key: value for key, value in kwargs.items() if value is not None
        })
    all_parser.add_argument("--fps", type=int, default=5)
    all_parser.add_argument("--frame-stride", type=int, default=2)
    all_parser.add_argument("--dpi", type=int, default=105)

    banks = subparsers.add_parser("episode-banks")
    banks.add_argument("--json", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "episode-banks":
        if args.json:
            print(json.dumps(EPISODE_BANKS, indent=2))
        else:
            for name, value in EPISODE_BANKS.items():
                print(f"{name:26s} {value}")
        return
    if args.command in ("collect", "all"):
        trace_path = collect(
            args.checkpoint, args.output_dir,
            id_episode=args.id_episode, ood_episode=args.ood_episode,
            gammas=args.gammas, id_profile=args.id_profile,
            ood_profile=args.ood_profile, device=args.device,
            verifier_workers=args.verifier_workers,
            sample_seed=args.sample_seed, audit_seed=args.audit_seed,
            ell=args.ell, T=args.T,
            expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            kazuki_safe_coef=args.kazuki_safe_coef,
            kazuki_goal_coef=args.kazuki_goal_coef,
        )
        if args.command == "collect":
            print(trace_path)
            return
        report = render(
            trace_path, args.output_dir, fps=args.fps,
            frame_stride=args.frame_stride, dpi=args.dpi,
        )
        print(report["mp4"])
        return
    report = render(
        args.trace_bundle, args.output_dir, fps=args.fps,
        frame_stride=args.frame_stride, dpi=args.dpi,
    )
    print(report["mp4"])


if __name__ == "__main__":
    main()
