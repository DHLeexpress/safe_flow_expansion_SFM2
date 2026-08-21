"""Render the 3x3 PRE / Champion / Kazuki comparison panels.

Style follows sfm_hp100_final_videos.render_raw_case with three declared
deviations requested by the user: per-method executed-sample colors
(PRE red, Champion blue, Kazuki magenta), a larger multi-step-safety badge,
and method/gamma labels on the RIGHT side of each panel (plot area clean,
tick numbers untouched).
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import torch

import _paths  # noqa: F401
import sfm_hp100_final_videos as FV
import sfm_hp100_paper_video_style as STYLE
import sfm_scene as SS

MAGENTA = "#C6007E"
SAFETY_ORANGE = "#FF7F0E"
GOAL_CYAN = "#00B7C3"
BADGE_FONTSIZE = 15.0
GUIDANCE_SCALE = 3.0
GUIDANCE_CAP = 1.8


def _badge(axis, safe: bool) -> None:
    color = STYLE.POSITIVE_BLUE if bool(safe) else STYLE.NEGATIVE_RED
    axis.text(
        0.025, 0.975,
        rf"$\mathrm{{Multi\!\!-\!step\ safety:}}\ \mathbf{{{str(bool(safe))}}}$",
        transform=axis.transAxes, ha="left", va="top", color=color,
        fontsize=BADGE_FONTSIZE, zorder=100,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white",
              "edgecolor": "none", "alpha": 0.78},
    )


def _right_label(axis, text: str) -> None:
    axis.text(
        1.035, 0.5, text, transform=axis.transAxes, rotation=270,
        ha="left", va="center", fontsize=12.5, color="#222222",
    )


def render_case_panel(
    case: dict, output: Path, *, positive_color: str, label: str,
    fps: int = 7, bounds=(-0.5, 6.5),
) -> int:
    """render_raw_case with per-method positive color + right label."""
    STYLE.apply_computer_modern_style()
    traces = list(case["traces"])
    indices = list(range(len(traces)))
    figure, axis = plt.subplots(figsize=(7.4, 6.8))
    figure.subplots_adjust(right=0.88)

    def update(index):
        axis.clear()
        current = traces[int(index)]
        FV._draw_pedestrians(axis, current["ped_xy"], current["ped_vel"])
        for row in traces[:int(index) + 1]:
            segment = np.asarray(row["proposal_result"]["segment"], float)
            positive = row["proposal_label"] == "full_h_positive"
            axis.plot(
                segment[:, 0], segment[:, 1],
                color=(positive_color if positive else STYLE.NEGATIVE_RED),
                lw=STYLE.SAMPLE_LW,
                alpha=0.18 if row is not current else 0.96, zorder=8,
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
            try:
                FV._draw_verifier(axis, float(case["gamma"]),
                                  current["proposal_result"])
            except (ValueError, KeyError):
                pass  # frame without a drawable GREEN certificate
        FV._draw_robot_goal(axis, current["state"])
        STYLE.fixed_world_frame(axis, bounds=bounds)
        _badge(axis, positive)
        _right_label(axis, label)
        return []

    movie = animation.FuncAnimation(
        figure, update, frames=indices, interval=1000 / fps, blit=False,
    )
    movie.save(output, writer=FV._writer(fps), dpi=130)
    plt.close(figure)
    return len(indices)


def render_kazuki_panel(
    run: dict, output: Path, *, label: str, fps: int = 7,
    bounds=(-0.5, 6.5),
) -> int:
    STYLE.apply_computer_modern_style()
    rows = list(run["trace"])
    audits = list(run["audits"])
    indices = list(range(len(rows)))
    collided = bool(run["collision"])
    figure, axis = plt.subplots(figsize=(7.4, 6.8))
    figure.subplots_adjust(right=0.88)

    def update(index):
        axis.clear()
        index = int(index)
        current = rows[index]
        FV._draw_pedestrians(
            axis, current["pedestrian_xy"], current["pedestrian_velocity"],
        )
        for row in rows[:index + 1]:
            plan = np.asarray(row["selected_plan_positions"], float)
            axis.plot(plan[:, 0], plan[:, 1], color=MAGENTA,
                      lw=STYLE.SAMPLE_LW,
                      alpha=0.18 if row is not current else 0.96, zorder=8)
        audit = audits[index]
        result = audit["verify"]
        valid = bool(result.get("resolved")) and int(result.get("y", 0)) == 1
        if not valid:
            plan = np.asarray(current["selected_plan_positions"], float)
            axis.plot(plan[-1, 0], plan[-1, 1], "x",
                      color=STYLE.NEGATIVE_RED, ms=5.5, mew=1.2, zorder=12)
        states = np.asarray(
            run["states"][:index + 2], float,
        )[:, :2]
        axis.plot(states[:, 0], states[:, 1], color=STYLE.EXECUTED_BLACK,
                  lw=STYLE.EXECUTED_LW, zorder=20)
        if collided and index == len(rows) - 1:
            axis.plot(states[-1, 0], states[-1, 1], "X",
                      color=STYLE.NEGATIVE_RED, ms=13, mew=2.0, zorder=40)
        start = np.asarray(current["state"], float)[:2]
        for key, color in (
            ("goal_guidance_action", GOAL_CYAN),
            ("safety_guidance_action", SAFETY_ORANGE),
        ):
            vector = GUIDANCE_SCALE * np.asarray(audit[key], float)
            norm = float(np.linalg.norm(vector))
            if norm > GUIDANCE_CAP:
                vector *= GUIDANCE_CAP / norm
            axis.add_patch(FancyArrowPatch(
                tuple(start), tuple(start + vector), arrowstyle="-|>",
                mutation_scale=15, lw=2.6, color=color,
                shrinkA=0, shrinkB=0, zorder=11,
            ))
        if valid:
            try:
                FV._draw_verifier(axis, float(run["gamma"]), dict(result))
            except (ValueError, KeyError):
                pass  # frame without a drawable GREEN certificate
        FV._draw_robot_goal(axis, current["state"])
        STYLE.fixed_world_frame(axis, bounds=bounds)
        _badge(axis, valid)
        _right_label(axis, label)
        return []

    movie = animation.FuncAnimation(
        figure, update, frames=indices, interval=1000 / fps, blit=False,
    )
    movie.save(output, writer=FV._writer(fps), dpi=130)
    plt.close(figure)
    return len(indices)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre-cases", required=True)
    ap.add_argument("--champion-cases", required=True)
    ap.add_argument("--kazuki-dir", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    gammas = (0.1, 0.5, 1.0)
    pre = torch.load(args.pre_cases, map_location="cpu", weights_only=False)
    champ = torch.load(args.champion_cases, map_location="cpu",
                       weights_only=False)
    ledger = {"episode": args.episode, "panels": {}}
    for gamma in gammas:
        tag = f"ep{args.episode}_g{gamma:g}".replace(".", "p")
        n1 = render_case_panel(
            pre["cases"][tag], out / f"panel_pre_g{gamma:g}.mp4",
            positive_color=STYLE.NEGATIVE_RED,
            label=rf"PRE ($\gamma={gamma:g}$)",
        )
        n2 = render_case_panel(
            champ["cases"][tag], out / f"panel_champion_g{gamma:g}.mp4",
            positive_color=STYLE.POSITIVE_BLUE,
            label=rf"Champion ($\gamma={gamma:g}$)",
        )
        run = torch.load(
            Path(args.kazuki_dir) / f"kazuki_{tag}.pt",
            map_location="cpu", weights_only=False,
        )
        n3 = render_kazuki_panel(
            run, out / f"panel_kazuki_g{gamma:g}.mp4",
            label=rf"CFM-MPPI ($\gamma={gamma:g}$)",
        )
        ledger["panels"][f"g{gamma:g}"] = {
            "pre_frames": n1, "champion_frames": n2, "kazuki_frames": n3,
            "pre_outcome": pre["cases"][tag]["outcome"],
            "champion_outcome": champ["cases"][tag]["outcome"],
            "kazuki_outcome": ("collision" if run["collision"]
                               else "success" if run["success"] else "timeout"),
        }
    (out / "PANELS_LEDGER.json").write_text(json.dumps(ledger, indent=2))
    print(json.dumps(ledger, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
