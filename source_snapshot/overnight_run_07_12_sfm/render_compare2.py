"""Render the 3x3 comparison panels, revision 2 (user style directives).

Changes versus ``render_compare.py``:

* tick numbers 1.5x, multi-step-safety badge box and text 1.5x;
* no in-panel method/gamma labels — titles live in a separate outside frame,
  and the panels carry tight margins so the assembled grid is compact;
* every row leaves the same kind of trail: each executed H10 action window is
  drawn in blue when that window was multi-step safe and in a blackish red
  when it was not, at 10% transparency with a solid black boundary;
* the executed state path is drawn in the method colour (PRE #CF2626, Safe
  Flow Expansion #2E9A30, CFM-MPPI #BB3FC4) instead of black, so the invalid
  trail red can no longer be confused with the PRE trajectory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import torch

import _paths  # noqa: F401
import sfm_hp100_final_videos as FV
import sfm_hp100_paper_video_style as STYLE

SAFETY_ORANGE = "#FF7F0E"
GOAL_CYAN = "#00B7C3"
TRAIL_VALID = STYLE.POSITIVE_BLUE      # multi-step safe window
TRAIL_INVALID = "#6B0C0C"              # blackish red, never the PRE path red
COLLISION_RED = "#E31A1C"
EXEC_PRE = "#CF2626"
EXEC_CHAMPION = "#2E9A30"
EXEC_KAZUKI = "#BB3FC4"

BADGE_FONTSIZE = 22.5                  # 1.5x the previous 15.0
BADGE_PAD = 0.42                       # 1.5x the previous 0.28
TICK_FONTSIZE = 13.5                   # 1.5x the style's 9.0
TRAIL_ALPHA = 0.90                     # 10% transparency
TRAIL_LW = STYLE.SAMPLE_LW * 0.80
TRAIL_EDGE_LW = 0.75
EXEC_LW = STYLE.EXECUTED_LW * 1.55
FIGSIZE = (6.8, 6.8)
DPI = 110
GUIDANCE_SCALE = 3.0
GUIDANCE_CAP = 1.8


def _badge(axis, safe: bool) -> None:
    color = TRAIL_VALID if bool(safe) else TRAIL_INVALID
    axis.text(
        0.022, 0.978,
        rf"$\mathrm{{Multi\!\!-\!step\ safety:}}\ \mathbf{{{str(bool(safe))}}}$",
        transform=axis.transAxes, ha="left", va="top", color=color,
        fontsize=BADGE_FONTSIZE, zorder=100,
        bbox={"boxstyle": f"round,pad={BADGE_PAD}", "facecolor": "white",
              "edgecolor": color, "linewidth": 1.1, "alpha": 0.88},
    )


def _frame(axis, bounds) -> None:
    STYLE.fixed_world_frame(axis, bounds=bounds)
    axis.tick_params(axis="both", labelsize=TICK_FONTSIZE, length=3.5, width=0.8)


def _trail(axis, segment: np.ndarray, valid: bool) -> None:
    """One executed action window: colour by safety, black solid boundary."""
    line, = axis.plot(
        segment[:, 0], segment[:, 1],
        color=(TRAIL_VALID if valid else TRAIL_INVALID),
        lw=TRAIL_LW, alpha=TRAIL_ALPHA, solid_capstyle="round", zorder=8,
    )
    line.set_path_effects([
        path_effects.Stroke(linewidth=TRAIL_LW + 2 * TRAIL_EDGE_LW,
                            foreground="black", alpha=TRAIL_ALPHA),
        path_effects.Normal(),
    ])


def _executed(axis, states: np.ndarray, color: str) -> None:
    line, = axis.plot(states[:, 0], states[:, 1], color=color, lw=EXEC_LW,
                      solid_capstyle="round", zorder=20)
    line.set_path_effects([
        path_effects.Stroke(linewidth=EXEC_LW + 1.4, foreground="white",
                            alpha=0.85),
        path_effects.Normal(),
    ])


def _new_axes():
    figure, axis = plt.subplots(figsize=FIGSIZE)
    figure.subplots_adjust(left=0.085, right=0.995, top=0.995, bottom=0.075)
    return figure, axis


def render_case_panel(case: dict, output: Path, *, exec_color: str,
                      fps: int = 7, bounds=(-0.5, 6.5)) -> int:
    STYLE.apply_computer_modern_style()
    traces = list(case["traces"])
    figure, axis = _new_axes()

    def update(index):
        axis.clear()
        index = int(index)
        current = traces[index]
        FV._draw_pedestrians(axis, current["ped_xy"], current["ped_vel"])
        for row in traces[:index + 1]:
            segment = np.asarray(row["proposal_result"]["segment"], float)
            valid = row["proposal_label"] == "full_h_positive"
            _trail(axis, segment, valid)
            if not valid:
                axis.plot(segment[-1, 0], segment[-1, 1], "x",
                          color=TRAIL_INVALID, ms=6.5, mew=1.5, zorder=12)
        states = [np.asarray(row["state"], float)[:2]
                  for row in traces[:index + 1]]
        states.append(np.asarray(current["next_state"], float)[:2])
        _executed(axis, np.asarray(states), exec_color)
        valid = current["proposal_label"] == "full_h_positive"
        if valid:
            try:
                FV._draw_verifier(axis, float(case["gamma"]),
                                  current["proposal_result"])
            except (ValueError, KeyError):
                pass  # frame without a drawable GREEN certificate
        FV._draw_robot_goal(axis, current["state"])
        _frame(axis, bounds)
        _badge(axis, valid)
        return []

    movie = animation.FuncAnimation(
        figure, update, frames=list(range(len(traces))),
        interval=1000 / fps, blit=False,
    )
    movie.save(output, writer=FV._writer(fps), dpi=DPI)
    plt.close(figure)
    return len(traces)


def render_kazuki_panel(run: dict, output: Path, *, fps: int = 7,
                        bounds=(-0.5, 6.5)) -> int:
    STYLE.apply_computer_modern_style()
    rows = list(run["trace"])
    audits = list(run["audits"])
    collided = bool(run["collision"])
    figure, axis = _new_axes()

    def valid_at(index: int) -> bool:
        result = audits[index]["verify"]
        return bool(result.get("resolved")) and int(result.get("y", 0)) == 1

    def update(index):
        axis.clear()
        index = int(index)
        current = rows[index]
        FV._draw_pedestrians(
            axis, current["pedestrian_xy"], current["pedestrian_velocity"],
        )
        for step, row in enumerate(rows[:index + 1]):
            plan = np.asarray(row["selected_plan_positions"], float)
            _trail(axis, plan, valid_at(step))
            if not valid_at(step):
                axis.plot(plan[-1, 0], plan[-1, 1], "x", color=TRAIL_INVALID,
                          ms=6.5, mew=1.5, zorder=12)
        states = np.asarray(run["states"][:index + 2], float)[:, :2]
        _executed(axis, states, EXEC_KAZUKI)
        if collided and index == len(rows) - 1:
            axis.plot(states[-1, 0], states[-1, 1], "X", color=COLLISION_RED,
                      ms=15, mew=2.2, zorder=40,
                      markeredgecolor="black")
        start = np.asarray(current["state"], float)[:2]
        audit = audits[index]
        for key, color in (("goal_guidance_action", GOAL_CYAN),
                           ("safety_guidance_action", SAFETY_ORANGE)):
            vector = GUIDANCE_SCALE * np.asarray(audit[key], float)
            norm = float(np.linalg.norm(vector))
            if norm > GUIDANCE_CAP:
                vector *= GUIDANCE_CAP / norm
            axis.add_patch(FancyArrowPatch(
                tuple(start), tuple(start + vector), arrowstyle="-|>",
                mutation_scale=16, lw=2.8, color=color,
                shrinkA=0, shrinkB=0, zorder=11,
            ))
        if valid_at(index):
            try:
                FV._draw_verifier(axis, float(run["gamma"]),
                                  dict(audit["verify"]))
            except (ValueError, KeyError):
                pass  # frame without a drawable GREEN certificate
        FV._draw_robot_goal(axis, current["state"])
        _frame(axis, bounds)
        _badge(axis, valid_at(index))
        return []

    movie = animation.FuncAnimation(
        figure, update, frames=list(range(len(rows))),
        interval=1000 / fps, blit=False,
    )
    movie.save(output, writer=FV._writer(fps), dpi=DPI)
    plt.close(figure)
    return len(rows)


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
        n1 = render_case_panel(pre["cases"][tag],
                               out / f"panel_pre_g{gamma:g}.mp4",
                               exec_color=EXEC_PRE)
        n2 = render_case_panel(champ["cases"][tag],
                               out / f"panel_champion_g{gamma:g}.mp4",
                               exec_color=EXEC_CHAMPION)
        run = torch.load(Path(args.kazuki_dir) / f"kazuki_{tag}.pt",
                         map_location="cpu", weights_only=False)
        n3 = render_kazuki_panel(run, out / f"panel_kazuki_g{gamma:g}.mp4")
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
