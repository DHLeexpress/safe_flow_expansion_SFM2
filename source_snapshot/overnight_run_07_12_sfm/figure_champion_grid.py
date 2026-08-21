"""One-off figure: Safe Flow Expansion row reshaped to gamma x time (3x2).

Rows are the three gammas, columns are two snapshots of the same rollout: an
early one (t = 4 s) and the final one.  Style follows the user's one-off
request: no multi-step-safety badge, pure blue / pure red verifier actions at
20% transparency with a black boundary, green executed trajectory, tight
spacing and axis labels.  The legend is written separately by
``figure_legend.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import torch

import _paths  # noqa: F401
import sfm_hp100_final_videos as FV
import sfm_hp100_paper_video_style as STYLE

TRUE_BLUE = "#0000FF"
TRUE_RED = "#FF0000"
EXEC_GREEN = "#2E9A30"
TRAIL_ALPHA = 0.80          # 20% transparency
TRAIL_LW = STYLE.SAMPLE_LW * 0.72
TRAIL_EDGE = 0.65
EXEC_LW = STYLE.EXECUTED_LW * 1.7
AXIS_FONTSIZE = 22
LABEL_FONTSIZE = 29.3          # gamma / time captions
TICK_FONTSIZE = 13.5
DT = 0.1
GAMMAS = (0.1, 0.5, 1.0)


def legend_handles():
    """The four declared entries, shared by the in-panel and standalone legend."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    stroke = lambda width: [
        path_effects.Stroke(linewidth=width, foreground="black",
                            alpha=TRAIL_ALPHA),
        path_effects.Normal(),
    ]
    return [
        Line2D([], [], color=EXEC_GREEN, lw=4.0, label="Executed trajectory"),
        Line2D([], [], color=TRUE_BLUE, lw=4.0, alpha=TRAIL_ALPHA,
               path_effects=stroke(5.6), label="Verifier positive action"),
        Line2D([], [], color=TRUE_RED, lw=4.0, alpha=TRAIL_ALPHA,
               path_effects=stroke(5.6), label="Verifier negative action"),
        Patch(facecolor="white", edgecolor=STYLE.VERIFIER_GREEN, linewidth=2.4,
              label="Verifier polytope"),
    ]


def _trail(axis, segment, valid: bool) -> None:
    line, = axis.plot(
        segment[:, 0], segment[:, 1],
        color=(TRUE_BLUE if valid else TRUE_RED),
        lw=TRAIL_LW, alpha=TRAIL_ALPHA, solid_capstyle="round", zorder=8,
    )
    line.set_path_effects([
        path_effects.Stroke(linewidth=TRAIL_LW + 2 * TRAIL_EDGE,
                            foreground="black", alpha=TRAIL_ALPHA),
        path_effects.Normal(),
    ])


def _panel(axis, traces, index: int, gamma: float, bounds) -> None:
    current = traces[index]
    FV._draw_pedestrians(axis, current["ped_xy"], current["ped_vel"])
    for row in traces[:index + 1]:
        segment = np.asarray(row["proposal_result"]["segment"], float)
        _trail(axis, segment, row["proposal_label"] == "full_h_positive")
    states = [np.asarray(row["state"], float)[:2] for row in traces[:index + 1]]
    states.append(np.asarray(current["next_state"], float)[:2])
    states = np.asarray(states)
    line, = axis.plot(states[:, 0], states[:, 1], color=EXEC_GREEN, lw=EXEC_LW,
                      solid_capstyle="round", zorder=20)
    line.set_path_effects([
        path_effects.Stroke(linewidth=EXEC_LW + 1.4, foreground="white",
                            alpha=0.85),
        path_effects.Normal(),
    ])
    if current["proposal_label"] == "full_h_positive":
        try:
            FV._draw_verifier(axis, float(gamma), current["proposal_result"])
        except (ValueError, KeyError):
            pass  # frame without a drawable GREEN certificate
    FV._draw_robot_goal(axis, current["state"])
    STYLE.fixed_world_frame(axis, bounds=bounds)
    axis.tick_params(axis="both", labelsize=TICK_FONTSIZE, length=3.0,
                     width=0.7)


def build(cases: dict, episode: int, output: Path, *, early_seconds: float = 4.0,
          late_seconds: float = 8.0, bounds=(-0.5, 6.5)) -> dict:
    STYLE.apply_computer_modern_style()
    figure, axes = plt.subplots(
        3, 2, figsize=(9.4, 13.2), sharex=True, sharey=True,
        gridspec_kw={"wspace": 0.03, "hspace": 0.03},
    )
    ledger = {"episode": episode, "panels": {}}
    early_index = int(round(early_seconds / DT))
    late_index = int(round(late_seconds / DT))
    for row, gamma in enumerate(GAMMAS):
        tag = f"ep{episode}_g{gamma:g}".replace(".", "p")
        traces = list(cases[tag]["traces"])
        last_index = len(traces) - 1
        picks = [min(early_index, last_index), min(late_index, last_index)]
        for column, index in enumerate(picks):
            _panel(axes[row][column], traces, index, gamma, bounds)
            ledger["panels"][f"g{gamma:g}_c{column}"] = {
                "step": index, "time_seconds": round(index * DT, 2),
                "is_terminal_frame": bool(index == last_index),
            }
        axes[row][0].set_ylabel(r"$y\ \mathrm{[m]}$",
                                fontsize=AXIS_FONTSIZE, labelpad=1)
    for column, seconds in enumerate((early_seconds, late_seconds)):
        axes[2][column].set_xlabel(r"$x\ \mathrm{[m]}$",
                                   fontsize=AXIS_FONTSIZE, labelpad=2)
    legend = axes[0][0].legend(
        handles=legend_handles(), loc="upper left", fontsize=15.0,
        handlelength=2.1, labelspacing=0.55, borderpad=0.55,
        framealpha=0.94, edgecolor="#333333",
    )
    legend.set_zorder(120)
    figure.subplots_adjust(left=0.175, right=0.995, top=0.995, bottom=0.108)
    for row, gamma in enumerate(GAMMAS):
        box = axes[row][0].get_position()
        figure.text(box.x0 - 0.105, box.y0 + box.height / 2,
                    rf"$\gamma = {gamma:.1f}$", rotation=90, ha="left",
                    va="center", fontsize=LABEL_FONTSIZE)
    for column, seconds in enumerate((early_seconds, late_seconds)):
        box = axes[2][column].get_position()
        figure.text(box.x0 + box.width / 2, 0.020,
                    rf"$t = {seconds:g}\ \mathrm{{[s]}}$", ha="center",
                    va="bottom", fontsize=LABEL_FONTSIZE)
    figure.savefig(output, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    ledger["column_captions"] = [early_seconds, late_seconds]
    return ledger


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--champion-cases", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--early-seconds", type=float, default=4.0)
    ap.add_argument("--late-seconds", type=float, default=8.0)
    args = ap.parse_args()
    champ = torch.load(args.champion_cases, map_location="cpu",
                       weights_only=False)
    ledger = build(champ["cases"], args.episode, Path(args.output),
                   early_seconds=args.early_seconds,
                   late_seconds=args.late_seconds)
    print(json.dumps(ledger, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
