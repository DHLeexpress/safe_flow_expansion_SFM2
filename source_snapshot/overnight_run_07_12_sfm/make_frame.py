"""Build the outside title/legend frame that wraps the 3x3 panel grid.

The panels themselves carry no text beyond tick numbers and the multi-step
safety badge; every name lives here: column titles (gamma), row titles
(method, in that method's trajectory colour) and one legend that says what
each drawn line means.  The PNG is transparent where the video grid goes, so
it can be overlaid on the padded grid by ffmpeg.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import _paths  # noqa: F401
import sfm_hp100_paper_video_style as STYLE
import render_compare2 as R2

ROWS = (
    ("Pretrained CFM (PRE)", R2.EXEC_PRE),
    ("Safe Flow Expansion (ours)", R2.EXEC_CHAMPION),
    ("CFM-MPPI (Kazuki)", R2.EXEC_KAZUKI),
)
GAMMAS = (0.1, 0.5, 1.0)


def build(panel_w: int, panel_h: int, left: int, top: int, bottom: int,
          output: Path, *, dpi: int = 110) -> tuple[int, int]:
    STYLE.apply_computer_modern_style()
    width = left + 3 * panel_w
    height = top + 3 * panel_h + bottom
    figure = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    figure.patch.set_alpha(0.0)
    axis = figure.add_axes([0, 0, 1, 1])
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)          # image coordinates: y grows downward
    axis.axis("off")
    axis.patch.set_alpha(0.0)

    for index, gamma in enumerate(GAMMAS):
        axis.text(
            left + panel_w * (index + 0.5), top * 0.62,
            rf"$\gamma = {gamma:.1f}$", ha="center", va="center",
            fontsize=34, color="#111111",
        )
    for index, (name, color) in enumerate(ROWS):
        axis.text(
            left * 0.52, top + panel_h * (index + 0.5), name,
            ha="center", va="center", rotation=90, fontsize=27, color=color,
        )

    handles = [
        Line2D([], [], color=R2.EXEC_PRE, lw=3.2,
               label="Executed trajectory - Pretrained CFM"),
        Line2D([], [], color=R2.EXEC_CHAMPION, lw=3.2,
               label="Executed trajectory - Safe Flow Expansion"),
        Line2D([], [], color=R2.EXEC_KAZUKI, lw=3.2,
               label="Executed trajectory - CFM-MPPI"),
        Line2D([], [], color=R2.TRAIL_VALID, lw=3.2,
               path_effects=[path_effects.Stroke(linewidth=5.0,
                                                 foreground="black"),
                             path_effects.Normal()],
               label="Executed H10 action window - multi-step safe"),
        Line2D([], [], color=R2.TRAIL_INVALID, lw=3.2,
               path_effects=[path_effects.Stroke(linewidth=5.0,
                                                 foreground="black"),
                             path_effects.Normal()],
               label="Executed H10 action window - not multi-step safe"),
        Patch(facecolor="white", edgecolor=STYLE.VERIFIER_GREEN, linewidth=2.0,
              label="Exact GREEN verifier polytope (audit geometry)"),
        Line2D([], [], color=R2.GOAL_CYAN, lw=3.2,
               label="CFM-MPPI goal guidance"),
        Line2D([], [], color=R2.SAFETY_ORANGE, lw=3.2,
               label="CFM-MPPI safety guidance"),
        Line2D([], [], color=R2.COLLISION_RED, lw=0, marker="X", ms=13,
               markeredgecolor="black", label="Collision"),
    ]
    legend = figure.legend(
        handles=handles, loc="lower center",
        bbox_to_anchor=(0.5, 0.004), ncol=3, frameon=False,
        fontsize=21, handlelength=2.6, labelspacing=0.85,
        columnspacing=2.6,
    )
    for text in legend.get_texts():
        text.set_color("#111111")

    figure.savefig(output, dpi=dpi, transparent=True)
    plt.close(figure)
    return width, height


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-w", type=int, required=True)
    ap.add_argument("--panel-h", type=int, required=True)
    ap.add_argument("--left", type=int, default=150)
    ap.add_argument("--top", type=int, default=120)
    ap.add_argument("--bottom", type=int, default=300)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    width, height = build(args.panel_w, args.panel_h, args.left, args.top,
                          args.bottom, Path(args.output))
    print(f"{width} {height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
