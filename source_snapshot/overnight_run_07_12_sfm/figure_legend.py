"""Standalone legend PDF for the one-off gamma x time figure.

Four entries, named exactly as requested: green = executed trajectory, blue =
verifier positive action, red = verifier negative action, green box = verifier
polytope.  Saved with a tight bounding box so the PDF is exactly the legend.
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
import figure_champion_grid as FIG


def build(output: Path, *, fontsize: float = 22, ncol: int = 1) -> None:
    STYLE.apply_computer_modern_style()
    handles = FIG.legend_handles()
    figure = plt.figure(figsize=(6.0, 2.4))
    legend = figure.legend(
        handles=handles, loc="center", frameon=True, ncol=int(ncol),
        fontsize=fontsize, handlelength=2.4, labelspacing=0.7,
        borderpad=0.7, edgecolor="#333333",
    )
    legend.get_frame().set_linewidth(0.9)
    for text in legend.get_texts():
        text.set_color("#111111")
    figure.canvas.draw()
    box = legend.get_window_extent().transformed(
        figure.dpi_scale_trans.inverted()
    )
    figure.savefig(output, format="pdf", bbox_inches=box.expanded(1.04, 1.10),
                   pad_inches=0.0)
    plt.close(figure)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--fontsize", type=float, default=22)
    ap.add_argument("--ncol", type=int, default=1)
    args = ap.parse_args()
    build(Path(args.output), fontsize=args.fontsize, ncol=args.ncol)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
