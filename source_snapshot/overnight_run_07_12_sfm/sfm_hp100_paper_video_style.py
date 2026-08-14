"""Shared, presentation-only style for the final SFM paper videos.

This module never changes planner, verifier, policy, or dynamics semantics.
It centralizes only typography, colors, camera geometry, and optional on-frame
certificate status.  Explanatory labels are rendered to a separate sheet so
that the MP4s remain composable paper assets.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import sfm_scene as SS


VERSION = "sfm_hp100_paper_video_style_v1"

# Color-blind-readable roles.  Positive/negative and executed-path colors are
# deliberately identical across SafeMPPI, raw-policy, and expansion videos.
POSITIVE_BLUE = "#0057FF"
NEGATIVE_RED = "#E31A1C"
VERIFIER_GREEN = "#00A651"
UNCERTAINTY_GRAY = "#8A8A8A"
WEIGHTED_GOLD = "#E69F00"
EXECUTED_BLACK = "#111111"
PEDESTRIAN_GRAY = "#777777"
GOAL_GOLD = "#F0B400"

ROLLOUT_LW = 1.35
SAMPLE_LW = 2.25
EXECUTED_LW = 1.55
POLYTOPE_LW = 1.05


def apply_computer_modern_style() -> None:
    """Use Matplotlib's bundled Computer Modern fonts reproducibly."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "font.size": 10.0,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "axes.linewidth": 0.75,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def fixed_world_frame(axis, *, bounds=None) -> None:
    """Apply one non-moving task-space camera with numeric ticks only."""
    if bounds is None:
        xlo, xhi, ylo, yhi = (
            float(SS.TASK_LO), float(SS.TASK_HI),
            float(SS.TASK_LO), float(SS.TASK_HI),
        )
    else:
        values = tuple(map(float, bounds))
        if len(values) == 2:
            xlo, xhi = values
            ylo, yhi = values
        elif len(values) == 4:
            xlo, xhi, ylo, yhi = values
        else:
            raise ValueError("camera bounds must contain 2 or 4 values")
    axis.set_xlim(xlo, xhi)
    axis.set_ylim(ylo, yhi)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.set_title("")
    axis.grid(alpha=0.10, linewidth=0.45)
    axis.tick_params(direction="out", length=2.5, width=0.55)


def safety_badge(axis, safe: bool, *, enabled: bool) -> None:
    """Draw the sole optional explanatory label allowed inside an MP4."""
    if not enabled:
        return
    color = POSITIVE_BLUE if bool(safe) else NEGATIVE_RED
    axis.text(
        0.025, 0.975,
        rf"$\mathrm{{Multi\!\!-\!step\ safety:}}\ \mathbf{{{str(bool(safe))}}}$",
        transform=axis.transAxes, ha="left", va="top", color=color,
        fontsize=10.5, zorder=100,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white", "edgecolor": "none", "alpha": 0.78,
        },
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def render_label_sheet(output_stem: str | Path) -> dict:
    """Render reusable labels separately from all scene videos."""
    apply_computer_modern_style()
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    axis.axis("off")
    handles = [
        Line2D([], [], color=POSITIVE_BLUE, lw=SAMPLE_LW,
               label=r"Exact full-$H$ positive sample"),
        Line2D([], [], color=NEGATIVE_RED, lw=SAMPLE_LW,
               label=r"Exact full-$H$ negative sample"),
        Line2D([], [], color=VERIFIER_GREEN, lw=POLYTOPE_LW,
               label=r"Candidate-specific verifier polytope and level sets"),
        Line2D([], [], color=UNCERTAINTY_GRAY, lw=ROLLOUT_LW,
               label=r"Candidate rollout"),
        Line2D([], [], color=WEIGHTED_GOLD, lw=ROLLOUT_LW,
               label=r"SafeMPPI weighted plan"),
        Line2D([], [], color=EXECUTED_BLACK, lw=EXECUTED_LW,
               label=r"Executed closed-loop trajectory"),
        Patch(facecolor="white", edgecolor=POSITIVE_BLUE,
              label=r"Multi-step safety: True"),
        Patch(facecolor="white", edgecolor=NEGATIVE_RED,
              label=r"Multi-step safety: False"),
    ]
    axis.legend(handles=handles, loc="upper center", frameon=False, ncol=1,
                fontsize=10.5, handlelength=3.0, labelspacing=0.75)
    axis.text(
        0.5, 0.34,
        "SafeMPPI expert\nPretrained generative policy\n"
        "Conditional uncertainty repair",
        transform=axis.transAxes, ha="center", va="bottom", fontsize=11.0,
    )
    figure.tight_layout(pad=0.2)
    figure.savefig(png, dpi=220, bbox_inches="tight", transparent=True)
    figure.savefig(pdf, bbox_inches="tight", transparent=True)
    plt.close(figure)
    return {
        "png": str(png.resolve()), "png_sha256": sha256_file(png),
        "pdf": str(pdf.resolve()), "pdf_sha256": sha256_file(pdf),
    }
