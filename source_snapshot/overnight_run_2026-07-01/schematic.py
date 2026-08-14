"""Overall schematic of the SafeFlow Exploration Loop pipeline (large fonts, real-figure thumbnails).
Stage 2 Safe-MPPI data engine -> Stage 3 gamma-flow policy -> Stage 4 SOCP certificate -> Stage 5 expansion loop,
with a bottom strip of real result figures.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.image as mpimg

import _paths
import config as C

PAPER_IMG = os.path.join(C.ROOT, "Dohyun_ICRA2026_SafeFlowExpansion", "images")


def box(ax, x, y, w, h, title, lines, fc, ec="#2c3e50", title_fs=22, body_fs=15.5, tcolor="#1a1a1a", title_dy=0.30):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18",
                                fc=fc, ec=ec, lw=2.8, zorder=2))
    if title:
        ax.text(x + w / 2, y + h - title_dy, title, ha="center", va="top", fontsize=title_fs,
                fontweight="bold", color=tcolor, zorder=3)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 1.02 - 0.56 * i, ln, ha="center", va="top", fontsize=body_fs,
                color=tcolor, zorder=3)


def arrow(ax, p0, p1, label="", fs=16, color="#c0392b", rad=0.0, lab_dxy=(0, 0.28)):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=28, lw=3.2, color=color,
                                 connectionstyle=f"arc3,rad={rad}", zorder=4))
    if label:
        mx, my = (p0[0] + p1[0]) / 2 + lab_dxy[0], (p0[1] + p1[1]) / 2 + lab_dxy[1]
        ax.text(mx, my, label, ha="center", va="bottom", fontsize=fs, color=color, fontweight="bold", zorder=5)


def strip(fig, path, rect, caption):
    if os.path.exists(path):
        axi = fig.add_axes(rect); axi.imshow(mpimg.imread(path)); axi.axis("off")
        axi.set_title(caption, fontsize=13.5, color="#34495e", pad=3)


def main():
    fig = plt.figure(figsize=(18, 11.2))
    ax = fig.add_axes([0, 0.0, 1, 1]); ax.set_xlim(0, 18); ax.set_ylim(0, 11.2); ax.axis("off")

    ax.text(9, 11.05, "Safe Flow Expansion", ha="center", va="top", fontsize=30, fontweight="bold", color="#0b3d5c")
    ax.text(9, 10.42, "Certifiably growing a safety-tunable generative policy",
            ha="center", va="top", fontsize=18, color="#2471a3", style="italic")

    # ---- top row: three stages
    box(ax, 0.4, 6.7, 5.0, 2.9, "Stage 2 · Safe-MPPI Data Engine",
        ["MPPI + polytope-DCBF rejection", "averages only certified rollouts",
         r"$\Rightarrow$ multi-modal safe demos $\mathcal{D}$"], fc="#e8f4fd")
    ax.text(2.55, 6.52, r"samples $\sim\mathcal{N}(\mu,\Sigma)$ via the free-space polytope",
            ha="center", va="top", fontsize=12.5, style="italic", color="#34495e")

    box(ax, 6.7, 6.7, 4.6, 2.9, r"Stage 3 · $\gamma$-Flow Policy",
        [r"noise $\epsilon \Rightarrow$ control window $U$", r"cond. on (polar grid $o$, $\gamma$, ctx $c$)",
         "windowed, real-time, no sampling"], fc="#eafaf1")

    box(ax, 12.6, 6.7, 5.0, 2.9, "Stage 4 · SOCP Certificate",
        [r"max-margin polytope, $K{=}12$ anchors", r"$H_{\mathcal{P}}(q_t)\geq(1-\gamma)^t$, closed form",
         r"a certified sample $\Rightarrow$ provably safe"], fc="#fdf2e9")

    arrow(ax, (5.4, 8.15), (6.7, 8.15), r"demos $\mathcal{D}$", color="#2471a3")
    arrow(ax, (11.3, 8.15), (12.6, 8.15), "trajectory", color="#2471a3")

    # gamma knob feeding the policy (clear of the subtitle)
    ax.add_patch(Circle((8.35, 9.82), 0.13, fc="#f1c40f", ec="#b7950b", lw=2.2, zorder=6))
    ax.text(8.55, 9.82, r"safety knob $\gamma\in(0,1]$", ha="left", va="center", fontsize=13.5,
            fontweight="bold", color="#7d6608")
    arrow(ax, (8.35, 9.70), (8.35, 9.61), color="#b7950b", label="")

    # ---- Stage 5 loop box
    box(ax, 1.7, 2.9, 14.6, 3.0, "", [], fc="#f4ecf7", ec="#7d3c98")
    ax.text(9, 5.74, "Stage 5 · SAFE FLOW EXPANSION LOOP",
            ha="center", va="top", fontsize=20, fontweight="bold", color="#5b2c6f")
    ax.text(9, 5.30, "iterate until the certified coverage fills the space near the obstacles",
            ha="center", va="top", fontsize=14, style="italic", color="#5b2c6f")
    nodes = [(2.5, "GENERATE", "policy samples\ncandidate trajectories", "#d6eaf8"),
             (7.2, "CERTIFY", "SOCP verifier\ngreen = safe / red = reject", "#d5f5e3"),
             (11.9, "EXPAND", "mode-balanced replay\non certified positives", "#fadbd8")]
    for x, t, sub, c in nodes:
        box(ax, x, 3.15, 3.6, 1.55, "", [], fc=c, ec="#5b2c6f")
        ax.text(x + 1.8, 4.45, t, ha="center", va="top", fontsize=17, fontweight="bold", color="#4a235a")
        ax.text(x + 1.8, 3.95, sub, ha="center", va="top", fontsize=12.5, color="#4a235a")
    arrow(ax, (6.1, 3.92), (7.2, 3.92), color="#7d3c98", label="")
    arrow(ax, (10.8, 3.92), (11.9, 3.92), color="#7d3c98", label="")
    # loop-back EXPAND -> GENERATE (rectangular route above the nodes)
    ax.plot([13.7, 13.7], [4.7, 4.98], color="#7d3c98", lw=3.0, zorder=4, solid_capstyle="round")
    ax.plot([13.7, 4.3], [4.98, 4.98], color="#7d3c98", lw=3.0, zorder=4, solid_capstyle="round")
    ax.add_patch(FancyArrowPatch((4.3, 4.98), (4.3, 4.72), arrowstyle="-|>", mutation_scale=26, lw=3.0,
                                 color="#7d3c98", zorder=4))
    ax.text(9, 4.98, "repeat", ha="center", va="center", fontsize=14, style="italic", fontweight="bold",
            color="#7d3c98", bbox=dict(fc="#f4ecf7", ec="none", pad=1), zorder=6)

    # couple loop to pipeline
    arrow(ax, (15.1, 6.7), (15.1, 5.9), r"query $y=\tilde v(x)$", color="#7d3c98", lab_dxy=(1.4, -0.35))
    arrow(ax, (3.5, 5.9), (8.0, 6.7), "", color="#27ae60", rad=-0.12)
    ax.text(6.15, 6.86, r"UpdateFlow $\theta_{t+1}$", ha="center", fontsize=15, fontweight="bold", color="#27ae60")

    # ---- bottom results strip (real figures)
    ax.text(9, 2.62, "results", ha="center", va="top", fontsize=15, fontweight="bold", color="#7d3c98")
    strip(fig, os.path.join(PAPER_IMG, "socp_narrow_gap.png"), [0.05, 0.035, 0.28, 0.165],
          "SOCP: radial fails, variable-tangent certifies")
    strip(fig, os.path.join(PAPER_IMG, "slalom_before_after.png"), [0.37, 0.035, 0.26, 0.16],
          "pretrained band  →  down / weave / over-the-top (certified)")
    strip(fig, os.path.join(PAPER_IMG, "slalom_2x2.png"), [0.69, 0.025, 0.235, 0.185],
          "coverage grows; kernel gains structure as modes emerge")

    out = os.path.join(C.FIGURES, "schematic_overview.png")
    fig.savefig(out, dpi=150); fig.savefig(out[:-4] + ".pdf")
    fig.savefig(os.path.join(PAPER_IMG, "schematic_overview.png"), dpi=150)
    plt.close(fig)
    print(f"schematic -> {out}  (+ paper images/)")


if __name__ == "__main__":
    main()
