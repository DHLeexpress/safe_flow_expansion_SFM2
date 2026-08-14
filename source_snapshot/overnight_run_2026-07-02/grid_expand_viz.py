"""Stage G viz — per-γ coverage/validity plots + the safe-expansion movie.

The movie shows, per γ side-by-side, the CERTIFIED GENERABLE SET expanding over ACTFLOW iterations: each
covered monotone staircase is drawn as a faint right/up lattice path (a discovered 'mode'); the fan grows
as coverage climbs. Current no-tilt sample trajectories (the real safe weaving paths) are overlaid, with the
live coverage/validity counters.
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

GAMMA_COLORS = {0.1: "#3b6fd6", 0.5: "#2ca02c", 1.0: "#d62728"}


def _grid(ax, obs, n_interior=16, gm=5.0):
    for k in range(6):
        ax.axvline(k, color="#ededed", lw=0.7, zorder=0); ax.axhline(k, color="#ededed", lw=0.7, zorder=0)
    ax.add_patch(Rectangle((0, 0), gm, gm, fill=False, edgecolor="#555", lw=1.6, zorder=0.5))
    for j, (ox, oy, r) in enumerate(obs):
        wall = j >= n_interior
        ax.add_patch(Circle((ox, oy), r, facecolor="#b8b8b8" if wall else "#c8a2c8",
                            edgecolor="#6f6f6f" if wall else "#7b3294", lw=0.5, alpha=0.85, zorder=3))
    ax.scatter([0], [0], s=55, marker="s", c="#00a000", edgecolor="k", zorder=7)
    ax.scatter([gm], [gm], marker="*", s=170, c="gold", edgecolor="k", zorder=7)
    ax.set_xlim(-0.7, gm + 0.7); ax.set_ylim(-0.7, gm + 0.7); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


def word_to_poly(word):
    x = y = 0.0
    pts = [(0.0, 0.0)]
    for c in word:
        if c == "R":
            x += 1
        else:
            y += 1
        pts.append((x, y))
    return np.array(pts, float)


def _stair_offset(word):
    h = (hash(word) % 1000) / 1000.0 - 0.5
    return 0.18 * h


def plot_metrics(hist_by_gamma, out_cov, out_val):
    for metric, out, ylab in [("coverage", out_cov, "coverage  (distinct staircases / 252)"),
                              ("validity", out_val, "validity  (fraction of deploys valid)")]:
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        for g, hist in hist_by_gamma.items():
            it = [h["iter"] for h in hist]; yv = [h[metric] * 100 for h in hist]
            ax.plot(it, yv, "-o", ms=4, lw=2, color=GAMMA_COLORS.get(g, None), label=f"γ={g}")
        ax.axhline(90, ls="--", color="#888", lw=1.2, label="90% goal")
        ax.set_xlabel("ACTFLOW iteration"); ax.set_ylabel(ylab); ax.set_ylim(0, 101)
        ax.legend(loc="best", fontsize=10); ax.grid(alpha=0.25)
        ax.set_title(f"Safe flow expansion — {metric} per γ (5×5 grid, 252 staircases)")
        fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def expansion_movie(snap_by_gamma, obs, out, fps=2, n_interior=16):
    gammas = list(snap_by_gamma.keys())
    nfr = max(len(s) for s in snap_by_gamma.values())
    fig, axes = plt.subplots(1, len(gammas), figsize=(4.5 * len(gammas), 4.9), squeeze=False)

    def frame(f):
        for ci, g in enumerate(gammas):
            ax = axes[0][ci]; ax.clear()
            snaps = snap_by_gamma[g]; s = snaps[min(f, len(snaps) - 1)]
            _grid(ax, obs, n_interior)
            for word in s["covered"]:                                  # the growing generable set (faint fan)
                p = word_to_poly(word); off = _stair_offset(word)
                ax.plot(p[:, 0] + off, p[:, 1] - off, "-", color=GAMMA_COLORS.get(g), lw=0.6, alpha=0.28, zorder=2)
            for path in s["paths"]:                                    # current safe deploys (bold)
                path = np.asarray(path)
                ax.plot(path[:, 0], path[:, 1], "-", color=GAMMA_COLORS.get(g), lw=1.4, alpha=0.9, zorder=5)
            cov = len(s["covered"]) / 252 * 100
            ax.set_title(f"γ={g}   iter {s['iter']}\ncoverage {cov:.0f}%  ({len(s['covered'])}/252)", fontsize=11)
        fig.suptitle("Safe Flow Expansion — certified generable set growing over ACTFLOW iterations", fontsize=13)
        return []

    anim = FuncAnimation(fig, frame, frames=nfr, interval=500)
    anim.save(out, writer=PillowWriter(fps=fps), dpi=90)
    try:
        anim.save(out[:-4] + ".mp4", writer=FFMpegWriter(fps=max(fps, 2), bitrate=2400), dpi=110)
    except Exception as e:
        print(f"[mp4] skip ({e})")
    plt.close(fig)


if __name__ == "__main__":
    import grid_scene as GS
    env = GS.make_grid(); obs = env.obstacles.numpy()
    demo_hist = {0.5: [dict(iter=i * 50, coverage=min(0.9, i * 0.1), validity=0.9) for i in range(6)]}
    plot_metrics(demo_hist, "/tmp/cov.png", "/tmp/val.png")
    words = ["RRRRRUUUUU", "RURURURURU", "UUUUURRRRR", "RRUURRUURU"]
    snaps = {0.5: [dict(iter=i * 50, covered=words[:i + 1],
                        paths=[np.array([[j * 0.1, j * 0.1] for j in range(51)])]) for i in range(4)]}
    expansion_movie(snaps, obs, "/tmp/exp.gif")
    print("viz smoke OK ->", os.path.exists("/tmp/exp.gif"), os.path.exists("/tmp/exp.mp4"))
