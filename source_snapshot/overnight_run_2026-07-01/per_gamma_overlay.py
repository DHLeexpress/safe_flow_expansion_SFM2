"""Per-γ SafeMPPI rollout OVERLAY (many seeds) to see the behavior distribution:
above / below / thread / collide / stuck. Answers 'is there only above/below?'.
"""
from __future__ import annotations

import argparse
import os
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import _paths
import scenes
from di_grid_viz import load_best_config, mppi_rollout

COLORS = {"above": "#1f77b4", "below": "#d62728", "thread": "#2ca02c",
          "collide": "k", "stuck": "0.6", "other": "orange"}


def classify(path, env):
    obs = env.obstacles.detach().cpu().numpy()
    mc = float((np.linalg.norm(path[:, None, :] - obs[None, :, :2], axis=2) - obs[None, :, 2]).min() - float(env.r_robot))
    reached = bool(np.linalg.norm(path[-1] - env.goal.detach().cpu().numpy()) < 0.5)
    # side at x = obstacle x-plane
    ox = float(obs[0, 0]); r = float(obs[:, 2].max())
    cross_y = None
    for i in range(len(path) - 1):
        if (path[i, 0] - ox) * (path[i + 1, 0] - ox) <= 0 and path[i + 1, 0] != path[i, 0]:
            t = (ox - path[i, 0]) / (path[i + 1, 0] - path[i, 0])
            cross_y = float(path[i, 1] + t * (path[i + 1, 1] - path[i, 1]))
            break
    if mc < 0:
        return "collide", cross_y
    if not reached:
        return "stuck", cross_y
    if cross_y is None:
        return "other", cross_y
    if env.name == "narrow_gap":
        _, _, half_body, _ = scenes.gap_geometry(env)
        if abs(cross_y) <= half_body:
            return "thread", cross_y
        return "above" if cross_y > 0 else "below", cross_y
    # single obstacle: above/below by sign, else it clipped the disk edge
    return ("above" if cross_y > 0 else "below"), cross_y


def render(env, gammas, n_seeds, out, title):
    cfg = load_best_config()
    fig, axes = plt.subplots(1, len(gammas), figsize=(4.3 * len(gammas), 4.2), squeeze=False)
    obs = env.obstacles.detach().cpu().numpy()
    all_counts = {}
    for ci, g in enumerate(gammas):
        ax = axes[0][ci]
        for (ox, oy, rr) in obs:
            ax.add_patch(Circle((ox, oy), rr, facecolor="#c8a2c8", edgecolor="#7b3294", lw=0.7, alpha=0.7, zorder=3))
        cnt = Counter()
        for seed in range(n_seeds):
            _, path = mppi_rollout(env, g, cfg, seed_base=seed * 97 + 1)
            cls, _cy = classify(path, env)
            cnt[cls] += 1
            ax.plot(path[:, 0], path[:, 1], "-", color=COLORS[cls], lw=0.7, alpha=0.5, zorder=4)
        ax.scatter([env.x0[0]], [env.x0[1]], s=45, c="#00a000", edgecolor="k", zorder=6)
        ax.scatter([env.goal[0]], [env.goal[1]], marker="*", s=140, c="gold", edgecolor="k", zorder=6)
        ax.set_xlim(*env.xlim); ax.set_ylim(*env.ylim); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        summ = "  ".join(f"{k}:{v}" for k, v in sorted(cnt.items()))
        ax.set_title(f"γ={g}\n{summ}", fontsize=9)
        all_counts[g] = dict(cnt)
    handles = [plt.Line2D([], [], color=c, lw=2, label=k) for k, c in COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return all_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=24)
    args = ap.parse_args()
    gammas = [0.1, 0.5, 1.0]
    cases = [
        ("single_r0.6", scenes.make_single_obstacle(r=0.6), "SINGLE obstacle r=0.6 (expect above/below)"),
        ("wall_gap_off0.35", scenes.make_narrow_gap(gap_offset=0.35, gap_r=0.35), "WALL-like gap off=0.35 r=0.35 (touching)"),
        ("gap_off0.66", scenes.make_narrow_gap(gap_offset=0.66, gap_r=0.35), "THREADABLE gap off=0.66 (corridor 0.11)"),
    ]
    for name, env, title in cases:
        out = os.path.join(_paths.HERE, "figures", f"overlay_{name}.png")
        counts = render(env, gammas, args.seeds, out, title)
        print(f"{name}: {counts}", flush=True)
        print("  saved", out, flush=True)


if __name__ == "__main__":
    main()
