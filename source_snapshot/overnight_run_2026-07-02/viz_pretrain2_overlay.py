"""Preliminary communication plot: the v2 W256 PRETRAINED policy vs the SafeMPPI EXPERT DATA it was trained
on, overlaid per γ (the pre-expansion starting point the safe-flow-expansion must broaden). 3 panels
(γ0.1/0.5/1.0): SafeMPPI successful expert trajectories (green) vs pretrained-model rollouts (orange).
Style matches overnight_run_2026-07-02's grid overlays. Read-only w.r.t. the running sweep."""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D

import _paths  # noqa: F401
import grid_scene as GS
import grid_policy2 as GP2
import grid_rollout as GR
import grid_metrics as GM

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FIG = os.path.join(os.path.dirname(__file__), "figures")
GAMMAS = [0.1, 0.5, 1.0]
N_EXPERT, N_MODEL = 22, 30


def draw_grid(ax, obs):
    for k in range(6):
        ax.axvline(k, color="#eee", lw=.6); ax.axhline(k, color="#eee", lw=.6)
    ax.add_patch(Rectangle((0, 0), 5, 5, fill=False, edgecolor="#555", lw=1.4))
    for j, (ox, oy, r) in enumerate(obs):
        ax.add_patch(Circle((ox, oy), r, facecolor="#b8b8b8" if j >= 16 else "#c8a2c8",
                            edgecolor="#777", lw=.4, alpha=.8))
    ax.scatter([0], [0], s=55, marker="s", c="#00a000", edgecolor="k", zorder=9)
    ax.scatter([5], [5], marker="*", s=200, c="gold", edgecolor="k", zorder=9)
    ax.set_xlim(-.6, 5.6); ax.set_ylim(-.6, 5.6); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def main(policy="pretrained2_w256.pt", out="prelim_pretrained2_vs_safemppi.png",
         label="v2 W256 pretrained rollouts"):
    env = GS.make_grid()
    obs = env.obstacles.numpy()
    cfg = GS.mode1_config()
    pol, _ = GP2.load_policy2(policy, device=DEV)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6))
    for ax, g in zip(axes, GAMMAS):
        draw_grid(ax, obs)
        # SafeMPPI expert successes (green)
        ne = 0
        for seed in range(80):
            if ne >= N_EXPERT:
                break
            try:
                p = GS.rollout_path(env, g, cfg, seed)
            except Exception:
                continue
            if GS.is_success(p, env):
                ax.plot(p[:, 0], p[:, 1], "-", color="#2ca02c", lw=1.1, alpha=.55, zorder=5)
                ne += 1
        # v2 W256 pretrained rollouts (orange)
        nm = 0
        for _ in range(N_MODEL):
            pth = GR.fm_deploy(pol, env, g, T=250, nfe=10, device=DEV)["path"]
            ok = GM.reaches_goal(pth, env.goal.numpy())
            ax.plot(pth[:, 0], pth[:, 1], "-", color="#ff7f0e", lw=1.0, alpha=.5, zorder=6)
            nm += int(ok)
        ax.set_title(f"γ={g}   SafeMPPI expert n={ne} · rollouts n={N_MODEL} (reach {nm}/{N_MODEL})", fontsize=10.5)
    handles = [Line2D([0], [0], color="#2ca02c", lw=2, label="SafeMPPI expert (success-only)"),
               Line2D([0], [0], color="#ff7f0e", lw=2, label=label)]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"PRELIMINARY — {label} vs SafeMPPI training data, per γ", fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    outp = os.path.join(FIG, out)
    fig.savefig(outp, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("saved", outp)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="pretrained2_w256.pt")
    ap.add_argument("--out", default="prelim_pretrained2_vs_safemppi.png")
    ap.add_argument("--label", default="v2 W256 pretrained rollouts")
    a = ap.parse_args()
    main(a.policy, a.out, a.label)
