"""Visualize OFF-DIAGONAL DR expert episodes (user 2026-07-06: |y-x| >= 0.5 starts, fixed goal).
Re-rolls the exact training seeds. -> figures/dr_test_overnight/od_data_viz.png"""
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import _paths  # noqa: F401
import grid_scene as GS
from gen_dr_data import rollout_dr

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures", "dr_test_overnight"); os.makedirs(FIG, exist_ok=True)

env = GS.make_grid(); cfg = GS.mode1_config()
goal = env.goal.detach().cpu().numpy()
obs = env.obstacles.detach().cpu().numpy()
fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
for j, g in enumerate((0.5, 1.0, 0.1)):
    ax = axes[j]
    for o in obs:
        ax.add_patch(Circle((o[0], o[1]), o[2] if len(o) > 2 else GS.OBS_R, fc="#555", ec="none", alpha=.7))
    ax.plot([0, 5], [0.5, 5.5], ls="--", c="#bbb", lw=1); ax.plot([0, 5], [-0.5, 4.5], ls="--", c="#bbb", lw=1)
    for s in range(9):
        states, _, start = rollout_dr(env, g, cfg, s, offdiag=0.5)
        ok, _ = GS.is_success(states[:, :2], env)
        ax.plot(states[:, 0], states[:, 1], lw=1.5, alpha=.8, color="#2ca02c" if ok else "#d62728")
        ax.plot(start[0], start[1], "o", ms=6, color="#1f77b4", zorder=5)
    ax.plot(*goal, marker="*", ms=17, color="gold", mec="k", zorder=6)
    ax.set_xlim(-.2, 5.2); ax.set_ylim(-.2, 5.2); ax.set_aspect("equal"); ax.grid(alpha=.15)
    ax.set_title(f"γ={g} — OD expert (|y-x|≥0.5 starts, dashed = excluded band)")
fig.suptitle("OVERNIGHT v2 data: off-diagonal domain-randomized expert episodes (goal fixed)", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "od_data_viz.png"), dpi=120, bbox_inches="tight")
print("saved figures/dr_test_overnight/od_data_viz.png", flush=True)
