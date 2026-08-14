"""PHASE-DR overlay test (user 2026-07-06): 3 rows x 3 γ.
Row 1: DR expert episodes (re-rolled with the SAME seeds as training data — random free-space starts).
Row 2: ORIGINAL res2w256_ft rollouts from (0,0)   } prediction: both diagonal — the spliced encoder must
Row 3: SPLICED res2w256_dr rollouts from (0,0)    } preserve in-distribution behavior (field drives).
Usage: python hp_dr_overlay.py [--n-fm 16 --n-expert 8]"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import _paths  # noqa: F401
import grid_scene as GS
import grid_rollout as GR
import hp_arch_sweep as ARCH
from gen_dr_data import rollout_dr

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures", "hp_test")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
GAMMAS = (0.5, 1.0, 0.1)


def draw_scene(ax, env):
    obs = env.obstacles.detach().cpu().numpy()
    for o in obs:
        ax.add_patch(Circle((o[0], o[1]), o[2] if len(o) > 2 else GS.OBS_R, fc="#555", ec="none", alpha=.75))
    g = env.goal.detach().cpu().numpy()
    ax.plot(*g, marker="*", ms=18, color="gold", mec="k", zorder=6)
    ax.set_xlim(-0.2, 5.2); ax.set_ylim(-0.2, 5.2); ax.set_aspect("equal"); ax.grid(alpha=.15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", default="results/hp_arch/res2w256_ft.pt")
    ap.add_argument("--spliced", default="results/hp_arch/res2w256_dr.pt")
    ap.add_argument("--n-fm", type=int, default=16)
    ap.add_argument("--n-expert", type=int, default=8)
    ap.add_argument("--tag", default="dr_overlay")
    a = ap.parse_args()
    env = GS.make_grid(); cfg = GS.mode1_config()
    pol_o, _ = ARCH.load_arch(a.orig, device=DEV)
    pol_s, _ = ARCH.load_arch(a.spliced, device=DEV)

    fig, axes = plt.subplots(3, 3, figsize=(15.5, 15))
    rows = ("DR EXPERT (random free-space starts, training data)",
            "ORIGINAL res2w256_ft from (0,0)", "DR-SPLICED res2w256_dr from (0,0)")
    goal = env.goal.detach().cpu().numpy()
    for j, g in enumerate(GAMMAS):
        ax = axes[0, j]; draw_scene(ax, env)
        for s in range(a.n_expert):
            states, _, start = rollout_dr(env, g, cfg, s)
            ok, _ = GS.is_success(states[:, :2], env)
            ax.plot(states[:, 0], states[:, 1], lw=1.4, alpha=.75,
                    color="#2ca02c" if ok else "#d62728")
            ax.plot(start[0], start[1], "o", ms=5, color="#1f77b4", zorder=5)
        ax.set_title(f"γ={g} — expert, random starts", fontsize=11)
        for i, pol in ((1, pol_o), (2, pol_s)):
            ax = axes[i, j]; draw_scene(ax, env)
            with torch.no_grad():
                paths = GR.deploy_many(pol, env, g, a.n_fm, T=env.T, temp=1.0, nfe=8, device=DEV)
            n_reach = 0
            for p in paths:
                xy = np.asarray(p["states"] if isinstance(p, dict) else p)[:, :2]
                reached = np.linalg.norm(xy - goal[None], axis=1).min() < 0.4
                n_reach += int(reached)
                ax.plot(xy[:, 0], xy[:, 1], lw=1.2, alpha=.6,
                        color="#2ca02c" if reached else "#9467bd")
            ax.set_title(f"γ={g} — reach {n_reach}/{a.n_fm}", fontsize=11)
    for i, r in enumerate(rows):
        axes[i, 0].set_ylabel(r, fontsize=10.5)
    fig.suptitle("PHASE-DR overlay test: expert DR data · original vs DR-spliced policy from (0,0) "
                 "(prediction: rows 2≈3 diagonal — encoder swap preserves in-dist behavior)", fontsize=13)
    fig.tight_layout()
    out = os.path.join(FIG, f"{a.tag}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
