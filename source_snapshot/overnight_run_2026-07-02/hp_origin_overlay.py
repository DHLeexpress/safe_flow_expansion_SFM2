"""Overlay rollouts from ORIGIN (0,0), per γ, for an HP checkpoint (user 2026-07-06, GRU model).
green = validity2 pass · red = fail · N deploys overlaid (temp 1.0, the measure protocol).
Usage: python hp_origin_overlay.py --ckpt results/hp_arch/res2w256_gru_ft.pt --tag gru_ft_origin"""
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
import grid_metrics2 as GM2
import hp_arch_sweep as ARCH

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures", "dr_test_overnight"); os.makedirs(FIG, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
GAMMAS = (0.5, 1.0, 0.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--tag", default="origin_overlay")
    a = ap.parse_args()
    pol, _ = ARCH.load_arch(a.ckpt, device=DEV)
    env = GS.make_grid()
    obs = env.obstacles.detach().cpu().numpy(); goal = env.goal.detach().cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    for j, g in enumerate(GAMMAS):
        ax = axes[j]
        for o in obs:
            ax.add_patch(Circle((o[0], o[1]), o[2] if len(o) > 2 else GS.OBS_R, fc="#555", ec="none", alpha=.7))
        ax.plot(*goal, marker="*", ms=17, color="gold", mec="k", zorder=6)
        ax.plot(0, 0, "s", ms=8, color="k", zorder=6)
        torch.manual_seed(1)
        paths = GR.deploy_many(pol, env, g, a.n, T=250, temp=1.0, nfe=8, device=DEV)
        nv = nr = 0
        for p in paths:
            P = np.asarray(p, np.float32)
            reached = np.linalg.norm(P[-1, :2] - goal) < 0.5
            v = GM2.traj_valid2(P, env, g)
            valid = bool(v[0] if isinstance(v, tuple) else v)
            nv += int(valid); nr += int(reached)
            ax.plot(P[:, 0], P[:, 1], lw=1.4, alpha=.7, color="#2ca02c" if valid else "#d62728")
        ax.set_xlim(-.2, 5.2); ax.set_ylim(-.2, 5.2); ax.set_aspect("equal"); ax.grid(alpha=.15)
        ax.set_title(f"γ={g} — valid2 {nv}/{a.n} · reach {nr}/{a.n}", fontsize=11)
    fig.suptitle(f"Rollouts from ORIGIN (0,0) — {os.path.basename(a.ckpt)} (green=validity2 pass, red=fail)", fontsize=13)
    fig.tight_layout()
    out = os.path.join(FIG, f"origin_overlay_{a.tag}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
