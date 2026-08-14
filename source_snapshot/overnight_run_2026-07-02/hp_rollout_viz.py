"""Quick look: random rollouts from an HP-full checkpoint, per γ, colored by validity2 (green pass / red fail),
exactly the measure protocol (temp 1.0, nfe 8, T 250). Usage: python hp_rollout_viz.py --ckpt <path> --n 8"""
import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import grid_scene as GS
import grid_rollout as GR
import grid_metrics2 as GM2
import grid_hp_expt as HP
import hp_arch_sweep as ARCH

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--tag", default="it15000")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    pol, _ = (ARCH.load_arch(a.ckpt, device=dev) if "variant" in ck else HP.load_hp(a.ckpt, device=dev))
    env = GS.make_grid()
    obs = env.obstacles.detach().cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.8))
    for ax, g in zip(axes, (0.1, 0.5, 1.0)):
        torch.manual_seed(0)
        paths = GR.deploy_many(pol, env, g, a.n, T=250, temp=1.0, nfe=8, device=dev)
        nv = 0
        for (ox, oy, r) in obs:
            ax.add_patch(Circle((ox, oy), r, facecolor="#d9c8e3", edgecolor="#9b72aa", lw=.6, alpha=.8))
        for p in paths:
            P = np.asarray(p, np.float32)
            ok = GM2.traj_valid2(P, env, g)
            ok = ok[0] if isinstance(ok, tuple) else bool(ok)
            nv += int(ok)
            ax.plot(P[:, 0], P[:, 1], "-", color="#2ca02c" if ok else "#d62728", lw=1.7, alpha=.85)
            ax.plot(P[-1, 0], P[-1, 1], "o", color="#2ca02c" if ok else "#d62728", ms=4)
        gl = env.goal.detach().cpu().numpy()
        ax.scatter([0], [0], marker="s", s=60, c="#333", zorder=6)
        ax.scatter([gl[0]], [gl[1]], marker="*", s=170, c="gold", edgecolor="k", zorder=6)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"γ={g} — {nv}/{a.n} valid2", fontsize=12,
                     color="#2ca02c" if nv >= a.n * 0.6 else ("#e08b00" if nv else "#d62728"))
    fig.suptitle(f"HP-full {a.tag}: {a.n} random rollouts per γ (measure protocol: temp 1.0, nfe 8) — "
                 "green = validity2 pass, red = fail", fontsize=12.5)
    fig.tight_layout()
    out = os.path.join(FIG, f"hp_rollouts_{a.tag}.png")
    fig.savefig(out, dpi=125, bbox_inches="tight")
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
