"""The GREEN verifier polytope MOVING along an expanded-FM closed-loop rollout, validating it (per γ).

Uses the ORIGINAL ieee verifier faces (verifier_polytope.certify_window) drawn in WORLD frame at each step —
the fitted max-margin polytope EXPANDS to certify the FM trajectory (vs the blue nominal in the di_grid viz).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

import _paths
import config as C
import verifier_polytope as VP
from windowed_policy import GridLowFlowPolicy, fm_rollout
import validity as VAL


def load_expanded(scene, device):
    ck = torch.load(C.scene_result(scene, "expanded.pt"), weights_only=False)
    pol = GridLowFlowPolicy(H_pred=ck["H_pred"], u_max=ck["u_max"]).to(device)
    pol.load_state_dict(ck["state_dict"])
    pol.eval()
    return pol


def pick_valid_path(pol, env, gamma, device, tries=8):
    best = None
    for _ in range(tries):
        paths, _ = fm_rollout(pol, env, gamma, n_traj=1, temp=0.7, device=device, record=False)
        p = paths[0]
        if VAL.is_valid(p, env, gamma):
            return p
        best = p
    return best


def draw_green_verifier(ax, seg, obstacles, r_robot, gamma, xlim, ylim):
    """Green verifier level sets + tangent faces in WORLD frame for the window `seg` (seg[0]=robot)."""
    ok, faces, raw, R_eff = VP.certify_window(seg, obstacles, r_robot, gamma)
    c = np.asarray(seg[0], float)
    gx = np.linspace(*xlim, 120); gy = np.linspace(*ylim, 90)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.stack([GX.ravel() - c[0], GY.ravel() - c[1]], 1)
    vals = [(f.m - pts @ f.a) / f.m for f in faces if f.feasible and f.m > 1e-9]
    if vals:
        Hh = np.min(np.stack(vals, 1), 1).reshape(GX.shape)
        lv = sorted({round((1 - gamma) ** i, 4) for i in range(6)} | {0.0})
        ax.contourf(GX, GY, Hh, levels=lv + [1.0001], cmap="Greens", alpha=0.42, zorder=1)
        ax.contour(GX, GY, Hh, levels=[0.0], colors="#006d2c", linewidths=1.4, zorder=3)
    for f in faces:                                                    # real-obstacle tangent faces (world)
        if getattr(f, "kind", "") == "real" and f.feasible and f.m > 1e-9:
            a = f.a; b = f.m + a @ c
            if abs(a[1]) > 1e-6:
                xs = np.array(xlim); ys = (b - a[0] * xs) / a[1]
                ax.plot(xs, ys, "-", color="#006d2c", lw=0.8, alpha=0.9, zorder=4)
    return ok


def render(scene, device="cpu", log=print):
    env = C.make_scene(scene)
    pol = load_expanded(scene, device)
    data = {g: pick_valid_path(pol, env, g, device) for g in C.GAMMAS}
    obs = env.obstacles.detach().cpu().numpy()
    H = C.VERIFIER["H_win"]
    nF = max(len(p) for p in data.values())
    fig, axes = plt.subplots(1, len(C.GAMMAS), figsize=(4.3 * len(C.GAMMAS), 4.4), squeeze=False)

    def frame(f):
        for ci, g in enumerate(C.GAMMAS):
            ax = axes[0][ci]; ax.clear()
            path = data[g]; t = min(f, len(path) - 1)
            seg = path[t:min(t + H + 1, len(path))]
            if len(seg) >= 2:
                draw_green_verifier(ax, seg, obs, float(env.r_robot), g, env.xlim, env.ylim)
            for (ox, oy, rr) in obs:
                ax.add_patch(Circle((ox, oy), rr, facecolor="#c8a2c8", edgecolor="#7b3294", lw=0.7, alpha=0.7, zorder=5))
            ax.plot(path[:t + 1, 0], path[:t + 1, 1], "-", color="#e6191b", lw=1.5, zorder=7)
            ax.scatter(path[:t + 1, 0], path[:t + 1, 1], s=11, c="k", zorder=7.5)
            ax.scatter([path[t, 0]], [path[t, 1]], s=42, c="#00a000", edgecolor="k", zorder=9)
            ax.scatter([env.goal[0]], [env.goal[1]], marker="*", s=140, c="gold", edgecolor="k", zorder=9)
            ax.set_xlim(*env.xlim); ax.set_ylim(*env.ylim); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"γ={g}", fontsize=11)
        fig.suptitle(f"[{scene}] expanded FM + GREEN verifier polytope (moving, validating)   t={f}", fontsize=11)
        return []

    anim = FuncAnimation(fig, frame, frames=nF, interval=200)
    gif = C.scene_fig(scene, "stage3_green_verifier.gif")
    anim.save(gif, writer=PillowWriter(fps=6), dpi=90)
    try:
        anim.save(gif[:-4] + ".mp4", writer=FFMpegWriter(fps=10, bitrate=2400), dpi=110)
    except Exception as e:
        log(f"[mp4] skip ({e})")
    frame(nF // 2); fig.savefig(gif[:-4] + ".png", dpi=120); plt.close(fig)
    log(f"[{scene}] green-verifier viz → {gif}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="gap", choices=C.SCENE_NAMES)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    render(args.scene, args.device)


if __name__ == "__main__":
    main()
