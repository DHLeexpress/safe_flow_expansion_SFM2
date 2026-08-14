"""Show the trained FM generating the THREE certified modes (around_down / weave / around_up) per gamma,
each with its moving green verifier polytope. mp4 + gif.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

import _paths
import config as C
import coverage as COV
import validity as VAL
import verifier_polytope as VP
from windowed_policy import GridLowFlowPolicy, fm_rollout

MODES = ["around_down", "weave", "around_up"]
MCOL = {"around_down": "#1f77b4", "weave": "#ff7f0e", "around_up": "#9467bd"}
MLAB = {"around_down": "around-down", "weave": "weave (through gap)", "around_up": "over-the-top"}


def load_expanded(scene, device):
    ck = torch.load(C.scene_result(scene, "expanded.pt"), weights_only=False)
    pol = GridLowFlowPolicy(H_pred=ck["H_pred"], u_max=ck["u_max"]).to(device)
    pol.load_state_dict(ck["state_dict"]); pol.eval()
    return pol


@torch.no_grad()
def get_mode_trajs(pol, env, gamma, device, budget=420, log=print):
    """Sample (temperature-escalating) until one CERTIFIED trajectory of each mode is found."""
    found, per = {}, 40
    for k in range(budget // per):
        temp = min(2.0, 1.35 + 0.09 * k)
        paths, _ = fm_rollout(pol, env, gamma, n_traj=per, temp=temp, device=device, record=False)
        for p in paths:
            m = COV.mode_of(p, env)
            if m in MODES and m not in found and VAL.is_valid(p, env, gamma):
                found[m] = p.astype(np.float32)
        if all(m in found for m in MODES):
            break
    log(f"  gamma {gamma}: found modes {sorted(found)} (of {MODES})")
    return found


def poly_boundary(ax, seg, obs, r_robot, gamma, xlim, ylim, color="#2ca02c"):
    ok, faces, *_ = VP.certify_window(seg, obs, r_robot, gamma)
    c = np.asarray(seg[0], float)
    gx = np.linspace(*xlim, 110); gy = np.linspace(*ylim, 84)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.stack([GX.ravel() - c[0], GY.ravel() - c[1]], 1)
    vals = [(f.m - pts @ f.a) / f.m for f in faces if f.feasible and f.m > 1e-9]
    if vals:
        Hh = np.min(np.stack(vals, 1), 1).reshape(GX.shape)
        ax.contour(GX, GY, Hh, levels=[0.0], colors=color, linewidths=1.2, alpha=0.6, zorder=3)


def render(scene="slalom", device="cpu", log=print):
    import os
    import pickle
    env = C.make_scene(scene); pol = load_expanded(scene, device)
    obs = env.obstacles.detach().cpu().numpy(); H = C.VERIFIER["H_win"]
    cache = C.scene_result(scene, "modes_cache.pkl")
    if os.path.exists(cache):
        data = pickle.load(open(cache, "rb")); log("[cache] loaded mode trajectories")
    else:
        data = {g: get_mode_trajs(pol, env, g, device, log=log) for g in C.GAMMAS}
        pickle.dump(data, open(cache, "wb"))
    nF = max([len(p) for g in C.GAMMAS for p in data[g].values()] + [2])
    fig, axes = plt.subplots(1, len(C.GAMMAS), figsize=(4.5 * len(C.GAMMAS), 4.4), squeeze=False)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.14, wspace=0.06)
    handles = [Line2D([0], [0], color=MCOL[m], lw=3.6, label=MLAB[m]) for m in MODES]
    handles.append(Line2D([0], [0], color="#2ca02c", lw=1.6, label="verifier polytope"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=12, framealpha=0.9)

    def frame(f):
        for ci, g in enumerate(C.GAMMAS):
            ax = axes[0][ci]; ax.clear()
            for (ox, oy, rr) in obs:
                ax.add_patch(Circle((ox, oy), rr, facecolor="#c8a2c8", edgecolor="#7b3294", lw=0.8, alpha=0.7, zorder=4))
            for m in MODES:
                if m not in data[g]:
                    continue
                p = data[g][m]; t = min(f, len(p) - 1)
                seg = p[t:min(t + H + 1, len(p))]
                if len(seg) >= 2:
                    poly_boundary(ax, seg, obs, float(env.r_robot), g, env.xlim, env.ylim)
                ax.plot(p[:t + 1, 0], p[:t + 1, 1], "-", color=MCOL[m], lw=3.4, zorder=8)
                ax.scatter(p[:t + 1, 0][::2], p[:t + 1, 1][::2], s=4, c="k", zorder=6, alpha=0.4)
                ax.scatter([p[t, 0]], [p[t, 1]], s=48, c=MCOL[m], edgecolor="k", zorder=9)
            ax.scatter([env.x0[0]], [env.x0[1]], s=42, c="#00a000", edgecolor="k", zorder=9)
            ax.scatter([env.goal[0]], [env.goal[1]], marker="*", s=150, c="gold", edgecolor="k", zorder=9)
            ax.set_xlim(*env.xlim); ax.set_ylim(*env.ylim); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"γ={g}   ({len(data[g])}/3 certified modes)", fontsize=12)
        fig.suptitle(f"[{scene}] trained FM generates the 3 certified modes per γ   (t={f})", fontsize=13)
        return []

    anim = FuncAnimation(fig, frame, frames=nF, interval=180)
    gif = C.scene_fig(scene, "exploration_modes.gif")
    anim.save(gif, writer=PillowWriter(fps=6), dpi=90)
    try:
        anim.save(gif[:-4] + ".mp4", writer=FFMpegWriter(fps=10, bitrate=2600), dpi=112)
    except Exception as e:
        log(f"[mp4] skip ({e})")
    frame(nF - 1); fig.savefig(gif[:-4] + ".png", dpi=125); plt.close(fig)
    log(f"[{scene}] 3-mode viz → {gif}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="slalom", choices=C.SCENE_NAMES)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    render(args.scene, args.device)


if __name__ == "__main__":
    main()
