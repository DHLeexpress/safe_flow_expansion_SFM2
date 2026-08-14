"""Safe-expansion EXPLORATION animation — at EVERY receding-horizon step the FM proposes candidate windows,
the GREEN verifier polytope (moving with the robot) certifies them (green) or rejects (red), and the
executed path grows. One column per γ. mp4 + gif.
"""
from __future__ import annotations

import argparse

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
from polar_grid import polar_grid
from local_frame import low_dim_features, goal_frame, to_world
from di_grid_viz import di_step
from green_verifier_viz import load_expanded, draw_green_verifier


def _minclr(seg, obs, r_robot):
    if not len(obs):
        return 9.0
    return float((np.linalg.norm(seg[:, None, :] - obs[None, :, :2], axis=2) - obs[None, :, 2] - r_robot).min())


@torch.no_grad()
def explore_rollout(pol, env, gamma, n_cand, device, reach=0.4):
    goal = env.goal.detach().cpu().numpy(); obs = env.obstacles.detach().cpu().numpy(); rr = float(env.r_robot)
    st = env.x0.detach().cpu().numpy().astype(np.float32)
    exec_path, frames, a_prev, pv = [st[:2].copy()], [], None, False
    for t in range(env.T):
        grid, _ = polar_grid(st[:2], goal, obs, r_robot=rr)
        low, _ = low_dim_features(st, goal, gamma, a_prev=a_prev, prev_valid=pv)
        ctx = pol.ctx_from(torch.tensor(grid[None], device=device), torch.tensor(low[None], device=device))
        e_g, e_lat, _ = goal_frame(st[:2], goal)
        cand_Ul = pol.sample(n_cand, ctx.expand(n_cand, -1), temp=1.5).detach().cpu().numpy()
        cand, certs = [], []
        for Ul in cand_Ul:
            Uw = to_world(Ul, e_g, e_lat); s2 = st.copy(); pp = [s2[:2].copy()]
            for u in Uw:
                s2 = di_step(s2, np.clip(u, -env.u_max, env.u_max), dt=env.dt); pp.append(s2[:2].copy())
            seg = np.array(pp)
            ok, *_ = VP.certify_window(seg, obs, rr, gamma)
            cand.append(seg); certs.append(bool(ok) and _minclr(seg, obs, rr) >= 0)
        exec_Uw = to_world(pol.sample(1, ctx, temp=0.6).detach().cpu().numpy()[0], e_g, e_lat)
        s2 = st.copy(); pp = [s2[:2].copy()]
        for u in exec_Uw:
            s2 = di_step(s2, np.clip(u, -env.u_max, env.u_max), dt=env.dt); pp.append(s2[:2].copy())
        frames.append(dict(pos=st[:2].copy(), cand=cand, certs=certs, exec_seg=np.array(pp)))
        u0 = np.clip(exec_Uw[0], -env.u_max, env.u_max)
        st = di_step(st, u0, dt=env.dt); a_prev = u0; pv = True; exec_path.append(st[:2].copy())
        if np.linalg.norm(st[:2] - goal) < reach:
            break
    return np.array(exec_path), frames


def render(scene, device="cpu", n_cand=14, log=print):
    env = C.make_scene(scene); pol = load_expanded(scene, device)
    obs = env.obstacles.detach().cpu().numpy()
    data = {g: explore_rollout(pol, env, g, n_cand, device) for g in C.GAMMAS}
    nF = max(len(f) for _, f in data.values())
    fig, axes = plt.subplots(1, len(C.GAMMAS), figsize=(4.4 * len(C.GAMMAS), 4.7), squeeze=False)

    def frame(i):
        for ci, g in enumerate(C.GAMMAS):
            ax = axes[0][ci]; ax.clear()
            exec_path, frames = data[g]; t = min(i, len(frames) - 1); fr = frames[t]
            draw_green_verifier(ax, fr["exec_seg"], obs, float(env.r_robot), g, env.xlim, env.ylim)
            for (ox, oy, rr) in obs:
                ax.add_patch(Circle((ox, oy), rr, facecolor="#c8a2c8", edgecolor="#7b3294", lw=0.7, alpha=0.7, zorder=5))
            for seg, ok in zip(fr["cand"], fr["certs"]):
                ax.plot(seg[:, 0], seg[:, 1], "-", color=("#00a000" if ok else "#d62728"),
                        lw=0.8, alpha=(0.75 if ok else 0.3), zorder=6)
            ax.plot(exec_path[:t + 1, 0], exec_path[:t + 1, 1], "-", color="#e6191b", lw=1.6, zorder=8)
            ax.scatter(exec_path[:t + 1, 0], exec_path[:t + 1, 1], s=11, c="k", zorder=8.5)
            ax.scatter([fr["pos"][0]], [fr["pos"][1]], s=42, c="#00a000", edgecolor="k", zorder=9)
            ax.scatter([env.goal[0]], [env.goal[1]], marker="*", s=140, c="gold", edgecolor="k", zorder=9)
            ax.text(0.02, 0.97, f"cert {sum(fr['certs'])}/{len(fr['certs'])}", transform=ax.transAxes,
                    va="top", fontsize=8, bbox=dict(boxstyle="round", fc="white", alpha=0.75))
            ax.set_xlim(*env.xlim); ax.set_ylim(*env.ylim); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"γ={g}", fontsize=11)
        fig.suptitle(f"[{scene}] safe-expansion EXPLORATION — GREEN verifier polytope + candidate windows "
                     f"(green=certified / red=rejected)   t={i}", fontsize=9.5)
        return []

    anim = FuncAnimation(fig, frame, frames=nF, interval=180)
    gif = C.scene_fig(scene, "exploration_process.gif")
    anim.save(gif, writer=PillowWriter(fps=6), dpi=90)
    try:
        anim.save(gif[:-4] + ".mp4", writer=FFMpegWriter(fps=10, bitrate=2600), dpi=110)
    except Exception as e:
        log(f"[mp4] skip ({e})")
    frame(nF // 2); fig.savefig(gif[:-4] + ".png", dpi=120); plt.close(fig)
    log(f"[{scene}] exploration process → {gif}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="gap", choices=C.SCENE_NAMES)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    render(args.scene, args.device)


if __name__ == "__main__":
    main()
