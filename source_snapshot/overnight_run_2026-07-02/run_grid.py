"""Deliverables for 2026-07-02:
 1) render_gamma_sweep  -> figures/grid_gamma_sweep.{mp4,gif,png}: gamma {0.1,0.5,1.0} simultaneously.
 2) overlay_trajectories-> figures/grid_overlay.png: success-only trajectories, gamma {0.1,0.2,0.4,0.8,1.0}.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

import _paths  # noqa: F401
import di_grid_viz as DV
import grid_scene as GS

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)


def _grid_lines(ax, n=5):
    from matplotlib.patches import Rectangle
    for k in range(n + 1):
        ax.axvline(k, color="#e8e8e8", lw=0.8, zorder=0)
        ax.axhline(k, color="#e8e8e8", lw=0.8, zorder=0)
    ax.add_patch(Rectangle((0, 0), n, n, fill=False, edgecolor="#555", lw=1.8, zorder=0.5))  # grid boundary (off = fail)


def _draw_obs(ax, obs, grid=5.0):
    for (ox, oy, r) in obs:
        wall = (ox < 0) or (ox > grid) or (oy < 0) or (oy > grid)
        fc, ec = ("#b3b3b3", "#6f6f6f") if wall else ("#c8a2c8", "#7b3294")
        ax.add_patch(Circle((ox, oy), r, facecolor=fc, edgecolor=ec, lw=0.6, alpha=0.85, zorder=4))


def _good_rollout(env_plan, g, cfg, env_true, tries=18):
    best, bestd = None, 1e9
    for s in range(tries):
        rec, path = DV.mppi_rollout(env_plan, g, cfg, seed_base=s * 13 + 1)   # plan on inflated obstacles
        ok, _ = GS.is_success(path, env_true)                                  # success at the true radius
        if ok:
            return rec, path
        d = float(np.linalg.norm(path[-1] - env_true.goal.detach().cpu().numpy()))
        if d < bestd:
            bestd, best = d, (rec, path)
    return best


def render_gamma_sweep(env, cfg, gammas, out, fps=9, log=print):
    obs = env.obstacles.detach().cpu().numpy(); rr = float(env.r_robot)
    goal = env.goal.detach().cpu().numpy(); x0 = env.x0.detach().cpu().numpy()
    env_plan = GS.inflated_env(env)
    data = {g: _good_rollout(env_plan, g, cfg, env) for g in gammas}
    for g in gammas:
        ok, clr = GS.is_success(data[g][1], env)
        log(f"  gamma {g}: success={ok} clearance={clr:+.2f}")
    # trim to when motion effectively stops
    def reach_len(path):
        for t in range(1, len(path)):
            if np.linalg.norm(path[t] - goal) < 0.45:
                return min(len(path), t + 6)
        return len(path)
    nF = max(reach_len(data[g][1]) for g in gammas)
    frames = list(range(0, nF, 2))
    fig, axes = plt.subplots(1, len(gammas), figsize=(4.4 * len(gammas), 4.8), squeeze=False)

    def frame(f):
        for ci, g in enumerate(gammas):
            ax = axes[0][ci]; ax.clear()
            rec, path = data[g]; st = rec[min(f, len(rec) - 1)]; p = st["p"]
            _grid_lines(ax)
            if st["poly"] is not None:
                gx = np.linspace(*env.xlim, 120); gy = np.linspace(*env.ylim, 120)
                GX, GY = np.meshgrid(gx, gy)
                Hh = DV.H_grid_nominal(st["poly"], GX, GY)
                lv = sorted({round((1 - g) ** i, 4) for i in range(8)} | {0.0})
                ax.contourf(GX, GY, Hh, levels=lv + [1.0001], cmap="Blues", alpha=0.40, zorder=1)
                ax.contour(GX, GY, Hh, levels=[0.0], colors="#08306b", linewidths=0.9, zorder=3)
            _draw_obs(ax, obs)
            traj, feas = st["traj"], st["feas"]
            for k in np.where(feas)[0][:35]:
                ax.plot(traj[k, :, 0], traj[k, :, 1], "-", color="#2ca02c", lw=0.5, alpha=0.45, zorder=5)
            hist = path[:min(f, len(path) - 1) + 1]
            ax.plot(hist[:, 0], hist[:, 1], "-", color="#e6191b", lw=1.6, zorder=7)
            ax.scatter(hist[:, 0], hist[:, 1], s=9, c="k", zorder=7.5)
            ax.add_patch(Circle((p[0], p[1]), rr, facecolor="#00a000", edgecolor="k", alpha=0.9, zorder=9))
            ax.scatter([x0[0]], [x0[1]], s=55, marker="s", c="#00a000", edgecolor="k", zorder=8)
            ax.scatter([goal[0]], [goal[1]], marker="*", s=200, c="gold", edgecolor="k", zorder=9)
            ax.set_xlim(*env.xlim); ax.set_ylim(*env.ylim); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            tag = "conservative" if g < 0.3 else ("aggressive" if g > 0.7 else "balanced")
            ax.set_title(f"γ={g}  ({tag})", fontsize=12)
        fig.suptitle(f"SafeMPPI (mode-1 Gaussian, range 2 m) — 5×5 grid, robot r={rr:g} (point)   t={f}", fontsize=12)
        return []

    anim = FuncAnimation(fig, frame, frames=frames, interval=140)
    anim.save(out, writer=PillowWriter(fps=fps), dpi=90)
    try:
        anim.save(out[:-4] + ".mp4", writer=FFMpegWriter(fps=max(fps, 12), bitrate=2600), dpi=110)
    except Exception as e:
        log(f"[mp4] skip ({e})")
    frame(frames[len(frames) // 2]); fig.savefig(out[:-4] + ".png", dpi=125); plt.close(fig)
    log(f"gamma sweep -> {out}")


def overlay_trajectories(env, cfg, gammas, n_target=30, max_seeds=90, out=None, log=print):
    cmap = plt.get_cmap("turbo")   # visible middle (coolwarm washes the mid-gamma to near-white)
    gcol = {g: cmap(x) for g, x in zip(gammas, np.linspace(0.1, 0.9, len(gammas)))}
    results, stats = {g: [] for g in gammas}, {}
    for g in gammas:
        succ = 0
        for seed in range(max_seeds):
            path = GS.rollout_path(env, g, cfg, seed)
            ok, _ = GS.is_success(path, env)
            if ok:
                results[g].append(path); succ += 1
                if len(results[g]) >= n_target:
                    break
        stats[g] = (succ, seed + 1)
        log(f"  gamma {g}: kept {len(results[g])} success / {seed + 1} seeds")
    obs = env.obstacles.detach().cpu().numpy()
    goal = env.goal.detach().cpu().numpy(); x0 = env.x0.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    _grid_lines(ax)
    _draw_obs(ax, obs)
    for g in gammas:
        for p in results[g]:
            ax.plot(p[:, 0], p[:, 1], "-", color=gcol[g], lw=1.0, alpha=0.5, zorder=5)
    ax.scatter([x0[0]], [x0[1]], s=90, marker="s", c="#00a000", edgecolor="k", zorder=8)
    ax.scatter([goal[0]], [goal[1]], marker="*", s=260, c="gold", edgecolor="k", zorder=8)
    handles = [Line2D([0], [0], color=gcol[g], lw=2.6, label=f"γ={g}  (n={len(results[g])})") for g in gammas]
    ax.legend(handles=handles, loc="upper left", fontsize=9, title="successful trajectories", framealpha=0.9)
    ax.set_xlim(*env.xlim); ax.set_ylim(*env.ylim); ax.set_aspect("equal")
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_title("Success-only SafeMPPI trajectories on the 5×5 grid (color = γ: blue conservative → red aggressive)",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=145); plt.close(fig)
    log(f"overlay -> {out}")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-target", type=int, default=30)
    ap.add_argument("--max-seeds", type=int, default=90)
    ap.add_argument("--skip-sweep", action="store_true")
    args = ap.parse_args()
    env = GS.make_grid(); cfg = GS.mode1_config()
    print(f"=== grid5 scene: {len(env.obstacles)} obstacles r={GS.OBS_R}, robot r={env.r_robot}, "
          f"mode-1 Gaussian, range {cfg['barrier_activation_radius']} m ===", flush=True)
    if not args.skip_sweep:
        print("--- deliverable 1: gamma sweep {0.1,0.5,1.0} ---", flush=True)
        render_gamma_sweep(env, cfg, [0.1, 0.5, 1.0], os.path.join(FIG, "grid_gamma_sweep.gif"))
    print("--- deliverable 2: success-only overlay {0.1,0.2,0.4,0.8,1.0} ---", flush=True)
    overlay_trajectories(env, cfg, [0.1, 0.2, 0.4, 0.8, 1.0], n_target=args.n_target,
                         max_seeds=args.max_seeds, out=os.path.join(FIG, "grid_overlay.png"))


if __name__ == "__main__":
    main()
