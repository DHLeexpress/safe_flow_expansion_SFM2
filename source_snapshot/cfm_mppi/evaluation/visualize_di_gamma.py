"""Clean DOUBLE-INTEGRATOR sanity demo (no UCY, no pedestrians): one static
obstacle in the path, sample-then-REJECT MPPI swept over gamma. Shows that the
(1-gamma)^i ruler alone (same samples, same obstacle) makes gamma change the
TRAJECTORY berth, and renders accept(green)/reject(red ✗) vividly.

  python -m cfm_mppi.evaluation.visualize_di_gamma --gammas 0.1 0.2 0.4 \
      --output results/benchmark_videos/di_gamma
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter

from cfm_mppi.safegpc_adapter.polytope import build_nominal_polytope
from cfm_mppi.evaluation.visualize_mirror_episode import _barrier_np, _norm_barrier, _di, DT


def reject_rollout(s0, goal, obstacles, gamma, dev, steps, num_samples=400, horizon=16,
                   sigma=2.0, amax=5.0, lam=0.5, vmax=1.2, seed=0):
    """PD nominal + Gaussian perturbations, DTCBF ruler b(x_i) >= (1-gamma)^i b(x_0),
    MPPI-average survivors. gamma enters ONLY through the ruler threshold."""
    st = s0.astype(np.float32).copy()
    obs_t = obstacles.astype(np.float32)                                    # [O,3] static
    traj = [st[:2].copy()]; clouds = []; polys = []; acc_hist = []
    rng = np.random.default_rng(seed)
    for t in range(steps):
        heading = st[2:4] if np.linalg.norm(st[2:4]) > 0.1 else (goal - st[:2])
        poly = build_nominal_polytope(
            torch.tensor(st[:2], device=dev), torch.tensor(heading, dtype=torch.float32, device=dev),
            torch.tensor(obs_t, device=dev), sensing_range=7.0, half_width=3.5, max_obstacles=8)
        A = poly.A.detach().cpu().numpy(); bb = poly.b.detach().cpu().numpy()
        b0 = float(_barrier_np(A, bb, st[:2][None])[0])
        to_goal = goal - st[:2]
        dist = np.linalg.norm(to_goal) + 1e-6
        v_des = vmax * to_goal / dist                                       # bounded-speed nominal toward goal
        nom_a = np.clip(2.0 * (v_des - st[2:4]), -amax, amax).astype(np.float32)
        nom = np.tile(nom_a, (horizon, 1))
        eps = (rng.standard_normal((num_samples, horizon, 2)) * sigma).astype(np.float32)
        U = np.clip(nom[None] + eps, -amax, amax)                           # [M,H,2]
        p = np.tile(st[:2], (num_samples, 1)).astype(np.float32)
        v = np.tile(st[2:4], (num_samples, 1)).astype(np.float32)
        pos = [p.copy()]
        for i in range(horizon):
            a = U[:, i]; p = p + DT * v + 0.5 * DT * DT * a; v = v + DT * a; pos.append(p.copy())
        pos = np.stack(pos, 1)                                              # [M,H+1,2]
        bvals = _barrier_np(A, bb, pos)                                     # [M,H+1]
        thresh = b0 * (1.0 - gamma) ** np.arange(horizon + 1)
        feasible = (bvals >= thresh[None, :] - 1e-6).all(1)                 # [M]
        term = np.linalg.norm(pos[:, -1] - goal, axis=1)
        d = np.linalg.norm(pos[:, :, None, :] - obs_t[None, None, :, :2], axis=3)
        clr = (obs_t[:, 2][None, None, :] + 0.3 - d).clip(min=0).max(axis=(1, 2))
        cost = term + 10.0 * clr
        w = np.where(feasible, np.exp(-(cost - cost.min()) / lam), 0.0)
        if w.sum() < 1e-8:
            best = int(bvals.min(1).argmax())                              # graceful: least-unsafe sample, NOT straight nominal
            act = U[best, 0]
        else:
            act = ((w / w.sum())[:, None, None] * U).sum(0)[0]
        acc_hist.append(int(feasible.sum()))
        clouds.append((pos, feasible)); polys.append(poly)
        st = _di(st, act); traj.append(st[:2].copy())
        if np.linalg.norm(st[:2] - goal) < 0.5:
            break
    return np.array(traj), clouds, polys, acc_hist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gammas", nargs="+", type=float, default=[0.1, 0.2, 0.4])
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--num-levels", type=int, default=16)
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--output", default="results/benchmark_videos/di_gamma")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    cli = p.parse_args()
    dev = torch.device(cli.device)

    # CLEAN scenario: robot heads right; a fat obstacle sits just below the straight
    # line so it MUST arc over -- and the ruler tightness (gamma) sets how high.
    s0 = np.array([0.0, 0.0, 1.0, 0.0], np.float32)
    goal = np.array([12.0, 0.0], np.float32)
    # obstacle ABOVE the straight line -> robot grazes it laterally and dips DOWN;
    # gamma sets how deep the dip (berth). No head-on inertia trap => no freeze.
    obstacles = np.array([[6.0, 1.3, 1.5]], np.float32)                     # [O,3] (x,y,r)

    runs = {}
    for g in cli.gammas:
        print(f"[gamma={g}] rolling out clean DI...", flush=True)
        runs[g] = reject_rollout(s0, goal, obstacles, g, dev, cli.steps,
                                 horizon=cli.num_levels, sigma=cli.sigma)
        tr, _, _, ah = runs[g]
        print(f"   dip min-y={tr[:,1].min():.2f}  reached={np.linalg.norm(tr[-1]-goal)<0.6}  min-accept={min(ah)}  median-accept={int(np.median(ah))}", flush=True)

    xlim = (-1.0, 13.0); ylim = (-3.0, 5.0)
    gx = np.linspace(*xlim, 90); gy = np.linspace(*ylim, 70)
    GX, GY = np.meshgrid(gx, gy)
    grid = torch.tensor(np.stack([GX.ravel(), GY.ravel()], 1), dtype=torch.float32, device=dev)

    ncol = len(cli.gammas)
    fig, axes = plt.subplots(1, ncol, figsize=(4.6 * ncol, 4.4))
    if ncol == 1:
        axes = [axes]
    S = cli.steps

    def draw(f):
        t = f
        for k, g in enumerate(cli.gammas):
            ax = axes[k]; ax.clear()
            ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            traj, clouds, polys, ah = runs[g]
            tt = min(t, len(polys) - 1)
            poly = polys[tt]
            H = _norm_barrier(poly, grid).detach().cpu().numpy().reshape(GX.shape)
            lv = sorted(set([0.0] + [round((1.0 - g) ** i, 4) for i in range(0, cli.num_levels + 1)]))
            ax.contourf(GX, GY, H, levels=lv + [1.0001], cmap="Blues", alpha=0.55, zorder=1)
            ax.contour(GX, GY, H, levels=lv[1:], colors="#2166ac", linewidths=0.5, alpha=0.7, zorder=3)
            ax.contour(GX, GY, H, levels=[0.0], colors="#08306b", linewidths=1.6, zorder=3)
            for o in obstacles:
                ax.add_patch(Circle((o[0], o[1]), o[2], facecolor=(0.6, 0.4, 0.8, 0.25), edgecolor="#6a3d9a", lw=1.2, zorder=2))
            ax.scatter(*goal, s=160, marker="*", c="gold", edgecolor="k", zorder=6)
            pos, feas = clouds[tt]
            acc = pos[feas]; rej = pos[~feas]
            for kk in range(0, rej.shape[0], max(1, rej.shape[0] // 40)):
                ax.plot(rej[kk, :, 0], rej[kk, :, 1], color="#e34a33", alpha=0.16, lw=0.5, zorder=3)
                ax.scatter(rej[kk, -1, 0], rej[kk, -1, 1], s=12, color="#e34a33", alpha=0.7, marker="x", lw=0.7, zorder=3)
            for kk in range(0, acc.shape[0], max(1, acc.shape[0] // 40)):
                ax.plot(acc[kk, :, 0], acc[kk, :, 1], color="#1a9850", alpha=0.40, lw=0.6, zorder=4)
            ax.plot(traj[:min(t, len(traj)-1)+1, 0], traj[:min(t, len(traj)-1)+1, 1], "-", color="#d62728", lw=2.4, zorder=5)
            ax.scatter(traj[min(t, len(traj)-1), 0], traj[min(t, len(traj)-1), 1], s=70, c="#1a9850", edgecolor="k", zorder=7)
            ax.text(0.03, 0.97, f"accept {int(feas.sum())}\nreject {int((~feas).sum())}", transform=ax.transAxes,
                    fontsize=9, va="top", ha="left", color="#222", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.8, ec="#999", lw=0.5))
            ax.set_title(f"double integrator  γ={g}", fontsize=11, color="#08519c")
        fig.suptitle("Clean DI: the (1-γ)^i REJECTION ruler alone bends the path — tighter γ ⇒ wider berth · accept=green, reject=red ✗",
                     fontsize=10.5, y=0.98)
        return []

    print("animating...", flush=True)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.06)
    anim = FuncAnimation(fig, draw, frames=S, interval=120)
    anim.save(cli.output + ".gif", writer=PillowWriter(fps=9), dpi=95)
    try:
        anim.save(cli.output + ".mp4", fps=9, dpi=120)
    except Exception as e:
        print("mp4 save failed:", e)
    print(f"saved {cli.output}.gif (+mp4)")


if __name__ == "__main__":
    main()
