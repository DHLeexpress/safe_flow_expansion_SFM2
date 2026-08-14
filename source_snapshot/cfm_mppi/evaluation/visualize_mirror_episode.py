"""Visualize Mirror-MPPI navigating a moving crowd, showing the REAL-TIME convex
polytope level-sets around the robot and the proposal sample cloud (the
visualizable form of the mirror-map / flow-matching proposal), swept over gamma.

Panels: [Mizuta | Mirror gamma=g1 | Mirror gamma=g2 | ...]. Each mirror panel
draws, per frame: the nested convex level sets of the polytope barrier H around
the robot (filled contours; lower gamma => more-inflated obstacles => tighter,
wider-berth polytope), the proposal sample cloud (next-step positions of the
feasible-by-construction samples), the robot path, pedestrians, and goal.

  python -m cfm_mppi.evaluation.visualize_mirror_episode --episode 123 \
      --gammas 0.4 0.7 1.0 --output results/benchmark_videos/mirror_levelsets_ep123
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

from cfm_mppi.safegpc_adapter.mirror_sampler import mirror_mppi_action
from cfm_mppi.safegpc_adapter.polytope import build_nominal_polytope
from cfm_mppi.evaluation.eval_benchmark import BenchmarkPolicies, DEFAULTS
from cfm_mppi.evaluation.render_validation_comparison import (
    get_parser as _rp, _make_scene, _frame_obstacles, _frame_velocities, _policy_args, _dynamics_step)

DT = 0.1


def _norm_barrier(poly, grid, beta=None):
    """Robot-normalized polytope barrier: h=1 at the robot (deepest), h=0 on the
    boundary. Per face: val_i(p) = (b_i - a_i.p)/(b_i - a_i.robot) (=1 at robot,
    =0 on face i); h = HARD min_i val_i => PIECEWISE-LINEAR, so the level sets
    {h=(1-gamma)^i} are exact tangent-polytope POLYGONS (matches the hand figure)."""
    mr = (poly.b - poly.A @ poly.ref).clamp_min(1e-3)          # margin at robot [F]
    val = (poly.b.unsqueeze(0) - grid @ poly.A.T) / mr.unsqueeze(0)  # [G,F], =1 at robot
    return val.min(dim=1).values                               # hard min => polygonal level sets


def _di(s, a, dt=DT):
    x = s.copy(); x[0] += dt*s[2]+0.5*dt*dt*a[0]; x[1] += dt*s[3]+0.5*dt*dt*a[1]
    x[2] += dt*a[0]; x[3] += dt*a[1]; return x


def _rollout_mirror(s0, goal, obs, vel, gamma, dev, steps, margin_gain=0.2, nav_sensing=5.0, viz_sensing=3.5, half_width=2.5, nlev=12):
    st = s0.astype(np.float32).copy()
    traj = [st[:2].copy()]; clouds = []; polys = []
    margin_eff = DEFAULTS["safety_margin"] + margin_gain * (1.0 - gamma)
    for t in range(steps):
        ob = _frame_obstacles(obs, t); ve = _frame_velocities(vel, t)
        ob_t = torch.tensor(ob, device=dev); ve_t = torch.tensor(ve, device=dev)
        a, info = mirror_mppi_action(torch.tensor(st, device=dev), torch.tensor(goal, device=dev),
                                     ob_t, ve_t, horizon=nlev, num_samples=320, gamma=gamma, eta=1.0,
                                     dual_sigma=1.2, margin_gain=margin_gain, temperature=0.3,
                                     clear_w=40.0, terminal_w=15.0, sensing_range=nav_sensing,
                                     seed=t, device=dev, return_rollouts=True)
        # sample cloud = feasible samples' next position
        dbg = info["debug_rollouts"]; ss = dbg["states"]; fz = dbg["feasible"]
        clouds.append((ss[:, :, :2], fz))  # FULL rollout trajectories [K, H+1, 2]
        # polytope around current robot for level-set contours (inflated by gamma margin)
        inflated = ob.copy()
        if inflated.shape[0]:
            inflated[:, 2] = inflated[:, 2] + margin_eff
        heading = st[2:4] if np.linalg.norm(st[2:4]) > 0.1 else (goal - st[:2])
        poly = build_nominal_polytope(torch.tensor(st[:2], device=dev), torch.tensor(heading, dtype=torch.float32, device=dev),
                                      torch.tensor(inflated, device=dev) if inflated.shape[0] else torch.zeros(0, 3, device=dev),
                                      sensing_range=viz_sensing, half_width=half_width, max_obstacles=10)
        polys.append(poly)
        st = _di(st, a.detach().cpu().numpy())
        traj.append(st[:2].copy())
    return np.array(traj), clouds, polys


def _barrier_np(A, b, P):
    """Raw polytope barrier min_j (b_j - a_j·p): >0 inside, =0 on boundary. P [...,2]."""
    return (b[None, :] - P.reshape(-1, 2) @ A.T).min(1).reshape(P.shape[:-1])


def _rollout_reject(s0, goal, obs, vel, gamma, dev, steps, num_samples=240, horizon=12,
                    sigma=2.0, amax=3.0, lam=0.6, sensing=3.5, half_width=2.5, margin_gain=0.0):
    """SAMPLE-THEN-REJECT MPPI (the user's framework): sample Gaussian perturbations
    about a PD nominal, roll out the double integrator, REJECT any sample whose
    rollout violates the DTCBF ruler b(x_i) >= (1-gamma)^i b(x_0), then MPPI-average
    the SURVIVORS. gamma only moves the ruler => same samples, different acceptance,
    different trajectory. Returns (traj, clouds=[(pos[M,H+1,2], feasible[M])], polys)."""
    st = s0.astype(np.float32).copy()
    traj = [st[:2].copy()]; clouds = []; polys = []
    rng = np.random.default_rng(0)
    for t in range(steps):
        ob = _frame_obstacles(obs, t)
        inflated = ob.copy()
        if inflated.shape[0] and margin_gain:
            inflated[:, 2] = inflated[:, 2] + margin_gain * (1.0 - gamma)
        heading = st[2:4] if np.linalg.norm(st[2:4]) > 0.1 else (goal - st[:2])
        poly = build_nominal_polytope(
            torch.tensor(st[:2], device=dev), torch.tensor(heading, dtype=torch.float32, device=dev),
            torch.tensor(inflated, device=dev) if inflated.shape[0] else torch.zeros(0, 3, device=dev),
            sensing_range=sensing, half_width=half_width, max_obstacles=10)
        A = poly.A.detach().cpu().numpy(); bb = poly.b.detach().cpu().numpy()
        b0 = float(_barrier_np(A, bb, st[:2][None])[0])
        # PD nominal accel toward goal, clipped
        to_goal = goal - st[:2]
        nom_a = np.clip(1.2 * to_goal - 1.6 * st[2:4], -amax, amax).astype(np.float32)  # [2]
        nom = np.tile(nom_a, (horizon, 1))                                              # [H,2]
        eps = (rng.standard_normal((num_samples, horizon, 2)) * sigma).astype(np.float32)
        U = np.clip(nom[None] + eps, -amax, amax)                                       # [M,H,2]
        # rollout double integrator
        p = np.tile(st[:2], (num_samples, 1)).astype(np.float32)
        v = np.tile(st[2:4], (num_samples, 1)).astype(np.float32)
        pos = [p.copy()]
        for i in range(horizon):
            a = U[:, i]; p = p + DT * v + 0.5 * DT * DT * a; v = v + DT * a; pos.append(p.copy())
        pos = np.stack(pos, 1)                                                          # [M,H+1,2]
        # DTCBF ruler: b(x_i) >= (1-gamma)^i b(x_0)  for ALL i  => accept
        bvals = _barrier_np(A, bb, pos)                                                 # [M,H+1]
        thresh = b0 * (1.0 - gamma) ** np.arange(horizon + 1)                           # [H+1]
        feasible = (bvals >= thresh[None, :] - 1e-6).all(1)                             # [M]
        # MPPI cost on survivors: terminal goal distance + path clearance to true obstacles
        term = np.linalg.norm(pos[:, -1] - goal, axis=1)
        cost = term.copy()
        if ob.shape[0]:
            d = np.linalg.norm(pos[:, :, None, :] - ob[None, None, :, :2], axis=3)      # [M,H+1,O]
            clr = (ob[:, 2][None, None, :] + 0.4 - d).clip(min=0).max(axis=(1, 2))      # penetration
            cost = cost + 8.0 * clr
        w = np.where(feasible, np.exp(-(cost - cost.min()) / lam), 0.0)
        if w.sum() < 1e-8:
            act = nom_a                                                                 # fallback: nominal
        else:
            w = w / w.sum()
            act = (w[:, None, None] * U).sum(0)[0]                                       # averaged FIRST control
        clouds.append((pos, feasible)); polys.append(poly)
        st = _di(st, act)
        traj.append(st[:2].copy())
    return np.array(traj), clouds, polys


def _rollout_mizuta(s0, goal, obs, vel, pol, dev, steps):
    pol._mizuta_episode = None
    st = s0.astype(np.float32).copy(); traj = [st[:2].copy()]; controls = []
    for t in range(steps):
        a, _ = pol.action("mizuta_cfm_mppi", st, goal, _frame_obstacles(obs, t), controls,
                          "doubleintegrator", 0.5, steps, obstacle_velocities=_frame_velocities(vel, t))
        st = _dynamics_step(st, a, "doubleintegrator", DT); controls.append(a); traj.append(st[:2].copy())
    return np.array(traj)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", nargs="+", type=int, default=[24, 98, 88])
    p.add_argument("--gammas", nargs="+", type=float, default=[0.1, 0.5])
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--margin-gain", type=float, default=0.5, help="how strongly gamma inflates obstacles (visual contrast)")
    p.add_argument("--num-levels", type=int, default=12, help="#polytope level-sets = MPPI horizon N (one (1-gamma)^i per step)")
    p.add_argument("--output", default="results/benchmark_videos/mirror_levelsets_ep123")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    cli = p.parse_args()
    dev = torch.device(cli.device)

    pol = BenchmarkPolicies(_policy_args(_rp().parse_args([])), dev)
    # roll out every episode, store data + per-episode plot limits/grid
    EP = []
    for ep in cli.episodes:
        b = _rp().parse_args([]); b.dataset = "ucy"; b.dynamics = "doubleintegrator"
        b.pedestrian_source = "validation"; b.episode = ep; b.steps = cli.steps
        s0, goal, obs, vel, label = _make_scene(b)
        print(f"[ep {ep}] rolling out mizuta + mirror {cli.gammas}...", flush=True)
        miz = _rollout_mizuta(s0, goal, obs, vel, pol, dev, cli.steps)
        mirror = {g: _rollout_reject(s0, goal, obs, vel, g, dev, cli.steps, horizon=cli.num_levels, margin_gain=0.0) for g in cli.gammas}
        allxy = np.concatenate([miz] + [mirror[g][0] for g in cli.gammas], axis=0)
        ped_xy = [_frame_obstacles(obs, t)[:, :2] for t in range(cli.steps) if _frame_obstacles(obs, t).shape[0]]
        if ped_xy:
            allxy = np.concatenate([allxy] + ped_xy, axis=0)
        pad = 1.5
        xlim = (float(allxy[:, 0].min()-pad), float(allxy[:, 0].max()+pad))
        ylim = (float(allxy[:, 1].min()-pad), float(allxy[:, 1].max()+pad))
        gx = np.linspace(*xlim, 70); gy = np.linspace(*ylim, 70); GX, GY = np.meshgrid(gx, gy)
        grid_pts = torch.tensor(np.stack([GX.ravel(), GY.ravel()], 1), dtype=torch.float32, device=dev)
        EP.append(dict(ep=ep, goal=goal, obs=obs, label=label, miz=miz, mirror=mirror,
                       xlim=xlim, ylim=ylim, GX=GX, GY=GY, grid=grid_pts))

    ncol = 1 + len(cli.gammas)
    fig, axes = plt.subplots(1, ncol, figsize=(4.2*ncol, 4.4))
    if ncol == 1:
        axes = [axes]
    S = cli.steps

    def draw(f):
        D = EP[min(f // S, len(EP)-1)]; t = f % S
        goal, obs, GX, GY, grid = D["goal"], D["obs"], D["GX"], D["GY"], D["grid"]
        obs_t = _frame_obstacles(obs, t)
        for ax in axes:
            ax.clear(); ax.set_xlim(D["xlim"]); ax.set_ylim(D["ylim"]); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for c in obs_t:
                ax.add_patch(Circle((c[0], c[1]), c[2]+0.5, facecolor=(0.6, 0.4, 0.8, 0.12), edgecolor="#9467bd", lw=0.8, zorder=2))
            ax.scatter(goal[0], goal[1], s=130, marker="*", c="gold", edgecolor="k", zorder=6)
        axes[0].set_title("Mizuta CFM-MPPI", fontsize=10, color="#444")
        miz = D["miz"]; axes[0].plot(miz[:t+1, 0], miz[:t+1, 1], "-", color="#d62728", lw=2.0, zorder=5)
        axes[0].scatter(miz[t, 0], miz[t, 1], s=55, c="#1a9850", edgecolor="k", zorder=7)
        for k, g in enumerate(cli.gammas):
            ax = axes[k+1]; traj, clouds, polys = D["mirror"][g]
            poly = polys[min(t, len(polys)-1)]
            H = _norm_barrier(poly, grid).detach().cpu().numpy().reshape(GX.shape)
            lv = sorted(set([0.0] + [round((1.0 - g) ** i, 4) for i in range(0, cli.num_levels + 1)]))
            ax.contourf(GX, GY, H, levels=lv + [1.0001], cmap="Blues", alpha=0.6, zorder=1)
            ax.contour(GX, GY, H, levels=lv[1:], colors="#2166ac", linewidths=0.5, alpha=0.7, zorder=3)
            ax.contour(GX, GY, H, levels=[0.0], colors="#08306b", linewidths=1.8, zorder=3)
            cl_xy, cl_f = clouds[min(t, len(clouds)-1)]   # [K,H+1,2], [K]
            cl_f = np.asarray(cl_f).astype(bool)
            acc = cl_xy[cl_f]                              # accepted: survive the (1-gamma)^i ruler
            rej = cl_xy[~cl_f]                             # rejected: violate the polytope ruler
            for kk in range(0, rej.shape[0], max(1, rej.shape[0] // 45)):   # rejected = RED, spill out
                ax.plot(rej[kk, :, 0], rej[kk, :, 1], color="#e34a33", alpha=0.16, lw=0.5, zorder=3)
                ax.scatter(rej[kk, -1, 0], rej[kk, -1, 1], s=11, color="#e34a33", alpha=0.7, marker="x", lw=0.7, zorder=3)
            for kk in range(0, acc.shape[0], max(1, acc.shape[0] // 45)):   # accepted = GREEN, stay in
                ax.plot(acc[kk, :, 0], acc[kk, :, 1], color="#1a9850", alpha=0.40, lw=0.6, zorder=4)
            na, nr = int(cl_f.sum()), int((~cl_f).sum())
            ax.text(0.03, 0.97, f"accept {na}\nreject {nr}", transform=ax.transAxes, fontsize=8.5,
                    va="top", ha="left", color="#222", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.75, ec="#999", lw=0.5))
            ax.plot(traj[:t+1, 0], traj[:t+1, 1], "-", color="#d62728", lw=2.0, zorder=5)
            ax.scatter(traj[t, 0], traj[t, 1], s=55, c="#1a9850", edgecolor="k", zorder=7)
            ax.set_title(f"Mirror-MPPI  γ={g}", fontsize=10, color="#08519c")
        fig.suptitle(f"{D['label']} — tangent-cut polytope (1-γ)^i level-sets · sample REJECTION: accepted=green, rejected=red ✗ | step {t}",
                     fontsize=10.5, y=0.99)
        return []

    print("animating...", flush=True)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.06)
    anim = FuncAnimation(fig, draw, frames=S*len(EP), interval=120)
    anim.save(cli.output + ".gif", writer=PillowWriter(fps=9), dpi=90)
    try:
        anim.save(cli.output + ".mp4", fps=9, dpi=110)
    except Exception as e:
        print("mp4 save failed (ffmpeg?):", e)
    print(f"saved {cli.output}.gif (+mp4) — {len(EP)} episodes in a row")


if __name__ == "__main__":
    main()
