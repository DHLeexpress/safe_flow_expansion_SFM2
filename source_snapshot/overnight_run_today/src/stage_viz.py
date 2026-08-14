"""Multi-stage GIFs that show the PRINCIPLE of each SafeFlow Exploration stage (the Figma loop):

  stage 0 (static)  verified_vs_candidate : conservative candidate polytope  vs  less-conservative
                    verified region the FM trajectories actually use.
  stage 1 (gif)     certified_planning    : per representative mode, the DTCBF clearance level sets
                    {h >= (1-gamma)^i h0} + the verified-polytope tangent faces certifying the path.
  stage 2 (gif)     safeflow_expansion    : the learning loop over rounds -- the seed's single leaf
                    opening into all homotopy modes, with coverage/validity/mode-coverage filling in.

Matplotlib idiom matches the repo (Blues level-set contours, obstacle circles, green=certified/red=rejected,
FuncAnimation -> PillowWriter gif).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch

from dynamics import rollout, clip_controls
from dtcbf import verify, build_candidate_polytope
from safeflow import validity_label
from flow_policy import FlowPolicy
import descriptors as D
from plots import MODE_COLORS


# --------------------------------------------------------------------- helpers

def _grid(env, nx=160, ny=140):
    gx = np.linspace(env.xlim[0], env.xlim[1], nx)
    gy = np.linspace(env.ylim[0], env.ylim[1], ny)
    GX, GY = np.meshgrid(gx, gy)
    return GX, GY


def clearance_field(env, GX, GY):
    """h(p) = min_j ( ||p - c_j|| - (r_j + r_robot) )  at t=0  (>0 = safe). Returns grid."""
    P = np.stack([GX.ravel(), GY.ravel()], 1)                      # [G,2]
    obs = env.obstacles.cpu().numpy()
    m = obs[:, 2] + env.r_robot
    d = np.linalg.norm(P[:, None, :] - obs[None, :, :2], axis=2) - m[None]   # [G,N]
    return d.min(1).reshape(GX.shape)


def _draw_obstacles(ax, env, light=False):
    for j in range(env.n_obs):
        cx, cy, r = [float(v) for v in env.obstacles[j].cpu()]
        a = 0.18 if light else 0.32
        ax.add_patch(Circle((cx, cy), r, facecolor="#7b3294", alpha=a, edgecolor="#4d004b", lw=1.2, zorder=4))
        ax.add_patch(Circle((cx, cy), r + env.r_robot, facecolor="none", edgecolor="#7b3294",
                             ls="--", lw=0.9, alpha=0.55, zorder=4))
    p0 = env.x0[:2].cpu(); g = env.goal.cpu()
    ax.scatter([p0[0]], [p0[1]], s=80, c="#1a9850", edgecolor="k", marker="o", zorder=9)
    ax.scatter([g[0]], [g[1]], s=150, c="gold", edgecolor="k", marker="*", zorder=9)
    ax.set_xlim(*env.xlim); ax.set_ylim(*env.ylim); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


@torch.no_grad()
def _representatives(policy, env, ctx, cfg, device, n=600):
    """One typical certified trajectory per macro-mode. Returns {mode: (states[T+1,2], req_gamma)}."""
    U = clip_controls(policy.sample(n, ctx, nfe=cfg.nfe), env)
    valid, safe, st, reqg = validity_label(U, env, cfg.gamma_max, cfg.n_angles)
    modes = D.macro_mode(st, env)
    lat = D.descriptor(st, env).mean(1)
    reps = {}
    for m in range(D.n_modes(env)):
        sel = torch.where(valid & (modes == m))[0]
        if sel.numel() == 0:
            continue
        med = lat[sel].median()
        pick = sel[(lat[sel] - med).abs().argmin()]
        reps[m] = (st[pick, :, :2].cpu().numpy(), float(reqg[pick]))
    return reps


# --------------------------------------------------------------------- stage 0 (static)

def _draw_candidate_polytope(ax, env):
    """Conservative candidate polytope boundary (red dotted) = the deterministic polytope.py reference."""
    cand = build_candidate_polytope(env)
    if cand is None:
        return
    A, b = cand[0].cpu().numpy(), cand[1].cpu().numpy()
    xs = np.linspace(env.xlim[0], env.xlim[1], 60)
    for a_i, b_i in zip(A, b):
        if abs(a_i[1]) > 1e-6:
            ys = (b_i - a_i[0] * xs) / a_i[1]
            ax.plot(xs, ys, color="#d62728", ls=":", lw=1.6, alpha=0.75, zorder=6)
        else:                                          # vertical half-space x = b/a_x
            ax.axvline(b_i / a_i[0], color="#d62728", ls=":", lw=1.6, alpha=0.75, zorder=6)


@torch.no_grad()
def render_seed_vs_expanded(seed_state, final_pol, env, ctx, cfg, path, width, depth, device="cpu"):
    """LEFT: the seed FM (conservative, one homotopy leaf). RIGHT: the expanded FM (all certified
    modes, hugging the free space incl. the narrow gap). Red dotted = conservative candidate polytope."""
    GX, GY = _grid(env)
    H = clearance_field(env, GX, GY)
    seed_pol = FlowPolicy(env.T, ctx.numel(), width=width, depth=depth, u_max=env.u_max).to(device)
    seed_pol.load_state_dict(seed_state); seed_pol.eval()
    cols = MODE_COLORS[env.name]; names = D.mode_names(env)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.7))
    titles = ["seed FM  (conservative · one leaf)", "expanded FM  (less conservative · multimodal)"]
    for ax, pol, title in zip(axes, (seed_pol, final_pol), titles):
        ax.contourf(GX, GY, H, levels=np.linspace(-0.5, H.max(), 14), cmap="Blues", alpha=0.5, zorder=1)
        ax.contour(GX, GY, H, levels=[0.0], colors="#08306b", linewidths=1.6, zorder=3)
        _draw_obstacles(ax, env)
        _draw_candidate_polytope(ax, env)
        U = clip_controls(pol.sample(400, ctx, nfe=cfg.nfe), env)
        valid, _, st, _ = validity_label(U, env, cfg.gamma_max, cfg.n_angles)
        st = st.cpu().numpy(); valid = valid.cpu().numpy()
        modes = D.macro_mode(torch.tensor(st), env).numpy()
        for i in np.where(valid)[0]:
            ax.plot(st[i, :, 0], st[i, :, 1], color=cols[int(modes[i])], alpha=0.25, lw=0.7, zorder=5)
        cnt = [int(((modes == mm) & valid).sum()) for mm in range(D.n_modes(env))]
        ax.set_title(f"{title}\n" + "  ".join(f"{names[mm]}:{cnt[mm]}" for mm in range(len(names))), fontsize=9.5)
    fig.suptitle(f"ENV {env.name} — Stage 0: seed → expanded  (red dotted = conservative candidate polytope)",
                 fontsize=11, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(path, dpi=140); plt.close(fig)


# --------------------------------------------------------------------- stage 1 (SafeMPPI ruler)

@torch.no_grad()
def render_safemppi_gif(env, cfg, path, side="right", gamma=0.3, device="cpu", log=print,
                        M=320, H_mppi=16, sigma=1.6, lam=0.5):
    """SafeMPPI sample-then-reject with the (1-gamma)^i DTCBF RULER (the data engine that seeds the FM).
    Receding horizon: sample controls about a one-sided PD nominal, roll out, REJECT any rollout that
    violates h(x_i) >= (1-gamma)^i h(x_0), MPPI-average the survivors. green=accepted, red=rejected."""
    GX, GY = _grid(env)
    Hf = clearance_field(env, GX, GY)
    obs = env.obstacles.cpu().numpy(); m = obs[:, 2] + env.r_robot
    dt, T = env.dt, env.T
    p0 = env.x0[:2].cpu().numpy(); g = env.goal.cpu().numpy()
    dvec = (g - p0) / (np.linalg.norm(g - p0) + 1e-9)
    e = np.array([-dvec[1], dvec[0]])
    lat = (2.0 if env.name == "gap" else 1.5) * (1 if side == "left" else -1)
    s = np.linspace(0, 1, T + 1)
    path_pts = p0[None] + s[:, None] * (g - p0)[None] + lat * np.sin(np.pi * s)[:, None] * e[None]
    rng = np.random.default_rng(0)

    st = env.x0.cpu().numpy().copy()
    executed = [st[:2].copy()]; frames = []
    for t in range(T):
        p, v = st[:2], st[2:4]
        # one-sided PD nominal toward a lookahead point on the biased path
        tgt = path_pts[min(t + 5, T)]
        nom_a = np.clip(6.0 * (tgt - p) - 4.0 * v, -env.u_max, env.u_max).astype(np.float32)
        nom = np.tile(nom_a, (H_mppi, 1))
        U = np.clip(nom[None] + sigma * rng.standard_normal((M, H_mppi, 2)), -env.u_max, env.u_max).astype(np.float32)
        pp = np.tile(p, (M, 1)).astype(np.float32); vv = np.tile(v, (M, 1)).astype(np.float32)
        pos = [pp.copy()]
        for i in range(H_mppi):
            a = U[:, i]; pp = pp + dt * vv + 0.5 * dt * dt * a; vv = vv + dt * a; pos.append(pp.copy())
        pos = np.stack(pos, 1)                                                  # [M,H+1,2]
        d = np.linalg.norm(pos[:, :, None, :] - obs[None, None, :, :2], axis=3) - m[None, None]
        h = d.min(2)                                                            # [M,H+1] clearance barrier
        h0 = float(np.min(np.linalg.norm(p[None] - obs[:, :2], axis=1) - m))
        thresh = h0 * (1.0 - gamma) ** np.arange(H_mppi + 1)
        feasible = (h >= thresh[None] - 1e-6).all(1)
        term = np.linalg.norm(pos[:, -1] - g, axis=1)
        cost = term + 8.0 * np.clip(-h.min(1), 0, None)
        w = np.where(feasible, np.exp(-(cost - cost.min()) / lam), 0.0)
        if w.sum() < 1e-8:
            act = U[int(h.min(1).argmax()), 0]
        else:
            act = ((w / w.sum())[:, None, None] * U).sum(0)[0]
        frames.append((pos, feasible, h0))
        st = st.copy(); st[:2] = p + dt * v + 0.5 * dt * dt * act; st[2:4] = v + dt * act
        executed.append(st[:2].copy())
        if np.linalg.norm(st[:2] - g) < 0.5:
            break
    executed = np.array(executed)
    nfr = len(frames)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))

    def draw(t):
        ax.clear()
        pos, feas, h0 = frames[t]
        ax.contourf(GX, GY, Hf, levels=np.linspace(-0.5, Hf.max(), 14), cmap="Blues", alpha=0.5, zorder=1)
        # the RULER: nested {h = (1-gamma)^i h0} level sets
        for i in range(0, H_mppi + 1, 2):
            lev = h0 * (1 - gamma) ** i
            if Hf.min() < lev < Hf.max():
                ax.contour(GX, GY, Hf, levels=[lev], colors="#e08214", linewidths=0.7, alpha=0.6, zorder=3)
        ax.contour(GX, GY, Hf, levels=[0.0], colors="#08306b", linewidths=1.5, zorder=3)
        _draw_obstacles(ax, env)
        rej = pos[~feas]; acc = pos[feas]
        for k in range(0, rej.shape[0], max(1, rej.shape[0] // 50)):
            ax.plot(rej[k, :, 0], rej[k, :, 1], color="#e34a33", alpha=0.14, lw=0.5, zorder=5)
            ax.scatter(rej[k, -1, 0], rej[k, -1, 1], s=10, color="#e34a33", marker="x", lw=0.6, alpha=0.7, zorder=5)
        for k in range(0, acc.shape[0], max(1, acc.shape[0] // 50)):
            ax.plot(acc[k, :, 0], acc[k, :, 1], color="#1a9850", alpha=0.35, lw=0.6, zorder=6)
        ax.plot(executed[:t + 1, 0], executed[:t + 1, 1], "-", color="#d62728", lw=2.4, zorder=8)
        ax.text(0.03, 0.97, f"step {t}\naccept {int(feas.sum())}  reject {int((~feas).sum())}",
                transform=ax.transAxes, fontsize=9, va="top", ha="left", zorder=10,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85, ec="#999", lw=0.5))
        ax.set_title(f"ENV {env.name} — Stage 1: SafeMPPI + (1-γ)^i ruler (γ={gamma})\n"
                     f"green=accepted  red=rejected ✗  → conservative one-leaf seed", fontsize=9.5)
        return []

    fig.tight_layout()
    anim = FuncAnimation(fig, draw, frames=nfr, interval=150)
    anim.save(path, writer=PillowWriter(fps=7), dpi=95)
    plt.close(fig)
    log(f"[stage1] saved {path}")


# --------------------------------------------------------------------- stage 2 (certified planning)

def render_certified_gif(policy, env, ctx, cfg, path, device="cpu", log=print):
    reps = _representatives(policy, env, ctx, cfg, device)
    if not reps:
        log("[stage1] no certified representatives; skipping"); return
    GX, GY = _grid(env)
    H = clearance_field(env, GX, GY)
    obs = env.obstacles.cpu().numpy()
    m = obs[:, 2] + env.r_robot
    p0 = env.x0[:2].cpu().numpy()
    h0 = float(np.min(np.linalg.norm(p0[None] - obs[:, :2], axis=1) - m))
    names = D.mode_names(env)
    modes_sorted = sorted(reps)
    ncol = len(modes_sorted)
    fig, axes = plt.subplots(1, ncol, figsize=(4.4 * ncol, 4.5), squeeze=False)
    axes = axes[0]
    T = env.T

    def draw(t):
        for k, mdx in enumerate(modes_sorted):
            ax = axes[k]; ax.clear()
            traj, reqg = reps[mdx]
            g = max(0.15, reqg)
            ax.contourf(GX, GY, H, levels=np.linspace(-0.5, H.max(), 14), cmap="Blues", alpha=0.5, zorder=1)
            ax.contour(GX, GY, H, levels=[0.0], colors="#08306b", linewidths=1.6, zorder=3)
            # DTCBF ruler at step t: boundary {h = (1-g)^t * h0}  (loosens over the horizon)
            lev = (1.0 - g) ** t * h0
            if H.min() < lev < H.max():
                ax.contour(GX, GY, H, levels=[lev], colors="#e08214", linewidths=1.8, zorder=5)
            _draw_obstacles(ax, env)
            ax.plot(traj[:t + 1, 0], traj[:t + 1, 1], color="#1a9850", lw=2.2, zorder=7)
            pt = traj[min(t, T)]
            ax.scatter([pt[0]], [pt[1]], s=70, c="#1a9850", edgecolor="k", zorder=9)
            # verified-polytope tangent faces (the per-step separating half-planes)
            for j in range(env.n_obs):
                c = obs[j, :2]
                d = pt - c; nd = np.linalg.norm(d) + 1e-9
                if nd - m[j] > 2.6:           # only draw active (nearby) faces
                    continue
                nvec = d / nd
                tang = c + nvec * m[j]         # tangent point on inflated obstacle
                perp = np.array([-nvec[1], nvec[0]])
                seg = np.stack([tang - 1.6 * perp, tang + 1.6 * perp])
                ax.plot(seg[:, 0], seg[:, 1], color="#d62728", lw=1.8, zorder=6)
                ax.arrow(tang[0], tang[1], 0.4 * nvec[0], 0.4 * nvec[1], color="#d62728",
                         width=0.02, head_width=0.12, zorder=6)
            ht = float(np.min(np.linalg.norm(pt[None] - obs[:, :2], axis=1) - m))
            ax.text(0.03, 0.97, f"{names[mdx]}\nstep {t}/{T}\nh={ht:.2f}  ≥ (1-γ)^t·h₀={lev:.2f}\nγ_req={reqg:.2f}",
                    transform=ax.transAxes, fontsize=8.5, va="top", ha="left", zorder=10,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85, ec="#999", lw=0.5))
        fig.suptitle(f"ENV {env.name} — Stage 2: FM field, certified · orange = DTCBF level set {{h ≥ (1-γ)^i h₀}} · "
                     f"red = verified-polytope faces", fontsize=10, y=0.99)
        return []

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    anim = FuncAnimation(fig, draw, frames=T + 1, interval=140)
    anim.save(path, writer=PillowWriter(fps=8), dpi=95)
    plt.close(fig)
    log(f"[stage1] saved {path}")


# --------------------------------------------------------------------- stage 2 (safe-flow expansion)

@torch.no_grad()
def render_expansion_gif(snaps, env, ctx, cfg, history, path, width, depth, device="cpu",
                         log=print, n_traj=280):
    rounds = sorted(snaps)
    pol = FlowPolicy(env.T, ctx.numel(), width=width, depth=depth, u_max=env.u_max).to(device)
    frames_data = []
    for r in rounds:
        pol.load_state_dict(snaps[r]); pol.eval()
        U = clip_controls(pol.sample(n_traj, ctx, nfe=cfg.nfe), env)
        valid, _, st, _ = validity_label(U, env, cfg.gamma_max, cfg.n_angles)
        modes = D.macro_mode(st, env)
        frames_data.append((st[:, :, :2].cpu().numpy(), valid.cpu().numpy(), modes.cpu().numpy()))
    hbr = {h["round"]: h for h in history}
    rr = [h["round"] for h in history]
    cov = [100 * h["coverage"] for h in history]
    val = [100 * h["validity"] for h in history]
    mc = [100 * h["mode_coverage"] for h in history]
    cols = MODE_COLORS[env.name]
    names = D.mode_names(env)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.6, 4.8))

    def draw(f):
        r = rounds[f]; st, valid, modes = frames_data[f]
        axL.clear()
        _draw_obstacles(axL, env, light=True)
        _draw_candidate_polytope(axL, env)        # fixed conservative reference: FM grows beyond it
        for i in np.where(~valid)[0][:90]:
            axL.plot(st[i, :, 0], st[i, :, 1], color="0.6", alpha=0.08, lw=0.6, zorder=5)
        for i in np.where(valid)[0]:
            axL.plot(st[i, :, 0], st[i, :, 1], color=cols[int(modes[i])], alpha=0.22, lw=0.7, zorder=6)
        cnt = [int(((modes == mm) & valid).sum()) for mm in range(D.n_modes(env))]
        sub = "  ".join(f"{names[mm]}:{cnt[mm]}" for mm in range(len(names)))
        axL.set_title(f"FM samples · round {r}  (red dotted = conservative candidate polytope)\n{sub}", fontsize=9)

        axR.clear()
        upto = sum(1 for x in rr if x <= r)
        axR.plot(rr[:upto], cov[:upto], "-o", color="#2166ac", lw=2, ms=3, label="coverage %")
        axR.plot(rr[:upto], val[:upto], "-s", color="#1a9850", lw=2, ms=3, label="validity %")
        axR.plot(rr[:upto], mc[:upto], "-^", color="#762a83", lw=2, ms=3, label="mode-coverage %")
        axR.set_xlim(0, max(rr)); axR.set_ylim(0, 105); axR.grid(alpha=0.2)
        axR.set_xlabel("expansion round"); axR.legend(loc="lower right", fontsize=8)
        axR.set_title("Safe Flow Expansion progress", fontsize=9.5)
        fig.suptitle(f"ENV {env.name} — Stage 3: Safe Flow Expansion (seed's single leaf → all certified modes)",
                     fontsize=10.5, y=0.99)
        return []

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    anim = FuncAnimation(fig, draw, frames=len(rounds), interval=400)
    anim.save(path, writer=PillowWriter(fps=3), dpi=95)
    plt.close(fig)
    log(f"[stage2] saved {path}")
