"""Make the abstractions CONCRETE and visual:
 (A) context encoding: a given environment/state -> polar safety grid o -> context vector c = ctx_from(o, low).
     Different environments yield visibly different context vectors.
 (B) flow feature: a given queried control window U -> noised-flow input U_s at level s -> feature phi_s(U).
     Three windows of different direction give different phi_s; their kernel k=<phi_s,phi_s'> separates them.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import _paths
import config as C
from windowed_policy import GridLowFlowPolicy
from polar_grid import polar_grid, N_THETA, N_R
from local_frame import low_dim_features, goal_frame, to_world
from di_grid_viz import di_step


def load_expanded(scene, device):
    ck = torch.load(C.scene_result(scene, "expanded.pt"), weights_only=False)
    pol = GridLowFlowPolicy(H_pred=ck["H_pred"], u_max=ck["u_max"]).to(device)
    pol.load_state_dict(ck["state_dict"]); pol.eval()
    return pol


@torch.no_grad()
def context_figure(pol, env, device, out):
    goal = env.goal.detach().cpu().numpy(); obs = env.obstacles.detach().cpu().numpy()
    x1 = float(obs[:, 0].min()); x2 = float(obs[:, 0].max())
    states = [("open (far from obstacles)", np.array([x1 - 1.6, 0.0, 1.4, 0.0], np.float32)),
              ("approaching upper obstacle", np.array([x1 - 0.15, float(obs[0, 1]) - 0.05, 1.2, 0.6], np.float32)),
              ("inside the gap", np.array([0.5 * (x1 + x2), 0.0, 1.4, 0.0], np.float32))]
    fig, axes = plt.subplots(3, len(states), figsize=(4.1 * len(states), 8.2),
                             gridspec_kw={"height_ratios": [1.25, 1.0, 0.5]})
    for j, (name, st) in enumerate(states):
        grid, _ = polar_grid(st[:2], goal, obs, r_robot=float(env.r_robot))
        low, _ = low_dim_features(st, goal, 0.5, a_prev=st[2:4] * 0, prev_valid=True)
        ctx = pol.ctx_from(torch.tensor(grid[None], device=device), torch.tensor(low[None], device=device))
        cvec = ctx.detach().cpu().numpy()[0]
        # (row0) scene
        a = axes[0][j]
        for (ox, oy, rr) in obs:
            a.add_patch(Circle((ox, oy), rr, facecolor="#c8a2c8", edgecolor="#7b3294", lw=0.8, alpha=0.75, zorder=3))
        a.scatter([st[0]], [st[1]], s=70, c="#00a000", edgecolor="k", zorder=5)
        a.arrow(st[0], st[1], 0.35 * st[2], 0.35 * st[3], head_width=0.12, color="k", zorder=6)
        a.scatter([goal[0]], [goal[1]], marker="*", s=130, c="gold", edgecolor="k", zorder=5)
        a.set_xlim(*env.xlim); a.set_ylim(*env.ylim); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        a.set_title(name, fontsize=11)
        # (row1) polar safety grid: H_P channel (theta x r)
        b = axes[1][j]
        im = b.imshow(grid[2], aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1, origin="lower")
        b.set_xlabel("radius bin", fontsize=8); b.set_yticks([0, N_THETA - 1]); b.set_yticklabels(["-π", "π"], fontsize=8)
        if j == 0:
            b.set_ylabel("angle θ  (grid channel: clipped H_P)", fontsize=9)
        b.set_title("safety grid  o[H_P]", fontsize=9)
        # (row2) context vector
        c = axes[2][j]
        c.imshow(cvec[None], aspect="auto", cmap="coolwarm", vmin=-2, vmax=2)
        c.set_yticks([]); c.set_xticks([0, 48, 144]); c.set_xticklabels(["0", "48", "144"], fontsize=8)
        c.axvline(48, color="k", lw=1.2)
        c.set_title(r"context $\bar c=[\,E_{low}(\ell)_{48}\;|\;E_{grid}(o)_{96}\,]\in\mathbb{R}^{144}$", fontsize=9)
    fig.suptitle(f"[{env.name}] CONTEXT ENCODING — this environment  →  this context vector "
                 r"$\bar c=\mathrm{ctx\_from}(o,\ell)$", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=135); plt.close(fig)
    print(f"context figure -> {out}")


@torch.no_grad()
def feature_figure(pol, env, device, out, s=0.9):
    goal = env.goal.detach().cpu().numpy(); obs = env.obstacles.detach().cpu().numpy()
    st = np.array([float(obs[:, 0].min()) - 1.0, 0.0, 1.3, 0.0], np.float32)
    grid, _ = polar_grid(st[:2], goal, obs, r_robot=float(env.r_robot))
    low, _ = low_dim_features(st, goal, 0.7, a_prev=np.array([1.3, 0.0]), prev_valid=True)
    ctx = pol.ctx_from(torch.tensor(grid[None], device=device), torch.tensor(low[None], device=device))
    e_g, e_lat, _ = goal_frame(st[:2], goal)
    # sample candidate windows, roll, pick 3 by terminal lateral (down / mid / up)
    U = pol.sample(48, ctx.expand(48, -1), temp=1.5)
    Uw = U.detach().cpu().numpy()
    lat = []
    for Ul in Uw:
        s2 = st.copy()
        for u in to_world(Ul, e_g, e_lat):
            s2 = di_step(s2, np.clip(u, -env.u_max, env.u_max), dt=env.dt)
        lat.append(s2[1])
    order = np.argsort(lat)
    pick = [order[2], order[len(order) // 2], order[-3]]
    names = ["steer down", "go straight", "steer up"]; cols = ["#1f77b4", "#ff7f0e", "#9467bd"]
    sel_U = U[pick]
    phi = pol.phi_s(sel_U, ctx.expand(len(pick), -1), s=s).detach().cpu().numpy()
    phin = phi / (np.linalg.norm(phi, axis=1, keepdims=True) + 1e-9)
    Kmat = phin @ phin.T
    # noise template for the U_s illustration (same seed the net averages over)
    eps = pol.noise_templates[0].detach().cpu().numpy().reshape(-1, 2) if hasattr(pol, "noise_templates") \
        else np.zeros_like(Uw[0])

    fig = plt.figure(figsize=(12.6, 7.6))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.2, 0.55, 0.9], width_ratios=[1, 1, 1, 1.15])
    for i, (pi, nm, col) in enumerate(zip(pick, names, cols)):
        Ul = Uw[pi]; Uworld = to_world(Ul, e_g, e_lat)
        # (row0) the queried control rolled out + its noised-flow input U_s rolled out
        a = fig.add_subplot(gs[0, i])
        for (ox, oy, rr) in obs:
            a.add_patch(Circle((ox, oy), rr, facecolor="#c8a2c8", edgecolor="#7b3294", lw=0.8, alpha=0.7, zorder=3))
        s2 = st.copy(); path = [s2[:2].copy()]
        for u in Uworld:
            s2 = di_step(s2, np.clip(u, -env.u_max, env.u_max), dt=env.dt); path.append(s2[:2].copy())
        path = np.array(path)
        Us_local = (1 - s) * eps * pol.u_max + s * Ul                      # noised-flow window (control space)
        s3 = st.copy(); pth = [s3[:2].copy()]
        for u in to_world(Us_local, e_g, e_lat):
            s3 = di_step(s3, np.clip(u, -env.u_max, env.u_max), dt=env.dt); pth.append(s3[:2].copy())
        pth = np.array(pth)
        a.plot(path[:, 0], path[:, 1], "-", color=col, lw=2.6, zorder=6, label="U (queried)")
        a.plot(pth[:, 0], pth[:, 1], "--", color="k", lw=1.2, alpha=0.7, zorder=5, label=f"U_s (s={s})")
        a.scatter([st[0]], [st[1]], s=45, c="#00a000", edgecolor="k", zorder=7)
        a.set_xlim(st[0] - 0.3, obs[:, 0].max() + 1.0); a.set_ylim(-2.0, 2.0); a.set_aspect("equal")
        a.set_xticks([]); a.set_yticks([]); a.set_title(f"queried control: {nm}", fontsize=10, color=col)
        if i == 0:
            a.legend(fontsize=7, loc="lower left")
        # (row1) the control window as (a_g, a_lat) sequences
        b = fig.add_subplot(gs[1, i])
        b.plot(Ul[:, 0], "-o", ms=3, color=col, label="a∥ (along goal)")
        b.plot(Ul[:, 1], "-s", ms=3, color="k", alpha=0.6, label="a⊥ (lateral)")
        b.set_xlabel("window step k", fontsize=8); b.set_ylim(-env.u_max, env.u_max)
        b.set_title(r"$U=\{a_k\}_{k=0}^{9}$", fontsize=9)
        if i == 0:
            b.legend(fontsize=6.5, loc="upper right")
        # (row2) the feature phi_s(U)
        c = fig.add_subplot(gs[2, i])
        c.imshow(phi[i][None], aspect="auto", cmap="magma")
        c.set_yticks([]); c.set_xlabel("feature dim (256)", fontsize=8)
        c.set_title(r"$\phi_s(U)\in\mathbb{R}^{256}$", fontsize=9)
    # kernel matrix (right column, spans)
    axk = fig.add_subplot(gs[:, 3])
    im = axk.imshow(Kmat, cmap="viridis", vmin=-1, vmax=1)
    axk.set_xticks(range(3)); axk.set_yticks(range(3))
    axk.set_xticklabels(names, rotation=30, fontsize=8); axk.set_yticklabels(names, fontsize=8)
    for ii in range(3):
        for jj in range(3):
            axk.text(jj, ii, f"{Kmat[ii, jj]:.2f}", ha="center", va="center",
                     color=("w" if Kmat[ii, jj] < 0.5 else "k"), fontsize=9)
    axk.set_title(r"kernel $k(U_i,U_j)=\langle\hat\phi_s(U_i),\hat\phi_s(U_j)\rangle$", fontsize=9.5)
    fig.suptitle(f"[{env.name}] FLOW FEATURE — this queried control  →  noised-flow input $U_s$  →  "
                 r"feature $\phi_s(U)$  (same conditioning $\bar c$)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=135); plt.close(fig)
    print(f"feature figure -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="slalom", choices=C.SCENE_NAMES)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    env = C.make_scene(args.scene); pol = load_expanded(args.scene, args.device)
    context_figure(pol, env, args.device, C.scene_fig(args.scene, "diag_context.png"))
    feature_figure(pol, env, args.device, C.scene_fig(args.scene, "diag_feature.png"))


if __name__ == "__main__":
    main()
