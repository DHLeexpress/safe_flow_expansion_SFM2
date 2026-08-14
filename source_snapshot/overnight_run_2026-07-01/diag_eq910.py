"""Make Eq.9 / Eq.10 / Algorithm-1 CONCRETE — show the queries and the kernel on our windowed policy.

At one conditioning state we:
  * sample candidate windows U from the FM (the design x)  → roll them out (the 'queries')
  * φ_s(U) = the FM velocity-net's noised-flow feature at level s   (Eq.10 representation)
  * kernel  k(x,x') = ⟨φ_s(x), φ_s(x')⟩  (linear, paper-faithful)  → K matrix
  * σ²(x) = k(x,x) − k(x,X)(K_buf+λI)⁻¹k(X,x)  over a 'known' buffer  (Eq.10 GP posterior variance)
  * Eq.9 tilt  w ∝ exp(σ/β)  → systematic-resample the B queries; report ESS (is the tilt informative?)
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
from polar_grid import polar_grid
from local_frame import low_dim_features, goal_frame, to_world
from di_grid_viz import di_step
from uncertainty import GPUncertainty


def load_expanded(scene, device):
    ck = torch.load(C.scene_result(scene, "expanded.pt"), weights_only=False)
    pol = GridLowFlowPolicy(H_pred=ck["H_pred"], u_max=ck["u_max"]).to(device)
    pol.load_state_dict(ck["state_dict"]); pol.eval()
    return pol


def systematic_resample(w, B):
    w = w / w.sum()
    cdf = np.cumsum(w)
    pts = (np.random.rand() / B) + np.arange(B) / B
    return np.clip(np.searchsorted(cdf, pts), 0, len(w) - 1)


def roll_windows(U_local, state, goal, env):
    e_g, e_lat, _ = goal_frame(state[:2], goal)
    out = []
    for Ul in U_local:
        Uw = to_world(Ul, e_g, e_lat); st = state.astype(np.float32).copy(); p = [st[:2].copy()]
        for u in Uw:
            st = di_step(st, np.clip(u, -env.u_max, env.u_max), dt=env.dt); p.append(st[:2].copy())
        out.append(np.array(p))
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="gap", choices=C.SCENE_NAMES)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--s-level", type=float, default=0.9)
    ap.add_argument("--beta", type=float, default=1.0 / 13)
    args = ap.parse_args()
    env = C.make_scene(args.scene)
    pol = load_expanded(args.scene, args.device)
    goal = env.goal.detach().cpu().numpy(); obs = env.obstacles.detach().cpu().numpy()
    # a conditioning state just BEFORE the obstacles (where the mode choice matters)
    state = np.array([float(obs[:, 0].min()) - 1.0, 0.0, 1.2, 0.0], np.float32)
    grid, _ = polar_grid(state[:2], goal, obs, r_robot=float(env.r_robot))
    low, _ = low_dim_features(state, goal, 0.7, a_prev=np.array([1.2, 0]), prev_valid=True)
    ctx = pol.ctx_from(torch.tensor(grid[None], device=args.device), torch.tensor(low[None], device=args.device))

    nb, nc = 64, 96
    buf_U = pol.sample(nb, ctx.expand(nb, -1), temp=0.7)                 # 'known' behavior
    cand_U = pol.sample(nc, ctx.expand(nc, -1), temp=1.7)               # exploratory candidates (the queries)
    phi_buf = pol.phi_s(buf_U, ctx.expand(nb, -1), s=args.s_level)       # Eq.10 representation φ_s
    phi_cand = pol.phi_s(cand_U, ctx.expand(nc, -1), s=args.s_level)
    unc = GPUncertainty(kernel="linear", lam=1e-2, normalize=True)       # k(x,x')=⟨φ,φ'⟩
    unc.set_buffer(phi_buf)
    sigma = unc.sigma(phi_cand).detach().cpu().numpy()                   # Eq.10 posterior std per candidate
    phn = torch.nn.functional.normalize(phi_cand, dim=1)
    K = (phn @ phn.T).detach().cpu().numpy()                             # the linear KERNEL matrix
    w = np.exp((sigma - sigma.max()) / max(args.beta, 1e-6)); w /= w.sum()   # Eq.9 tilt
    ess = 1.0 / (w ** 2).sum()
    sel = systematic_resample(w, 24)                                    # the B selected queries
    cand_paths = roll_windows(cand_U.detach().cpu().numpy(), state, goal, env)

    fig, ax = plt.subplots(2, 2, figsize=(11.5, 9.2))
    # (a) candidate query windows colored by σ (uncertainty)
    a = ax[0][0]
    for (ox, oy, rr) in obs:
        a.add_patch(Circle((ox, oy), rr, facecolor="#c8a2c8", edgecolor="#7b3294", lw=0.7, alpha=0.7, zorder=3))
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(sigma.min(), sigma.max()))
    for p, s in zip(cand_paths, sigma):
        a.plot(p[:, 0], p[:, 1], "-", color=sm.to_rgba(s), lw=1.0, alpha=0.7, zorder=4)
    a.scatter([state[0]], [state[1]], s=45, c="#00a000", edgecolor="k", zorder=6)
    a.set_xlim(state[0] - 0.4, obs[:, 0].max() + 1.2); a.set_ylim(-2.2, 2.2); a.set_aspect("equal")
    a.set_title("Eq.9 QUERIES: candidate windows colored by σ (Eq.10 uncertainty)", fontsize=9)
    plt.colorbar(sm, ax=a, label="σ (posterior std)")
    # (b) kernel matrix
    im = ax[0][1].imshow(K, cmap="magma", vmin=-1, vmax=1)
    ax[0][1].set_title("linear KERNEL  k(x,x')=⟨φ_s(x),φ_s(x')⟩  (candidates)", fontsize=9)
    plt.colorbar(im, ax=ax[0][1])
    # (c) σ histogram + ESS
    ax[1][0].hist(sigma, bins=24, color="#4477aa")
    ax[1][0].set_title(f"σ over candidates (Eq.10)  |  ESS(Eq.9 tilt)={ess:.1f}/{nc}\n"
                       f"σ∈[{sigma.min():.3f},{sigma.max():.3f}]  β={args.beta:.3f}", fontsize=9)
    ax[1][0].set_xlabel("σ (posterior std)")
    # (d) selected queries (the resampled B)
    d = ax[1][1]
    for (ox, oy, rr) in obs:
        d.add_patch(Circle((ox, oy), rr, facecolor="#c8a2c8", edgecolor="#7b3294", lw=0.7, alpha=0.7, zorder=3))
    for i in sel:
        d.plot(cand_paths[i][:, 0], cand_paths[i][:, 1], "-", color="#d62728", lw=1.1, alpha=0.7, zorder=4)
    d.scatter([state[0]], [state[1]], s=45, c="#00a000", edgecolor="k", zorder=6)
    d.set_xlim(state[0] - 0.4, obs[:, 0].max() + 1.2); d.set_ylim(-2.2, 2.2); d.set_aspect("equal")
    d.set_title("selected B=24 queries after Eq.9 σ-tilt + systematic resample", fontsize=9)
    fig.suptitle(f"[{args.scene}] Eq.9 (active query) + Eq.10 (GP σ over φ_s) + Alg.1 — s={args.s_level}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = C.scene_fig(args.scene, "diag_eq910.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[{args.scene}] Eq.9/10 diag  σ∈[{sigma.min():.3f},{sigma.max():.3f}] ESS={ess:.1f}/{nc} "
          f"(ESS≈{nc}→tilt≈uniform)  → {out}", flush=True)


if __name__ == "__main__":
    main()
