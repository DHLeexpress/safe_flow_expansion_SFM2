"""Visualize ONE Eq-9 σ-tilt step. At a receding-horizon step the policy draws candidate windows; each is
scored by σ = GPUncertainty.sigma(φ_s) (posterior std vs the QUERIED buffer), tilted by w=exp((σ−maxσ)/β),
and one is systematic-resampled. Two panels (small vs grown buffer) show the buffer starting EMPTY (cold-start
σ≈1, near-uniform tilt) and accumulating as the trajectory queries windows — the buffer is fixed within a
trajectory and grows between steps here only to illustrate; in the real loop it is refit once per iteration.
"""
import _paths, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import grid_scene as GS, grid_policy as GP, grid_rollout as GR, grid_feats as GF
from di_grid_viz import di_step
from uncertainty import GPUncertainty

dev = "cuda"; env = GS.make_grid(); obs = env.obstacles.numpy(); rr = float(env.r_robot); goal = env.goal.numpy()
pol, _ = GP.load_policy("pretrained.pt", device=dev)
BETA, S, NF, NB = 1 / 50, 0.3, 40, 40

# plain roll recording (state, grid, low5, hist, U) per step
st = env.x0.numpy().astype(np.float32); hist = []; states = [st.copy()]; recs = []
for t in range(70):
    g = GF.axis_grid(st[:2], obs, rr); l = GF.low5(st, goal, 0.5)
    h = GF.hist_pad(np.array(hist[-16:]) if hist else np.zeros((0, 2)), 16)
    with torch.no_grad():
        U = pol.sample_window(torch.tensor(g, device=dev), torch.tensor(l, device=dev),
                              torch.tensor(h, device=dev), n=1, nfe=8)[0].cpu().numpy()
    recs.append((g, l, h)); a = U[0]; st = di_step(st, a, dt=env.dt); hist.append(a); states.append(st.copy())
states = np.array(states)


def panel(ax, k):
    g, l, h = recs[k]; s0 = states[k]
    unc = GPUncertainty(kernel="rbf", lengthscale=0.2, lam=1e-2, normalize=True)
    if k > 0:                                                     # buffer = φ_s of windows queried so far
        with torch.no_grad():
            G = torch.tensor(np.array([r[0] for r in recs[:k]]), device=dev)
            L = torch.tensor(np.array([r[1] for r in recs[:k]]), device=dev)
            H = torch.tensor(np.array([r[2] for r in recs[:k]]), device=dev)
            uu = []
            for i in range(k):                                   # the executed windows (recompute a sample each)
                uu.append(pol.sample_window(G[i], L[i], H[i], n=1, nfe=6)[0])
            ctx = pol.ctx_from(G, L, H); unc.set_buffer(pol.phi_s(torch.stack(uu), ctx, s=S))
    else:
        unc.set_buffer(None)                                     # cold start
    with torch.no_grad():
        gT, lT, hT = torch.tensor(g, device=dev), torch.tensor(l, device=dev), torch.tensor(h, device=dev)
        Uc = pol.sample_window(gT, lT, hT, n=NF, temp=1.3, nfe=6)
        Uc = torch.cat([Uc, GR.broad_proposal(s0, goal, env, NB, dev)], 0)
        sig = unc.sigma(pol.phi_s_at(Uc, gT, lT, hT, s=S)).cpu().numpy()
    Uc_np = Uc.cpu().numpy()
    w = np.exp((sig - sig.max()) / BETA); sel = int(GR.systematic_resample(torch.tensor(w), 1)[0])
    pos = GR.di_rollout_batch(s0, Uc_np, env.dt)                 # [M,H,2] candidate window rollouts
    for kk in range(6):
        ax.axvline(kk, color="#eee", lw=.5); ax.axhline(kk, color="#eee", lw=.5)
    ax.add_patch(Rectangle((0, 0), 5, 5, fill=False, edgecolor="#999", lw=1.0))
    for j, (ox, oy, r) in enumerate(obs):
        ax.add_patch(Circle((ox, oy), r, facecolor="#b8b8b8" if j >= 16 else "#c8a2c8", edgecolor="#777", lw=.4, alpha=.8))
    ax.plot(states[:k + 1, 0], states[:k + 1, 1], "-", color="#444", lw=1.0, alpha=.6)          # trail
    nrm = Normalize(sig.min(), sig.max())
    for i in range(len(Uc_np)):                                  # candidate windows colored by σ
        ax.plot(pos[i, :, 0], pos[i, :, 1], "-", color=plt.cm.viridis(nrm(sig[i])), lw=0.8, alpha=.55, zorder=5)
    ax.plot(pos[sel, :, 0], pos[sel, :, 1], "-", color="#e6191b", lw=2.4, zorder=8)             # resampled winner
    ax.scatter([s0[0]], [s0[1]], s=55, marker="o", c="white", edgecolor="k", zorder=9)
    ax.set_xlim(-.7, 5.7); ax.set_ylim(-.7, 5.7); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"step {k}: buffer={k} windows,  σ∈[{sig.min():.2f},{sig.max():.2f}],  β=1/50\n"
                 f"red = exp(σ/β)-resampled winner  (spread {sig.max()-sig.min():.2f})", fontsize=9)
    return nrm


fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
n0 = panel(axes[0], 3); n1 = panel(axes[1], 45)
fig.colorbar(ScalarMappable(norm=n1, cmap="viridis"), ax=axes, fraction=0.025, label="σ  (GP posterior std vs buffer)")
fig.suptitle("One Eq-9 σ-tilt step — 80 candidate windows scored by σ=GP(φ_s), tilted by exp(σ/β), 1 resampled",
             fontsize=12)
fig.savefig("figures/sigma_tilt_step.png", dpi=140, bbox_inches="tight")
print("saved figures/sigma_tilt_step.png; cold-start σ (empty buffer) =",
      round(float(GPUncertainty(kernel="rbf", lengthscale=0.2, normalize=True).sigma(
          pol.phi_s_at(pol.sample_window(torch.tensor(recs[0][0], device=dev), torch.tensor(recs[0][1], device=dev),
                       torch.tensor(recs[0][2], device=dev), n=3), torch.tensor(recs[0][0], device=dev),
                       torch.tensor(recs[0][1], device=dev), torch.tensor(recs[0][2], device=dev), s=S)).mean()), 3),
      "(expect 1.0 => everything maximally novel at empty buffer)")
