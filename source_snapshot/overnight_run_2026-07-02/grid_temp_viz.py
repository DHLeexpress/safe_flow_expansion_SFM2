"""8x4 temperature/β study of the Eq-9 σ-tilt on the margin-0 pretrained FM (no broad proposal).
Rows 1-4: FM-candidate sampling temperature {0.5,1,1.5,2} at the current β=1/50.
Rows 5-8: same temperatures at β×10=1/5 (softer tilt).
Columns: 4 scenario states (snapshots along a pretrained rollout).
Each cell: 64 FM candidate windows colored by σ=GP(φ_s) vs a fixed buffer, red = exp(σ/β)-resampled winner.
Shared σ colorbar. Shows how temperature widens exploration and how small β over-greedily picks OOD candidates.
"""
import os
import _paths, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import grid_scene as GS, grid_policy as GP, grid_rollout as GR, grid_feats as GF
from di_grid_viz import di_step
from uncertainty import GPUncertainty

POLICY = os.environ.get("POLICY", "pretrained.pt"); SUF = os.environ.get("SUFFIX", "")
dev = "cuda"; env = GS.make_grid(); obs = env.obstacles.numpy(); rr = float(env.r_robot); goal = env.goal.numpy()
pol, _ = GP.load_policy(POLICY, device=dev)                        # POLICY env var (default: margin-0 pretrained)
S, BETA, N = 0.3, 1.0 / 50, 64
TEMPS = [0.5, 1.0, 1.5, 2.0]; BETAS = [BETA, BETA * 10]
SCEN_STEPS = [6, 22, 38, 54]

# roll a plain γ=0.5 trajectory: collect buffer windows + the 4 scenario states
bg, bl, bh, bU, scen = [], [], [], [], []
st = env.x0.numpy().astype(np.float32); hist = []
for t in range(64):
    g = GF.axis_grid(st[:2], obs, rr); l = GF.low5(st, goal, 0.5)
    h = GF.hist_pad(np.array(hist[-16:]) if hist else np.zeros((0, 2)), 16)
    with torch.no_grad():
        U = pol.sample_window(torch.tensor(g, device=dev), torch.tensor(l, device=dev),
                              torch.tensor(h, device=dev), n=1, nfe=8)[0].cpu().numpy()
    bg.append(g); bl.append(l); bh.append(h); bU.append(U)
    if t in SCEN_STEPS:
        scen.append((st.copy(), g, l, h))
    a = U[0]; st = di_step(st, a, dt=env.dt); hist.append(a)

unc = GPUncertainty(kernel="rbf", lengthscale=0.2, lam=1e-2, normalize=True)
with torch.no_grad():
    G = torch.tensor(np.array(bg), device=dev); L = torch.tensor(np.array(bl), device=dev)
    H = torch.tensor(np.array(bh), device=dev); U = torch.tensor(np.array(bU), device=dev)
    unc.set_buffer(pol.phi_s(U, pol.ctx_from(G, L, H), s=S))

cells, allsig = {}, []
for ti, temp in enumerate(TEMPS):
    for bi, beta in enumerate(BETAS):
        row = bi * 4 + ti
        for col, (st0, g, l, h) in enumerate(scen):
            gT, lT, hT = torch.tensor(g, device=dev), torch.tensor(l, device=dev), torch.tensor(h, device=dev)
            with torch.no_grad():
                Uc = pol.sample_window(gT, lT, hT, n=N, temp=temp, nfe=8)
                sig = unc.sigma(pol.phi_s_at(Uc, gT, lT, hT, s=S)).cpu().numpy()
            Uc_np = Uc.cpu().numpy()
            w = np.exp((sig - sig.max()) / beta); sel = int(GR.systematic_resample(torch.tensor(w), 1)[0])
            cells[(row, col)] = (st0, GR.di_rollout_batch(st0, Uc_np, env.dt), sig, sel, temp, beta)
            allsig.append(sig)
nrm = Normalize(float(np.concatenate(allsig).min()), float(np.concatenate(allsig).max()))

fig, axes = plt.subplots(8, 4, figsize=(15, 26))
for (row, col), (st0, pos, sig, sel, temp, beta) in cells.items():
    ax = axes[row][col]
    for (ox, oy, r) in obs:
        if abs(ox - st0[0]) < 2.3 and abs(oy - st0[1]) < 2.3:
            ax.add_patch(Circle((ox, oy), r, facecolor="#c8a2c8", edgecolor="#777", lw=.4, alpha=.7))
    order = np.argsort(sig)
    for i in order:
        ax.plot(pos[i, :, 0], pos[i, :, 1], "-", color=plt.cm.viridis(nrm(sig[i])), lw=0.8, alpha=.6, zorder=4)
    ax.plot(pos[sel, :, 0], pos[sel, :, 1], "-", color="#e6191b", lw=2.4, zorder=6)
    ax.scatter([st0[0]], [st0[1]], s=40, marker="o", c="white", edgecolor="k", zorder=7)
    ax.set_xlim(st0[0] - 2.0, st0[0] + 2.0); ax.set_ylim(st0[1] - 2.0, st0[1] + 2.0)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if col == 0:
        ax.set_ylabel(f"temp={temp}\nβ=1/{1/beta:.0f}", fontsize=10)
    if row == 0:
        ax.set_title(f"scenario {col+1}\n(x={st0[0]:.1f}, y={st0[1]:.1f})", fontsize=9)
fig.colorbar(ScalarMappable(norm=nrm, cmap="viridis"), ax=axes, fraction=0.015, pad=0.01,
             label="σ = GP posterior std (φ_s) vs buffer")
fig.suptitle("Eq-9 σ-tilt: FM-candidate temperature (rows, per β-block) × scenario (cols) — 64 candidates by σ, "
             "red = exp(σ/β) winner.  Top 4 rows β=1/50, bottom 4 rows β=1/5 (10×, softer).", fontsize=12, y=0.995)
fig.savefig(f"figures/temp_beta_study{SUF}.png", dpi=125, bbox_inches="tight")
print(f"saved figures/temp_beta_study{SUF}.png; global σ range", round(nrm.vmin, 3), round(nrm.vmax, 3))
