"""10k deliverables: (1) trajectory overlay pretrained vs 10k, (2) coverage+validity curves, (3) output-variance
vs iteration (all from the current-loss 10k run), (4) 16-case snapshot at the FIXED training config temp=2,β=0.1."""
import os, json
import _paths, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import grid_scene as GS, grid_policy as GP, grid_rollout as GR, grid_feats as GF
from di_grid_viz import di_step
from uncertainty import GPUncertainty

dev = "cuda"; env = GS.make_grid(); obs = env.obstacles.numpy(); rr = float(env.r_robot); goal = env.goal.numpy()
FIG = "figures"; TEN = "results/expt_longrun/current_g0.5.pt"


def draw_grid(ax):
    for k in range(6):
        ax.axvline(k, color="#eee", lw=.6); ax.axhline(k, color="#eee", lw=.6)
    ax.add_patch(Rectangle((0, 0), 5, 5, fill=False, edgecolor="#555", lw=1.4))
    for j, (ox, oy, r) in enumerate(obs):
        ax.add_patch(Circle((ox, oy), r, facecolor="#b8b8b8" if j >= 16 else "#c8a2c8", edgecolor="#777", lw=.4, alpha=.8))
    ax.scatter([0], [0], s=55, marker="s", c="#00a000", edgecolor="k", zorder=8)
    ax.scatter([5], [5], marker="*", s=200, c="gold", edgecolor="k", zorder=8)
    ax.set_xlim(-.6, 5.6); ax.set_ylim(-.6, 5.6); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


# ---------- (1) trajectory overlay: pretrained vs 10k ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for ax, (name, path) in zip(axes, [("pretrained (collapsed)", "pretrained.pt"), ("after 10k expansion", TEN)]):
    pol, _ = GP.load_policy(path, device=dev)
    draw_grid(ax)
    for i in range(30):
        p = GR.fm_deploy(pol, env, 0.5, T=250, nfe=10, device=dev)["path"]
        ax.plot(p[:, 0], p[:, 1], "-", color="#2ca02c", lw=1.0, alpha=0.5, zorder=5)
    ax.set_title(f"{name} — 30 rollouts (γ=0.5)", fontsize=12)
fig.suptitle("Trajectory overlay: how 10k ACTFLOW iterations de-collapse the policy off the diagonal", fontsize=13)
fig.tight_layout(); fig.savefig(f"{FIG}/trajectory_overlay_pretrained_vs_10k.png", dpi=140); plt.close(fig)
print("saved trajectory_overlay_pretrained_vs_10k.png")

# ---------- (2)+(3) coverage, validity, output-variance vs iteration ----------
h = json.load(open("results/expt_longrun/current_history.json"))
it = [x["iter"] for x in h]
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
ax[0].plot(it, [x["coverage"] * 100 for x in h], "-o", ms=3, color="#1f77b4"); ax[0].axhline(90, ls="--", color="#999")
ax[1].plot(it, [x["validity"] * 100 for x in h], "-o", ms=3, color="#2ca02c")
ov = [(x["iter"], x["out_var"]) for x in h if "out_var" in x]
ax[2].plot([a for a, _ in ov], [b for _, b in ov], "-o", ms=3, color="#d62728")
for a, t in zip(ax, ["coverage % (distinct / 252)", "validity %", "output-variance (policy spread, m²)"]):
    a.set_xlabel("ACTFLOW iteration"); a.set_title(t); a.grid(alpha=.25)
fig.suptitle("10k current-loss run (γ=0.5, temp=2, β=1/10): coverage · validity · output-variance vs iteration",
             fontsize=12)
fig.tight_layout(); fig.savefig(f"{FIG}/expt_10k_curves.png", dpi=140); plt.close(fig)
print("saved expt_10k_curves.png")

# ---------- (4) 16-case snapshot at the FIXED training config (temp=2, β=1/10) ----------
pol, _ = GP.load_policy(TEN, device=dev); S, TEMP, BETA, N = 0.3, 2.0, 0.1, 64
bg, bl, bh, bU, states = [], [], [], [], []
st = env.x0.numpy().astype(np.float32); hist = []
for t in range(90):
    g = GF.axis_grid(st[:2], obs, rr); l = GF.low5(st, goal, 0.5)
    hh = GF.hist_pad(np.array(hist[-16:]) if hist else np.zeros((0, 2)), 16)
    with torch.no_grad():
        U = pol.sample_window(torch.tensor(g, device=dev), torch.tensor(l, device=dev), torch.tensor(hh, device=dev),
                              n=1, nfe=8)[0].cpu().numpy()
    bg.append(g); bl.append(l); bh.append(hh)
    states.append((st.copy(), g, l, hh)); a = U[0]; st = di_step(st, a, dt=env.dt); hist.append(a)
unc = GPUncertainty(kernel="rbf", lengthscale=0.2, lam=1e-2, normalize=True)
with torch.no_grad():
    G = torch.tensor(np.array(bg), device=dev); L = torch.tensor(np.array(bl), device=dev); H = torch.tensor(np.array(bh), device=dev)
    Ub = torch.stack([pol.sample_window(G[i], L[i], H[i], n=1, nfe=6)[0] for i in range(len(bg))])
    unc.set_buffer(pol.phi_s(Ub, pol.ctx_from(G, L, H), s=S))
sel_states = states[6::5][:16]
cells, allsig = [], []
for (st0, g, l, hh) in sel_states:
    gT, lT, hT = torch.tensor(g, device=dev), torch.tensor(l, device=dev), torch.tensor(hh, device=dev)
    with torch.no_grad():
        Uc = pol.sample_window(gT, lT, hT, n=N, temp=TEMP, nfe=8)
        sig = unc.sigma(pol.phi_s_at(Uc, gT, lT, hT, s=S)).cpu().numpy()
    Uc_np = Uc.cpu().numpy(); w = np.exp((sig - sig.max()) / BETA); selk = int(GR.systematic_resample(torch.tensor(w), 1)[0])
    cells.append((st0, GR.di_rollout_batch(st0, Uc_np, env.dt), sig, selk)); allsig.append(sig)
nrm = Normalize(float(np.concatenate(allsig).min()), float(np.concatenate(allsig).max()))
fig, axes = plt.subplots(4, 4, figsize=(15, 15))
for ax, (st0, pos, sig, selk) in zip(axes.ravel(), cells):
    for (ox, oy, r) in obs:
        if abs(ox - st0[0]) < 2.3 and abs(oy - st0[1]) < 2.3:
            ax.add_patch(Circle((ox, oy), r, facecolor="#c8a2c8", edgecolor="#777", lw=.4, alpha=.7))
    for i in np.argsort(sig):
        ax.plot(pos[i, :, 0], pos[i, :, 1], "-", color=plt.cm.viridis(nrm(sig[i])), lw=0.8, alpha=.6, zorder=4)
    ax.plot(pos[selk, :, 0], pos[selk, :, 1], "-", color="#e6191b", lw=2.3, zorder=6)
    ax.scatter([st0[0]], [st0[1]], s=40, marker="o", c="white", edgecolor="k", zorder=7)
    ax.set_xlim(st0[0] - 2, st0[0] + 2); ax.set_ylim(st0[1] - 2, st0[1] + 2); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"x={st0[0]:.1f},y={st0[1]:.1f}", fontsize=8)
fig.colorbar(ScalarMappable(norm=nrm, cmap="viridis"), ax=axes, fraction=0.02, label="σ (GP posterior std, φ_s)")
fig.suptitle("16 scenarios at the FIXED training config (10k policy, temp=2, β=1/10) — 64 candidates by σ, "
             "red = winner", fontsize=13)
fig.savefig(f"{FIG}/snapshot_16case_temp2_b10.png", dpi=120, bbox_inches="tight"); plt.close(fig)
print("saved snapshot_16case_temp2_b10.png")
