"""v2 expansion visualizations (user 3d).

--mode multimodal : columns = checkpoints (iter 0 / 500 / 1000 / 2000), rows =
   (1) local map: 64 FM candidate windows colored by σ, RED = Eq-9 winner, orange dashed = max-σ,
       blue dashed = min-σ (correspondence markers repeat in rows 2-3);
   (2) σ distribution: grey = p (current FM policy, uniform weights), red = q* ∝ p·exp(σ/β)
       (importance-tilted), with winner/max/min markers and Var(σ) in the title — the 20-D window
       distribution projected onto the ONE statistic the tilt acts on;
   (3) kernel matrix K(φ_i,φ_j) among the candidates, ordered by net-displacement angle — block
       structure = modes; unimodal → multimodal across the checkpoint columns.
   Animated over rollout wall-clock t = 0,2,…,20; the GP buffer at frame t = that rollout's own past
   windows (cold start at t=0 → σ≡1, uniform q*).
--mode progress   : expansion movie from a run's snapshots: covered staircases growing on the map +
   the σ-tilted exploration rollout every snapshot + live A) validity2 B) var(σ) C) coverage_cum
   D) coverage_final curves.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

import _paths  # noqa: F401
import grid_scene as GS
import grid_feats as GF
import grid_policy2 as GP2
import grid_rollout as GR
from di_grid_viz import di_step
from uncertainty import GPUncertainty

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
GCOL = {"0.1": "#3b6fd6", "0.5": "#2ca02c", "1.0": "#d62728"}


def _l2n(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def save_anim(anim, base, fps):
    gif = base + ".gif"
    anim.save(gif, writer=PillowWriter(fps=fps), dpi=90)
    try:
        anim.save(base + ".mp4", writer=FFMpegWriter(fps=max(fps, 4), bitrate=2400), dpi=115)
    except Exception as e:
        print(f"[mp4] skip ({e})")
    print(f"saved {base}.{{gif,mp4}}")


# ===================================================================== multimodal
def rollout_with_windows(policy, env, gamma, T, device, seed=11):
    """Plain rollout that also stores each step's conditioning + executed window (the GP buffer source)."""
    torch.manual_seed(seed); np.random.seed(seed)
    obs = env.obstacles.detach().cpu().numpy(); rr = float(env.r_robot)
    goal = env.goal.detach().cpu().numpy()
    st = env.x0.detach().cpu().numpy().astype(np.float32)
    hist, steps = [], []
    for t in range(T):
        g = GF.axis_grid(st[:2], obs, rr); l = GF.low5(st, goal, gamma)
        h = GF.hist_pad(np.array(hist[-GF.K_HIST:]) if hist else np.zeros((0, 2)), GF.K_HIST)
        with torch.no_grad():
            U = policy.sample_window(torch.tensor(g, device=device), torch.tensor(l, device=device),
                                     torch.tensor(h, device=device), n=1, nfe=8)[0].cpu().numpy()
        steps.append(dict(state=st.copy(), g=g, l=l, h=h, U=U.astype(np.float32)))
        a = U[0]
        st = di_step(st, np.asarray(a, np.float32), dt=env.dt)
        hist.append(np.asarray(a, np.float32))
        if np.linalg.norm(st[:2] - goal) < 0.45:
            break
    return steps


def frame_data(policy, env, steps, t, cfg, device, N=64):
    """Candidates + σ + tilt at rollout time t, buffer = this rollout's own windows [0..t)."""
    t = min(t, len(steps) - 1)
    sd = steps[t]
    gT = torch.tensor(sd["g"], device=device); lT = torch.tensor(sd["l"], device=device)
    hT = torch.tensor(sd["h"], device=device)
    unc = GPUncertainty(kernel="rbf", lengthscale=cfg["ell"], lam=1e-2, normalize=True)
    if t > 0:
        with torch.no_grad():
            feats = []
            for p in steps[:t]:
                feats.append(policy.phi_s_at(torch.tensor(p["U"], device=device)[None],
                                             torch.tensor(p["g"], device=device),
                                             torch.tensor(p["l"], device=device),
                                             torch.tensor(p["h"], device=device), s=cfg["s"]))
            unc.set_buffer(torch.cat(feats))
    with torch.no_grad():
        Uc = policy.sample_window(gT, lT, hT, n=N, temp=cfg["temp"], nfe=8)
        phi = policy.phi_s_at(Uc, gT, lT, hT, s=cfg["s"])
        sig = unc.sigma(phi).cpu().numpy()
    w = np.exp((sig - sig.max()) / max(cfg["beta"], 1e-6))
    sel = int(GR.systematic_resample(torch.tensor(w / w.sum()), 1)[0])
    pos = GR.di_rollout_batch(sd["state"], Uc.cpu().numpy(), env.dt)
    K = None
    with torch.no_grad():
        ph = _l2n(phi)
        d2 = torch.cdist(ph, ph) ** 2
        K = torch.exp(-d2 / (2 * cfg["ell"] ** 2)).cpu().numpy()
    net = pos[:, -1, :] - sd["state"][:2]
    order = np.argsort(np.arctan2(net[:, 1], net[:, 0]))
    return dict(state=sd["state"], pos=pos, sig=sig, w=w, sel=sel,
                imax=int(sig.argmax()), imin=int(sig.argmin()), K=K, order=order,
                trail=np.array([p["state"][:2] for p in steps[:t + 1]]))


def mode_multimodal(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    env = GS.make_grid(); obs = env.obstacles.detach().cpu().numpy()
    cfg = dict(temp=1.3, beta=0.1, s=0.9, ell=0.2)
    cf = os.path.join(args.rundir, "config.json")
    if os.path.exists(cf):
        c = json.load(open(cf))
        cfg = dict(temp=c.get("temp", 1.3), beta=c.get("beta", 0.1), s=c.get("s", 0.9), ell=c.get("ell", 0.2))
    cols = [("iter 0\n(pretrained)", args.policy0)]
    for it in (500, 1000, 2000):
        p = os.path.join(args.rundir, f"ckpt_{it}.pt" if it < 2000 else "final.pt")
        if os.path.exists(p):
            cols.append((f"iter {it}", p))
    times = list(range(0, 21, 2))
    print(f"multimodal: {len(cols)} checkpoints × t={times}, γ={args.gamma}, cfg={cfg}", flush=True)

    data = []
    for name, path in cols:
        pol, _ = GP2.load_policy2(path, device=dev)
        steps = rollout_with_windows(pol, env, args.gamma, T=max(times) + 12, device=dev)
        data.append((name, [frame_data(pol, env, steps, t, cfg, dev) for t in times]))
    smax = max(fd["sig"].max() for _, fr in data for fd in fr) + 1e-6
    nrm = Normalize(0.0, smax)
    bins = np.linspace(0, smax, 22)

    nc = len(data)
    fig, axes = plt.subplots(3, nc, figsize=(3.9 * nc, 11.3))
    axes = axes.reshape(3, nc)
    cbar_ax = fig.add_axes([0.92, 0.68, 0.013, 0.24])
    fig.colorbar(ScalarMappable(norm=nrm, cmap="viridis"), cax=cbar_ax, label="σ (GP posterior std)")

    def draw(fi):
        for ci, (name, frames) in enumerate(data):
            fd = frames[fi]
            st = fd["state"]; sig = fd["sig"]; pos = fd["pos"]
            axm, axh, axk = axes[0][ci], axes[1][ci], axes[2][ci]
            for a in (axm, axh, axk):
                a.clear()
            # (1) map
            for (ox, oy, r) in obs:
                if abs(ox - st[0]) < 2.4 and abs(oy - st[1]) < 2.4:
                    axm.add_patch(Circle((ox, oy), r, facecolor="#c8a2c8", edgecolor="#777", lw=.4, alpha=.7))
            for i in np.argsort(sig):
                axm.plot(pos[i, :, 0], pos[i, :, 1], "-", color=plt.cm.viridis(nrm(sig[i])), lw=.8, alpha=.55)
            axm.plot(pos[fd["imax"], :, 0], pos[fd["imax"], :, 1], "--", color="#ff7f0e", lw=1.8, zorder=6)
            axm.plot(pos[fd["imin"], :, 0], pos[fd["imin"], :, 1], "--", color="#1f77b4", lw=1.8, zorder=6)
            axm.plot(pos[fd["sel"], :, 0], pos[fd["sel"], :, 1], "-", color="#e6191b", lw=3.2, alpha=.65, zorder=7)
            if len(fd["trail"]) > 1:
                axm.plot(fd["trail"][:, 0], fd["trail"][:, 1], "-", color="#444", lw=1.0, alpha=.7)
            axm.scatter([st[0]], [st[1]], s=42, c="white", edgecolor="k", zorder=8)
            axm.set_xlim(st[0] - 2, st[0] + 2); axm.set_ylim(st[1] - 2, st[1] + 2)
            axm.set_aspect("equal"); axm.set_xticks([]); axm.set_yticks([])
            axm.set_title(f"{name}   t={times[fi]}", fontsize=10)
            # (2) sigma distribution: p vs q*
            wq = fd["w"] / fd["w"].sum()
            axh.hist(sig, bins=bins, weights=np.full(len(sig), 1.0 / len(sig)), color="#999", alpha=.6,
                     label="p (FM policy)")
            axh.hist(sig, bins=bins, weights=wq, histtype="step", color="#e6191b", lw=2.0,
                     label="q* ∝ p·e^{σ/β}")
            for idx, col in ((fd["sel"], "#e6191b"), (fd["imax"], "#ff7f0e"), (fd["imin"], "#1f77b4")):
                axh.axvline(sig[idx], color=col, lw=1.4, ls=":" if idx != fd["sel"] else "-")
            axh.set_xlim(0, smax); axh.set_ylim(0, 1.0)
            axh.set_title(f"Var(σ)={sig.var():.4f}", fontsize=10)
            if ci == 0:
                axh.legend(fontsize=7); axh.set_ylabel("prob. mass")
            axh.set_xlabel("σ", fontsize=8); axh.grid(alpha=.2)
            # (3) kernel matrix, angle-ordered
            o = fd["order"]
            axk.imshow(fd["K"][np.ix_(o, o)], cmap="magma", vmin=0, vmax=1, interpolation="nearest")
            ppos = {int(v): k for k, v in enumerate(o)}
            for idx, col in ((fd["sel"], "#e6191b"), (fd["imax"], "#ff7f0e"), (fd["imin"], "#1f77b4")):
                j = ppos[idx]
                alpha = 0.30 if idx == fd["sel"] else 0.0
                if alpha:
                    axk.axhspan(j - .5, j + .5, color=col, alpha=alpha)
                    axk.axvspan(j - .5, j + .5, color=col, alpha=alpha)
                axk.plot([j], [j], "s", color=col, ms=5)
            axk.set_xticks([]); axk.set_yticks([])
            axk.set_title("kernel K(φᵢ,φⱼ) — angle-ordered; blocks = modes", fontsize=8.5)
        fig.suptitle(f"De-collapse over ACTFLOW iterations (γ={args.gamma}, temp={cfg['temp']}, β={cfg['beta']}) — "
                     "red = Eq-9 winner, orange/blue = max/min-σ candidate (same marks in all rows)",
                     fontsize=11.5)
        return []

    anim = FuncAnimation(fig, draw, frames=len(times), interval=800)
    fig.subplots_adjust(right=0.90, top=0.92, hspace=0.3)
    save_anim(anim, os.path.join(FIG, f"expand2_multimodal_g{args.gamma}"), fps=2)
    plt.close(fig)


# ===================================================================== progress
def stair_vertices(word):
    vx, vy = 0, 0
    pts = [(0.0, 0.0)]
    for ch in word:
        vx, vy = (vx + 1, vy) if ch == "R" else (vx, vy + 1)
        pts.append((float(vx), float(vy)))
    return np.array(pts)


def mode_progress(args):
    env = GS.make_grid(); obs = env.obstacles.detach().cpu().numpy()
    rid = os.path.basename(os.path.normpath(args.rundir))
    snaps = pickle.load(open(os.path.join(args.rundir, "snapshots.pkl"), "rb"))
    hist = json.load(open(os.path.join(args.rundir, "history.json")))
    ex = [s for s in snaps if s["kind"] == "explore"]
    meas = {(s["iter"], s["gamma"]): s for s in snaps if s["kind"] == "measure"}
    gammas = ["0.5", "1.0", "0.1"]
    hx = [r["iter"] for r in hist]
    hv = [float(np.mean([r[f"g{g}"]["validity"] for g in map(float, gammas)])) * 100 for r in hist]
    hvs = [r.get("var_sigma", 0.0) for r in hist]
    hc = [float(np.mean([r[f"g{g}"]["coverage_cum"] for g in map(float, gammas)])) * 100 for r in hist]
    hf = [float(np.mean([r[f"g{g}"]["coverage_final"] for g in map(float, gammas)])) * 100 for r in hist]

    fig = plt.figure(figsize=(13.6, 7.4))
    axm = fig.add_axes([0.04, 0.06, 0.52, 0.86])
    axs = [fig.add_axes([0.63, 0.76 - i * 0.235, 0.34, 0.185]) for i in range(4)]

    def draw(fi):
        s = ex[fi]
        axm.clear()
        for k in range(6):
            axm.axvline(k, color="#eee", lw=.6); axm.axhline(k, color="#eee", lw=.6)
        axm.add_patch(Rectangle((0, 0), 5, 5, fill=False, edgecolor="#555", lw=1.3))
        for j, (ox, oy, r) in enumerate(obs):
            axm.add_patch(Circle((ox, oy), r, facecolor="#b8b8b8" if j >= 16 else "#c8a2c8",
                                 edgecolor="#777", lw=.3, alpha=.8))
        for gi, g in enumerate(gammas):
            off = (gi - 1) * 0.045
            for wd in s.get("covered_sets", {}).get(g, []):
                v = stair_vertices(wd)
                axm.plot(v[:, 0] + off, v[:, 1] + off, "-", color=GCOL[g], lw=1.0, alpha=.16, zorder=2)
        g = str(s["gamma"])
        p = s["path"]
        axm.plot(p[:, 0], p[:, 1], "-", color=GCOL.get(g, "#333"), lw=2.2, alpha=.95, zorder=6,
                 label=f"σ-tilted exploration (γ={g})")
        mp = meas.get((s["iter"] - s["iter"] % 200, float(g)))
        axm.scatter([0], [0], s=45, marker="s", c="#00a000", edgecolor="k", zorder=8)
        axm.scatter([5], [5], marker="*", s=160, c="gold", edgecolor="k", zorder=8)
        cov_txt = "  ".join(f"γ{gg}: {s['covered'][gg]}/252" for gg in gammas if gg in s.get("covered", {}))
        axm.set_title(f"iter {s['iter']} — covered staircases (translucent, by γ)   {cov_txt}", fontsize=10.5)
        axm.legend(loc="upper left", fontsize=8)
        axm.set_xlim(-.55, 5.55); axm.set_ylim(-.55, 5.55); axm.set_aspect("equal")
        axm.set_xticks([]); axm.set_yticks([])
        it = s["iter"]
        for a, (ys, ttl, col) in zip(axs, [(hv, "A) validity2 % (γ-mean)", "#2ca02c"),
                                           (hvs, "B) var(σ)", "#8c1aa8"),
                                           (hc, "C) coverage_cumulative %", "#1f77b4"),
                                           (hf, "D) coverage_final %", "#d62728")]):
            a.clear()
            a.plot(hx, ys, "-o", ms=2.5, color=col, lw=1.3)
            k = int(np.searchsorted(hx, it, side="right")) - 1
            if k >= 0:
                a.plot([hx[k]], [ys[k]], "o", ms=7, mfc="none", mec="k")
            a.axvline(it, color="#bbb", lw=.8, ls="--")
            a.set_title(ttl, fontsize=9); a.grid(alpha=.25)
            a.tick_params(labelsize=7)
        axs[-1].set_xlabel("iteration", fontsize=8)
        fig.suptitle(f"Safe Flow Expansion v2 — run {rid}", fontsize=12)
        return []

    anim = FuncAnimation(fig, draw, frames=len(ex), interval=600)
    save_anim(anim, os.path.join(FIG, f"expand2_progress_{rid}"), fps=2)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", required=True)
    ap.add_argument("--mode", choices=["multimodal", "progress"], required=True)
    ap.add_argument("--policy0", default="pretrained2_w256.pt")
    ap.add_argument("--gamma", type=float, default=0.5)
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    if args.mode == "multimodal":
        mode_multimodal(args)
    else:
        mode_progress(args)
