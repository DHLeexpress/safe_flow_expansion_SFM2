"""HP capacity check (user 'Important'): is the pretrained FM's FIRST-CONTROL distribution multimodal at
obstacle-encounter states? Probe states from an unseen expert rollout (t=0, first encounter, mid). At each:
sample n windows at temp 1.0 and 1.3 → panels: [local scene + rolled candidates | u₀ scatter in control space],
annotated with the LEFT/RIGHT split of window endpoints w.r.t. the bearing to the nearest obstacle.
Usage: python hp_mm_check.py --policy results/hp_chessboard/pretrained_hp.pt --tag pretrained450"""
import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

import grid_scene as GS
import grid_feats as GF
import grid_rollout as GR
import stage2_grid_data as SD
import grid_hp_expt as HP
import hp_arch_sweep as ARCH

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "hp_test")
os.makedirs(FIG, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def pillars_of(obs):
    m = (obs[:, 0] > 0.7) & (obs[:, 0] < 5.3) & (obs[:, 1] > 0.7) & (obs[:, 1] < 5.3)
    return obs[m]


def probe_states(env, gamma=0.5, seed=700):
    cfg = GS.mode1_config()
    states, controls = SD.rollout_full(env, gamma, cfg, seed)
    obs = env.obstacles.detach().cpu().numpy()
    pil = pillars_of(obs)
    enc = []                                          # states with an interior PILLAR AHEAD within 0.9 m
    for t in range(2, len(controls) - 11):
        v = states[t, 2:4]
        if np.linalg.norm(v) < 0.15:
            continue
        dvec = pil[:, :2] - states[t, :2][None]
        dist = np.linalg.norm(dvec, axis=1) - pil[:, 2]
        ahead = (dvec @ (v / np.linalg.norm(v))) > 0.2
        if ((dist < 0.9) & ahead).any():
            enc.append(t)
    t_enc = enc[0] if enc else len(controls) // 3
    t_mid = enc[len(enc) // 2] if len(enc) > 2 else min(len(controls) - 11, t_enc + 15)
    picks = [1, t_enc, t_mid]
    g, l, h, u = SD.windows_from(states, controls, env, gamma)
    return states, [(t, g[t], l[t], h[t]) for t in picks]


def lateral_split(st, ends, obs):
    pil = pillars_of(obs)                             # split w.r.t. nearest interior PILLAR, not boundary walls
    d = pil[:, :2] - st[None, :2]
    j = int(np.argmin(np.linalg.norm(d, axis=1)))
    b = d[j] / max(1e-6, np.linalg.norm(d[j]))                     # bearing to nearest obstacle
    nvec = np.array([-b[1], b[0]])
    lat = (ends - st[None, :2]) @ nvec
    L, R = float((lat < -0.05).mean()), float((lat > 0.05).mean())
    return min(L, R), lat, j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="results/hp_chessboard/pretrained_hp.pt")
    ap.add_argument("--tag", default="pretrained450")
    ap.add_argument("--n", type=int, default=256)
    a = ap.parse_args()
    ck = torch.load(a.policy, map_location="cpu", weights_only=False)
    pol, _ = (ARCH.load_arch(a.policy, device=DEV) if "variant" in ck else HP.load_hp(a.policy, device=DEV))
    env = GS.make_grid()
    obs = env.obstacles.detach().cpu().numpy()
    states, probes = probe_states(env)
    fig, axes = plt.subplots(len(probes), 3, figsize=(15, 4.6 * len(probes)))
    scores = []
    for ri, (t, g, l, h) in enumerate(probes):
        st = states[t]
        gt, lt, ht = (torch.tensor(np.asarray(x), device=DEV) for x in (g, l, h))
        row = {}
        for temp in (1.0, 1.3):
            with torch.no_grad():
                U = pol.sample_window(gt, lt, ht, n=a.n, temp=temp, nfe=8).detach().cpu().numpy()
            rolls = GR.di_rollout_batch(st, U, 0.1)
            sc, lat, j = lateral_split(st, rolls[:, -1], obs)
            row[temp] = (U, rolls, sc, lat)
        scores.append((t, row[1.0][2], row[1.3][2]))
        ax = axes[ri][0]
        for (ox, oy, r) in obs:
            ax.add_patch(Circle((ox, oy), r, facecolor="#d9c8e3", edgecolor="#9b72aa", lw=.5, alpha=.8))
        U, rolls, sc, lat = row[1.0]
        for i in range(0, a.n, 2):
            P = np.r_[st[None, :2], rolls[i]]
            ax.plot(P[:, 0], P[:, 1], "-", color="#2ca02c" if lat[i] > 0.05 else ("#1f77b4" if lat[i] < -0.05 else "#999"),
                    lw=.7, alpha=.5)
        ax.plot(states[:t + 1, 0], states[:t + 1, 1], "k-", lw=1.4, alpha=.7)
        ax.scatter([st[0]], [st[1]], s=55, c="white", edgecolor="k", zorder=6)
        ax.set_xlim(st[0] - 1.8, st[0] + 1.8); ax.set_ylim(st[1] - 1.8, st[1] + 1.8)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"t={t} — window endpoints, split {sc*100:.0f}%/side (temp 1.0)\nblue=left green=right of nearest-obstacle bearing", fontsize=9)
        for ci, temp in enumerate((1.0, 1.3)):
            U, rolls, sc, lat = row[temp]
            ax = axes[ri][ci + 1]
            ax.add_patch(Rectangle((-1, -1), 2, 2, fill=False, edgecolor="#999", ls="--", lw=.8))
            ax.scatter(U[:, 0, 0], U[:, 0, 1], s=9, c=np.where(lat > 0.05, "#2ca02c", np.where(lat < -0.05, "#1f77b4", "#bbb")), alpha=.7)
            ax.axhline(0, color="#eee"); ax.axvline(0, color="#eee")
            ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.25); ax.set_aspect("equal")
            ax.set_title(f"u₀ first-control distribution — temp {temp}, split {sc*100:.0f}%/side", fontsize=9.5)
            ax.grid(alpha=.2)
    fig.suptitle(f"MULTIMODALITY CHECK [{a.tag}] — {a.n} samples/probe; rows: start / FIRST ENCOUNTER / mid. "
                 "A capable model must place u₀ peaks on BOTH sides at the encounter row.", fontsize=12)
    fig.tight_layout()
    out = os.path.join(FIG, f"mm_check_{a.tag}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("SPLITS (t, temp1.0, temp1.3):", [(t, f"{s1*100:.0f}%", f"{s13*100:.0f}%") for t, s1, s13 in scores], flush=True)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
