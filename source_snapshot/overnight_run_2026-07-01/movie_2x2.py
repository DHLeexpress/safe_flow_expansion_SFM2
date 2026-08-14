"""Render the 2x2 Safe-Flow-Expansion movie from movie_record.pkl:
  (0,0) live certified trajectories, colored by mode, NEW mode highlighted at its discovery round;
  (0,1) live kernel matrix K=<phi_s,phi_s'> (sorted by window direction) — block structure emerges with modes;
  (1,0) live sigma histogram (GP over phi_s) + ESS;
  (1,1) certified coverage-near-obstacles growth + mode-discovery markers.
Discovery rounds are held longer and flashed.
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

import _paths
import config as C

PALETTE = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e", "#17becf"]


def render(scene, hold=5, log=print):
    with open(C.scene_result(scene, "movie_record.pkl"), "rb") as f:
        D = pickle.load(f)
    recs = D["records"]; obs = D["obs"]; goal = D["goal"]; x0 = D["x0"]
    xlim, ylim, all_modes = D["xlim"], D["ylim"], D["all_modes"]
    cmap = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(all_modes)}

    acc_by_round, disc_round, acc = [], {}, {m: [] for m in all_modes}
    for i, rec in enumerate(recs):
        for (p, m, g) in rec["trajs"]:
            if m in acc:
                acc[m].append(p)
        for m in rec["new"]:
            disc_round.setdefault(m, i)
        acc_by_round.append({m: list(v) for m, v in acc.items()})
    covs = [rec["cov"] for rec in recs]
    R = len(recs)
    frame_rounds = []
    for i, rec in enumerate(recs):
        frame_rounds += [i] * (hold + (10 if rec["new"] else 0))

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    def frame(fi):
        i = frame_rounds[fi]; rec = recs[i]
        for a in ax.ravel():
            a.clear()
        # (0,0) live certified trajectories
        a = ax[0][0]
        for (ox, oy, rr) in obs:
            a.add_patch(Circle((ox, oy), rr, facecolor="#c8a2c8", edgecolor="#7b3294", lw=0.7, alpha=0.7, zorder=3))
        for m, paths in acc_by_round[i].items():
            for p in paths[-45:]:
                a.plot(p[:, 0], p[:, 1], "-", color=cmap[m], lw=0.7, alpha=0.32, zorder=4)
        new_here = rec["new"]
        for (p, m, g) in rec["trajs"]:
            if m in new_here:
                a.plot(p[:, 0], p[:, 1], "-", color=cmap.get(m, "k"), lw=2.6, alpha=1.0, zorder=8)
        a.scatter([x0[0]], [x0[1]], s=42, c="#00a000", edgecolor="k", zorder=9)
        a.scatter([goal[0]], [goal[1]], marker="*", s=150, c="gold", edgecolor="k", zorder=9)
        a.set_xlim(*xlim); a.set_ylim(*ylim); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        if new_here:
            a.set_title("NEW MODE: " + ", ".join(new_here) + " !", fontsize=12, color="#d62728", fontweight="bold")
        else:
            a.set_title("certified trajectories (color = mode)", fontsize=11)
        # legend of modes discovered
        for j, (m, dr) in enumerate(sorted(disc_round.items(), key=lambda kv: kv[1])):
            if dr <= i:
                a.text(0.02, 0.97 - 0.06 * j, f"● {m}", color=cmap[m], transform=a.transAxes,
                       fontsize=8, va="top", fontweight="bold")
        # (0,1) live kernel matrix
        b = ax[0][1]
        b.imshow(rec["K"], cmap="magma", vmin=-1, vmax=1)
        b.set_title(r"live kernel $K=\langle\phi_s,\phi_s'\rangle$ (sorted by window direction)", fontsize=10)
        b.set_xticks([]); b.set_yticks([])
        # (1,0) live sigma histogram
        c = ax[1][0]
        c.hist(rec["sigma"], bins=22, color="#4477aa", range=(0, max(0.35, rec["sigma"].max())))
        c.set_title(fr"$\sigma$ histogram (GP over $\phi_s$)   ESS={rec['ess']:.0f}/{len(rec['sigma'])}", fontsize=10)
        c.set_xlabel(r"$\sigma$ (posterior std)")
        # (1,1) coverage growth
        d = ax[1][1]
        d.plot(range(1, i + 2), covs[:i + 1], "-o", color="#2ca02c", ms=3)
        for m, dr in disc_round.items():
            if dr <= i:
                d.axvline(dr + 1, color=cmap[m], ls="--", lw=1.2, alpha=0.8)
        d.set_xlim(1, R); d.set_ylim(0, max(0.6, max(covs) * 1.12))
        d.set_xlabel("iteration (round)")
        d.set_title(f"certified coverage near obstacles = {rec['cov']:.2f}   "
                    f"modes {len(rec['modes'])}/{len(all_modes)}", fontsize=10)
        fig.suptitle(f"[{scene}] Safe Flow Expansion — round {rec['round']}/{R}  "
                     f"(iterate until the space near obstacles is covered)", fontsize=12)
        return []

    anim = FuncAnimation(fig, frame, frames=len(frame_rounds), interval=140)
    mp4 = C.scene_fig(scene, "expansion_2x2.mp4")
    try:
        anim.save(mp4, writer=FFMpegWriter(fps=8, bitrate=3000), dpi=110)
    except Exception as e:
        log(f"[mp4] fail ({e})")
    anim.save(mp4[:-4] + ".gif", writer=PillowWriter(fps=6), dpi=80)
    frame(len(frame_rounds) - 1); fig.savefig(mp4[:-4] + ".png", dpi=120); plt.close(fig)
    log(f"[{scene}] 2x2 movie ({len(recs)} rounds) → {mp4}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="gap", choices=C.SCENE_NAMES)
    args = ap.parse_args()
    render(args.scene)


if __name__ == "__main__":
    main()
