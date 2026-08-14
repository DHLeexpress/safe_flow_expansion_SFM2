"""Preview ETH/UCY ego+crowd episodes as a grid of animations, so we can eyeball the dataset before committing
to full processing. Each cell animates the surrounding pedestrians (purple, with trails) and the EGO (green) over
the 80-step window.

  python -m cfm_mppi.data.visualize_crowd_episodes --pkl dataset/eth_crowd_scenes.pkl
"""
from __future__ import annotations
import argparse, os, pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

FIG = "overnight_run_2026-06-28/figures"; os.makedirs(FIG, exist_ok=True)


def pick(eps, sources, k, min_crowd=6, min_ego_move=1.0):
    out = []
    for src in sources:
        cand = []
        for i, e in enumerate(eps):
            if e["source"] != src:
                continue
            nc = (~np.isnan(e["obstacles_seq"][..., 0]).all(0)).sum()
            move = float(np.linalg.norm(e["goal"] - e["start"]))
            if nc >= min_crowd and move >= min_ego_move:
                cand.append((nc, i))
        cand.sort(reverse=True)
        out += [i for _, i in cand[:k]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="dataset/eth_crowd_scenes.pkl")
    ap.add_argument("--sources", nargs="+", default=["biwi_eth", "biwi_hotel", "crowds_zara02", "students003"])
    ap.add_argument("--per-source", type=int, default=2)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--out", default="crowd_episodes_preview.gif")
    args = ap.parse_args()
    eps = pickle.load(open(args.pkl, "rb"))
    idxs = pick(eps, args.sources, args.per_source)
    print("previewing episodes:", idxs)
    chosen = [eps[i] for i in idxs]
    T = chosen[0]["obstacles_seq"].shape[0]

    cols = args.per_source; rows = len(args.sources)
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.4 * rows), squeeze=False)
    flat = [axes[r][c] for r in range(rows) for c in range(cols)]
    # per-episode window
    lims = []
    for e in chosen:
        obs = e["obstacles_seq"][..., :2].reshape(-1, 2); obs = obs[~np.isnan(obs).any(1)]
        pts = np.vstack([obs, e["ego_seq"][:, :2]])
        pad = 1.0
        lims.append(((pts[:, 0].min() - pad, pts[:, 0].max() + pad), (pts[:, 1].min() - pad, pts[:, 1].max() + pad)))

    def draw(t):
        for k, (ax, e) in enumerate(zip(flat, chosen)):
            ax.clear(); (xl, yl) = lims[k]
            ob = e["obstacles_seq"][t]                          # [N,3]
            ok = ~np.isnan(ob[:, :2]).any(1); peds = ob[ok]
            tr0 = max(0, t - 8)
            # crowd trails + current
            for j in range(e["obstacles_seq"].shape[1]):
                seg = e["obstacles_seq"][tr0:t + 1, j, :2]; seg = seg[~np.isnan(seg).any(1)]
                if len(seg) >= 2:
                    ax.plot(seg[:, 0], seg[:, 1], "-", color="#7b3294", lw=0.6, alpha=0.3, zorder=2)
            ax.scatter(peds[:, 0], peds[:, 1], s=40, c="#7b3294", alpha=0.7, edgecolor="#4d004b", zorder=4)
            # ego
            ego = e["ego_seq"]
            ax.plot(ego[:t + 1, 0], ego[:t + 1, 1], "-", color="#1a9850", lw=1.6, zorder=5)
            ax.scatter([ego[t, 0]], [ego[t, 1]], s=70, c="#1a9850", edgecolor="k", zorder=9)
            ax.scatter([ego[0, 0]], [ego[0, 1]], marker="o", s=30, c="none", edgecolor="#1a9850", zorder=6)
            ax.scatter([ego[-1, 0]], [ego[-1, 1]], marker="*", s=120, c="#d62728", edgecolor="k", zorder=6)
            ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            nc = (~np.isnan(e["obstacles_seq"][..., 0]).all(0)).sum()
            ax.set_title(f"{e['source']}  ({nc} peds)", fontsize=8)
        fig.suptitle(f"ETH/UCY ego(green)+crowd(purple) episodes · t={t}/{T}  ·  star=ego goal", fontsize=11)
        return []

    anim = FuncAnimation(fig, draw, frames=T, interval=1000 // args.fps)
    p = os.path.join(FIG, args.out)
    anim.save(p, writer=PillowWriter(fps=args.fps), dpi=85); print("saved", p)


if __name__ == "__main__":
    main()
