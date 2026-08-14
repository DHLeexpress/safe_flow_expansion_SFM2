"""TREE VIZ (user spec 2026-07-05): per checkpoint row, one σ-tilt trunk rollout with decaying candidate SPRAYS
at every 1 s node — k = [5,4,4,3,3,2,2,1,1,...] branches importance-resampled via p∝exp(σ/β) from N=64 temp-1.3
candidates, each rolled 1 s in parallel, σ-colored; validity failures RED + labeled (✗G goal / ✗T taskspace /
✗S SOCP) and terminated; trunk continues via the tilt winner. Rows: pretrained, ckpt_1000, ckpt_2000, ...
Usage: python hp_tree_viz.py --ckpts results/hp_arch/res2w256.pt [more.pt ...] --labels pretrained [it1000 ...]
       --gamma 0.5 --temp 1.3 --beta 0.1 --tag <name>"""
import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

import grid_scene as GS
import grid_feats as GF
import grid_metrics as GM
import grid_metrics2 as GM2
import grid_hp_expt as HP
import hp_arch_sweep as ARCH
from uncertainty import GPUncertainty

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "hp_test")
os.makedirs(FIG, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SCHED = [5, 4, 4, 3, 3, 2, 2]                       # then 1 forever
FCOL = {"✗G": "#ff7f0e", "✗T": "#9467bd", "✗S": "#d62728"}   # goal-seeking / taskspace / SOCP
DT, H = 0.1, 10


def load_any(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "variant" in ck:
        return ARCH.load_arch(path, device=DEV)[0]
    if any(k.startswith("trunk.blocks") for k in ck["state_dict"]):   # expansion ckpt of a ResTrunk variant
        gru = any(k.startswith("gru.") for k in ck["state_dict"])     # GRU expansion ckpt (2026-07-06)
        pol = ARCH.build("res2w256_gru" if gru else "res2w256")
        pol.load_state_dict(ck["state_dict"])
        return pol.to(DEV).eval()
    return HP.load_hp(path, device=DEV)[0]


def dataset_buffer(pol, n=512, s=0.9):
    """σ buffer stand-in: φ_s of n random TRAINING windows under this policy (documented in goal md)."""
    Gs, Ls, Hs, Us = [], [], [], []
    for g in ("0.1", "0.5", "1.0"):
        d = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", f"windows_g{g}.pt"))
        idx = torch.randperm(d["grid"].shape[0], generator=torch.Generator().manual_seed(0))[:n // 3]
        Gs.append(d["grid"][idx]); Ls.append(d["low5"][idx]); Hs.append(d["hist"][idx]); Us.append(d["U"][idx])
    G, L, Hh, U = (torch.cat(x).to(DEV) for x in (Gs, Ls, Hs, Us))
    with torch.no_grad():
        return pol.phi_s(U, pol.ctx_from(G, L, Hh), s=s)


def roll_states(st, U):
    """DI open-loop rollout keeping full state. st[4], U[H,2] → states[H+1,4]."""
    out = [st.copy()]
    s = st.copy()
    for a in U:
        v = s[2:4] + DT * np.asarray(a)
        p = s[:2] + DT * s[2:4] + 0.5 * DT * DT * np.asarray(a)
        s = np.array([p[0], p[1], v[0], v[1]], np.float32)
        out.append(s.copy())
    return np.stack(out)


def seg_verdict(seg_xy, env, gamma, goal):
    """(ok, tag) for one 1-s branch segment: ✗G goal-seeking, ✗T taskspace, ✗S SOCP."""
    D = np.linalg.norm(seg_xy - goal[None], axis=1)
    if not GM2.approach_ok(D):
        return False, "✗G"
    if not GM.in_taskspace(seg_xy):
        return False, "✗T"
    if not GM.socp_ok(seg_xy, env, gamma):
        return False, "✗S"
    return True, ""


def build_tree(pol, env, gamma, temp, beta, unc, tilt, seed=0, N=64, T=250, max_alive=128):
    """TRUE RECURSIVE TREE (user 2026-07-05: 'if you make a branch then rollout separately'). BFS by 1-s levels:
    every surviving branch spawns k=SCHED[level] children (importance-resampled p∝exp(σ/β) from N temp-candidates),
    each child rolls its window and lives or dies by the validity verdict (dead ⇒ its subtree never exists).
    Frontier capped at max_alive (fair subsample; raw product would be 2880 by 7 s)."""
    goal = env.goal.detach().cpu().numpy()
    obs = env.obstacles.detach().cpu().numpy()
    rng = np.random.RandomState(1000 + seed)
    torch.manual_seed(seed)
    frontier = [np.zeros(4, np.float32)]
    branch_pts = []
    segs = []                                        # (seg[11,4], sigma, ok, tag, level, reached)
    n_reached = 0
    for j in range(T // H):
        if not frontier:
            break
        k = SCHED[j] if j < len(SCHED) else 1
        # SPLIT BUDGET (user 2026-07-05: branching must be visible at EVERY 1s level): as many branches as
        # capacity allows split with the FULL schedule k; the rest continue single-file. Deaths free slots.
        budget = max(0, max_alive - len(frontier))
        n_split = len(frontier) if k <= 1 else min(len(frontier), max(1, budget // max(1, k - 1)))
        order_idx = rng.permutation(len(frontier))
        new_frontier = []
        for oi, fi in enumerate(order_idx):
            st = frontier[fi]
            k_i = k if (k > 1 and oi < n_split) else 1
            grid = GF.axis_grid(st[:2], obs, float(env.r_robot))
            low5 = GF.low5(st, goal, gamma)
            hist = GF.hist_pad(np.zeros((0, 2)), GF.K_HIST)
            gt, lt, ht = (torch.tensor(np.asarray(x), device=DEV) for x in (grid, low5, hist))
            with torch.no_grad():
                Uc = pol.sample_window(gt, lt, ht, n=N, temp=temp, nfe=8)
                sig_t = unc.sigma(pol.phi_s_at(Uc, gt, lt, ht, s=0.9))
            sig = (sig_t.detach().cpu().numpy() if torch.is_tensor(sig_t) else np.asarray(sig_t)).reshape(-1)
            Uc = Uc.detach().cpu().numpy()
            if tilt:
                w = np.exp((sig - sig.max()) / beta); w /= w.sum()
                picks = rng.choice(N, size=min(k_i, N), replace=False, p=w)
            else:
                picks = rng.choice(N, size=min(k_i, N), replace=False)
            if len(picks) > 1:
                branch_pts.append(st[:2].copy())     # a real branching event (k>=2)
            for i in picks:
                seg = roll_states(st, Uc[i])
                ok, tag = seg_verdict(seg[:, :2], env, gamma, goal)
                hit = ok and np.linalg.norm(seg[-1, :2] - goal) < 0.5
                segs.append((seg, float(sig[i]), ok, tag, j, hit))
                if hit:
                    n_reached += 1
                elif ok:
                    new_frontier.append(seg[-1])
        frontier = new_frontier                      # no subsample-drop: every drawn branch continues
    return dict(segs=segs, n_reached=n_reached, alive_end=len(frontier), branch_pts=branch_pts)


def draw_tree(ax, tree, env, smin, smax):
    obs = env.obstacles.detach().cpu().numpy()
    goal = env.goal.detach().cpu().numpy()
    nrm = Normalize(smin, max(smax, smin + 1e-9))
    for (ox, oy, r) in obs:
        ax.add_patch(Circle((ox, oy), r, facecolor="#d9c8e3", edgecolor="#9b72aa", lw=.4, alpha=.7))
    lab_used = set()
    for (seg, sg, ok, tag, lvl, hit) in tree["segs"]:
        al = max(0.22, 1.0 * (0.86 ** lvl))              # first branches opaque, deeper ever more transparent
        lw = max(0.7, 2.4 * (0.82 ** lvl))
        if ok:
            ax.plot(seg[:, 0], seg[:, 1], "-", color=plt.cm.viridis(nrm(sg)), lw=lw, alpha=al)
            if hit:
                ax.plot(seg[-1, 0], seg[-1, 1], "o", color="#2ca02c", ms=4.5, mec="k", mew=.4, zorder=8)
        else:
            c = FCOL.get(tag, "#d62728")
            ax.plot(seg[:, 0], seg[:, 1], "-", color=c, lw=max(1.0, lw), alpha=min(1.0, al + .15))
            ax.plot(seg[-1, 0], seg[-1, 1], "x", color=c, ms=5, mew=1.6, zorder=7)
            ax.annotate(tag[1], (seg[-1, 0], seg[-1, 1]), fontsize=6.5, color=c,
                        xytext=(2, 2), textcoords="offset points", fontweight="bold")
    bp = np.array(tree["branch_pts"]) if tree["branch_pts"] else np.zeros((0, 2))
    if len(bp):
        ax.plot(bp[:, 0], bp[:, 1], "o", color="k", ms=3.2, ls="", zorder=6)   # true branching events only
    ax.scatter([0], [0], marker="s", s=55, c="#333", zorder=7)
    ax.scatter([goal[0]], [goal[1]], marker="*", s=200, c="gold", edgecolor="k", zorder=9)   # THE goal (once)
    ax.legend(handles=[
        Line2D([], [], color="#ff7f0e", lw=2, label="G: goal-seeking fail"),
        Line2D([], [], color="#9467bd", lw=2, label="T: taskspace fail"),
        Line2D([], [], color="#d62728", lw=2, label="S: SOCP fail"),
        Line2D([], [], color="#2ca02c", marker="o", ls="", label="reached goal"),
    ], loc="lower right", fontsize=7.5, framealpha=.9)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    return nrm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--temp", type=float, default=1.3)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--ell", type=float, default=0.5)   # calibrated ell* (hp_ell_calib: 0.2 gives sigma≡1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="tree")
    ap.add_argument("--outdir", default=None, help="override output dir (default figures/hp_test)")
    ap.add_argument("--ncols", type=int, default=0, help="grid columns (0=single row; 3 -> 2x3 for 6 trees)")
    a = ap.parse_args()
    if a.outdir:
        globals()["FIG"] = a.outdir
        os.makedirs(a.outdir, exist_ok=True)
    env = GS.make_grid()
    rows = []
    for ri, (cp, lb) in enumerate(zip(a.ckpts, a.labels)):
        pol = load_any(cp)
        unc = GPUncertainty(kernel="rbf", lengthscale=a.ell, lam=1e-2, normalize=True)
        unc.set_buffer(dataset_buffer(pol))
        tree = build_tree(pol, env, a.gamma, a.temp, a.beta, unc, tilt=(ri > 0), seed=a.seed)
        rows.append((lb, tree))
        nf = sum(1 for b in tree["segs"] if not b[2])
        print(f"[{lb}] branches {len(tree['segs'])} failed {nf} reached-goal {tree['n_reached']} "
              f"alive-at-end {tree['alive_end']}", flush=True)
    sigs = [b[1] for _, t in rows for b in t["segs"]]
    smin, smax = min(sigs), max(sigs)
    nc = a.ncols if a.ncols and a.ncols > 0 else len(rows)     # --ncols grid (e.g. 3 -> 2x3 for 6 trees, legible)
    nr = int(np.ceil(len(rows) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(7.2 * nc, 7.2 * nr))
    axes = np.atleast_1d(axes).ravel()
    for ax, (lb, tree) in zip(axes, rows):
        nrm = draw_tree(ax, tree, env, smin, smax)
        nf = sum(1 for b in tree["segs"] if not b[2])
        ax.set_title(f"{lb} — {len(tree['segs'])} branches · {nf} died · {tree['n_reached']} reached goal",
                     fontsize=12, color="#2ca02c" if tree["n_reached"] else "#d62728")
    for ax in axes[len(rows):]:
        ax.axis("off")
    fig.colorbar(ScalarMappable(norm=Normalize(smin, smax), cmap="viridis"), ax=axes.tolist(), fraction=.02, label="σ")
    fig.suptitle(f"SAFE-EXPANSION TREE (recursive) — γ{a.gamma}, temp {a.temp}, β {a.beta}, ell {a.ell} · every branch rolls out "
                 "separately, k=[5,4,4,3,3,2,2,1,…]/branch (throttled, never orphaned) · deaths color-coded G/T/S · dots=branch events",
                 fontsize=12)
    out = os.path.join(FIG, f"tree_{a.tag}.png")
    fig.savefig(out, dpi=125, bbox_inches="tight")
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
