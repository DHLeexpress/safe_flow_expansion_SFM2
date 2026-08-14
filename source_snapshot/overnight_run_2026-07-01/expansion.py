"""STAGE 3b — windowed VERIFIER-FILTERED safe expansion (Pillar 5) + before/after viz + comparison.

Loop (per the spec):
    pretrain on MPPI windows   (stage3_pretrain)
    for each round:
        sample H_pred windows from the FM (closed-loop rollout, exploratory temp) + a broad proposal
        roll out with dynamics, VERIFY (collision ∧ goal ∧ SOCP-certified) per γ
        keep only certified-safe positives (their windows)
        finetune the FM on  MPPI demos  ∪  verified positives
Coverage/validity are the swappable `coverage`/`validity` modules; Ω* = broad proposal gated by the verifier.
Explicit ACTFLOW note: this is the finite-β / broad-proposal instantiation of Eq.9 (the broad proposal is the
importance support q may deviate into); we omit the σ-GP (Eq.10) tilt here (it was near-uniform in prior runs)
and keep the honest broad-proposal + verifier-filter as the exploration engine.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import _paths
import config as C
from cfm_mppi.data.windowed_dataset import WindowedDataset
from windowed_policy import GridLowFlowPolicy, fm_rollout, windows_of
import validity as VAL
import coverage as COV
import verifier_polytope as VP
import di_grid_viz as DV
import wandb_utils as W


def _load(scene, which, device):
    ck = torch.load(C.scene_result(scene, f"{which}.pt"), weights_only=False)
    pol = GridLowFlowPolicy(H_pred=ck["H_pred"], u_max=ck["u_max"]).to(device)
    pol.load_state_dict(ck["state_dict"])
    pol.eval()
    return pol


def load_pretrained(scene, device):
    return _load(scene, "pretrained", device)


def finetune(policy, demo, pos, steps, batch, lr, device):
    """Finetune on MPPI demos ∪ verified positives, with MODE-BALANCED replay of the positives so rare
    discovered modes (e.g. weave) are not forgotten."""
    from collections import Counter
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    dG, dL, dU = demo.grid.to(device), demo.low_dim.to(device), demo.U_local.to(device)
    nD = dG.shape[0]
    pG = pL = pU = w_t = None
    if pos["grid"] and len(pos["grid"]) >= 8:
        pG = torch.tensor(np.array(pos["grid"]), device=device).float()
        pL = torch.tensor(np.array(pos["low"]), device=device).float()
        pU = torch.tensor(np.array(pos["U"]), device=device).float()
        cnt = Counter(pos["mode"])
        w = np.array([1.0 / cnt[m] for m in pos["mode"]], dtype=np.float64)
        w_t = torch.tensor(w / w.sum(), device=device)
    policy.train()
    last = 0.0
    for _ in range(steps):
        bi = torch.randint(0, nD, (batch,), device=device)
        grids, lows, Us = dG[bi], dL[bi], dU[bi]
        if pG is not None:
            pi = torch.multinomial(w_t, min(batch, pG.shape[0]), replacement=True)   # inverse-freq mode balance
            grids = torch.cat([grids, pG[pi]], 0); lows = torch.cat([lows, pL[pi]], 0); Us = torch.cat([Us, pU[pi]], 0)
        ctx = policy.ctx_from(grids, lows)
        loss = policy.cfm_loss(Us, ctx) + 0.3 * policy.aux_safety_loss(grids)   # keep safety-encoding intact
        opt.zero_grad(); loss.backward(); opt.step()
        last = float(loss.detach())
    policy.eval()
    return last


def collect_fm_positives(policy, env, gamma, n_traj, temp, device):
    paths, (G, L, U) = fm_rollout(policy, env, gamma, n_traj=n_traj, temp=temp, device=device, record=True)
    pg, pl, pu, pm = [], [], [], []
    for i in range(len(paths)):
        if VAL.is_valid(paths[i], env, gamma):
            m = COV.mode_of(paths[i], env)
            pg += G[i]; pl += L[i]; pu += U[i]; pm += [m] * len(G[i])
    return paths, pg, pl, pu, pm


def collect_broad_positives(env, gamma, n, seed, H_pred, device):
    S, Uc = COV.broad_rollouts(env, n, seed=seed)
    pg, pl, pu, pm = [], [], [], []
    for i in range(len(S)):
        if VAL.is_valid(S[i, :, :2], env, gamma):
            g, l, u = windows_of(S[i], Uc[i], env, gamma, H_pred, device=device)
            m = COV.mode_of(S[i, :, :2], env)
            pg += g; pl += l; pu += u; pm += [m] * len(g)
    return pg, pl, pu, pm


@torch.no_grad()
def eval_all_gamma(policy, env, star, gammas, n_eval, device):
    out = {}
    for g in gammas:
        paths, _ = fm_rollout(policy, env, g, n_traj=n_eval, temp=1.0, device=device, record=False)
        out[g] = COV.evaluate(paths, env, g, star)
    return out


# --------------------------------------------------------------------- viz
@torch.no_grad()
def plot_before_after(pre, post, env, gammas, star, out, n=40, device="cpu"):
    fig, axes = plt.subplots(2, len(gammas), figsize=(4.3 * len(gammas), 8.0), squeeze=False)
    obs = env.obstacles.detach().cpu().numpy()
    for row, (pol, tag) in enumerate([(pre, "pretrained"), (post, "after expansion")]):
        for ci, g in enumerate(gammas):
            ax = axes[row][ci]
            for (ox, oy, rr) in obs:
                ax.add_patch(Circle((ox, oy), rr, facecolor="#c8a2c8", edgecolor="#7b3294", lw=0.7, alpha=0.7, zorder=3))
            paths, _ = fm_rollout(pol, env, g, n_traj=n, temp=1.0, device=device, record=False)
            for p in paths:
                ok = VAL.is_valid(p, env, g)
                ax.plot(p[:, 0], p[:, 1], "-", color=("#2ca02c" if ok else "0.7"),
                        lw=0.7, alpha=(0.5 if ok else 0.2), zorder=(5 if ok else 4))
            ax.scatter([env.x0[0]], [env.x0[1]], s=40, c="#00a000", edgecolor="k", zorder=6)
            ax.scatter([env.goal[0]], [env.goal[1]], marker="*", s=130, c="gold", edgecolor="k", zorder=6)
            ax.set_xlim(*env.xlim); ax.set_ylim(*env.ylim); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f"γ={g}", fontsize=11)
            if ci == 0:
                ax.set_ylabel(tag, fontsize=11)
    fig.suptitle(f"Windowed FM [{env.name}] — green=verifier-valid. Pretrained → after safe expansion", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


@torch.no_grad()
def plot_comparison(post, env, gammas, out, n_fm=40, n_mppi=30, device="cpu"):
    """1×2: SafeMPPI total generated trajectories vs FM (expanded) total generated trajectories."""
    cfg = DV.load_best_config()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2))
    obs = env.obstacles.detach().cpu().numpy()
    # left: SafeMPPI
    for (ox, oy, rr) in obs:
        axes[0].add_patch(Circle((ox, oy), rr, facecolor="#c8a2c8", edgecolor="#7b3294", lw=0.7, alpha=0.7, zorder=3))
    for g in gammas:
        for seed in range(n_mppi):
            _, path = DV.mppi_rollout(env, g, cfg, seed_base=seed * 131 + 1)
            axes[0].plot(path[:, 0], path[:, 1], "-", color="#08519c", lw=0.5, alpha=0.25, zorder=4)
    axes[0].set_title(f"SafeMPPI expert — {n_mppi}×{len(gammas)} rollouts", fontsize=11)
    # right: FM expanded
    for (ox, oy, rr) in obs:
        axes[1].add_patch(Circle((ox, oy), rr, facecolor="#c8a2c8", edgecolor="#7b3294", lw=0.7, alpha=0.7, zorder=3))
    for g in gammas:
        paths, _ = fm_rollout(post, env, g, n_traj=n_fm, temp=1.0, device=device, record=False)
        for p in paths:
            axes[1].plot(p[:, 0], p[:, 1], "-", color="#2ca02c", lw=0.5, alpha=0.3, zorder=4)
    axes[1].set_title(f"Expanded windowed FM — {n_fm}×{len(gammas)} rollouts", fontsize=11)
    for ax in axes:
        ax.scatter([env.x0[0]], [env.x0[1]], s=45, c="#00a000", edgecolor="k", zorder=6)
        ax.scatter([env.goal[0]], [env.goal[1]], marker="*", s=140, c="gold", edgecolor="k", zorder=6)
        ax.set_xlim(*env.xlim); ax.set_ylim(*env.ylim); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"[{env.name}] total generated trajectories — SafeMPPI vs expanded FM", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="gap", choices=C.SCENE_NAMES)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--init", default="pretrained", choices=["pretrained", "expanded"])
    ap.add_argument("--explore-temp", type=float, default=1.2)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    W.add_wandb_args(ap)
    args = ap.parse_args()
    device = args.device
    n_traj, n_broad, n_eval, ft_steps = (24, 200, 32, 150)
    if args.smoke:
        args.rounds, n_traj, n_broad, n_eval, ft_steps = 3, 8, 60, 16, 40

    env = C.make_scene(args.scene)
    demo = WindowedDataset(os.path.join(C.dataset_dir(args.scene), "train.pt"))
    pre = load_pretrained(args.scene, device)                       # the seed, always used for before/after
    if args.init == "expanded" and os.path.exists(C.scene_result(args.scene, "expanded.pt")):
        policy = _load(args.scene, "expanded", device)
        print("[init] continuing from expanded policy", flush=True)
    else:
        policy = load_pretrained(args.scene, device)
    gmax = C.VERIFIER["gamma_max"]
    print(f"=== STAGE 3b [{args.scene}] safe expansion: rounds={args.rounds} demos={len(demo)} ===", flush=True)
    run = W.init_run(args, name=f"{args.scene}-expand", dir=C.RESULTS, group=args.scene,
                     config={"stage": "expand", "scene": args.scene, "rounds": args.rounds,
                             "n_traj": n_traj, "n_broad": n_broad, "gammas": C.GAMMAS})

    star = COV.build_omega_star(env, gmax, n=(400 if args.smoke else 1500))
    pos = {"grid": [], "low": [], "U": [], "mode": []}
    for g in C.GAMMAS:                                          # seed D_0 with verified broad windows
        bg, bl, bu, bm = collect_broad_positives(env, g, n_broad, seed=int(g * 7), H_pred=C.H_PRED, device=device)
        pos["grid"] += bg; pos["low"] += bl; pos["U"] += bu; pos["mode"] += bm

    m0 = eval_all_gamma(policy, env, star, C.GAMMAS, n_eval, device)
    for g in C.GAMMAS:
        W.log(run, {f"cov/g{g}": m0[g]["spatial_coverage"], f"val/g{g}": m0[g]["validity"],
                    f"modecov/g{g}": m0[g]["mode_coverage"], f"vendi/g{g}": m0[g]["vendi"]}, step=0)
    print(f"[pretrained] " + " ".join(f"g{g}:cov{m0[g]['spatial_coverage']:.2f}/val{m0[g]['validity']:.2f}/"
          f"mode{m0[g]['mode_coverage']:.2f}" for g in C.GAMMAS), flush=True)

    history = [{"round": 0, **{f"cov_g{g}": m0[g]["spatial_coverage"] for g in C.GAMMAS}}]
    best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
    best_score, best_round, best_m = -1.0, 0, m0                    # peak = max mean(mode+spatial) over rounds
    t0 = time.time()
    for r in range(1, args.rounds + 1):
        for g in C.GAMMAS:
            _, pg, pl, pu, pm = collect_fm_positives(policy, env, g, n_traj, temp=args.explore_temp, device=device)
            pos["grid"] += pg; pos["low"] += pl; pos["U"] += pu; pos["mode"] += pm
            bg, bl, bu, bm = collect_broad_positives(env, g, n_broad // 2, seed=r * 100 + int(g * 10),
                                                     H_pred=C.H_PRED, device=device)
            pos["grid"] += bg; pos["low"] += bl; pos["U"] += bu; pos["mode"] += bm
        ft_loss = finetune(policy, demo, pos, ft_steps, batch=128, lr=2e-4, device=device)
        m = eval_all_gamma(policy, env, star, C.GAMMAS, n_eval, device)
        rec = {"round": r, "ft_loss": ft_loss, "n_pos": len(pos["grid"])}
        for g in C.GAMMAS:
            rec[f"cov_g{g}"] = m[g]["spatial_coverage"]
            W.log(run, {f"cov/g{g}": m[g]["spatial_coverage"], f"val/g{g}": m[g]["validity"],
                        f"modecov/g{g}": m[g]["mode_coverage"], f"vendi/g{g}": m[g]["vendi"],
                        "expand/ft_loss": ft_loss, "expand/n_pos": len(pos["grid"])}, step=r)
        history.append(rec)
        score = float(np.mean([m[g]["mode_coverage"] + m[g]["spatial_coverage"] for g in C.GAMMAS]))
        if score > best_score:
            best_score, best_round, best_m = score, r, m
            best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
        print(f"[r{r}] " + " ".join(f"g{g}:cov{m[g]['spatial_coverage']:.2f}/mode{m[g]['mode_coverage']:.2f}"
              f"({','.join(m[g]['modes'])})" for g in C.GAMMAS) + f"  npos={len(pos['grid'])} ({time.time()-t0:.0f}s)",
              flush=True)

    policy.load_state_dict(best_state)                             # deploy the PEAK round, not the fluctuating final
    m = best_m
    print(f"[best] deploying round {best_round}/{args.rounds} (score {best_score:.3f})", flush=True)
    torch.save({"state_dict": policy.state_dict(), "H_pred": C.H_PRED, "u_max": float(env.u_max),
                "scene": args.scene, "history": history, "m0": m0, "final": m, "best_round": best_round},
               C.scene_result(args.scene, "expanded.pt"))
    ba = C.scene_fig(args.scene, "stage3_before_after.png")
    plot_before_after(pre, policy, env, C.GAMMAS, star, ba, device=device)
    cmp = C.scene_fig(args.scene, "stage3_comparison.png")
    plot_comparison(policy, env, C.GAMMAS, cmp, device=device)
    W.log_image(run, "before_after", ba)
    W.log_image(run, "comparison", cmp)
    W.finish(run, summary={f"final_cov_g{g}": m[g]["spatial_coverage"] for g in C.GAMMAS})
    with open(C.scene_result(args.scene, "expand_history.json"), "w") as f:
        json.dump({"m0": {str(g): m0[g] for g in C.GAMMAS}, "final": {str(g): m[g] for g in C.GAMMAS},
                   "history": history}, f, indent=2, default=float)
    print(f"[{args.scene}] expansion done → {ba}, {cmp}", flush=True)


if __name__ == "__main__":
    main()
