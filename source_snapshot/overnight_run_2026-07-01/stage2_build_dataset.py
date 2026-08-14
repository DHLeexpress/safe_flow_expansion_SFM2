"""STAGE 2 — build the windowed MPPI-expert dataset (100 × 3 episodes) + per-γ plots.

For γ∈{0.1,0.5,1.0}, run N SafeMPPI episodes (seeds) on the locked narrow gap; keep goal-reaching,
collision-free episodes; per executed step t emit (polar_grid[3,16,12], low_dim[7], γ) → U_local[H_pred,2].
Saves dataset/windowed_narrow_gap/{train,val,test}.pt and per-γ trajectory-overlay plots. Logs to W&B.

    python stage2_build_dataset.py --episodes 100        # 100 × 3
    python stage2_build_dataset.py --smoke               # 6 × 3 quick
"""
from __future__ import annotations

import argparse
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
import scenes
from di_grid_viz import load_best_config, di_step
from polar_grid import polar_grid
from local_frame import low_dim_features, goal_frame, to_local
from cfm_mppi.safegpc_adapter.safemppi import SafeMPPIAdapter
from cfm_mppi.data.windowed_dataset import save_windowed_splits
import wandb_utils as W


def run_episode(env, gamma, seed, cfg, reach_thresh=0.4):
    ad = SafeMPPIAdapter(**cfg)
    st = env.x0.detach().cpu().numpy().astype(np.float32)
    goal = env.goal.detach().cpu().numpy()
    goal_t = env.goal.detach().cpu().float()
    obs_t = env.obstacles.detach().cpu().float()
    states, uwins, execs, reached, reach_step = [st.copy()], [], [], False, env.T
    for t in range(env.T):
        a, _ = ad.plan(torch.tensor(st, dtype=torch.float32), goal_t, obs_t, gamma=gamma, seed=seed * 1000 + t)
        uwins.append(ad._u_prev.detach().cpu().numpy().astype(np.float32))    # [H_pred,2] planned window
        execs.append(a.detach().cpu().numpy().astype(np.float32))
        st = di_step(st, execs[-1], dt=env.dt)
        states.append(st.copy())
        if np.linalg.norm(st[:2] - goal) < reach_thresh:
            reached, reach_step = True, t + 1
            break
    states = np.array(states, np.float32)
    obs = env.obstacles.detach().cpu().numpy()
    minclr = float((np.linalg.norm(states[:, None, :2] - obs[None, :, :2], axis=2)
                    - obs[None, :, 2]).min() - float(env.r_robot))
    return states, np.array(uwins), np.array(execs), reached, minclr


def featurize(states, uwins, execs, goal, gamma, obs, r_robot):
    grids, lows, ulocs = [], [], []
    for t in range(len(uwins)):
        pos = states[t, :2]
        a_prev = execs[t - 1] if t > 0 else None
        grid, _ = polar_grid(pos, goal, obs, r_robot=r_robot)
        low, _ = low_dim_features(states[t], goal, gamma, a_prev=a_prev, prev_valid=(t > 0))
        e_g, e_lat, _ = goal_frame(pos, goal)
        ulocs.append(to_local(uwins[t], e_g, e_lat).astype(np.float32))
        grids.append(grid); lows.append(low)
    return np.array(grids), np.array(lows), np.array(ulocs)


def plot_per_gamma(paths_by_gamma, env, out):
    gammas = sorted(paths_by_gamma.keys())
    fig, axes = plt.subplots(1, len(gammas), figsize=(4.2 * len(gammas), 4.0), squeeze=False)
    obs = env.obstacles.detach().cpu().numpy()
    for ci, g in enumerate(gammas):
        ax = axes[0][ci]
        for (ox, oy, rr) in obs:
            ax.add_patch(Circle((ox, oy), rr, facecolor="#c8a2c8", edgecolor="#7b3294", lw=0.6, alpha=0.7))
        for path in paths_by_gamma[g]:
            ax.plot(path[:, 0], path[:, 1], "-", color="#08519c", lw=0.5, alpha=0.25)
        ax.scatter([env.x0[0]], [env.x0[1]], s=50, c="#00a000", edgecolor="k", zorder=5)
        ax.scatter([env.goal[0]], [env.goal[1]], marker="*", s=140, c="gold", edgecolor="k", zorder=5)
        ax.set_xlim(*env.xlim); ax.set_ylim(*env.ylim); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"γ={g}  ({len(paths_by_gamma[g])} eps)", fontsize=11)
    fig.suptitle("SafeMPPI expert rollouts per γ (locked narrow gap)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="gap", choices=C.SCENE_NAMES)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    W.add_wandb_args(ap)
    args = ap.parse_args()
    n_ep = 6 if args.smoke else args.episodes

    cfg = load_best_config()
    env = C.make_scene(args.scene)
    print(f"=== STAGE 2 [{args.scene}]: windowed dataset  {n_ep} eps × γ{C.GAMMAS}  H_pred={C.H_PRED}  "
          f"scene={env.name} obs={env.n_obs} r_robot={float(env.r_robot)} ===", flush=True)
    run = W.init_run(args, name=f"{args.scene}-dataset-{n_ep}x{len(C.GAMMAS)}", dir=C.RESULTS,
                     group=args.scene,
                     config={"stage": "dataset", "scene": args.scene, "episodes": n_ep,
                             "gammas": C.GAMMAS, "H_pred": C.H_PRED, "scene_cfg": C.SCENES[args.scene]})

    G, L, U, GA = [], [], [], []
    paths_by_gamma = {g: [] for g in C.GAMMAS}
    t0, n_try, n_keep = time.time(), 0, 0
    for g in C.GAMMAS:
        kept = 0
        for seed in range(n_ep):
            n_try += 1
            states, uwins, execs, reached, minclr = run_episode(env, g, seed, cfg)
            if not (reached and minclr >= 0.0):
                continue
            grids, lows, ulocs = featurize(states, uwins, execs, env.goal.numpy(), g,
                                           env.obstacles.numpy(), float(env.r_robot))
            G.append(grids); L.append(lows); U.append(ulocs); GA.append(np.full(len(ulocs), g, np.float32))
            paths_by_gamma[g].append(states[:, :2].copy())
            kept += 1; n_keep += 1
        print(f"  γ={g}: kept {kept}/{n_ep} episodes ({n_keep} total, {len(np.concatenate(U)) if U else 0} windows, "
              f"{time.time()-t0:.0f}s)", flush=True)
        W.log(run, {f"kept_eps_g{g}": kept}, step=int(g * 100))

    grid = torch.tensor(np.concatenate(G)); low = torch.tensor(np.concatenate(L))
    U_local = torch.tensor(np.concatenate(U)); gamma = torch.tensor(np.concatenate(GA))
    meta = {"scene_name": args.scene, "scene": C.SCENES[args.scene], "gammas": C.GAMMAS, "H_pred": C.H_PRED,
            "obstacles": env.obstacles.detach().cpu(), "goal": env.goal.detach().cpu(),
            "x0": env.x0.detach().cpu(), "r_robot": float(env.r_robot), "dt": float(env.dt),
            "u_max": float(env.u_max), "T": int(env.T)}
    stats = save_windowed_splits(C.dataset_dir(args.scene), grid, low, gamma, U_local, meta=meta)
    print(f"windows: {grid.shape[0]} total | grid {tuple(grid.shape)} U_local {tuple(U_local.shape)} | {stats}", flush=True)

    fig_path = C.scene_fig(args.scene, "stage2_per_gamma_rollouts.png")
    plot_per_gamma(paths_by_gamma, env, fig_path)
    print("saved", fig_path, flush=True)
    W.log(run, {"n_windows": int(grid.shape[0]), "n_episodes_kept": n_keep, "keep_rate": n_keep / max(n_try, 1)})
    W.log_image(run, "per_gamma_rollouts", fig_path)
    W.finish(run, summary={**stats, "n_windows": int(grid.shape[0])})


if __name__ == "__main__":
    main()
