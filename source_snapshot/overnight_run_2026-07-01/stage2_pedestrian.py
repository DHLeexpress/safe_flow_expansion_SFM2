"""HEAVY Phase-4 data engine — windowed SafeMPPI dataset on REAL moving-pedestrian crowds (UCY / SDD).

Same windowed schema as the toy scenes (polar polytope-occupancy grid [3,16,12] + goal-aligned low_dim[7]
→ MPPI planned window U_local[H_pred,2]) but on time-varying pedestrian scenes loaded via
`cfm_mppi.mppi.sweep._load` (obstacle_velocities passed so the DTCBF rejection sees the moving crowd).
NO verifier here — this is the expensive expert-data generation; expansion/verify on crowds comes later.
Point ego (r_robot=0) among r=0.5 pedestrian disks; keep goal-reaching, collision-free episodes.

    python stage2_pedestrian.py --datasets ucy sdd --episodes 150
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

import _paths
import config as C
from cfm_mppi.mppi import sweep
from cfm_mppi.safegpc_adapter.safemppi import SafeMPPIAdapter
from di_grid_viz import load_best_config, di_step
from polar_grid import polar_grid
from local_frame import low_dim_features, goal_frame, to_local
from cfm_mppi.data.windowed_dataset import save_windowed_splits
import wandb_utils as W

PED_R_ROBOT = 0.0        # point ego among r=0.5 pedestrian disks


def _active(ob, vl):
    ok = ~np.isnan(np.asarray(ob)[:, :2]).any(1)
    return np.asarray(ob)[ok], np.asarray(vl)[ok]


def run_ped_episode(dataset, ep, gamma, cfg, steps, reach_thresh=0.6):
    s0, goal, obs_seq, vel_seq = sweep._load(dataset, ep, steps)
    ad = SafeMPPIAdapter(**cfg)
    st = np.asarray(s0, np.float32).copy()
    goal = np.asarray(goal, float)
    states, uwins, execs, obs_at = [st.copy()], [], [], []
    T = min(steps, len(obs_seq) - 1)
    reached = False
    H = int(cfg["horizon"])
    for t in range(T):
        ob, vl = _active(obs_seq[min(t, len(obs_seq) - 1)], vel_seq[min(t, len(vel_seq) - 1)])
        if len(ob) == 0:
            a, _ = ad.plan(torch.tensor(st), torch.tensor(goal, dtype=torch.float32),
                           torch.zeros(0, 3), gamma=gamma, seed=ep * 1000 + t)
        else:
            a, _ = ad.plan(torch.tensor(st), torch.tensor(goal, dtype=torch.float32),
                           torch.tensor(ob, dtype=torch.float32), gamma=gamma,
                           obstacle_velocities=torch.tensor(vl, dtype=torch.float32), seed=ep * 1000 + t)
        uw = ad._u_prev.detach().cpu().numpy().astype(np.float32) if getattr(ad, "_u_prev", None) is not None \
            else np.zeros((H, 2), np.float32)
        uwins.append(uw); execs.append(a.detach().cpu().numpy().astype(np.float32)); obs_at.append(ob.copy())
        st = di_step(st, execs[-1], dt=cfg["dt"])
        states.append(st.copy())
        if np.linalg.norm(st[:2] - goal) < reach_thresh:
            reached = True
            break
    states = np.array(states, np.float32)
    # min clearance vs the active peds along the executed path
    minclr = np.inf
    for t, ob in enumerate(obs_at):
        if len(ob):
            d = np.linalg.norm(states[t, :2][None] - ob[:, :2], axis=1) - ob[:, 2] - PED_R_ROBOT
            minclr = min(minclr, float(d.min()))
    return states, np.array(uwins), np.array(execs), obs_at, goal, reached, (minclr if np.isfinite(minclr) else 9.0)


def featurize(states, uwins, execs, obs_at, goal, gamma, H_pred):
    grids, lows, ulocs = [], [], []
    for t in range(len(uwins)):
        pos = states[t, :2]
        ob = obs_at[t] if t < len(obs_at) else np.zeros((0, 3))
        a_prev = execs[t - 1] if t > 0 else None
        grid, _ = polar_grid(pos, goal, ob if len(ob) else np.zeros((0, 3)), r_robot=PED_R_ROBOT)
        low, _ = low_dim_features(states[t], goal, gamma, a_prev=a_prev, prev_valid=(t > 0))
        e_g, e_lat, _ = goal_frame(pos, goal)
        grids.append(grid); lows.append(low)
        ulocs.append(to_local(uwins[t], e_g, e_lat).astype(np.float32))
    return np.array(grids), np.array(lows), np.array(ulocs)


def build(dataset, n_ep, gammas, cfg, log=print):
    G, L, U, GA = [], [], [], []
    t0, n_try, n_keep = time.time(), 0, 0
    for ep in range(n_ep):
        for g in gammas:
            n_try += 1
            try:
                states, uwins, execs, obs_at, goal, reached, minclr = run_ped_episode(dataset, ep, g, cfg, 80)
            except Exception as exc:
                log(f"  [{dataset} ep{ep} g{g}] skipped ({exc})")
                continue
            if not (reached and minclr >= 0.0 and len(uwins) >= 5):
                continue
            grids, lows, ulocs = featurize(states, uwins, execs, obs_at, goal, g, cfg["horizon"])
            G.append(grids); L.append(lows); U.append(ulocs); GA.append(np.full(len(ulocs), g, np.float32))
            n_keep += 1
        if ep % 20 == 0:
            nw = sum(len(x) for x in U) if U else 0
            log(f"  {dataset} ep {ep}/{n_ep}: kept {n_keep}/{n_try} eps, {nw} windows ({time.time()-t0:.0f}s)", )
    if not U:
        raise RuntimeError(f"{dataset}: no valid episodes")
    grid = torch.tensor(np.concatenate(G)); low = torch.tensor(np.concatenate(L))
    U_local = torch.tensor(np.concatenate(U)); gamma = torch.tensor(np.concatenate(GA))
    return grid, low, U_local, gamma, {"n_try": n_try, "n_keep": n_keep, "secs": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ucy", "sdd"])
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--smoke", action="store_true")
    W.add_wandb_args(ap)
    args = ap.parse_args()
    n_ep = 5 if args.smoke else args.episodes
    cfg = load_best_config()
    for dataset in args.datasets:
        print(f"\n=== HEAVY PEDESTRIAN dataset [{dataset}]: {n_ep} eps × γ{C.GAMMAS} (moving crowd) ===", flush=True)
        run = W.init_run(args, name=f"pedestrian-{dataset}-{n_ep}", dir=C.RESULTS, group="pedestrian",
                         config={"stage": "pedestrian_dataset", "dataset": dataset, "episodes": n_ep,
                                 "gammas": C.GAMMAS, "H_pred": C.H_PRED})
        grid, low, U_local, gamma, stats = build(dataset, n_ep, C.GAMMAS, cfg)
        meta = {"dataset": dataset, "gammas": C.GAMMAS, "H_pred": C.H_PRED, "r_robot": PED_R_ROBOT,
                "source": "moving_pedestrian", **stats}
        ddir = os.path.join(C.ROOT, "dataset", f"windowed_pedestrian_{dataset}")
        split = save_windowed_splits(ddir, grid, low, gamma, U_local, meta=meta)
        print(f"[{dataset}] windows {grid.shape[0]} | grid {tuple(grid.shape)} | kept {stats['n_keep']}/{stats['n_try']} "
              f"| {split} | {stats['secs']:.0f}s → {ddir}", flush=True)
        W.log(run, {"n_windows": int(grid.shape[0]), "n_keep": stats["n_keep"], "keep_rate": stats["n_keep"] / max(stats["n_try"], 1)})
        W.finish(run, summary={"n_windows": int(grid.shape[0]), **split})
    print("\n=== PEDESTRIAN generation complete ===", flush=True)


if __name__ == "__main__":
    main()
