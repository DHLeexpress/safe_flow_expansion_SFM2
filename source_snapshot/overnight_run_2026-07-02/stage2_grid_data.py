"""Stage A — SafeMPPI grid data generation for the FM policy.

For each γ∈{0.1,0.5,1.0}: run the mode-1 (Gaussian, u_max=1, tripled-variance) SafeMPPI expert over many
seeds on the walled 5x5 grid, keep only SUCCESSFUL trajectories (reach ∧ collision-free ∧ on-grid), and slice
each into per-step windowed records for the GRU+CNN FM policy:
  grid[3,16,12] (axis-aligned polytope grid) · low5 (relgoal,vel,γ) · hist[K,2] (past executed controls) ·
  U[H,2] (target world control window, raw).
Saved to dataset/windows_g<γ>.pt (revisitable — rerun with more --seeds to append).
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

import _paths  # noqa: F401
import grid_scene as GS
import grid_feats as GF
from di_grid_viz import di_step
from cfm_mppi.safegpc_adapter.safemppi import SafeMPPIAdapter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "dataset"); os.makedirs(DATA, exist_ok=True)


def rollout_full(env, gamma, cfg, seed, reach=0.4):
    """Receding-horizon SafeMPPI rollout -> (states [n+1,4], controls [n,2]) (stops at reach)."""
    ad = SafeMPPIAdapter(**cfg)
    st = env.x0.detach().cpu().numpy().astype(np.float32)
    goal_t = env.goal.detach().cpu().float()
    obs_plan = GS.planner_obstacles(env)                      # inflated (r_robot + margin)
    goal = env.goal.detach().cpu().numpy()
    states, controls = [st.copy()], []
    for t in range(env.T):
        a, _ = ad.plan(torch.tensor(st, dtype=torch.float32), goal_t, obs_plan, gamma=gamma, seed=seed * 1000 + t)
        a = a.detach().cpu().numpy().astype(np.float32)
        st = di_step(st, a, dt=env.dt)
        states.append(st.copy()); controls.append(a)
        if np.linalg.norm(st[:2] - goal) < reach:
            break
    return np.array(states, np.float32), np.array(controls, np.float32)


def windows_from(states, controls, env, gamma, K=GF.K_HIST, H=GF.H_PRED):
    """Slice a trajectory into per-step (grid, low5, hist, U) records."""
    obs = env.obstacles.detach().cpu().numpy(); rr = float(env.r_robot); goal = env.goal.detach().cpu().numpy()
    n = len(controls); G, L, Hh, U = [], [], [], []
    for t in range(n):
        G.append(GF.axis_grid(states[t, :2], obs, rr))
        L.append(GF.low5(states[t], goal, gamma))
        Hh.append(GF.hist_pad(controls[max(0, t - K):t], K))
        u = controls[t:t + H]
        if len(u) < H:
            u = np.concatenate([u, np.repeat(u[-1:], H - len(u), 0)], 0)
        U.append(u.astype(np.float32))
    return G, L, Hh, U


def generate(gamma, seeds, env, cfg, log=print):
    G, L, Hh, U = [], [], [], []
    n_ok = 0; t0 = time.time()
    for s in range(seeds):
        states, controls = rollout_full(env, gamma, cfg, s)
        ok, _ = GS.is_success(states[:, :2], env)
        if not ok or len(controls) < 2:
            continue
        n_ok += 1
        g, l, h, u = windows_from(states, controls, env, gamma)
        G += g; L += l; Hh += h; U += u
        if (s + 1) % 25 == 0:
            log(f"  γ{gamma}: {s+1}/{seeds} seeds, {n_ok} success, {len(G)} windows, {(time.time()-t0)/(s+1):.2f}s/seed")
    return dict(grid=torch.tensor(np.array(G)), low5=torch.tensor(np.array(L)),
                hist=torch.tensor(np.array(Hh)), U=torch.tensor(np.array(U)),
                gamma=float(gamma), n_traj=n_ok, n_seeds=seeds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=150)
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.1, 0.5, 1.0])
    ap.add_argument("--append", action="store_true", help="append to existing shard (seed offset)")
    args = ap.parse_args()
    env = GS.make_grid(); cfg = GS.mode1_config()
    print(f"=== Stage A data: {len(env.obstacles)} obstacles, u_max={cfg['u_max']}, "
          f"noise={[round(x,3) for x in cfg['noise_sigma']]}, K={GF.K_HIST}, H={GF.H_PRED} ===", flush=True)
    for g in args.gammas:
        d = generate(g, args.seeds, env, cfg)
        out = os.path.join(DATA, f"windows_g{g}.pt")
        if args.append and os.path.exists(out):
            old = torch.load(out)
            for k in ("grid", "low5", "hist", "U"):
                d[k] = torch.cat([old[k], d[k]], 0)
            d["n_traj"] += old.get("n_traj", 0); d["n_seeds"] += old.get("n_seeds", 0)
        torch.save(d, out)
        print(f"γ{g}: saved {d['grid'].shape[0]} windows from {d['n_traj']} success trajectories -> {out}", flush=True)


if __name__ == "__main__":
    main()
