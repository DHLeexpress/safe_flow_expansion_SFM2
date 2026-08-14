"""PHASE-DR data gen (user 2026-07-06): domain-randomized STARTS, fixed goal (5,5), per-γ SafeMPPI expert.

Purpose: expose the H_P CNN+AAP encoder to off-diagonal obstacle patterns it never sees from the (0,0) start.
Start (x,y) ~ U over the grid interior, rejected if too close to an obstacle or the goal; v0 = 0.
Successes sliced into the SAME window records as stage2 (grid/low5/hist/U) -> dataset/dr_windows_g<γ>.pt.
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
from stage2_grid_data import windows_from

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "dataset"); os.makedirs(DATA, exist_ok=True)


def sample_start(env, rng, goal_clear=1.5, obs_margin=0.15, lo=0.25, hi=4.75, tries=400, offdiag=0.0):
    """offdiag>0 (user 2026-07-06): require |y-x| >= offdiag — starts OFF the diagonal band, so expert
    trajectories cannot just replay the diagonal behavior."""
    obs = env.obstacles.detach().cpu().numpy()
    oc = obs[:, :2] if len(obs) else obs
    orad = obs[:, 2] if len(obs) and obs.shape[1] > 2 else np.full(len(obs), GS.OBS_R)
    goal = env.goal.detach().cpu().numpy()
    for _ in range(tries):
        p = rng.uniform(lo, hi, size=2)
        if offdiag > 0 and abs(p[1] - p[0]) < offdiag:
            continue
        if np.linalg.norm(p - goal) < goal_clear:
            continue
        if len(obs) and (np.linalg.norm(oc - p[None], axis=1) - orad).min() < float(env.r_robot) + obs_margin:
            continue
        return np.array([p[0], p[1], 0.0, 0.0], np.float32)
    return env.x0.detach().cpu().numpy().astype(np.float32)   # fallback: canonical start


def rollout_dr(env, gamma, cfg, seed, reach=0.4, offdiag=0.0, obs_margin=0.15):
    """stage2.rollout_full with a randomized start (rng independent of the planner seeds)."""
    ad = SafeMPPIAdapter(**cfg)
    rng = np.random.default_rng(10_000_000 + seed)
    st = sample_start(env, rng, offdiag=offdiag, obs_margin=obs_margin)
    start = st.copy()
    goal_t = env.goal.detach().cpu().float()
    obs_plan = GS.planner_obstacles(env)
    goal = env.goal.detach().cpu().numpy()
    states, controls = [st.copy()], []
    for t in range(env.T):
        a, _ = ad.plan(torch.tensor(st, dtype=torch.float32), goal_t, obs_plan, gamma=gamma, seed=seed * 1000 + t)
        a = a.detach().cpu().numpy().astype(np.float32)
        st = di_step(st, a, dt=env.dt)
        states.append(st.copy()); controls.append(a)
        if np.linalg.norm(st[:2] - goal) < reach:
            break
    return np.array(states, np.float32), np.array(controls, np.float32), start


def generate(gamma, seeds, env, cfg, s0=0, offdiag=0.0, obs_margin=0.15, log=print):
    G, L, Hh, U, starts = [], [], [], [], []
    n_ok = 0; t0 = time.time()
    for s in range(s0, s0 + seeds):
        states, controls, start = rollout_dr(env, gamma, cfg, s, offdiag=offdiag, obs_margin=obs_margin)
        ok, _ = GS.is_success(states[:, :2], env)
        if not ok or len(controls) < 2:
            continue
        n_ok += 1
        g, l, h, u = windows_from(states, controls, env, gamma)
        G += g; L += l; Hh += h; U += u; starts.append(start)
        if (s - s0 + 1) % 50 == 0:
            log(f"  γ{gamma}: {s-s0+1}/{seeds} seeds, {n_ok} success, {len(G)} windows, "
                f"{(time.time()-t0)/(s-s0+1):.2f}s/seed", flush=True)
    return dict(grid=torch.tensor(np.array(G)), low5=torch.tensor(np.array(L)),
                hist=torch.tensor(np.array(Hh)), U=torch.tensor(np.array(U)),
                starts=torch.tensor(np.array(starts)) if starts else torch.zeros(0, 4),
                gamma=float(gamma), n_traj=n_ok, n_seeds=seeds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=400)
    ap.add_argument("--s0", type=int, default=0, help="seed offset (append runs)")
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.1, 0.5, 1.0])
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--offdiag", type=float, default=0.0, help="require |y-x| >= this at the start")
    ap.add_argument("--obs-margin", type=float, default=0.15,
                    help="min clearance (m) between robot body and obstacle surface at the start")
    ap.add_argument("--out-prefix", default="dr_", help='shard prefix ("" appends to the MAIN windows_g*.pt)')
    args = ap.parse_args()
    env = GS.make_grid(); cfg = GS.mode1_config()
    print(f"=== PHASE-DR data: random starts (goal fixed), {len(env.obstacles)} obstacles, "
          f"seeds {args.s0}..{args.s0+args.seeds}, offdiag {args.offdiag}, obs_margin {args.obs_margin}, "
          f"prefix '{args.out_prefix}' ===",
          flush=True)
    for g in args.gammas:
        d = generate(g, args.seeds, env, cfg, s0=args.s0, offdiag=args.offdiag, obs_margin=args.obs_margin)
        out = os.path.join(DATA, f"{args.out_prefix}windows_g{g}.pt")
        if args.append and os.path.exists(out):
            old = torch.load(out)
            for k in ("grid", "low5", "hist", "U", "starts"):
                if k in old:
                    d[k] = torch.cat([old[k], d[k]], 0)
                elif k == "starts":
                    d.pop(k, None)        # main shards have no starts key — keep schema unchanged
            d["n_traj"] += old.get("n_traj", 0); d["n_seeds"] += old.get("n_seeds", 0)
        torch.save(d, out)
        print(f"γ{g}: saved {d['grid'].shape[0]} DR windows from {d['n_traj']} successes "
              f"({d['n_traj']}/{d['n_seeds']} rate) -> {out}", flush=True)


if __name__ == "__main__":
    main()
