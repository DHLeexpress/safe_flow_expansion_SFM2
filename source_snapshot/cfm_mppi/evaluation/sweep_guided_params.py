"""Parameter sweep for Guided Safe MPPI.

Objective: maximize success subject to ZERO collisions (safety is enforced by
hard rejection, so we can freely push goal-seeking). Searches gamma, eta, barrier
margin, goal/progress cost weights, samples, horizon over a fixed validation set
of moving-pedestrian episodes. No Mizuta in the loop => fast. Ranks configs by
score = success_rate - PENALTY * collision_rate.

  python -m cfm_mppi.evaluation.sweep_guided_params --dataset ucy \
      --num-episodes 24 --num-configs 40 --device cuda \
      --output overnight_run_2026-06-23/param_sweep
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from cfm_mppi.safegpc_adapter.safemppi import SafeMPPIAdapter
from cfm_mppi.evaluation.render_validation_comparison import (
    get_parser as _render_parser,
    _make_scene,
    _frame_obstacles,
    _frame_velocities,
    _compute_dynamic_episode_metrics,
)

DT = 0.1
COLLISION_PENALTY = 5.0

SEARCH = {
    "gamma": [0.15, 0.2, 0.25, 0.3],
    "eta": [0.4, 0.6, 0.9, 1.2],
    "barrier_extra_margin": [0.15, 0.25, 0.35],
    "progress_weight": [4.0, 6.0, 9.0],
    "terminal_goal_weight": [120.0, 200.0],
    "running_goal_weight": [0.4, 0.6, 0.9],
    "num_samples": [384, 512],
    "horizon": [30, 40],
    "guidance_horizon": [10, 16],
    "aniso_tangent_scale": [1.4, 1.7, 2.0],
}


def _di_step(state, action, dt):
    s = state.copy()
    s[0] += dt * state[2] + 0.5 * dt * dt * action[0]
    s[1] += dt * state[3] + 0.5 * dt * dt * action[1]
    s[2] += dt * action[0]
    s[3] += dt * action[1]
    return s


def _load_scenes(cli):
    base = _render_parser().parse_args([])
    base.dataset = cli.dataset
    base.dynamics = "doubleintegrator"
    base.pedestrian_source = "validation"
    base.steps = cli.steps
    base.seed = 0
    scenes = []
    for ep in range(cli.num_episodes):
        base.episode = ep
        state0, goal, obs_seq, vel_seq, _ = _make_scene(base)
        scenes.append((state0, goal, obs_seq, vel_seq))
    return scenes


def _eval_config(cfg, scenes, device):
    succ, coll, clears = [], [], []
    for (state0, goal, obs_seq, vel_seq) in scenes:
        adapter = SafeMPPIAdapter(
            horizon=cfg["horizon"], dt=DT, num_samples=cfg["num_samples"], gamma=cfg["gamma"],
            noise_sigma=(0.4, 0.4), u_min=(-2.0, -2.0), u_max=(2.0, 2.0), safety_margin=0.5,
            dynamics_type="doubleintegrator", use_ho_barrier=True, eta=cfg["eta"],
            use_guidance=True, guidance_horizon=cfg["guidance_horizon"], use_aniso_cov=True,
            aniso_tangent_scale=cfg["aniso_tangent_scale"], barrier_extra_margin=cfg["barrier_extra_margin"],
            barrier_activation_radius=3.5, progress_weight=cfg["progress_weight"],
            terminal_goal_weight=cfg["terminal_goal_weight"], running_goal_weight=cfg["running_goal_weight"],
            filter_output=True,
        )
        steps = obs_seq.shape[0] - 1
        state = state0.astype(np.float32).copy()
        states = [state.copy()]
        controls = []
        for t in range(steps):
            obs = _frame_obstacles(obs_seq, t)
            vel = _frame_velocities(vel_seq, t)
            a, _ = adapter.plan(
                torch.tensor(state, dtype=torch.float32, device=device),
                torch.tensor(goal, dtype=torch.float32, device=device),
                torch.tensor(obs, dtype=torch.float32, device=device),
                gamma=cfg["gamma"],
                obstacle_velocities=torch.tensor(vel, dtype=torch.float32, device=device),
                seed=t,
            )
            a = a.detach().cpu().numpy()
            state = _di_step(state, a, DT)
            states.append(state.copy())
            controls.append(a)
        m = _compute_dynamic_episode_metrics(
            np.asarray(states, dtype=np.float32), np.asarray(controls, dtype=np.float32),
            obs_seq, goal, safety_margin=0.5, success_threshold=0.5, planning_times=[],
            min_barrier_h=None, num_barrier_violations=0,
        )
        succ.append(m["success"]); coll.append(m["collision"])
        if np.isfinite(m["min_clearance"]):
            clears.append(m["min_clearance"])
    sr = float(np.mean(succ)); cr = float(np.mean(coll))
    return {"success_rate": sr, "collision_rate": cr,
            "mean_min_clearance": float(np.mean(clears)) if clears else float("nan"),
            "score": sr - COLLISION_PENALTY * cr}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="ucy")
    p.add_argument("--num-episodes", type=int, default=24)
    p.add_argument("--num-configs", type=int, default=40)
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default="overnight_run_2026-06-23/param_sweep")
    cli = p.parse_args()

    device = torch.device(cli.device)
    rng = random.Random(cli.seed)
    scenes = _load_scenes(cli)
    print(f"loaded {len(scenes)} scenes from {cli.dataset}", flush=True)

    # sample distinct configs
    seen = set()
    configs = []
    while len(configs) < cli.num_configs and len(seen) < 10000:
        cfg = {k: rng.choice(v) for k, v in SEARCH.items()}
        key = tuple(sorted(cfg.items()))
        if key in seen:
            continue
        seen.add(key)
        configs.append(cfg)

    results = []
    for i, cfg in enumerate(configs):
        res = _eval_config(cfg, scenes, device)
        results.append({"config": cfg, **res})
        print(f"[{i+1}/{len(configs)}] score={res['score']:.3f} "
              f"succ={res['success_rate']:.2f} coll={res['collision_rate']:.2f} "
              f"clr={res['mean_min_clearance']:.2f} :: g={cfg['gamma']} eta={cfg['eta']} "
              f"m={cfg['barrier_extra_margin']} prog={cfg['progress_weight']} "
              f"S={cfg['num_samples']} H={cfg['horizon']}", flush=True)

    results.sort(key=lambda r: (r["score"], r["success_rate"]), reverse=True)
    out = Path(cli.output); out.mkdir(parents=True, exist_ok=True)
    (out / "sweep_results.json").write_text(json.dumps(results, indent=2))
    print("\n=== TOP 8 CONFIGS (zero-collision preferred) ===")
    for r in results[:8]:
        print(f"score={r['score']:.3f} succ={r['success_rate']:.2f} coll={r['collision_rate']:.2f} "
              f"clr={r['mean_min_clearance']:.2f} | {r['config']}")
    zero_coll = [r for r in results if r["collision_rate"] == 0.0]
    if zero_coll:
        best = max(zero_coll, key=lambda r: r["success_rate"])
        print(f"\nBEST ZERO-COLLISION: succ={best['success_rate']:.2f} clr={best['mean_min_clearance']:.2f}")
        print(json.dumps(best["config"], indent=2))
    print(f"\nWrote {out}/sweep_results.json")


if __name__ == "__main__":
    main()
