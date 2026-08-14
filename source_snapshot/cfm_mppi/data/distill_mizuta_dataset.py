"""Flavor B: distill the SOTA policy (Mizuta) into a canonical dataset.

Runs Mizuta CFM-MPPI over real UCY scenes, records the executed (context, control)
trajectories, keeps the successful ones, and saves in canonical format so our
CFM/drifting proposal can be trained to imitate Mizuta — then wrapped in our
certificate (cfm_proposal_mppi / guided_drifting) for a self-contained, fast,
provably-safe, tunable policy.

  python -m cfm_mppi.data.distill_mizuta_dataset --episode-start 0 --episode-end 100 \
      --output-dir dataset/mizuta_distill --device cuda --success-only
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from cfm_mppi.evaluation.eval_benchmark import BenchmarkPolicies
from cfm_mppi.evaluation.render_validation_comparison import (
    get_parser as _render_parser, _make_scene, _frame_obstacles, _frame_velocities,
    _policy_args, _dynamics_step,
)
from cfm_mppi.data.canonical_dataset import _make_histories, save_canonical_splits

DT = 0.1


def _nearest_rel(pos, obs, vel):
    if obs.shape[0] == 0:
        return np.zeros(4, dtype=np.float32)
    d = np.linalg.norm(obs[:, :2] - pos[None, :], axis=1) - obs[:, 2]
    j = int(np.argmin(d))
    rv = vel[j] if vel.shape[0] > j else np.zeros(2, dtype=np.float32)
    return np.array([obs[j, 0] - pos[0], obs[j, 1] - pos[1], rv[0], rv[1]], dtype=np.float32)


def run(cli):
    base = _render_parser().parse_args([])
    base.dataset = "ucy"; base.dynamics = "doubleintegrator"; base.pedestrian_source = "validation"
    base.steps = cli.steps
    pol = BenchmarkPolicies(_policy_args(base), torch.device(cli.device))

    A = {k: [] for k in ["states", "controls", "obsrel", "gamma", "start", "goal"]}
    eps_reps = [(ep, r) for ep in range(cli.episode_start, cli.episode_end) for r in range(cli.repeats)]
    for ep, rep in eps_reps:
        base.episode = ep
        s0, goal, obs_seq, vel_seq, _ = _make_scene(base)
        pol._mizuta_episode = None
        state = s0.astype(np.float32).copy()
        states = [state.copy()]; controls = []; obsrel = []
        for t in range(cli.steps):
            ob = _frame_obstacles(obs_seq, t); ve = _frame_velocities(vel_seq, t)
            a, _ = pol.action("mizuta_cfm_mppi", state, goal, ob, controls,
                              "doubleintegrator", 0.5, cli.steps, obstacle_velocities=ve)
            obsrel.append(_nearest_rel(state[:2], ob, ve))
            state = _dynamics_step(state, a, "doubleintegrator", DT)
            states.append(state.copy()); controls.append(a.astype(np.float32))
        states = np.asarray(states, dtype=np.float32)
        if cli.success_only:
            fd = float(np.linalg.norm(states[-1, :2] - goal)); mc = np.inf
            for t in range(states.shape[0]):
                ob = _frame_obstacles(obs_seq, min(t, obs_seq.shape[0]-1))
                if ob.shape[0]:
                    mc = min(mc, float(np.min(np.linalg.norm(ob[:, :2] - states[t, :2], axis=1) - ob[:, 2] - 0.5)))
            if not (fd <= 0.5 and mc >= 0.0):
                continue
        A["states"].append(states); A["controls"].append(np.asarray(controls, dtype=np.float32))
        A["obsrel"].append(np.asarray(obsrel, dtype=np.float32)); A["gamma"].append(0.4)
        A["start"].append(s0[:2].astype(np.float32)); A["goal"].append(goal.astype(np.float32))
        if (ep+1) % 10 == 0:
            print(f"[distill] ep {ep} kept={len(A['states'])}", flush=True)

    n = len(A["states"])
    states = torch.from_numpy(np.stack(A["states"])); controls = torch.from_numpy(np.stack(A["controls"]))
    obsrel = torch.from_numpy(np.stack(A["obsrel"]))
    eh, ah, oh = _make_histories(states, controls, obsrel, cli.history_len)
    data = {"states": states, "controls_dyn": controls.clone(), "controls_si": controls.clone(),
            "start": torch.from_numpy(np.stack(A["start"])), "goal": torch.from_numpy(np.stack(A["goal"])),
            "ego_history": eh, "action_history": ah, "nearest_obstacle_history": oh,
            "obstacles": torch.zeros(n, 1, 3), "gamma": torch.tensor(A["gamma"], dtype=torch.float32),
            "dynamics_type": ["doubleintegrator"]*n, "safety_margin": torch.full((n,), 0.5),
            "source": ["mizuta_distill"]*n,
            "metadata": {"schema_version": 1, "source_format": "mizuta_distill", "dt": DT, "history_len": cli.history_len}}
    paths = save_canonical_splits(data, Path(cli.output_dir), seed=cli.seed)
    print(f"[distill] saved {n} Mizuta trajectories -> {cli.output_dir}: {paths}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episode-start", type=int, default=0)
    p.add_argument("--episode-end", type=int, default=100)
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--history-len", type=int, default=10)
    p.add_argument("--success-only", action="store_true")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default="dataset/mizuta_distill")
    run(p.parse_args())


if __name__ == "__main__":
    main()
