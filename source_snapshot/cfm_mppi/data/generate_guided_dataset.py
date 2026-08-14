"""Generate a canonical training dataset by running Guided Safe MPPI over many
simulated social-force (SFM) crowd scenes across a gamma grid.

Design: TRAIN on simulated SFM crowds (unlimited seeds), so the held-out real
UCY/SDD pedestrian tracks remain a clean generalization test set. Each (scene,
gamma) produces one canonical trajectory item conditioned on initial context and
the scalar safety parameter gamma. Output matches the canonical schema consumed
by train_safe_cfm / train_drifting.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from cfm_mppi.safegpc_adapter.safemppi import SafeMPPIAdapter
from cfm_mppi.data.canonical_dataset import _make_histories, save_canonical_splits
from cfm_mppi.evaluation.render_validation_comparison import (
    get_parser as _render_parser,
    _make_sfm_social_force_scene,
)

DT = 0.1


def _nearest_rel(state_xy, obstacles_t, velocities_t):
    if obstacles_t.shape[0] == 0:
        return np.zeros(4, dtype=np.float32)
    d = np.linalg.norm(obstacles_t[:, :2] - state_xy[None, :], axis=1) - obstacles_t[:, 2]
    j = int(np.argmin(d))
    rel = obstacles_t[j, :2] - state_xy
    relv = velocities_t[j] if velocities_t.shape[0] > j else np.zeros(2, dtype=np.float32)
    return np.array([rel[0], rel[1], relv[0], relv[1]], dtype=np.float32)


def _di_step(state, action, dt):
    s = state.copy()
    s[0] += dt * state[2] + 0.5 * dt * dt * action[0]
    s[1] += dt * state[3] + 0.5 * dt * dt * action[1]
    s[2] += dt * action[0]
    s[3] += dt * action[1]
    return s


def _si_step(state, action, dt):
    # single integrator: control IS velocity; position += dt*u, velocity channels track u
    s = state.copy()
    s[0] += dt * action[0]
    s[1] += dt * action[1]
    s[2] = action[0]
    s[3] = action[1]
    return s


def run(cli):
    device = torch.device(cli.device)
    base = _render_parser().parse_args([])
    base.dynamics = cli.dynamics
    base.steps = cli.steps
    base.num_pedestrians = cli.num_pedestrians
    base.pedestrian_radius = 0.0

    gammas = list(cli.gamma_grid)
    all_states, all_controls, all_obsrel, all_gamma = [], [], [], []
    all_start, all_goal = [], []

    rng = np.random.RandomState(cli.seed)
    validation = cli.scene_source in ("ucy", "sdd")
    if validation:
        from cfm_mppi.evaluation.render_validation_comparison import _make_scene
        base.dataset = cli.scene_source
        base.pedestrian_source = "validation"
        scene_indices = list(range(cli.episode_start, cli.episode_end))
    else:
        scene_indices = list(range(cli.num_scenes))
    for si in scene_indices:
        base.seed = cli.seed + si
        base.episode = si
        if validation:
            state0, goal, obstacles_seq, velocities_seq, _ = _make_scene(base)
        else:
            state0, goal, obstacles_seq, velocities_seq, _ = _make_sfm_social_force_scene(base)
        if cli.randomize_pose and not validation:
            # rigid rotation about the start + goal-distance scaling => diverse
            # start/goal directions & distances so the proposal learns RELATIVE
            # goal-seeking and generalizes beyond the fixed SFM (0,0)->(6,6).
            th = rng.uniform(0, 2 * np.pi)
            s = rng.uniform(0.6, 1.7)
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=np.float32)
            origin = state0[:2].copy()
            def _tf(xy):
                return ((xy - origin) * s) @ R.T + origin
            goal = _tf(goal.astype(np.float32)).astype(np.float32)
            obstacles_seq = obstacles_seq.copy()
            obstacles_seq[..., :2] = _tf(obstacles_seq[..., :2].reshape(-1, 2)).reshape(obstacles_seq[..., :2].shape)
            velocities_seq = (velocities_seq.reshape(-1, 2) @ R.T).reshape(velocities_seq.shape).astype(np.float32)
        for rep in range(cli.repeats):
          for gamma in gammas:
            if cli.dynamics == "singleintegrator":
                # relative-degree-1: position-only affine DCBF (no HO/braking term).
                # control IS velocity, so u bounds are speed limits and noise is in velocity.
                adapter = SafeMPPIAdapter(
                    horizon=cli.horizon, dt=DT, num_samples=cli.samples, gamma=gamma,
                    noise_sigma=(0.5, 0.5), u_min=(-cli.umax, -cli.umax), u_max=(cli.umax, cli.umax),
                    safety_margin=0.5, dynamics_type="singleintegrator",
                    use_ho_barrier=False, eta=0.0, use_guidance=True,
                    use_aniso_cov=True, barrier_topk=6,
                    barrier_activation_radius=cli.sensing_range,
                )
            else:
                adapter = SafeMPPIAdapter(
                    horizon=cli.horizon, dt=DT, num_samples=cli.samples, gamma=gamma,
                    noise_sigma=(0.4, 0.4), u_min=(-3.0, -3.0), u_max=(3.0, 3.0),
                    safety_margin=0.5, dynamics_type="doubleintegrator",
                    use_ho_barrier=True, eta=cli.eta, use_guidance=True,
                    use_aniso_cov=True, barrier_topk=6,
                    barrier_activation_radius=cli.sensing_range,
                )
            T = cli.steps
            state = state0.astype(np.float32).copy()
            states = [state.copy()]
            controls = []
            obsrel = []
            for t in range(T):
                obs_t = obstacles_seq[min(t, obstacles_seq.shape[0] - 1)]
                vel_t = velocities_seq[min(t, velocities_seq.shape[0] - 1)]
                a, _ = adapter.plan(
                    torch.tensor(state, dtype=torch.float32, device=device),
                    torch.tensor(goal, dtype=torch.float32, device=device),
                    torch.tensor(obs_t, dtype=torch.float32, device=device),
                    gamma=gamma,
                    obstacle_velocities=torch.tensor(vel_t, dtype=torch.float32, device=device),
                    seed=si * 100000 + rep * 1000 + t,
                )
                a = a.detach().cpu().numpy()
                obsrel.append(_nearest_rel(state[:2], obs_t, vel_t))
                state = _si_step(state, a, DT) if cli.dynamics == "singleintegrator" else _di_step(state, a, DT)
                states.append(state.copy())
                controls.append(a.astype(np.float32))
            states_arr = np.asarray(states, dtype=np.float32)
            if cli.success_only:
                # keep only goal-reaching, collision-free trajectories (the
                # "success cases" the proposal should imitate)
                fd = float(np.linalg.norm(states_arr[-1, :2] - goal))
                mc = np.inf
                for t in range(states_arr.shape[0]):
                    ob = obstacles_seq[min(t, obstacles_seq.shape[0] - 1)]
                    if ob.shape[0]:
                        mc = min(mc, float(np.min(np.linalg.norm(ob[:, :2] - states_arr[t, :2], axis=1) - ob[:, 2] - 0.5)))
                if not (fd <= 0.5 and mc >= 0.0):
                    continue
            all_states.append(np.asarray(states, dtype=np.float32))
            all_controls.append(np.asarray(controls, dtype=np.float32))
            all_obsrel.append(np.asarray(obsrel, dtype=np.float32))
            all_gamma.append(float(gamma))
            all_start.append(state0[:2].astype(np.float32))
            all_goal.append(goal.astype(np.float32))
        if (si + 1) % 5 == 0:
            print(f"[gen] scene idx {si}  items={len(all_states)}", flush=True)

    n = len(all_states)
    states = torch.from_numpy(np.stack(all_states))          # [n, T+1, 4]
    controls = torch.from_numpy(np.stack(all_controls))      # [n, T, 2]
    obsrel = torch.from_numpy(np.stack(all_obsrel))          # [n, T, 4]
    gamma = torch.tensor(all_gamma, dtype=torch.float32)
    start = torch.from_numpy(np.stack(all_start))
    goal = torch.from_numpy(np.stack(all_goal))
    hist_len = cli.history_len
    ego_hist, act_hist, obs_hist = _make_histories(states, controls, obsrel, hist_len)

    data = {
        "states": states,
        "controls_dyn": controls.clone(),
        "controls_si": controls.clone(),
        "start": start,
        "goal": goal,
        "ego_history": ego_hist,
        "action_history": act_hist,
        "nearest_obstacle_history": obs_hist,
        "obstacles": torch.zeros(n, 1, 3, dtype=torch.float32),
        "gamma": gamma,
        "dynamics_type": [cli.dynamics] * n,
        "safety_margin": torch.full((n,), 0.5, dtype=torch.float32),
        "source": [f"guided_safemppi_{cli.scene_source}"] * n,
        "metadata": {"schema_version": 1, "source_format": "guided_safemppi",
                     "dt": DT, "history_len": hist_len, "gamma_grid": gammas,
                     "eta": cli.eta, "num_scenes": len(scene_indices),
                     "scene_source": cli.scene_source, "dynamics": cli.dynamics,
                     "sensing_range": cli.sensing_range},
    }
    paths = save_canonical_splits(data, Path(cli.output_dir), seed=cli.seed)
    print(f"[gen] saved {n} items to {cli.output_dir}: {paths}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-scenes", type=int, default=400)
    p.add_argument("--num-pedestrians", type=int, default=12)
    p.add_argument("--gamma-grid", nargs="+", type=float, default=[0.1, 0.3, 0.5, 0.7, 1.0])
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--eta", type=float, default=0.6)
    p.add_argument("--history-len", type=int, default=10)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default="dataset/canonical_guided")
    p.add_argument("--randomize-pose", action="store_true",
                   help="random rigid rotation + distance scaling of start/goal/obstacles per scene")
    p.add_argument("--scene-source", default="sfm", choices=["sfm", "ucy", "sdd"],
                   help="sfm = social-force synthetic; ucy/sdd = real validation pedestrian scenes")
    p.add_argument("--episode-start", type=int, default=0)
    p.add_argument("--episode-end", type=int, default=100)
    p.add_argument("--success-only", action="store_true",
                   help="keep only goal-reaching, collision-free trajectories")
    p.add_argument("--repeats", type=int, default=1,
                   help="stochastic rollouts per (episode, gamma) for data volume/diversity")
    p.add_argument("--dynamics", default="doubleintegrator",
                   choices=["singleintegrator", "doubleintegrator"],
                   help="robot dynamics model (single integrator = control is velocity)")
    p.add_argument("--umax", type=float, default=3.0,
                   help="control bound; for single integrator this is the speed limit (m/s)")
    p.add_argument("--sensing-range", type=float, default=3.5,
                   help="finite sensing range: obstacle barriers activate within this radius")
    run(p.parse_args())


if __name__ == "__main__":
    main()
