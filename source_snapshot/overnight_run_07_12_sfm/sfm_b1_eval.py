"""Strict raw SFM evaluation: temp=1, NFE=8, one window, first action."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os

import numpy as np
import torch

import _paths  # noqa: F401
import grid_feats as GF
import grid_policy_sfm as GPS
import sfm_hp_history as HH
import sfm_scene as SS


def integrate_latents(policy, latents, context, *, nfe=8):
    """Unguided Euler flow used identically by raw evaluation and B1 proposals."""
    value = torch.as_tensor(latents, device=policy.head.weight.device)
    context = torch.as_tensor(context, device=value.device)
    if context.ndim == 1:
        context = context[None].expand(len(value), -1)
    if len(context) != len(value):
        raise ValueError("one context is required per latent")
    for index in range(int(nfe)):
        time = torch.full((len(value),), index / int(nfe), device=value.device, dtype=value.dtype)
        value = value + policy.forward(value, time, context) / int(nfe)
    return (value.reshape(len(value), policy.H_pred, 2) * policy.u_max).clamp(-policy.u_max, policy.u_max)


def generate_windows(policy, hp10, low5, hist, *, K, nfe=8, temp=1.0, generator=None):
    """One batched proposal call across all alive contexts."""
    context = policy.ctx_from(hp10, low5, hist)
    expanded = context.repeat_interleave(int(K), dim=0)
    latents = float(temp) * torch.randn(
        len(expanded), policy.d, device=expanded.device, dtype=expanded.dtype, generator=generator
    )
    windows = integrate_latents(policy, latents, expanded, nfe=nfe)
    return windows.reshape(len(context), int(K), policy.H_pred, 2)


def _step(state, action):
    state = np.asarray(state, np.float32).copy()
    action = np.asarray(action, np.float32)
    state[:2] += SS.DT * state[2:4] + 0.5 * SS.DT ** 2 * action
    state[2:4] += SS.DT * action
    return state


def classify_candidate(segment, pedestrian_prediction):
    robot = np.asarray(segment, float)
    peds = np.asarray(pedestrian_prediction, float)
    if len(robot) < 2 or not peds.size:
        return "yield"
    distance = np.linalg.norm(robot[:, None, :] - peds, axis=2)
    horizon, pedestrian = np.unravel_index(int(np.argmin(distance)), distance.shape)
    progress = np.linalg.norm(robot[0] - SS.GOAL) - np.linalg.norm(robot[-1] - SS.GOAL)
    speed = np.linalg.norm(np.diff(robot, axis=0), axis=1).max(initial=0.0)
    if progress < 0.04 or speed < 0.015:
        return "yield"
    direction = SS.GOAL - robot[0]
    relative = peds[horizon, pedestrian] - robot[horizon]
    cross = direction[0] * relative[1] - direction[1] * relative[0]
    return "left" if cross >= 0.0 else "right"


def raw_rollout(policy, episode, gamma, *, device="cpu", T=180, n_ped=20, temp=1.0, nfe=8,
                reach=0.5, ped_speed_range=SS.OOD_PED_SPEED_RANGE, sample_seed=700_000,
                collect_trace=False):
    if float(temp) != 1.0 or int(nfe) != 8:
        raise ValueError("the frozen raw evaluator requires temp=1 and NFE=8")
    humans = SS.make_humans(episode, 0, n_ped, ped_speed_range)
    state = np.zeros(4, np.float32)
    controls = []
    states = [state.copy()]
    pedestrian_rows = []
    history = HH.HpHistory()
    minimum_clearance = float("inf")
    collision = reached = False
    trace = []
    episode_modes = Counter()
    generator = torch.Generator(device=device).manual_seed(int(sample_seed) + int(episode) * 1000)
    for step in range(int(T)):
        ped_xy, ped_vel = SS.collect_humans(humans)
        clearance = float(np.linalg.norm(ped_xy - state[:2][None], axis=1).min() - SS.R_PED)
        minimum_clearance = min(minimum_clearance, clearance)
        if clearance < 0.0:
            collision = True
            break
        if float(np.linalg.norm(state[:2] - SS.GOAL)) < float(reach):
            reached = True
            break
        obstacles = np.concatenate([ped_xy, np.full((n_ped, 1), SS.R_PED, np.float32)], axis=1)
        raw_grid = torch.as_tensor(
            GF.axis_grid(state[:2], obstacles, 0.0, R=SS.R_SENSE, sensing=SS.R_SENSE), device=device
        )
        hp10 = history.append(raw_grid).to(device)
        low = torch.as_tensor(GF.low5(state, SS.GOAL, gamma), device=device)
        control_history = torch.as_tensor(
            GF.hist_pad(np.asarray(controls[-16:]) if controls else np.zeros((0, 2)), 16), device=device
        )
        with torch.no_grad():
            window = generate_windows(
                policy, hp10[None], low[None], control_history[None], K=1,
                nfe=nfe, temp=temp, generator=generator,
            )[0, 0]
        action = window[0].detach().cpu().numpy().astype(np.float32)
        before = state.copy()
        state = _step(state, action)
        controls.append(action)
        states.append(state.copy())
        pedestrian_rows.append(ped_xy.copy())
        plan = [before.copy()]
        cursor = before.copy()
        for proposal_action in window.detach().cpu().numpy():
            cursor = _step(cursor, proposal_action)
            plan.append(cursor.copy())
        ped_pred = ped_xy[None] + np.arange(11)[:, None, None] * SS.DT * ped_vel[None]
        mode = classify_candidate(np.asarray(plan)[:, :2], ped_pred)
        episode_modes[mode] += 1
        if collect_trace:
            trace.append(dict(
                step=step, state=before, action=action, controls=window.detach().cpu().numpy(),
                planned_states=np.asarray(plan), ped_xy=ped_xy.copy(), ped_vel=ped_vel.copy(), mode=mode,
            ))
        SS.advance_humans(humans, state)
    if not collision and not reached:
        terminal_xy, _ = SS.collect_humans(humans)
        terminal_clearance = float(np.linalg.norm(terminal_xy - state[:2][None], axis=1).min() - SS.R_PED)
        minimum_clearance = min(minimum_clearance, terminal_clearance)
        collision = terminal_clearance < 0.0
        reached = bool(not collision and np.linalg.norm(state[:2] - SS.GOAL) < float(reach))
    states = np.asarray(states, np.float32)
    successful_clearance = minimum_clearance if reached and not collision else None
    return dict(
        episode=int(episode), gamma=float(gamma), success=bool(reached and not collision),
        collision=bool(collision), reached=bool(reached), timeout=bool(not reached and not collision),
        steps=len(controls), time_to_goal=(len(controls) * SS.DT if reached and not collision else None),
        min_clearance=float(minimum_clearance), successful_clearance=successful_clearance,
        states=states, controls=np.asarray(controls, np.float32), peds=np.asarray(pedestrian_rows, np.float32),
        n_ped=int(n_ped), ped_speed_range=tuple(map(float, ped_speed_range)),
        trace=trace if collect_trace else None, mode_counts=dict(episode_modes),
        raw_semantics="temp=1,NFE=8,one generated window per context,execute first action,plain flow only",
    )


def wilson(successes, total, z=1.959963984540054):
    if int(total) == 0:
        return [None, None]
    proportion = float(successes) / int(total)
    denominator = 1.0 + z * z / int(total)
    center = (proportion + z * z / (2 * int(total))) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / int(total) + z * z / (4 * total * total)) / denominator
    return [center - radius, center + radius]


def bootstrap_mean(values, *, seed=0, draws=10_000):
    values = np.asarray([value for value in values if value is not None], float)
    if not len(values):
        return dict(mean=None, interval95=[None, None], n=0)
    generator = np.random.default_rng(int(seed))
    means = values[generator.integers(0, len(values), size=(int(draws), len(values)))].mean(axis=1)
    return dict(mean=float(values.mean()), interval95=list(map(float, np.quantile(means, [.025, .975]))), n=len(values))


def summarize(rows, *, bootstrap_seed=0):
    def one(values, seed):
        total = len(values)
        successes = sum(row["success"] for row in values)
        collisions = sum(row["collision"] for row in values)
        support = Counter()
        for row in values:
            support.update(row.get("mode_counts", {}))
        return dict(
            n=total, SR=successes / total, SR_wilson95=wilson(successes, total),
            CR=collisions / total, CR_wilson95=wilson(collisions, total),
            successful_clearance=bootstrap_mean([row["successful_clearance"] for row in values], seed=seed),
            successful_time_to_goal=bootstrap_mean([row["time_to_goal"] for row in values], seed=seed + 1),
            unconditional_min_clearance=bootstrap_mean([row["min_clearance"] for row in values], seed=seed + 2),
            support=dict(support),
        )
    per_gamma = {str(gamma): one([row for row in rows if row["gamma"] == gamma], bootstrap_seed + index * 10)
                 for index, gamma in enumerate(SS.GAMMAS)}
    return dict(pooled=one(rows, bootstrap_seed + 100), per_gamma=per_gamma)


def selection_key(summary):
    per_gamma = list(summary["per_gamma"].values())
    worst_cr = max(row["CR"] for row in per_gamma)
    worst_sr = min(row["SR"] for row in per_gamma)
    clearance = summary["pooled"]["successful_clearance"]["mean"]
    time = summary["pooled"]["successful_time_to_goal"]["mean"]
    support = summary["pooled"]["support"]
    noncollapsed = min((support.get(mode, 0) for mode in ("left", "right", "yield")), default=0)
    return (
        0 if worst_cr < 0.05 else 1, summary["pooled"]["CR"],
        -worst_sr, -summary["pooled"]["SR"],
        -(clearance if clearance is not None else -float("inf")),
        time if time is not None else float("inf"), -noncollapsed,
    )


def evaluate_policy(policy, bank, *, device, scene_profile, collect_trace=False, sample_seed=700_000):
    environment = SS.scene_profile(scene_profile)
    rows = []
    for gamma in SS.GAMMAS:
        for episode in bank[str(gamma)]:
            rows.append(raw_rollout(
                policy, episode, gamma, device=device, sample_seed=sample_seed,
                n_ped=environment["n_ped"],
                ped_speed_range=tuple(environment["ped_speed_range"]),
                collect_trace=collect_trace,
            ))
    return rows, summarize(rows)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ep0", type=int, required=True)
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--scene-profile", required=True, choices=SS.SCIENTIFIC_EVAL_PROFILES)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    policy, _ = GPS.load_sfm_policy(args.checkpoint, device=args.device)
    bank = {str(gamma): list(range(args.ep0, args.ep0 + args.M)) for gamma in SS.GAMMAS}
    rows, summary = evaluate_policy(
        policy, bank, device=args.device, scene_profile=args.scene_profile,
    )
    payload = dict(
        checkpoint=os.path.abspath(args.checkpoint), checkpoint_sha256=sha256_file(args.checkpoint),
        bank=bank, environment=SS.scene_profile(args.scene_profile), summary=summary,
        rows=[{key: value for key, value in row.items()
              if key not in ("states", "controls", "peds", "trace")} for row in rows],
        empirical_target_note="CR<5% is an empirical target, not a proof under real SFM dynamics",
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as stream:
        json.dump(payload, stream, indent=2)


if __name__ == "__main__":
    main()
