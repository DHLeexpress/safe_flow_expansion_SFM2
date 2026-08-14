"""Publication figure: the sampling distribution + DCBF rejection at one frame.

Renders, for a chosen moving-pedestrian episode and timestep, the MPPI proposal
cloud mapped to predicted next positions, colored accepted (feasible, blue) vs
rejected (red), with the moving affine half-planes and the chosen action. Works
for the Gaussian/guided proposal now; pass --proposal-controls (from a learned
q_theta) to visualize the LEARNED sampling distribution in the same axes.

  python -m cfm_mppi.evaluation.visualize_sampling --dataset ucy --episode 110 \
      --frame 20 --gamma 0.2 --output overnight_run_2026-06-23/figs/sampling_frame.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from cfm_mppi.safegpc_adapter.safemppi import SafeMPPIAdapter
from cfm_mppi.evaluation.render_validation_comparison import (
    get_parser as _render_parser, _make_scene, _frame_obstacles, _frame_velocities,
)

DT = 0.1


def _di_step(state, a, dt):
    s = state.copy()
    s[0] += dt * state[2] + 0.5 * dt * dt * a[0]; s[1] += dt * state[3] + 0.5 * dt * dt * a[1]
    s[2] += dt * a[0]; s[3] += dt * a[1]
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="ucy")
    p.add_argument("--episode", type=int, default=110)
    p.add_argument("--frame", type=int, default=20)
    p.add_argument("--gamma", type=float, default=0.2)
    p.add_argument("--samples", type=int, default=384)
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default="overnight_run_2026-06-23/figs/sampling_frame.png")
    cli = p.parse_args()

    base = _render_parser().parse_args([])
    base.dataset = cli.dataset; base.dynamics = "doubleintegrator"
    base.pedestrian_source = "validation"; base.steps = cli.steps; base.episode = cli.episode
    state0, goal, obs_seq, vel_seq, label = _make_scene(base)
    device = torch.device(cli.device)

    adapter = SafeMPPIAdapter(
        horizon=cli.horizon, dt=DT, num_samples=cli.samples, gamma=cli.gamma,
        dynamics_type="doubleintegrator", u_min=(-2., -2.), u_max=(2., 2.), safety_margin=0.5,
        use_ho_barrier=True, eta=0.6, use_guidance=True, use_aniso_cov=True,
        barrier_extra_margin=0.25, filter_output=True, progress_weight=9.0,
        terminal_goal_weight=200.0, running_goal_weight=0.4, guidance_horizon=10, debug_max_rollouts=cli.samples,
    )
    # roll forward to the requested frame
    state = state0.astype(np.float32).copy()
    for t in range(cli.frame):
        obs = _frame_obstacles(obs_seq, t); vel = _frame_velocities(vel_seq, t)
        a, _ = adapter.plan(torch.tensor(state, device=device), torch.tensor(goal, device=device),
                            torch.tensor(obs, device=device), gamma=cli.gamma,
                            obstacle_velocities=torch.tensor(vel, device=device), seed=t)
        state = _di_step(state, a.numpy(), DT)
    # frame of interest: collect rollouts
    obs = _frame_obstacles(obs_seq, cli.frame); vel = _frame_velocities(vel_seq, cli.frame)
    a, info = adapter.plan(torch.tensor(state, device=device), torch.tensor(goal, device=device),
                           torch.tensor(obs, device=device), gamma=cli.gamma,
                           obstacle_velocities=torch.tensor(vel, device=device), seed=cli.frame,
                           return_rollouts=True)
    dbg = info["debug_rollouts"]
    rollouts = dbg["states"]            # [R, H+1, 4]
    feasible = dbg["feasible"]          # [R]
    best = dbg["best_state"]            # [H+1, 4]

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    # pedestrians (safety disks)
    for c in obs:
        ax.add_patch(Circle((c[0], c[1]), c[2] + 0.5, facecolor=(1, .55, 0, .10), edgecolor="#f46d43", lw=1.4, zorder=3))
        ax.scatter(c[0], c[1], s=18, c="#f46d43", zorder=5)
    # rollout cloud (the sampling distribution mapped to predicted positions)
    for r, ok in zip(rollouts, feasible):
        ax.plot(r[:, 0], r[:, 1], color=("#2b8cbe" if ok else "#d73027"),
                alpha=(0.22 if ok else 0.10), lw=0.6, zorder=2)
    ax.plot(best[:, 0], best[:, 1], color="#084081", lw=2.2, zorder=6, label="chosen (certified)")
    ax.scatter(state[0], state[1], s=70, marker="o", c="#1a9850", edgecolor="k", zorder=7, label="robot")
    ax.scatter(goal[0], goal[1], s=120, marker="*", c="gold", edgecolor="k", zorder=7, label="goal")
    acc = float(np.mean(feasible))
    ax.set_title(f"{label}  |  frame {cli.frame}  γ={cli.gamma}\n"
                 f"accept-rate={acc:.2f}  (blue=accepted, red=rejected)")
    ax.set_aspect("equal"); ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.2)
    out = Path(cli.output); out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=160); print(f"wrote {out}  accept_rate={acc:.3f}")


if __name__ == "__main__":
    main()
