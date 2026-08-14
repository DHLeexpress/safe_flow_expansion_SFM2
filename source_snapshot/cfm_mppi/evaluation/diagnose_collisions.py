"""Diagnose the SOURCE of residual collisions in Guided Safe MPPI.

With the output PSF on, sample-infeasibility cannot cause collisions. This logs,
per step, whether the feasible set H(o,γ) was EMPTY (projection could not reach
feasibility => Assumption-1 / pointwise-feasibility violated, the robot is
cornered) and classifies every collision as:
  - 'set_infeasible' : H=∅ at/near the collision step (deep cause)
  - 'prediction/approx' : H was feasible but the obstacle moved / multi-obstacle
                          projection residual / activation edge.
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
from cfm_mppi.safegpc_adapter.safemppi import SafeMPPIAdapter
from cfm_mppi.evaluation.render_validation_comparison import (
    get_parser as _render_parser, _make_scene, _frame_obstacles, _frame_velocities,
)
DT = 0.1


def _di_step(s, a, dt):
    x = s.copy()
    x[0] += dt*s[2] + 0.5*dt*dt*a[0]; x[1] += dt*s[3] + 0.5*dt*dt*a[1]
    x[2] += dt*a[0]; x[3] += dt*a[1]
    return x


def _clearance(pos, obs, margin=0.5):
    if obs.shape[0] == 0:
        return np.inf
    return float(np.min(np.linalg.norm(obs[:, :2] - pos[None, :], axis=1) - obs[:, 2] - margin))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="ucy")
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--gamma", type=float, default=0.2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    cli = p.parse_args()
    base = _render_parser().parse_args([])
    base.dataset = cli.dataset; base.dynamics = "doubleintegrator"
    base.pedestrian_source = "validation"; base.steps = 80
    device = torch.device(cli.device)

    n_coll = 0; n_eps = 0
    cls = {"set_infeasible": 0, "prediction/approx": 0}
    infeasible_step_total = 0; step_total = 0
    for ep in range(100, 100 + cli.episodes):
        base.episode = ep
        s0, goal, obs_seq, vel_seq, _ = _make_scene(base)
        ad = SafeMPPIAdapter(horizon=30, dt=DT, num_samples=512, gamma=cli.gamma,
            dynamics_type="doubleintegrator", u_min=(-2.,-2.), u_max=(2.,2.), safety_margin=0.5,
            use_ho_barrier=True, eta=0.6, use_guidance=True, use_aniso_cov=True,
            barrier_extra_margin=0.25, filter_output=True, progress_weight=9.0,
            terminal_goal_weight=200.0, running_goal_weight=0.4, guidance_horizon=10)
        st = s0.astype(np.float32).copy()
        coll_step = -1; infeas_flags = []
        steps = obs_seq.shape[0]-1
        for t in range(steps):
            obs = _frame_obstacles(obs_seq, t); vel = _frame_velocities(vel_seq, t)
            a, info = ad.plan(torch.tensor(st, device=device), torch.tensor(goal, device=device),
                              torch.tensor(obs, device=device), gamma=cli.gamma,
                              obstacle_velocities=torch.tensor(vel, device=device), seed=t)
            infeas = bool(info.get("filter_infeasible", False))
            infeas_flags.append(infeas)
            infeasible_step_total += int(infeas); step_total += 1
            st = _di_step(st, a.detach().cpu().numpy(), DT)
            clr = _clearance(st[:2], _frame_obstacles(obs_seq, t+1))
            if clr < 0 and coll_step < 0:
                coll_step = t
        n_eps += 1
        if coll_step >= 0:
            n_coll += 1
            window = infeas_flags[max(0, coll_step-3):coll_step+1]
            if any(window):
                cls["set_infeasible"] += 1
            else:
                cls["prediction/approx"] += 1
    print(f"dataset={cli.dataset} eps={n_eps} collisions={n_coll} ({100*n_coll/n_eps:.1f}%)")
    print(f"per-step H=empty rate: {100*infeasible_step_total/max(step_total,1):.2f}%")
    print("collision attribution:")
    for k, v in cls.items():
        frac = (100*v/n_coll) if n_coll else 0.0
        print(f"  {k}: {v}/{n_coll} ({frac:.0f}%)")


if __name__ == "__main__":
    main()
