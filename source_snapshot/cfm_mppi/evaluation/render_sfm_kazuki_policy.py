from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FFMpegWriter, PillowWriter
from matplotlib.patches import Circle

from cfm_mppi.evaluation.eval_utils import CFMConfig, synthesize_control
from cfm_mppi.models.transformer import TransformerModel
from cfm_mppi.mppi.flowmppi import FlowMPPI
from cfm_mppi.mppi.utils import (
    doubleintegrator_dynamics,
    stage_cost,
    terminal_cost,
    unicycle_dynamics,
)
from cfm_mppi.utils import AgentHistory, HumanAgent


ODE_TIMES = [0.5, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]
ODE_TIMES_WARM = [0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]
SAFE_COEF = [0.1, 0.3, 0.5, 0.7, 0.9]
GOAL_COEF = 0.1
SCALE = 10.0
DT = 0.1


@dataclass
class EpisodeRollout:
    episode: int
    states: np.ndarray
    controls: np.ndarray
    pedestrians: np.ndarray
    pedestrian_velocities: np.ndarray
    planned_xy: List[np.ndarray]
    planning_times: np.ndarray
    goal: np.ndarray
    metrics: dict


def _load_kazuki_model(checkpoint: Path, device: torch.device) -> TransformerModel:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing Kazuki checkpoint: {checkpoint}")
    model = TransformerModel().to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def _make_solver(
    *,
    dynamics: str,
    goal: torch.Tensor,
    horizon: int,
    num_samples: int,
    device: torch.device,
) -> FlowMPPI:
    if dynamics == "unicycle":
        dim_state = 3
        dynamics_fn = unicycle_dynamics
        sigma = torch.tensor([0.3, 0.6], dtype=torch.float32)
    else:
        dim_state = 4
        dynamics_fn = doubleintegrator_dynamics
        sigma = torch.tensor([0.4, 0.4], dtype=torch.float32)
    return FlowMPPI(
        num_samples=num_samples,
        dim_state=dim_state,
        dim_control=2,
        dynamics=dynamics_fn,
        stage_cost=stage_cost,
        terminal_cost=terminal_cost,
        u_min=torch.tensor([-2.0, -2.0], dtype=torch.float32),
        u_max=torch.tensor([2.0, 2.0], dtype=torch.float32),
        sigmas=sigma,
        lambda_=0.1,
        goal=goal.squeeze(0),
        horizon=horizon,
        dt=DT,
        device=device,
        dynamics_type=dynamics,
    )


def _dynamics_step_np(state: np.ndarray, action: np.ndarray, dynamics: str) -> np.ndarray:
    x = state.astype(np.float32).copy()
    a = action.astype(np.float32)
    if dynamics == "unicycle":
        x[0] = x[0] + DT * a[0] * np.cos(x[2])
        x[1] = x[1] + DT * a[0] * np.sin(x[2])
        x[2] = np.arctan2(np.sin(x[2] + DT * a[1]), np.cos(x[2] + DT * a[1]))
        return x
    x[0] = x[0] + x[2] * DT
    x[1] = x[1] + x[3] * DT
    x[2] = x[2] + a[0] * DT
    x[3] = x[3] + a[1] * DT
    return x


def _rollout_plan_xy(
    state: np.ndarray,
    controls_dyn: torch.Tensor,
    *,
    dynamics: str,
    max_steps: int,
) -> np.ndarray:
    seq = controls_dyn[0].detach().cpu().numpy().T
    x = state.astype(np.float32).copy()
    xy = [x[:2].copy()]
    for action in seq[:max_steps]:
        x = _dynamics_step_np(x, action, dynamics)
        xy.append(x[:2].copy())
    return np.asarray(xy, dtype=np.float32)


def _collect_humans(humans: List[HumanAgent]) -> tuple[np.ndarray, np.ndarray]:
    xy = np.asarray([h.state for h in humans], dtype=np.float32)
    vel = np.asarray([h.control for h in humans], dtype=np.float32)
    return xy, vel


def _advance_humans(
    humans: List[HumanAgent],
    *,
    robot_xy: np.ndarray,
    robot_control_si: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    xy, vel = _collect_humans(humans)
    for i, human in enumerate(humans):
        others_xy = np.vstack([xy[:i], xy[i + 1 :], robot_xy.reshape(1, 2)])
        others_vel = np.vstack([vel[:i], vel[i + 1 :], robot_control_si.reshape(1, 2)])
        human.social_force_step(others_xy, others_vel)
    return _collect_humans(humans)


def _episode_metrics(
    states: np.ndarray,
    controls: np.ndarray,
    pedestrians: np.ndarray,
    goal: np.ndarray,
    *,
    safe_margin: float,
    planning_times: np.ndarray,
) -> dict:
    robot_xy = states[:, :2]
    frames = min(robot_xy.shape[0], pedestrians.shape[0])
    clearances = []
    for i in range(frames):
        d = np.linalg.norm(pedestrians[i] - robot_xy[i][None, :], axis=1) - safe_margin
        clearances.append(float(np.min(d)) if d.size else float("inf"))
    clearances_np = np.asarray(clearances, dtype=np.float32)
    final_goal_distance = float(np.linalg.norm(robot_xy[-1] - goal))
    path_length = float(np.linalg.norm(np.diff(robot_xy, axis=0), axis=1).sum())
    return {
        "success": bool(final_goal_distance <= 0.5 and float(np.min(clearances_np)) >= 0.0),
        "collision": bool(float(np.min(clearances_np)) < 0.0),
        "min_clearance": float(np.min(clearances_np)),
        "mean_clearance": float(np.mean(clearances_np[np.isfinite(clearances_np)])),
        "final_goal_distance": final_goal_distance,
        "path_length": path_length,
        "control_effort": float(np.sum(controls**2)) if controls.size else 0.0,
        "planning_wall_time_mean": float(np.mean(planning_times)) if planning_times.size else 0.0,
        "planning_wall_time_p95": float(np.percentile(planning_times, 95)) if planning_times.size else 0.0,
    }


def rollout_episode(
    *,
    model: TransformerModel,
    episode: int,
    seed: int,
    dynamics: str,
    steps: int,
    planning_horizon: int,
    num_samples: int,
    num_pedestrians: int,
    safe_margin: float,
    plan_viz_steps: int,
    device: torch.device,
) -> EpisodeRollout:
    state_dim = 3 if dynamics == "unicycle" else 4
    state_t = torch.zeros(1, state_dim, dtype=torch.float32, device=device)
    goal_t = torch.tensor([[6.0, 6.0]], dtype=torch.float32, device=device)
    goal_np = goal_t.squeeze(0).detach().cpu().numpy()

    rng = np.random.RandomState(int(seed) + int(episode))
    humans = [HumanAgent(goal_np, random_generator=rng) for _ in range(num_pedestrians)]
    solver = _make_solver(
        dynamics=dynamics,
        goal=goal_t,
        horizon=planning_horizon,
        num_samples=num_samples,
        device=device,
    )

    histories = {
        "ego_state": AgentHistory(max_length=10),
        "ego_control_sin": AgentHistory(max_length=10),
        "obs_state": AgentHistory(max_length=10),
        "obs_control": AgentHistory(max_length=10),
    }

    x_t = torch.randn(num_samples, 2, planning_horizon, dtype=torch.float32, device=device)
    noise_level_value = torch.tensor([0.8], dtype=torch.float32, device=device)

    states = [state_t.squeeze(0).detach().cpu().numpy().copy()]
    controls = []
    ped_xy, ped_vel = _collect_humans(humans)
    pedestrians = [ped_xy.copy()]
    pedestrian_velocities = [ped_vel.copy()]
    planned_xy: List[np.ndarray] = []
    planning_times = []

    for step in range(steps):
        pos_obs = torch.tensor(ped_xy, dtype=torch.float32, device=device).view(1, -1, 2)
        vel_obs = torch.tensor(ped_vel, dtype=torch.float32, device=device).view(1, -1, 2)

        if step == 0:
            t_curr = torch.tensor([0.0], dtype=torch.float32, device=device)
            ode_times = ODE_TIMES
            current_horizon = planning_horizon
        else:
            t_curr = noise_level_value
            ode_times = ODE_TIMES_WARM
            current_horizon = x_t.shape[-1]

        config = CFMConfig(
            ode_times=ode_times,
            dt=DT,
            agent_radius=safe_margin,
            space_scale=SCALE,
            safe_margin_coefs=SAFE_COEF,
            goal_margin_coef=GOAL_COEF,
            device=str(device),
        )

        start = time.perf_counter()
        controls_dyn, controls_sin = synthesize_control(
            model,
            solver,
            config,
            state_t,
            goal_t,
            x_t,
            t_curr,
            pos_obs,
            vel_obs,
            current_horizon,
            histories=histories,
            d=0.1,
            k_p=3.0,
        )
        planning_times.append(time.perf_counter() - start)

        state_np = state_t.squeeze(0).detach().cpu().numpy()
        planned_xy.append(
            _rollout_plan_xy(
                state_np,
                controls_dyn,
                dynamics=dynamics,
                max_steps=min(plan_viz_steps, controls_dyn.shape[-1]),
            )
        )

        action_t = controls_dyn[:, :, 0]
        if dynamics == "unicycle":
            state_t = unicycle_dynamics(state_t, action_t, DT)
        else:
            state_t = doubleintegrator_dynamics(state_t, action_t, DT)
        controls.append(action_t.squeeze(0).detach().cpu().numpy().copy())

        control_history_len = len(histories["ego_control_sin"])
        noise = torch.randn(
            num_samples,
            controls_sin.shape[1],
            controls_sin.shape[2],
            dtype=torch.float32,
            device=device,
        )
        x_t = noise_level_value * controls_sin / SCALE + (1.0 - noise_level_value) * noise
        x_t = x_t[:, :, (control_history_len + 1) :]

        histories["ego_control_sin"].update(controls_sin[:, :, control_history_len])
        histories["ego_state"].update(state_t)
        histories["obs_state"].update(pos_obs)
        histories["obs_control"].update(vel_obs)

        control_history_sin = histories["ego_control_sin"].get()
        x_t = torch.cat([control_history_sin.expand(num_samples, -1, -1) / SCALE, x_t], dim=-1)

        robot_control_si = control_history_sin[0, :, -1].detach().cpu().numpy()
        ped_xy, ped_vel = _advance_humans(
            humans,
            robot_xy=state_t[0, :2].detach().cpu().numpy(),
            robot_control_si=robot_control_si,
        )
        states.append(state_t.squeeze(0).detach().cpu().numpy().copy())
        pedestrians.append(ped_xy.copy())
        pedestrian_velocities.append(ped_vel.copy())

    states_np = np.asarray(states, dtype=np.float32)
    controls_np = np.asarray(controls, dtype=np.float32)
    pedestrians_np = np.asarray(pedestrians, dtype=np.float32)
    ped_vel_np = np.asarray(pedestrian_velocities, dtype=np.float32)
    planning_np = np.asarray(planning_times, dtype=np.float32)
    metrics = _episode_metrics(
        states_np,
        controls_np,
        pedestrians_np,
        goal_np,
        safe_margin=safe_margin,
        planning_times=planning_np,
    )
    return EpisodeRollout(
        episode=episode,
        states=states_np,
        controls=controls_np,
        pedestrians=pedestrians_np,
        pedestrian_velocities=ped_vel_np,
        planned_xy=planned_xy,
        planning_times=planning_np,
        goal=goal_np,
        metrics=metrics,
    )


def _limits(run: EpisodeRollout, safe_margin: float) -> tuple[float, float, float, float]:
    pts = [run.states[:, :2], run.pedestrians.reshape(-1, 2), run.goal.reshape(1, 2), np.zeros((1, 2))]
    cloud = np.vstack(pts)
    finite = np.isfinite(cloud).all(axis=1)
    cloud = cloud[finite]
    pad = max(1.0, safe_margin + 0.75)
    xmin, ymin = cloud.min(axis=0) - pad
    xmax, ymax = cloud.max(axis=0) + pad
    span = max(xmax - xmin, ymax - ymin, 1.0)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    half = 0.5 * span
    return cx - half, cx + half, cy - half, cy + half


def _draw_frame(
    axes: np.ndarray,
    runs: List[EpisodeRollout],
    frame: int,
    *,
    safe_margin: float,
    limits: List[tuple[float, float, float, float]],
) -> None:
    for ax, run, lim in zip(axes.flat, runs, limits):
        ax.clear()
        f = min(frame, run.states.shape[0] - 1)
        peds = run.pedestrians[f]
        state_xy = run.states[f, :2]
        ax.set_xlim(lim[0], lim[1])
        ax.set_ylim(lim[2], lim[3])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="#e5e7eb", linewidth=0.6)
        ax.set_facecolor("#fbfbf8")

        trail_start = max(0, f - 24)
        ax.plot(
            run.states[: f + 1, 0],
            run.states[: f + 1, 1],
            color="#d62728",
            linewidth=2.2,
            label="robot",
            zorder=5,
        )
        if f < len(run.planned_xy):
            plan = run.planned_xy[f]
            ax.plot(plan[:, 0], plan[:, 1], color="#ef4444", alpha=0.35, linewidth=1.6, zorder=4)

        for i in range(peds.shape[0]):
            trail = run.pedestrians[trail_start : f + 1, i, :]
            ax.plot(trail[:, 0], trail[:, 1], color="#6b46c1", alpha=0.22, linewidth=0.8, zorder=1)
            ax.add_patch(
                Circle(
                    tuple(peds[i]),
                    safe_margin,
                    facecolor="#8b5cf6",
                    edgecolor="#5b21b6",
                    linewidth=0.7,
                    alpha=0.16,
                    zorder=2,
                )
            )
        ax.scatter(peds[:, 0], peds[:, 1], s=18, color="#5b21b6", alpha=0.85, zorder=3)
        ax.scatter([state_xy[0]], [state_xy[1]], s=48, color="#dc2626", edgecolor="white", linewidth=0.9, zorder=6)
        ax.scatter([run.goal[0]], [run.goal[1]], marker="*", s=130, color="#2563eb", edgecolor="white", linewidth=0.8, zorder=6)
        ax.scatter([0.0], [0.0], marker="o", s=28, color="#111827", alpha=0.45, zorder=4)

        m = run.metrics
        status = "success" if m["success"] else ("collision" if m["collision"] else "running")
        ax.set_title(
            f"SFM ep {run.episode} | {status} | clr {m['min_clearance']:.2f} | goal {m['final_goal_distance']:.2f}\n"
            f"Kazuki CFM-MPPI, frame {f:03d}",
            fontsize=9,
        )

    for ax in axes.flat[len(runs) :]:
        ax.axis("off")


def render_video(
    runs: List[EpisodeRollout],
    *,
    output: Path,
    gif_output: Optional[Path],
    fps: int,
    frame_stride: int,
    safe_margin: float,
    dpi: int,
    cols: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if gif_output is not None:
        gif_output.parent.mkdir(parents=True, exist_ok=True)
    cols = max(1, min(cols, len(runs)))
    rows = int(np.ceil(len(runs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.6 * cols, 5.2 * rows), squeeze=False)
    limits = [_limits(run, safe_margin) for run in runs]
    max_frames = max(run.states.shape[0] for run in runs)
    frames = list(range(0, max_frames, max(1, frame_stride)))
    if frames[-1] != max_frames - 1:
        frames.append(max_frames - 1)

    writer = FFMpegWriter(fps=fps, metadata={"title": "Kazuki CFM-MPPI on SFM crowd"})
    with writer.saving(fig, str(output), dpi=dpi):
        for frame in frames:
            _draw_frame(axes, runs, frame, safe_margin=safe_margin, limits=limits)
            fig.tight_layout()
            writer.grab_frame()

    if gif_output is not None:
        gif_writer = PillowWriter(fps=fps)
        with gif_writer.saving(fig, str(gif_output), dpi=dpi):
            for frame in frames:
                _draw_frame(axes, runs, frame, safe_margin=safe_margin, limits=limits)
                fig.tight_layout()
                gif_writer.grab_frame()
    plt.close(fig)


def _write_summary(path: Path, args: argparse.Namespace, runs: List[EpisodeRollout]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    args_json = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    payload = {
        "args": args_json,
        "episodes": [
            {
                "episode": run.episode,
                **run.metrics,
                "planning_wall_time_ms": float(1000.0 * run.metrics["planning_wall_time_mean"]),
                "steps": int(run.controls.shape[0]),
                "num_pedestrians": int(run.pedestrians.shape[1]),
            }
            for run in runs
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Render Kazuki Mizuta's CFM-MPPI policy in an online SFM crowd without editing original eval files."
    )
    p.add_argument("--episodes", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dynamics", choices=["doubleintegrator", "unicycle"], default="doubleintegrator")
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--planning-horizon", type=int, default=80)
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--num-pedestrians", type=int, default=20)
    p.add_argument("--safe-margin", type=float, default=0.5)
    p.add_argument("--plan-viz-steps", type=int, default=18)
    p.add_argument("--checkpoint", type=Path, default=Path("output_dir/cfm_transformer/checkpoint.pth"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=Path, default=Path("results/benchmark_videos/kazuki_sfm_policy.mp4"))
    p.add_argument("--gif-output", type=Path, default=None)
    p.add_argument("--summary", type=Path, default=Path("results/benchmark_videos/kazuki_sfm_policy_summary.json"))
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--cols", type=int, default=3)
    return p


def main() -> None:
    args = get_parser().parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = _load_kazuki_model(args.checkpoint, device)
    runs: List[EpisodeRollout] = []
    for episode in args.episodes:
        t0 = time.time()
        run = rollout_episode(
            model=model,
            episode=int(episode),
            seed=args.seed,
            dynamics=args.dynamics,
            steps=args.steps,
            planning_horizon=args.planning_horizon,
            num_samples=args.num_samples,
            num_pedestrians=args.num_pedestrians,
            safe_margin=args.safe_margin,
            plan_viz_steps=args.plan_viz_steps,
            device=device,
        )
        runs.append(run)
        print(
            f"episode {episode}: success={int(run.metrics['success'])} "
            f"collision={int(run.metrics['collision'])} "
            f"min_clearance={run.metrics['min_clearance']:.3f} "
            f"goal={run.metrics['final_goal_distance']:.3f} "
            f"sim_time={time.time() - t0:.1f}s",
            flush=True,
        )
    render_video(
        runs,
        output=args.output,
        gif_output=args.gif_output,
        fps=args.fps,
        frame_stride=args.frame_stride,
        safe_margin=args.safe_margin,
        dpi=args.dpi,
        cols=args.cols,
    )
    _write_summary(args.summary, args, runs)
    print(f"wrote {args.output}", flush=True)
    if args.gif_output is not None:
        print(f"wrote {args.gif_output}", flush=True)
    print(f"wrote {args.summary}", flush=True)


if __name__ == "__main__":
    main()
