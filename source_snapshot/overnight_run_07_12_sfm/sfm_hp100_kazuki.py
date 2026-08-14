"""Locked Kazuki generate--guide--refine comparator for SFM HP100.

This is an additive comparator: it uses the same HP100 checkpoint and scene as
the raw policy, but applies the existing Kazuki ODE reward guidance and MPPI
refinement with the historically locked coefficients.  No shield, template,
privileged simulator lookahead, or fallback is enabled.  Unlike the legacy
Hp10 comparator, every robot rollout and executed transition obeys the HP100
componentwise acceleration and velocity caps.
"""
from __future__ import annotations

import numpy as np
import torch

import _paths  # noqa: F401
import sfm_hp100_dynamics as DYN
import sfm_hp100_features as HPF
import sfm_hp100_history as HPH
import sfm_kazuki as BASE
import sfm_scene as SS


VERSION = "sfm_hp100_kazuki_locked_clipped_v1"
SAFE_COEF = 0.3
GOAL_COEF = 0.5
SAMPLE_SEED = 700_000

# Every Kazuki knob is listed here intentionally.  The HP100 comparator must
# not change when a default in the much broader legacy controller evolves.
# ``locked_config`` also checks the dataclass schema so a newly added default
# fails closed until it is reviewed and pinned here.
LOCKED_CONFIG_ITEMS = (
    ("ode_times", (0.0, 0.5, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0)),
    ("warm_s", 0.8),
    ("safe_coefs", (SAFE_COEF,)),
    ("goal_coef", GOAL_COEF),
    ("safe_coef_gamma_span", 0.0),
    ("goal_coef_gamma_span", 0.0),
    ("a_cbf", 1.0),
    ("k_worst", 5),
    ("markup", 1.01),
    ("refinement_cost", "b1_safemppi"),
    ("collision_weight", 20.0),
    ("goal_weight", 2.0),
    ("beta_mppi", 20.0),
    ("n_sample", 200),
    ("n_elite", 10),
    ("n_copy", 200),
    ("mppi_lambda", 0.1),
    ("mppi_sigma", 0.4),
    ("warm_consistency_weight", 0.1),
    ("collision_margin", 0.05),
    ("hard_clearance_select", False),
    ("refined_clearance_margin", 0.0),
    ("exact_sfm_step_filter", False),
    ("step_filter_margin", 0.03),
    ("step_filter_gamma_margin_span", 0.0),
    ("step_filter_horizon", 10),
    ("step_filter_goal_plans", 0),
    ("step_filter_avoid_plans", 0),
    ("step_filter_always_select", False),
    ("step_filter_min_progress", 0.0),
    ("step_filter_goal_score_weight", 1.0),
    ("step_filter_clearance_weight", 0.05),
    ("step_filter_gamma_clearance_target_span", 0.0),
    ("step_filter_clearance_target_weight", 0.0),
    ("step_filter_escape_patience", 0),
    ("step_filter_escape_burst", 0),
    ("step_filter_fallback_lookahead", 0),
    ("step_filter_viability_lookahead", 0),
    ("step_filter_viability_band", 0.05),
    ("step_filter_viability_goal_weight", 0.0),
    ("step_filter_viability_escalate", False),
    ("step_filter_viability_escalation_band", 0.05),
    ("step_filter_viability_escalation_min_progress", 0.0),
    ("step_filter_viability_escalation_entry_progress", 0.0),
    ("step_filter_viability_escalation_burst", 0),
    ("step_filter_release_steps", 0),
    ("step_filter_stagnation_gamma_max", 0.0),
    ("step_filter_stagnation_window", 0),
    ("step_filter_stagnation_progress", 0.0),
    ("step_filter_stagnation_horizon", 0),
    ("step_filter_stagnation_burst", 0),
    ("output_filter", False),
    ("filter_eta", 0.6),
    ("filter_margin", 0.05),
    ("filter_iters", 5),
    ("filter_solver", "jacobi"),
    ("controller_gammas", ()),
    ("safe_coef_by_gamma", ()),
    ("goal_coef_by_gamma", ()),
    ("step_filter_margin_by_gamma", ()),
    ("step_filter_goal_score_weight_by_gamma", ()),
    ("step_filter_clearance_weight_by_gamma", ()),
    ("step_filter_clearance_target_weight_by_gamma", ()),
)


def locked_config() -> BASE.KazukiConfig:
    """Return and audit the immutable, non-privileged comparator recipe."""
    expected = dict(LOCKED_CONFIG_ITEMS)
    declared_fields = tuple(BASE.KazukiConfig.__dataclass_fields__)
    if tuple(expected) != declared_fields:
        raise RuntimeError(
            "KazukiConfig schema changed; review and fully repin the HP100 comparator: "
            f"locked={tuple(expected)}, current={declared_fields}"
        )
    config = BASE.KazukiConfig(**expected).validate()
    if config.to_dict() != expected:
        raise RuntimeError("locked HP100 Kazuki config does not match its declared contract")
    return config


def clipped_di_rollout_t(state, controls, dt=DYN.DT):
    """Differentiable batched rollout under the exact HP100 cap ordering."""
    if controls.ndim != 3 or controls.shape[-1] != 2:
        raise ValueError(f"expected controls [B,H,2], got {tuple(controls.shape)}")
    batch, horizon, _ = controls.shape
    current = torch.as_tensor(
        state, dtype=controls.dtype, device=controls.device,
    ).reshape(1, 4).expand(batch, 4).clone()
    positions, velocities = [], []
    for step in range(horizon):
        current = DYN.step_torch(current, controls[:, step], dt=float(dt))
        positions.append(current[:, :2])
        velocities.append(current[:, 2:4])
    return torch.stack(positions, dim=1), torch.stack(velocities, dim=1)


def _obstacles(pedestrian_xy) -> np.ndarray:
    pedestrian_xy = np.asarray(pedestrian_xy, np.float32).reshape(-1, 2)
    return np.concatenate(
        [pedestrian_xy, np.full((len(pedestrian_xy), 1), SS.R_PED, np.float32)],
        axis=1,
    )


def _clearance(state, pedestrian_xy) -> float:
    pedestrian_xy = np.asarray(pedestrian_xy, np.float32).reshape(-1, 2)
    if not len(pedestrian_xy):
        return float("inf")
    return float(
        np.linalg.norm(pedestrian_xy - np.asarray(state[:2])[None], axis=1).min()
        - SS.R_PED
    )


def kazuki_hp100_deploy(
    policy,
    episode,
    gamma,
    *,
    scene_profile,
    T=180,
    reach=0.5,
    device="cpu",
    sample_seed=SAMPLE_SEED,
    collect_diagnostics=False,
):
    """Deploy the locked comparator on one deterministic scenario.

    ``scene_profile`` is required so the caller cannot silently evaluate a
    different pedestrian bank than the paired raw HP100 policy.
    """
    config = locked_config()
    environment = SS.scene_profile(scene_profile)
    humans = SS.make_humans(
        int(episode), seed=0, n_ped=int(environment["n_ped"]),
        speed_range=tuple(environment["ped_speed_range"]),
    )
    state = np.zeros(4, np.float32)
    goal = torch.as_tensor(SS.GOAL, dtype=torch.float32, device=device)
    horizon, latent_dim = int(policy.H_pred), int(policy.d)
    hp_history = HPH.Hp100History()
    control_history = []
    previous_latent = previous_window = None
    states, controls, pedestrian_positions, pedestrian_velocities = [state.copy()], [], [], []
    trace = []
    reached = collision = False
    minimum_clearance = float("inf")

    for step in range(int(T)):
        pedestrian_xy, pedestrian_velocity = SS.collect_humans(humans)
        pedestrian_xy = np.asarray(pedestrian_xy, np.float32)
        pedestrian_velocity = np.asarray(pedestrian_velocity, np.float32)
        clearance = _clearance(state, pedestrian_xy)
        minimum_clearance = min(minimum_clearance, clearance)
        if clearance < 0.0:
            collision = True
            break
        if float(np.linalg.norm(state[:2] - SS.GOAL)) < float(reach):
            reached = True
            break

        hp_frame = HPF.hp100_frame(
            state[:2], _obstacles(pedestrian_xy), sensing=SS.R_SENSE,
            n_base=HPF.POLYTOPE_N_BASE,
            obstacle_velocities=pedestrian_velocity,
            robot_velocity=state[2:4],
            predict_gain=HPF.PREDICT_GAIN,
            predict_tau=HPF.PREDICT_TAU,
        )
        hp_tensor = hp_history.append(hp_frame).to(device)
        low_tensor = torch.as_tensor(
            HPF.low5(state, SS.GOAL, gamma), device=device,
        )
        history_tensor = torch.as_tensor(
            HPF.hist_pad(control_history[-HPF.K_HIST:]), device=device,
        )
        context = policy.ctx_from(
            hp_tensor[None], low_tensor[None], history_tensor[None],
        ).squeeze(0)

        torch.manual_seed(int(sample_seed) + int(episode) * 1000 + step)
        if previous_latent is None:
            latent = torch.randn(config.n_sample, latent_dim, device=device)
            ode_times = config.ode_times
        else:
            latent = float(config.warm_s) * previous_latent[None].expand(
                config.n_sample, latent_dim,
            ) + (1.0 - float(config.warm_s)) * torch.randn(
                config.n_sample, latent_dim, device=device,
            )
            ode_times = tuple(
                value for value in config.ode_times
                if value >= float(config.warm_s) - 1.0e-12
            )

        pedestrian_prediction = BASE.predict_pedestrians_t(
            pedestrian_xy, pedestrian_velocity, horizon, DYN.DT,
            device, latent.dtype,
        )
        pedestrian_velocity_tensor = torch.as_tensor(
            pedestrian_velocity, dtype=latent.dtype, device=device,
        )
        generated, guidance_diagnostics, unguided, _guidance_components = BASE.guided_generate(
            policy, context, state, goal, pedestrian_prediction,
            pedestrian_velocity_tensor, SS.R_PED + config.collision_margin,
            latent, ode_times, config,
            collect_diagnostics=collect_diagnostics,
            rollout_fn=clipped_di_rollout_t,
        )
        generated_windows = DYN.clip_action_torch(
            generated.reshape(config.n_sample, horizon, 2) * float(policy.u_max)
        )
        selected_window, refine_diagnostics = BASE.flow_mppi_refine(
            policy, state, goal, pedestrian_xy, pedestrian_velocity_tensor,
            pedestrian_prediction,
            SS.R_PED + config.collision_margin, generated_windows,
            previous_window, config, collect_diagnostics=collect_diagnostics,
            rollout_fn=clipped_di_rollout_t,
        )
        action = DYN.clip_action_numpy(
            selected_window[0].detach().cpu().numpy()
        ).astype(np.float32, copy=False)

        pedestrian_positions.append(pedestrian_xy.copy())
        pedestrian_velocities.append(pedestrian_velocity.copy())
        controls.append(action.copy())
        control_history.append(action.copy())
        previous_state = state.copy()
        state = DYN.step_numpy(state, action).astype(np.float32, copy=False)
        states.append(state.copy())
        SS.advance_humans(humans, state)

        if collect_diagnostics:
            selected_positions, _ = clipped_di_rollout_t(
                previous_state, selected_window[None], DYN.DT,
            )
            trace.append(dict(
                step=int(step), state=previous_state, action=action.copy(),
                pedestrian_xy=pedestrian_xy.copy(),
                pedestrian_velocity=pedestrian_velocity.copy(),
                guidance=guidance_diagnostics, refinement=refine_diagnostics,
                selected_plan_positions=np.concatenate([
                    previous_state[:2][None],
                    selected_positions[0].detach().cpu().numpy(),
                ], axis=0).astype(np.float32),
                unguided_available=unguided is not None,
            ))

        shifted = torch.cat([selected_window[1:], selected_window[-1:]], dim=0)
        previous_window = shifted.detach()
        previous_latent = (
            shifted / float(policy.u_max)
        ).reshape(-1).detach()

    if not reached and not collision:
        pedestrian_xy, _ = SS.collect_humans(humans)
        clearance = _clearance(state, pedestrian_xy)
        minimum_clearance = min(minimum_clearance, clearance)
        collision = clearance < 0.0
        reached = (
            not collision
            and float(np.linalg.norm(state[:2] - SS.GOAL)) < float(reach)
        )

    return dict(
        states=np.asarray(states, np.float32),
        controls=np.asarray(controls, np.float32).reshape(-1, 2),
        peds=np.asarray(pedestrian_positions, np.float32),
        ped_vels=np.asarray(pedestrian_velocities, np.float32),
        path=np.asarray(states, np.float32)[:, :2],
        success=bool(reached and not collision), collision=bool(collision),
        reached=bool(reached), steps=len(controls),
        min_clear=float(minimum_clearance), gamma=float(gamma),
        episode=int(episode), scene=environment,
        config=config.to_dict(), dynamics=DYN.contract(),
        trace=trace if collect_diagnostics else None,
    )
