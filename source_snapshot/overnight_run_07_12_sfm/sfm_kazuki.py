"""Claude-style Kazuki/Mizuta generate--guide--refine baseline on the SAME SFM learned prior.

The checkpoint is never modified.  Its gamma-conditioned velocity field supplies the nominal generative policy;
reward gradients are added at the ODE integration knots, then the generated modes are refined by MPPI.  Warm
starts resume at exactly s=0.8.  Moving pedestrians are predicted at constant velocity over the H=10 guidance
horizon, while execution and collision checking use the same SFM crowd and local ``di_step`` as every SFM arm.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import torch

import grid_feats as GF
import sfm_features as SF
import sfm_hp_history as HH
import sfm_b1_cost as BC
import sfm_scene as SS
from di_grid_viz import di_step
from cfm_mppi.evaluation.render_sfm_kazuki_policy import _advance_humans, _collect_humans
from cfm_mppi.safegpc_adapter.safemppi import SafeMPPIAdapter
from cfm_mppi.safegpc_adapter.barrier import affine_barrier_h_ho_all


@dataclass(frozen=True)
class KazukiConfig:
    # Original UnifiedGenRefine knot schedule, with the user-locked exact warm-start knot s=.8.
    ode_times: tuple = (0.0, 0.5, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0)
    warm_s: float = 0.8
    safe_coefs: tuple = (0.3,)
    goal_coef: float = 0.5
    # Optional learned-guidance schedule. Effective safety is ``coef + safe_span*(1-gamma)`` and effective
    # goal guidance is ``goal_coef + goal_span*gamma``. Zero spans preserve the original Kazuki adaptation.
    safe_coef_gamma_span: float = 0.0
    goal_coef_gamma_span: float = 0.0
    a_cbf: float = 1.0
    k_worst: int = 5
    markup: float = 1.01
    # Keep Kazuki's generate--guide--refine mechanics, but score every generated,
    # perturbed, and refined window with the same frozen proposal cost as B1.
    # ``legacy_kazuki`` remains available only to reproduce pre-match artifacts.
    refinement_cost: str = "b1_safemppi"
    # Claude's H=10 adaptation of the original MPPI cost scaling.
    collision_weight: float = 20.0
    goal_weight: float = 2.0
    beta_mppi: float = 20.0
    n_sample: int = 200
    n_elite: int = 10
    n_copy: int = 200
    mppi_lambda: float = 0.1
    mppi_sigma: float = 0.4  # original absolute DI noise for u_max=2
    warm_consistency_weight: float = 0.1
    collision_margin: float = 0.05
    # Opt-in hard selection over the already-refined modes.  The locked Kazuki comparator leaves this disabled.
    hard_clearance_select: bool = False
    refined_clearance_margin: float = 0.0
    # Candidate-specific one-step SFM dynamics shield.  This is an opt-in expanded-policy method; the frozen
    # Kazuki comparator never sees simulator-state prediction or action replacement.
    exact_sfm_step_filter: bool = False
    step_filter_margin: float = 0.03
    # Optional adaptive margin: ``margin + span * (1-gamma)``.  The base is therefore the gamma=1 margin,
    # while lower gamma receives a larger hard clearance target.  Zero exactly preserves the fixed-margin
    # controller and the frozen Kazuki comparator never enables this expanded-policy option.
    step_filter_gamma_margin_span: float = 0.0
    step_filter_horizon: int = 10
    step_filter_goal_plans: int = 0
    step_filter_avoid_plans: int = 0
    step_filter_always_select: bool = False
    step_filter_min_progress: float = 0.0
    step_filter_goal_score_weight: float = 1.0
    step_filter_clearance_weight: float = 0.05
    # Optional ranking-only adaptive risk target. Unlike ``gamma_margin_span`` this leaves the hard feasible
    # set unchanged: low gamma can prefer wider clearance and high gamma a direct margin-near plan.
    step_filter_gamma_clearance_target_span: float = 0.0
    step_filter_clearance_target_weight: float = 0.0
    # Optional bounded escape hysteresis for repeated no-progress replans. Zero preserves the stateless recipe.
    step_filter_escape_patience: int = 0
    step_filter_escape_burst: int = 0
    step_filter_fallback_lookahead: int = 0
    step_filter_viability_lookahead: int = 0
    step_filter_viability_band: float = 0.05
    step_filter_viability_goal_weight: float = 0.0
    step_filter_viability_escalate: bool = False
    step_filter_viability_escalation_band: float = 0.05
    step_filter_viability_escalation_min_progress: float = 0.0
    step_filter_viability_escalation_entry_progress: float = 0.0
    step_filter_viability_escalation_burst: int = 0
    # Optional initial yielding interval. A hold is executed only after the same exact-SFM horizon check passes;
    # otherwise deployment falls through to the normally selected safe action.
    step_filter_release_steps: int = 0
    # Low-gamma executed-progress watchdog. All-zero values disable it. When enabled, measured stagnation
    # temporarily regenerates the complete longer-horizon candidate family, then returns to H=10.
    step_filter_stagnation_gamma_max: float = 0.0
    step_filter_stagnation_window: int = 0
    step_filter_stagnation_progress: float = 0.0
    step_filter_stagnation_horizon: int = 0
    step_filter_stagnation_burst: int = 0
    # Optional post-refinement shield for the expanded-policy arm.  The frozen Kazuki comparator keeps this
    # False; enabling it is a separately named method, never a silent change to the baseline.
    output_filter: bool = False
    filter_eta: float = 0.6
    filter_margin: float = 0.05
    filter_iters: int = 5
    filter_solver: str = "jacobi"
    # Optional explicit operating table for the locked gamma list. Empty tuples preserve scalar configuration;
    # when populated, every requested gamma must have one row and only the listed fields are overridden.
    controller_gammas: tuple = ()
    safe_coef_by_gamma: tuple = ()
    goal_coef_by_gamma: tuple = ()
    step_filter_margin_by_gamma: tuple = ()
    step_filter_goal_score_weight_by_gamma: tuple = ()
    step_filter_clearance_weight_by_gamma: tuple = ()
    step_filter_clearance_target_weight_by_gamma: tuple = ()

    def validate(self):
        times = tuple(map(float, self.ode_times))
        if times[0] != 0.0 or times[-1] != 1.0 or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError(f"invalid ODE knots: {times}")
        if not any(abs(x - self.warm_s) < 1e-10 for x in times):
            raise ValueError(f"warm s={self.warm_s} must be an exact ODE knot")
        if not self.safe_coefs or any(x < 0 for x in self.safe_coefs):
            raise ValueError("safe_coefs must be nonnegative")
        if self.safe_coef_gamma_span < 0 or self.goal_coef_gamma_span < 0:
            raise ValueError("gamma guidance spans must be nonnegative")
        if self.refinement_cost not in {"b1_safemppi", "legacy_kazuki"}:
            raise ValueError(f"unknown refinement cost: {self.refinement_cost}")
        if self.n_sample < 1 or self.n_elite < 1 or self.n_copy < 1:
            raise ValueError("sample/elite/copy counts must be positive")
        if self.filter_solver not in {"jacobi", "exact"}:
            raise ValueError(f"unknown output-filter solver: {self.filter_solver}")
        if self.refined_clearance_margin < 0:
            raise ValueError("refined_clearance_margin must be nonnegative")
        if self.step_filter_margin < 0:
            raise ValueError("step_filter_margin must be nonnegative")
        if self.step_filter_gamma_margin_span < 0:
            raise ValueError("step-filter gamma margin span must be nonnegative")
        if self.step_filter_horizon < 1:
            raise ValueError("step_filter_horizon must be positive")
        if self.step_filter_goal_plans < 0:
            raise ValueError("step_filter_goal_plans must be nonnegative")
        if self.step_filter_avoid_plans < 0:
            raise ValueError("step_filter_avoid_plans must be nonnegative")
        if self.step_filter_min_progress < 0:
            raise ValueError("step_filter_min_progress must be nonnegative")
        if self.step_filter_goal_score_weight <= 0:
            raise ValueError("step-filter goal score weight must be positive")
        if self.step_filter_clearance_weight < 0:
            raise ValueError("step_filter_clearance_weight must be nonnegative")
        if self.step_filter_gamma_clearance_target_span < 0 or self.step_filter_clearance_target_weight < 0:
            raise ValueError("step-filter adaptive clearance target span/weight must be nonnegative")
        if self.step_filter_escape_patience < 0 or self.step_filter_escape_burst < 0:
            raise ValueError("step-filter escape patience/burst must be nonnegative")
        if bool(self.step_filter_escape_patience) != bool(self.step_filter_escape_burst):
            raise ValueError("step-filter escape patience and burst must both be zero or both be positive")
        if self.step_filter_fallback_lookahead < 0:
            raise ValueError("step-filter fallback lookahead must be nonnegative")
        if (self.step_filter_viability_lookahead < 0 or self.step_filter_viability_band < 0
                or self.step_filter_viability_goal_weight < 0
                or self.step_filter_viability_escalation_band < 0
                or self.step_filter_viability_escalation_min_progress < 0
                or self.step_filter_viability_escalation_entry_progress < 0
                or self.step_filter_viability_escalation_burst < 0):
            raise ValueError("step-filter viability lookahead/band/goal weight must be nonnegative")
        if self.step_filter_release_steps < 0:
            raise ValueError("step-filter release steps must be nonnegative")
        if self.step_filter_release_steps and not self.exact_sfm_step_filter:
            raise ValueError("step-filter release steps require the exact SFM step filter")
        stagnation_values = (self.step_filter_stagnation_gamma_max, self.step_filter_stagnation_window,
                             self.step_filter_stagnation_progress, self.step_filter_stagnation_horizon,
                             self.step_filter_stagnation_burst)
        if any(float(x) < 0 for x in stagnation_values):
            raise ValueError("step-filter stagnation settings must be nonnegative")
        if any(stagnation_values) and not all(float(x) > 0 for x in stagnation_values):
            raise ValueError("all step-filter stagnation settings must be positive when enabled")
        if self.step_filter_stagnation_horizon and self.step_filter_stagnation_horizon < self.step_filter_horizon:
            raise ValueError("stagnation horizon must be at least the ordinary step-filter horizon")
        table_fields = (
            self.safe_coef_by_gamma, self.goal_coef_by_gamma, self.step_filter_margin_by_gamma,
            self.step_filter_goal_score_weight_by_gamma, self.step_filter_clearance_weight_by_gamma,
            self.step_filter_clearance_target_weight_by_gamma)
        if self.controller_gammas:
            n = len(self.controller_gammas)
            if len(set(map(float, self.controller_gammas))) != n:
                raise ValueError("controller gamma table must contain unique gamma values")
            if any(values and len(values) != n for values in table_fields):
                raise ValueError("every populated controller table field must match controller_gammas")
            nonnegative = (self.safe_coef_by_gamma, self.goal_coef_by_gamma,
                           self.step_filter_margin_by_gamma, self.step_filter_clearance_weight_by_gamma,
                           self.step_filter_clearance_target_weight_by_gamma)
            if any(any(float(x) < 0 for x in values) for values in nonnegative if values):
                raise ValueError("controller table coefficients/margins/weights must be nonnegative")
            if (self.step_filter_goal_score_weight_by_gamma
                    and any(float(x) <= 0 for x in self.step_filter_goal_score_weight_by_gamma)):
                raise ValueError("controller-table goal score weights must be positive")
        elif any(table_fields):
            raise ValueError("controller table values require controller_gammas")
        return self

    def to_dict(self):
        return asdict(self)


def di_rollout_t(state, U, dt=SS.DT):
    """Differentiable batched rollout matching ``di_step`` exactly."""
    B, H, _ = U.shape
    p = torch.as_tensor(state[:2], dtype=U.dtype, device=U.device).expand(B, 2).clone()
    v = torch.as_tensor(state[2:], dtype=U.dtype, device=U.device).expand(B, 2).clone()
    ps, vs = [], []
    for t in range(H):
        u = U[:, t]
        p = p + float(dt) * v + 0.5 * float(dt) ** 2 * u
        v = v + float(dt) * u
        ps.append(p); vs.append(v)
    return torch.stack(ps, dim=1), torch.stack(vs, dim=1)


def predict_pedestrians_t(ped_xy, ped_vel, H, dt, device, dtype=torch.float32):
    xy = torch.as_tensor(ped_xy, dtype=dtype, device=device)
    vel = torch.as_tensor(ped_vel, dtype=dtype, device=device)
    tau = torch.arange(1, H + 1, dtype=dtype, device=device)[:, None, None] * float(dt)
    return xy[None] + tau * vel[None]


def _adaptive_step_filter_margin(cfg, gamma):
    """Return the expanded controller's gamma-conditioned hard horizon margin."""
    g = float(np.clip(float(gamma), 0.0, 1.0))
    return float(cfg.step_filter_margin) + float(cfg.step_filter_gamma_margin_span) * (1.0 - g)


def _adaptive_step_filter_clearance_target(cfg, gamma, margin):
    """Preferred clearance for ranking hard-safe candidates; ``None`` disables the extra term."""
    if float(cfg.step_filter_clearance_target_weight) <= 0.0:
        return None
    g = float(np.clip(float(gamma), 0.0, 1.0))
    return float(margin) + float(cfg.step_filter_gamma_clearance_target_span) * (1.0 - g)


def _gamma_guidance_config(cfg, gamma):
    """Materialize the learned ODE-guidance coefficients for one gamma-conditioned episode."""
    g = float(np.clip(float(gamma), 0.0, 1.0))
    return replace(
        cfg,
        safe_coefs=tuple(float(x) + float(cfg.safe_coef_gamma_span) * (1.0 - g)
                         for x in cfg.safe_coefs),
        goal_coef=float(cfg.goal_coef) + float(cfg.goal_coef_gamma_span) * g,
    )


def _gamma_controller_config(cfg, gamma):
    """Apply one exact row of the optional, explicitly reported seven-gamma operating table."""
    if not cfg.controller_gammas:
        return cfg
    matches = [i for i, value in enumerate(cfg.controller_gammas)
               if abs(float(value) - float(gamma)) <= 1e-9]
    if len(matches) != 1:
        raise ValueError(f"gamma={gamma} is not uniquely represented in controller_gammas")
    i = matches[0]
    kwargs = {}
    if cfg.safe_coef_by_gamma:
        kwargs["safe_coefs"] = (float(cfg.safe_coef_by_gamma[i]),)
    for table_name, scalar_name in (
            ("goal_coef_by_gamma", "goal_coef"),
            ("step_filter_margin_by_gamma", "step_filter_margin"),
            ("step_filter_goal_score_weight_by_gamma", "step_filter_goal_score_weight"),
            ("step_filter_clearance_weight_by_gamma", "step_filter_clearance_weight"),
            ("step_filter_clearance_target_weight_by_gamma", "step_filter_clearance_target_weight")):
        values = getattr(cfg, table_name)
        if values:
            kwargs[scalar_name] = float(values[i])
    return replace(cfg, **kwargs)


def interpolated_gamma_controller_config(cfg, gamma):
    """Continuously interpolate the optional operating table for adaptive-gamma deployment.

    The locked fixed-gamma experiments continue to use :func:`_gamma_controller_config`, which requires an
    exact table row.  This helper is intentionally separate: at every listed knot it returns the identical
    row, while values between knots are piecewise-linear.  Values outside the table are clamped to its end
    rows (the adaptive selector itself is normally clipped to ``[0.1, 1.0]``).
    """
    if not cfg.controller_gammas:
        return cfg
    order = np.argsort(np.asarray(cfg.controller_gammas, dtype=float))
    knots = np.asarray(cfg.controller_gammas, dtype=float)[order]
    g = float(np.clip(float(gamma), knots[0], knots[-1]))
    kwargs = {}
    if cfg.safe_coef_by_gamma:
        values = np.asarray(cfg.safe_coef_by_gamma, dtype=float)[order]
        kwargs["safe_coefs"] = (float(np.interp(g, knots, values)),)
    for table_name, scalar_name in (
            ("goal_coef_by_gamma", "goal_coef"),
            ("step_filter_margin_by_gamma", "step_filter_margin"),
            ("step_filter_goal_score_weight_by_gamma", "step_filter_goal_score_weight"),
            ("step_filter_clearance_weight_by_gamma", "step_filter_clearance_weight"),
            ("step_filter_clearance_target_weight_by_gamma", "step_filter_clearance_target_weight")):
        values = getattr(cfg, table_name)
        if values:
            values = np.asarray(values, dtype=float)[order]
            kwargs[scalar_name] = float(np.interp(g, knots, values))
    return replace(cfg, **kwargs)


def cbf_reward(pos, vel, ped_pred, ped_vel, r_col, cfg):
    """Moving-obstacle form of Claude's faithful K-worst CBF guidance reward."""
    d = pos.unsqueeze(2) - ped_pred.unsqueeze(0)                         # [B,H,N,2]
    h = (d ** 2).sum(-1) - float(r_col) ** 2
    rel_v = vel.unsqueeze(2) - ped_vel[None, None]
    hdot = 2.0 * (d * rel_v).sum(-1)
    cbf = torch.clamp(hdot + float(cfg.a_cbf) * h, max=0.0)
    k = min(int(cfg.k_worst), ped_pred.shape[1])
    worst = torch.topk(cbf, k=k, dim=2, largest=False).values
    w = torch.arange(k, 0, -1, dtype=pos.dtype, device=pos.device)[None, None]
    return (worst * w).sum(dim=(1, 2)), cbf


def goal_reward(pos, goal):
    return -torch.linalg.vector_norm(pos[:, -1] - goal[None], dim=1)


def stage_cost_batch(pos, U, goal, ped_pred, r_col, cfg, prev_U=None):
    """Legacy SFM Kazuki port cost, retained only for artifact reproduction."""
    H = pos.shape[1]
    goal_c = torch.linalg.vector_norm(pos - goal[None, None], dim=2)
    d = torch.linalg.vector_norm(pos.unsqueeze(2) - ped_pred.unsqueeze(0), dim=3)
    proximity = torch.clamp(torch.exp(-float(cfg.beta_mppi) * (d - float(r_col))), max=1.0).sum(2)
    time_weight = float(cfg.collision_weight) * (
        1.0 + 0.99 ** torch.arange(H, dtype=pos.dtype, device=pos.device))[None]
    cost = (float(cfg.goal_weight) * goal_c + time_weight * proximity).sum(1)
    cost = cost + float(cfg.goal_weight) * torch.linalg.vector_norm(pos[:, -1] - goal[None], dim=1)
    if prev_U is not None:
        cost = cost + float(cfg.warm_consistency_weight) * ((U - prev_U[None]) ** 2).sum((1, 2))
    return cost


def refinement_cost_batch(state, U, goal, ped_xy, ped_vel, ped_pred, r_col, cfg, prev_U=None):
    """Dispatch the declared MPPI objective without changing Kazuki's proposal mechanics."""
    if cfg.refinement_cost == "b1_safemppi":
        return BC.safemppi_proposal_cost(
            state, U, goal, ped_xy, ped_vel, config=BC.frozen_expert_config()
        )
    pos, _ = di_rollout_t(state, U, SS.DT)
    return stage_cost_batch(pos, U, goal, ped_pred, r_col, cfg, prev_U)


def _select_refined_index(refined_cost, refined_min_clear, cfg):
    """Select the cheapest predicted-safe mode, preserving the legacy argmin when disabled."""
    if not cfg.hard_clearance_select:
        return int(torch.argmin(refined_cost))
    eligible = refined_min_clear >= float(cfg.refined_clearance_margin)
    if bool(eligible.any()):
        masked = torch.where(eligible, refined_cost, torch.full_like(refined_cost, torch.inf))
        return int(torch.argmin(masked))
    best_clear = refined_min_clear.max()
    tied = refined_min_clear >= best_clear - 1e-8
    masked = torch.where(tied, refined_cost, torch.full_like(refined_cost, torch.inf))
    return int(torch.argmin(masked))


def _sample_safe_coefficients(cfg, device, dtype):
    vals = torch.tensor(cfg.safe_coefs, dtype=dtype, device=device)
    # Claude's port used contiguous coefficient groups.  The floor map also allocates a non-divisible remainder
    # deterministically without dropping samples.
    idx = torch.div(torch.arange(cfg.n_sample, device=device) * len(vals), cfg.n_sample,
                    rounding_mode="floor")
    return vals[idx][:, None, None]


def exact_ho_filter_action(state, obstacles, obstacle_velocities, action, gamma, *, eta=.6,
                           margin=.05, activation_radius=SS.R_SENSE):
    """Exact 2-D Euclidean projection onto active affine HO-DCBF half-spaces and the control box."""
    x0 = torch.as_tensor(state, dtype=torch.float32).reshape(1, 4)
    obs = torch.as_tensor(obstacles, dtype=torch.float32).clone()
    vel = torch.as_tensor(obstacle_velocities, dtype=torch.float32)
    obs[:, 2] += float(margin)
    obs_next = obs.clone(); obs_next[:, :2] += SS.DT * vel
    zero = torch.zeros(1, 2, dtype=x0.dtype)
    x1zero = x0.clone()
    x1zero[:, :2] = x0[:, :2] + SS.DT * x0[:, 2:4]
    # zero acceleration leaves velocity unchanged
    h_old, _, active = affine_barrier_h_ho_all(
        x0, x0, obs, vel, float(eta), 0, float(activation_radius))
    h_zero, grad, _ = affine_barrier_h_ho_all(
        x0, x1zero, obs_next, vel, float(eta), 0, float(activation_radius))
    scale = 0.5 * SS.DT * SS.DT + float(eta) * SS.DT
    A = (grad[0] * scale).numpy()
    b = (((1.0 - float(gamma)) * h_old[0] - h_zero[0]).numpy())
    mask = active[0].numpy().astype(bool)
    A, b = A[mask], b[mask]
    # u in [-u_max,u_max]^2, expressed in the same A u >= b convention.
    box_A = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]], float)
    box_b = np.array([-SS.U_MAX, -SS.U_MAX, -SS.U_MAX, -SS.U_MAX], float)
    A = np.concatenate([A, box_A], axis=0); b = np.concatenate([b, box_b], axis=0)
    target = np.asarray(action, float).reshape(2)

    def feasible(u, tol=2e-6):
        return bool(np.all(A @ u >= b - tol))

    candidates = [np.clip(target, -SS.U_MAX, SS.U_MAX)]
    # Orthogonal target projection onto every active boundary.
    for ai, bi in zip(A, b):
        den = float(ai @ ai)
        if den > 1e-12:
            candidates.append(target + max(0.0, float(bi - ai @ target)) / den * ai)
    # Every vertex of a 2-D half-space intersection is an intersection of two active boundaries.
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            M = np.stack([A[i], A[j]])
            det = float(np.linalg.det(M))
            if abs(det) > 1e-10:
                candidates.append(np.linalg.solve(M, np.array([b[i], b[j]], float)))
    good = [u for u in candidates if np.isfinite(u).all() and feasible(u)]
    if good:
        u = min(good, key=lambda z: float(np.sum((z - target) ** 2)))
        feasible_out = True
    else:
        # Infeasible intersection: choose the enumerated bounded point with the best worst constraint slack.
        bounded = [np.clip(u, -SS.U_MAX, SS.U_MAX) for u in candidates if np.isfinite(u).all()]
        u = max(bounded, key=lambda z: float(np.min(A @ z - b)))
        feasible_out = False
    slack = A @ u - b
    return np.asarray(u, np.float32), dict(
        filter_feasible=bool(feasible_out), filter_max_deficit=float(max(0.0, -float(slack.min()))),
        filter_num_active=int(mask.sum()), correction_magnitude=float(np.linalg.norm(u - target)),
        filter_solver="exact")


def exact_sfm_step_filter_action(humans, state, action, *, margin=.03):
    """Backward-compatible one-step wrapper around the candidate-specific SFM horizon shield."""
    plan = np.asarray(action, np.float32).reshape(1, 2)
    selected, diag, _ = exact_sfm_horizon_filter_action(
        humans, state, plan, plan[None], margin=margin, horizon=1)
    return selected, diag


def _sfm_repulsive_force(r_ab, v_rel, dt=SS.DT):
    """Vectorized ``-grad_barrier_exp`` from ``cfm_mppi.utils`` (A=2.1, B=.5)."""
    r = np.asarray(r_ab, float); v = np.asarray(v_rel, float)
    rp = r - float(dt) * v
    dist = np.linalg.norm(r, axis=-1); pred_dist = np.linalg.norm(rp, axis=-1)
    q = dist + pred_dist
    expr = q * q - np.sum((float(dt) * v) ** 2, axis=-1)
    active = expr > 1e-6
    b = .5 * np.sqrt(np.maximum(expr, 1e-6))
    unit = r / np.maximum(dist[..., None], 1e-9) + rp / np.maximum(pred_dist[..., None], 1e-9)
    db = q[..., None] / np.maximum(4.0 * b[..., None], 1e-9) * unit
    potential = 2.1 * np.exp(-b / .5)
    return np.where(active[..., None], potential[..., None] / .5 * db, 0.0)


def _simulate_sfm_plans(humans, state, plans, horizon):
    """Batch-exact SFM prediction, logically terminating each candidate at first safe goal entry."""
    plans = np.asarray(plans, np.float32); B = len(plans); P = len(humans)
    H = min(int(horizon), plans.shape[1])
    robot = np.repeat(np.asarray(state, np.float32)[None], B, axis=0)
    hxy0 = np.asarray([h.state for h in humans], float)
    hvel0 = np.asarray([h.control for h in humans], float)
    goals = np.asarray([h.goal for h in humans], float)
    desired_speed = np.asarray([h.sfm_des_speed for h in humans], float)
    hxy = np.repeat(hxy0[None], B, axis=0)
    hvel = np.repeat(hvel0[None], B, axis=0)
    min_clear = np.full(B, np.inf, float); inside = np.ones(B, bool)
    reach_step = np.full(B, H + 1, int)
    eye = np.eye(P, dtype=bool)[None, :, :, None]
    for t in range(H):
        active = reach_step > H
        u = plans[:, t]
        robot[:, :2] += SS.DT * robot[:, 2:4] + .5 * SS.DT * SS.DT * u
        robot[:, 2:4] += SS.DT * u
        # Human-human terms use the pre-step snapshot for every agent, matching _advance_humans.
        r_pair = hxy[:, :, None] - hxy[:, None, :]
        v_pair = hvel[:, :, None] - hvel[:, None, :]
        pair_force = np.where(eye, 0.0, _sfm_repulsive_force(r_pair, v_pair)).sum(axis=2)
        robot_force = _sfm_repulsive_force(
            hxy - robot[:, None, :2], hvel - robot[:, None, 2:4])
        to_goal = goals[None] - hxy
        goal_dist = np.linalg.norm(to_goal, axis=2)
        desired = desired_speed[None, :, None] * to_goal / np.maximum(goal_dist[:, :, None], 1e-9)
        acceleration = 2.0 * (desired - hvel) + pair_force + robot_force
        arrived = goal_dist < .1
        hvel = hvel + acceleration * SS.DT
        speed = np.linalg.norm(hvel, axis=2)
        cap = 1.3 * desired_speed[None]
        hvel = np.where((speed > cap)[:, :, None], hvel / np.maximum(speed[:, :, None], 1e-9) * cap[:, :, None], hvel)
        hvel = np.where(arrived[:, :, None], 0.0, hvel)
        hxy = hxy + hvel * SS.DT
        clear = np.linalg.norm(hxy - robot[:, None, :2], axis=2).min(axis=1) - SS.R_PED
        min_clear = np.where(active, np.minimum(min_clear, clear), min_clear)
        inside_now = ((robot[:, :2] >= -0.5) & (robot[:, :2] <= 6.5)).all(axis=1)
        inside &= (~active) | inside_now
        reached_now = active & (clear >= 0.0) & (np.linalg.norm(robot[:, :2] - SS.GOAL, axis=1) < .5)
        reach_step[reached_now] = t + 1
    return min_clear, inside, robot, hxy, reach_step


def _goal_control_plans(state, n, horizon):
    """Deterministic bounded DI plans that accelerate toward the goal and brake before overshoot."""
    state = np.asarray(state, np.float32)
    normal = np.array([-1.0, 1.0], np.float32) / np.sqrt(2.0)
    plans = []
    for q in range(int(n)):
        st = state.copy(); seq = []
        accel = float(SS.U_MAX) * (0.75 + 0.25 * ((q % 4) + 1) / 4.0)
        brake_bias = (-0.20, 0.0, 0.15)[(q // 4) % 3]
        lateral = ((q % 3) - 1) * 0.08 * normal
        for _ in range(int(horizon)):
            delta = np.asarray(SS.GOAL, np.float32) - st[:2]
            sign = np.sign(delta)
            forward_speed = st[2:4] * sign
            brake_distance = np.maximum(forward_speed, 0.0) ** 2 / max(2.0 * accel, 1e-6)
            action = np.where(
                forward_speed < -0.05, sign * accel,
                np.where(np.abs(delta) <= brake_distance + brake_bias,
                         -np.sign(st[2:4]) * accel, sign * accel))
            action = np.clip(action + lateral, -SS.U_MAX, SS.U_MAX).astype(np.float32)
            seq.append(action); st = di_step(st, action, dt=SS.DT)
        plans.append(seq)
    return np.asarray(plans, np.float32)


def _direct_goal_continuation(state, horizon):
    """Single deterministic bang--brake tail used only beyond a learned plan's native H=10."""
    st = np.asarray(state, np.float32).copy(); seq = []
    accel = float(SS.U_MAX)
    for _ in range(int(horizon)):
        delta = np.asarray(SS.GOAL, np.float32) - st[:2]
        sign = np.sign(delta)
        forward_speed = st[2:4] * sign
        brake_distance = np.maximum(forward_speed, 0.0) ** 2 / max(2.0 * accel, 1e-6)
        action = np.where(
            forward_speed < -0.05, sign * accel,
            np.where(np.abs(delta) <= brake_distance,
                     -np.sign(st[2:4]) * accel, sign * accel))
        action = np.clip(action, -SS.U_MAX, SS.U_MAX).astype(np.float32)
        seq.append(action); st = di_step(st, action, dt=SS.DT)
    return np.asarray(seq, np.float32)


def _brake_control_plan(state, horizon):
    """Bounded plan that brings the DI velocity to zero and then holds position."""
    st = np.asarray(state, np.float32).copy(); seq = []
    for h in range(int(horizon)):
        # Preserve the original filter's four-step braking schedule exactly.
        remaining = max(1, 4 - h)
        action = np.clip(-st[2:4] / max(SS.DT * remaining, SS.DT),
                         -SS.U_MAX, SS.U_MAX).astype(np.float32)
        seq.append(action); st = di_step(st, action, dt=SS.DT)
    return np.asarray(seq, np.float32)


def _extend_plan_with_goal(state, plan, target_len):
    """Preserve the supplied prefix exactly and append a deterministic goal continuation if needed."""
    plan = np.asarray(plan, np.float32)
    if len(plan) >= int(target_len):
        return plan.copy()
    st = np.asarray(state, np.float32).copy()
    for action in plan:
        st = di_step(st, action, dt=SS.DT)
    tail = _direct_goal_continuation(st, int(target_len) - len(plan))
    return np.concatenate([plan, tail], axis=0)


def _avoidance_control_plans(humans, state, n, horizon):
    """Goal-progressing left/right tangent homotopies around the nearest moving pedestrians."""
    if int(n) <= 0 or not humans:
        return np.empty((0, int(horizon), 2), np.float32)
    state = np.asarray(state, np.float32)
    xy = np.asarray([h.state for h in humans], np.float32)
    vel = np.asarray([h.control for h in humans], np.float32)
    nearest = np.argsort(np.linalg.norm(xy - state[:2], axis=1))[:min(3, len(xy))]
    specs = [(int(j), side, speed) for j in nearest for side in (-1.0, 1.0)
             for speed in (1.2, 1.8, 2.4)]
    plans = []
    for q in range(int(n)):
        j, side, speed = specs[q % len(specs)]
        st = state.copy(); seq = []
        for h in range(int(horizon)):
            ped = xy[j] + (h + 1) * SS.DT * vel[j]
            away = st[:2] - ped
            distance = max(float(np.linalg.norm(away)), 1e-6)
            away /= distance
            tangent = side * np.array([-away[1], away[0]], np.float32)
            goal = np.asarray(SS.GOAL, np.float32) - st[:2]
            goal /= max(float(np.linalg.norm(goal)), 1e-6)
            threat = float(np.clip((1.4 - distance) / 1.1, 0.15, 1.0))
            direction = goal + threat * (0.45 * away + 0.85 * tangent)
            direction /= max(float(np.linalg.norm(direction)), 1e-6)
            desired = speed * direction
            action = np.clip(2.8 * (desired - st[2:4]), -SS.U_MAX, SS.U_MAX).astype(np.float32)
            seq.append(action); st = di_step(st, action, dt=SS.DT)
        plans.append(seq)
    return np.asarray(plans, np.float32)


def exact_sfm_horizon_filter_action(humans, state, nominal_plan, candidate_plans, *, margin=.03, horizon=10,
                                    n_goal_plans=0, n_avoid_plans=0, always_select=False, min_progress=0.0,
                                    goal_score_weight=1.0, clearance_weight=.05,
                                    fallback_clearance=False, fallback_lookahead=0,
                                    viability_lookahead=0, viability_band=.05,
                                    viability_goal_weight=0.0, viability_escalate=False,
                                    viability_escalation_band=.05,
                                    viability_escalation_min_progress=0.0,
                                    clearance_target=None, clearance_target_weight=0.0):
    """Select a candidate whose full short-horizon rollout is safe under the known SFM transition."""
    state = np.asarray(state, np.float32)
    nominal_plan = np.asarray(nominal_plan, np.float32)
    base_len = len(nominal_plan)
    H = max(1, int(horizon))
    plan_len = max(base_len, H)
    plans = [nominal_plan]
    plans.extend(np.asarray(candidate_plans, np.float32))
    brake = _brake_control_plan(state, len(nominal_plan))
    plans.append(brake)
    plans.extend(_goal_control_plans(state, n_goal_plans, plan_len))
    plans.extend(_avoidance_control_plans(humans, state, n_avoid_plans, plan_len))
    # Sustained accelerations provide deterministic escape modes when every refined trajectory shares a bad
    # homotopy.  They remain within the same DI control box and are evaluated by the identical SFM simulator.
    for ux in np.linspace(-SS.U_MAX, SS.U_MAX, 5):
        for uy in np.linspace(-SS.U_MAX, SS.U_MAX, 5):
            plans.append(np.repeat(np.array([[ux, uy]], np.float32), plan_len, axis=0))
    unique = []
    for plan in plans:
        plan = _extend_plan_with_goal(state, plan, plan_len)
        plan = np.clip(np.asarray(plan, np.float32), -SS.U_MAX, SS.U_MAX)
        if not any(np.allclose(plan, old, atol=1e-7) for old in unique):
            unique.append(plan)

    clear, inside, terminal, _, reach_step = _simulate_sfm_plans(humans, state, np.stack(unique), H)
    stop_hi = terminal[:, :2] + np.maximum(terminal[:, 2:4], 0.0) ** 2 / (2.0 * SS.U_MAX)
    stop_lo = terminal[:, :2] - np.maximum(-terminal[:, 2:4], 0.0) ** 2 / (2.0 * SS.U_MAX)
    reached = reach_step <= H
    recoverable = inside & (reached | ((stop_hi <= 6.5).all(axis=1) & (stop_lo >= -0.5).all(axis=1)))
    nominal_clear = float(clear[0])
    if not bool(always_select) and bool(recoverable[0]) and nominal_clear >= float(margin):
        return unique[0][0].copy(), dict(
            filter_solver="exact_sfm_horizon", filter_feasible=True, horizon=H,
            nominal_recoverable=True,
            nominal_terminal=terminal[0].tolist(),
            nominal_stop_lo=stop_lo[0].tolist(), nominal_stop_hi=stop_hi[0].tolist(),
            nominal_horizon_clear=float(nominal_clear), selected_horizon_clear=float(nominal_clear),
            correction_magnitude=0.0, candidates_checked=1), unique[0][:base_len].copy()
    evaluated = [(plan, float(clear[i]), bool(inside[i]), bool(recoverable[i]), terminal[i], int(reach_step[i]))
                 for i, plan in enumerate(unique)]
    feasible = [x for x in evaluated if x[3] and x[1] >= float(margin)]
    selected_future = None
    if feasible:
        current_goal_distance = float(np.linalg.norm(state[:2] - SS.GOAL))

        def progress(x):
            _, _, _, _, terminal, arrival = x
            if arrival <= H:
                return current_goal_distance
            return current_goal_distance - float(np.linalg.norm(terminal[:2] - SS.GOAL))

        def score(x):
            plan, clear, _, _, terminal, arrival = x
            goal_cost = (.04 * float(arrival) if arrival <= H
                         else float(np.linalg.norm(terminal[:2] - SS.GOAL)))
            target_cost = (0.0 if clearance_target is None else
                           float(clearance_target_weight)
                           * abs(min(clear, 1.0) - float(clearance_target)))
            return (float(goal_score_weight) * goal_cost
                    + .015 * float(np.mean(plan * plan))
                    + .02 * float(np.sum((plan[0] - nominal_plan[0]) ** 2))
                    - float(clearance_weight) * min(clear, 1.0)
                    + target_cost)
        progressing = [x for x in feasible if progress(x) >= float(min_progress)]
        if progressing:
            selected = min(progressing, key=score)
            selection_reason = "progressing_score"
            standard_viability = selected[1] <= float(margin) + float(viability_band)
            escalation_audit = (bool(viability_escalate)
                                and selected[1] <= (float(margin)
                                                    + float(viability_escalation_band)))
            if int(viability_lookahead) > H and (standard_viability or escalation_audit):
                L = int(viability_lookahead)
                extended = np.stack([_extend_plan_with_goal(state, plan, L) for plan, *_ in progressing])
                vclear, vinside, vterminal, _, vreach = _simulate_sfm_plans(humans, state, extended, L)
                vstop_hi = vterminal[:, :2] + np.maximum(vterminal[:, 2:4], 0.0) ** 2 / (2.0 * SS.U_MAX)
                vstop_lo = vterminal[:, :2] - np.maximum(-vterminal[:, 2:4], 0.0) ** 2 / (2.0 * SS.U_MAX)
                vrecover = (vinside & ((vreach <= L) | ((vstop_hi <= 6.5).all(axis=1)
                                                        & (vstop_lo >= -0.5).all(axis=1))))
                vpool = [i for i in range(len(progressing))
                         if vrecover[i] and float(vclear[i]) >= float(margin)]
                if vpool and standard_viability:
                    def viability_score(i):
                        future_goal_cost = (.04 * float(vreach[i]) if vreach[i] <= L
                                            else float(np.linalg.norm(vterminal[i, :2] - SS.GOAL)))
                        return score(progressing[i]) + float(viability_goal_weight) * future_goal_cost
                    vi = min(vpool, key=viability_score)
                    selected = progressing[vi]
                    selection_reason = "progressing_viable"
                    selected_future = dict(
                        lookahead=L, clearance=float(vclear[vi]),
                        progress=(current_goal_distance if vreach[vi] <= L else
                                  current_goal_distance
                                  - float(np.linalg.norm(vterminal[vi, :2] - SS.GOAL))),
                        arrival=(int(vreach[vi]) if vreach[vi] <= L else None))
                elif bool(viability_escalate):
                    # A direct-goal extension can reject every H-step prefix even when a genuine longer
                    # avoidance homotopy exists. Regenerate the complete candidate family at L and accept it
                    # only if the ordinary exact L-step filter is feasible.
                    e_action, e_diag, e_plan = exact_sfm_horizon_filter_action(
                        humans, state, nominal_plan, candidate_plans,
                        margin=margin, horizon=L, n_goal_plans=n_goal_plans,
                        n_avoid_plans=n_avoid_plans, always_select=True,
                        min_progress=min_progress, goal_score_weight=goal_score_weight,
                        clearance_weight=clearance_weight, fallback_clearance=fallback_clearance,
                        fallback_lookahead=0, viability_lookahead=0,
                        viability_band=viability_band, viability_goal_weight=0.0,
                        viability_escalate=False,
                        viability_escalation_band=viability_escalation_band,
                        viability_escalation_min_progress=viability_escalation_min_progress,
                        clearance_target=clearance_target,
                        clearance_target_weight=clearance_target_weight)
                    escalation_progress_floor = max(
                        float(min_progress), float(viability_escalation_min_progress))
                    if (e_diag.get("filter_feasible")
                            and float(e_diag.get("selected_progress", -np.inf))
                            >= escalation_progress_floor):
                        e_diag.update(selection_reason="proactive_horizon_escalation",
                                      escalated_from_horizon=H)
                        return e_action, e_diag, e_plan
        elif int(fallback_lookahead) > H:
            # Rank only the already H-safe candidates by a longer exact-SFM continuation.  The lookahead is a
            # liveness tie-breaker, not an execution authority: the selected first action still comes from the
            # unchanged H-step hard-margin feasible set and is reverified at the next MPC step.
            L = int(fallback_lookahead)
            extended = np.stack([_extend_plan_with_goal(state, plan, L) for plan, *_ in feasible])
            fclear, finside, fterminal, _, freach = _simulate_sfm_plans(humans, state, extended, L)
            fstop_hi = fterminal[:, :2] + np.maximum(fterminal[:, 2:4], 0.0) ** 2 / (2.0 * SS.U_MAX)
            fstop_lo = fterminal[:, :2] - np.maximum(-fterminal[:, 2:4], 0.0) ** 2 / (2.0 * SS.U_MAX)
            frecover = (finside & ((freach <= L) | ((fstop_hi <= 6.5).all(axis=1)
                                                   & (fstop_lo >= -0.5).all(axis=1))))
            fpool = [i for i in range(len(feasible))
                     if frecover[i] and float(fclear[i]) >= 0.0]
            if fpool:
                def future_score(i):
                    plan = feasible[i][0]
                    goal_cost = (.04 * float(freach[i]) if freach[i] <= L
                                 else float(np.linalg.norm(fterminal[i, :2] - SS.GOAL)))
                    return (goal_cost + .005 * float(np.mean(plan * plan))
                            - .02 * min(float(fclear[i]), 1.0))
                fi = min(fpool, key=future_score)
                selected = feasible[fi]
                selection_reason = "future_detour"
                selected_future = dict(
                    lookahead=L, clearance=float(fclear[fi]),
                    progress=(current_goal_distance if freach[fi] <= L else
                              current_goal_distance - float(np.linalg.norm(fterminal[fi, :2] - SS.GOAL))),
                    arrival=(int(freach[fi]) if freach[fi] <= L else None))
            elif fallback_clearance:
                selected = max(feasible, key=lambda x: (x[1], progress(x),
                                                         -float(np.mean(x[0] * x[0]))))
                selection_reason = "clearance_escape"
            else:
                selected = max(feasible, key=lambda x: (progress(x), x[1],
                                                         -float(np.mean(x[0] * x[0]))))
                selection_reason = "least_regressive_safe"
        elif fallback_clearance:
            # A bounded caller-controlled escape burst breaks a persistent stationary local minimum.  Unlike
            # the old stateless rule this cannot maximize clearance indefinitely and drive back to the start.
            selected = max(feasible, key=lambda x: (x[1], progress(x),
                                                     -float(np.mean(x[0] * x[0]))))
            selection_reason = "clearance_escape"
        else:
            # Every member already satisfies the hard horizon-clearance margin.  Maximizing clearance here
            # caused long receding-horizon loops: the controller repeatedly drove back toward the start even
            # when another safe candidate lost only a few centimetres of goal progress.  Preserve viability by
            # choosing the least-regressive safe plan, then use clearance and effort only as tie-breakers.
            selected = max(feasible, key=lambda x: (progress(x), x[1],
                                                     -float(np.mean(x[0] * x[0]))))
            selection_reason = "least_regressive_safe"
        ok = True
    else:
        # If the requested margin is unavailable, preserve taskspace/recoverability before maximizing clearance.
        selected = max(evaluated, key=lambda x: (x[3], x[2], x[1],
                                                  -float(np.sum((x[0][0] - nominal_plan[0]) ** 2))))
        selection_reason = "margin_unavailable"
        ok = False
    selected_plan, selected_clear, _, selected_recoverable, _, selected_arrival = selected
    selected_progress = (float(np.linalg.norm(state[:2] - SS.GOAL)) if selected_arrival <= H else
                         float(np.linalg.norm(state[:2] - SS.GOAL)
                               - np.linalg.norm(selected[4][:2] - SS.GOAL)))
    return selected_plan[0].copy(), dict(
        filter_solver="exact_sfm_horizon", filter_feasible=bool(ok), horizon=H,
        nominal_recoverable=bool(recoverable[0]),
        nominal_terminal=terminal[0].tolist(),
        nominal_stop_lo=stop_lo[0].tolist(), nominal_stop_hi=stop_hi[0].tolist(),
        recoverable=bool(selected_recoverable),
        selection_reason=selection_reason,
        fallback_future=selected_future,
        selected_progress=float(selected_progress),
        goal_score_weight=float(goal_score_weight),
        clearance_target=(None if clearance_target is None else float(clearance_target)),
        clearance_target_weight=float(clearance_target_weight),
        viability_goal_weight=float(viability_goal_weight),
        viability_escalate=bool(viability_escalate),
        viability_escalation_band=float(viability_escalation_band),
        viability_escalation_min_progress=float(viability_escalation_min_progress),
        predicted_arrival_step=(int(selected_arrival) if selected_arrival <= H else None),
        nominal_horizon_clear=float(nominal_clear), selected_horizon_clear=float(selected_clear),
        correction_magnitude=float(np.linalg.norm(selected_plan[0] - nominal_plan[0])),
        candidates_checked=len(evaluated)), selected_plan[:base_len].copy()


def guided_generate(policy, ctx, state, goal, ped_pred, ped_vel, r_col, z_init, taus, cfg,
                    collect_diagnostics=False, rollout_fn=di_rollout_t):
    """Add goal/CBF reward gradients to the frozen learned field at every ODE integration step.

    Diagnostic runs also integrate an unguided copy from the *same* initial latent samples and ODE knots.  The
    difference between the two final physical controls is therefore the accumulated net guidance effect; it is
    not the gradient from one arbitrarily selected intermediate knot.
    """
    N, H = len(z_init), policy.H_pred
    z = z_init; ctxN = policy._expand_ctx(ctx, N)
    z_unguided = z_init.detach().clone() if collect_diagnostics else None
    goal_integral = torch.zeros_like(z) if collect_diagnostics else None
    safety_integral = torch.zeros_like(z) if collect_diagnostics else None
    safe_coef = _sample_safe_coefficients(cfg, z.device, z.dtype)
    markup = float(cfg.markup) ** torch.arange(H - 1, -1, -1, dtype=z.dtype, device=z.device)
    markup = markup[None, :, None]
    trace = []
    for tau, tau_next in zip(taus, taus[1:]):
        tau = float(tau); tau_next = float(tau_next)
        tt = torch.full((N,), max(tau, 1e-4), dtype=z.dtype, device=z.device)
        with torch.no_grad():
            base = policy.forward(z, tt, ctxN)
            base_unguided = (policy.forward(z_unguided, tt, ctxN)
                             if collect_diagnostics else None)
        x1 = (z + (1.0 - tau) * base).detach().requires_grad_(True)
        U1 = x1.reshape(N, H, 2) * float(policy.u_max)
        pos, vel = rollout_fn(state, U1, SS.DT)
        r_cbf, cbf_terms = cbf_reward(pos, vel, ped_pred, ped_vel, r_col, cfg)
        r_goal = goal_reward(pos, goal)
        g_cbf, = torch.autograd.grad(r_cbf.sum(), x1, retain_graph=True)
        g_goal, = torch.autograd.grad(r_goal.sum(), x1)
        base_norm = torch.linalg.vector_norm(base)
        raw_cbf_norm = torch.linalg.vector_norm(g_cbf)
        raw_goal_norm = torch.linalg.vector_norm(g_goal)
        g_cbf = g_cbf * base_norm / (raw_cbf_norm + 1e-8)
        g_goal = g_goal * base_norm / (raw_goal_norm + 1e-8)
        goal_guidance = float(cfg.goal_coef) * g_goal.reshape(N, H, 2)
        safety_guidance = safe_coef * g_cbf.reshape(N, H, 2) * markup
        guidance = goal_guidance + safety_guidance
        z = z + (tau_next - tau) * (base + guidance.reshape(N, -1))
        if collect_diagnostics:
            z_unguided = z_unguided + (tau_next - tau) * base_unguided
            goal_integral = goal_integral + (
                tau_next - tau
            ) * goal_guidance.reshape(N, -1)
            safety_integral = safety_integral + (
                tau_next - tau
            ) * safety_guidance.reshape(N, -1)
            trace.append(dict(
                tau=tau, dtau=tau_next - tau, base_norm=float(base_norm.detach().cpu()),
                raw_cbf_grad_norm=float(raw_cbf_norm.detach().cpu()),
                raw_goal_grad_norm=float(raw_goal_norm.detach().cpu()),
                guidance_norm=float(torch.linalg.vector_norm(guidance).detach().cpu()),
                cbf_active_fraction=float((cbf_terms < 0).float().mean().detach().cpu()),
                endpoint_min_pred_clear=float((torch.linalg.vector_norm(
                    pos.unsqueeze(2) - ped_pred.unsqueeze(0), dim=3) - r_col).min().detach().cpu()),
            ))
    components = None
    if collect_diagnostics:
        components = dict(
            goal=goal_integral,
            safety=safety_integral,
            semantics=(
                "ODE-time integrals of the additive normalized goal and CBF "
                "guidance fields evaluated along the guided latent path"
            ),
        )
    return z, trace, z_unguided, components


@torch.no_grad()
def flow_mppi_refine(policy, state, goal, ped_xy, ped_vel, ped_pred, r_col,
                     U_gen, prev_U, cfg, collect_diagnostics=False,
                     rollout_fn=di_rollout_t):
    pos, _ = rollout_fn(state, U_gen, SS.DT)
    generated_cost = refinement_cost_batch(
        state, U_gen, goal, ped_xy, ped_vel, ped_pred, r_col, cfg, prev_U
    )
    n_elite = min(int(cfg.n_elite), len(U_gen))
    top = torch.topk(generated_cost, k=n_elite, largest=False).indices
    elites = U_gen[top]; E = len(elites)
    pert = elites.repeat_interleave(int(cfg.n_copy), dim=0)
    pert = torch.clamp(pert + float(cfg.mppi_sigma) * torch.randn_like(pert),
                       -float(policy.u_max), float(policy.u_max))
    ppos, _ = rollout_fn(state, pert, SS.DT)
    pcost = refinement_cost_batch(
        state, pert, goal, ped_xy, ped_vel, ped_pred, r_col, cfg, prev_U
    ).reshape(E, int(cfg.n_copy))
    shift = pcost.min(dim=1, keepdim=True).values
    weight = torch.softmax(-(pcost - shift) / float(cfg.mppi_lambda), dim=1)
    refined = (weight[:, :, None, None] * pert.reshape(E, int(cfg.n_copy), *pert.shape[1:])).sum(1)
    rpos, _ = rollout_fn(state, refined, SS.DT)
    refined_cost = refinement_cost_batch(
        state, refined, goal, ped_xy, ped_vel, ped_pred, r_col, cfg, prev_U
    )
    rclear = torch.linalg.vector_norm(rpos.unsqueeze(2) - ped_pred.unsqueeze(0), dim=3) - r_col
    refined_min_clear = rclear.amin(dim=(1, 2))
    best = _select_refined_index(refined_cost, refined_min_clear, cfg); U_best = refined[best]
    if not collect_diagnostics and not cfg.exact_sfm_step_filter:
        return U_best, None
    def draw_positions(x, cap):
        n = len(x)
        if n <= int(cap):
            return x.detach().cpu().numpy()
        idx = torch.linspace(0, n - 1, steps=int(cap), device=x.device).round().long().unique()
        return x[idx].detach().cpu().numpy()
    gclear = torch.linalg.vector_norm(pos.unsqueeze(2) - ped_pred.unsqueeze(0), dim=3) - r_col
    diag = dict(
        generated_cost_min=float(generated_cost.min().cpu()), generated_cost_median=float(generated_cost.median().cpu()),
        generated_min_pred_clear=float(gclear.min().cpu()), generated_collision_fraction=float((gclear.min(dim=2).values.min(dim=1).values < 0).float().mean().cpu()),
        refined_cost_best=float(refined_cost[best].cpu()), refined_min_pred_clear=float(rclear[best].min().cpu()),
        refined_safe_mode_fraction=float((refined_min_clear >= float(cfg.refined_clearance_margin)).float().mean().cpu()),
        hard_clearance_select=bool(cfg.hard_clearance_select),
        refinement_cost=str(cfg.refinement_cost),
        refinement_cost_manifest=(BC.scorer_manifest()
                                  if cfg.refinement_cost == "b1_safemppi" else None),
        selected_elite=int(best), selected_generated_index=int(top[best].cpu()),
        elite_generated_indices=top.detach().cpu().numpy(),
        best_planned_positions=rpos[best].cpu().numpy(),
        # Match the original notebook's four animation layers without persisting all 2,000 MPPI rollouts.
        generated_positions=draw_positions(pos, 40),
        elite_generated_positions=pos[top].detach().cpu().numpy(),
        mppi_perturbed_positions=draw_positions(ppos, 60),
        refined_mode_positions=rpos.detach().cpu().numpy(),
    )
    if cfg.exact_sfm_step_filter:
        diag["_refined_controls"] = refined.detach().cpu().numpy()
    return U_best, diag


def kazuki_sfm_deploy(policy, episode, gamma, cfg=None, n_ped=20, T=180, reach=0.5, device="cpu",
                      ped_speed_range=SS.ID_PED_SPEED_RANGE, sample_seed=700000, collect_diagnostics=False,
                      gamma_selector=None):
    base_cfg = (cfg or KazukiConfig()).validate()
    dynamic_gamma = gamma_selector is not None
    cfg = (interpolated_gamma_controller_config(base_cfg, gamma) if dynamic_gamma
           else _gamma_controller_config(base_cfg, gamma)).validate()
    guidance_cfg = _gamma_guidance_config(cfg, gamma)
    output_filter = None
    if cfg.output_filter and cfg.filter_solver == "jacobi":
        output_filter = SafeMPPIAdapter(
            horizon=policy.H_pred, dt=SS.DT, dynamics_type="doubleintegrator",
            u_min=(-float(policy.u_max), -float(policy.u_max)),
            u_max=(float(policy.u_max), float(policy.u_max)), use_polytope_barrier=False,
            use_ho_barrier=True, eta=float(cfg.filter_eta), barrier_activation_radius=SS.R_SENSE,
            safety_margin=float(cfg.filter_margin), filter_output=True, filter_iters=int(cfg.filter_iters))
    humans = SS.make_humans(episode, 0, n_ped, speed_range=ped_speed_range)
    state = np.zeros(4, np.float32); goal = torch.tensor(SS.GOAL, dtype=torch.float32, device=device)
    H, d = policy.H_pred, policy.d
    states, controls, peds, ped_vels, trace, gamma_history = [state.copy()], [], [], [], [], []
    history = []; prev_z = prev_U = None
    step_filter_stall = step_filter_escape_remaining = 0
    step_filter_escalation_remaining = 0
    step_filter_stagnation_remaining = 0
    step_filter_goal_distance_history = []
    hp_history = HH.HpHistory()
    reached = collision = False; min_clear = float("inf"); collision_ped = None
    terminal_ped_xy = terminal_ped_vel = None
    for t in range(T):
        ped_xy, ped_vel = _collect_humans(humans)
        terminal_ped_xy, terminal_ped_vel = ped_xy.copy(), ped_vel.copy()
        clear = np.linalg.norm(ped_xy - state[:2][None], axis=1) - SS.R_PED
        jmin = int(np.argmin(clear)); min_clear = min(min_clear, float(clear[jmin]))
        if clear[jmin] < 0:
            collision = True; collision_ped = jmin; break
        if float(np.linalg.norm(state[:2] - SS.GOAL)) < reach:
            reached = True; break
        obs = np.concatenate([ped_xy, np.full((n_ped, 1), SS.R_PED, np.float32)], axis=1)
        grid_raw = torch.tensor(GF.axis_grid(state[:2], obs, 0.0, R=SS.R_SENSE, sensing=SS.R_SENSE),
                                device=device)
        G = hp_history.append(grid_raw)
        Hp = torch.tensor(GF.hist_pad(np.asarray(history[-GF.K_HIST:]) if history else np.zeros((0, 2)),
                                      GF.K_HIST), device=device)
        gamma_diag = None
        gamma_step = float(gamma_history[-1]) if gamma_history else float(gamma)
        if dynamic_gamma:
            gamma_step, gamma_diag = gamma_selector.select(
                policy=policy, grid=G, state=state, history=Hp, ped_xy=ped_xy, ped_vel=ped_vel,
                previous_gamma=gamma_step, step=t,
                seed=int(sample_seed) + int(episode) * 1000 + t,
                latent_anchor=prev_z, warm_s=float(base_cfg.warm_s), ode_times=base_cfg.ode_times)
            gamma_step = float(gamma_step)
            cfg = interpolated_gamma_controller_config(base_cfg, gamma_step).validate()
            guidance_cfg = _gamma_guidance_config(cfg, gamma_step)
        gamma_history.append(gamma_step)
        L = torch.tensor(GF.low5(state, SS.GOAL, gamma_step), device=device)
        ctx = policy.ctx_from(G[None], L[None], Hp[None]).squeeze(0)
        torch.manual_seed(int(sample_seed) + int(episode) * 1000 + t)
        if prev_z is None:
            z = torch.randn(cfg.n_sample, d, device=device); taus = cfg.ode_times
        else:
            z = float(cfg.warm_s) * prev_z[None].expand(cfg.n_sample, d) \
                + (1.0 - float(cfg.warm_s)) * torch.randn(cfg.n_sample, d, device=device)
            taus = tuple(x for x in cfg.ode_times if x >= cfg.warm_s - 1e-12)
        ped_pred = predict_pedestrians_t(ped_xy, ped_vel, H, SS.DT, device, z.dtype)
        ped_vel_t = torch.tensor(ped_vel, dtype=z.dtype, device=device)
        warm_positions = None
        if collect_diagnostics and prev_U is not None:
            warm_positions = di_rollout_t(state, prev_U[None], SS.DT)[0][0].detach().cpu().numpy()
        z1, ode_diag, z1_unguided, guidance_components = guided_generate(
            policy, ctx, state, goal, ped_pred, ped_vel_t,
            SS.R_PED + cfg.collision_margin, z, taus, guidance_cfg,
            collect_diagnostics=collect_diagnostics)
        U_gen = torch.clamp(z1.reshape(cfg.n_sample, H, 2) * float(policy.u_max),
                            -float(policy.u_max), float(policy.u_max))
        U_unguided = (torch.clamp(z1_unguided.reshape(cfg.n_sample, H, 2) * float(policy.u_max),
                                  -float(policy.u_max), float(policy.u_max))
                      if collect_diagnostics else None)
        U_best, refine_diag = flow_mppi_refine(policy, state, goal, ped_xy, ped_vel, ped_pred,
                                               SS.R_PED + cfg.collision_margin, U_gen, prev_U, guidance_cfg,
                                               collect_diagnostics=collect_diagnostics)
        action = U_best[0].detach().cpu().numpy().astype(np.float32)
        filter_diag = None
        if cfg.output_filter and cfg.filter_solver == "exact":
            action, filter_diag = exact_ho_filter_action(
                state, obs, ped_vel, action, gamma_step, eta=cfg.filter_eta,
                margin=cfg.filter_margin, activation_radius=SS.R_SENSE)
        elif output_filter is not None:
            obs_t = torch.tensor(
                np.concatenate([ped_xy, np.full((n_ped, 1), SS.R_PED, np.float32)], axis=1),
                dtype=torch.float32, device=device)
            action_t, filter_diag = output_filter.safety_filter_action(
                torch.tensor(state, dtype=torch.float32, device=device), obs_t,
                torch.tensor(action, dtype=torch.float32, device=device), gamma=float(gamma_step),
                obstacle_velocities=torch.tensor(ped_vel, dtype=torch.float32, device=device),
                iters=int(cfg.filter_iters))
            action = action_t.detach().cpu().numpy().astype(np.float32)
        if cfg.exact_sfm_step_filter:
            refined_pool = refine_diag.pop("_refined_controls")
            if prev_U is not None:
                # Preserve the previously selected certified homotopy explicitly.  MPPI also receives this warm
                # plan, but refinement can move every returned mode to a different side of the same pedestrian.
                refined_pool = np.concatenate(
                    [prev_U.detach().cpu().numpy()[None], refined_pool], axis=0)
            effective_margin = _adaptive_step_filter_margin(cfg, gamma_step)
            clearance_target = _adaptive_step_filter_clearance_target(cfg, gamma_step, effective_margin)
            escalation_was_active = step_filter_escalation_remaining > 0
            escalation_progress_floor = float(cfg.step_filter_viability_escalation_min_progress)
            if (not escalation_was_active
                    and float(cfg.step_filter_viability_escalation_entry_progress) > 0.0):
                escalation_progress_floor = float(cfg.step_filter_viability_escalation_entry_progress)
            stagnation_enabled = bool(
                cfg.step_filter_stagnation_window > 0
                and float(gamma_step) <= float(cfg.step_filter_stagnation_gamma_max) + 1e-9)
            current_goal_distance = float(np.linalg.norm(state[:2] - SS.GOAL))
            if stagnation_enabled and step_filter_stagnation_remaining == 0:
                step_filter_goal_distance_history.append(current_goal_distance)
                W = int(cfg.step_filter_stagnation_window)
                if (len(step_filter_goal_distance_history) >= W + 1
                        and (step_filter_goal_distance_history[-W - 1]
                             - step_filter_goal_distance_history[-1])
                        < float(cfg.step_filter_stagnation_progress)):
                    step_filter_stagnation_remaining = int(cfg.step_filter_stagnation_burst)
                    step_filter_goal_distance_history = [current_goal_distance]
            filter_horizon = (int(cfg.step_filter_stagnation_horizon)
                              if step_filter_stagnation_remaining > 0
                              else int(cfg.step_filter_horizon))
            current_margin_violation = bool(
                cfg.step_filter_escape_patience > 0 and float(clear[jmin]) < effective_margin)
            action, step_filter_diag, selected_plan = exact_sfm_horizon_filter_action(
                humans, state, U_best.detach().cpu().numpy(), refined_pool,
                margin=effective_margin, horizon=filter_horizon,
                n_goal_plans=cfg.step_filter_goal_plans,
                n_avoid_plans=cfg.step_filter_avoid_plans,
                always_select=cfg.step_filter_always_select,
                min_progress=cfg.step_filter_min_progress,
                goal_score_weight=cfg.step_filter_goal_score_weight,
                clearance_weight=cfg.step_filter_clearance_weight,
                fallback_clearance=bool(step_filter_escape_remaining > 0 or current_margin_violation),
                fallback_lookahead=cfg.step_filter_fallback_lookahead,
                viability_lookahead=cfg.step_filter_viability_lookahead,
                viability_band=cfg.step_filter_viability_band,
                viability_goal_weight=cfg.step_filter_viability_goal_weight,
                viability_escalate=cfg.step_filter_viability_escalate,
                viability_escalation_band=cfg.step_filter_viability_escalation_band,
                viability_escalation_min_progress=escalation_progress_floor,
                clearance_target=clearance_target,
                clearance_target_weight=cfg.step_filter_clearance_target_weight)
            if t < int(cfg.step_filter_release_steps):
                hold_plan = _brake_control_plan(state, H)
                hclear, hinside, hterminal, _, _ = _simulate_sfm_plans(
                    humans, state, hold_plan[None], H)
                hold_stop_hi = (hterminal[0, :2]
                                + np.maximum(hterminal[0, 2:4], 0.0) ** 2 / (2.0 * SS.U_MAX))
                hold_stop_lo = (hterminal[0, :2]
                                - np.maximum(-hterminal[0, 2:4], 0.0) ** 2 / (2.0 * SS.U_MAX))
                hold_safe = bool(hinside[0] and float(hclear[0]) >= effective_margin
                                 and (hold_stop_hi <= 6.5).all() and (hold_stop_lo >= -0.5).all())
                if hold_safe:
                    action = hold_plan[0].copy(); selected_plan = hold_plan.copy()
                    step_filter_diag.update(
                        filter_feasible=True, selection_reason="gamma_release_hold",
                        selected_horizon_clear=float(hclear[0]),
                        selected_progress=0.0, correction_magnitude=float(
                            np.linalg.norm(hold_plan[0] - U_best[0].detach().cpu().numpy())))
                step_filter_diag.update(release_step=int(t), release_hold_safe=bool(hold_safe),
                                        release_steps=int(cfg.step_filter_release_steps))
            reason = step_filter_diag.get("selection_reason")
            if reason == "proactive_horizon_escalation" and not escalation_was_active:
                step_filter_escalation_remaining = int(cfg.step_filter_viability_escalation_burst)
            elif escalation_was_active:
                step_filter_escalation_remaining = max(0, step_filter_escalation_remaining - 1)
            if step_filter_stagnation_remaining > 0:
                step_filter_stagnation_remaining -= 1
                if step_filter_stagnation_remaining == 0:
                    step_filter_goal_distance_history = [current_goal_distance]
            if reason == "clearance_escape":
                if not current_margin_violation:
                    step_filter_escape_remaining = max(0, step_filter_escape_remaining - 1)
                if step_filter_escape_remaining == 0 and not current_margin_violation:
                    step_filter_stall = 0
            elif reason == "least_regressive_safe" and cfg.step_filter_escape_patience > 0:
                step_filter_stall += 1
                if step_filter_stall >= int(cfg.step_filter_escape_patience):
                    step_filter_stall = 0
                    step_filter_escape_remaining = int(cfg.step_filter_escape_burst)
            else:
                step_filter_stall = 0
                step_filter_escape_remaining = 0
            step_filter_diag.update(stall_count=int(step_filter_stall),
                                    escape_remaining=int(step_filter_escape_remaining),
                                    current_margin_violation=bool(current_margin_violation),
                                    current_clearance=float(clear[jmin]),
                                    base_margin=float(cfg.step_filter_margin),
                                    gamma_margin_span=float(cfg.step_filter_gamma_margin_span),
                                    effective_margin=float(effective_margin),
                                    adaptive_clearance_target=(None if clearance_target is None
                                                               else float(clearance_target)),
                                    escalation_progress_floor=float(escalation_progress_floor),
                                    escalation_remaining=int(step_filter_escalation_remaining),
                                    stagnation_horizon=int(filter_horizon),
                                    stagnation_remaining=int(step_filter_stagnation_remaining))
            U_best = torch.tensor(selected_plan, dtype=U_best.dtype, device=U_best.device)
            filter_diag = dict(filter_diag or {}, **step_filter_diag)
        guidance_diag = selected_plan_positions = None
        if collect_diagnostics:
            seed_index = int(refine_diag["selected_generated_index"])
            guided_final_action = U_gen[seed_index, 0].detach().cpu().numpy().astype(np.float32)
            unguided_final_action = U_unguided[seed_index, 0].detach().cpu().numpy().astype(np.float32)
            net_guidance_action = guided_final_action - unguided_final_action
            goal_guidance_action = (
                guidance_components["goal"][seed_index]
                .reshape(H, 2)[0].detach().cpu().numpy().astype(np.float32)
                * float(policy.u_max)
            )
            safety_guidance_action = (
                guidance_components["safety"][seed_index]
                .reshape(H, 2)[0].detach().cpu().numpy().astype(np.float32)
                * float(policy.u_max)
            )
            guidance_diag = dict(
                selected_generated_index=seed_index,
                guided_final_action=guided_final_action,
                unguided_final_action=unguided_final_action,
                net_guidance_action=net_guidance_action,
                net_guidance_norm=float(np.linalg.norm(net_guidance_action)),
                goal_guidance_action=goal_guidance_action,
                goal_guidance_norm=float(np.linalg.norm(goal_guidance_action)),
                safety_guidance_action=safety_guidance_action,
                safety_guidance_norm=float(np.linalg.norm(safety_guidance_action)),
                component_semantics=guidance_components["semantics"],
                semantics=(
                    "net: guided final acceleration minus unguided final "
                    "acceleration from the same latent sample; goal/safety: "
                    "separate integrated additive guidance components"
                ))
            viz_controls = U_best.detach().clone()
            viz_controls[0] = torch.as_tensor(action, dtype=viz_controls.dtype, device=viz_controls.device)
            with torch.no_grad():
                viz_pos, _ = di_rollout_t(state, viz_controls[None], SS.DT)
            selected_plan_positions = np.concatenate(
                [state[:2][None], viz_pos[0].detach().cpu().numpy()], axis=0).astype(np.float32)
        pre_state = state.copy(); peds.append(ped_xy.copy()); ped_vels.append(ped_vel.copy())
        state = di_step(state, action, dt=SS.DT)
        controls.append(action); history.append(action); states.append(state.copy())
        _advance_humans(humans, robot_xy=state[:2].copy(), robot_control_si=state[2:4].copy())
        if collect_diagnostics:
            trace.append(dict(step=t, state=pre_state, action=action, current_clear=float(clear[jmin]),
                              closest_ped=jmin, ode=ode_diag, refine=refine_diag,
                              output_filter=filter_diag,
                              gamma=float(gamma_step), gamma_selection=gamma_diag,
                              ped_xy=ped_xy.copy(), ped_vel=ped_vel.copy(),
                              warm_start_positions=warm_positions,
                              accumulated_guidance=guidance_diag,
                              selected_plan_positions=selected_plan_positions))
        shifted = torch.cat([U_best[1:], U_best[-1:]], dim=0)
        prev_z = (shifted / float(policy.u_max)).reshape(-1).detach(); prev_U = shifted.detach()
    return dict(
        states=np.asarray(states, np.float32), controls=np.asarray(controls, np.float32),
        peds=np.asarray(peds, np.float32), ped_vels=np.asarray(ped_vels, np.float32),
        path=np.asarray(states, np.float32)[:, :2], success=bool(reached and not collision),
        collision=bool(collision), reached=bool(reached), steps=len(controls), min_clear=float(min_clear),
        collision_step=len(controls) if collision else None, collision_ped=collision_ped,
        gamma=float(gamma), episode=int(episode), ped_speed_range=tuple(map(float, ped_speed_range)),
        gammas=np.asarray(gamma_history, np.float32), dynamic_gamma=bool(dynamic_gamma),
        config=base_cfg.to_dict(), effective_controller=cfg.to_dict(),
        effective_guidance=dict(safe_coefs=tuple(map(float, guidance_cfg.safe_coefs)),
                                goal_coef=float(guidance_cfg.goal_coef)),
        trace=trace if collect_diagnostics else None,
        terminal_ped_xy=terminal_ped_xy, terminal_ped_vel=terminal_ped_vel,
    )
