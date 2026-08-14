"""Mirror-map proposal for Safe MPPI (Dohyun, 2026-06-24).

Samples control sequences whose double-integrator rollouts are FEASIBLE BY
CONSTRUCTION w.r.t. the per-step convex polytope (one separating half-space per
nearby pedestrian, pulled back through the dynamics, intersected with the control
box). The feasibility comes from the log-barrier MIRROR MAP: for a convex polytope
{u : C u <= d}, with barrier phi(u) = -sum_l log(d_l - c_l^T u), the conjugate
decode u = grad phi*(y) maps ANY dual point y into the polytope interior. So a
Gaussian in the dual (later: a learned flow in the dual) yields feasible controls
with NO rejection/projection. This fixes the ~1% accept rate of Gaussian-around-
nominal sampling. Single obstacle => single half-space (fallback to the affine method).

References: Mirror Diffusion (Liu et al., NeurIPS 2023, arXiv:2310.01236); Mirror
Flow Matching on convex domains (Guan et al., arXiv:2510.08929, 2025); log-barrier
/ Dikin geometry (Boyd & Vandenberghe; Kannan & Narayanan 2012).
"""
from __future__ import annotations
from typing import Optional, Tuple

import torch


def logbarrier_grad(u: torch.Tensor, C: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """grad phi(u) = sum_l c_l / (d_l - c_l^T u).  u:[M,2] C:[M,F,2] d:[M,F] -> [M,2]."""
    s = (d - torch.einsum("mfk,mk->mf", C, u)).clamp_min(1e-4)  # slacks [M,F] (floored for stability)
    return torch.einsum("mf,mfk->mk", 1.0 / s, C)


def logbarrier_hess(u: torch.Tensor, C: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Dikin Hessian H(u) = sum_l c_l c_l^T / (d_l - c_l^T u)^2 -> [M,2,2]."""
    s = (d - torch.einsum("mfk,mk->mf", C, u)).clamp_min(1e-4)  # floored => bounded 1/s^2
    w = (1.0 / s) ** 2  # [M,F]
    return torch.einsum("mf,mfi,mfj->mij", w, C, C)


def mirror_decode(
    y: torch.Tensor, C: torch.Tensor, d: torch.Tensor, u0: torch.Tensor,
    iters: int = 8, ridge: float = 1e-3, bt_iters: int = 8,
) -> torch.Tensor:
    """Newton solve of grad phi(u) = y (i.e. u = grad phi*(y)) from interior u0,
    with backtracking that keeps every slack d_l - c_l^T u > 0 (stay feasible).
    Uses an analytic 2x2 inverse (+ridge) so it can never raise on singular H."""
    u = u0.clone()
    for _ in range(iters):
        g = logbarrier_grad(u, C, d) - y           # residual [M,2]
        H = logbarrier_hess(u, C, d)               # [M,2,2]
        a = H[:, 0, 0] + ridge; b = H[:, 0, 1]; c = H[:, 1, 1] + ridge
        det = (a * c - b * b).clamp_min(1e-9)
        # delta = -H^{-1} g  (analytic 2x2)
        d0 = -(c * g[:, 0] - b * g[:, 1]) / det
        d1 = -(-b * g[:, 0] + a * g[:, 1]) / det
        delta = torch.stack([d0, d1], dim=1)        # [M,2]
        # backtracking line search to keep all slacks positive
        alpha = torch.ones(u.shape[0], device=u.device, dtype=u.dtype)
        for _bt in range(bt_iters):
            u_try = u + alpha.unsqueeze(-1) * delta
            slack = d - torch.einsum("mfk,mk->mf", C, u_try)
            bad = (slack <= 1e-9).any(dim=1)
            if not bool(bad.any()):
                break
            alpha = torch.where(bad, alpha * 0.5, alpha)
        u = u + alpha.unsqueeze(-1) * delta
    return u


def deep_interior_2d(
    C: torch.Tensor, d: torch.Tensor, u_init: torch.Tensor,
    iters: int = 20, target: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Push u toward a deep-interior (Chebyshev-like) point by repeatedly relaxing
    the tightest face: to raise margin_j = d_j - c_j^T u, step u along -c_j.
    Returns (u [M,2], min_slack [M]); min_slack<=0 flags an empty polytope (cornered)."""
    u = u_init.clone()
    M = C.shape[0]
    ar = torch.arange(M, device=C.device)
    for _ in range(iters):
        slack = d - torch.einsum("mfk,mk->mf", C, u)      # [M,F]
        jmin = torch.argmin(slack, dim=1)
        s_j = slack[ar, jmin]                              # [M]
        a = C[ar, jmin]                                    # [M,2]
        a2 = (a * a).sum(dim=1).clamp_min(1e-9)
        bump = torch.clamp(target - s_j, min=0.0)          # want margin >= target
        u = u - (bump / a2).unsqueeze(1) * a               # move away from tightest face
    slack = d - torch.einsum("mfk,mk->mf", C, u)
    return u, slack.min(dim=1).values


def mirror_sample_controls(
    state: torch.Tensor,              # [4] (px,py,vx,vy)
    goal: torch.Tensor,              # [2]
    obstacles: torch.Tensor,         # [N,3] (cx,cy,radius) RAW (not yet inflated)
    obstacle_velocities: Optional[torch.Tensor] = None,  # [N,2]
    *,
    horizon: int = 30,
    num_samples: int = 256,
    dt: float = 0.1,
    u_min: Tuple[float, float] = (-2.0, -2.0),
    u_max: Tuple[float, float] = (2.0, 2.0),
    safety_margin: float = 0.5,
    dual_sigma: float = 1.2,
    eta: float = 0.6,
    sensing_range: float = 6.0,
    seed: int = 0,
    device: Optional[torch.device] = None,
):
    """Sample `num_samples` double-integrator control sequences whose per-step
    positions stay inside the per-step convex polytope (one separating half-space
    per nearby pedestrian, pulled back through the dynamics + control box), via the
    log-barrier mirror map. Returns (controls [M,H,2], states [M,H+1,4],
    feasible [M] bool) — feasible-by-construction (modulo genuinely cornered steps,
    where it brakes). At step 0 all tracks share one polytope, so the MPPI average
    of their first controls is also feasible (constrain-then-average, exercised)."""
    device = device or state.device
    dt2 = 0.5 * dt * dt
    M, H, N = num_samples, horizon, obstacles.shape[0]
    umin = torch.tensor(u_min, device=device, dtype=state.dtype)
    umax = torch.tensor(u_max, device=device, dtype=state.dtype)
    centers = obstacles[:, :2].to(device)
    radii = (obstacles[:, 2].to(device) + safety_margin)
    velo = (obstacle_velocities.to(device) if obstacle_velocities is not None
            else torch.zeros(N, 2, device=device, dtype=state.dtype))
    gen = torch.Generator(device=device); gen.manual_seed(int(seed))

    box_C = torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]], device=device, dtype=state.dtype)
    box_d = torch.stack([umax[0], -umin[0], umax[1], -umin[1]])

    P = state[:2].to(device).expand(M, 2).clone()
    V = state[2:4].to(device).expand(M, 2).clone()
    goal2 = goal[:2].to(device)
    ar = torch.arange(M, device=device)
    ctrls, states = [], [torch.cat([P, V], dim=1)]
    cornered_any = torch.zeros(M, device=device, dtype=torch.bool)

    for i in range(H):
        ci = centers + velo * (dt * i)                       # [N,2] predicted obstacle pos
        diff = ci.unsqueeze(0) - P.unsqueeze(1)              # [M,N,2]
        dist = torch.linalg.norm(diff, dim=2).clamp_min(1e-6)
        n = diff / dist.unsqueeze(2)                          # unit, toward obstacle
        p_o = ci.unsqueeze(0) - n * radii.view(1, N, 1)       # boundary points
        Cobs = dt2 * n                                        # face normals in u-space [M,N,2]
        rhs = torch.einsum("mnk,mnk->mn", n, p_o - P.unsqueeze(1) - dt * V.unsqueeze(1))
        if eta > 0:
            vrel = V.unsqueeze(1) - velo.unsqueeze(0)         # [M,N,2]
            rhs = rhs - eta * torch.relu(torch.einsum("mnk,mnk->mn", n, vrel))
        far = (dist - radii.view(1, N)) > sensing_range
        rhs = torch.where(far, torch.full_like(rhs, 1e6), rhs)
        C = torch.cat([box_C.expand(M, 4, 2), Cobs], dim=1)   # [M,4+N,2]
        d = torch.cat([box_d.expand(M, 4), rhs], dim=1)       # [M,4+N]

        u_nom = torch.clamp(0.45 * (goal2 - P) + 0.8 * (-V), umin, umax)
        uc, minslack = deep_interior_2d(C, d, u_nom, iters=30, target=0.1)
        cornered = minslack <= 1e-3
        cornered_any |= cornered
        y0 = logbarrier_grad(uc, C, d)
        y = y0 + dual_sigma * torch.randn(M, 2, generator=gen, device=device, dtype=state.dtype)
        u = mirror_decode(y, C, d, uc)
        # cornered (polytope empty) => actively EVADE toward max clearance (away from
        # the crowd, weighted by proximity) + partial brake, rather than just braking
        # into a pedestrian (the stall-then-collide failure).
        clr_cur = dist - radii.view(1, N)                     # [M,N] current clearance
        w_ev = torch.relu(sensing_range - clr_cur)            # weight nearby obstacles
        evade = -(w_ev.unsqueeze(2) * n).sum(dim=1)           # away from crowd [M,2]
        en = torch.linalg.norm(evade, dim=1, keepdim=True).clamp_min(1e-6)
        u_evade = (evade / en) * float(umax[0])               # full-speed away
        u_corner = torch.clamp(0.7 * u_evade - 0.3 * V / dt, umin, umax)
        u = torch.where(cornered.unsqueeze(1), u_corner, u)
        u = torch.clamp(u, umin, umax)
        ctrls.append(u)
        P = P + dt * V + dt2 * u
        V = V + dt * u
        states.append(torch.cat([P, V], dim=1))

    controls = torch.stack(ctrls, dim=1)                      # [M,H,2]
    states = torch.stack(states, dim=1)                       # [M,H+1,4]
    return controls, states, ~cornered_any


def mirror_mppi_action(
    state: torch.Tensor, goal: torch.Tensor, obstacles: torch.Tensor,
    obstacle_velocities: Optional[torch.Tensor] = None, *,
    horizon: int = 30, num_samples: int = 256, dt: float = 0.1,
    u_min=(-2.0, -2.0), u_max=(2.0, 2.0), safety_margin: float = 0.5,
    dual_sigma: float = 1.2, eta: float = 0.6, sensing_range: float = 6.0,
    temperature: float = 1.0, terminal_w: float = 8.0, running_w: float = 0.3,
    clear_w: float = 40.0, effort_w: float = 0.02, prox_w: float = 3.0, prox_scale: float = 0.6,
    seed: int = 0, gamma: float = 1.0, margin_gain: float = 0.25, mode_aware: bool = True,
    device: Optional[torch.device] = None, return_rollouts: bool = False,
    return_samples: bool = False,
):
    """MPPI over the mirror-sampled (feasible-by-construction) proposal: cost the
    rollouts, importance-weight, and AVERAGE the first controls. Since all tracks'
    step-0 controls live in one shared polytope, the average is feasible too
    (constrain-then-average, genuinely exercised). Returns (action[2], info)."""
    device = device or state.device
    # gamma knob: lower gamma => wider berth (the polytope is inflated more).
    margin_eff = safety_margin + margin_gain * (1.0 - float(gamma))
    ctrl, states, _ = mirror_sample_controls(
        state, goal, obstacles, obstacle_velocities, horizon=horizon,
        num_samples=num_samples, dt=dt, u_min=u_min, u_max=u_max,
        safety_margin=margin_eff, dual_sigma=dual_sigma, eta=eta,
        sensing_range=sensing_range, seed=seed, device=device)
    M, H1, _ = states.shape
    g2 = goal[:2].to(device)
    centers = obstacles[:, :2].to(device); radii = obstacles[:, 2].to(device) + safety_margin
    velo = (obstacle_velocities.to(device) if obstacle_velocities is not None
            else torch.zeros(obstacles.shape[0], 2, device=device, dtype=state.dtype))
    gdist = torch.linalg.norm(states[:, :, :2] - g2.view(1, 1, 2), dim=2)   # [M,H1]
    cost = terminal_w * gdist[:, -1] ** 2 + running_w * (gdist ** 2).sum(dim=1)
    cost = cost + effort_w * (ctrl ** 2).sum(dim=(1, 2))
    minclear = torch.full((M,), 1e9, device=device, dtype=state.dtype)
    for i in range(H1):
        ci = centers + velo * (dt * i)
        dd = torch.linalg.norm(states[:, i, :2].unsqueeze(1) - ci.unsqueeze(0), dim=2) - radii.unsqueeze(0)
        c = dd.min(dim=1).values
        minclear = torch.minimum(minclear, c)
        cost = cost + clear_w * torch.relu(-c) ** 2
        cost = cost + prox_w * torch.exp(-c.clamp_min(-1.0) / prox_scale)  # openness: prefer wide berths
    if mode_aware and M > 1:
        # Avoid blending left/right homotopy classes into a middle collision
        # (Mizuta's insight): restrict the average to the best sample's pass-side.
        rob = state[:2].to(device); gd = g2 - rob
        gd = gd / torch.linalg.norm(gd).clamp_min(1e-6)
        perp = torch.stack([-gd[1], gd[0]])
        mid = H1 // 2
        lateral = (states[:, mid, :2] - rob.view(1, 2)) @ perp   # [M] signed pass-side
        best = int(torch.argmin(cost))
        same_side = (lateral >= 0) == (lateral[best] >= 0)
        cost = torch.where(same_side, cost, cost + 1e6)          # drop the other mode
    w = torch.softmax(-(cost - cost.min()) / max(temperature, 1e-6), dim=0)
    action = (w.unsqueeze(1) * ctrl[:, 0, :]).sum(dim=0)
    collision_free = (minclear >= 0.0)
    info = {
        "accept_rate": float(collision_free.float().mean().detach().cpu()),
        "min_clearance": float(minclear.max().detach().cpu()),  # best sample's clearance
        "num_samples": int(M),
    }
    if return_rollouts:
        keep = min(48, M)
        idx = torch.randperm(M, device=device)[:keep]
        info["debug_rollouts"] = {
            "states": states[idx].detach().cpu().numpy(),
            "feasible": collision_free[idx].detach().cpu().numpy(),
            "best_state": states[int(torch.argmin(cost))].detach().cpu().numpy(),
        }
    if return_samples:
        # reward-tilted training data: top-K feasible control sequences + their
        # self-normalized MPPI weights w ∝ exp(-S/λ) (the tilt). q_θ regresses these.
        k = min(96, M)
        topw, topi = torch.topk(w, k)
        info["samples"] = {
            "controls": ctrl[topi].detach().cpu().numpy(),          # [k, H, 2]
            "weights": (topw / topw.sum()).detach().cpu().numpy(),  # [k] tilt weights
        }
    return action, info
