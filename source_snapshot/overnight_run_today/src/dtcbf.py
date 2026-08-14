"""Affine DTCBF barrier + candidate polytope + the VERIFIER (certificate optimization).

The verifier `v_cert` is a per-obstacle convex feasibility check (see ../design/VERIFIER.md):
for some unit normal n and the tightest separating offset b_t = n.c_t - m, with
    h_t = n.(c_t - p_t) - m   (>=0  == containment+separation),
require  h_{t+1} >= (1-gamma) h_t  for some gamma <= gamma_max.
With tightest b this reduces to a CLOSED FORM (no LP): swept over the normal angle.
The closed form is SOUND (never certifies an unsafe trajectory); it may be marginally
conservative (reject some certifiable ones), which is the safe side.
"""
from __future__ import annotations

from typing import Optional

import torch

from dynamics import Env, rollout


def clearances(states: torch.Tensor, env: Env) -> torch.Tensor:
    """Signed clearance min over time & obstacles. states [B,T+1,4] -> [B] (>=0 safe)."""
    p = states[:, :, :2]                                  # [B,T+1,2]
    c = env.obstacle_centers_over_time().to(states.device)   # [T+1,N,2]
    r = env.obstacles[:, 2].to(states.device)            # [N]
    m = r + env.r_robot                                   # [N]
    d = torch.linalg.norm(p[:, :, None, :] - c[None], dim=-1)  # [B,T+1,N]
    sd = d - m[None, None, :]                             # signed dist to inflated boundary
    return sd.amin(dim=(1, 2))                            # [B]


def v_collision(states: torch.Tensor, env: Env, tol: float = 0.0) -> torch.Tensor:
    return clearances(states, env) >= -tol


@torch.no_grad()
def verify(
    states: torch.Tensor,
    env: Env,
    gamma_max: float = 0.7,
    tol: float = 1e-5,
    h_eps: float = 1e-3,
    chunk: int = 8192,
    return_certificate: bool = False,
    **_,
):
    """v_cert via the distance DTCBF (per-step normal = grad of clearance; the repo's decay form).

    For each obstacle j the barrier is the true clearance  h_{j,t} = ||p_t - c_{j,t}|| - (r_j+r_robot).
    The trajectory is certifiable at rate gamma_max iff, for every obstacle:
      (a) h_{j,t} >= 0 for all t            (collision-free / containment), and
      (b) exists gamma_j <= gamma_max with  h_{j,t+1} >= (1-gamma_j) h_{j,t}  for all t   (DTCBF decay).
    The per-step separating tangent planes (normal (p_t-c_t)/||.||) are the time-varying polytope faces;
    their {h >= (1-gamma)^i h_0} nested sets are the "verified polytope level sets".  Dimension-agnostic:
    the same check is O(dim) in 3-D (no normal search / SOCP needed) -- see ../design/VERIFIER.md.

    states: [B,T+1,4].  Returns dict: safe [B] bool, req_gamma [B] (worst-obstacle needed gamma; inf if unsafe),
    and (optionally) cert_normal [B,N,T+1,2] per-step certifying normals.
    """
    device = states.device
    B = states.shape[0]
    c_all = env.obstacle_centers_over_time().to(device)             # [T+1,N,2]
    m_all = (env.obstacles[:, 2] + env.r_robot).to(device)         # [N]
    N = env.n_obs

    safe = torch.ones(B, dtype=torch.bool, device=device)
    req_gamma = torch.full((B,), float("inf"), device=device)

    for s in range(0, B, chunk):
        e = min(B, s + chunk)
        p = states[s:e, :, :2]                                      # [b,T+1,2]
        d = torch.linalg.norm(p[:, :, None, :] - c_all[None], dim=-1)   # [b,T+1,N]
        h = d - m_all[None, None, :]                                # [b,T+1,N]
        contain = h.amin(dim=1) >= -tol                            # [b,N]
        ht, htp1 = h[:, :-1, :], h[:, 1:, :]                       # [b,T,N]
        ratio = htp1 / ht.clamp_min(h_eps)
        need = torch.where(ht > h_eps, 1.0 - ratio, torch.full_like(ratio, -float("inf")))
        req = need.amax(dim=1).clamp_min(0.0)                       # [b,N] needed gamma per obstacle
        feas = contain & (req <= gamma_max)                        # [b,N]
        safe[s:e] = feas.all(dim=1)
        req_obs = torch.where(contain, req, torch.full_like(req, float("inf")))
        req_gamma[s:e] = req_obs.amax(dim=1)                        # worst obstacle (inf if any collision)

    req_gamma = torch.where(safe, req_gamma, torch.full_like(req_gamma, float("inf")))
    out = {"safe": safe, "req_gamma": req_gamma}
    if return_certificate:
        p = states[:, :, :2]
        diff = p[:, :, None, :] - c_all[None]                       # [B,T+1,N,2]
        out["cert_normal"] = diff / torch.linalg.norm(diff, dim=-1, keepdim=True).clamp_min(1e-9)
    return out


@torch.no_grad()
def verify_controls(U: torch.Tensor, env: Env, **kw):
    """Convenience: roll out then verify. Returns (safe[B], info)."""
    states = rollout(U, env)
    info = verify(states, env, **kw)
    info["states"] = states
    return info["safe"], info


# --------------------------------------------------------- candidate polytope (deterministic)

@torch.no_grad()
def build_candidate_polytope(env: Env, heading: Optional[torch.Tensor] = None, sensing: float = 6.0):
    """Conservative half-space set {p : A p <= b} from the START state (the polytope.py analog).

    One tangent half-space per obstacle within sensing range, normal = start->obstacle.
    Returned only for plotting/baseline; the loop trains on verifier certificates, not this.
    """
    p0 = env.x0[:2]
    if heading is None:
        heading = (env.goal - p0)
    heading = heading / heading.norm().clamp_min(1e-9)
    A, b = [], []
    for j in range(env.n_obs):
        c = env.obstacles[j, :2]
        r = env.obstacles[j, 2] + env.r_robot
        d = c - p0
        dist = d.norm().clamp_min(1e-9)
        if dist > sensing:
            continue
        nrm = d / dist                       # outward normal toward obstacle
        # tangent plane to inflated obstacle on the robot side: n.p <= n.c - r
        A.append(nrm)
        b.append((nrm @ c) - r)
    if not A:
        return None
    return torch.stack(A), torch.stack(b)
