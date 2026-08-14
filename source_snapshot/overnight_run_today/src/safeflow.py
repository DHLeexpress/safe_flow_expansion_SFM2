"""Safe Flow Expansion = ACTFLOW transplanted to control/DTCBF (see ../design/SAFEFLOW_GLOSSARY.md).

Loop:  re-extract phi_s^t -> fit sigma_t (Eq.10) -> active exploration (Eq.9 tilt) -> verifier
       -> buffer -> UpdateFlow (signed CFM grad).  Diagnostics D1..D9 logged each eval.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from dynamics import Env, rollout, clip_controls
from dtcbf import verify, clearances
from flow_policy import FlowPolicy
from uncertainty import GPUncertainty
import descriptors as D


# --------------------------------------------------------------------- validity label

@torch.no_grad()
def reaches_goal(states: torch.Tensor, env: Env, radius: float = 1.0) -> torch.Tensor:
    dist = torch.linalg.norm(states[:, :, :2] - env.goal.to(states.device), dim=-1)
    return dist.amin(dim=1) < radius            # passes near goal at some point


@torch.no_grad()
def validity_label(U: torch.Tensor, env: Env, gamma_max: float, n_angles: int):
    """y = v_cert(safe) AND reaches_goal.  Returns (valid[B], safe[B], states, req_gamma)."""
    states = rollout(U, env)
    info = verify(states, env, gamma_max=gamma_max, n_angles=n_angles)
    safe = info["safe"]
    valid = safe & reaches_goal(states, env)
    return valid, safe, states, info["req_gamma"]


# --------------------------------------------------------------------- seed (one leaf)

@torch.no_grad()
def controller_rollout(env: Env, n: int, lateral: torch.Tensor, sigma: float,
                       kp: float = 6.0, kd: float = 4.0, device="cpu", seed: int | None = 0):
    """PD-to-waypoint controller with a lateral bump + noise. Returns realized (U, states).
    lateral [n]: signed lateral bump amplitude (peaks at mid-longitude)."""
    p0 = env.x0[:2].to(device); g = env.goal.to(device)
    d = (g - p0); d = d / d.norm().clamp_min(1e-9)
    e = torch.stack([-d[1], d[0]])
    s = torch.linspace(0, 1, env.T + 1, device=device)
    base = p0[None] + s[:, None] * (g - p0)[None]                  # [T+1,2]
    bump = torch.sin(torch.pi * s)                                  # [T+1]
    p_des = base[None] + lateral[:, None, None] * bump[None, :, None] * e[None, None]  # [n,T+1,2]
    v_des = torch.zeros_like(p_des)
    v_des[:, :-1] = (p_des[:, 1:] - p_des[:, :-1]) / env.dt
    x = env.x0.to(device).expand(n, 4).clone()
    Us, states = [], [x]
    gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    for t in range(env.T):
        p, v = x[:, :2], x[:, 2:4]
        u = kp * (p_des[:, t] - p) + kd * (v_des[:, t] - v)
        noise = torch.randn(n, 2, generator=gen, device=device) if gen is not None \
            else torch.randn(n, 2, device=device)
        u = u + sigma * noise
        u = u.clamp(-env.u_max, env.u_max)
        Us.append(u)
        p_n = p + env.dt * v + 0.5 * env.dt ** 2 * u
        v_n = v + env.dt * u
        x = torch.cat([p_n, v_n], dim=1)
        states.append(x)
    return torch.stack(Us, 1), torch.stack(states, 1)


@torch.no_grad()
def make_seed(env: Env, leaf: str, n_raw: int, gamma_max: float, n_angles: int, device="cpu"):
    """Generate a ONE-leaf safe+goal-reaching seed dataset (the conservative SafeMPPI output)."""
    if env.name == "single":
        rng = {"right": (-1.9, -1.15), "left": (1.15, 1.9)}[leaf]
    else:  # gap: left = above both
        rng = {"left": (1.7, 2.3), "right": (-2.3, -1.7)}[leaf]
    lat = torch.empty(n_raw, device=device).uniform_(*rng)
    U, states = controller_rollout(env, n_raw, lat, sigma=0.6, device=device)
    valid, _, _, _ = validity_label(U, env, gamma_max, n_angles)
    return U[valid]


@torch.no_grad()
def surrounding_proposal(env: Env, n: int, device="cpu", seed: int | None = None):
    """Broad 'surrounding constrained' proposal covering all homotopy leaves (the SafeMPPI-style
    wide sampler). Provides the candidate support that finite-beta Eq.9 may deviate into; the
    sigma-tilt then selects the informative ones and the verifier filters. Returns U [n,T,2]."""
    if n <= 0:
        return torch.empty(0, env.T, 2, device=device)
    lo, hi = env.ylim
    n2 = n // 3                                   # oversample the central band (narrow gaps / straight-through)
    n1 = n - n2
    lat = torch.cat([torch.empty(n1, device=device).uniform_(lo * 0.95, hi * 0.95),
                     torch.empty(n2, device=device).uniform_(-0.6, 0.6)])
    U, _ = controller_rollout(env, n, lat, sigma=0.9, device=device, seed=seed)
    return clip_controls(U, env)


# --------------------------------------------------------------------- Eq.9 exploration

def systematic_resample(weights: torch.Tensor, B: int) -> torch.Tensor:
    w = weights / weights.sum().clamp_min(1e-12)
    cdf = torch.cumsum(w, 0)
    u0 = torch.rand(1, device=w.device) / B
    pts = u0 + torch.arange(B, device=w.device) / B
    idx = torch.searchsorted(cdf, pts.clamp(max=1.0 - 1e-6))
    return idx.clamp(max=w.numel() - 1)


@torch.no_grad()
def active_exploration(policy: FlowPolicy, unc: GPUncertainty, ctx, env: Env,
                       N: int, B: int, beta: float, s: float, nfe: int,
                       temp: float = 1.0, churn: float = 0.0, extra_U=None):
    """Eq.9: candidate pool = FM-policy samples (+ optional broad 'surrounding' pool extra_U),
    tilt by exp(sigma/beta), resample B queries. Returns (U_query, diag)."""
    n_extra = 0 if extra_U is None else extra_U.shape[0]
    U_fm = clip_controls(policy.sample(max(N - n_extra, 0), ctx, nfe=nfe, temp=temp, churn=churn), env)
    U = U_fm if n_extra == 0 else torch.cat([U_fm, extra_U], 0)
    phi = policy.phi_s(U, ctx, s=s)
    sig = unc.sigma(phi)                                    # [pool]
    logw = (sig - sig.max()) / max(beta, 1e-6)
    w = torch.exp(logw.clamp(min=-30, max=30))
    ess = float((w.sum() ** 2) / (w ** 2).sum().clamp_min(1e-12))   # D7
    idx = systematic_resample(w, B)
    diag = {"ESS": ess, "ESS_frac": ess / U.shape[0],
            "sigma_sel_mean": float(sig[idx].mean()), "sigma_all_mean": float(sig.mean()),
            "frac_broad_selected": float((idx >= U_fm.shape[0]).float().mean())}
    return U[idx], diag


# --------------------------------------------------------------------- UpdateFlow

def update_flow(policy, opt, D_pos, D_neg, ctx, steps, batch, alpha, env=None):
    if D_pos.shape[0] == 0:
        return 0.0
    # mode-balanced per-sample weights: give rare modes (e.g. the narrow GAP) equal voice
    wpos = None
    if env is not None:
        with torch.no_grad():
            modes = D.macro_mode(rollout(D_pos, env), env)
        K = D.n_modes(env)
        freq = torch.tensor([max(int((modes == m).sum()), 1) for m in range(K)],
                            device=D_pos.device).float()
        invf = 1.0 / freq
        wpos = invf[modes]; wpos = wpos / wpos.mean()
    policy.train()
    last = 0.0
    for _ in range(steps):
        bi = torch.randint(0, D_pos.shape[0], (min(batch, D_pos.shape[0]),), device=D_pos.device)
        loss = policy.cfm_loss(D_pos[bi], ctx, weights=(wpos[bi] if wpos is not None else None))
        if alpha > 0 and D_neg.shape[0] > 0:
            ni = torch.randint(0, D_neg.shape[0], (min(batch, D_neg.shape[0]),), device=D_neg.device)
            loss = loss - alpha * policy.cfm_loss(D_neg[ni], ctx)
        opt.zero_grad(); loss.backward(); opt.step()
        last = float(loss)
    policy.eval()
    return last


# --------------------------------------------------------------------- config + main loop

@dataclass
class SFConfig:
    rounds: int = 60
    N: int = 256              # candidates per round (Eq.9 pool)
    B: int = 64              # verifier queries per round
    beta: float = 1.0 / 13   # Eq.9 KL trade-off
    s: float = 0.9           # phi_s noise level
    nfe: int = 12            # ODE steps
    explore_temp: float = 1.1   # Eq.9 FM-proposal tail fattening
    explore_churn: float = 0.05 # per-step exploration noise
    rho0: float = 0.7        # initial broad 'surrounding' fraction of the candidate pool
    rho_min: float = 0.05    # final broad fraction (anneal exploration -> exploitation)
    gamma_max: float = 0.7   # DTCBF non-conservativeness ceiling
    n_angles: int = 144      # verifier normal sweep
    kernel: str = "linear"   # uncertainty kernel
    lengthscale: float = 0.4
    lam: float = 1e-2
    alpha: float = 0.12      # negative (unlearning) weight: pushes mass out of inter-mode invalid gap
    warmup_pos: int = 64     # no UpdateFlow until this many valid samples
    inner_steps: int = 200
    batch: int = 128
    lr: float = 2e-4
    eval_every: int = 5
    eval_K: int = 2500       # samples for coverage/validity eval
    nbins: int = 40
    tau: float = 0.01        # generable-set density threshold (paper)
    no_finetune: bool = False  # REC-NF baseline
    no_tilt: bool = False      # REC-F baseline (beta -> inf : uniform resample)


@torch.no_grad()
def evaluate(policy, env, ctx, star_bins, ranges, cfg) -> dict:
    U = clip_controls(policy.sample(cfg.eval_K, ctx, nfe=cfg.nfe), env)
    valid, safe, states, _ = validity_label(U, env, cfg.gamma_max, cfg.n_angles)
    desc = D.descriptor(states, env)
    modes = D.macro_mode(states, env)
    desc_valid = desc[valid]
    cov = D.coverage(desc_valid, star_bins, ranges, cfg.nbins, cfg.tau)
    mc, probs = D.mode_coverage(modes[valid], cfg.eval_K, env)
    vendi = D.vendi_score(desc_valid)
    return {
        "validity": float(valid.float().mean()),
        "safe_rate": float(safe.float().mean()),
        "coverage": cov,
        "mode_coverage": mc,
        "mode_probs": probs,
        "vendi": vendi,
        "n_valid": int(valid.sum()),
    }


def run_safeflow(env: Env, ctx, policy: FlowPolicy, star_bins, ranges, cfg: SFConfig,
                 device="cpu", log=print, snapshot_rounds=None, init_pos=None):
    import copy
    snapshot_rounds = set(snapshot_rounds or [])
    snapshots = {}
    unc = GPUncertainty(kernel=cfg.kernel, lengthscale=cfg.lengthscale,
                        lam=cfg.lam, normalize=True)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    # Optional seed buffer D_0 = verified-safe demos (paradigm's D_0); default = empty (unchanged).
    D_pos = (init_pos.to(device) if init_pos is not None and init_pos.numel()
             else torch.empty(0, env.T, 2, device=device))
    D_neg = torch.empty(0, env.T, 2, device=device)
    history = []

    for t in range(cfg.rounds):
        # 1) refit sigma_t on CURRENT phi_s^t over the buffer (designs queried so far)
        buf = torch.cat([D_pos, D_neg], 0) if (D_pos.numel() + D_neg.numel()) else None
        phi_buf = policy.phi_s(buf, ctx, s=cfg.s) if buf is not None else None
        unc.set_buffer(phi_buf)

        # 2) Eq.9 active exploration over (FM pool + broad 'surrounding' pool), tilted by sigma.
        #    REC-F: no sigma tilt (beta->inf, uniform select) but SAME broad pool -> isolates Eq.9/10 value.
        beta = 1e9 if cfg.no_tilt else cfg.beta
        temp = 1.0 if cfg.no_tilt else cfg.explore_temp
        churn = 0.0 if cfg.no_tilt else cfg.explore_churn
        rho = cfg.rho0 + (cfg.rho_min - cfg.rho0) * (t / max(cfg.rounds - 1, 1))
        broad_U = surrounding_proposal(env, int(rho * cfg.N), device=device, seed=1000 + t)
        U_q, eq9 = active_exploration(policy, unc, ctx, env, cfg.N, cfg.B, beta, cfg.s, cfg.nfe,
                                      temp=temp, churn=churn, extra_U=broad_U)
        eq9["rho"] = rho

        # 3) verifier query -> labels
        valid, safe, states, req_g = validity_label(U_q, env, cfg.gamma_max, cfg.n_angles)
        D_pos = torch.cat([D_pos, U_q[valid]], 0)
        D_neg = torch.cat([D_neg, U_q[~valid]], 0)
        yield_rate = float(valid.float().mean())          # D9

        # 4) UpdateFlow (warm-up gate; REC-NF skips)
        upd_loss = 0.0
        if (not cfg.no_finetune) and D_pos.shape[0] >= cfg.warmup_pos:
            upd_loss = update_flow(policy, opt, D_pos, D_neg, ctx,
                                   cfg.inner_steps, cfg.batch, cfg.alpha, env=env)

        # 5) eval + diagnostics
        if t % cfg.eval_every == 0 or t == cfg.rounds - 1:
            m = evaluate(policy, env, ctx, star_bins, ranges, cfg)
            # Eq.10 diagnostics on a fresh policy batch
            U_fresh = clip_controls(policy.sample(min(512, cfg.eval_K), ctx, nfe=cfg.nfe), env)
            phi_fresh = policy.phi_s(U_fresh, ctx, s=cfg.s)
            diag = unc.diagnostics(phi_buf, phi_fresh)
            rec = {"round": t, "n_pos": int(D_pos.shape[0]), "n_neg": int(D_neg.shape[0]),
                   "yield": yield_rate, "upd_loss": upd_loss, **m, **eq9, **diag}
            history.append(rec)
            log(f"[{env.name} r{t:03d}] cov={m['coverage']:.2f} val={m['validity']:.2f} "
                f"modecov={m['mode_coverage']:.2f} vendi={m['vendi']:.2f} "
                f"yield={yield_rate:.2f} ESSf={eq9['ESS_frac']:.2f} "
                f"sig(fresh={diag['sigma_fresh_mean']:.3f}"
                f"{'/buf=' + format(diag.get('sigma_buffer_mean', float('nan')), '.3f') if 'sigma_buffer_mean' in diag else ''}) "
                f"npos={D_pos.shape[0]}")
        if t in snapshot_rounds or t == cfg.rounds - 1:
            snapshots[t] = copy.deepcopy(policy.state_dict())
    return policy, history, snapshots
