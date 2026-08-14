"""Trajectory descriptors + coverage / mode-coverage / Vendi (see ../design/METRICS.md).

The design U is too high-dim to histogram, so we measure coverage on a low-dim descriptor that
encodes the homotopy class (passing side) -- the thing that makes left/right/middle distinct.
"""
from __future__ import annotations

import torch


def _chord_frame(env):
    p0 = env.x0[:2]
    g = env.goal
    d = (g - p0); d = d / d.norm().clamp_min(1e-9)
    e = torch.stack([-d[1], d[0]])            # left-perpendicular (+ = left of motion)
    return p0, d, e


@torch.no_grad()
def descriptor(states: torch.Tensor, env) -> torch.Tensor:
    """states [B,T+1,4] -> lateral offset at each obstacle's longitudinal slice  [B, N_obs]."""
    p = states[:, :, :2]                       # [B,T+1,2]
    p0, d, e = _chord_frame(env)
    p0 = p0.to(states.device); d = d.to(states.device); e = e.to(states.device)
    lon = torch.einsum("btd,d->bt", p - p0, d)        # [B,T+1] longitudinal coord
    lat = torch.einsum("btd,d->bt", p - p0, e)        # [B,T+1] lateral coord
    outs = []
    for j in range(env.n_obs):
        Lj = (env.obstacles[j, :2].to(states.device) - p0) @ d
        idx = (lon - Lj).abs().argmin(dim=1)          # [B] closest step to obstacle longitude
        outs.append(lat[torch.arange(p.shape[0], device=states.device), idx])
    return torch.stack(outs, dim=1)                    # [B,N_obs]


@torch.no_grad()
def macro_mode(states: torch.Tensor, env) -> torch.Tensor:
    """Human-meaningful mode label per trajectory.
       single: 0=LEFT(+y), 1=RIGHT(-y)
       gap:    0=LEFT(above top), 1=GAP(middle), 2=RIGHT(below bottom)
    """
    lat = descriptor(states, env)                      # [B,N]
    if env.name == "single":
        return (lat[:, 0] < 0).long()                  # 0 left(+), 1 right(-)
    # gap: classify by lateral at the (shared) slice vs obstacle y-centers
    ys = env.obstacles[:, 1].to(states.device)
    y_top, y_bot = ys.max(), ys.min()
    ell = lat.mean(dim=1)
    mode = torch.full_like(ell, 1, dtype=torch.long)   # default GAP
    mode[ell >= y_top] = 0                              # LEFT (above both)
    mode[ell <= y_bot] = 2                              # RIGHT (below both)
    return mode


def n_modes(env) -> int:
    return 2 if env.name == "single" else 3


def mode_names(env):
    return ["LEFT", "RIGHT"] if env.name == "single" else ["LEFT", "GAP", "RIGHT"]


# ------------------------------------------------------------------- coverage

def _occupied_bins(desc: torch.Tensor, ranges, nbins: int, min_count: int):
    """desc [M,D] -> set of occupied bin tuples (count >= min_count)."""
    if desc.shape[0] == 0:
        return set()
    D = desc.shape[1]
    idxs = []
    for k in range(D):
        lo, hi = ranges[k]
        b = ((desc[:, k] - lo) / (hi - lo) * nbins).long().clamp(0, nbins - 1)
        idxs.append(b)
    keys = idxs[0]
    for k in range(1, D):
        keys = keys * nbins + idxs[k]
    uniq, counts = torch.unique(keys, return_counts=True)
    return set(int(u) for u, c in zip(uniq.tolist(), counts.tolist()) if c >= min_count)


def coverage(desc_safe: torch.Tensor, star_bins: set, ranges, nbins: int, tau: float):
    """Fraction of reachable-safe bins B* populated by the policy's safe samples (>= tau density)."""
    if len(star_bins) == 0:
        return 0.0
    M = desc_safe.shape[0]
    min_count = max(1, int(tau * M)) if M > 0 else 1
    theta_bins = _occupied_bins(desc_safe, ranges, nbins, min_count)
    return len(theta_bins & star_bins) / len(star_bins)


def build_star_bins(desc_star_safe: torch.Tensor, ranges, nbins: int, min_count: int = 1):
    return _occupied_bins(desc_star_safe, ranges, nbins, min_count)


def mode_coverage(modes_safe: torch.Tensor, total_sampled: int, env, p_min: float = 0.01):
    """Fraction of macro-modes the policy generates with prob >= p_min (safe count / total)."""
    K = n_modes(env)
    if total_sampled == 0:
        return 0.0, [0.0] * K
    probs = []
    for m in range(K):
        probs.append(float((modes_safe == m).sum()) / total_sampled)
    covered = sum(1 for p in probs if p >= p_min)
    return covered / K, probs


# ------------------------------------------------------------------- diversity (Vendi)

@torch.no_grad()
def vendi_score(desc_safe: torch.Tensor, ell: float = 0.5, max_n: int = 512) -> float:
    M = desc_safe.shape[0]
    if M < 2:
        return float(M)
    x = desc_safe
    if M > max_n:
        sel = torch.randperm(M, device=x.device)[:max_n]
        x = x[sel]
    d2 = torch.cdist(x, x) ** 2
    K = torch.exp(-d2 / (2 * ell ** 2))
    K = K / K.shape[0]                                  # normalize so trace=1
    ev = torch.linalg.eigvalsh(K).clamp_min(1e-12)
    ev = ev / ev.sum()
    H = -(ev * ev.log()).sum()
    return float(torch.exp(H))
