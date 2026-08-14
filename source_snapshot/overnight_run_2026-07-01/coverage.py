"""Swappable COVERAGE module (behavior-space coverage — the control space is too large; see COVERAGE.md).

Metrics on the ROLLED-OUT paths, normalized by the verifier-reachable-safe set Ω* (a broad proposal gated
by the SAME validity module):
  * spatial_coverage (headline) — free-cells covered by valid FM paths ∩ Ω* cells / |Ω* cells|
  * mode_coverage — fraction of Ω*'s behavior MODES the FM reproduces
      gap:    center / upper / lower (which part of the corridor the thread passes)
      slalom: around_up / around_down / weave (go around vs weave between)
  * vendi — diversity of a low-dim path descriptor
Swap ideas by editing `mode_of` / the descriptor; the loop only calls `evaluate` / `build_omega_star`.
"""
from __future__ import annotations

import numpy as np

import _paths  # noqa: F401
from di_grid_viz import di_step
import validity as VAL


def _cross_y(path, xline):
    p = np.asarray(path, float)
    for i in range(len(p) - 1):
        if (p[i, 0] - xline) * (p[i + 1, 0] - xline) <= 0 and p[i + 1, 0] != p[i, 0]:
            t = (xline - p[i, 0]) / (p[i + 1, 0] - p[i, 0])
            return float(p[i, 1] + t * (p[i + 1, 1] - p[i, 1]))
    return None


def mode_of(path, env):
    obs = env.obstacles.detach().cpu().numpy()
    if env.name == "narrow_gap":
        off = float(obs[0, 1]); r = float(obs[0, 2])
        cy = _cross_y(path, float(obs[0, 0]))
        if cy is None:
            return "none"
        band = (off - r) * 0.45
        return "center" if abs(cy) <= band else ("upper" if cy > 0 else "lower")
    if env.name == "slalom":
        ax, ay = float(obs[0, 0]), float(obs[0, 1])          # A = UPPER obstacle (ay>0)
        bx, by = float(obs[1, 0]), float(obs[1, 1])          # B = LOWER obstacle (by<0)
        yA, yB = _cross_y(path, ax), _cross_y(path, bx)
        if yA is None or yB is None:
            return "none"
        # collision-free => yA is outside A's disk, so (yA>ay) <=> passes ABOVE the upper obstacle; likewise yB,B.
        overA, overB = yA > ay, yB > by
        if overA and overB:
            return "around_up"      # truly OVER THE TOP of both obstacles (highest, hardest — after weave)
        if (not overA) and (not overB):
            return "around_down"    # under the bottom of both
        if (not overA) and overB:
            return "weave"          # below upper-A, above lower-B: threads the diagonal gap
        return "over_under"         # above A then below B (rare long detour)
    return "single_" + ("up" if np.asarray(path)[:, 1].max() > 0 else "down")


def spatial_cells(paths, env, cell=0.2):
    lo_x, lo_y = env.xlim[0], env.ylim[0]
    cells = set()
    for path in paths:
        p = np.asarray(path, float)
        ix = np.floor((p[:, 0] - lo_x) / cell).astype(int)
        iy = np.floor((p[:, 1] - lo_y) / cell).astype(int)
        cells |= set(zip(ix.tolist(), iy.tolist()))
    return cells


def descriptor(path, env, n_slice=6):
    """Lateral offset at n_slice longitudinal slices (chord-frame) — low-dim behavior descriptor."""
    p = np.asarray(path, float)
    xs = np.linspace(env.xlim[0] + 0.5, env.goal[0].item(), n_slice)
    out = []
    for x in xs:
        cy = _cross_y(p, x)
        out.append(cy if cy is not None else 0.0)
    return np.array(out)


def vendi_score(descs, ell=0.6):
    if len(descs) < 2:
        return float(len(descs))
    X = np.stack(descs)
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    K = np.exp(-d2 / (2 * ell ** 2)) / len(X)
    ev = np.clip(np.linalg.eigvalsh(K), 1e-12, None)
    ev = ev / ev.sum()
    return float(np.exp(-(ev * np.log(ev)).sum()))


# ----------------------------------------------------------------- broad proposal → Ω*
def broad_rollouts(env, n, seed=0):
    """Diverse full trajectories (PD-to-waypoint with random lateral + weave).
    Returns (states [n,T+1,4], controls [n,T,2]) so expansion can extract windows."""
    rng = np.random.RandomState(seed)
    x0 = env.x0.detach().cpu().numpy().astype(np.float32)
    goal = env.goal.detach().cpu().numpy()
    obs = env.obstacles.detach().cpu().numpy()
    T, dt, umax = env.T, env.dt, float(env.u_max)
    s = np.linspace(0, 1, T + 1)
    x_des = x0[0] + s * (goal[0] - x0[0])
    S, U = [], []
    for _ in range(n):
        pick = rng.rand()
        if env.name == "slalom" and pick < 0.32:               # dedicated gap-threading WEAVE proposal
            ax, bx = float(obs[0, 0]), float(obs[1, 0])
            xmid, xsc = 0.5 * (ax + bx), max(0.3, 0.35 * (bx - ax))
            direction = rng.choice([-1.0, 1.0], p=[0.25, 0.75])  # +1: below-A→above-B (the weave)
            D = rng.uniform(0.7, 1.3)
            # tanh threads the gap; sin(pi s) window returns the path to the goal line (y=0) at the ends
            y_des = direction * D * np.tanh((x_des - xmid) / xsc) * np.sin(np.pi * s)
        elif env.name == "slalom" and pick < 0.62:             # dedicated OVER-THE-TOP proposal (around_up)
            D = rng.uniform(1.4, 2.1)                            # high single arc clears BOTH obstacles above
            y_des = D * np.sin(np.pi * s)
        else:
            lat = rng.uniform(-1.8, 1.8)
            wamp = rng.uniform(-0.8, 0.8)
            wfreq = rng.choice([1, 2, 3])
            phase = rng.uniform(0, np.pi)
            y_des = lat * np.sin(np.pi * s) + wamp * np.sin(wfreq * 2 * np.pi * s + phase)
        st = x0.copy(); states = [st.copy()]; ctrls = []
        for t in range(T):
            p_des = np.array([x_des[t + 1], y_des[t + 1]])
            v_des = np.array([x_des[t + 1] - x_des[t], y_des[t + 1] - y_des[t]]) / dt
            u = np.clip(6.0 * (p_des - st[:2]) + 4.0 * (v_des - st[2:4]) + rng.randn(2) * 0.4, -umax, umax)
            st = di_step(st, u, dt); states.append(st.copy()); ctrls.append(u.astype(np.float32))
        S.append(np.array(states, np.float32)); U.append(np.array(ctrls, np.float32))
    return np.array(S), np.array(U)


def broad_paths(env, n, seed=0):
    S, _ = broad_rollouts(env, n, seed=seed)
    return S[:, :, :2]


def build_omega_star(env, gamma_max, n=1500, cell=0.2, seed=7, log=print):
    """Broad proposal → validity gate → reachable-safe cells / modes / descriptors (the denominator)."""
    paths = broad_paths(env, n, seed=seed)
    valid = [p for p in paths if VAL.is_valid(p, env, gamma_max)]
    star_cells = spatial_cells(valid, env, cell) if valid else set()
    star_modes = set(mode_of(p, env) for p in valid)
    log(f"[omega*] {len(valid)}/{n} broad paths valid  cells={len(star_cells)}  modes={sorted(star_modes)}")
    return dict(cells=star_cells, modes=star_modes, n_valid=len(valid))


def evaluate(fm_paths, env, gamma, star, cell=0.2):
    """Coverage of the FM's paths (at one γ) vs Ω*. fm_paths: list of rolled-out [T+1,2]."""
    valid = [p for p in fm_paths if VAL.is_valid(p, env, gamma)]
    n = len(fm_paths)
    theta_cells = spatial_cells(valid, env, cell) if valid else set()
    spatial = len(theta_cells & star["cells"]) / max(len(star["cells"]), 1)
    theta_modes = set(mode_of(p, env) for p in valid)
    modecov = len(theta_modes & star["modes"]) / max(len(star["modes"]), 1)
    vendi = vendi_score([descriptor(p, env) for p in valid]) if valid else 0.0
    return {"validity": len(valid) / max(n, 1), "spatial_coverage": spatial,
            "mode_coverage": modecov, "vendi": vendi, "n_valid": len(valid),
            "modes": sorted(theta_modes)}
