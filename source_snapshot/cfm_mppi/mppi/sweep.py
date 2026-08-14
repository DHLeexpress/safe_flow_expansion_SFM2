"""Reusable SafeMPPI evaluation + parameter sweep (the MPPI framework is shared across stages).

Measures SafeMPPI success rate (goal-reach + collision-free) over N eval episodes, and — gated on success rate
<= a threshold — sweeps the key knobs (num_samples first, then noise_sigma / horizon / lambda) to clear the bar.
Single-integrator by default; works for double-integrator too. GPU-friendly.

  LD_PRELOAD=.../libstdc++.so.6 python -m cfm_mppi.mppi.sweep --dataset ucy --episodes 100 --gamma 0.5
"""
from __future__ import annotations
import argparse, json, os
from copy import deepcopy
import numpy as np
import torch

from cfm_mppi.safegpc_adapter.safemppi import SafeMPPIAdapter
from cfm_mppi.evaluation.render_validation_comparison import get_parser as _vparser, _make_scene

DT = 0.1
R_ROBOT = 0.2

# Single-integrator production base config (obstacles inflated by safety_margin for the barrier).
DEFAULT_CFG = dict(
    horizon=10, dt=DT, num_samples=200, noise_sigma=(0.4, 0.4),
    u_min=(-2.0, -2.0), u_max=(2.0, 2.0), safety_margin=0.5, temperature=1.0,
    dynamics_type="singleintegrator", use_ho_barrier=False, eta=0.0,
    use_guidance=True, use_aniso_cov=True, barrier_topk=0, barrier_activation_radius=3.5,
)

# Isolation config (Step-2 plot): guidance/aniso OFF so the ONLY gamma-dependent mechanism is the polytope
# rejection -> isolates whether MPPI temperature lets gamma actually move the executed trajectory.
STEP2_CFG = dict(DEFAULT_CFG, safety_margin=0.0, use_guidance=False, use_aniso_cov=True)

GAMMAS = [0.1, 0.3, 0.5, 0.7, 1.0]


def _si_step(state, a, dt):
    s = state.copy(); s[0] += dt * a[0]; s[1] += dt * a[1]; s[2] = a[0]; s[3] = a[1]; return s


def velocity_inflate(ob, vel, robot_xy, kappa, tau):
    """Suggestion 1: inflate each obstacle radius by kappa*tau*max(0, approach speed toward the robot) so
    pedestrians heading at the robot get a larger margin (velocity-predictive). kappa=0 -> unchanged."""
    if kappa <= 0.0 or ob.shape[0] == 0:
        return ob
    rel = robot_xy[None, :] - ob[:, :2]; dist = np.linalg.norm(rel, axis=1, keepdims=True)
    m_to_robot = rel / np.clip(dist, 1e-9, None)                # obstacle -> robot
    approach = np.maximum(0.0, np.sum(m_to_robot * vel, axis=1))  # obstacle speed toward the robot
    out = ob.copy(); out[:, 2] = ob[:, 2] + kappa * tau * approach
    return out


def _load(dataset, ep, steps):
    a = _vparser().parse_args([])
    a.dataset = dataset; a.pedestrian_source = "validation"; a.dynamics = "doubleintegrator"
    a.steps = steps; a.pedestrian_radius = 0.5; a.episode = ep; a.seed = ep
    s0, goal, obs, vel, _ = _make_scene(a)
    s0 = np.asarray(s0, float).reshape(-1)
    s0 = np.array([s0[0], s0[1], 0.0, 0.0]) if s0.shape[0] >= 2 else np.zeros(4)
    return s0.astype(np.float32), np.asarray(goal, float).reshape(2), np.asarray(obs, float), np.asarray(vel, float)


@torch.no_grad()
def run_episode(cfg, dataset, ep, steps, gamma, dev):
    cfg = dict(cfg); kappa = float(cfg.pop("predict_gain", 0.0)); tau = float(cfg.get("horizon", 10)) * DT
    s0, goal, obs, vel = _load(dataset, ep, steps)
    adapter = SafeMPPIAdapter(**cfg)
    state = s0.copy(); T = obs.shape[0]
    min_clear = np.inf; reached = False; comp = steps
    for t in range(steps):
        ob = obs[min(t, T - 1)]; vl = vel[min(t, vel.shape[0] - 1)]
        ok = ~np.isnan(ob[:, :2]).any(1); ob = ob[ok]; vl = vl[ok] if vl.shape[0] == ok.shape[0] else vl
        ob_plan = velocity_inflate(ob, vl, state[:2], kappa, tau)
        a, _ = adapter.plan(
            torch.tensor(state, dtype=torch.float32, device=dev),
            torch.tensor(goal, dtype=torch.float32, device=dev),
            torch.tensor(ob_plan, dtype=torch.float32, device=dev),
            gamma=gamma, obstacle_velocities=torch.tensor(vl, dtype=torch.float32, device=dev),
            seed=ep * 100000 + t,
        )
        state = _si_step(state, a.detach().cpu().numpy(), DT)
        if ob.shape[0]:
            cl = float(np.min(np.linalg.norm(ob[:, :2] - state[:2], axis=1) - ob[:, 2] - R_ROBOT))
            min_clear = min(min_clear, cl)
        if not reached and np.linalg.norm(state[:2] - goal) < 0.5:
            reached = True; comp = t + 1
    collided = min_clear < 0.0
    success = bool(reached and not collided)
    return dict(success=success, reached=reached, collided=collided,
                min_clear=float(min_clear), comp_step=comp)


def evaluate(cfg, dataset, n_eps, gamma, dev, verbose=False):
    rows = [run_episode(cfg, dataset, ep, 80, gamma, dev) for ep in range(n_eps)]
    sr = float(np.mean([r["success"] for r in rows]))
    rr = float(np.mean([r["reached"] for r in rows]))
    cr = float(np.mean([r["collided"] for r in rows]))
    mc = float(np.mean([r["min_clear"] for r in rows]))
    cs = float(np.mean([r["comp_step"] for r in rows]))
    if verbose:
        print(f"    success={sr*100:.1f}% reach={rr*100:.1f}% collide={cr*100:.1f}% "
              f"mean_clear={mc:.2f} mean_comp={cs:.1f}", flush=True)
    return dict(success_rate=sr, reach_rate=rr, collide_rate=cr, mean_clear=mc, mean_comp=cs)


@torch.no_grad()
def _rollout_full(cfg, dataset, ep, gamma, dev, steps=80):
    """Run one episode, return the full 80-step trajectory (held at goal after reach) + metrics."""
    cfg = dict(cfg); kappa = float(cfg.pop("predict_gain", 0.0)); tau = float(cfg.get("horizon", 10)) * DT
    s0, goal, obs, vel = _load(dataset, ep, steps)
    adapter = SafeMPPIAdapter(**cfg); state = s0.copy(); T = obs.shape[0]
    traj = [state[:2].copy()]; min_clear = np.inf; reached = False; comp = steps
    for t in range(steps):
        if not reached:
            ob = obs[min(t, T - 1)]; vl = vel[min(t, vel.shape[0] - 1)]
            ok = ~np.isnan(ob[:, :2]).any(1); ob = ob[ok]; vl = vl[ok] if vl.shape[0] == ok.shape[0] else vl
            ob_plan = velocity_inflate(ob, vl, state[:2], kappa, tau)
            a, _ = adapter.plan(
                torch.tensor(state, dtype=torch.float32, device=dev),
                torch.tensor(goal, dtype=torch.float32, device=dev),
                torch.tensor(ob_plan, dtype=torch.float32, device=dev),
                gamma=gamma, obstacle_velocities=torch.tensor(vl, dtype=torch.float32, device=dev),
                seed=ep * 100000 + t)
            state = _si_step(state, a.detach().cpu().numpy(), DT)
            if ob.shape[0]:
                min_clear = min(min_clear, float(np.min(np.linalg.norm(ob[:, :2] - state[:2], axis=1) - ob[:, 2] - R_ROBOT)))
            if np.linalg.norm(state[:2] - goal) < 0.5:
                reached = True; comp = t + 1
        traj.append(state[:2].copy())
    success = bool(reached and min_clear >= 0.0)
    return np.array(traj), float(min_clear), int(comp), success


def gamma_study(cfg, dataset, n_eps, dev, gammas=GAMMAS):
    """PROPER MEASURES of whether gamma moves the trajectory:
       - traj_spread : mean over episodes of mean pairwise per-step L2 distance between the per-gamma trajectories
       - clr_range / comp_range : spread of min-clearance / completion-step across gamma (averaged over episodes)
       - corr_g_clear / corr_g_comp : Spearman corr(gamma, clearance|completion) -- should be NEGATIVE & strong
         if gamma works as intended (higher gamma -> less conservative -> lower clearance, faster).
    """
    from scipy.stats import spearmanr
    traj_spreads, clr_ranges, comp_ranges = [], [], []
    g_all, clr_all, comp_all, succ_all = [], [], [], []
    for ep in range(n_eps):
        trs, clrs, comps = {}, {}, {}
        for g in gammas:
            tr, mc, cp, ok = _rollout_full(cfg, dataset, ep, g, dev)
            trs[g] = tr; clrs[g] = mc; comps[g] = cp
            g_all.append(g); clr_all.append(mc); comp_all.append(cp); succ_all.append(ok)
        # pairwise per-step trajectory distance across gamma
        gs = list(gammas); ds = []
        for i in range(len(gs)):
            for j in range(i + 1, len(gs)):
                ds.append(float(np.mean(np.linalg.norm(trs[gs[i]] - trs[gs[j]], axis=1))))
        traj_spreads.append(np.mean(ds))
        cl = [clrs[g] for g in gammas]; cp = [comps[g] for g in gammas]
        clr_ranges.append(max(cl) - min(cl)); comp_ranges.append(max(cp) - min(cp))
    cg = spearmanr(g_all, clr_all).correlation
    cc = spearmanr(g_all, comp_all).correlation
    return dict(traj_spread=float(np.mean(traj_spreads)), clr_range=float(np.mean(clr_ranges)),
                comp_range=float(np.mean(comp_ranges)), corr_g_clear=float(cg), corr_g_comp=float(cc),
                success_rate=float(np.mean(succ_all)))


def sweep_param(base, param, values, dataset, n_eps, gamma, dev):
    out = []
    for v in values:
        cfg = deepcopy(base)
        cfg[param] = (v, v) if param == "noise_sigma" else v
        print(f"  {param}={v}", flush=True)
        m = evaluate(cfg, dataset, n_eps, gamma, dev, verbose=True)
        out.append({param: v, **m})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ucy")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--mode", default="gamma-temp", choices=["gamma-temp", "success"])
    ap.add_argument("--temps", nargs="+", type=float, default=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0])
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--threshold", type=float, default=0.99)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="overnight_run_2026-06-28/figures/mppi_sweep.json")
    args = ap.parse_args()
    dev = args.device

    if args.mode == "gamma-temp":
        # Does MPPI TEMPERATURE make the executed trajectory depend on gamma? (isolation config: guidance off)
        print(f"[gamma-temp] {args.episodes} {args.dataset} episodes, gammas={GAMMAS}, isolation cfg (guidance off)")
        rows = []
        for temp in args.temps:
            cfg = deepcopy(STEP2_CFG); cfg["temperature"] = temp
            m = gamma_study(cfg, args.dataset, args.episodes, dev)
            rows.append({"temperature": temp, **m})
            print(f"  T={temp:<5}: traj_spread={m['traj_spread']:.3f}  clr_range={m['clr_range']:.2f}  "
                  f"comp_range={m['comp_range']:.1f}  corr(γ,clear)={m['corr_g_clear']:+.2f}  "
                  f"corr(γ,comp)={m['corr_g_comp']:+.2f}  success={m['success_rate']*100:.0f}%", flush=True)
        # best = most gamma-separation (traj spread) with the intended monotone sign and non-trivial success
        cand = [r for r in rows if r["corr_g_clear"] < -0.15 and r["success_rate"] >= 0.3] or rows
        best = max(cand, key=lambda r: r["traj_spread"])
        print(f"[best temperature] T={best['temperature']}: traj_spread={best['traj_spread']:.3f} "
              f"corr(γ,clear)={best['corr_g_clear']:+.2f} success={best['success_rate']*100:.0f}%")
        result = {"mode": "gamma-temp", "cfg": {k: v for k, v in STEP2_CFG.items()}, "rows": rows, "best": best}
    else:
        base = deepcopy(DEFAULT_CFG)
        print(f"[success] {args.episodes} {args.dataset} episodes, gamma={args.gamma}")
        base_m = evaluate(base, args.dataset, args.episodes, args.gamma, dev, verbose=True)
        result = {"mode": "success", "base_cfg": {k: v for k, v in base.items()}, "base_metrics": base_m, "sweeps": {}}
        if base_m["success_rate"] <= args.threshold:
            result["sweeps"]["num_samples"] = sweep_param(base, "num_samples", [64, 128, 200, 400, 800],
                                                          args.dataset, args.episodes, args.gamma, dev)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
