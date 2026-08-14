"""Generate reward-tilted training data for the learned proposal q_θ(U|o,γ).

Runs mirror-MPPI on real UCY scenes; at each step records a TRANSLATION-INVARIANT
context o (robot velocity, goal-relative, nearest-K pedestrian relative pos/vel),
the safety knob γ, and the top-K feasible control sequences with their
self-normalized MPPI weights w ∝ exp(-S/λ) (the reward tilt). Trained with the
energy/reward-weighted conditional flow-matching loss (EFM, arXiv:2503.04975).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from cfm_mppi.safegpc_adapter.mirror_sampler import mirror_mppi_action
from cfm_mppi.evaluation.render_validation_comparison import (
    get_parser as _rp, _make_scene, _frame_obstacles, _frame_velocities)

DT = 0.1
KOBS = 6
ODIM = 4 + 4 * KOBS  # [vx,vy, gdx,gdy, K*(dx,dy,dvx,dvy)]


def _obs_features(st, goal, obs, vel):
    p = st[:2]; v = st[2:4]
    gd = goal[:2] - p
    feats = [v[0], v[1], gd[0], gd[1]]
    if obs.shape[0]:
        d = np.linalg.norm(obs[:, :2] - p[None, :], axis=1) - obs[:, 2]
        order = np.argsort(d)[:KOBS]
        for j in order:
            rel = obs[j, :2] - p
            rv = vel[j] if vel.shape[0] > j else np.zeros(2)
            feats += [rel[0], rel[1], rv[0], rv[1]]
    while len(feats) < ODIM:
        feats += [8.0, 8.0, 0.0, 0.0]  # far/no obstacle
    return np.asarray(feats[:ODIM], dtype=np.float32)


def _di(s, a, dt=DT):
    x = s.copy(); x[0] += dt*s[2]+0.5*dt*dt*a[0]; x[1] += dt*s[3]+0.5*dt*dt*a[1]
    x[2] += dt*a[0]; x[3] += dt*a[1]; return x


def run(cli):
    dev = torch.device(cli.device)
    O, C, W, G = [], [], [], []   # contexts[S,ODIM], controls[S,K,H,2], weights[S,K], gamma[S]
    for ei, ep in enumerate(range(cli.ep_start, cli.ep_end)):
        b = _rp().parse_args([]); b.dataset = "ucy"; b.dynamics = "doubleintegrator"
        b.pedestrian_source = "validation"; b.episode = ep; b.steps = cli.steps
        s0, goal, obs, vel, _ = _make_scene(b)
        for g in cli.gammas:
            st = s0.astype(np.float32).copy()
            for t in range(0, cli.steps, cli.stride):
                ob = _frame_obstacles(obs, t); ve = _frame_velocities(vel, t)
                a, info = mirror_mppi_action(
                    torch.tensor(st, device=dev), torch.tensor(goal, device=dev),
                    torch.tensor(ob, device=dev), torch.tensor(ve, device=dev),
                    horizon=cli.horizon, num_samples=cli.samples, gamma=g, eta=1.0,
                    dual_sigma=1.4, margin_gain=0.25, temperature=cli.lam, clear_w=80.0,
                    terminal_w=15.0, sensing_range=6.0, seed=t, device=dev, return_samples=True)
                smp = info["samples"]
                ctrl = smp["controls"]; w = smp["weights"]  # [k,H,2], [k]
                k = min(cli.keep, ctrl.shape[0])
                # advance the realized state to the next stored step (apply a few steps)
                O.append(_obs_features(st, goal, ob, ve))
                C.append(ctrl[:k]); W.append(w[:k] / w[:k].sum()); G.append(np.float32(g))
                for _ in range(cli.stride):
                    st = _di(st, a.detach().cpu().numpy());
        if (ei + 1) % 5 == 0:
            print(f"[tilt-gen] ep {ep}  items={len(O)}", flush=True)

    O = np.stack(O); C = np.stack(C); W = np.stack(W); G = np.stack(G)
    out = Path(cli.output); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"context": torch.from_numpy(O), "controls": torch.from_numpy(C),
                "weights": torch.from_numpy(W), "gamma": torch.from_numpy(G),
                "meta": {"odim": ODIM, "kobs": KOBS, "horizon": cli.horizon, "lam": cli.lam,
                         "ep_range": [cli.ep_start, cli.ep_end], "gammas": cli.gammas}}, out)
    print(f"[tilt-gen] saved {len(O)} steps x {C.shape[1]} samples -> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ep-start", type=int, default=0)
    p.add_argument("--ep-end", type=int, default=60)
    p.add_argument("--gammas", nargs="+", type=float, default=[0.2, 0.5, 0.8])
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--horizon", type=int, default=25)
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--keep", type=int, default=48)
    p.add_argument("--lam", type=float, default=0.3, help="MPPI temperature lambda (tilt sharpness)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default="dataset/tilted/tilt_data.pt")
    run(p.parse_args())


if __name__ == "__main__":
    main()
