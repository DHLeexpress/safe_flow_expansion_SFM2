"""Iterate Mirror-MPPI hyperparameters to maximize accept-rate + success at low
collision. Sharded so two instances run on two GPUs. Objective:
score = success_rate - 3*collision_rate  (+ accept_rate reported).

  CUDA_VISIBLE_DEVICES=0 python -m cfm_mppi.evaluation.sweep_mirror_params --shard 0 --nshards 2 --num-configs 24 --val-eps 130 12
  CUDA_VISIBLE_DEVICES=1 python -m cfm_mppi.evaluation.sweep_mirror_params --shard 1 --nshards 2 --num-configs 24 --val-eps 130 12
then: python -m cfm_mppi.evaluation.sweep_mirror_params --merge
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from cfm_mppi.safegpc_adapter.mirror_sampler import mirror_mppi_action
from cfm_mppi.evaluation.render_validation_comparison import (
    get_parser as _rp, _make_scene, _frame_obstacles, _frame_velocities)

OUT = Path("overnight_run_2026-06-23/mirror_sweep")
SEARCH = {
    "dual_sigma": [1.0, 1.4, 1.8],
    "eta": [0.6, 1.0, 1.4],
    "margin_gain": [0.15, 0.25, 0.4],
    "gamma": [0.5, 0.7, 0.9],
    "temperature": [0.2, 0.5],
    "clear_w": [60.0, 120.0],
    "terminal_w": [12.0, 20.0],
    "prox_w": [0.0, 4.0, 8.0],
    "sensing_range": [5.0, 6.5],
    "num_samples": [320],
    "horizon": [25, 35],
}


def _di(s, a, dt=0.1):
    x = s.copy(); x[0] += dt*s[2]+0.5*dt*dt*a[0]; x[1] += dt*s[3]+0.5*dt*dt*a[1]
    x[2] += dt*a[0]; x[3] += dt*a[1]; return x


def _eval(cfg, eps, dev):
    succ, coll, acc, clr = [], [], [], []
    for ep in eps:
        b = _rp().parse_args([]); b.dataset = "ucy"; b.dynamics = "doubleintegrator"
        b.pedestrian_source = "validation"; b.episode = ep; b.steps = 80
        s0, goal, obs, vel, _ = _make_scene(b)
        st = s0.astype(np.float32).copy(); a_acc = []; mc = 1e9
        for t in range(80):
            a, info = mirror_mppi_action(
                torch.tensor(st, device=dev), torch.tensor(goal, device=dev),
                torch.tensor(_frame_obstacles(obs, t), device=dev),
                torch.tensor(_frame_velocities(vel, t), device=dev),
                horizon=cfg["horizon"], num_samples=cfg["num_samples"], gamma=cfg["gamma"],
                eta=cfg["eta"], dual_sigma=cfg["dual_sigma"], margin_gain=cfg["margin_gain"],
                temperature=cfg["temperature"], clear_w=cfg["clear_w"], terminal_w=cfg["terminal_w"],
                prox_w=cfg["prox_w"], sensing_range=cfg["sensing_range"],
                mode_aware=cfg.get("mode_aware", True), seed=t, device=dev)
            a_acc.append(info["accept_rate"]); st = _di(st, a.detach().cpu().numpy())
            ot = _frame_obstacles(obs, t+1)
            if ot.shape[0]:
                mc = min(mc, float(np.min(np.linalg.norm(ot[:, :2]-st[:2], axis=1)-ot[:, 2]-0.5)))
        fd = float(np.linalg.norm(st[:2]-goal))
        succ.append(fd <= 0.5 and mc >= 0); coll.append(mc < 0)
        acc.append(float(np.mean(a_acc))); clr.append(mc if np.isfinite(mc) else 5.0)
    sr, cr = float(np.mean(succ)), float(np.mean(coll))
    return {"success_rate": sr, "collision_rate": cr, "accept_rate": float(np.mean(acc)),
            "mean_clearance": float(np.mean(clr)), "score": sr - 3.0*cr}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=int, default=0); p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--num-configs", type=int, default=24)
    p.add_argument("--val-eps", nargs=2, type=int, default=[130, 12], help="start count")
    p.add_argument("--val-list", nargs="+", type=int, default=None, help="explicit validation episodes (overrides --val-eps)")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--merge", action="store_true")
    cli = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if cli.merge:
        allres = []
        for f in sorted(OUT.glob("shard_*.json")):
            allres += json.loads(f.read_text())
        allres.sort(key=lambda r: (r["score"], r["accept_rate"]), reverse=True)
        (OUT/"merged.json").write_text(json.dumps(allres, indent=2))
        print("=== TOP 8 mirror configs (score = succ - 3*coll) ===")
        for r in allres[:8]:
            print(f"score={r['score']:.2f} succ={r['success_rate']:.2f} coll={r['collision_rate']:.2f} "
                  f"acc={r['accept_rate']:.2f} clr={r['mean_clearance']:.2f} | {r['config']}")
        return

    rng = random.Random(cli.seed)
    seen, configs = set(), []
    while len(configs) < cli.num_configs and len(seen) < 5000:
        c = {k: rng.choice(v) for k, v in SEARCH.items()}
        key = tuple(sorted(c.items()))
        if key in seen: continue
        seen.add(key); configs.append(c)
    mine = configs[cli.shard::cli.nshards]
    eps = list(range(cli.val_eps[0], cli.val_eps[0]+cli.val_eps[1]))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for i, c in enumerate(mine):
        r = {"config": c, **_eval(c, eps, dev)}
        results.append(r)
        print(f"[shard{cli.shard} {i+1}/{len(mine)}] score={r['score']:.2f} succ={r['success_rate']:.2f} "
              f"coll={r['collision_rate']:.2f} acc={r['accept_rate']:.2f} | {c}", flush=True)
        (OUT/f"shard_{cli.shard}.json").write_text(json.dumps(results, indent=2))
    print(f"shard {cli.shard} done: {len(results)} configs")


if __name__ == "__main__":
    main()
