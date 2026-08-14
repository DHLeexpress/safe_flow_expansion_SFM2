"""Headless batch benchmark over moving-pedestrian episodes (UCY/SDD).

Reuses the exact scene construction, per-step rollout, and metric definitions of
``render_validation_comparison`` (same success/collision/clearance), but loops
over many episodes and methods, and emits aggregate statistics with bootstrap
confidence intervals and paired significance tests (guided vs Mizuta).

Example:
  python -m cfm_mppi.evaluation.eval_pedestrian_benchmark \
      --dataset ucy --dynamics doubleintegrator \
      --methods mizuta_cfm_mppi safemppi_gamma guided_safemppi \
      --gamma-grid 0.1 0.3 0.5 1.0 --num-episodes 100 \
      --safemppi-samples 512 --safemppi-horizon 40 --device cuda \
      --output overnight_run_2026-06-23/ped_bench
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from cfm_mppi.evaluation.render_validation_comparison import (
    get_parser as _render_parser,
    _make_scene,
    _rollout,
    _policy_args,
    BenchmarkPolicies,
)
import torch


METHOD_GAMMA = {"safemppi_gamma", "guided_safemppi", "safe_cfm", "cfm_proposal_mppi", "mizuta_safe", "mirror_mppi"}


def _base_args(cli: argparse.Namespace) -> argparse.Namespace:
    args = _render_parser().parse_args([])
    args.dataset = cli.dataset
    args.dynamics = cli.dynamics
    args.pedestrian_source = cli.pedestrian_source
    args.steps = cli.steps
    args.seed = cli.seed
    args.device = cli.device
    args.safemppi_samples = cli.safemppi_samples
    args.safemppi_horizon = cli.safemppi_horizon
    args.debug_rollouts = 0
    args.smoke = False
    args.num_pedestrians = cli.num_pedestrians
    if getattr(cli, "safe_cfm_checkpoint", None):
        args.safe_cfm_checkpoint = cli.safe_cfm_checkpoint
    if getattr(cli, "drifting_checkpoint", None):
        args.drifting_checkpoint = cli.drifting_checkpoint
    for k in ("guided_eta", "guided_extra_margin", "guided_activation_radius",
              "guided_progress_weight", "guided_terminal_goal_weight",
              "guided_running_goal_weight", "guided_guidance_horizon"):
        v = getattr(cli, k, None)
        if v is not None:
            setattr(args, k, v)
    return args


def _bootstrap_ci(values, fn, n_boot=2000, alpha=0.05, seed=0):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    stats = np.empty(n_boot)
    n = values.size
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        stats[b] = fn(values[idx])
    point = fn(values)
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return (float(point), lo, hi)


def _mcnemar(success_a, success_b):
    """Paired test on per-episode binary success. Returns (b, c, p_two_sided).
    b = a wins (a=1,b=0), c = b wins (a=0,b=1). Exact binomial p."""
    a = np.asarray(success_a, dtype=bool)
    b = np.asarray(success_b, dtype=bool)
    n01 = int(np.sum(a & ~b))   # a success, b fail
    n10 = int(np.sum(~a & b))   # a fail, b success
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    # two-sided exact binomial with p=0.5
    p = 0.0
    for i in range(0, k + 1):
        p += math.comb(n, i) * 0.5 ** n
    p = min(1.0, 2.0 * p)
    return n01, n10, p


def _wilcoxon_signed_rank(x, y):
    """Paired Wilcoxon signed-rank on per-episode metric (x - y). Normal approx.
    Returns (median_diff, p_two_sided)."""
    d = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    d = d[d != 0]
    n = d.size
    if n == 0:
        return 0.0, 1.0
    order = np.argsort(np.abs(d))
    ranks = np.empty(n)
    ranks[order] = np.arange(1, n + 1)
    # average ties on |d|
    absd = np.abs(d)
    uniq = np.unique(absd)
    for u in uniq:
        mask = absd == u
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    w_plus = ranks[d > 0].sum()
    w_minus = ranks[d < 0].sum()
    T = min(w_plus, w_minus)
    mean_T = n * (n + 1) / 4.0
    sd_T = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sd_T == 0:
        return float(np.median(d)), 1.0
    z = (T - mean_T) / sd_T
    p = math.erfc(abs(z) / math.sqrt(2))
    return float(np.median(np.asarray(x) - np.asarray(y))), float(p)


def run(cli: argparse.Namespace) -> None:
    base = _base_args(cli)
    device = torch.device(cli.device)
    policies = BenchmarkPolicies(_policy_args(base), device)

    episodes = list(range(cli.num_episodes)) if cli.episode_list is None else cli.episode_list
    method_variants = []
    for m in cli.methods:
        if m in METHOD_GAMMA:
            for g in cli.gamma_grid:
                method_variants.append((m, float(g)))
        else:
            method_variants.append((m, None))

    out_dir = Path(cli.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = (out_dir / "episodes.jsonl").open("w")

    # records[(method,gamma)] -> list of metric dicts (one per episode), indexed by episode order
    records: dict = {mv: [] for mv in method_variants}
    t_start = time.time()
    for ei, ep in enumerate(episodes):
        base.episode = int(ep)
        state0, goal, obstacles_seq, velocities_seq, scene_label = _make_scene(base)
        for (method, gamma) in method_variants:
            run_args = base
            run_obj = _rollout(run_args, policies, method, gamma, state0, goal, obstacles_seq, velocities_seq)
            m = dict(run_obj.metrics)
            m["episode"] = int(ep)
            m["method"] = method
            m["gamma"] = gamma
            records[(method, gamma)].append(m)
            jsonl.write(json.dumps({k: (None if isinstance(v, float) and math.isinf(v) else v)
                                    for k, v in m.items() if not isinstance(v, (list, dict, np.ndarray))}) + "\n")
        if (ei + 1) % 5 == 0 or ei == len(episodes) - 1:
            el = time.time() - t_start
            print(f"[{ei+1}/{len(episodes)}] ep={ep} elapsed={el:.0f}s "
                  f"({el/(ei+1):.1f}s/ep)", flush=True)
    jsonl.close()

    # aggregate
    summary = []
    for (method, gamma), rows in records.items():
        succ = np.array([r["success"] for r in rows], dtype=float)
        coll = np.array([r["collision"] for r in rows], dtype=float)
        clr = np.array([r["min_clearance"] for r in rows], dtype=float)
        clr = np.where(np.isfinite(clr), clr, np.nan)
        gd = np.array([r["final_goal_distance"] for r in rows], dtype=float)
        gr = np.array([r["goal_reached"] for r in rows], dtype=float)
        pl = np.array([r["path_length"] for r in rows], dtype=float)
        eff = np.array([r["control_effort"] for r in rows], dtype=float)
        pt = np.array([r.get("planning_wall_time_mean", 0.0) for r in rows], dtype=float)
        s_pt, s_lo, s_hi = _bootstrap_ci(succ, np.mean)
        c_pt, c_lo, c_hi = _bootstrap_ci(coll, np.mean)
        clr_valid = clr[~np.isnan(clr)]
        summary.append({
            "method": method, "gamma": gamma, "episodes": len(rows),
            "success_rate": s_pt, "success_ci": [s_lo, s_hi],
            "collision_rate": c_pt, "collision_ci": [c_lo, c_hi],
            "goal_reached_rate": float(np.mean(gr)),
            "mean_min_clearance": float(np.nanmean(clr)) if clr_valid.size else float("nan"),
            "median_min_clearance": float(np.nanmedian(clr)) if clr_valid.size else float("nan"),
            "mean_final_goal_distance": float(np.mean(gd)),
            "mean_path_length": float(np.mean(pl)),
            "mean_control_effort": float(np.mean(eff)),
            "mean_planning_time_ms": float(np.mean(pt) * 1000.0),
        })

    # paired tests: each guided/safemppi variant vs mizuta (episode-aligned)
    paired = []
    mizuta_rows = records.get(("mizuta_cfm_mppi", None))
    if mizuta_rows is not None:
        m_succ = [r["success"] for r in mizuta_rows]
        m_clr = [r["min_clearance"] if np.isfinite(r["min_clearance"]) else 10.0 for r in mizuta_rows]
        for (method, gamma), rows in records.items():
            if method == "mizuta_cfm_mppi":
                continue
            o_succ = [r["success"] for r in rows]
            o_clr = [r["min_clearance"] if np.isfinite(r["min_clearance"]) else 10.0 for r in rows]
            n01, n10, p_mc = _mcnemar(o_succ, m_succ)  # ours vs mizuta
            med_diff, p_w = _wilcoxon_signed_rank(o_clr, m_clr)
            paired.append({
                "method": method, "gamma": gamma,
                "success_ours_minus_mizuta": float(np.mean(o_succ) - np.mean(m_succ)),
                "mcnemar_ours_win": n01, "mcnemar_mizuta_win": n10, "mcnemar_p": p_mc,
                "clearance_median_diff_ours_minus_mizuta": med_diff, "wilcoxon_p": p_w,
            })

    result = {"summary": summary, "paired_vs_mizuta": paired,
              "config": vars(cli), "episodes": len(episodes)}
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))

    # CSV
    cols = ["method", "gamma", "episodes", "success_rate", "collision_rate",
            "goal_reached_rate", "mean_min_clearance", "median_min_clearance",
            "mean_final_goal_distance", "mean_path_length", "mean_control_effort",
            "mean_planning_time_ms"]
    with (out_dir / "summary.csv").open("w") as f:
        f.write(",".join(cols) + "\n")
        for s in summary:
            f.write(",".join(str(s[c]) for c in cols) + "\n")

    # markdown
    md = ["# Pedestrian benchmark — moving obstacles", "",
          f"dataset={cli.dataset} dynamics={cli.dynamics} episodes={len(episodes)} "
          f"samples={cli.safemppi_samples} horizon={cli.safemppi_horizon}", "",
          "| method | gamma | succ% [95% CI] | coll% [95% CI] | min-clear (mean/med) | goal-dist | path | t(ms) |",
          "|---|---|---|---|---|---|---|---|"]
    for s in summary:
        md.append("| {m} | {g} | {sr:.1f} [{sl:.1f},{sh:.1f}] | {cr:.1f} [{cl:.1f},{ch:.1f}] | "
                  "{mc:.3f}/{mdc:.3f} | {gd:.2f} | {pl:.1f} | {pt:.1f} |".format(
                      m=s["method"], g=("-" if s["gamma"] is None else f"{s['gamma']:.2g}"),
                      sr=100*s["success_rate"], sl=100*s["success_ci"][0], sh=100*s["success_ci"][1],
                      cr=100*s["collision_rate"], cl=100*s["collision_ci"][0], ch=100*s["collision_ci"][1],
                      mc=s["mean_min_clearance"], mdc=s["median_min_clearance"],
                      gd=s["mean_final_goal_distance"], pl=s["mean_path_length"], pt=s["mean_planning_time_ms"]))
    md += ["", "## Paired vs Mizuta (same episodes)",
           "| method | gamma | Δsucc | McNemar (ours/miz win) p | Δclear(med) Wilcoxon p |",
           "|---|---|---|---|---|"]
    for p in paired:
        md.append("| {m} | {g} | {ds:+.3f} | {a}/{b} p={pmc:.3g} | {dc:+.3f} p={pw:.3g} |".format(
            m=p["method"], g=("-" if p["gamma"] is None else f"{p['gamma']:.2g}"),
            ds=p["success_ours_minus_mizuta"], a=p["mcnemar_ours_win"], b=p["mcnemar_mizuta_win"],
            pmc=p["mcnemar_p"], dc=p["clearance_median_diff_ours_minus_mizuta"], pw=p["wilcoxon_p"]))
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nWrote {out_dir}/summary.{{json,csv,md}} and episodes.jsonl")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="ucy", choices=["ucy", "sdd", "sfm"])
    p.add_argument("--dynamics", default="doubleintegrator", choices=["doubleintegrator", "unicycle"])
    p.add_argument("--pedestrian-source", default="validation")
    p.add_argument("--methods", nargs="+",
                   default=["mizuta_cfm_mppi", "safemppi_gamma", "guided_safemppi"])
    p.add_argument("--gamma-grid", nargs="+", type=float, default=[0.1, 0.2, 0.4, 0.8])  # log-scale working range
    p.add_argument("--num-episodes", type=int, default=100)
    p.add_argument("--episode-list", nargs="+", type=int, default=None)
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--safemppi-samples", type=int, default=512)
    p.add_argument("--safemppi-horizon", type=int, default=40)
    p.add_argument("--num-pedestrians", type=int, default=20)
    p.add_argument("--output", default="overnight_run_2026-06-23/ped_bench")
    p.add_argument("--safe-cfm-checkpoint", default=None)
    p.add_argument("--drifting-checkpoint", default=None)
    p.add_argument("--guided-eta", type=float, default=None)
    p.add_argument("--guided-extra-margin", type=float, default=None)
    p.add_argument("--guided-activation-radius", type=float, default=None)
    p.add_argument("--guided-progress-weight", type=float, default=None)
    p.add_argument("--guided-terminal-goal-weight", type=float, default=None)
    p.add_argument("--guided-running-goal-weight", type=float, default=None)
    p.add_argument("--guided-guidance-horizon", type=int, default=None)
    run(p.parse_args())


if __name__ == "__main__":
    main()
