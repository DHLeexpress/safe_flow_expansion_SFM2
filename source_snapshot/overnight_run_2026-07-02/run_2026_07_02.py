"""Stage F — autonomous safe-flow-expansion driver (per γ) + Stage G deliverables.

For each γ: fresh copy of the overfit pretrained policy -> ACTFLOW expansion (grid_expand.run_expand) with the
chosen (ell, beta, alpha, demo_frac); save history/covered/snapshots/expanded policy. Then per-γ coverage +
validity plots and the safe-expansion movie (3 γ side-by-side), and a results summary. W&B throughout.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import torch

import _paths  # noqa: F401
import grid_scene as GS
import grid_policy as GP
import grid_expand as GE
import grid_expand_viz as GV
import wandb_utils as W

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "expansion"); os.makedirs(RES, exist_ok=True)
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.5, 1.0, 0.1])
    ap.add_argument("--ell", type=float, default=0.2)
    ap.add_argument("--beta", type=float, default=1.0 / 50)
    ap.add_argument("--s", type=float, default=0.3)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--demo-frac", type=float, default=0.55)
    ap.add_argument("--n-measure", type=int, default=50)
    ap.add_argument("--measure-every", type=int, default=50)
    ap.add_argument("--baseline", type=int, default=500)
    ap.add_argument("--demo-frac-only", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--pretrained", default=os.path.join(HERE, "pretrained.pt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    W.add_wandb_args(ap)
    args = ap.parse_args()
    dev = args.device
    env = GS.make_grid()

    run = W.init_run(args, name=f"expand-ell{args.ell}-b{args.beta:.3f}", config=vars(args), group="grid-safeflow")
    hist_by_gamma, snap_by_gamma, results = {}, {}, {}
    step = 0
    for g in args.gammas:
        pol, _ = GP.load_policy(args.pretrained, device=dev)          # fresh copy per γ
        demo = GE.load_demo(g)
        cfg = GE.SFGridConfig(iters=args.iters, ell=args.ell, beta=args.beta, s=args.s, alpha=args.alpha,
                              demo_frac=args.demo_frac, n_measure=args.n_measure, measure_every=args.measure_every,
                              baseline_deploys=args.baseline)
        print(f"\n===== EXPAND γ={g}  (ell={cfg.ell}, beta={cfg.beta:.3f}, alpha={cfg.alpha}, "
              f"demo_frac={cfg.demo_frac}, iters={cfg.iters}) =====", flush=True)
        try:
            r = GE.run_expand(pol, env, g, cfg, demo=demo, device=dev, run=run, step0=step)
        except Exception as exc:
            print(f"[γ{g}] expansion crashed: {exc}", flush=True)
            continue
        step += cfg.iters + 10
        hist_by_gamma[g] = r["history"]; snap_by_gamma[g] = r["snapshots"]
        GP.save_policy(r["policy"], os.path.join(RES, f"expanded_g{g}.pt"),
                       extra={"gamma": g, "covered": sorted(r["covered"]), "final": r["final"]})
        with open(os.path.join(RES, f"history_g{g}.json"), "w") as f:
            json.dump(r["history"], f, indent=2)
        with open(os.path.join(RES, f"snapshots_g{g}.pkl"), "wb") as f:
            pickle.dump(snap_by_gamma[g], f)
        results[g] = dict(coverage=r["final"]["coverage"], validity=r["final"]["validity"],
                          avg_steps=r["final"]["avg_steps"], covered=len(r["covered"]),
                          reached_goal=r["reached_goal"])
        print(f"[γ{g}] FINAL cov={r['final']['coverage']*100:.1f}% val={r['final']['validity']*100:.1f}% "
              f"covered={len(r['covered'])}/252 goal={r['reached_goal']}", flush=True)

    # ---- deliverables ----
    if hist_by_gamma:
        GV.plot_metrics(hist_by_gamma, os.path.join(FIG, "expand_coverage.png"),
                        os.path.join(FIG, "expand_validity.png"))
        GV.expansion_movie(snap_by_gamma, env.obstacles.numpy(),
                           os.path.join(FIG, "safe_expansion_grid.gif"))
        W.log_image(run, "expand/coverage_plot", os.path.join(FIG, "expand_coverage.png"))
        W.log_image(run, "expand/validity_plot", os.path.join(FIG, "expand_validity.png"))
        W.log_video(run, "expand/movie", os.path.join(FIG, "safe_expansion_grid.mp4"))
    with open(os.path.join(RES, "results.json"), "w") as f:
        json.dump(dict(params=dict(ell=args.ell, beta=args.beta, alpha=args.alpha, demo_frac=args.demo_frac,
                                   iters=args.iters), results=results), f, indent=2)
    print("\n===== SUMMARY =====")
    for g, m in results.items():
        print(f"  γ{g}: coverage {m['coverage']*100:.1f}%  validity {m['validity']*100:.1f}%  "
              f"steps {m['avg_steps']:.0f}  goal={m['reached_goal']}", flush=True)
    all_goal = all(m["reached_goal"] for m in results.values()) and len(results) == len(args.gammas)
    print(f"  ALL γ > 90% cov+val: {all_goal}", flush=True)
    W.finish(run, summary={f"final_cov_g{g}": m["coverage"] for g, m in results.items()})


if __name__ == "__main__":
    main()
