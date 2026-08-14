"""Resume the safe-flow expansion +N iters from the saved expanded policies (cumulative coverage carries over),
combine old+new history/snapshots, and regenerate the coverage/validity plots + movie with γ order 0.1,0.5,1.0."""
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
RES = os.path.join(HERE, "results", "expansion")
FIG = os.path.join(HERE, "figures")
GAMMAS = [0.1, 0.5, 1.0]                          # γ0.1 on the LEFT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--more", type=int, default=500)
    ap.add_argument("--offset", type=int, default=450)
    W.add_wandb_args(ap)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    env = GS.make_grid()
    run = W.init_run(args, name="expand-resume-500", config=vars(args), group="grid-safeflow")

    hist_by_gamma, snap_by_gamma, results = {}, {}, {}
    step = 0
    for g in GAMMAS:
        pol, ck = GP.load_policy(os.path.join(RES, f"expanded_g{g}.pt"), device=dev)
        covered0 = set(ck.get("covered", []))
        cfg = GE.SFGridConfig(iters=args.more, ell=0.2, beta=0.02, s=0.3, demo_frac=0.5,
                              n_measure=50, measure_every=50, baseline_deploys=100)
        print(f"\n=== RESUME γ={g}: from covered={len(covered0)}/252, +{args.more} iters ===", flush=True)
        r = GE.run_expand(pol, env, g, cfg, demo=GE.load_demo(g), device=dev, run=run, step0=step,
                          init_covered=covered0, iter_offset=args.offset)
        step += args.more + 10
        old_h = json.load(open(os.path.join(RES, f"history_g{g}.json")))
        hist_by_gamma[g] = old_h + r["history"][1:]                  # drop the resume-baseline duplicate
        sp = os.path.join(RES, f"snapshots_g{g}.pkl")
        old_snap = pickle.load(open(sp, "rb")) if os.path.exists(sp) else []
        snap_by_gamma[g] = old_snap + r["snapshots"]
        GP.save_policy(r["policy"], os.path.join(RES, f"expanded_g{g}.pt"),
                       extra={"gamma": g, "covered": sorted(r["covered"]), "final": r["final"]})
        json.dump(hist_by_gamma[g], open(os.path.join(RES, f"history_g{g}.json"), "w"), indent=2)
        pickle.dump(snap_by_gamma[g], open(os.path.join(RES, f"snapshots_g{g}.pkl"), "wb"))
        results[g] = dict(coverage=r["final"]["coverage"], validity=r["final"]["validity"], covered=len(r["covered"]))
        print(f"[γ{g}] FINAL cov={r['final']['coverage']*100:.1f}% val={r['final']['validity']*100:.1f}% "
              f"covered={len(r['covered'])}/252", flush=True)

    GV.plot_metrics(hist_by_gamma, os.path.join(FIG, "expand_coverage.png"), os.path.join(FIG, "expand_validity.png"))
    GV.expansion_movie(snap_by_gamma, env.obstacles.numpy(), os.path.join(FIG, "safe_expansion_grid.gif"))
    W.log_image(run, "expand/coverage_plot", os.path.join(FIG, "expand_coverage.png"))
    W.log_image(run, "expand/validity_plot", os.path.join(FIG, "expand_validity.png"))
    W.log_video(run, "expand/movie", os.path.join(FIG, "safe_expansion_grid.mp4"))
    print(f"\n=== SUMMARY (after +{args.more}, total ~{args.offset + args.more} iters) ===")
    for g, m in results.items():
        print(f"  γ{g}: coverage {m['coverage']*100:.1f}%  validity {m['validity']*100:.1f}%  "
              f"covered {m['covered']}/252", flush=True)
    W.finish(run)


if __name__ == "__main__":
    main()
