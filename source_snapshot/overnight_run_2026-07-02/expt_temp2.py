"""Experiment: faithful φ_s main line with temperature=2.0, β=1/10 (softer/less-greedy), FM-only 64 candidates
(no broad), 1000 iters/γ. Separate outputs (results/expt_temp2, figures/expt_temp2_*); existing run untouched."""
import argparse, json, os
import _paths  # noqa: F401
import grid_scene as GS, grid_policy as GP, grid_expand as GE, grid_expand_viz as GV, wandb_utils as W

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "expt_temp2"); FIG = os.path.join(HERE, "figures")
os.makedirs(RES, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.1, 0.5, 1.0])
    W.add_wandb_args(ap)
    args = ap.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    env = GS.make_grid()
    run = W.init_run(args, name="expt-temp2-b10",
                     config=dict(temp=2.0, beta=0.1, N=64, broad=0, feature="phi_s", iters=args.iters),
                     group="grid-safeflow")
    hist_by, snap_by, results = {}, {}, {}
    step = 0
    for g in args.gammas:
        pol, _ = GP.load_policy("pretrained.pt", device=dev)
        cfg = GE.SFGridConfig(iters=args.iters, feature="phi_s", N=64, broad=0, safe_filter=True,
                              temp_explore=2.0, beta=0.1, s=0.3, ell=0.2, demo_frac=0.5,
                              n_measure=50, measure_every=50, baseline_deploys=200)
        print(f"\n===== γ={g}: temp=2.0, β=1/10, FM-only(64), {args.iters} iters =====", flush=True)
        r = GE.run_expand(pol, env, g, cfg, device=dev, run=run, step0=step)
        step += args.iters + 10
        hist_by[g] = r["history"]; snap_by[g] = r["snapshots"]
        results[g] = dict(coverage=r["final"]["coverage"], validity=r["final"]["validity"], covered=len(r["covered"]))
        print(f"[γ{g}] FINAL coverage {r['final']['coverage']*100:.1f}%  validity {r['final']['validity']*100:.1f}%  "
              f"covered {len(r['covered'])}/252", flush=True)
        GP.save_policy(r["policy"], os.path.join(RES, f"expanded_g{g}.pt"),
                       extra={"gamma": g, "covered": sorted(r["covered"]), "final": r["final"]})
        json.dump(r["history"], open(os.path.join(RES, f"history_g{g}.json"), "w"), indent=2)
    GV.plot_metrics(hist_by, os.path.join(FIG, "expt_temp2_coverage.png"), os.path.join(FIG, "expt_temp2_validity.png"))
    GV.expansion_movie(snap_by, env.obstacles.numpy(), os.path.join(FIG, "expt_temp2_movie.gif"))
    json.dump(dict(config=dict(temp=2.0, beta=0.1, N=64, broad=0, feature="phi_s"), results=results),
              open(os.path.join(RES, "results.json"), "w"), indent=2)
    print("\n===== SUMMARY (temp=2.0, β=1/10, 1000 iters) =====")
    for g, m in results.items():
        print(f"  γ{g}: coverage {m['coverage']*100:.1f}%  validity {m['validity']*100:.1f}%  covered {m['covered']}/252",
              flush=True)
    W.finish(run)


if __name__ == "__main__":
    main()
