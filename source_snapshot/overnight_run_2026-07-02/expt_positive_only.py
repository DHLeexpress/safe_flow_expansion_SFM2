"""Next phase: POSITIVE-ONLY training (demo_frac=0, no SafeMPPI anchor), MODERATE temperature (1.3, near the
policy's own/initial distribution) and MODERATE β (1/10). Tests whether removing the diagonal demo anchor lets
the field de-collapse further, at the risk of forgetting the safe behavior. γ=0.5, 10k iters, track_variance."""
import argparse, json, os
import _paths  # noqa: F401
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import grid_scene as GS, grid_policy as GP, grid_expand as GE, wandb_utils as W

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "expt_positive"); FIG = os.path.join(HERE, "figures"); os.makedirs(RES, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--temp", type=float, default=1.3)
    ap.add_argument("--beta", type=float, default=0.1)
    W.add_wandb_args(ap)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    env = GS.make_grid()
    run = W.init_run(args, name=f"positive-only-{args.iters}", config=vars(args), group="grid-safeflow")
    pol, _ = GP.load_policy("pretrained.pt", device=dev)
    cfg = GE.SFGridConfig(iters=args.iters, feature="phi_s", N=64, broad=0, safe_filter=True,
                          temp_explore=args.temp, beta=args.beta, demo_frac=0.0, inner_steps=12, lr=2e-4,
                          s=0.3, ell=0.2, track_variance=True, n_measure=30, measure_every=250,
                          baseline_deploys=150, cap_pos=60000)
    print(f"===== positive-only (demo_frac=0): {args.iters} iters, γ={args.gamma}, temp={args.temp}, β={args.beta} =====",
          flush=True)
    r = GE.run_expand(pol, env, args.gamma, cfg, device=dev, run=run, step0=0)
    GP.save_policy(r["policy"], os.path.join(RES, f"positive_g{args.gamma}.pt"), extra={"covered": sorted(r["covered"])})
    json.dump(r["history"], open(os.path.join(RES, "positive_history.json"), "w"), indent=2)
    print(f"[positive-only] FINAL cov {r['final']['coverage']*100:.1f}%  val {r['final']['validity']*100:.1f}%  "
          f"covered {len(r['covered'])}/252  out_var {r['history'][-1].get('out_var', 0):.4f}", flush=True)

    h = r["history"]; it = [x["iter"] for x in h]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    ax[0].plot(it, [x["coverage"] * 100 for x in h], "-o", ms=3, color="#8c1aa8"); ax[0].axhline(90, ls="--", color="#999")
    ax[1].plot(it, [x["validity"] * 100 for x in h], "-o", ms=3, color="#8c1aa8")
    ov = [(x["iter"], x["out_var"]) for x in h if "out_var" in x]
    ax[2].plot([a for a, _ in ov], [b for _, b in ov], "-o", ms=3, color="#8c1aa8")
    for a, t in zip(ax, ["coverage % (distinct / 252)", "validity %", "output-variance (policy spread, m²)"]):
        a.set_xlabel("ACTFLOW iteration"); a.set_title(t); a.grid(alpha=.25)
    fig.suptitle(f"POSITIVE-ONLY (demo_frac=0, γ={args.gamma}, temp={args.temp}, β={args.beta}): "
                 "coverage · validity · output-variance", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "expt_positive_curves.png"), dpi=140)
    W.log_image(run, "positive/plot", os.path.join(FIG, "expt_positive_curves.png"))
    print("saved figures/expt_positive_curves.png"); W.finish(run)


if __name__ == "__main__":
    main()
