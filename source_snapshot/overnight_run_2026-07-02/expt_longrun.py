"""Automated 10k-iteration de-collapse study (γ=0.5, temp=2.0, β=1/10, FM-only 64, faithful φ_s).
Question: can the verifier slowly re-shape the overfit (near-deterministic, diagonal) velocity field so the FM
learns OFF-DIAGONAL behavior? Tracks the FM OUTPUT VARIANCE over iterations (OOD = off-diagonal spread), and
compares the CURRENT loss vs an AGGRESSIVE loss (less demo anchoring, more inner steps, higher lr = "backprop
more"). Separate outputs (results/expt_longrun, figures/expt_longrun.png); existing runs untouched.
"""
import argparse, json, os
import _paths  # noqa: F401
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import grid_scene as GS, grid_policy as GP, grid_expand as GE, wandb_utils as W

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "expt_longrun"); FIG = os.path.join(HERE, "figures"); os.makedirs(RES, exist_ok=True)
CONDS = {
    "current":         dict(demo_frac=0.5, inner_steps=12, lr=2e-4, color="#1f77b4"),
    "aggressive-loss": dict(demo_frac=0.3, inner_steps=24, lr=4e-4, color="#d62728"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--gamma", type=float, default=0.5)
    W.add_wandb_args(ap)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    env = GS.make_grid()
    run = W.init_run(args, name=f"longrun-{args.iters}", config=vars(args), group="grid-safeflow")
    hist = {}
    step = 0
    for name, kw in CONDS.items():
        color = kw.pop("color")
        pol, _ = GP.load_policy("pretrained.pt", device=dev)
        cfg = GE.SFGridConfig(iters=args.iters, feature="phi_s", N=64, broad=0, safe_filter=True,
                              temp_explore=2.0, beta=0.1, s=0.3, ell=0.2, track_variance=True,
                              n_measure=30, measure_every=250, baseline_deploys=150, cap_pos=60000, **kw)
        print(f"\n===== {name}: {args.iters} iters, γ={args.gamma}, temp=2 β=1/10 FM-only =====", flush=True)
        r = GE.run_expand(pol, env, args.gamma, cfg, device=dev, run=run, step0=step)
        step += args.iters + 10
        hist[name] = (r["history"], color)
        GP.save_policy(r["policy"], os.path.join(RES, f"{name}_g{args.gamma}.pt"), extra={"covered": sorted(r["covered"])})
        json.dump(r["history"], open(os.path.join(RES, f"{name}_history.json"), "w"), indent=2)
        print(f"[{name}] FINAL cov {r['final']['coverage']*100:.1f}%  val {r['final']['validity']*100:.1f}%  "
              f"covered {len(r['covered'])}/252  out_var {r['history'][-1].get('out_var', 0):.4f}", flush=True)

    fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
    for name, (h, color) in hist.items():
        it = [x["iter"] for x in h]
        ax[0].plot(it, [x["coverage"] * 100 for x in h], "-", color=color, lw=1.8, label=name)
        ax[1].plot(it, [x["validity"] * 100 for x in h], "-", color=color, lw=1.8, label=name)
        iv = [(x["iter"], x["out_var"]) for x in h if "out_var" in x]
        ax[2].plot([a for a, _ in iv], [b for _, b in iv], "-", color=color, lw=1.8, label=name)
    for a, t in zip(ax, ["coverage % (distinct / 252)", "validity %",
                         "FM output variance @ probes (off-diagonal spread)"]):
        a.set_xlabel("ACTFLOW iteration"); a.set_title(t); a.grid(alpha=.25); a.legend(fontsize=9)
    fig.suptitle(f"{args.iters}-iter de-collapse study (γ={args.gamma}, temp=2, β=1/10, FM-only) — "
                 "current vs aggressive loss", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "expt_longrun.png"), dpi=140)
    W.log_image(run, "longrun/plot", os.path.join(FIG, "expt_longrun.png"))
    json.dump({n: {"final": h[-1]} for n, (h, _) in hist.items()}, open(os.path.join(RES, "results.json"), "w"), indent=2)
    print("\nsaved figures/expt_longrun.png"); W.finish(run)


if __name__ == "__main__":
    main()
