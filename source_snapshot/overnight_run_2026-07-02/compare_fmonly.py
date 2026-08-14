"""Faithful ablation: pure FM-only Eq-9 SNIS (N=96 FM candidates, NO broad proposal, NO safety filter) vs the
current FM+broad heuristic (40 FM + 40 broad + safety filter). Both run for the SAME wall-clock budget on γ=0.5;
compare coverage/validity vs iteration (and note iters completed, since FM-only trajectories die faster)."""
import _paths, torch, time, os, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import grid_scene as GS, grid_policy as GP, grid_expand as GE

dev = "cuda"; env = GS.make_grid()
BUDGET = float(os.environ.get("BUDGET", 1200))
CONFIGS = {
    "FM-only (N=96, no broad, no filter)": dict(N=96, broad=0, safe_filter=False, color="#1f77b4"),
    "FM+broad (40+40, safety filter)":     dict(N=40, broad=40, safe_filter=True, color="#d62728"),
}
res = {}
for name, kw in CONFIGS.items():
    color = kw.pop("color")
    pol, _ = GP.load_policy("pretrained.pt", device=dev)
    cfg = GE.SFGridConfig(iters=100000, ell=0.2, beta=0.02, s=0.3, demo_frac=0.5,
                          n_measure=40, measure_every=25, baseline_deploys=40, **kw)
    print(f"\n=== {name}: wall-clock budget {BUDGET:.0f}s ===", flush=True)
    t0 = time.time()
    r = GE.run_expand(pol, env, 0.5, cfg, device=dev, time_budget=BUDGET)
    res[name] = dict(history=r["history"], covered=len(r["covered"]), color=color,
                     iters=r["history"][-1]["iter"], wall=time.time() - t0,
                     cov=r["final"]["coverage"], val=r["final"]["validity"])
    print(f"  -> {res[name]['iters']} iters in {res[name]['wall']/60:.1f} min | "
          f"coverage {res[name]['cov']*100:.1f}%  validity {res[name]['val']*100:.1f}%  "
          f"covered {res[name]['covered']}/252", flush=True)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
for name, d in res.items():
    it = [h["iter"] for h in d["history"]]
    ax[0].plot(it, [h["coverage"] * 100 for h in d["history"]], "-o", ms=3, color=d["color"], label=name)
    ax[1].plot(it, [h["validity"] * 100 for h in d["history"]], "-o", ms=3, color=d["color"], label=name)
for a, t in zip(ax, ["coverage (%/252)", "validity (%)"]):
    a.set_xlabel("iteration (same wall-clock budget)"); a.set_ylabel(t); a.grid(alpha=.25); a.legend(fontsize=8)
ax[0].set_title(f"Coverage — FM-only Eq-9 vs FM+broad ({BUDGET/60:.0f} min each, γ=0.5)")
ax[1].set_title("Validity")
fig.tight_layout(); fig.savefig("figures/fmonly_vs_broad.png", dpi=140); print("\nsaved figures/fmonly_vs_broad.png")
json.dump({n: {k: d[k] for k in ("iters", "wall", "cov", "val", "covered")} for n, d in res.items()},
          open("results/fmonly_vs_broad.json", "w"), indent=2)
