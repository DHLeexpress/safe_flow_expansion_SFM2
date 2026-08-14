"""Sweep (s = φ_s representation noise level, β = σ-tilt temperature) on the pretrained model.
Hypothesis (user): decreasing BOTH s and β increases exploration -> higher coverage. γ=0.5, short runs."""
import _paths, torch, time
import grid_scene as GS, grid_policy as GP, grid_expand as GE

dev = "cuda"
env = GS.make_grid()
ITERS = 90
combos = [(0.9, 1/25), (0.7, 1/25), (0.5, 1/25), (0.3, 1/25),
          (0.7, 1/50), (0.5, 1/50), (0.3, 1/50), (0.5, 1/80)]
print(f"(s, beta) sweep on γ=0.5, {ITERS} iters each  [pure σ-tilt]\n")
print(f"{'s':>4} {'beta':>7} | {'cov@45':>7} {'cov@90':>7} {'val@90':>7} {'covered':>7} | time")
rows = []
for s, b in combos:
    pol, _ = GP.load_policy("pretrained.pt", device=dev)
    cfg = GE.SFGridConfig(iters=ITERS, measure_every=45, n_measure=40, baseline_deploys=40, warmup_pos=40,
                          beta=b, s=s, ell=0.2, N=40, broad=40, cap_pos=50000, use_style=False,
                          inner_steps=10, lr=1.5e-4, demo_frac=0.5, nfe_measure=10)
    t0 = time.time()
    r = GE.run_expand(pol, env, 0.5, cfg, device=dev, log=lambda *a, **k: None)
    h = r["history"]
    c45 = h[1]["coverage"] * 100 if len(h) > 1 else 0
    c90, v90 = h[-1]["coverage"] * 100, h[-1]["validity"] * 100
    print(f"{s:>4} 1/{1/b:>4.0f} | {c45:>6.1f}% {c90:>6.1f}% {v90:>6.1f}% {len(r['covered']):>7} | {(time.time()-t0)/60:.1f}min", flush=True)
    rows.append((s, b, c90, v90, len(r["covered"])))
best = max(rows, key=lambda z: z[4] + (z[3] >= 75) * 5)     # most covered, prefer validity >=75
print(f"\nBEST (most covered, val>=75%): s={best[0]} beta=1/{1/best[1]:.0f} -> covered {best[4]}, val {best[3]:.0f}%")
