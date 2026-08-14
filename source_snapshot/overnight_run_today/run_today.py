"""SafeFlow Exploration — overnight_run_today entrypoint.

Pipeline:  build Omega* -> pretrain one-leaf seed (theta_0) -> Safe Flow Expansion (ACTFLOW loop)
           -> coverage/validity/vendi curves + multi-modal overlays.

Usage:
    python overnight_run_today/run_today.py --env single --rounds 60 --device cuda
    python overnight_run_today/run_today.py --env gap    --rounds 80 --device cuda
    python overnight_run_today/run_today.py --env single --rounds 4 --smoke
    python overnight_run_today/run_today.py --env single --rounds 60 --baseline recf   # no-tilt ablation
"""
from __future__ import annotations

import argparse, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

import torch

from dynamics import make_env
from flow_policy import FlowPolicy, env_context
from dtcbf import build_candidate_polytope
import descriptors as D
from safeflow import (SFConfig, run_safeflow, make_seed, controller_rollout,
                      validity_label, update_flow)
import plots


def build_omega_star(env, cfg, n, device, log):
    """Broad proposal -> verify -> the reachable-safe descriptor bins B* (coverage denominator)."""
    lo, hi = env.ylim
    lat = torch.empty(n, device=device).uniform_(lo * 0.95, hi * 0.95)
    U, _ = controller_rollout(env, n, lat, sigma=0.8, device=device)
    valid, safe, states, _ = validity_label(U, env, cfg.gamma_max, cfg.n_angles)
    desc = D.descriptor(states, env)[valid]
    ranges = [env.ylim for _ in range(env.n_obs)]
    star_bins = D.build_star_bins(desc, ranges, cfg.nbins, min_count=1)
    modes = D.macro_mode(states, env)[valid]
    present = sorted(set(int(m) for m in modes.tolist()))
    log(f"[omega*] valid {int(valid.sum())}/{n}  bins={len(star_bins)}  "
        f"modes_present={[D.mode_names(env)[m] for m in present]}")
    return star_bins, ranges, present


def pretrain(policy, U_seed, ctx, steps, lr, batch, device, log):
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    policy.train()
    for i in range(steps):
        bi = torch.randint(0, U_seed.shape[0], (min(batch, U_seed.shape[0]),), device=device)
        loss = policy.cfm_loss(U_seed[bi], ctx)
        opt.zero_grad(); loss.backward(); opt.step()
        if i % max(1, steps // 5) == 0:
            log(f"  pretrain {i}/{steps} loss={float(loss.detach()):.4f}")
    policy.eval()


def overlay_from_state(state, env, ctx, cfg, path, title, device, width, depth):
    pol = FlowPolicy(env.T, ctx.numel(), width=width, depth=depth, u_max=env.u_max).to(device)
    pol.load_state_dict(state); pol.eval()
    plots.plot_overlay(pol, env, ctx, cfg, path, title, n=cfg.eval_K // 4 if cfg.eval_K >= 400 else 200, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=["single", "gap"], default="single")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--baseline", choices=["none", "recf", "recnf"], default="none")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=None)       # negative-unlearning weight
    ap.add_argument("--inner", type=int, default=None)         # UpdateFlow steps/round
    ap.add_argument("--temp", type=float, default=None)        # FM explore temperature
    ap.add_argument("--churn", type=float, default=None)
    ap.add_argument("--rho0", type=float, default=None)        # broad fraction (start)
    ap.add_argument("--pretrain", type=int, default=None)
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    env = make_env(args.env, device=device)
    ctx = env_context(env, device)

    cfg = SFConfig()
    cfg.rounds = args.rounds if args.rounds is not None else (60 if args.env == "single" else 80)
    cfg.no_tilt = (args.baseline == "recf")
    cfg.no_finetune = (args.baseline == "recnf")
    pretrain_steps, omega_n, seed_n = 2500, 20000, 8000
    if args.alpha is not None: cfg.alpha = args.alpha
    if args.inner is not None: cfg.inner_steps = args.inner
    if args.temp is not None: cfg.explore_temp = args.temp
    if args.churn is not None: cfg.explore_churn = args.churn
    if args.rho0 is not None: cfg.rho0 = args.rho0
    if args.pretrain is not None: pretrain_steps = args.pretrain
    if args.smoke:
        cfg.rounds = min(cfg.rounds, 4)
        cfg.N, cfg.B, cfg.eval_K, cfg.inner_steps = 96, 24, 400, 25
        cfg.eval_every, cfg.warmup_pos = 1, 24
        pretrain_steps, omega_n, seed_n = 400, 4000, 3000

    tag = f"{args.env}" + ("" if args.baseline == "none" else f"_{args.baseline}") + ("_smoke" if args.smoke else "")
    out = args.out or os.path.join(HERE, "results")
    figdir = os.path.join(HERE, "figures")
    os.makedirs(out, exist_ok=True); os.makedirs(figdir, exist_ok=True)
    logf = open(os.path.join(out, f"{tag}_log.txt"), "w")
    def log(*a):
        msg = " ".join(str(x) for x in a); print(msg); logf.write(msg + "\n"); logf.flush()

    t0 = time.time()
    log(f"=== SafeFlow Exploration  env={args.env} baseline={args.baseline} rounds={cfg.rounds} "
        f"device={device} smoke={args.smoke} ===")
    cand = build_candidate_polytope(env)
    log(f"candidate polytope faces: {0 if cand is None else cand[0].shape[0]} (conservative seed/baseline)")

    # 1) Omega*
    star_bins, ranges, modes_present = build_omega_star(env, cfg, omega_n, device, log)
    gt_modes = D.n_modes(env)
    log(f"ground-truth macro-modes: {gt_modes} ({D.mode_names(env)})")

    # 2) seed (one leaf) + pretrain theta_0
    leaf = "right" if args.env == "single" else "left"
    U_seed = make_seed(env, leaf, seed_n, cfg.gamma_max, cfg.n_angles, device=device)
    log(f"seed leaf={leaf}: {U_seed.shape[0]} valid one-leaf sequences")
    policy = FlowPolicy(env.T, ctx.numel(), width=args.width, depth=args.depth, u_max=env.u_max).to(device)
    pretrain(policy, U_seed, ctx, pretrain_steps, lr=5e-4, batch=256, device=device, log=log)
    import copy as _copy
    seed_state = _copy.deepcopy(policy.state_dict())   # the conservative seed FM (for stage-0/stage-1 viz)

    # pretrained overlay + metrics
    from safeflow import evaluate
    m0 = evaluate(policy, env, ctx, star_bins, ranges, cfg)
    log(f"[pretrained] cov={m0['coverage']:.2f} val={m0['validity']:.2f} "
        f"modecov={m0['mode_coverage']:.2f} vendi={m0['vendi']:.2f} probs={['%.2f'%p for p in m0['mode_probs']]}")
    plots.plot_overlay(policy, env, ctx, cfg, os.path.join(figdir, f"{tag}_overlay_pretrained.png"),
                       f"ENV {args.env} — pretrained seed ({leaf})", n=max(200, cfg.eval_K // 4), device=device)

    # 3) Safe Flow Expansion  (snapshot EVERY eval round so the expansion GIF can animate it)
    snap_rounds = set(range(0, cfg.rounds, cfg.eval_every)) | {cfg.rounds - 1}
    policy, history, snaps = run_safeflow(env, ctx, policy, star_bins, ranges, cfg,
                                          device=device, log=log, snapshot_rounds=snap_rounds)

    # 4) static plots + key overlays
    plots.plot_curves(history, env, os.path.join(figdir, f"{tag}_coverage_validity.png"))
    plots.plot_modecov_vendi(history, env, os.path.join(figdir, f"{tag}_modecov_vendi.png"))
    for rnd in sorted({0, cfg.rounds // 2, cfg.rounds - 1} & set(snaps)):
        overlay_from_state(snaps[rnd], env, ctx, cfg,
                           os.path.join(figdir, f"{tag}_overlay_r{rnd:03d}.png"),
                           f"ENV {args.env} — round {rnd}", device, args.width, args.depth)

    # 5) multi-stage principle GIFs (the Figma loop, separated like the old gifs)
    if not args.no_gif:
        import stage_viz
        final_pol = FlowPolicy(env.T, ctx.numel(), width=args.width, depth=args.depth,
                               u_max=env.u_max).to(device)
        final_pol.load_state_dict(snaps[cfg.rounds - 1]); final_pol.eval()
        # stage 0 (static): seed FM -> expanded FM, vs conservative candidate polytope
        stage_viz.render_seed_vs_expanded(
            seed_state, final_pol, env, ctx, cfg,
            os.path.join(figdir, f"{tag}_stage0_seed_vs_expanded.png"),
            args.width, args.depth, device=device)
        # stage 1 (gif): SafeMPPI sample-then-reject with the (1-gamma)^i RULER (the data engine)
        stage_viz.render_safemppi_gif(
            env, cfg, os.path.join(figdir, f"{tag}_stage1_safemppi_ruler.gif"),
            side=leaf, device=device, log=log)
        # stage 2 (gif): the FM generative field under the ruler + verified-polytope faces (final policy)
        stage_viz.render_certified_gif(
            final_pol, env, ctx, cfg, os.path.join(figdir, f"{tag}_stage2_fm_field_certified.gif"),
            device=device, log=log)
        # stage 3 (gif): Safe Flow Expansion -- FM fits the safe set better; polytope changes over rounds
        stage_viz.render_expansion_gif(
            snaps, env, ctx, cfg, history, os.path.join(figdir, f"{tag}_stage3_safeflow_expansion.gif"),
            args.width, args.depth, device=device, log=log)

    summary = {
        "env": args.env, "baseline": args.baseline, "rounds": cfg.rounds,
        "cfg": cfg.__dict__, "modes_present_omega": modes_present, "gt_modes": gt_modes,
        "pretrained": m0, "final": history[-1] if history else None,
        "history": history, "wall_time_s": time.time() - t0,
    }
    with open(os.path.join(out, f"{tag}_history.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log(f"DONE in {time.time()-t0:.1f}s  ->  {out}/{tag}_history.json  &  {figdir}/{tag}_*.png")
    logf.close()


if __name__ == "__main__":
    main()
