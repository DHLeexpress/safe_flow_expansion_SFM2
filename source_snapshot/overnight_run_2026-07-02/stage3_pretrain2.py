"""Stage B'/C' — v2 pretrain (2026-07-03): cfm-ONLY loss (no aux decoder), proper validation, and the
module-characterization the user asked for (1b): per-module grad flow, input-branch ablation, encoder
output-spread collapse guard. Trains ONE width; run once per candidate width {256,192,128}; the sweep
orchestrator picks the lightest within tolerance.

Outputs: pretrained2_w{W}.pt (best-val checkpoint) + _last.pt, results/pretrain2/w{W}.json,
figures/pretrain2_w{W}.png.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _paths  # noqa: F401
import grid_scene as GS
import grid_policy2 as GP2
import grid_rollout as GR
import grid_metrics2 as GM2
import wandb_utils as W

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "pretrain2")
FIG = os.path.join(HERE, "figures")
GAMMAS = [0.1, 0.5, 1.0]


def load_pool():
    import glob
    G, L, H, U = [], [], [], []
    for f in sorted(glob.glob(os.path.join(HERE, "dataset", "windows_g*.pt"))):
        d = torch.load(f)
        G.append(d["grid"]); L.append(d["low5"]); H.append(d["hist"]); U.append(d["U"])
    return (torch.cat(G), torch.cat(L), torch.cat(H), torch.cat(U))


def grad_rms(policy):
    """Per-param RMS gradient per module group — comparable across module sizes."""
    out = {}
    for k, m in policy.module_groups().items():
        g2 = 0.0
        n = 0
        for p in m.parameters():
            if p.grad is not None:
                g2 += float((p.grad ** 2).sum())
            n += p.numel()
        out[k] = (g2 / max(n, 1)) ** 0.5
    return out


@torch.no_grad()
def val_eval(policy, va, seed=1234):
    """val_cfm with IDENTICAL noise draws (seeded) so numbers are comparable across epochs/ablations.
    RNG state (CPU+CUDA) is saved/restored so the eval never perturbs training randomness."""
    policy.eval()
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    ctx = policy.ctx_from(va[0], va[1], va[2])
    out = float(policy.cfm_loss(va[3], ctx))
    torch.random.set_rng_state(cpu_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)
    return out


@torch.no_grad()
def ablation_eval(policy, va, seed=1234):
    """Zero one conditioning branch at a time (same noise draws) -> Delta val_cfm = branch importance."""
    g, l, h, u = va
    full = val_eval(policy, (g, l, h, u), seed)
    zg = val_eval(policy, (torch.zeros_like(g), l, h, u), seed)
    l0 = l.clone(); l0[:, :4] = 0.0                      # zero relgoal+vel, KEEP gamma
    zl = val_eval(policy, (g, l0, h, u), seed)
    zh = val_eval(policy, (g, l, torch.zeros_like(h), u), seed)
    return dict(full=full, zero_grid=zg, zero_lowfeat=zl, zero_hist=zh,
                d_grid=zg - full, d_lowfeat=zl - full, d_hist=zh - full)


@torch.no_grad()
def encoder_spread(policy, va):
    toks = policy.encoder_tokens(va[0], va[1], va[2])       # dict of existing encoder outputs (empty=reduced)
    return {f"std_{k}": float(v.std(0).mean()) for k, v in toks.items()}


def baseline_eval2(policy, env, n_deploy, device):
    out = {}
    policy.eval()
    for g in GAMMAS:
        paths = GR.deploy_many(policy, env, g, n_deploy, T=env.T, nfe=8, device=device)
        m = GM2.measure2(paths, env, g, covered=set())
        out[str(g)] = {k: m[k] for k in ("validity", "coverage_cum", "coverage_final", "reach_rate", "avg_steps")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--no-gru", action="store_true", help="drop GRU history from ctx")
    ap.add_argument("--no-enc-low", action="store_true", help="drop E_l encoder (raw low -> trunk)")
    ap.add_argument("--no-grid", action="store_true", help="drop grid CNN E_g (no encoded safety)")
    ap.add_argument("--raw-hist", action="store_true", help="append raw last-k executed actions to ctx")
    ap.add_argument("--raw-hist-k", type=int, default=10)
    ap.add_argument("--enc-hist", action="store_true", help="route raw hist THROUGH E_l (25->64->48)")
    ap.add_argument("--dropout", type=float, default=0.0, help="trunk dropout (regularizer tweak)")
    ap.add_argument("--tag", default=None, help="output filename tag (e.g. 'reduced' -> pretrained2_reduced.pt)")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-eval", type=int, default=15)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    W.add_wandb_args(ap)
    args = ap.parse_args()
    dev = args.device
    os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)

    grid, low5, hist, U = load_pool()
    n = grid.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    grid, low5, hist, U = grid[perm], low5[perm], hist[perm], U[perm]
    nval = max(2048, n // 10)
    tr = TensorDataset(grid[nval:], low5[nval:], hist[nval:], U[nval:])
    va = (grid[:nval].to(dev), low5[:nval].to(dev), hist[:nval].to(dev), U[:nval].to(dev))
    vg = va[1][:, 4]                                        # per-gamma val masks
    per_g_idx = {g: torch.nonzero((vg - g).abs() < 1e-3).flatten() for g in GAMMAS}
    dl = DataLoader(tr, batch_size=args.batch, shuffle=True, drop_last=True)
    TAG = args.tag or f"w{args.width}"
    variant = f"use_gru={not args.no_gru} encode_low={not args.no_enc_low} use_grid={not args.no_grid}"
    print(f"=== pretrain2 [{TAG}] W{args.width} ({variant}): {n} windows "
          f"({n - nval} train / {nval} val), cfm-only ===", flush=True)

    pol = GP2.build_policy2(width=args.width, use_gru=not args.no_gru, encode_low=not args.no_enc_low,
                            use_grid=not args.no_grid, raw_hist=args.raw_hist, raw_hist_k=args.raw_hist_k,
                            enc_hist=args.enc_hist, dropout=args.dropout, device=dev)
    rep = GP2.param_report(pol)
    print("params:", rep, flush=True)
    opt = torch.optim.Adam(pol.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    run = W.init_run(args, name=f"pretrain2-{TAG}", config={**vars(args), **rep}, group="sweep-0703")

    hist_rows = []
    best = dict(val=float("inf"), ep=-1, sd=None)
    step = 0
    for ep in range(args.epochs):
        pol.train()
        ep_cfm = 0.0
        gsum = {k: 0.0 for k in pol.module_groups()}
        nb = 0
        for gb, lb, hb, ub in dl:
            gb, lb, hb, ub = gb.to(dev), lb.to(dev), hb.to(dev), ub.to(dev)
            loss = pol.cfm_loss(ub, pol.ctx_from(gb, lb, hb))
            opt.zero_grad(); loss.backward()
            for k, v in grad_rms(pol).items():
                gsum[k] += v
            opt.step()
            ep_cfm += float(loss); nb += 1; step += 1
        sched.step()
        vall = val_eval(pol, va)
        vper = {g: val_eval(pol, tuple(x[per_g_idx[g]] for x in va)) for g in GAMMAS}
        spread = encoder_spread(pol, va)
        row = dict(epoch=ep, train_cfm=ep_cfm / nb, val_cfm=vall,
                   **{f"val_cfm_g{g}": vper[g] for g in GAMMAS},
                   **{f"grad_{k}": v / nb for k, v in gsum.items()}, **spread)
        hist_rows.append(row)
        W.log(run, {f"pretrain2/{k}": v for k, v in row.items() if k != "epoch"}, step=ep)
        if vall < best["val"]:
            best = dict(val=vall, ep=ep, sd={k: v.detach().cpu().clone() for k, v in pol.state_dict().items()})
        if ep % 10 == 0 or ep == args.epochs - 1:
            gstr = " ".join(f"{k[5:]} {v:.1e}" for k, v in row.items() if k.startswith("grad_"))
            print(f"ep {ep:03d}: train {row['train_cfm']:.4f} val {vall:.4f} "
                  f"(γ: {' '.join(f'{vper[g]:.3f}' for g in GAMMAS)}) grads {gstr}", flush=True)

    out_last = os.path.join(HERE, f"pretrained2_{TAG}_last.pt")
    GP2.save_policy2(pol, out_last, extra={"pretrain_args": vars(args)})
    pol.load_state_dict(best["sd"])                          # best-val checkpoint is THE model
    out_best = os.path.join(HERE, f"pretrained2_{TAG}.pt")
    GP2.save_policy2(pol, out_best, extra={"pretrain_args": vars(args), "best_val": best["val"], "best_ep": best["ep"]})
    print(f"saved {out_best} (best ep {best['ep']}, val {best['val']:.4f}) + _last.pt", flush=True)

    abl = ablation_eval(pol, va)
    spread = encoder_spread(pol, va)
    env = GS.make_grid()
    base = baseline_eval2(pol, env, args.n_eval, dev)
    for g in GAMMAS:
        b = base[str(g)]
        print(f"  baseline2 γ{g}: validity2 {b['validity']*100:.0f}% reach {b['reach_rate']*100:.0f}% "
              f"cov_final {b['coverage_final']*100:.1f}% steps {b['avg_steps']:.0f}", flush=True)
    print(f"  ablation Δval_cfm: grid +{abl['d_grid']:.4f}  lowfeat +{abl['d_lowfeat']:.4f} "
          f"hist +{abl['d_hist']:.4f} (bigger = branch matters more)", flush=True)

    summary = dict(width=args.width, params=rep, best_val_cfm=best["val"], best_epoch=best["ep"],
                   ablation=abl, encoder_spread=spread, baseline=base,
                   val_cfm_per_gamma={str(g): hist_rows[-1][f"val_cfm_g{g}"] for g in GAMMAS},
                   history=hist_rows)
    with open(os.path.join(RES, f"{TAG}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    ep_x = [r["epoch"] for r in hist_rows]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    ax[0].plot(ep_x, [r["train_cfm"] for r in hist_rows], label="train")
    ax[0].plot(ep_x, [r["val_cfm"] for r in hist_rows], label="val")
    ax[0].axvline(best["ep"], ls="--", color="#999", lw=1); ax[0].set_yscale("log"); ax[0].legend()
    ax[0].set_title(f"[{TAG}] cfm (best ep {best['ep']})")
    for k in ("E_g", "E_l", "GRU", "trunk", "head"):
        if all(f"grad_{k}" in r for r in hist_rows):
            ax[1].plot(ep_x, [r[f"grad_{k}"] for r in hist_rows], label=k)
    ax[1].set_yscale("log"); ax[1].legend(fontsize=8); ax[1].set_title("per-module grad RMS (flow of learning)")
    ax[2].bar(["grid", "lowfeat", "hist"], [abl["d_grid"], abl["d_lowfeat"], abl["d_hist"]], color="#4a7fb5")
    ax[2].set_title("input-branch ablation Δval_cfm")
    for a in ax:
        a.grid(alpha=.25)
    fig.suptitle(f"pretrain2 [{TAG}] W{args.width}: validation + module characterization", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"pretrain2_{TAG}.png"), dpi=130)
    W.log_image(run, "pretrain2/plot", os.path.join(FIG, f"pretrain2_{TAG}.png"))
    W.finish(run, summary={"best_val_cfm": best["val"]})


if __name__ == "__main__":
    main()
