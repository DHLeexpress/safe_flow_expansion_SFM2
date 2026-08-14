"""Stage C — pretrain the γ-conditioned GridGRUFlowPolicy on pooled SafeMPPI grid data (cfm + aux).

One policy conditioned on γ (a low-dim feature). Loss = cfm_loss(U|ctx) + aux_w · aux_safety_loss(grid),
Adam + CosineAnnealing, W&B per-step/epoch. Ends with a quick per-γ baseline coverage/validity so we can
see how small the pretrained generable set is (the thing expansion must grow).
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

import _paths  # noqa: F401
import grid_scene as GS
import grid_policy as GP
import grid_rollout as GR
import grid_metrics as GM
import wandb_utils as W

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "dataset")


def load_pool(device="cpu"):
    G, L, H, U = [], [], [], []
    for f in sorted(glob.glob(os.path.join(DATA, "windows_g*.pt"))):
        d = torch.load(f)
        G.append(d["grid"]); L.append(d["low5"]); H.append(d["hist"]); U.append(d["U"])
    return (torch.cat(G), torch.cat(L), torch.cat(H), torch.cat(U))


def baseline_eval(policy, env, gammas, n_deploy, device):
    out = {}
    policy.eval()
    for g in gammas:
        paths = GR.deploy_many(policy, env, g, n_deploy, T=env.T, nfe=8, device=device)
        val, cov, steps, _ = GM.measure(paths, env, g)
        out[g] = dict(validity=val, coverage=cov, avg_steps=steps)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--aux-weight", type=float, default=0.3)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--gru-dim", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(HERE, "pretrained.pt"))
    ap.add_argument("--n-eval", type=int, default=30)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    W.add_wandb_args(ap)
    args = ap.parse_args()
    dev = args.device

    grid, low5, hist, U = load_pool()
    n = grid.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    grid, low5, hist, U = grid[perm], low5[perm], hist[perm], U[perm]
    nval = max(512, n // 10)
    tr = TensorDataset(grid[nval:], low5[nval:], hist[nval:], U[nval:])
    va = (grid[:nval].to(dev), low5[:nval].to(dev), hist[:nval].to(dev), U[:nval].to(dev))
    dl = DataLoader(tr, batch_size=args.batch, shuffle=True, drop_last=True)
    print(f"=== Stage C pretrain: {n} windows ({n-nval} train / {nval} val), depth={args.depth}, "
          f"gru={args.gru_dim}, dev={dev} ===", flush=True)

    pol = GP.build_policy(depth=args.depth, gru_dim=args.gru_dim, device=dev)
    opt = torch.optim.Adam(pol.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    run = W.init_run(args, name=f"pretrain-d{args.depth}-g{args.gru_dim}",
                     config=vars(args), group="grid-safeflow")

    step = 0
    for ep in range(args.epochs):
        pol.train()
        ep_cfm = ep_aux = 0.0
        for gb, lb, hb, ub in dl:
            gb, lb, hb, ub = gb.to(dev), lb.to(dev), hb.to(dev), ub.to(dev)
            ctx = pol.ctx_from(gb, lb, hb)
            cfm = pol.cfm_loss(ub, ctx)
            aux = pol.aux_safety_loss(gb)
            loss = cfm + args.aux_weight * aux
            opt.zero_grad(); loss.backward(); opt.step()
            ep_cfm += float(cfm); ep_aux += float(aux); step += 1
            if step % 50 == 0:
                W.log(run, {"pretrain/cfm": float(cfm), "pretrain/aux": float(aux),
                            "pretrain/lr": opt.param_groups[0]["lr"]}, step=step)
        sched.step()
        nb = len(dl)
        with torch.no_grad():
            pol.eval()
            vctx = pol.ctx_from(va[0], va[1], va[2])
            vcfm = float(pol.cfm_loss(va[3], vctx)); vaux = float(pol.aux_safety_loss(va[0]))
        W.log(run, {"pretrain/epoch_cfm": ep_cfm / nb, "pretrain/epoch_aux": ep_aux / nb,
                    "pretrain/val_cfm": vcfm, "pretrain/val_aux": vaux}, step=step)
        print(f"ep {ep:02d}: cfm {ep_cfm/nb:.4f} aux {ep_aux/nb:.4f} | val_cfm {vcfm:.4f}", flush=True)

    GP.save_policy(pol, args.out, extra={"pretrain_args": vars(args)})
    print(f"saved pretrained -> {args.out}", flush=True)

    env = GS.make_grid()
    base = baseline_eval(pol, env, GS.__dict__.get("GAMMAS", [0.1, 0.5, 1.0]) if False else [0.1, 0.5, 1.0],
                         args.n_eval, dev)
    for g, m in base.items():
        print(f"  baseline γ{g}: coverage {m['coverage']*100:.1f}% validity {m['validity']*100:.1f}% "
              f"avg_steps {m['avg_steps']:.0f}", flush=True)
        W.log(run, {f"baseline/coverage_g{g}": m["coverage"], f"baseline/validity_g{g}": m["validity"]})
    W.finish(run, summary={"val_cfm": vcfm})


if __name__ == "__main__":
    main()
