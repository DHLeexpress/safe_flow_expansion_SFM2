"""STAGE 3a — pretrain the windowed γ+grid-conditioned CFM policy on the MPPI windows (W&B live loss).

Trains `GridLowFlowPolicy` (ctx = concat(low_token, grid_token)) with CFM (noise→U_local). Saves
pretrained_<scene>.pt and a per-γ closed-loop FM-rollout overlay (the 'pretrained policies' figure).
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import _paths
import config as C
from cfm_mppi.data.windowed_dataset import WindowedDataset, windowed_collate
from windowed_policy import GridLowFlowPolicy, fm_rollout
import wandb_utils as W


@torch.no_grad()
def plot_per_gamma_fm(policy, env, gammas, out, n=24, device="cpu", title=""):
    fig, axes = plt.subplots(1, len(gammas), figsize=(4.3 * len(gammas), 4.2), squeeze=False)
    obs = env.obstacles.detach().cpu().numpy()
    cmap = plt.get_cmap("viridis")
    for ci, g in enumerate(gammas):
        ax = axes[0][ci]
        for (ox, oy, rr) in obs:
            ax.add_patch(Circle((ox, oy), rr, facecolor="#c8a2c8", edgecolor="#7b3294", lw=0.7, alpha=0.7, zorder=3))
        paths, _ = fm_rollout(policy, env, g, n_traj=n, temp=1.0, device=device, record=False)
        col = cmap((g - min(gammas)) / max(max(gammas) - min(gammas), 1e-6))
        for p in paths:
            ax.plot(p[:, 0], p[:, 1], "-", color=col, lw=0.7, alpha=0.4, zorder=4)
        ax.scatter([env.x0[0]], [env.x0[1]], s=45, c="#00a000", edgecolor="k", zorder=6)
        ax.scatter([env.goal[0]], [env.goal[1]], marker="*", s=140, c="gold", edgecolor="k", zorder=6)
        ax.set_xlim(*env.xlim); ax.set_ylim(*env.ylim); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"γ={g}", fontsize=11)
    fig.suptitle(title or "Pretrained windowed FM policy — closed-loop rollouts per γ", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="gap", choices=C.SCENE_NAMES)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--aux-weight", type=float, default=0.3)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    W.add_wandb_args(ap)
    args = ap.parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 6)

    device = args.device
    ddir = C.dataset_dir(args.scene)
    train = WindowedDataset(os.path.join(ddir, "train.pt"))
    val = WindowedDataset(os.path.join(ddir, "val.pt"))
    loader = DataLoader(train, batch_size=args.batch, shuffle=True, collate_fn=windowed_collate, drop_last=True)
    env = C.make_scene(args.scene)
    policy = GridLowFlowPolicy(H_pred=C.H_PRED, u_max=float(env.u_max)).to(device)
    print(f"=== STAGE 3a [{args.scene}] pretrain: {len(train)} windows, ctx_dim={policy.ctx_dim}, "
          f"epochs={args.epochs} ===", flush=True)
    run = W.init_run(args, name=f"{args.scene}-pretrain", dir=C.RESULTS, group=args.scene,
                     config={"stage": "pretrain", "scene": args.scene, "n_windows": len(train),
                             "epochs": args.epochs, "batch": args.batch, "lr": args.lr, "H_pred": C.H_PRED})
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    lam_aux = args.aux_weight
    gstep = 0
    for epoch in range(args.epochs):
        policy.train()
        cfms, auxs = [], []
        for batch in loader:
            grid = batch["grid"].to(device); low = batch["low_dim"].to(device)
            U = batch["U_local"].to(device)
            ctx = policy.ctx_from(grid, low)
            cfm = policy.cfm_loss(U, ctx)
            aux = policy.aux_safety_loss(grid)                       # polytope→context reconstruction
            loss = cfm + lam_aux * aux
            opt.zero_grad(); loss.backward(); opt.step()
            cfms.append(float(cfm.detach())); auxs.append(float(aux.detach()))
            W.log(run, {"pretrain/cfm_loss": cfms[-1], "pretrain/aux_safety_loss": auxs[-1],
                        "pretrain/lr": sched.get_last_lr()[0]}, step=gstep)
            gstep += 1
        sched.step()
        policy.eval()
        with torch.no_grad():
            vb = windowed_collate([val[i] for i in range(min(len(val), 1024))])
            vctx = policy.ctx_from(vb["grid"].to(device), vb["low_dim"].to(device))
            vcfm = float(policy.cfm_loss(vb["U_local"].to(device), vctx))
            vaux = float(policy.aux_safety_loss(vb["grid"].to(device)))
        W.log(run, {"pretrain/epoch_cfm": float(np.mean(cfms)), "pretrain/epoch_aux": float(np.mean(auxs)),
                    "pretrain/val_cfm": vcfm, "pretrain/val_aux": vaux}, step=gstep)
        print(f"  epoch {epoch}: cfm {np.mean(cfms):.4f}  aux {np.mean(auxs):.4f}  val_cfm {vcfm:.4f}  "
              f"val_aux {vaux:.4f}", flush=True)
    vloss = vcfm

    out = C.scene_result(args.scene, "pretrained.pt")
    torch.save({"state_dict": policy.state_dict(), "scene": args.scene, "H_pred": C.H_PRED,
                "u_max": float(env.u_max)}, out)
    print("saved", out, flush=True)
    fig = C.scene_fig(args.scene, "stage3_pretrained_per_gamma.png")
    plot_per_gamma_fm(policy, env, C.GAMMAS, fig, device=device,
                      title=f"Pretrained windowed FM [{args.scene}] — closed-loop per γ")
    W.log_image(run, "pretrained_per_gamma", fig)
    W.finish(run, summary={"val_loss": vloss})
    print("saved", fig, flush=True)


if __name__ == "__main__":
    main()
