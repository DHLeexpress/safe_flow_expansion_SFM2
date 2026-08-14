"""Diagnostic — does the grid encoder ENCODE THE SAFETY FUNCTION and find patterns?

(1) input polar safety grid  vs  reconstruction from grid_token (the aux 'polytope→context' head).
(2) PCA of grid_token over many states, colored by the true safety proximity (min clearance) — if the
    token space organizes by safety, the encoder found the pattern.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _paths
import config as C
from windowed_policy import GridLowFlowPolicy
from polar_grid import polar_grid, N_THETA, N_R
from di_grid_viz import load_best_config, mppi_rollout


def load(scene, device):
    ck = torch.load(C.scene_result(scene, "pretrained.pt"), weights_only=False)
    pol = GridLowFlowPolicy(H_pred=ck["H_pred"], u_max=ck["u_max"]).to(device)
    pol.load_state_dict(ck["state_dict"])
    pol.eval()
    return pol


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="gap", choices=C.SCENE_NAMES)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    env = C.make_scene(args.scene)
    pol = load(args.scene, args.device)
    goal = env.goal.detach().cpu().numpy(); obs = env.obstacles.detach().cpu().numpy()
    cfg = load_best_config()

    # gather states along a few SafeMPPI rollouts
    states = []
    for g in C.GAMMAS:
        _, path = mppi_rollout(env, g, cfg, steps=50, seed_base=int(g * 1000))
        states.extend(path[3:45])
    states = np.array(states)
    grids = np.stack([polar_grid(s, goal, obs)[0] for s in states])            # [M,3,16,12]
    gt = pol.grid_token(torch.tensor(grids, device=args.device)).cpu().numpy()  # [M, token]
    recon = pol.safety_decoder(torch.tensor(gt, device=args.device)).cpu().numpy().reshape(-1, 3, N_THETA, N_R)
    minclr = np.array([float((np.linalg.norm(s[None] - obs[:, :2], axis=1) - obs[:, 2]).min()) for s in states])

    # (1) input vs recon for 4 representative states (by increasing danger)
    order = np.argsort(minclr)
    picks = [order[int(f * (len(order) - 1))] for f in (0.05, 0.35, 0.65, 0.95)]
    ch = ["occupancy", "polytope_mask", "H_P"]
    fig, axes = plt.subplots(len(picks), 6, figsize=(13, 2.1 * len(picks)))
    for r, i in enumerate(picks):
        for c in range(3):
            axes[r][c].imshow(grids[i, c], aspect="auto", cmap="magma", vmin=-1, vmax=1)
            axes[r][c + 3].imshow(recon[i, c], aspect="auto", cmap="magma", vmin=-1, vmax=1)
            if r == 0:
                axes[r][c].set_title(f"in: {ch[c]}", fontsize=8)
                axes[r][c + 3].set_title(f"recon: {ch[c]}", fontsize=8)
            for a in (axes[r][c], axes[r][c + 3]):
                a.set_xticks([]); a.set_yticks([])
        axes[r][0].set_ylabel(f"clr={minclr[i]:+.2f}", fontsize=8)
    recon_mse = float(((grids - recon) ** 2).mean())
    fig.suptitle(f"[{args.scene}] safety-grid encode→reconstruct (aux MSE={recon_mse:.3f}) — θ(rows)×r(cols)", fontsize=11)
    fig.tight_layout()
    p1 = C.scene_fig(args.scene, "diag_safety_recon.png")
    fig.savefig(p1, dpi=130); plt.close(fig)

    # (2) PCA of grid_token colored by safety proximity
    X = gt - gt.mean(0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    Z = X @ Vt[:2].T
    fig2, ax = plt.subplots(figsize=(6.2, 5.2))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=minclr, cmap="RdYlGn", s=14, vmin=-0.2, vmax=1.5)
    plt.colorbar(sc, label="true min clearance (safety)")
    ax.set_title(f"[{args.scene}] grid_token PCA — colored by safety proximity\n"
                 f"(organized by safety ⇒ encoder found the pattern)", fontsize=10)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig2.tight_layout()
    p2 = C.scene_fig(args.scene, "diag_safety_token_pca.png")
    fig2.savefig(p2, dpi=130); plt.close(fig2)
    print(f"[{args.scene}] recon MSE={recon_mse:.4f}  → {p1}, {p2}", flush=True)


if __name__ == "__main__":
    main()
