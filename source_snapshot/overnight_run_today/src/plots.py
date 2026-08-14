"""Plots: multi-modal trajectory overlays + coverage-validity / vendi curves."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import torch

from dynamics import rollout, clip_controls
from safeflow import validity_label
import descriptors as D

MODE_COLORS = {
    "single": {0: "#1a9850", 1: "#2166ac"},                 # LEFT green, RIGHT blue
    "gap": {0: "#1a9850", 1: "#f46d43", 2: "#2166ac"},      # LEFT green, GAP orange, RIGHT blue
}


def _draw_env(ax, env):
    for j in range(env.n_obs):
        cx, cy, r = [float(v) for v in env.obstacles[j].cpu()]
        ax.add_patch(Circle((cx, cy), r, facecolor="#7b3294", alpha=0.35, edgecolor="#4d004b", zorder=3))
        ax.add_patch(Circle((cx, cy), r + env.r_robot, facecolor="none",
                             edgecolor="#7b3294", ls="--", lw=1.0, alpha=0.6, zorder=3))
    p0 = env.x0[:2].cpu(); g = env.goal.cpu()
    ax.scatter([p0[0]], [p0[1]], s=90, c="#1a9850", edgecolor="k", marker="o", zorder=7, label="start")
    ax.scatter([g[0]], [g[1]], s=160, c="gold", edgecolor="k", marker="*", zorder=7, label="goal")
    ax.set_xlim(*env.xlim); ax.set_ylim(*env.ylim); ax.set_aspect("equal")
    ax.grid(alpha=0.2)


@torch.no_grad()
def plot_overlay(policy, env, ctx, cfg, path, title, n=300, device="cpu"):
    U = clip_controls(policy.sample(n, ctx, nfe=cfg.nfe), env)
    valid, safe, states, _ = validity_label(U, env, cfg.gamma_max, cfg.n_angles)
    modes = D.macro_mode(states, env)
    states = states.cpu(); valid = valid.cpu(); modes = modes.cpu()
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    _draw_env(ax, env)
    cols = MODE_COLORS[env.name]
    # unsafe/invalid faint grey
    for i in torch.where(~valid)[0][:120]:
        tr = states[i, :, :2]
        ax.plot(tr[:, 0], tr[:, 1], color="0.6", alpha=0.10, lw=0.6, zorder=4)
    # valid colored by mode
    for i in torch.where(valid)[0]:
        tr = states[i, :, :2]
        ax.plot(tr[:, 0], tr[:, 1], color=cols[int(modes[i])], alpha=0.22, lw=0.7, zorder=5)
    names = D.mode_names(env)
    counts = [int((modes[valid] == m).sum()) for m in range(D.n_modes(env))]
    sub = "  ".join(f"{names[m]}:{counts[m]}" for m in range(len(names)))
    ax.set_title(f"{title}\nvalid {int(valid.sum())}/{n}   {sub}", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_curves(history, env, path):
    r = [h["round"] for h in history]
    cov = [100 * h["coverage"] for h in history]
    val = [100 * h["validity"] for h in history]
    fig, ax1 = plt.subplots(figsize=(5.6, 4.0))
    ax1.plot(r, cov, "-o", color="#2166ac", lw=2, ms=4, label="coverage")
    ax1.set_xlabel("round"); ax1.set_ylabel("coverage (%)", color="#2166ac")
    ax1.set_ylim(0, 105); ax1.tick_params(axis="y", labelcolor="#2166ac")
    ax2 = ax1.twinx()
    ax2.plot(r, val, "-s", color="#1a9850", lw=2, ms=4, label="validity")
    ax2.set_ylabel("validity (%)", color="#1a9850"); ax2.set_ylim(0, 105)
    ax2.tick_params(axis="y", labelcolor="#1a9850")
    ax1.set_title(f"ENV {env.name}: coverage & validity vs round")
    ax1.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_modecov_vendi(history, env, path):
    r = [h["round"] for h in history]
    mc = [100 * h["mode_coverage"] for h in history]
    vd = [h["vendi"] for h in history]
    fig, ax1 = plt.subplots(figsize=(5.6, 4.0))
    ax1.plot(r, mc, "-o", color="#762a83", lw=2, ms=4)
    ax1.set_xlabel("round"); ax1.set_ylabel("mode coverage (%)", color="#762a83")
    ax1.set_ylim(0, 105)
    ax2 = ax1.twinx()
    ax2.plot(r, vd, "-^", color="#e08214", lw=2, ms=4)
    ax2.set_ylabel("Vendi diversity", color="#e08214")
    ax2.set_ylim(0.8, max(vd) * 1.15 if vd else 2.0)
    ax1.set_title(f"ENV {env.name}: mode coverage & diversity")
    ax1.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
