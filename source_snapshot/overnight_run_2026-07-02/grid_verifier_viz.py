"""Visualize the verifier polytope + DTCBF level sets (green) moving with the FM rollout, per γ, and snapshot
the multi-step-safety FAILURE case.

For one 10-window seg (center c=seg[0]) the verifier builds faces {a_k·(x−c) ≤ m_k}; the barrier is
H_P(x)=min_k (m_k − a_k·(x−c))/m_k (=1 at c, 0 on a face). The DTCBF certificate requires every step to stay
in the shrinking safe level set:  H_P(x_t) ≥ (1−γ)^t. A step that drops below its level (red) is a multi-step
safety violation — even if it is collision-free. draw_verifier_window() renders that; verifier_movie() animates
it along the FM rollout for the 3 γ; snapshot_failures() saves the first failing window per γ.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

import _paths  # noqa: F401
import verifier_polytope as VP

GCOL = {0.1: "#3b6fd6", 0.5: "#2ca02c", 1.0: "#d62728"}


def _hp_field(faces, c, GX, GY):
    P = np.stack([GX.ravel() - c[0], GY.ravel() - c[1]], 1)
    H = np.full(P.shape[0], np.inf)
    for f in faces:
        m = float(f.m)
        if m > 1e-9:
            H = np.minimum(H, (m - P @ np.asarray(f.a, float)) / m)
    return H.reshape(GX.shape)


def _hp_at(faces, c, pts):
    P = np.asarray(pts, float) - c
    H = np.full(P.shape[0], np.inf)
    for f in faces:
        m = float(f.m)
        if m > 1e-9:
            H = np.minimum(H, (m - P @ np.asarray(f.a, float)) / m)
    return H


def draw_verifier_window(ax, seg, obs, r_robot, gamma, trail=None, xlim=(-0.7, 5.7), ylim=(-0.7, 5.7)):
    """Draw one window's verifier: green level sets, polytope boundary, the H-step plan with per-step
    barrier satisfaction, obstacles, and the running trail. Returns (ok, n_violating_steps)."""
    seg = np.asarray(seg, float); c = seg[0]
    ok, faces, raw, R_eff = VP.certify_window(seg, obs, r_robot, gamma, R=2.5, n_theta=180)
    gx = np.linspace(c[0] - R_eff, c[0] + R_eff, 130); gy = np.linspace(c[1] - R_eff, c[1] + R_eff, 130)
    GX, GY = np.meshgrid(gx, gy); H = _hp_field(faces, c, GX, GY)
    # GAMMA-DEPENDENT level sets: the DTCBF requires H_P(x_t) >= (1-gamma)^t, so shade the shrinking allowed
    # region by those thresholds. Conservative gamma -> levels bunched near the center (tight tube);
    # aggressive gamma=1 -> single band = the whole polytope (H_P>=0).
    Hn = len(seg)
    lv = sorted(set([0.0, 1.0] + [round(float((1.0 - gamma) ** t), 4) for t in range(Hn)]))
    ax.contourf(GX, GY, H, levels=lv, cmap="Greens", alpha=.55, zorder=1)
    ax.contour(GX, GY, H, levels=[0.0], colors="#006d2c", linewidths=1.6, zorder=3)              # polytope boundary
    tube = float((1.0 - gamma) ** (Hn - 1))                                                       # loosest (final-step) CBF level
    ax.contour(GX, GY, H, levels=[tube], colors="#08519c", linewidths=1.3, linestyles="--", zorder=3.2)  # reachable tube
    for k in range(6):
        ax.axvline(k, color="#eee", lw=.5, zorder=0); ax.axhline(k, color="#eee", lw=.5, zorder=0)
    ax.add_patch(Rectangle((0, 0), 5, 5, fill=False, edgecolor="#999", lw=1.0, zorder=0.5))
    for j, (ox, oy, r) in enumerate(obs):
        ax.add_patch(Circle((ox, oy), r, facecolor="#b8b8b8" if j >= 16 else "#c8a2c8",
                            edgecolor="#777", lw=.4, alpha=.85, zorder=4))
    if trail is not None and len(trail) > 1:
        ax.plot(np.asarray(trail)[:, 0], np.asarray(trail)[:, 1], "-", color="#444", lw=1.0, alpha=.6, zorder=5)
    # two failure modes: (a) COLLISION = step inside an obstacle (infeasible face); (b) BARRIER = H_P(x_t) < (1-γ)^t
    alpha = (1.0 - gamma) ** np.arange(len(seg))
    hp = _hp_at(faces, c, seg)
    barrier = hp < alpha - 1e-6
    dcl = np.linalg.norm(seg[:, None] - obs[None, :, :2], axis=2) - obs[None, :, 2] - r_robot
    coll = dcl.min(1) < 0.0
    good = ~(barrier | coll)
    ax.plot(seg[:, 0], seg[:, 1], "-", color="#111", lw=1.0, zorder=6)
    ax.scatter(seg[good, 0], seg[good, 1], s=20, c="#111", zorder=7)
    if barrier.any():
        ax.scatter(seg[barrier, 0], seg[barrier, 1], s=75, marker="X", c="#e6191b", edgecolor="k", lw=.6,
                   zorder=8, label="barrier<(1−γ)^t")
    if coll.any():
        ax.scatter(seg[coll, 0], seg[coll, 1], s=95, marker="o", facecolor="none", edgecolor="#e6191b", lw=2.0,
                   zorder=8, label="collision")
    ax.scatter([c[0]], [c[1]], s=45, marker="o", c="white", edgecolor="k", zorder=9)
    imin = int(dcl.min(1).argmin())                              # tightest step (the infeasible-face culprit)
    if not ok:
        ax.scatter([seg[imin, 0]], [seg[imin, 1]], s=120, marker="D", facecolor="none", edgecolor="#b30000",
                   lw=1.8, zorder=8)
        ax.annotate(f"min clr {dcl.min():+.3f} m", (seg[imin, 0], seg[imin, 1]), textcoords="offset points",
                    xytext=(6, 6), fontsize=7.5, color="#b30000")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    why = "" if ok else ("  collision" if coll.any() else ("  barrier-violation" if barrier.any() else "  infeasible-face (grazes obstacle)"))
    ax.set_title(f"γ={gamma}   cert={'OK' if ok else 'FAIL'}{why}", fontsize=10,
                 color=("#006d2c" if ok else "#e6191b"))
    if not ok and (barrier.any() or coll.any()):
        ax.legend(loc="upper left", fontsize=7, framealpha=.85)
    return bool(ok), int(barrier.sum() + coll.sum())


def snapshot_failures(policy, env, gammas, out, deploy_fn, H=10, n_try=25):
    """For each γ, deploy until a trajectory has a FAILING window; render that window. 3 columns."""
    obs = env.obstacles.detach().cpu().numpy(); rr = float(env.r_robot)
    fig, axes = plt.subplots(1, len(gammas), figsize=(4.6 * len(gammas), 4.9), squeeze=False)
    for ci, g in enumerate(gammas):
        chosen = None
        for _ in range(n_try):
            p = deploy_fn(policy, env, g)
            for t in range(0, len(p) - H - 1, 2):
                seg = p[t:t + H + 1]                              # H-step window = H+1 points, seg[0]=center
                ok, *_ = VP.certify_window(seg, obs, rr, g, R=2.5, n_theta=180)
                if not ok:
                    chosen = (p, t); break
            if chosen:
                break
        ax = axes[0][ci]
        if chosen is None:
            ax.text(.5, .5, f"γ={g}\nno failing window found", ha="center", va="center"); ax.axis("off"); continue
        p, t = chosen
        draw_verifier_window(ax, p[t:t + H + 1], obs, rr, g, trail=p[:t + 1])
    fig.suptitle("FM multi-step-safety FAILURE snapshot — green = verifier safe set; ✕ = step below barrier "
                 "H_P≥(1−γ)^t, ◯ = collision, ◇ = tightest step (infeasible face). All windows: cert=FAIL.",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def verifier_movie(policy, env, gammas, out, deploy_fn, H=10, fps=10):
    """Animate the FM rollout with the moving verifier polytope + level sets, 3 γ side-by-side."""
    obs = env.obstacles.detach().cpu().numpy(); rr = float(env.r_robot)
    rolls = {g: deploy_fn(policy, env, g) for g in gammas}
    nF = max(len(p) for p in rolls.values())
    fig, axes = plt.subplots(1, len(gammas), figsize=(4.6 * len(gammas), 4.9), squeeze=False)

    def frame(f):
        for ci, g in enumerate(gammas):
            ax = axes[0][ci]; ax.clear()
            p = rolls[g]; t = min(f, len(p) - H - 1); t = max(t, 0)
            draw_verifier_window(ax, p[t:t + H + 1], obs, rr, g, trail=p[:t + 1])
        fig.suptitle("Verifier polytope + DTCBF level sets moving with the FM rollout  "
                     "(green bands = H_P≥(1−γ)^t allowed region, tighter for smaller γ; dashed = reachable tube)",
                     fontsize=11)
        return []

    anim = FuncAnimation(fig, frame, frames=range(0, nF - H, 2), interval=120)
    anim.save(out, writer=PillowWriter(fps=fps), dpi=85)
    try:
        anim.save(out[:-4] + ".mp4", writer=FFMpegWriter(fps=max(fps, 12), bitrate=2600), dpi=110)
    except Exception as e:
        print(f"[mp4] skip ({e})")
    plt.close(fig)


if __name__ == "__main__":
    import grid_scene as GS, grid_policy as GP, grid_rollout as GR
    dev = "cuda"
    POLICY = os.environ.get("POLICY", "pretrained.pt"); SUF = os.environ.get("SUFFIX", "")
    env = GS.make_grid(); pol, _ = GP.load_policy(POLICY, device=dev)
    FIG = os.path.join(os.path.dirname(__file__), "figures")

    def deploy(policy, env, g):
        return GR.fm_deploy(policy, env, g, T=250, nfe=10, device=dev)["path"]

    snapshot_failures(pol, env, [0.1, 0.5, 1.0], os.path.join(FIG, f"verifier_failure{SUF}.png"), deploy)
    print(f"saved verifier_failure{SUF}.png")
    verifier_movie(pol, env, [0.1, 0.5, 1.0], os.path.join(FIG, f"verifier_rollout{SUF}.gif"), deploy)
    print(f"saved verifier_rollout{SUF}.{{gif,mp4}}")
