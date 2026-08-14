"""Diagnose the 'constant offset between robot and obstacles' in the SafeMPPI polytope viz.
Render the exact nominal polytope H_P at 3 positions, from TRUE obstacles (r=0.2) vs the PLANNER's INFLATED
obstacles (r=0.2+PLAN_MARGIN), and measure the robot->polytope-boundary clearance."""
import _paths, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import grid_scene as GS
from polar_grid import polytope_HP

env = GS.make_grid()
obs_true = env.obstacles.numpy()
obs_infl = GS.planner_obstacles(env).numpy()          # r_robot(0)+PLAN_MARGIN(0.1) => r=0.3
positions = [(0.0, 0.0), (1.5, 0.5), (2.5, 2.5)]
gx = np.linspace(-0.7, 5.7, 260); gy = np.linspace(-0.7, 5.7, 260); GX, GY = np.meshgrid(gx, gy)
flat = np.stack([GX.ravel(), GY.ravel()], 1)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for row, (obs, tag) in enumerate([(obs_true, "TRUE obstacles r=0.2"),
                                  (obs_infl, "PLANNER obstacles r=0.3 (inflated by PLAN_MARGIN=0.1)")]):
    for col, c in enumerate(positions):
        ax = axes[row][col]
        HP, _ = polytope_HP(np.array(c), obs, sensing=2.0, n_base=16)
        H = HP(flat).reshape(GX.shape)
        ax.contourf(GX, GY, H, levels=[0, 1e-6, 1], colors=["#ffffff", "#cfe0f5"], alpha=0.6)
        ax.contour(GX, GY, H, levels=[0.0], colors="#08306b", linewidths=1.6)
        for k in range(6):
            ax.axvline(k, color="#eee", lw=.6); ax.axhline(k, color="#eee", lw=.6)
        ax.add_patch(Rectangle((0, 0), 5, 5, fill=False, edgecolor="#555", lw=1.4))
        for j, (ox, oy, r) in enumerate(obs_true):                    # always draw TRUE obstacles
            ax.add_patch(Circle((ox, oy), r, facecolor="#b8b8b8" if j >= 16 else "#c8a2c8",
                                edgecolor="#777", lw=.5, alpha=.8))
        ax.scatter([c[0]], [c[1]], s=70, marker="o", c="#00a000", edgecolor="k", zorder=8)
        # clearance from robot center to the H_P=0 boundary (min over the disk where H_P just crosses 0)
        d_edge = (np.linalg.norm(flat - np.array(c), axis=1))
        onb = np.abs(HP(flat)) < 0.02
        clr = d_edge[onb].min() if onb.any() else np.nan
        # nearest true-obstacle surface distance
        do = (np.linalg.norm(np.array(c)[None] - obs_true[:, :2], axis=1) - obs_true[:, 2]).min()
        ax.set_title(f"c={c}  |  nearest true-obs surface {do:+.2f} m", fontsize=10)
        ax.set_xlim(-0.7, 5.7); ax.set_ylim(-0.7, 5.7); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    axes[row][0].set_ylabel(tag, fontsize=11)
fig.suptitle("Nominal SafeMPPI polytope (blue = safe set H_P≥0) centered at the robot (green) — "
             "top: true obstacles, bottom: inflated (planner)", fontsize=13)
fig.tight_layout(); fig.savefig("figures/diag_offset.png", dpi=130)
print("saved figures/diag_offset.png")
# quantify the corner leak at (0,0)
HP0, _ = polytope_HP(np.array([0.0, 0.0]), obs_true, sensing=2.0, n_base=16)
print("H_P at (0,0)=%.2f (center, expect 1); at (-0.4,-0.4)=%.2f (out-of-grid corner; >0 => polytope LEAKS out)"
      % (HP0(np.array([[0.0, 0.0]]))[0], HP0(np.array([[-0.4, -0.4]]))[0]))
