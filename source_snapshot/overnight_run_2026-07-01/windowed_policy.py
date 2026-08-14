"""Windowed trajectory-level CFM policy (SOTA diffusion/flow-matching policy style).

Extends `overnight_run_today/src/flow_policy.FlowPolicy` (which concatenates an arbitrary context vector
into the trunk). Context = concat(low_token, grid_token):
  * enc_low : MLP over low_dim[7] (goal-aligned state incl. γ)          → low_token
  * enc_grid: flatten+MLP over the polar polytope-occupancy grid [3,16,12] → grid_token   (CNN later)
Target = the MPPI planned window in the goal-aligned LOCAL frame `U_local ∈ R^{H_pred×2}` (CFM: noise→U_local).
Inference rotates the sampled U_local back to world (local_frame.to_world) before rollout/verify.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import _paths  # noqa: F401
from flow_policy import FlowPolicy
from local_frame import goal_frame, to_world
from polar_grid import polar_grid
from local_frame import low_dim_features
from di_grid_viz import di_step


class GridLowFlowPolicy(FlowPolicy):
    def __init__(self, H_pred=10, grid_shape=(3, 16, 12), low_dim=7,
                 grid_token=96, low_token=48, width=256, depth=3, u_max=2.0,
                 grid_hidden=256, low_hidden=64):
        ctx_dim = grid_token + low_token
        super().__init__(T=H_pred, ctx_dim=ctx_dim, width=width, depth=depth, u_max=u_max)
        g_in = int(np.prod(grid_shape))
        self.enc_grid = nn.Sequential(nn.Flatten(), nn.Linear(g_in, grid_hidden), nn.SiLU(),
                                      nn.Linear(grid_hidden, grid_token), nn.SiLU())
        self.enc_low = nn.Sequential(nn.Linear(low_dim, low_hidden), nn.SiLU(),
                                     nn.Linear(low_hidden, low_token), nn.SiLU())
        # auxiliary SAFETY-ENCODING head: reconstruct the polar safety grid from grid_token, so the
        # encoder is forced to encode the safety function (occupancy / polytope-mask / H_P), not just
        # whatever the CFM loss happens to need. Gives a monitorable "polytope→context" loss.
        self.safety_decoder = nn.Sequential(nn.Linear(grid_token, grid_hidden), nn.SiLU(),
                                            nn.Linear(grid_hidden, g_in))
        self.grid_shape = grid_shape
        self.g_in = g_in
        self.H_pred = H_pred

    def grid_token(self, grid):
        if grid.dim() == 3:
            grid = grid[None]
        return self.enc_grid(grid.float())

    def ctx_from(self, grid, low_dim):
        """grid [B,3,Nθ,Nr], low_dim [B,7] -> ctx [B, ctx_dim] (encoders trained jointly)."""
        if grid.dim() == 3:
            grid = grid[None]
        if low_dim.dim() == 1:
            low_dim = low_dim[None]
        return torch.cat([self.enc_low(low_dim.float()), self.enc_grid(grid.float())], dim=1)

    def aux_safety_loss(self, grid):
        """Reconstruction MSE of the safety grid from grid_token (the 'polytope→context' loss)."""
        gt = self.grid_token(grid)
        recon = self.safety_decoder(gt)
        return F.mse_loss(recon, grid.float().reshape(grid.shape[0], -1))

    # -------------------------------------------------------------- closed-loop deployment
    @torch.no_grad()
    def sample_world_window(self, state, goal, gamma, obstacles, a_prev=None, prev_valid=False,
                            r_robot=0.0, n=1, temp=1.0, nfe=12, device="cpu"):
        """One conditioning state -> sampled U_local -> world control window(s) [n, H_pred, 2]."""
        pos = np.asarray(state, float)[:2]
        grid, _ = polar_grid(pos, goal, obstacles, r_robot=r_robot)
        low, _ = low_dim_features(state, goal, gamma, a_prev=a_prev, prev_valid=prev_valid)
        ctx = self.ctx_from(torch.tensor(grid, device=device), torch.tensor(low, device=device))
        U_local = self.sample(n, ctx.expand(n, -1), nfe=nfe, temp=temp)                 # [n,H_pred,2]
        e_g, e_lat, _ = goal_frame(pos, goal)
        U_world = to_world(U_local.cpu().numpy(), e_g, e_lat)                            # [n,H_pred,2]
        return U_world.astype(np.float32), U_local.cpu().numpy().astype(np.float32)


@torch.no_grad()
def fm_rollout(policy, env, gamma, n_traj=32, temp=1.0, H_exec=1, nfe=12, reach_thresh=0.4,
               device="cpu", record=True):
    """Closed-loop (receding-horizon) FM rollout of n_traj trajectories. Returns
    (paths[n,steps+1,2], (per-traj grids, lows, U_local windows)) — windows are the training positives."""
    goal = env.goal.detach().cpu().numpy()
    obs = env.obstacles.detach().cpu().numpy()
    r_robot = float(env.r_robot); T = int(env.T); dt = float(env.dt); umax = float(env.u_max)
    st = np.tile(env.x0.detach().cpu().numpy().astype(np.float32), (n_traj, 1))
    a_prev = np.zeros((n_traj, 2), np.float32); prev_valid = np.zeros(n_traj, bool)
    reached = np.zeros(n_traj, bool)
    paths = [st[:, :2].copy()]
    G = [[] for _ in range(n_traj)]; L = [[] for _ in range(n_traj)]; U = [[] for _ in range(n_traj)]
    t = 0
    while t < T:
        grids = np.stack([polar_grid(st[i, :2], goal, obs, r_robot=r_robot)[0] for i in range(n_traj)])
        lows = np.stack([low_dim_features(st[i], goal, gamma, a_prev[i], bool(prev_valid[i]))[0]
                         for i in range(n_traj)])
        ctx = policy.ctx_from(torch.tensor(grids, device=device), torch.tensor(lows, device=device))
        U_local = policy.sample(n_traj, ctx, nfe=nfe, temp=temp).cpu().numpy()
        for i in range(n_traj):
            if record:
                G[i].append(grids[i]); L[i].append(lows[i]); U[i].append(U_local[i].astype(np.float32))
            e_g, e_lat, _ = goal_frame(st[i, :2], goal)
            Uw = to_world(U_local[i], e_g, e_lat)
            for k in range(H_exec):
                if reached[i]:
                    break
                u = np.clip(Uw[min(k, len(Uw) - 1)], -umax, umax).astype(np.float32)
                st[i] = di_step(st[i], u, dt); a_prev[i] = u; prev_valid[i] = True
                if np.linalg.norm(st[i, :2] - goal) < reach_thresh:
                    reached[i] = True
        paths.append(st[:, :2].copy())
        t += H_exec
    return np.stack(paths, 1), (G, L, U)


def windows_of(states, controls, env, gamma, H_pred, device="cpu"):
    """Per-step (grid, low, U_local) along a rolled-out (states[T+1,4], controls[T,2]) trajectory."""
    goal = env.goal.detach().cpu().numpy(); obs = env.obstacles.detach().cpu().numpy()
    r_robot = float(env.r_robot)
    grids, lows, Us = [], [], []
    n = len(controls)
    for t in range(n):
        pos = states[t, :2]
        a_prev = controls[t - 1] if t > 0 else None
        grid, _ = polar_grid(pos, goal, obs, r_robot=r_robot)
        low, _ = low_dim_features(states[t], goal, gamma, a_prev=a_prev, prev_valid=(t > 0))
        e_g, e_lat, _ = goal_frame(pos, goal)
        win = np.asarray(controls[t:t + H_pred], float)
        if len(win) < H_pred:
            win = np.vstack([win, np.tile(win[-1:], (H_pred - len(win), 1))])
        from local_frame import to_local
        grids.append(grid); lows.append(low); Us.append(to_local(win, e_g, e_lat).astype(np.float32))
    return grids, lows, Us


if __name__ == "__main__":
    import config as C
    pol = GridLowFlowPolicy(H_pred=C.H_PRED)
    B = 8
    grid = torch.randn(B, 3, 16, 12)
    low = torch.randn(B, 7)
    U = torch.randn(B, C.H_PRED, 2)
    ctx = pol.ctx_from(grid, low)
    print("ctx", tuple(ctx.shape), "ctx_dim", pol.ctx_dim)
    loss = pol.cfm_loss(U, ctx)
    print("cfm_loss", float(loss))
    s = pol.sample(4, ctx[:4])
    print("sample", tuple(s.shape))
    env = C.make_scene()
    Uw, Ul = pol.sample_world_window([0, 0, 0, 0], env.goal.numpy(), 0.5, env.obstacles.numpy(), n=3)
    print("world window", Uw.shape, "local", Ul.shape)
    print("OK")
