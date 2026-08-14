"""Inference for the reward-tilted flow proposal: build context, sample q_θ(U|o,γ),
and run the samples through the safety certificate (HOCBF rejection + averaging +
output projection). Safety stays proposal-agnostic (Props 3-4); the learned tilt
provides performance/multimodality."""
from __future__ import annotations
import numpy as np
import torch
from cfm_mppi.models.tilted_flow import TiltedFlowProposal

KOBS = 6
ODIM = 4 + 4 * KOBS


def load_tilted_flow(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = TiltedFlowProposal(horizon=ck["horizon"], cond_dim=ck["cond_dim"], hidden=ck["hidden"]).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    return {"model": model, "mu": ck["ctx_mu"].to(device), "sd": ck["ctx_sd"].to(device), "horizon": ck["horizon"]}


def tilted_features(state, goal, obstacles, obstacle_velocities):
    """Translation-invariant context (must match generate_tilted_dataset)."""
    st = np.asarray(state, dtype=np.float32); goal = np.asarray(goal, dtype=np.float32)
    obs = np.asarray(obstacles, dtype=np.float32)
    vel = np.asarray(obstacle_velocities, dtype=np.float32) if obstacle_velocities is not None else np.zeros((0, 2), np.float32)
    p = st[:2]; v = st[2:4]; gd = goal[:2] - p
    feats = [v[0], v[1], gd[0], gd[1]]
    if obs.shape[0]:
        d = np.linalg.norm(obs[:, :2] - p[None, :], axis=1) - obs[:, 2]
        for j in np.argsort(d)[:KOBS]:
            rel = obs[j, :2] - p
            rv = vel[j] if vel.shape[0] > j else np.zeros(2)
            feats += [rel[0], rel[1], rv[0], rv[1]]
    while len(feats) < ODIM:
        feats += [8.0, 8.0, 0.0, 0.0]
    return np.asarray(feats[:ODIM], dtype=np.float32)


@torch.no_grad()
def tilted_sample(tilted, state, goal, obstacles, obstacle_velocities, gamma, n, nfe=8, device=None):
    """Sample n control sequences [n,H,2] from q_θ(U|o,γ)."""
    device = device or tilted["mu"].device
    o = tilted_features(state, goal, obstacles, obstacle_velocities)
    o_n = (torch.tensor(o, device=device) - tilted["mu"]) / tilted["sd"]
    cond = torch.cat([o_n, torch.tensor([float(gamma)], device=device)])
    return tilted["model"].sample(cond, n, nfe=nfe)
