"""Reward-tilted conditional flow proposal q_θ(U | o, γ).

A conditional flow-matching (rectified-flow / OT path) generator over the control
sequence U ∈ R^{H×2}, conditioned on a translation-invariant context o (robot
velocity, goal-relative, nearest-K pedestrian relative pos/vel) and the safety
knob γ. Trained by the energy/reward-weighted CFM loss (EFM, arXiv:2503.04975):
sampling the training control ∝ MPPI weight exp(-S/λ) makes the terminal marginal
equal the reward tilt p(U|o,γ) ∝ 1[U∈F] exp(-S/λ). Few-NFE Euler ODE at inference.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


def _sinusoidal(t: torch.Tensor, dim: int = 64) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    ang = t.view(-1, 1) * freqs.view(1, -1)
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class TiltedFlowProposal(nn.Module):
    def __init__(self, horizon: int = 25, cond_dim: int = 29, hidden: int = 384, depth: int = 4):
        super().__init__()
        self.H = horizon
        self.udim = horizon * 2
        self.cond_dim = cond_dim
        self.cond_mlp = nn.Sequential(nn.Linear(cond_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.t_mlp = nn.Sequential(nn.Linear(64, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.inp = nn.Linear(self.udim, hidden)
        blocks = []
        for _ in range(depth):
            blocks.append(nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)))
        self.blocks = nn.ModuleList(blocks)
        self.out = nn.Linear(hidden, self.udim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x:[B,udim] noised control, t:[B] in [0,1], cond:[B,cond_dim] -> v:[B,udim]."""
        h = self.inp(x) + self.t_mlp(_sinusoidal(t)) + self.cond_mlp(cond)
        for blk in self.blocks:
            h = h + blk(h)
        return self.out(h)

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, n: int, nfe: int = 8, u_clip: float = 2.0) -> torch.Tensor:
        """Sample n control sequences for a single context cond:[cond_dim].
        Euler-integrate the flow ODE from noise. Returns [n, H, 2]."""
        dev = cond.device
        c = cond.view(1, -1).expand(n, -1)
        x = torch.randn(n, self.udim, device=dev)
        for i in range(nfe):
            t = torch.full((n,), i / nfe, device=dev)
            x = x + (1.0 / nfe) * self.forward(x, t, c)
        return x.view(n, self.H, 2).clamp(-u_clip, u_clip)
