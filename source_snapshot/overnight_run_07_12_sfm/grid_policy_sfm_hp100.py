"""Strict high-resolution HP100 SFM conditional-flow policy."""
from __future__ import annotations

import torch
import torch.nn as nn

import _paths  # noqa: F401
from flow_policy import FlowPolicy
from sfm_hp100_dynamics import U_MAX, V_MAX
from sfm_hp100_features import (
    HP100_SHAPE, K_HIST, POLYTOPE_N_BASE, PREDICT_GAIN, PREDICT_TAU, R_SENSE,
)

ARCH = "v3-sfm-hp100-residual"
H_PRED = 10
GRID_SHAPE = (10, *HP100_SHAPE)
GRU_DIM = 16
LOW_TOKEN = 48
VISUAL_TOKEN = 128
WIDTH = 256
N_RES_BLOCKS = 2
RES_DROPOUT = 0.05
ANGULAR_POOL = 4
T_DIM = 32


class ResidualTrunk(nn.Module):
    def __init__(self, in_dim, width=WIDTH, n_blocks=N_RES_BLOCKS, dropout=RES_DROPOUT):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(in_dim, width), nn.SiLU())
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(),
                nn.Linear(width, width), nn.Dropout(dropout),
            )
            for _ in range(n_blocks)
        ])

    def forward(self, value):
        hidden = self.inp(value)
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return hidden


class GridSFMHP100FlowPolicy(FlowPolicy):
    """HP100 encoder + GRU16/low48 conditioning + residual flow trunk."""

    def __init__(self):
        raw_low = 4 + GRU_DIM + 1
        ctx_dim = LOW_TOKEN + VISUAL_TOKEN
        super().__init__(
            T=H_PRED, ctx_dim=ctx_dim, width=WIDTH, depth=1,
            u_max=U_MAX, t_dim=T_DIM,
        )
        self.H_pred = H_PRED
        self.grid_shape = GRID_SHAPE
        self.K_hist = K_HIST
        self.gru_dim = GRU_DIM
        self.low_token = LOW_TOKEN
        self.visual_token = VISUAL_TOKEN
        self.n_res_blocks = N_RES_BLOCKS
        self.res_dropout = RES_DROPOUT

        self.gru = nn.GRU(input_size=2, hidden_size=GRU_DIM, num_layers=1, batch_first=True)
        self.enc_low = nn.Sequential(
            nn.Linear(raw_low, 64), nn.SiLU(),
            nn.Linear(64, LOW_TOKEN), nn.SiLU(),
        )
        self.grid_conv = nn.Sequential(
            nn.Conv2d(GRID_SHAPE[0], 16, 3, padding=1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.SiLU(),
        )
        # ``None`` preserves all 100 radial cells: only the angular axis is pooled.
        self.angular_pool = nn.AdaptiveAvgPool2d((ANGULAR_POOL, None))
        self.grid_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * ANGULAR_POOL * HP100_SHAPE[1], VISUAL_TOKEN),
            nn.SiLU(),
        )
        self.trunk = ResidualTrunk(
            self.d + self.ctx_dim + self.t_dim, WIDTH, N_RES_BLOCKS, RES_DROPOUT
        )

    def _batched_inputs(self, grid, low5, hist):
        if grid.dim() == 3:
            grid = grid.unsqueeze(0)
        if low5.dim() == 1:
            low5 = low5.unsqueeze(0)
        if hist.dim() == 2:
            hist = hist.unsqueeze(0)
        if tuple(grid.shape[1:]) != GRID_SHAPE:
            raise ValueError(f"expected grid [B,{GRID_SHAPE}], got {tuple(grid.shape)}")
        if low5.ndim != 2 or low5.shape[1] != 5:
            raise ValueError(f"expected low5 [B,5], got {tuple(low5.shape)}")
        if hist.ndim != 3 or tuple(hist.shape[1:]) != (K_HIST, 2):
            raise ValueError(f"expected history [B,{K_HIST},2], got {tuple(hist.shape)}")
        if not (grid.shape[0] == low5.shape[0] == hist.shape[0]):
            raise ValueError("grid/low5/history batch sizes differ")
        return grid, low5, hist

    def visual_features(self, grid):
        if grid.dim() == 3:
            grid = grid.unsqueeze(0)
        if tuple(grid.shape[1:]) != GRID_SHAPE:
            raise ValueError(f"expected grid [B,{GRID_SHAPE}], got {tuple(grid.shape)}")
        features = self.grid_conv(grid.float())
        features = self.angular_pool(features)
        if features.shape[-1] != HP100_SHAPE[1]:
            raise RuntimeError("HP100 visual encoder pooled the radial axis")
        return self.grid_projection(features)

    def ctx_from(self, grid, low5, hist):
        grid, low5, hist = self._batched_inputs(grid, low5, hist)
        _, hidden = self.gru(hist.float())
        raw_low = torch.cat([low5[:, :4], hidden[-1], low5[:, 4:5]], dim=1)
        return torch.cat([self.enc_low(raw_low), self.visual_features(grid)], dim=1)

    @torch.no_grad()
    def sample_window(self, grid, low5, hist, n=1, temp=1.0, nfe=12, churn=0.0):
        context = self.ctx_from(grid, low5, hist)
        if context.shape[0] == 1:
            context = context[0]
        return self.sample(n, context, nfe=nfe, temp=temp, churn=churn)

    def phi_s_at(self, controls, grid, low5, hist, s=0.9):
        context = self.ctx_from(grid, low5, hist)
        if context.shape[0] == 1:
            context = context[0]
        return self.phi_s(controls, context, s=s)

    def module_groups(self):
        return dict(
            E_grid=nn.ModuleList([self.grid_conv, self.angular_pool, self.grid_projection]),
            GRU=self.gru, E_low=self.enc_low, trunk=self.trunk, head=self.head,
        )

    def config(self):
        return dict(
            arch=ARCH, H_pred=H_PRED, grid_shape=GRID_SHAPE, K_hist=K_HIST,
            radial_bins=HP100_SHAPE[1], angular_bins=HP100_SHAPE[0],
            sensing_radius=R_SENSE, polytope_n_base=POLYTOPE_N_BASE,
            predict_gain=PREDICT_GAIN, predict_tau=PREDICT_TAU,
            conv_channels=(GRID_SHAPE[0], 16, 32), angular_pool=ANGULAR_POOL,
            radial_pool=None, visual_token=VISUAL_TOKEN, gru_dim=GRU_DIM,
            low_token=LOW_TOKEN, width=WIDTH, n_res_blocks=N_RES_BLOCKS,
            res_dropout=RES_DROPOUT, t_dim=T_DIM, u_max=float(U_MAX),
            v_max=float(V_MAX), output_dim=self.d,
        )


def build_sfm_hp100_policy(device="cpu"):
    return GridSFMHP100FlowPolicy().to(device)


def configure_head_only_expansion(policy):
    """Freeze every pre-head parameter for the declared HP100 expansion arm."""
    if not isinstance(policy, GridSFMHP100FlowPolicy):
        raise TypeError("head-only expansion requires GridSFMHP100FlowPolicy")
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    for parameter in policy.head.parameters():
        parameter.requires_grad_(True)
    assert_head_only_expansion(policy)
    return policy


def assert_head_only_expansion(policy):
    trainable = {name for name, value in policy.named_parameters() if value.requires_grad}
    expected = {f"head.{name}" for name, _ in policy.head.named_parameters()}
    if trainable != expected:
        raise RuntimeError(
            f"HP100 head-only contract violated: trainable={sorted(trainable)}, "
            f"expected={sorted(expected)}"
        )


def save_sfm_hp100_policy(policy, path, extra=None):
    if not isinstance(policy, GridSFMHP100FlowPolicy):
        raise TypeError("save_sfm_hp100_policy requires GridSFMHP100FlowPolicy")
    payload = {"state_dict": policy.state_dict(), "config": policy.config()}
    if extra:
        overlap = {"state_dict", "config"}.intersection(extra)
        if overlap:
            raise ValueError(f"extra metadata cannot replace strict checkpoint fields: {sorted(overlap)}")
        payload.update(extra)
    torch.save(payload, path)


def _canonical_config(config):
    config = dict(config)
    for key in ("grid_shape", "conv_channels"):
        if key in config:
            config[key] = tuple(config[key])
    return config


def load_sfm_hp100_policy(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "config" not in checkpoint or "state_dict" not in checkpoint:
        raise ValueError("HP100 checkpoint requires config and state_dict")
    policy = GridSFMHP100FlowPolicy()
    expected = policy.config()
    actual = _canonical_config(checkpoint["config"])
    if actual != expected:
        differing = sorted(
            key for key in set(expected) | set(actual) if expected.get(key) != actual.get(key)
        )
        raise ValueError(f"checkpoint is not a strict SFM HP100 checkpoint; differing={differing}")
    policy.load_state_dict(checkpoint["state_dict"], strict=True)
    return policy.to(device).eval(), checkpoint
