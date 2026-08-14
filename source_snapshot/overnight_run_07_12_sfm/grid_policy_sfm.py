"""GridSFMFlowPolicy with the frozen Hp10 visual input."""
from __future__ import annotations

import torch
import torch.nn as nn

import _paths  # noqa: F401
import grid_policy2 as GP2
from flow_policy import fourier_time


class ResidualTrunk(nn.Module):
    def __init__(self, in_dim, width, n_blocks=2, dropout=0.05):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(in_dim, width), nn.SiLU())
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(),
                          nn.Linear(width, width), nn.Dropout(dropout))
            for _ in range(n_blocks)
        ])

    def forward(self, value, dropout_scale=None):
        hidden = self.inp(value)
        if dropout_scale is not None:
            expected = (len(hidden), len(self.blocks), hidden.shape[1])
            if tuple(dropout_scale.shape) != expected:
                raise ValueError(f"dropout_scale {tuple(dropout_scale.shape)} != {expected}")
        for index, block in enumerate(self.blocks):
            residual = block[3](block[2](block[1](block[0](hidden))))
            residual = (block[4](residual) if dropout_scale is None
                        else residual * dropout_scale[:, index])
            hidden = hidden + residual
        return hidden


class GridSFMFlowPolicy(GP2.GridGRUFlowPolicy2):
    def __init__(self, n_res_blocks=2, res_dropout=0.05, **kwargs):
        super().__init__(**kwargs)
        self.trunk = ResidualTrunk(
            self.d + self.ctx_dim + self.t_dim, self.width, n_res_blocks, res_dropout
        )
        self.n_res_blocks = int(n_res_blocks)
        self.res_dropout = float(res_dropout)

    def config(self):
        return dict(
            arch="v2-sfm-hp10-residual", H_pred=self.H_pred, grid_shape=self.grid_shape,
            K_hist=self.K_hist, gru_dim=self.gru_dim, width=self.width, depth=2,
            u_max=self.u_max, use_gru=self.use_gru, encode_low=self.encode_low,
            use_grid=self.use_grid, raw_hist=self.raw_hist, raw_hist_k=self.raw_hist_k,
            dropout=self.dropout, enc_hist=self.enc_hist, n_res_blocks=self.n_res_blocks,
            res_dropout=self.res_dropout,
        )

    def cfm_loss_designed(self, controls, context, *, x0, tau, dropout_scale, weights=None):
        """CFM loss under an explicit per-query random design.

        The caller owns ``x0``, ``tau``, and every residual-dropout mask, so
        one verifier query receives the same stochastic design regardless of
        replay batching or the requested number of optimizer steps.
        """
        batch = int(controls.shape[0])
        x1 = (controls / self.u_max).reshape(batch, self.d)
        x0 = torch.as_tensor(x0, dtype=x1.dtype, device=x1.device)
        tau = torch.as_tensor(tau, dtype=x1.dtype, device=x1.device).reshape(batch)
        dropout_scale = torch.as_tensor(
            dropout_scale, dtype=x1.dtype, device=x1.device,
        )
        if tuple(x0.shape) != tuple(x1.shape):
            raise ValueError(f"designed x0 {tuple(x0.shape)} != {tuple(x1.shape)}")
        x_tau = (1.0 - tau)[:, None] * x0 + tau[:, None] * x1
        features = self.trunk(
            torch.cat([
                x_tau, self._expand_ctx(context, batch), fourier_time(tau, self.t_dim),
            ], dim=1),
            dropout_scale=dropout_scale,
        )
        prediction = self.head(features)
        per_sample = ((prediction - (x1 - x0)) ** 2).mean(dim=1)
        if weights is not None:
            per_sample = per_sample * weights
        return per_sample.mean()


def build_sfm_policy(width=256, u_max=2.0, n_res_blocks=2, res_dropout=0.05,
                     grid_shape=(10, 16, 12), device="cpu"):
    if tuple(grid_shape) != (10, 16, 12):
        raise ValueError(f"Hp10 policy requires grid_shape=(10,16,12), got {grid_shape}")
    return GridSFMFlowPolicy(
        width=width, depth=2, u_max=u_max, use_gru=True, encode_low=True, use_grid=True,
        grid_shape=grid_shape, n_res_blocks=n_res_blocks, res_dropout=res_dropout,
    ).to(device)


def save_sfm_policy(policy, path, extra=None):
    payload = {"state_dict": policy.state_dict(), "config": policy.config()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_sfm_policy(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    if config.get("arch") != "v2-sfm-hp10-residual" or tuple(config["grid_shape"]) != (10, 16, 12):
        raise ValueError("checkpoint is not a strict SFM Hp10 checkpoint")
    policy = GridSFMFlowPolicy(
        H_pred=config["H_pred"], grid_shape=tuple(config["grid_shape"]), K_hist=config["K_hist"],
        gru_dim=config["gru_dim"], width=config["width"], depth=config["depth"],
        u_max=config["u_max"], n_res_blocks=config["n_res_blocks"],
        res_dropout=config["res_dropout"],
    )
    policy.load_state_dict(checkpoint["state_dict"], strict=True)
    return policy.to(device).eval(), checkpoint
