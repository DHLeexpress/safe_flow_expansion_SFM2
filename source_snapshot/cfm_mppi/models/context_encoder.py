from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


def fit_last_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    if x.shape[-1] == dim:
        return x
    if x.shape[-1] > dim:
        return x[..., :dim]
    pad_shape = (*x.shape[:-1], dim - x.shape[-1])
    return torch.cat([x, torch.zeros(pad_shape, dtype=x.dtype, device=x.device)], dim=-1)


class ContextEncoder(nn.Module):
    """
    Encodes nearest-obstacle and ego/action histories into transformer condition tokens.

    Default feature assumptions mirror the local safeGPC v4.2 data:
    ego state [px, py, vx, vy], action [ax, ay], obstacle history
    [relative_px, relative_py, relative_vx, relative_vy].
    """

    def __init__(
        self,
        d_model: int = 256,
        state_dim: int = 4,
        action_dim: int = 2,
        obstacle_dim: int = 4,
        history_len: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.obstacle_dim = int(obstacle_dim)
        self.history_len = int(history_len)

        self.start_proj = nn.Linear(2, d_model)
        self.goal_proj = nn.Linear(2, d_model)
        self.ego_current_proj = nn.Linear(state_dim, d_model)
        self.ego_hist_proj = nn.Linear(state_dim, d_model)
        self.action_hist_proj = nn.Linear(action_dim, d_model)
        self.obs_hist_proj = nn.Linear(obstacle_dim, d_model)
        self.scalar_proj = nn.Linear(2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _mean_history(self, x: Optional[torch.Tensor], dim: int, batch: int, device: torch.device) -> torch.Tensor:
        if x is None:
            return torch.zeros(batch, dim, device=device)
        x = fit_last_dim(x.float(), dim)
        if x.ndim == 2:
            return x
        valid = torch.isfinite(x).all(dim=-1, keepdim=True)
        x = torch.where(valid, x, torch.zeros_like(x))
        denom = valid.float().sum(dim=1).clamp_min(1.0)
        return x.sum(dim=1) / denom

    def forward(
        self,
        *,
        start: torch.Tensor,
        goal: torch.Tensor,
        ego_current: Optional[torch.Tensor] = None,
        ego_history: Optional[torch.Tensor] = None,
        action_history: Optional[torch.Tensor] = None,
        nearest_obstacle_history: Optional[torch.Tensor] = None,
        gamma: Optional[torch.Tensor] = None,
        safety_margin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch = start.shape[0]
        device = start.device
        start = fit_last_dim(start.float(), 2)
        goal = fit_last_dim(goal.float(), 2)
        if ego_current is None:
            if ego_history is not None:
                ego_current = ego_history[:, -1]
            else:
                ego_current = torch.zeros(batch, self.state_dim, device=device)
        ego_current = fit_last_dim(ego_current.float(), self.state_dim)
        ego_hist = self._mean_history(ego_history, self.state_dim, batch, device)
        action_hist = self._mean_history(action_history, self.action_dim, batch, device)
        obstacle_hist = self._mean_history(nearest_obstacle_history, self.obstacle_dim, batch, device)
        if gamma is None:
            gamma = torch.zeros(batch, device=device)
        gamma = torch.nan_to_num(gamma.float().view(batch), nan=0.0)
        if safety_margin is None:
            safety_margin = torch.zeros(batch, device=device)
        safety_margin = torch.nan_to_num(safety_margin.float().view(batch), nan=0.0)
        scalars = torch.stack([gamma, safety_margin], dim=-1)

        tokens = torch.stack(
            [
                self.start_proj(start),
                self.goal_proj(goal),
                self.ego_current_proj(ego_current),
                self.ego_hist_proj(ego_hist),
                self.action_hist_proj(action_hist),
                self.obs_hist_proj(obstacle_hist),
                self.scalar_proj(scalars),
            ],
            dim=1,
        )
        return self.dropout(self.norm(tokens))


def context_kwargs_from_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    states = batch["states"]
    return {
        "start": batch["start"],
        "goal": batch["goal"],
        "ego_current": states[:, 0],
        "ego_history": batch.get("ego_history"),
        "action_history": batch.get("action_history"),
        "nearest_obstacle_history": batch.get("nearest_obstacle_history"),
        "gamma": batch.get("gamma"),
        "safety_margin": batch.get("safety_margin"),
    }
