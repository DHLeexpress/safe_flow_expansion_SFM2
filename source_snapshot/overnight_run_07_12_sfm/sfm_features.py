"""Compatibility helpers; the frozen B1 policy itself uses sfm_hp_history.HpHistory."""
from __future__ import annotations

import torch


def policy_grid(policy, current, previous=None):
    """Match the checkpoint's declared channel count without changing the current-grid semantics."""
    channels = int(policy.grid_shape[0])
    if channels == 3:
        return current
    if channels == 6:
        prev = current if previous is None else previous
        dim = 0 if current.ndim == 3 else 1
        return torch.cat([current, prev], dim=dim)
    if channels == 10:
        raise ValueError("Hp10 callers must maintain sfm_hp_history.HpHistory, not a one-frame helper")
    raise ValueError(f"unsupported SFM policy grid channels: {channels}")


def add_previous_grid(grid, episode, step):
    """Concatenate current and previous same-trajectory grids for an ordered window dataset."""
    if grid.ndim != 4 or grid.shape[1] != 3:
        raise ValueError(f"expected [N,3,H,W], got {tuple(grid.shape)}")
    episode = episode.reshape(-1); step = step.reshape(-1)
    if len(grid) != len(episode) or len(grid) != len(step):
        raise ValueError("grid/episode/step lengths differ")
    prev = grid.clone()
    if len(grid) > 1:
        contiguous = (episode[1:] == episode[:-1]) & (step[1:] == step[:-1] + 1)
        idx = torch.nonzero(contiguous, as_tuple=False).flatten() + 1
        prev[idx] = grid[idx - 1]
    return torch.cat([grid, prev], dim=1)
