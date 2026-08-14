"""Leak-free Hp10 construction shared by offline and online SFM callers."""
from __future__ import annotations

from collections import deque
import numpy as np
import torch

HP_CHANNEL = 2
HP_HISTORY = 10
HP_SHAPE = (16, 12)


def _hp(frame):
    value = torch.as_tensor(frame)
    if value.shape == HP_SHAPE:
        return value
    if value.shape[-3:] != (3, *HP_SHAPE):
        raise ValueError(f"expected [3,16,12] grid or [16,12] Hp, got {tuple(value.shape)}")
    return value[..., HP_CHANNEL, :, :]


class HpHistory:
    """Online newest-to-oldest deque; reset padding repeats the first current Hp."""

    def __init__(self, length=HP_HISTORY):
        if int(length) != HP_HISTORY:
            raise ValueError("the frozen study requires exactly Hp10")
        self._frames = deque(maxlen=HP_HISTORY)

    def reset(self):
        self._frames.clear()

    def append(self, grid_or_hp):
        frame = _hp(grid_or_hp).detach().clone()
        if tuple(frame.shape) != HP_SHAPE:
            raise ValueError(f"expected one Hp frame, got {tuple(frame.shape)}")
        self._frames.append(frame)
        return self.tensor()

    def tensor(self):
        if not self._frames:
            raise RuntimeError("append the current frame before requesting Hp10")
        oldest = self._frames[0]
        newest_first = list(reversed(self._frames))
        newest_first.extend([oldest] * (HP_HISTORY - len(newest_first)))
        return torch.stack(newest_first, dim=0)


def build_hp10(grids, episodes, steps):
    """Build [N,10,16,12] from same-episode current/past frames only.

    Input order is irrelevant. Missing pre-start history repeats the earliest frame;
    missing interior steps are rejected rather than backfilled ambiguously.
    """
    grids = torch.as_tensor(grids)
    episodes = torch.as_tensor(episodes, dtype=torch.int64).reshape(-1)
    steps = torch.as_tensor(steps, dtype=torch.int64).reshape(-1)
    if grids.ndim != 4 or tuple(grids.shape[1:]) != (3, *HP_SHAPE):
        raise ValueError(f"expected [N,3,16,12], got {tuple(grids.shape)}")
    if not (len(grids) == len(episodes) == len(steps)):
        raise ValueError("grid/episode/step lengths differ")
    lookup = {}
    earliest = {}
    for i, (episode, step) in enumerate(zip(episodes.tolist(), steps.tolist())):
        key = (int(episode), int(step))
        if key in lookup:
            raise ValueError(f"duplicate episode/step record: {key}")
        lookup[key] = i
        earliest[int(episode)] = min(int(step), earliest.get(int(episode), int(step)))
    out = []
    for episode, step in zip(episodes.tolist(), steps.tolist()):
        first = earliest[int(episode)]
        indices = []
        for lag in range(HP_HISTORY):
            wanted = max(first, int(step) - lag)
            key = (int(episode), wanted)
            if key not in lookup:
                raise ValueError(f"non-contiguous episode {episode}: missing past step {wanted}")
            indices.append(lookup[key])
        out.append(grids[indices, HP_CHANNEL])
    return torch.stack(out, dim=0)


def online_hp10_sequence(grids):
    history = HpHistory()
    values = []
    for grid in grids:
        values.append(history.append(grid))
    return torch.stack(values)


def hp10_numpy(value):
    array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    if array.shape != (HP_HISTORY, *HP_SHAPE):
        raise ValueError(f"expected [10,16,12], got {array.shape}")
    return array.astype(np.float32, copy=False)
