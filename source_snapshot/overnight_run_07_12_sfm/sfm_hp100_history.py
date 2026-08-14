"""Leak-free ten-frame history for high-resolution SFM :math:`H_P` rasters."""
from __future__ import annotations

from collections import deque

import numpy as np
import torch

from sfm_hp100_features import HP100_SHAPE

HP_HISTORY = 10


def _frame(value):
    frame = torch.as_tensor(value)
    if tuple(frame.shape) != HP100_SHAPE:
        raise ValueError(f"expected one Hp100 frame {HP100_SHAPE}, got {tuple(frame.shape)}")
    return frame


class Hp100History:
    """Online newest-to-oldest history with first-observation reset padding."""

    def __init__(self, length=HP_HISTORY):
        if int(length) != HP_HISTORY:
            raise ValueError("the HP100 policy requires exactly ten frames")
        self._frames = deque(maxlen=HP_HISTORY)

    def reset(self):
        self._frames.clear()

    def append(self, hp_frame):
        self._frames.append(_frame(hp_frame).detach().clone())
        return self.tensor()

    def tensor(self):
        if not self._frames:
            raise RuntimeError("append the current frame before requesting HP100 history")
        oldest = self._frames[0]
        newest_first = list(reversed(self._frames))
        newest_first.extend([oldest] * (HP_HISTORY - len(newest_first)))
        return torch.stack(newest_first, dim=0)


def build_hp100(frames, episodes, steps):
    """Build ``[N,10,32,100]`` using only same-episode current/past frames.

    Input order is irrelevant.  Missing pre-start history repeats the earliest
    frame in that episode, while an interior gap fails closed.
    """
    frames = torch.as_tensor(frames)
    episodes = torch.as_tensor(episodes, dtype=torch.int64).reshape(-1)
    steps = torch.as_tensor(steps, dtype=torch.int64).reshape(-1)
    if frames.ndim != 3 or tuple(frames.shape[1:]) != HP100_SHAPE:
        raise ValueError(f"expected [N,32,100], got {tuple(frames.shape)}")
    if not (len(frames) == len(episodes) == len(steps)):
        raise ValueError("frames/episodes/steps lengths differ")

    lookup = {}
    earliest = {}
    for index, (episode, step) in enumerate(zip(episodes.tolist(), steps.tolist())):
        key = (int(episode), int(step))
        if key in lookup:
            raise ValueError(f"duplicate episode/step record: {key}")
        lookup[key] = index
        earliest[int(episode)] = min(int(step), earliest.get(int(episode), int(step)))

    rows = []
    for episode, step in zip(episodes.tolist(), steps.tolist()):
        first = earliest[int(episode)]
        indices = []
        for lag in range(HP_HISTORY):
            wanted = max(first, int(step) - lag)
            key = (int(episode), wanted)
            if key not in lookup:
                raise ValueError(f"non-contiguous episode {episode}: missing past step {wanted}")
            indices.append(lookup[key])
        rows.append(frames[indices])
    return torch.stack(rows, dim=0)


def hp100_numpy(value):
    array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    expected = (HP_HISTORY, *HP100_SHAPE)
    if array.shape != expected:
        raise ValueError(f"expected {expected}, got {array.shape}")
    return array.astype(np.float32, copy=False)

