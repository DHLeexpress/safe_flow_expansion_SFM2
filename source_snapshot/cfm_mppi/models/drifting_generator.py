from __future__ import annotations

import torch
import torch.nn as nn

from cfm_mppi.models.contextual_transformer import ContextualTransformerModel
from cfm_mppi.models.context_encoder import context_kwargs_from_batch


class DriftingGenerator(nn.Module):
    """
    One-step PyTorch generator baseline using the same transformer depth defaults
    as ContextualTransformerModel. Inference is a single forward pass.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.backbone = ContextualTransformerModel.from_mizuta_defaults(**kwargs)

    @classmethod
    def from_mizuta_defaults(cls, **overrides):
        return cls(**overrides)

    def forward(self, noise, timesteps=None, **context_kwargs):
        if timesteps is None:
            timesteps = torch.ones(noise.shape[0], device=noise.device, dtype=noise.dtype)
        return self.backbone(noise, timesteps, **context_kwargs)

    def forward_batch(self, noise, batch, timesteps=None):
        return self.forward(noise, timesteps, **context_kwargs_from_batch(batch))
