from __future__ import annotations

import torch
import torch.nn as nn

from cfm_mppi.models.context_encoder import ContextEncoder, context_kwargs_from_batch
from cfm_mppi.models.model_configs import MODEL_CONFIGS
from cfm_mppi.models.transformer import (
    ConditionalTransformerEncoderLayer,
    PositionalEncoding,
    SinusoidalTimeEmbedding,
)


class ContextualTransformerModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 500,
        state_dim: int = 4,
        action_dim: int = 2,
        obstacle_dim: int = 4,
        history_len: int = 10,
    ):
        super().__init__()
        self.time_embed_dim = d_model
        self.input_linear = nn.Linear(in_channels, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.time_embed = SinusoidalTimeEmbedding(d_model)
        self.context_encoder = ContextEncoder(
            d_model=d_model,
            state_dim=state_dim,
            action_dim=action_dim,
            obstacle_dim=obstacle_dim,
            history_len=history_len,
            dropout=dropout,
        )
        self.layers = nn.ModuleList(
            [
                ConditionalTransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation="relu",
                    time_embed_dim=d_model,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_linear = nn.Linear(d_model, out_channels)

    @classmethod
    def from_mizuta_defaults(cls, **overrides):
        cfg = dict(MODEL_CONFIGS["transformer"])
        cfg.update(overrides)
        return cls(**cfg)

    def forward(self, x, timesteps, context_tokens=None, **context_kwargs):
        """
        x: [B, 2, T] or [B, T, 2].
        timesteps: [B].
        returns [B, 2, T].
        """
        if x.ndim != 3:
            raise ValueError(f"x must be rank-3, got {tuple(x.shape)}")
        if x.shape[1] == 2:
            x_bt = x.permute(0, 2, 1)
            return_channels_first = True
        else:
            x_bt = x
            return_channels_first = False
        h = self.input_linear(x_bt)
        h = self.pos_encoding(h)
        if context_tokens is None:
            context_tokens = self.context_encoder(**context_kwargs)
        h = torch.cat([context_tokens, h], dim=1)
        t_emb = self.time_embed(timesteps.view(-1))
        for layer in self.layers:
            h = layer(h, t_emb=t_emb)
        h = h[:, context_tokens.shape[1] :, :]
        out = self.output_linear(h)
        if return_channels_first:
            out = out.permute(0, 2, 1)
        return out

    def forward_batch(self, x, timesteps, batch):
        return self.forward(x, timesteps, **context_kwargs_from_batch(batch))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
