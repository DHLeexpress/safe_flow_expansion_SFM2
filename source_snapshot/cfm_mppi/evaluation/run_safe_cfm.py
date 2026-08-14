from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch

from cfm_mppi.models.context_encoder import context_kwargs_from_batch
from cfm_mppi.models.contextual_transformer import ContextualTransformerModel


def load_safe_cfm(checkpoint_path: str | Path, device: torch.device) -> ContextualTransformerModel:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = ContextualTransformerModel.from_mizuta_defaults(history_len=int(args.get("history_len", 10))).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def sample_safe_cfm_controls(
    model: ContextualTransformerModel,
    batch: Dict[str, torch.Tensor],
    *,
    horizon: int,
    nfe: int = 8,
    device: torch.device,
) -> torch.Tensor:
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    x = torch.randn(batch["start"].shape[0], 2, horizon, device=device)
    times = torch.linspace(0.0, 1.0, nfe + 1, device=device)
    context = context_kwargs_from_batch(batch)
    for i in range(nfe):
        t = torch.full((x.shape[0],), float(times[i]), device=device)
        dt = times[i + 1] - times[i]
        x = x + dt * model(x, t, **context)
    return x
