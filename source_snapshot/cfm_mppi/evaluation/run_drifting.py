from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch

from cfm_mppi.models.drifting_generator import DriftingGenerator


def load_drifting(checkpoint_path: str | Path, device: torch.device) -> DriftingGenerator:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = DriftingGenerator.from_mizuta_defaults(history_len=int(args.get("history_len", 10))).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def sample_drifting_controls(
    model: DriftingGenerator,
    batch: Dict[str, torch.Tensor],
    *,
    horizon: int,
    device: torch.device,
) -> torch.Tensor:
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    noise = torch.randn(batch["start"].shape[0], 2, horizon, device=device)
    return model.forward_batch(noise, batch)
