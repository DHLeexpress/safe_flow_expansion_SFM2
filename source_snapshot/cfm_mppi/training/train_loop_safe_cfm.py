from __future__ import annotations

import math
import time
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F

from cfm_mppi.flow_matching.path import CondOTProbPath
from cfm_mppi.models.context_encoder import context_kwargs_from_batch


def _controls_channels_first(batch: Dict[str, torch.Tensor], key: str = "controls_si") -> torch.Tensor:
    controls = batch[key].float()
    return controls.transpose(1, 2).contiguous()


def safe_cfm_loss(model: torch.nn.Module, batch: Dict[str, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, Dict[str, float]]:
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    target = _controls_channels_first(batch)
    noise = torch.randn_like(target)
    t = torch.rand(target.shape[0], device=device).clamp(1e-4, 1.0)
    path = CondOTProbPath()
    sample = path.sample(t=t, x_0=noise, x_1=target)
    pred = model(sample.x_t, t, **context_kwargs_from_batch(batch))
    loss = F.mse_loss(pred, sample.dx_t)
    return loss, {"loss": float(loss.detach().cpu())}


def train_one_epoch_safe_cfm(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    grad_clip: float | None = 1.0,
) -> Dict[str, float]:
    model.train()
    total = 0.0
    n = 0
    grad_norm_value = 0.0
    t0 = time.perf_counter()
    for batch in data_loader:
        loss, _ = safe_cfm_loss(model, batch, device)
        if not math.isfinite(float(loss.detach().cpu())):
            raise ValueError(f"Non-finite safe CFM loss: {loss.item()}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            grad_norm_value = float(grad_norm.detach().cpu())
        optimizer.step()
        total += float(loss.detach().cpu())
        n += 1
    return {
        "loss": total / max(n, 1),
        "gradient_norm": grad_norm_value,
        "epoch_time": time.perf_counter() - t0,
    }


@torch.no_grad()
def evaluate_safe_cfm(model: torch.nn.Module, data_loader: Iterable, device: torch.device) -> Dict[str, float]:
    model.eval()
    total = 0.0
    n = 0
    for batch in data_loader:
        loss, _ = safe_cfm_loss(model, batch, device)
        total += float(loss.detach().cpu())
        n += 1
    return {"loss": total / max(n, 1)}
