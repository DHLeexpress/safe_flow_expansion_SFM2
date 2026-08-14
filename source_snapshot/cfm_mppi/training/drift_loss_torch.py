from __future__ import annotations

import torch
import torch.nn.functional as F


def drifting_loss(
    gen: torch.Tensor,
    fixed_pos: torch.Tensor,
    fixed_neg: torch.Tensor | None = None,
    *,
    temperature: float = 1.0,
    attraction_weight: float = 1.0,
    repulsion_weight: float = 0.2,
) -> torch.Tensor:
    """
    PyTorch analogue of a one-step drifting affinity objective.

    The positive term pulls generated controls toward safeMPPI expert controls.
    The negative term pushes generated controls away from unsafe/bad controls. If
    no negatives are available, noisy/generated controls are used by the caller as
    documented fallback.
    """
    if gen.shape != fixed_pos.shape:
        raise ValueError(f"gen and fixed_pos must match, got {tuple(gen.shape)} vs {tuple(fixed_pos.shape)}")
    gen_flat = gen.flatten(1)
    pos_flat = fixed_pos.flatten(1)
    pos_dist = torch.mean((gen_flat - pos_flat) ** 2, dim=1)
    loss = attraction_weight * pos_dist.mean()
    if fixed_neg is not None:
        neg_flat = fixed_neg.flatten(1)
        neg_dist = torch.mean((gen_flat - neg_flat) ** 2, dim=1)
        repulsion = F.softplus((temperature - neg_dist) / max(temperature, 1e-6)).mean()
        loss = loss + repulsion_weight * repulsion
    return loss
