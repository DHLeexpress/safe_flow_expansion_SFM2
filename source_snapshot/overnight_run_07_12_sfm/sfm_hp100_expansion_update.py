"""Positive-minus-alpha-negative CFM update for SFM2 predictive expansion.

The literal declared objective is

    L(theta) = mean(L_CFM(D+)) - alpha * mean(L_CFM(D-))

with ``alpha=0`` as the mandatory control.  There is no normalized
signed-gradient rho, no per-lineage alpha scaling, and no P1/P2/Ncausal/D0
replay role: the historical ``phased_update`` convention is deliberately not
reused.  Negative mass is ``full_set_mean``: every optimizer step evaluates
the negative term over the entire D- set, so D- carries total objective mass
exactly ``alpha`` regardless of its cardinality and no rare negative silently
receives arbitrary mass through oversampling.

The default trainable surface is the complete flow trunk plus head
(``trunk.inp + blocks[0] + blocks[1] + head``); the declared reduced arm
(``last_two_blocks_and_head``) leaves ``trunk.inp`` frozen as well, and the
declared minimal arm (``last_block_and_head``) additionally freezes
``blocks[0]``.  Every frozen entry — all condition encoders, plus each trunk
layer a narrowed scope excludes — has its state digest asserted bitwise
before and after each round.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math

import numpy as np
import torch

import _paths  # noqa: F401
import sfm_hp100_ball_adapter as PORT
from sfm_hp100_ball_core.expansion import _counter_seed
import sfm_hp100_exhaustive_hybrid as HYBRID


VERSION = "sfm_hp100_expansion_update_v1"
OPTIMIZER_SCOPE = "trunk_and_head"
TRAINABLE_PARAMETER_COUNT = 327_956
FROZEN_TRAINABLE_NAMES = (
    "policy.head.bias",
    "policy.head.weight",
    "policy.trunk.blocks.0.0.bias",
    "policy.trunk.blocks.0.0.weight",
    "policy.trunk.blocks.0.1.bias",
    "policy.trunk.blocks.0.1.weight",
    "policy.trunk.blocks.0.3.bias",
    "policy.trunk.blocks.0.3.weight",
    "policy.trunk.blocks.1.0.bias",
    "policy.trunk.blocks.1.0.weight",
    "policy.trunk.blocks.1.1.bias",
    "policy.trunk.blocks.1.1.weight",
    "policy.trunk.blocks.1.3.bias",
    "policy.trunk.blocks.1.3.weight",
    "policy.trunk.inp.0.bias",
    "policy.trunk.inp.0.weight",
)
# Declared reduced surface: both residual blocks plus head with the trunk
# input layer left frozen alongside the condition encoders.  The scope name
# is the frozen adapter's own "last_two_blocks_and_head"; on the two-block
# HP100 trunk that is exactly blocks[0] + blocks[1] + head.
REDUCED_OPTIMIZER_SCOPE = "last_two_blocks_and_head"
REDUCED_TRAINABLE_PARAMETER_COUNT = 269_332
REDUCED_TRAINABLE_NAMES = tuple(
    name for name in FROZEN_TRAINABLE_NAMES
    if not name.startswith("policy.trunk.inp.")
)
# Declared minimal surface: the last residual block plus head only, leaving
# trunk.inp and blocks[0] frozen alongside the condition encoders.  The scope
# name is the frozen adapter's own "last_block_and_head".
MINIMAL_OPTIMIZER_SCOPE = "last_block_and_head"
MINIMAL_TRAINABLE_PARAMETER_COUNT = 137_236
MINIMAL_TRAINABLE_NAMES = tuple(
    name for name in FROZEN_TRAINABLE_NAMES
    if name.startswith(("policy.head.", "policy.trunk.blocks.1."))
)
DECLARED_TRAINABLE_SURFACES = {
    OPTIMIZER_SCOPE: (FROZEN_TRAINABLE_NAMES, TRAINABLE_PARAMETER_COUNT),
    REDUCED_OPTIMIZER_SCOPE: (
        REDUCED_TRAINABLE_NAMES, REDUCED_TRAINABLE_PARAMETER_COUNT,
    ),
    MINIMAL_OPTIMIZER_SCOPE: (
        MINIMAL_TRAINABLE_NAMES, MINIMAL_TRAINABLE_PARAMETER_COUNT,
    ),
}
# Trunk state each narrowed scope excludes from training and must therefore
# leave bitwise untouched, digested under its own key.
_EXCLUDED_TRUNK_PREFIXES = {
    REDUCED_OPTIMIZER_SCOPE: (("trunk_inp", "trunk.inp."),),
    MINIMAL_OPTIMIZER_SCOPE: (
        ("trunk_inp", "trunk.inp."),
        ("trunk_block_0", "trunk.blocks.0."),
    ),
}
NEGATIVE_LOSS_ABORT_FACTOR = 10.0


@dataclass(frozen=True)
class UpdateConfig:
    alpha: float = 0.0
    # The declared fixed learning rate for the exposure sweep.
    learning_rate: float = 1.0e-5
    batch_size: int = 64
    # E: complete deterministically reshuffled passes over the eligible round
    # archive.  This is the declared primary recipe variable of the sweep
    # (E in {1, 4, 16}); every pass exposes each D+ row exactly once, so total
    # exposure is E-fold by design and is audited truthfully below.
    exposure_passes: int = 1
    negative_mass: str = "full_set_mean"
    grad_clip_norm: float = 1.0
    max_relative_parameter_drift: float = 0.25
    # Historical expansion modules trained the adapter in eval mode (dropout
    # inactive, deterministic gradients); pretraining used train mode.  The
    # choice is declared here rather than inherited silently.
    train_mode: str = "eval"
    # Declared trainable surface: the full trunk_and_head default, the
    # reduced last_two_blocks_and_head arm that leaves trunk.inp frozen, or
    # the minimal last_block_and_head arm that also freezes blocks[0].
    optimizer_scope: str = OPTIMIZER_SCOPE
    seed: int = 2

    def validate(self) -> None:
        if self.alpha < 0.0:
            raise ValueError("alpha must be nonnegative")
        if (
            self.learning_rate <= 0.0 or self.batch_size < 1
            or self.exposure_passes < 1
        ):
            raise ValueError(
                "learning rate, batch size, and exposure passes must be positive"
            )
        if self.negative_mass != "full_set_mean":
            raise ValueError("only the declared full_set_mean negative mass is allowed")
        if self.grad_clip_norm <= 0.0:
            raise ValueError("gradient clip norm must be positive")
        if not 0.0 < self.max_relative_parameter_drift < 1.0:
            raise ValueError("relative drift gate must lie in (0,1)")
        if self.train_mode not in {"eval", "train"}:
            raise ValueError("train_mode must be eval or train")
        if self.optimizer_scope not in DECLARED_TRAINABLE_SURFACES:
            raise ValueError(
                "optimizer_scope must be one of "
                f"{sorted(DECLARED_TRAINABLE_SURFACES)}"
            )


def configure_trainable(
    adapter: PORT.HP100ExpansionPolicy,
    scope: str = OPTIMIZER_SCOPE,
) -> tuple[list[torch.nn.Parameter], list[str]]:
    """Apply one declared optimizer scope and prove its exact surface."""
    if scope not in DECLARED_TRAINABLE_SURFACES:
        raise ValueError(f"undeclared optimizer scope: {scope!r}")
    declared_names, declared_count = DECLARED_TRAINABLE_SURFACES[scope]
    parameters = adapter.expansion_optimizer_parameters(scope)
    names = sorted(
        name for name, parameter in adapter.named_parameters()
        if parameter.requires_grad
    )
    if tuple(names) != declared_names:
        raise RuntimeError(
            f"trainable surface drifted: {names} != {list(declared_names)}"
        )
    count = sum(parameter.numel() for parameter in parameters)
    if count != declared_count:
        raise RuntimeError(
            f"{scope} parameter count drifted: {count} != {declared_count}"
        )
    return parameters, names


def encoder_state_sha256(adapter: PORT.HP100ExpansionPolicy) -> dict:
    """Bitwise digest of every frozen condition-encoder state entry."""
    grouped: dict[str, hashlib._hashlib.HASH] = {}
    combined = hashlib.sha256()
    for name, tensor in sorted(adapter.policy.state_dict().items()):
        if name.startswith("trunk.") or name.startswith("head."):
            continue
        module = name.split(".", 1)[0]
        value = tensor.detach().cpu().contiguous()
        for digest in (grouped.setdefault(module, hashlib.sha256()), combined):
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
            digest.update(value.numpy().tobytes())
    if not grouped:
        raise RuntimeError("policy exposes no frozen condition-encoder state")
    return {
        **{module: digest.hexdigest() for module, digest in sorted(grouped.items())},
        "combined": combined.hexdigest(),
    }


def frozen_surface_sha256(
    adapter: PORT.HP100ExpansionPolicy,
    scope: str = OPTIMIZER_SCOPE,
) -> dict:
    """Bitwise digest of everything the declared scope must leave untouched.

    Always covers every condition encoder; each trunk layer a narrowed scope
    excludes (``trunk.inp`` for the reduced scope, plus ``trunk.blocks.0``
    for the minimal scope) joins the asserted-frozen set under its own key.
    """
    if scope not in DECLARED_TRAINABLE_SURFACES:
        raise ValueError(f"undeclared optimizer scope: {scope!r}")
    digest = encoder_state_sha256(adapter)
    for key, prefix in _EXCLUDED_TRUNK_PREFIXES.get(scope, ()):
        group = hashlib.sha256()
        seen = False
        for name, tensor in sorted(adapter.policy.state_dict().items()):
            if not name.startswith(prefix):
                continue
            seen = True
            value = tensor.detach().cpu().contiguous()
            group.update(name.encode())
            group.update(str(value.dtype).encode())
            group.update(np.asarray(value.shape, dtype=np.int64).tobytes())
            group.update(value.numpy().tobytes())
        if not seen:
            raise RuntimeError(f"policy exposes no {prefix} state to freeze")
        digest[key] = group.hexdigest()
    return digest


def _validate_roles(positives, negatives) -> None:
    for row in positives:
        if row["role"] != "positive" or not row["verification"]["valid"]:
            raise ValueError("D+ rows must be executed exact positives")
        if row.get("negative_reason") is not None:
            raise ValueError("D+ rows cannot carry a negative reason")
    declared = {"all_negative_nvp", "realized_collision", "realized_oob"}
    for row in negatives:
        if row["role"] != "negative" or row.get("negative_reason") not in declared:
            raise ValueError("D- rows must carry a declared negative reason")
        realized = row["negative_reason"] != "all_negative_nvp"
        if bool(row["verification"]["valid"]) != realized:
            raise ValueError("D- verifier label disagrees with its declared reason")


def _row_keys(rows) -> list[tuple]:
    return [
        (
            row["lineage"], int(row["scenario_id"]), int(row["step"]),
            int(row["attempt"]), int(row.get("round", 0)), int(row.get("block", 0)),
        )
        for row in rows
    ]


def _in_archive_duplicate_rows(rows) -> int:
    keys = _row_keys(rows)
    return len(keys) - len(set(keys))


def expansion_update(
    adapter: PORT.HP100ExpansionPolicy,
    positives: list[dict],
    negatives: list[dict],
    config: UpdateConfig,
    *,
    round_index: int,
    optimizer: torch.optim.Adam | None = None,
) -> dict:
    """One declared update round: E complete reshuffled passes over D+.

    When ``optimizer`` is provided it must already hold exactly the declared
    trainable parameters; its momentum state then persists across rounds so a
    cumulative arm can be resumed bitwise from any saved round.
    """
    config.validate()
    if not positives:
        raise ValueError("expansion update requires at least one D+ row")
    _validate_roles(positives, negatives)
    parameters, trainable_names = configure_trainable(
        adapter, config.optimizer_scope,
    )
    device = parameters[0].device
    encoder_before = frozen_surface_sha256(adapter, config.optimizer_scope)
    before = HYBRID._parameter_snapshot(parameters)
    getattr(adapter, config.train_mode)()

    negative_stack = None
    if negatives:
        negative_stack = HYBRID._stack_rows(negatives, device)

    if optimizer is None:
        optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    else:
        held = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        if len(held) != len(parameters) or any(
            a is not b for a, b in zip(held, parameters)
        ):
            raise RuntimeError(
                "the persistent optimizer does not hold the declared surface"
            )
    positive_losses, negative_losses, objectives = [], [], []
    grad_pre_clip, grad_post_clip = [], []
    positive_grad_norms, negative_grad_norms, grad_cosines = [], [], []
    per_pass_positive_loss, per_pass_grad_norm = [], []
    clipped_steps = 0
    steps = 0
    initial_negative_loss = None
    abort_reason = None

    for epoch in range(config.exposure_passes):
        pass_losses, pass_norms = [], []
        order = np.random.default_rng(_counter_seed(
            config.seed, "sfm2_Dplus_order", int(round_index), epoch,
        )).permutation(len(positives))
        ordered = [positives[int(index)] for index in order]
        for batch_index, start in enumerate(
            range(0, len(ordered), config.batch_size)
        ):
            batch = ordered[start:start + config.batch_size]
            contexts, candidates = HYBRID._stack_rows(batch, device)
            HYBRID._set_step_seed(_counter_seed(
                config.seed, "sfm2_cfm_noise", int(round_index),
                epoch, batch_index,
            ), device)
            positive_per_sample = adapter.cfm_loss(
                contexts, candidates, reduction="none",
            )
            positive_loss = positive_per_sample.mean()
            negative_loss_value = None
            if negative_stack is not None:
                if config.alpha > 0.0:
                    negative_per_sample = adapter.cfm_loss(
                        *negative_stack, reduction="none",
                    )
                    negative_loss = negative_per_sample.mean()
                    negative_loss_value = float(negative_loss.detach())
                    positive_grad = torch.autograd.grad(
                        positive_loss, parameters, allow_unused=True,
                    )
                    negative_grad = torch.autograd.grad(
                        negative_loss, parameters, allow_unused=True,
                    )
                    positive_norm = HYBRID._gradient_norm(positive_grad, device)
                    negative_norm = HYBRID._gradient_norm(negative_grad, device)
                    dot = sum(
                        (
                            (pos * neg).sum()
                            for pos, neg in zip(positive_grad, negative_grad)
                            if pos is not None and neg is not None
                        ),
                        torch.zeros((), device=device),
                    )
                    positive_grad_norms.append(float(positive_norm))
                    negative_grad_norms.append(float(negative_norm))
                    grad_cosines.append(float(
                        dot / (positive_norm * negative_norm + 1.0e-24)
                    ))
                    optimizer.zero_grad()
                    for parameter, pos, neg in zip(
                        parameters, positive_grad, negative_grad,
                    ):
                        if pos is None and neg is None:
                            parameter.grad = None
                        elif pos is None:
                            parameter.grad = -config.alpha * neg.detach()
                        elif neg is None:
                            parameter.grad = pos.detach()
                        else:
                            parameter.grad = (
                                pos.detach() - config.alpha * neg.detach()
                            )
                else:
                    # alpha=0 control: the negative term contributes exactly
                    # zero gradient but is still evaluated (without a graph)
                    # so both arms consume identical CFM noise draws.
                    with torch.no_grad():
                        negative_loss_value = float(adapter.cfm_loss(
                            *negative_stack, reduction="none",
                        ).mean())
                    optimizer.zero_grad()
                    positive_loss.backward()
            else:
                optimizer.zero_grad()
                positive_loss.backward()

            pre_clip = float(torch.nn.utils.clip_grad_norm_(
                parameters, config.grad_clip_norm,
            ))
            post_clip = float(HYBRID._gradient_norm(
                [parameter.grad for parameter in parameters], device,
            ))
            clipped_steps += int(pre_clip > config.grad_clip_norm)
            optimizer.step()
            steps += 1

            positive_loss_value = float(positive_loss.detach())
            positive_losses.append(positive_loss_value)
            pass_losses.append(positive_loss_value)
            pass_norms.append(pre_clip)
            grad_pre_clip.append(pre_clip)
            grad_post_clip.append(post_clip)
            if negative_loss_value is None:
                objectives.append(positive_loss_value)
            else:
                negative_losses.append(negative_loss_value)
                objectives.append(
                    positive_loss_value - config.alpha * negative_loss_value
                )
                if initial_negative_loss is None:
                    initial_negative_loss = negative_loss_value
                elif config.alpha > 0.0 and negative_loss_value > (
                    NEGATIVE_LOSS_ABORT_FACTOR
                    * max(initial_negative_loss, 1.0e-12)
                ):
                    abort_reason = "negative_loss_divergence"
                    break
        per_pass_positive_loss.append(
            float(np.mean(pass_losses)) if pass_losses else None
        )
        per_pass_grad_norm.append(
            float(np.mean(pass_norms)) if pass_norms else None
        )
        if abort_reason is not None:
            break

    drift = HYBRID._relative_parameter_drift(parameters, before)
    finite = HYBRID._finite_parameters(parameters)
    accepted = bool(
        finite
        and abort_reason is None
        and drift <= config.max_relative_parameter_drift
    )
    if not accepted:
        HYBRID._restore_parameters(parameters, before)
    encoder_after = frozen_surface_sha256(adapter, config.optimizer_scope)
    if encoder_after != encoder_before:
        raise RuntimeError(
            "a frozen surface entry (condition encoder or excluded trunk.inp) "
            "changed during the update"
        )

    positive_loss_mean = float(np.mean(positive_losses)) if positive_losses else None
    negative_loss_mean = float(np.mean(negative_losses)) if negative_losses else None
    objective_mean = float(np.mean(objectives)) if objectives else None
    if (
        positive_loss_mean is not None
        and negative_loss_mean is not None
        and objective_mean is not None
        and not math.isclose(
            objective_mean,
            positive_loss_mean - config.alpha * negative_loss_mean,
            rel_tol=1.0e-9, abs_tol=1.0e-9,
        )
    ):
        raise RuntimeError("objective violates positive-minus-alpha-negative identity")
    return {
        "version": VERSION,
        "config": asdict(config),
        "round": int(round_index),
        "optimizer_scope": config.optimizer_scope,
        "trainable_names": list(trainable_names),
        "trainable_parameters":
            DECLARED_TRAINABLE_SURFACES[config.optimizer_scope][1],
        "steps": int(steps),
        "adam_steps": int(steps),
        "exposure_passes_declared": int(config.exposure_passes),
        "exposure_passes_completed": len(per_pass_positive_loss),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "negative_counts_by_reason": {
            reason: sum(row["negative_reason"] == reason for row in negatives)
            for reason in (
                "all_negative_nvp", "realized_collision", "realized_oob",
            )
        },
        # Exact exposure audit.  Each completed pass exposes every D+ row once;
        # the negative term touches the full D- set every step by declared
        # full_set_mean semantics.  duplicate_exposures counts every exposure
        # beyond a unique sample's first, so the E-fold design is reported
        # truthfully instead of being hidden as 0.
        "unique_positive_samples": len(set(_row_keys(positives))),
        "unique_negative_samples": len(set(_row_keys(negatives))),
        "positive_exposures": len(per_pass_positive_loss) * len(positives),
        "negative_exposures": int(steps) * len(negatives),
        "duplicate_exposures": (
            len(per_pass_positive_loss) * len(positives)
            - len(set(_row_keys(positives)))
        ),
        "in_archive_duplicate_rows": _in_archive_duplicate_rows(positives)
        + _in_archive_duplicate_rows(negatives),
        "positive_loss_mean": positive_loss_mean,
        "negative_loss_mean": negative_loss_mean,
        "objective_mean": objective_mean,
        "grad_norm_pre_clip": grad_pre_clip,
        "grad_norm_post_clip": grad_post_clip,
        "per_pass_positive_loss": per_pass_positive_loss,
        "per_pass_grad_norm": per_pass_grad_norm,
        "clipped_fraction": (clipped_steps / steps if steps else None),
        "positive_grad_norm": (
            float(np.mean(positive_grad_norms)) if positive_grad_norms else None
        ),
        "negative_grad_norm": (
            float(np.mean(negative_grad_norms)) if negative_grad_norms else None
        ),
        "grad_cosine": (float(np.mean(grad_cosines)) if grad_cosines else None),
        "relative_parameter_drift": float(drift),
        "drift_gate": float(config.max_relative_parameter_drift),
        "finite": bool(finite),
        "abort_reason": abort_reason,
        "accepted": bool(accepted),
        "encoder_state_sha256_before": encoder_before,
        "encoder_state_sha256_after": encoder_after,
        "negative_mass": config.negative_mass,
        "sample_order": (
            "E deterministic reshuffled complete passes over D+; each D+ row "
            "exposed exactly once per pass; negative term over the full D- "
            "set each step"
        ),
    }


def checkpoint_payload(
    adapter: PORT.HP100ExpansionPolicy,
    *,
    parent_checkpoint_sha256: str,
    pretrained_checkpoint_sha256: str,
    round_index: int,
    alpha: float,
    exposure_passes: int = 1,
    optimizer_scope: str = OPTIMIZER_SCOPE,
) -> dict:
    """Strict HP100 checkpoint schema so the raw evaluator loads it unchanged."""
    if optimizer_scope not in DECLARED_TRAINABLE_SURFACES:
        raise ValueError(f"undeclared optimizer scope: {optimizer_scope!r}")
    return {
        "scientific_status": "SFM2_PREDICTIVE_EXPANSION_ROUND",
        "state_dict": {
            key: value.detach().cpu().clone()
            for key, value in adapter.policy.state_dict().items()
        },
        "config": adapter.policy.config(),
        "parent_checkpoint_sha256": str(parent_checkpoint_sha256),
        "pretrained_checkpoint_sha256": str(pretrained_checkpoint_sha256),
        "optimizer_scope": str(optimizer_scope),
        "round": int(round_index),
        "alpha": float(alpha),
        "exposure_passes": int(exposure_passes),
        "promotable": False,
    }
