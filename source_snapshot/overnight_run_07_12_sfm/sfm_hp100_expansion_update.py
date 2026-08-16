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

``negative_mode="hinge"`` is the declared bounded variant of that negative
term: it becomes ``+ alpha * mean(relu(negative_margin - L_CFM(D-)))`` over
the same full D- set, so each negative's likelihood is pushed down only until
its CFM loss reaches the margin, where its gradient vanishes.  The unbounded
literal term rewards diverging on D- without limit (observed at mega-scale:
D- CFM loss 0.6 -> ~11 within one round, tripping the 10x divergence guard);
the hinge cannot, though the guard stays armed in both modes.

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
# Declared D+ mass modes.  pooled_mean is the original per-batch mean; the
# two weighted modes compute one fixed weight vector over the whole D+ set
# per round (summing to exactly 1) and each optimizer step contributes
# sum(w_i * L_i) over its batch: batch sums are partial masses of that fixed
# global weighting, never re-normalized per batch, so one complete pass
# carries total positive mass exactly 1.
POSITIVE_MASS_MODES = (
    "pooled_mean", "per_gamma_balanced", "progress_weighted",
    "mode_gamma_tree",
)
# The weighted modes floor every weight at this fraction of the uniform
# 1/|D+| weight before the single final renormalization, so no executed
# positive is ever silently erased from the objective.
PROGRESS_WEIGHT_FLOOR_FRACTION = 0.25
# mode_gamma_tree: inside the avoidance branch, rows where safety actively
# changed the selected candidate (hard_avoid) carry this share of the
# branch mass; safe-pass and unknown-shadow rows share the remainder.
HARD_AVOID_SHARE = 2.0 / 3.0


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
    # Declared negative-term mode.  "literal" is the exact declared objective
    # L = mean(L+) - alpha*mean(L-), kept bit-for-bit; "hinge" replaces the
    # negative term with + alpha * mean(relu(negative_margin - L_CFM(D-))):
    # a bounded push that reduces each D- sample's likelihood only until its
    # CFM loss reaches the margin, where its gradient vanishes, so the term
    # cannot diverge the way the unbounded literal term can at scale.  The
    # 10x negative-loss divergence guard stays armed in both modes.
    negative_mode: str = "literal"
    # hinge only: the per-sample CFM-loss level beyond which a D- row stops
    # contributing gradient.  Ignored by literal.
    negative_margin: float = 2.0
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
    # Replay window over per-round archives: round k trains on the union of
    # the D+/D- archives of rounds max(1, k-N+1)..k, N=1 being the original
    # fresh-archive-only behavior.  Only the two declared roles ever enter
    # the union (no relabeling, no historical P1/P2 revival), the health gate
    # keeps evaluating the fresh archive alone, and the window is part of the
    # declared recipe identity checked on resume.
    replay_window: int = 1
    # Declared D+ mass mode (see POSITIVE_MASS_MODES).  pooled_mean keeps the
    # original per-batch-mean code path bit-for-bit; per_gamma_balanced gives
    # every gamma present equal total mass; progress_weighted keeps each
    # gamma's pooled mass share but redistributes it inside the gamma
    # proportionally to the selected candidate's H10 goal progress;
    # mode_gamma_tree anchors on goal-seeking rows while concentrating
    # learning on tagged avoidance rows (see positive_mass_weights).
    positive_mass: str = "pooled_mean"
    # mode_gamma_tree only: total objective mass carried by the avoidance
    # branch (interaction rows); goal-seeking rows anchor the remainder.
    # Ignored by every other mass mode.
    avoid_mass: float = 0.65
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
        if self.negative_mode not in {"literal", "hinge"}:
            raise ValueError("negative_mode must be literal or hinge")
        if self.negative_margin <= 0.0:
            raise ValueError("negative_margin must be positive")
        if self.grad_clip_norm <= 0.0:
            raise ValueError("gradient clip norm must be positive")
        if not 0.0 < self.max_relative_parameter_drift < 1.0:
            raise ValueError("relative drift gate must lie in (0,1)")
        if self.train_mode not in {"eval", "train"}:
            raise ValueError("train_mode must be eval or train")
        if self.replay_window < 1:
            raise ValueError("replay window must be a positive round count")
        if self.positive_mass not in POSITIVE_MASS_MODES:
            raise ValueError(
                f"positive_mass must be one of {list(POSITIVE_MASS_MODES)}"
            )
        if not 0.0 < self.avoid_mass < 1.0:
            raise ValueError("avoid_mass must lie in (0,1)")
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


def _leaf_gamma_uniform(
    weights: np.ndarray, indices: np.ndarray, gammas: np.ndarray, mass: float,
) -> None:
    """Split ``mass`` equally across the gammas present in one tree leaf,
    uniformly per row inside each gamma (the per_gamma_balanced convention,
    applied leaf-locally)."""
    leaf_gammas = gammas[indices]
    unique = np.unique(leaf_gammas)
    for gamma in unique:
        members = indices[leaf_gammas == gamma]
        weights[members] = mass / (unique.size * members.size)


def positive_mass_weights(
    positives: list[dict], mode: str, *, avoid_mass: float = 0.65,
) -> np.ndarray:
    """One fixed, global D+ weight vector for the declared mass mode.

    The vector sums to exactly 1 over the full D+ set and is computed once
    per round; batches index into it, so batch sums are partial masses of a
    fixed global weighting (there is no per-batch renormalization).

    - ``pooled_mean``: uniform ``1/n`` (returned for the audit only — the
      update itself keeps the original per-batch-mean code path bit-for-bit).
    - ``per_gamma_balanced``: every gamma present carries equal total mass
      ``1/n_gammas``, split uniformly inside the gamma.
    - ``progress_weighted``: each gamma keeps its pooled mass share
      ``n_gamma/n``; inside the gamma, mass is proportional to the selected
      candidate's H10 goal progress
      (``row["prediction_audit"]["H10_goal_progress"]`` — the value the
      selector already cross-checked against the exact verifier's progress),
      with nonpositive progress clipped to zero.  Every weight is floored at
      ``PROGRESS_WEIGHT_FLOOR_FRACTION`` of uniform before one final global
      renormalization, so no row's mass vanishes.
    - ``mode_gamma_tree``: two-level declared mass tree over the
      ``row["mode_tags"]`` set by ``sfm_hp100_mode_tags.tag_rows`` (fail
      closed on untagged rows).  Avoidance rows (``interaction`` True) carry
      total mass ``avoid_mass`` and goal-seeking rows anchor the remainder;
      inside avoidance, hard_avoid rows (``changed`` True) carry
      ``HARD_AVOID_SHARE`` of the branch and safe-pass/unknown rows the rest.
      An empty branch's mass moves to its sibling.  Every leaf splits its
      mass per gamma equally, then uniformly per row; weights are floored at
      ``PROGRESS_WEIGHT_FLOOR_FRACTION`` of uniform and renormalized once
      globally to total mass exactly 1.
    """
    if mode not in POSITIVE_MASS_MODES:
        raise ValueError(f"undeclared positive mass mode: {mode!r}")
    count = len(positives)
    if count == 0:
        raise ValueError("positive mass weights require at least one D+ row")
    uniform = 1.0 / count
    if mode == "pooled_mean":
        return np.full(count, uniform, dtype=np.float64)
    gammas = np.asarray([float(row["gamma"]) for row in positives])
    unique = np.unique(gammas)
    weights = np.empty(count, dtype=np.float64)
    if mode == "per_gamma_balanced":
        for gamma in unique:
            mask = gammas == gamma
            weights[mask] = 1.0 / (unique.size * int(mask.sum()))
        return weights
    if mode == "mode_gamma_tree":
        if not 0.0 < float(avoid_mass) < 1.0:
            raise ValueError("avoid_mass must lie in (0,1)")
        # Fail closed on untagged rows: KeyError on either level.
        interaction = np.asarray([
            bool(row["mode_tags"]["interaction"]) for row in positives
        ])
        hard = np.asarray([
            row["mode_tags"]["changed"] is True for row in positives
        ])
        indices = np.arange(count)
        goal = indices[~interaction]
        hard_avoid = indices[interaction & hard]
        soft_avoid = indices[interaction & ~hard]
        weights[:] = 0.0
        # Empty-branch reassignment at the top level...
        avoidance_total = float(avoid_mass)
        goal_total = 1.0 - avoidance_total
        if goal.size == 0:
            avoidance_total, goal_total = 1.0, 0.0
        if hard_avoid.size + soft_avoid.size == 0:
            avoidance_total, goal_total = 0.0, 1.0
        # ...and inside the avoidance branch.
        hard_total = avoidance_total * HARD_AVOID_SHARE
        soft_total = avoidance_total - hard_total
        if hard_avoid.size == 0:
            hard_total, soft_total = 0.0, avoidance_total
        if soft_avoid.size == 0 and hard_avoid.size > 0:
            hard_total, soft_total = avoidance_total, 0.0
        for leaf, mass in (
            (goal, goal_total), (hard_avoid, hard_total),
            (soft_avoid, soft_total),
        ):
            if leaf.size and mass > 0.0:
                _leaf_gamma_uniform(weights, leaf, gammas, mass)
        weights = np.maximum(weights, PROGRESS_WEIGHT_FLOOR_FRACTION * uniform)
        return weights / float(weights.sum())
    progress = np.asarray([
        float(row["prediction_audit"]["H10_goal_progress"])
        for row in positives
    ])
    for gamma in unique:
        mask = gammas == gamma
        share = float(mask.sum()) / count
        base = np.clip(progress[mask], 0.0, None)
        total = float(base.sum())
        if total <= 0.0:
            weights[mask] = share / int(mask.sum())
        else:
            weights[mask] = share * base / total
    weights = np.maximum(weights, PROGRESS_WEIGHT_FLOOR_FRACTION * uniform)
    return weights / float(weights.sum())


def _branch_masses(positives: list[dict], weights: np.ndarray) -> dict:
    """Audited effective mass per mode_gamma_tree behavior branch."""
    masses = {"goal_seeking": 0.0, "hard_avoid": 0.0,
              "safe_pass_or_unknown": 0.0}
    counts = {key: 0 for key in masses}
    for row, weight in zip(positives, weights):
        tags = row["mode_tags"]
        if not tags["interaction"]:
            branch = "goal_seeking"
        elif tags["changed"] is True:
            branch = "hard_avoid"
        else:
            branch = "safe_pass_or_unknown"
        masses[branch] += float(weight)
        counts[branch] += 1
    return {"mass": masses, "rows": counts}


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
    mass_weights = positive_mass_weights(
        positives, config.positive_mass, avoid_mass=config.avoid_mass,
    )
    row_gammas = np.asarray([float(row["gamma"]) for row in positives])
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
    negative_hinge_values = []
    grad_pre_clip, grad_post_clip = [], []
    positive_grad_norms, negative_grad_norms, grad_cosines = [], [], []
    per_pass_positive_loss, per_pass_grad_norm = [], []
    per_pass_beyond_margin = []
    clipped_steps = 0
    steps = 0
    initial_negative_loss = None
    abort_reason = None

    for epoch in range(config.exposure_passes):
        pass_losses, pass_norms, pass_beyond = [], [], []
        order = np.random.default_rng(_counter_seed(
            config.seed, "sfm2_Dplus_order", int(round_index), epoch,
        )).permutation(len(positives))
        ordered = [positives[int(index)] for index in order]
        ordered_weights = mass_weights[order]
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
            if config.positive_mass == "pooled_mean":
                # Original semantics, untouched: unit objective mass per step.
                positive_loss = positive_per_sample.mean()
            else:
                # Partial mass of the fixed global weighting: one complete
                # pass sums to total positive mass exactly 1.
                batch_weights = torch.as_tensor(
                    ordered_weights[start:start + config.batch_size],
                    device=device, dtype=positive_per_sample.dtype,
                )
                positive_loss = (batch_weights * positive_per_sample).sum()
            negative_loss_value = None
            negative_term_value = None
            if negative_stack is not None:
                if config.alpha > 0.0:
                    negative_per_sample = adapter.cfm_loss(
                        *negative_stack, reduction="none",
                    )
                    negative_loss = negative_per_sample.mean()
                    negative_loss_value = float(negative_loss.detach())
                    if config.negative_mode == "hinge":
                        # Bounded push: the objective ADDS
                        # + alpha * mean(relu(margin - L-)) over the full D-
                        # set (full_set_mean semantics unchanged), so a D-
                        # sample whose CFM loss already sits at or beyond the
                        # margin contributes exactly zero gradient.
                        negative_term = torch.relu(
                            config.negative_margin - negative_per_sample
                        ).mean()
                        term_sign = 1.0
                        negative_term_value = float(negative_term.detach())
                        pass_beyond.append(float(
                            (
                                negative_per_sample.detach()
                                >= config.negative_margin
                            ).float().mean()
                        ))
                    else:
                        # Literal declared objective: - alpha * mean(L-).
                        negative_term = negative_loss
                        term_sign = -1.0
                    positive_grad = torch.autograd.grad(
                        positive_loss, parameters, allow_unused=True,
                    )
                    negative_grad = torch.autograd.grad(
                        negative_term, parameters, allow_unused=True,
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
                    # term_sign folds the mode into one combination rule:
                    # literal is pos - alpha*grad(mean L-) exactly as before
                    # (IEEE sign flips commute with multiply/add, so the
                    # literal path stays bitwise), hinge is
                    # pos + alpha*grad(mean relu(margin - L-)).
                    for parameter, pos, neg in zip(
                        parameters, positive_grad, negative_grad,
                    ):
                        if pos is None and neg is None:
                            parameter.grad = None
                        elif pos is None:
                            parameter.grad = (
                                term_sign * config.alpha * neg.detach()
                            )
                        elif neg is None:
                            parameter.grad = pos.detach()
                        else:
                            parameter.grad = (
                                pos.detach()
                                + term_sign * config.alpha * neg.detach()
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
                if negative_term_value is not None:
                    # hinge with alpha > 0: the realized objective adds the
                    # bounded term; negative_loss_value stays the raw mean
                    # CFM loss on D- for the audit and divergence guard.
                    negative_hinge_values.append(negative_term_value)
                    objectives.append(
                        positive_loss_value
                        + config.alpha * negative_term_value
                    )
                else:
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
        per_pass_beyond_margin.append(
            float(np.mean(pass_beyond)) if pass_beyond else None
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
    negative_hinge_mean = (
        float(np.mean(negative_hinge_values)) if negative_hinge_values else None
    )
    objective_mean = float(np.mean(objectives)) if objectives else None
    if positive_loss_mean is not None and objective_mean is not None:
        if negative_hinge_mean is not None:
            # hinge with alpha > 0.
            expected_objective = (
                positive_loss_mean + config.alpha * negative_hinge_mean
            )
        elif negative_loss_mean is not None:
            expected_objective = (
                positive_loss_mean - config.alpha * negative_loss_mean
            )
        else:
            expected_objective = None
        if expected_objective is not None and not math.isclose(
            objective_mean, expected_objective,
            rel_tol=1.0e-9, abs_tol=1.0e-9,
        ):
            raise RuntimeError(
                "objective violates positive-minus-alpha-negative identity"
            )
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
        # Raw mean CFM loss on D- in BOTH negative modes (the divergence
        # guard's statistic); the hinge term itself is audited separately.
        "negative_loss_mean": negative_loss_mean,
        "negative_mode": config.negative_mode,
        "negative_margin": float(config.negative_margin),
        "negative_hinge_mean": negative_hinge_mean,
        # hinge only: per-pass fraction of D- rows whose CFM loss already
        # sits at or beyond the margin (zero-gradient rows).  None entries
        # mean the pass ran without an active hinge term.
        "fraction_of_dminus_beyond_margin": per_pass_beyond_margin,
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
        # Declared D+ mass mode with its effective per-gamma objective mass
        # and the fixed global weight vector's statistics.  For the weighted
        # modes each per-step positive term is a partial mass of this fixed
        # weighting; for pooled_mean the weights are the uniform reference.
        "positive_mass": config.positive_mass,
        "positive_mass_per_gamma": {
            f"{gamma:g}": float(mass_weights[row_gammas == gamma].sum())
            for gamma in np.unique(row_gammas)
        },
        "positive_weight_stats": {
            "min": float(mass_weights.min()),
            "max": float(mass_weights.max()),
            "mean": float(mass_weights.mean()),
        },
        # mode_gamma_tree only: effective mass per declared behavior branch
        # (goal_seeking / hard_avoid / safe-pass-or-unknown) and the declared
        # avoidance split; None for the other modes.
        "avoid_mass": (
            float(config.avoid_mass)
            if config.positive_mass == "mode_gamma_tree" else None
        ),
        "positive_mass_per_branch": (
            _branch_masses(positives, mass_weights)
            if config.positive_mass == "mode_gamma_tree" else None
        ),
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
