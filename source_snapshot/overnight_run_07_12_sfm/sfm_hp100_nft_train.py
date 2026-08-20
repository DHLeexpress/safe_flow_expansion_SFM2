"""Bounded contrastive velocity regression (NFT-style) on verifier groups.

Adapts DiffusionNFT (arXiv 2509.16117) to the conditional CFM policy: every
B=32 candidate of a harvested group trains one of two least-squares branches
built from the same interpolant. With ``v`` the CFM velocity target
(``x1 - x0`` in normalized action space, exactly the ``cfm_loss``
semantics), ``v_ref`` the frozen r0 velocity at the identical
``(x_tau, tau, context)``, and label ``r`` from the exact verifier:

    positive (r=1):  L = || (1-beta) * v_ref + beta * v_theta - v ||^2
    negative (r=0):  L = || (1+beta) * v_ref - beta * v_theta - v ||^2

Both branches are ordinary regressions with finite optima anchored on the
reference: the positive optimum is ``v_ref + (v - v_ref)/beta`` (a 1/beta
amplified step toward the data), the negative optimum is
``v_ref - (v - v_ref)/beta`` (an explicit finite target away from the unsafe
candidate) — unlike a hinge, the negative branch never saturates to a zero
gradient and never ascends a loss without a destination.

Group weighting (declared): within a group, exact positives carry total mass
1/2 split by ``softmax(H10_progress / PROGRESS_TEMPERATURE)`` — the
progress half of the acquisition tilt — and exact negatives carry total mass
1/2 uniformly; a group with no negatives cannot occur (harvest keeps
contested contexts only). Groups are per-gamma balanced. The batch loss is
the weighted mean (batch mass reported in the audit).

Positive rows optionally use their STORED flow base as the interpolant
origin (``--coupled-x0``): the (x0, action) coupling realized by the sampler
that generated the candidate — the rectified-flow variant.

Optional SDPO-style guard (``--lambda-safe``): when the positive- and
negative-branch batch gradients conflict (negative inner product), the
negative gradient is scaled by
``clamp((1-mu) * ||g_pos||^2 / -(g_pos . g_neg), 0, 1)`` so the combined
step still decreases the positive-branch loss at first order.

Everything is additive; trainable surfaces, digests, provider, and payloads
are imported from the audited modules unmodified.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch

import _paths  # noqa: F401
import grid_policy_sfm_hp100 as GPS
import sfm_hp100_ball_adapter as PORT
import sfm_hp100_ball_launch as BASE
import sfm_hp100_exhaustive_hybrid as HYBRID
import sfm_hp100_expansion_update as UPD
import sfm_hp100_predictive_execution as PRED
import sfm_hp100_raw_obs_dataset as RDS

VERSION = "sfm_hp100_nft_train_v1"
STATUS = "SFM2_NFT_TRAIN_COMPLETE"
GROUP_ARCHIVE_STATUS = "SFM2_LABELED_GROUP_ARCHIVE"
PROTOCOL_AMENDMENT = (
    "all-B contrastive training is a declared user-directed protocol "
    "amendment for the NFT research line; the original handoff rule (train "
    "only on the two archive roles) still governs the authoritative "
    "expansion recipes"
)
PROGRESS_TEMPERATURE = 0.5
LAMBDA_SAFE_MU = 0.05


def candidate_labels(group: dict) -> list[bool]:
    """Exact-verifier label per B-local: valid and not errored."""
    return [bool(v) for v in group["valid"]]


def group_candidate_weights(group: dict) -> torch.Tensor:
    """Within-group mass: positives 1/2 by progress softmax, negatives 1/2."""
    labels = candidate_labels(group)
    count = len(labels)
    weights = torch.zeros(count, dtype=torch.float64)
    pos = [i for i in range(count) if labels[i]]
    neg = [i for i in range(count) if not labels[i]]
    if pos:
        progress = torch.tensor(
            [float(group["H10_progress"][i]) for i in pos], dtype=torch.float64,
        )
        soft = torch.softmax(progress / PROGRESS_TEMPERATURE, dim=0)
        share = 0.5 if neg else 1.0
        for local, value in zip(pos, soft):
            weights[local] = share * float(value)
    if neg:
        share = 0.5 if pos else 1.0
        for local in neg:
            weights[local] = share / len(neg)
    return weights.to(torch.float32)


def dataset_weights(groups: list[dict]) -> list[torch.Tensor]:
    """Per-candidate weights: per-gamma balanced groups x within-group mass."""
    by_gamma = defaultdict(list)
    for index, group in enumerate(groups):
        by_gamma[f"{group['gamma']:g}"].append(index)
    gamma_count = len(by_gamma)
    scale = {}
    for gamma, indices in by_gamma.items():
        for index in indices:
            scale[index] = 1.0 / (gamma_count * len(indices))
    return [
        group_candidate_weights(group) * float(scale[index])
        for index, group in enumerate(groups)
    ]


def nft_batch_loss(
    theta: PORT.HP100ExpansionPolicy,
    reference: PORT.HP100ExpansionPolicy,
    theta_tokens: torch.Tensor,
    ref_tokens: torch.Tensor,
    batch_groups: list[dict],
    batch_weights: list[torch.Tensor],
    *,
    beta: float,
    coupled_x0: bool,
    noise_seed: int,
    device: torch.device,
    return_branches: bool = False,
):
    """Weighted-mean NFT loss over the candidates of a group batch."""
    d = theta.policy.d
    u_max = float(theta.policy.u_max)
    actions, weights, labels, x0_rows = [], [], [], []
    theta_rows, ref_rows = [], []
    for slot, (group, group_w) in enumerate(zip(batch_groups, batch_weights)):
        flags = candidate_labels(group)
        acts = group["actions"].to(device)
        bases = group["flow_bases"].to(device)
        for local in range(len(flags)):
            if float(group_w[local]) <= 0.0:
                continue
            actions.append(acts[local])
            weights.append(float(group_w[local]))
            labels.append(bool(flags[local]))
            x0_rows.append(
                bases[local].reshape(d)
                if (coupled_x0 and flags[local]) else None
            )
            theta_rows.append(slot)
            ref_rows.append(slot)
    count = len(actions)
    if count == 0:
        raise RuntimeError("empty NFT batch")
    x1 = torch.stack(actions).reshape(count, d) / u_max
    HYBRID._set_step_seed(int(noise_seed), device)
    x0 = torch.randn_like(x1)
    tau = torch.rand(count, device=x1.device).clamp(1.0e-4, 1.0)
    for row, stored in enumerate(x0_rows):
        if stored is not None:
            x0[row] = stored
    x_tau = (1.0 - tau)[:, None] * x0 + tau[:, None] * x1
    target = x1 - x0
    label_mask = torch.tensor(labels, device=device, dtype=torch.bool)
    weight = torch.tensor(weights, device=device, dtype=x1.dtype)

    theta_tok = theta_tokens[torch.tensor(theta_rows, device=device)]
    ref_tok = ref_tokens[torch.tensor(ref_rows, device=device)]
    v_theta = theta.policy(x_tau, tau, theta_tok)
    with torch.no_grad():
        v_ref = reference.policy(x_tau, tau, ref_tok)

    beta = float(beta)
    plus = (1.0 - beta) * v_ref + beta * v_theta
    minus = (1.0 + beta) * v_ref - beta * v_theta
    per_plus = (plus - target).square().reshape(count, PORT.H, 2).mean(dim=(1, 2))
    per_minus = (minus - target).square().reshape(count, PORT.H, 2).mean(dim=(1, 2))

    pos_mass = weight[label_mask].sum()
    neg_mass = weight[~label_mask].sum()
    pos_loss = (
        (weight[label_mask] * per_plus[label_mask]).sum()
        / pos_mass.clamp_min(1.0e-12)
    )
    neg_loss = (
        (weight[~label_mask] * per_minus[~label_mask]).sum()
        / neg_mass.clamp_min(1.0e-12)
    )
    total_mass = (pos_mass + neg_mass).clamp_min(1.0e-12)
    total = (pos_mass * pos_loss + neg_mass * neg_loss) / total_mass
    audit = {
        "pos_branch_loss": float(pos_loss.detach()),
        "neg_branch_loss": float(neg_loss.detach()),
        "candidates": count,
        "positives": int(label_mask.sum()),
        "batch_mass": float(total_mass.detach()),
        "theta_minus_ref_pos": float(
            (v_theta - v_ref).detach()[label_mask].square().mean()
        ) if bool(label_mask.any()) else None,
        "theta_minus_ref_neg": float(
            (v_theta - v_ref).detach()[~label_mask].square().mean()
        ) if bool((~label_mask).any()) else None,
    }
    if return_branches:
        return total, pos_loss, neg_loss, pos_mass, neg_mass, audit
    return total, audit


def lambda_safe(g_pos: torch.Tensor, g_neg: torch.Tensor,
                mu: float = LAMBDA_SAFE_MU) -> float:
    """SDPO-style scale for the negative gradient (batch level).

    Descending on ``g_pos + lambda * g_neg`` changes the positive-branch loss
    at first order by ``-(||g_pos||^2 + lambda * <g_pos, g_neg>)``. When the
    branches conflict (negative inner product), the largest safe scale that
    still decreases the positive branch at rate ``mu * ||g_pos||^2`` is
    ``(1 - mu) * ||g_pos||^2 / -<g_pos, g_neg>`` (clamped to [0, 1]); aligned
    branches need no scaling.
    """
    dot = float(torch.dot(g_pos, g_neg))
    if dot >= 0.0:
        return 1.0
    return float(min(max(
        (1.0 - mu) * float(g_pos.square().sum()) / (-dot), 0.0,
    ), 1.0))


def _flat(grads, parameters) -> torch.Tensor:
    return torch.cat([
        (g if g is not None else torch.zeros_like(p)).reshape(-1)
        for g, p in zip(grads, parameters)
    ])


def run(args) -> dict:
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing nft-train output: {output}")
    BASE._gpu_contract(args.device, int(args.physical_gpu))
    start_sha = PRED.sha256_file(args.checkpoint)
    if start_sha != str(args.expected_checkpoint_sha256).lower():
        raise RuntimeError("start checkpoint SHA256 mismatch")
    ref_sha = PRED.sha256_file(args.reference_checkpoint)
    if ref_sha != str(args.expected_reference_checkpoint_sha256).lower():
        raise RuntimeError("reference checkpoint SHA256 mismatch")

    policy, _ = GPS.load_sfm_hp100_policy(args.checkpoint, device=args.device)
    theta = PORT.HP100ExpansionPolicy(policy).eval()
    ref_policy, _ = GPS.load_sfm_hp100_policy(
        args.reference_checkpoint, device=args.device,
    )
    reference = PORT.HP100ExpansionPolicy(ref_policy).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    device = next(theta.parameters()).device

    scope = str(args.optimizer_scope)
    parameters, trainable_names = UPD.configure_trainable(theta, scope)
    frozen_before = UPD.frozen_surface_sha256(theta, scope)

    groups: list[dict] = []
    for path in args.groups:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("status") != GROUP_ARCHIVE_STATUS:
            raise RuntimeError("input is not a labeled-group archive")
        groups.extend(payload["groups"])
    if int(args.max_groups) > 0 and len(groups) > int(args.max_groups):
        rng = random.Random(7)
        rng.shuffle(groups)
        groups = groups[: int(args.max_groups)]
    if not groups:
        raise RuntimeError("no groups to train on")
    weights = dataset_weights(groups)

    provider = None
    token_audit = None
    if args.context_path == "raw":
        if not args.raw_manifest:
            raise ValueError("--context-path raw requires --raw-manifest")
        dataset = RDS.RawObsDataset(
            args.raw_manifest, lru_shards=int(args.lru_shards),
        )
        dataset.assert_joined(groups)
        audit_rows = groups[:: max(1, int(args.raw_audit_every))]
        token_audit = RDS.audit_tokens(
            dataset, theta, audit_rows, atol=float(args.raw_audit_atol),
        )
        if not token_audit["ok"]:
            raise RuntimeError(
                "raw-obs token audit failed: max deviation "
                f"{token_audit['max_abs_deviation']} > atol {token_audit['atol']}"
            )
        provider = RDS.make_context_provider(
            dataset, theta,
            open_encoders=(scope == UPD.ALL_OPEN_OPTIMIZER_SCOPE),
        )
    elif scope != UPD.OPTIMIZER_SCOPE:
        raise ValueError(
            "stored-context NFT training only supports the trunk_and_head "
            "scope; wider scopes need --context-path raw"
        )

    optimizer = torch.optim.Adam(parameters, lr=float(args.learning_rate))
    snapshot_full = HYBRID._parameter_snapshot(parameters)
    groups_per_batch = int(args.groups_per_batch)
    passes = int(args.exposure_passes)
    order_rng = random.Random(
        HYBRID._counter_seed(int(args.update_seed), "nft_order")
    )
    use_lambda_safe = bool(args.lambda_safe)

    snapshot_every = int(args.snapshot_every or 0)
    snapshot_records: list[dict] = []
    if snapshot_every > 0:
        output.mkdir(parents=True)

    def save_snapshot(step: int, running: dict) -> None:
        payload = UPD.checkpoint_payload(
            theta,
            parent_checkpoint_sha256=start_sha,
            pretrained_checkpoint_sha256=ref_sha,
            round_index=1,
            alpha=0.0,
            exposure_passes=passes,
            optimizer_scope=scope,
        )
        payload["nft"] = {
            "beta": float(args.beta),
            "coupled_x0": bool(args.coupled_x0),
            **running,
        }
        path = output / f"snapshot_step{int(step):05d}.pt"
        torch.save(payload, path)
        snapshot_records.append({
            "step": int(step), "path": str(path),
            "sha256": PRED.sha256_file(path), **running,
        })

    step_index = 0
    per_pass = []
    aborted = None
    for pass_index in range(passes):
        indices = list(range(len(groups)))
        order_rng.shuffle(indices)
        pass_audit = defaultdict(list)
        for start in range(0, len(indices), groups_per_batch):
            chosen = indices[start:start + groups_per_batch]
            batch_groups = [groups[i] for i in chosen]
            batch_weights = [weights[i] for i in chosen]
            ref_contexts = torch.stack(
                [group["context"] for group in batch_groups]
            ).to(device)
            ref_tokens = reference._policy_context(ref_contexts)
            if provider is not None:
                theta_tokens = provider(batch_groups, device)
            else:
                theta_tokens = theta._policy_context(ref_contexts)
            noise_seed = HYBRID._counter_seed(
                int(args.update_seed), "nft_noise", pass_index, step_index,
            )
            optimizer.zero_grad(set_to_none=True)
            if use_lambda_safe:
                total, pos_loss, neg_loss, pos_mass, neg_mass, audit = (
                    nft_batch_loss(
                        theta, reference, theta_tokens, ref_tokens,
                        batch_groups, batch_weights,
                        beta=float(args.beta),
                        coupled_x0=bool(args.coupled_x0),
                        noise_seed=noise_seed, device=device,
                        return_branches=True,
                    )
                )
                total_mass = (pos_mass + neg_mass).clamp_min(1.0e-12)
                g_pos = torch.autograd.grad(
                    pos_mass / total_mass * pos_loss, parameters,
                    retain_graph=True, allow_unused=True,
                )
                g_neg = torch.autograd.grad(
                    neg_mass / total_mass * neg_loss, parameters,
                    allow_unused=True,
                )
                flat_pos = _flat(g_pos, parameters)
                flat_neg = _flat(g_neg, parameters)
                scale = lambda_safe(flat_pos, flat_neg)
                for parameter, gp, gn in zip(parameters, g_pos, g_neg):
                    combined = None
                    if gp is not None:
                        combined = gp.clone()
                    if gn is not None:
                        combined = (
                            gn * scale if combined is None
                            else combined + gn * scale
                        )
                    parameter.grad = combined
                audit["lambda_safe"] = float(scale)
            else:
                total, audit = nft_batch_loss(
                    theta, reference, theta_tokens, ref_tokens,
                    batch_groups, batch_weights,
                    beta=float(args.beta),
                    coupled_x0=bool(args.coupled_x0),
                    noise_seed=noise_seed, device=device,
                )
                total.backward()
            grad_norm = HYBRID._gradient_norm(
                [parameter.grad for parameter in parameters], device,
            )
            torch.nn.utils.clip_grad_norm_(
                parameters, float(args.grad_clip_norm),
            )
            optimizer.step()
            step_index += 1
            if not HYBRID._finite_parameters(parameters):
                aborted = f"non-finite parameters at step {step_index}"
                break
            for key, value in audit.items():
                if value is not None:
                    pass_audit[key].append(float(value))
            pass_audit["grad_norm"].append(float(grad_norm))
            if snapshot_every and step_index % snapshot_every == 0:
                save_snapshot(step_index, {
                    "pos_branch_running": float(
                        sum(pass_audit["pos_branch_loss"])
                        / len(pass_audit["pos_branch_loss"])
                    ),
                    "neg_branch_running": float(
                        sum(pass_audit["neg_branch_loss"])
                        / len(pass_audit["neg_branch_loss"])
                    ),
                    "relative_parameter_drift": float(
                        HYBRID._relative_parameter_drift(
                            parameters, snapshot_full,
                        )
                    ),
                })
        per_pass.append({
            key: float(sum(values) / len(values))
            for key, values in pass_audit.items() if values
        })
        if aborted:
            break

    drift = float(HYBRID._relative_parameter_drift(parameters, snapshot_full))
    accepted = (
        aborted is None
        and drift <= float(args.max_relative_parameter_drift)
    )
    if not accepted:
        HYBRID._restore_parameters(parameters, snapshot_full)
    frozen_after = UPD.frozen_surface_sha256(theta, scope)
    if frozen_before != frozen_after:
        raise RuntimeError("frozen surface drifted during NFT training")

    output.mkdir(parents=True, exist_ok=True)
    checkpoint = None
    if accepted:
        payload = UPD.checkpoint_payload(
            theta,
            parent_checkpoint_sha256=start_sha,
            pretrained_checkpoint_sha256=ref_sha,
            round_index=1,
            alpha=0.0,
            exposure_passes=passes,
            optimizer_scope=scope,
        )
        payload["nft"] = {
            "beta": float(args.beta),
            "coupled_x0": bool(args.coupled_x0),
            "lambda_safe": bool(args.lambda_safe),
        }
        checkpoint = output / "checkpoint_r1.pt"
        torch.save(payload, checkpoint)

    marker = {
        "status": STATUS,
        "version": VERSION,
        "protocol_amendment": PROTOCOL_AMENDMENT,
        "accepted": bool(accepted),
        "aborted": aborted,
        "groups": len(groups),
        "steps": step_index,
        "beta": float(args.beta),
        "coupled_x0": bool(args.coupled_x0),
        "lambda_safe": bool(args.lambda_safe),
        "progress_temperature": PROGRESS_TEMPERATURE,
        "optimizer_scope": scope,
        "trainable_parameters": int(sum(p.numel() for p in parameters)),
        "trainable_names": list(trainable_names),
        "start_checkpoint_sha256": start_sha,
        "reference_checkpoint_sha256": ref_sha,
        "relative_parameter_drift": drift,
        "per_pass": per_pass,
        "token_audit": token_audit,
        "snapshots": snapshot_records or None,
        "checkpoint": (
            None if checkpoint is None
            else {"path": str(checkpoint), "sha256": PRED.sha256_file(checkpoint)}
        ),
    }
    PRED._write_json(output / "NFT_TRAIN_COMPLETE.json", marker)
    print(json.dumps({
        "status": STATUS, "accepted": bool(accepted), "steps": step_index,
        "drift": drift,
        "pos_branch_last_pass": (
            per_pass[-1].get("pos_branch_loss") if per_pass else None
        ),
        "neg_branch_last_pass": (
            per_pass[-1].get("neg_branch_loss") if per_pass else None
        ),
    }))
    return marker


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--checkpoint", required=True)
    value.add_argument("--expected-checkpoint-sha256", required=True)
    value.add_argument("--reference-checkpoint", required=True)
    value.add_argument("--expected-reference-checkpoint-sha256", required=True)
    value.add_argument("--groups", action="append", required=True)
    value.add_argument("--raw-manifest", action="append", default=[])
    value.add_argument("--context-path", choices=("raw", "stored"), default="raw")
    value.add_argument("--optimizer-scope",
                       choices=sorted(UPD.DECLARED_TRAINABLE_SURFACES),
                       default="trunk_head_and_projection")
    value.add_argument("--beta", type=float, default=1.0)
    value.add_argument("--coupled-x0", action="store_true")
    value.add_argument("--lambda-safe", action="store_true")
    value.add_argument("--max-groups", type=int, default=0)
    value.add_argument("--groups-per-batch", type=int, default=2)
    value.add_argument("--learning-rate", type=float, default=1.0e-5)
    value.add_argument("--exposure-passes", type=int, default=1)
    value.add_argument("--grad-clip-norm", type=float, default=1.0)
    value.add_argument("--max-relative-parameter-drift", type=float, default=0.25)
    value.add_argument("--update-seed", type=int, default=2)
    value.add_argument("--snapshot-every", type=int, default=0)
    value.add_argument("--lru-shards", type=int, default=RDS.DEFAULT_LRU_SHARDS)
    value.add_argument("--raw-audit-atol", type=float, default=5.0e-3)
    value.add_argument("--raw-audit-every", type=int, default=500)
    value.add_argument("--output", required=True)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--physical-gpu", type=int, required=True)
    return value


def main(argv=None) -> int:
    run(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
