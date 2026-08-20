"""Paired contrastive (DPO-style) CFM post-training on same-context pairs.

Targets the diagnosed blind spot directly: plain CFM fine-tuning raises the
density of safe demonstrations but moves the same-context tail mass on
unsafe proposals too slowly, and single-sample temperature-1 deployment
compounds that per-step tail risk over the episode. Given pairs (context,
pos_action = executed exact positive, neg_action = exact negative from the
SAME B block, harvested by ``sfm_hp100_pair_harvest``), each optimizer step
minimizes, with a frozen reference policy (the training start point):

    margin_i = (Lp_theta_i - Lp_ref_i) - (Ln_theta_i - Ln_ref_i)
    L = mean_i[ -log sigmoid(-beta * margin_i) ] + w_anchor * mean_i[Lp_theta_i]

where Lp/Ln are per-sample CFM losses of the positive/negative action at the
shared context. Driving ``margin`` negative moves conditional density from
the unsafe candidate onto the safe one *relative to the reference*, which is
exactly a same-context tail-mass transfer; the anchor term keeps the plain
demonstration-likelihood pressure so the policy does not trade its base
competence for pure separation.

Noise discipline: all four CFM evaluations of one batch (pos/neg x
theta/ref) consume the identical (x0, tau) draws — the global RNG is
re-seeded with the same per-step counter seed before each call — so the
pairwise comparison is low-variance by construction.

Everything is additive: trainable-surface selection, frozen-surface digests,
the raw context provider, and the strict checkpoint payload are imported
from the existing audited modules; none of them is modified.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

import _paths  # noqa: F401
import grid_policy_sfm_hp100 as GPS
import sfm_hp100_ball_adapter as PORT
import sfm_hp100_ball_launch as BASE
import sfm_hp100_exhaustive_hybrid as HYBRID
import sfm_hp100_expansion_update as UPD
import sfm_hp100_pair_harvest as PAIRS
import sfm_hp100_predictive_execution as PRED
import sfm_hp100_raw_obs_dataset as RDS

VERSION = "sfm_hp100_dpo_train_v1"
STATUS = "SFM2_DPO_TRAIN_COMPLETE"


def dpo_batch_losses(
    theta: PORT.HP100ExpansionPolicy,
    reference: PORT.HP100ExpansionPolicy,
    theta_contexts: torch.Tensor,
    ref_contexts: torch.Tensor,
    pos_actions: torch.Tensor,
    neg_actions: torch.Tensor,
    *,
    beta: float,
    anchor_weight: float,
    noise_seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """One batch objective with the shared-noise discipline."""

    def _seed() -> None:
        HYBRID._set_step_seed(int(noise_seed), device)

    _seed()
    lp_theta = theta.cfm_loss(theta_contexts, pos_actions, reduction="none")
    _seed()
    ln_theta = theta.cfm_loss(theta_contexts, neg_actions, reduction="none")
    with torch.no_grad():
        _seed()
        lp_ref = reference.cfm_loss(ref_contexts, pos_actions, reduction="none")
        _seed()
        ln_ref = reference.cfm_loss(ref_contexts, neg_actions, reduction="none")
    margin = (lp_theta - lp_ref) - (ln_theta - ln_ref)
    # -log sigmoid(-beta * margin) == softplus(beta * margin), numerically safe.
    dpo = F.softplus(float(beta) * margin).mean()
    anchor = lp_theta.mean()
    total = dpo + float(anchor_weight) * anchor
    audit = {
        "dpo_loss": float(dpo.detach()),
        "anchor_loss": float(anchor.detach()),
        "margin_mean": float(margin.detach().mean()),
        "margin_negative_fraction": float((margin.detach() < 0).float().mean()),
        "lp_theta": float(lp_theta.detach().mean()),
        "ln_theta": float(ln_theta.detach().mean()),
        "lp_ref": float(lp_ref.mean()),
        "ln_ref": float(ln_ref.mean()),
    }
    return total, audit


def _stack(pairs, key) -> torch.Tensor:
    return torch.stack([pair[key] for pair in pairs])


def run(args) -> dict:
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing dpo-train output: {output}")
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

    pair_payloads = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in args.pairs
    ]
    for payload in pair_payloads:
        if payload.get("status") != PAIRS.STATUS:
            raise RuntimeError("input is not a pair archive")
    pairs = [pair for payload in pair_payloads for pair in payload["pairs"]]
    if float(args.subset_fraction) < 1.0:
        rng = random.Random(7)
        rng.shuffle(pairs)
        pairs = pairs[: max(1, int(len(pairs) * float(args.subset_fraction)))]
    if not pairs:
        raise RuntimeError("no pairs to train on")

    provider = None
    token_audit = None
    if args.context_path == "raw":
        if not args.raw_manifest:
            raise ValueError("--context-path raw requires --raw-manifest")
        dataset = RDS.RawObsDataset(
            args.raw_manifest, lru_shards=int(args.lru_shards),
        )
        dataset.assert_joined(pairs)
        audit_rows = pairs[:: max(1, int(args.raw_audit_every))]
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
            "stored-context DPO training only supports the trunk_and_head "
            "scope; wider scopes need --context-path raw"
        )

    optimizer = torch.optim.Adam(parameters, lr=float(args.learning_rate))
    snapshot_full = HYBRID._parameter_snapshot(parameters)
    batch_size = int(args.batch_size)
    passes = int(args.exposure_passes)
    order_rng = random.Random(
        HYBRID._counter_seed(int(args.update_seed), "dpo_order")
    )

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
        payload["dpo"] = {
            "beta": float(args.beta),
            "anchor_weight": float(args.anchor_weight),
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
        indices = list(range(len(pairs)))
        order_rng.shuffle(indices)
        pass_audit = defaultdict(list)
        for start in range(0, len(indices), batch_size):
            batch = [pairs[i] for i in indices[start:start + batch_size]]
            ref_contexts = _stack(batch, "context").to(device)
            if provider is not None:
                theta_contexts = provider(batch, device)
            else:
                theta_contexts = ref_contexts
            pos_actions = _stack(batch, "pos_action").to(device)
            neg_actions = _stack(batch, "neg_action").to(device)
            noise_seed = HYBRID._counter_seed(
                int(args.update_seed), "dpo_noise", pass_index, step_index,
            )
            total, audit = dpo_batch_losses(
                theta, reference, theta_contexts, ref_contexts,
                pos_actions, neg_actions,
                beta=float(args.beta),
                anchor_weight=float(args.anchor_weight),
                noise_seed=noise_seed, device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            grad_norm = HYBRID._gradient_norm(parameters)
            torch.nn.utils.clip_grad_norm_(
                parameters, float(args.grad_clip_norm),
            )
            optimizer.step()
            step_index += 1
            if not HYBRID._finite_parameters(parameters):
                aborted = f"non-finite parameters at step {step_index}"
                break
            for key, value in audit.items():
                pass_audit[key].append(value)
            pass_audit["grad_norm"].append(float(grad_norm))
            if snapshot_every and step_index % snapshot_every == 0:
                save_snapshot(step_index, {
                    "dpo_loss_running_mean": float(
                        sum(pass_audit["dpo_loss"]) / len(pass_audit["dpo_loss"])
                    ),
                    "margin_mean_running": float(
                        sum(pass_audit["margin_mean"])
                        / len(pass_audit["margin_mean"])
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
        raise RuntimeError("frozen surface drifted during DPO training")

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
        payload["dpo"] = {
            "beta": float(args.beta),
            "anchor_weight": float(args.anchor_weight),
        }
        checkpoint = output / "checkpoint_r1.pt"
        torch.save(payload, checkpoint)

    marker = {
        "status": STATUS,
        "version": VERSION,
        "accepted": bool(accepted),
        "aborted": aborted,
        "pairs": len(pairs),
        "steps": step_index,
        "beta": float(args.beta),
        "anchor_weight": float(args.anchor_weight),
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
    PRED._write_json(output / "DPO_TRAIN_COMPLETE.json", marker)
    print(json.dumps({
        "status": STATUS, "accepted": bool(accepted), "steps": step_index,
        "drift": drift,
        "margin_mean_last_pass": (
            per_pass[-1].get("margin_mean") if per_pass else None
        ),
    }))
    return marker


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--checkpoint", required=True)
    value.add_argument("--expected-checkpoint-sha256", required=True)
    value.add_argument("--reference-checkpoint", required=True)
    value.add_argument("--expected-reference-checkpoint-sha256", required=True)
    value.add_argument("--pairs", action="append", required=True)
    value.add_argument("--raw-manifest", action="append", default=[])
    value.add_argument("--context-path", choices=("raw", "stored"), default="raw")
    value.add_argument("--optimizer-scope",
                       choices=sorted(UPD.DECLARED_TRAINABLE_SURFACES),
                       default="trunk_head_and_projection")
    value.add_argument("--beta", type=float, default=1.0)
    value.add_argument("--anchor-weight", type=float, default=0.5)
    value.add_argument("--subset-fraction", type=float, default=1.0)
    value.add_argument("--learning-rate", type=float, default=1.0e-5)
    value.add_argument("--batch-size", type=int, default=64)
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
