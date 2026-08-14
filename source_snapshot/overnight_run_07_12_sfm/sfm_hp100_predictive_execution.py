"""Always-on uncertainty acquisition with prediction-aware progress execution.

This default-off diagnostic keeps the HP100 flow policy and exact paper-SOCP
verifier unchanged.  At every replan it generates K=64 flow windows, acquires
B=32 using the frozen-RBF uncertainty mechanism, and executes one exact-positive
window.  The selector is deliberately weight-free:

    max lexicographically (H10 goal progress, predicted clearance,
                           selected uncertainty, -candidate index).

Exact positivity already includes the all-pedestrian constant-velocity
collision test.  Predicted clearance is therefore an audit/tie-break, not a
second eligibility gate.  If B contains no exact positive, the state is held
fixed and the Gaussian flow base scale increases 1.0, 1.1, ... .  Exhausting
the declared attempts produces one resolved exact-negative counterfactual and
terminates that lineage without executing it.

The module gathers evidence only; it never updates a checkpoint or GP buffer.
It is the frozen mechanism handoff for a later positive-minus-alpha-negative
Safe Flow Expansion study.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import torch

import grid_policy_sfm_hp100 as GPS
import sfm_hp100_ball_adapter as PORT
import sfm_hp100_ball_launch as BASE
from sfm_hp100_ball_core.expansion import (
    calibrate_fixed_beta,
    mean_pairwise_lengthscale,
    normalized_ess,
)
import sfm_hp100_early_acquisition as ACQ
import sfm_hp100_exhaustive_hybrid as HYBRID
import sfm_scene as SS


VERSION = "sfm_hp100_predictive_execution_v1"
STATUS = "SFM_HP100_PREDICTIVE_EXECUTION_COMPLETE"
TRACE_STATUS = "SFM_HP100_PREDICTIVE_EXECUTION_TRACE"
K = 64
B = 32


@dataclass(frozen=True)
class PredictiveConfig:
    gammas: tuple[float, ...] = (0.1, 0.3, 0.5, 1.0)
    lineages_per_gamma: int = 2
    max_steps: int = 180
    max_attempts: int = 32
    base_std_start: float = 1.0
    base_std_step: float = 0.1
    K: int = K
    B: int = B
    ess_target: float = 0.1
    rbf_noise: float = 1.0e-2
    seed: int = 41

    def validate(self) -> None:
        if not self.gammas or self.lineages_per_gamma < 1:
            raise ValueError("declare at least one gamma and lineage")
        if self.max_steps < 1 or self.max_attempts < 1:
            raise ValueError("step and attempt limits must be positive")
        if (self.K, self.B) != (K, B):
            raise ValueError("predictive execution fixes K=64 and B=32")
        if not 1.0 / self.K <= self.ess_target <= 1.0:
            raise ValueError("ESS target must lie in [1/K,1]")
        if self.base_std_start != 1.0 or self.base_std_step != 0.1:
            raise ValueError("retry schedule must be 1.0, 1.1, ...")
        if self.rbf_noise <= 0.0:
            raise ValueError("RBF noise must be positive")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _model_state_sha256(module: torch.nn.Module) -> str:
    return HYBRID._model_state_sha256(module)


def _lineage_keys(config: PredictiveConfig) -> tuple[HYBRID.LineageKey, ...]:
    return tuple(
        HYBRID.LineageKey(float(gamma), replica)
        for gamma in config.gammas
        for replica in range(config.lineages_per_gamma)
    )


def _reset_states(
    task: PORT.SFMHP100ExpansionTask,
    keys: Sequence[HYBRID.LineageKey],
    config: PredictiveConfig,
) -> dict[HYBRID.LineageKey, PORT.SFMState]:
    return {
        key: task.reset(
            key.gamma,
            episode=int(round(key.gamma * 10_000)) * config.lineages_per_gamma
            + key.replica,
            seed=HYBRID._scene_seed(config.seed, key.replica),
        )
        for key in keys
    }


def prediction_metrics(
    task: PORT.SFMHP100ExpansionTask,
    context: torch.Tensor,
    candidate: torch.Tensor,
) -> dict:
    """Independent CV audit for one H10 candidate at the current replan."""
    robot, ped_xy, ped_vel = task.decode_context(context)
    controls = np.asarray(candidate.detach().cpu(), np.float32).reshape(-1, 2)
    states = PORT.clipped_plan_states(robot, controls)
    segment = states[:, :2]
    pedestrians = PORT.VERIFY.predict_pedestrians(
        ped_xy, ped_vel, H=len(controls),
    )
    if pedestrians.shape[1]:
        distance = np.linalg.norm(
            segment[:, None, :] - pedestrians, axis=2,
        ) - float(SS.R_PED)
        clearance = float(distance[1:].min())
        sensed = np.linalg.norm(
            np.asarray(ped_xy, float) - np.asarray(robot[:2], float)[None],
            axis=1,
        ) - float(SS.R_PED) <= float(SS.R_SENSE)
        entering = (distance <= float(SS.R_SENSE)).any(axis=0)
    else:
        clearance = float("inf")
        sensed = np.zeros(0, bool)
        entering = np.zeros(0, bool)
    goal = np.asarray(SS.GOAL, float)
    initial_distance = float(np.linalg.norm(np.asarray(robot[:2], float) - goal))
    return {
        "predicted_min_clearance": clearance,
        "predicted_collision_free": bool(clearance >= -1.0e-9),
        "sensed_pedestrians": int(sensed.sum()),
        "sensed_or_predicted_entering_pedestrians": int((sensed | entering).sum()),
        "H10_goal_progress": float(
            initial_distance - np.linalg.norm(segment[-1] - goal)
        ),
        "one_step_goal_progress": float(
            initial_distance - np.linalg.norm(segment[1] - goal)
        ),
    }


def select_predictive_progress(
    results: Sequence,
    audits: Sequence[dict],
    selected_sigma: Sequence[float],
) -> int | None:
    """Select an exact-positive candidate without a hand-tuned scalar cost."""
    if not (len(results) == len(audits) == len(selected_sigma)):
        raise ValueError("selector inputs differ in length")
    eligible = []
    for index, (result, audit) in enumerate(zip(results, audits)):
        if result.error or not result.valid or not result.progress_eligible:
            continue
        if not audit["predicted_collision_free"]:
            raise RuntimeError(
                "exact-positive candidate failed its duplicate CV collision audit"
            )
        if not math.isclose(
            float(result.progress), float(audit["H10_goal_progress"]),
            rel_tol=0.0, abs_tol=2.0e-6,
        ):
            raise RuntimeError("verifier and selector H10 progress differ")
        eligible.append(index)
    if not eligible:
        return None
    return min(eligible, key=lambda index: (
        -float(results[index].progress),
        -float(audits[index]["predicted_min_clearance"]),
        -float(selected_sigma[index]),
        int(index),
    ))


def _negative_counterfactual(
    results: Sequence,
    audits: Sequence[dict],
) -> int:
    resolved = [index for index, result in enumerate(results) if not result.error]
    if not resolved or any(results[index].valid for index in resolved):
        raise ValueError("terminal counterfactual requires an all-negative resolved B")
    return min(resolved, key=lambda index: (
        -float(results[index].progress),
        -float(audits[index]["predicted_min_clearance"]),
        int(index),
    ))


def _sample_record(
    *,
    role: str,
    key: HYBRID.LineageKey,
    state: PORT.SFMState,
    step: int,
    attempt: int,
    context: torch.Tensor,
    candidate: torch.Tensor,
    flow_base: torch.Tensor,
    result,
    audit: dict,
    executed: bool,
    negative_reason: str | None = None,
) -> dict:
    if role not in {"positive", "negative"}:
        raise ValueError(f"unknown predictive sample role {role!r}")
    if role == "positive" and (not result.valid or negative_reason is not None):
        raise ValueError("positive sample must retain an exact-positive label")
    if role == "negative":
        declared = {"all_negative_nvp", "realized_collision", "realized_oob"}
        if negative_reason not in declared:
            raise ValueError("negative sample lacks a declared failure reason")
        if bool(result.valid) != bool(negative_reason != "all_negative_nvp"):
            raise ValueError("negative reason disagrees with exact verifier label")
    return {
        "role": role,
        "gamma": float(key.gamma),
        "replica": int(key.replica),
        "lineage": key.label,
        "scenario_id": int(state.scenario_id),
        "step": int(step),
        "attempt": int(attempt),
        "executed": bool(executed),
        "negative_reason": negative_reason,
        "context": context.detach().cpu().to(torch.float32).clone(),
        "candidate": candidate.detach().cpu().to(torch.float32).clone(),
        "flow_base": flow_base.detach().cpu().to(torch.float32).clone(),
        "verification": HYBRID._verification_row(result),
        "prediction_audit": dict(audit),
    }


def _event(
    key: HYBRID.LineageKey,
    state: PORT.SFMState,
    context: torch.Tensor,
    step: int,
    task: PORT.SFMHP100ExpansionTask,
) -> dict:
    _, ped_xy, ped_vel = task.decode_context(context)
    return {
        "gamma": float(key.gamma),
        "replica": int(key.replica),
        "lineage": key.label,
        "scenario_id": int(state.scenario_id),
        "step": int(step),
        "context": context.detach().cpu().to(torch.float32).clone(),
        "state_before": np.asarray(state.robot, np.float32).copy(),
        "state_after": np.asarray(state.robot, np.float32).copy(),
        "ped_xy": np.asarray(ped_xy, np.float32).copy(),
        "ped_vel": np.asarray(ped_vel, np.float32).copy(),
        "attempts": [],
        "executed_role": None,
        "terminal": None,
    }


@torch.inference_mode()
def gather_predictive(
    adapter: PORT.HP100ExpansionPolicy,
    reference: PORT.HP100ExpansionPolicy,
    task: PORT.SFMHP100ExpansionTask,
    *,
    keys: Sequence[HYBRID.LineageKey],
    config: PredictiveConfig,
    lengthscale: float,
    support_by_gamma: dict[float, torch.Tensor],
    verifier: HYBRID._OrderedSidecarVerifier,
) -> dict:
    """Run one no-update always-on K64/B32 acquisition transaction."""
    config.validate()
    states = _reset_states(task, keys, config)
    active = set(keys)
    samples = []
    events = []
    timers = defaultdict(float)
    posterior = HYBRID._posterior_by_gamma(
        config, float(lengthscale), support_by_gamma,
    )

    for step in range(config.max_steps):
        ordered = sorted(active)
        if not ordered:
            break
        pending = []
        event_by_key = {}
        for key in ordered:
            context = task.context(states[key], key.gamma)
            current = _event(key, states[key], context, step, task)
            events.append(current)
            event_by_key[key] = current
            pending.append({
                "key": key, "context": context, "state": states[key],
            })
        chosen = {}
        chosen_payload = {}
        final_all_negative = {}

        for attempt in range(config.max_attempts):
            if not pending:
                break
            base_std = config.base_std_start + config.base_std_step * attempt
            contexts = [row["context"] for row in pending]
            seeds = [
                HYBRID._sampling_seed(
                    config.seed, "predictive_always_on", row["key"], step,
                    microcycle=0, attempt=attempt,
                )
                for row in pending
            ]
            started = time.perf_counter()
            plans_by_context, bases_by_context, _, generators = ACQ._sample_blocks(
                adapter, contexts, seeds, K=config.K, flow_base_std=base_std,
            )
            feature_blocks = HYBRID._frozen_reference_features(
                reference, contexts, plans_by_context, bases_by_context,
            )
            timers["flow_and_frozen_phi"] += time.perf_counter() - started

            prepared = []
            for row, plans, bases, features, generator in zip(
                pending, plans_by_context, bases_by_context,
                feature_blocks, generators,
            ):
                gp = posterior[float(row["key"].gamma)]
                marginal_sigma = gp.sigma(features)
                beta = calibrate_fixed_beta(
                    [marginal_sigma], target=config.ess_target,
                )
                selected, selected_sigma, conditional_ess = gp.acquire(
                    features, config.B, beta, generator,
                )
                prepared.append({
                    **row,
                    "plans": plans,
                    "bases": bases,
                    "features": features,
                    "selected": selected,
                    "selected_sigma": selected_sigma,
                    "conditional_ess": conditional_ess,
                    "marginal_sigma": marginal_sigma,
                    "beta": float(beta),
                    "queried": plans[selected],
                    "queried_bases": bases[selected],
                })

            started = time.perf_counter()
            verified = verifier.verify_many([
                (row["context"], row["queried"], row["key"].gamma)
                for row in prepared
            ])
            timers["exact_B32_verification"] += time.perf_counter() - started
            next_pending = []
            for row, verified_block in zip(prepared, verified):
                results, sidecars = verified_block
                if len(results) != config.B or any(result.error for result in results):
                    raise RuntimeError("predictive verifier did not resolve B=32")
                audits = [
                    prediction_metrics(task, row["context"], candidate)
                    for candidate in row["queried"]
                ]
                selected_sigma = list(map(float, row["selected_sigma"]))
                predictive_local = select_predictive_progress(
                    results, audits, selected_sigma,
                )
                max_margin_local = HYBRID._chosen_max_margin(results)
                positive_first16 = sum(bool(result.valid) for result in results[:16])
                positive_B32 = sum(bool(result.valid) for result in results)
                attempt_row = {
                    "attempt": int(attempt),
                    "base_std": float(base_std),
                    "candidate_ids": list(map(int, row["selected"])),
                    "K_segments": np.stack([
                        HYBRID._segment(row["state"], candidate)
                        for candidate in row["plans"]
                    ]).astype(np.float32, copy=False),
                    "B_segments": np.stack([
                        HYBRID._segment(row["state"], candidate)
                        for candidate in row["queried"]
                    ]).astype(np.float32, copy=False),
                    "verification": [
                        HYBRID._verification_row(result) for result in results
                    ],
                    "prediction_audits": audits,
                    "predictive_local": (
                        None if predictive_local is None else int(predictive_local)
                    ),
                    "max_margin_reference_local": (
                        None if max_margin_local is None else int(max_margin_local)
                    ),
                    "selected_sigma": selected_sigma,
                    "marginal_sigma": list(map(float, row["marginal_sigma"])),
                    "conditional_ess": list(map(float, row["conditional_ess"])),
                    "beta": float(row["beta"]),
                    "marginal_ESS_over_K": float(normalized_ess(
                        row["marginal_sigma"], row["beta"],
                    )),
                    "positive_first16": int(positive_first16),
                    "positive_B32": int(positive_B32),
                }
                event_by_key[row["key"]]["attempts"].append(attempt_row)
                final_all_negative[row["key"]] = (
                    row, results, sidecars, audits, attempt_row,
                )
                if predictive_local is None:
                    next_pending.append({
                        key: row[key]
                        for key in ("key", "context", "state")
                    })
                    continue

                result = results[predictive_local]
                candidate = row["queried"][predictive_local]
                sidecar = HYBRID._validated_sidecar(
                    task, row["context"], candidate, row["key"].gamma,
                    result, sidecars[predictive_local],
                )
                attempt_row["predictive_sidecar"] = sidecar
                chosen[row["key"]] = candidate
                chosen_payload[row["key"]] = {
                    "row": row, "result": result,
                    "audit": audits[predictive_local],
                    "local": int(predictive_local),
                    "attempt": int(attempt),
                }
            pending = next_pending

        exhausted = {row["key"] for row in pending}
        for key in ordered:
            current = event_by_key[key]
            if key in exhausted:
                row, results, _sidecars, audits, attempt_row = final_all_negative[key]
                negative_local = _negative_counterfactual(results, audits)
                attempt_row["negative_counterfactual_local"] = int(negative_local)
                samples.append(_sample_record(
                    role="negative", key=key, state=row["state"], step=step,
                    attempt=int(attempt_row["attempt"]), context=row["context"],
                    candidate=row["queried"][negative_local],
                    flow_base=row["queried_bases"][negative_local],
                    result=results[negative_local], audit=audits[negative_local],
                    executed=False, negative_reason="all_negative_nvp",
                ))
                current["terminal"] = "nvp"
                active.remove(key)
                continue
            states[key] = task.advance(states[key], chosen[key])
            current["state_after"] = np.asarray(
                states[key].robot, np.float32,
            ).copy()
            terminal = task.terminal(states[key])
            payload = chosen_payload[key]
            realized_failure = terminal in {"COLLISION", "OOB"}
            role = "negative" if realized_failure else "positive"
            negative_reason = (
                f"realized_{str(terminal).lower()}" if realized_failure else None
            )
            samples.append(_sample_record(
                role=role, key=key, state=payload["row"]["state"], step=step,
                attempt=int(payload["attempt"]),
                context=payload["row"]["context"], candidate=chosen[key],
                flow_base=payload["row"]["queried_bases"][payload["local"]],
                result=payload["result"], audit=payload["audit"],
                executed=True, negative_reason=negative_reason,
            ))
            current["executed_role"] = (
                negative_reason if realized_failure else "positive"
            )
            if terminal is not None:
                current["terminal"] = str(terminal).lower()
                active.remove(key)

    for key in sorted(active):
        rows = [row for row in events if row["lineage"] == key.label]
        if rows:
            rows[-1]["terminal"] = "timeout"
        active.remove(key)

    outcomes = {}
    for key in keys:
        rows = [row for row in events if row["lineage"] == key.label]
        outcomes[key.label] = {
            "gamma": float(key.gamma),
            "replica": int(key.replica),
            "scenario_id": int(states[key].scenario_id),
            "status": rows[-1]["terminal"] if rows else "empty",
            "executed_steps": int(states[key].steps),
            "net_goal_progress": float(
                np.linalg.norm(SS.GOAL)
                - np.linalg.norm(np.asarray(states[key].robot)[:2] - SS.GOAL)
            ),
        }
    return {
        "outcomes": outcomes,
        "samples": samples,
        "events": events,
        "sample_counts": dict(sorted(Counter(
            row["role"] for row in samples
        ).items())),
        "timers_seconds": dict(sorted(timers.items())),
    }


def summarize(result: dict, config: PredictiveConfig) -> dict:
    rows = []
    for event in result["events"]:
        for attempt in event["attempts"]:
            verifications = attempt["verification"]
            chosen = attempt["predictive_local"]
            reference = attempt["max_margin_reference_local"]
            rows.append({
                "gamma": float(event["gamma"]),
                "lineage": str(event["lineage"]),
                "step": int(event["step"]),
                "attempt": int(attempt["attempt"]),
                "positive_first16": int(attempt["positive_first16"]),
                "positive_B32": int(attempt["positive_B32"]),
                "base_std": float(attempt["base_std"]),
                "chosen": chosen,
                "reference": reference,
                "choice_changed": (
                    chosen is not None and reference is not None
                    and int(chosen) != int(reference)
                ),
                "chosen_progress": (
                    None if chosen is None
                    else float(verifications[int(chosen)]["H10_progress"])
                ),
                "reference_progress": (
                    None if reference is None
                    else float(verifications[int(reference)]["H10_progress"])
                ),
                "chosen_clearance": (
                    None if chosen is None else float(
                        attempt["prediction_audits"][int(chosen)][
                            "predicted_min_clearance"
                        ]
                    )
                ),
                "reference_clearance": (
                    None if reference is None else float(
                        attempt["prediction_audits"][int(reference)][
                            "predicted_min_clearance"
                        ]
                    )
                ),
                "chosen_step_margin": (
                    None if chosen is None else float(
                        verifications[int(chosen)]["step_margin"]
                    )
                ),
                "reference_step_margin": (
                    None if reference is None else float(
                        verifications[int(reference)]["step_margin"]
                    )
                ),
            })

    def aggregate(block: Sequence[dict]) -> dict:
        def mean(field):
            values = [float(row[field]) for row in block if row[field] is not None]
            return None if not values else float(np.mean(values))

        comparable = [
            row for row in block
            if row["chosen"] is not None and row["reference"] is not None
        ]

        return {
            "attempts": len(block),
            "mean_positive_first16": mean("positive_first16"),
            "median_positive_first16": (
                None if not block else float(np.median([
                    row["positive_first16"] for row in block
                ]))
            ),
            "zero_positive_first16_fraction": (
                None if not block else float(np.mean([
                    row["positive_first16"] == 0 for row in block
                ]))
            ),
            "mean_positive_B32": mean("positive_B32"),
            "median_positive_B32": (
                None if not block else float(np.median([
                    row["positive_B32"] for row in block
                ]))
            ),
            "zero_positive_B32_fraction": (
                None if not block else float(np.mean([
                    row["positive_B32"] == 0 for row in block
                ]))
            ),
            "selector_change_fraction": (
                None if not comparable else float(np.mean([
                    row["choice_changed"] for row in comparable
                ]))
            ),
            "predictive_H10_progress": mean("chosen_progress"),
            "max_margin_H10_progress": mean("reference_progress"),
            "predictive_minus_max_margin_H10_progress": (
                None if mean("chosen_progress") is None else float(
                    mean("chosen_progress") - mean("reference_progress")
                )
            ),
            "predictive_clearance": mean("chosen_clearance"),
            "max_margin_clearance": mean("reference_clearance"),
            "predictive_step_margin": mean("chosen_step_margin"),
            "max_margin_step_margin": mean("reference_step_margin"),
            "mean_base_std": mean("base_std"),
        }

    per_gamma = {
        str(gamma): aggregate([
            row for row in rows if math.isclose(
                row["gamma"], float(gamma), rel_tol=0.0, abs_tol=1.0e-8,
            )
        ])
        for gamma in config.gammas
    }
    first_attempts = [row for row in rows if row["attempt"] == 0]
    final_keys = {
        (str(event["lineage"]), int(event["step"])): int(
            event["attempts"][-1]["attempt"]
        )
        for event in result["events"] if event["attempts"]
    }
    final_attempts = [
        row for row in rows
        if row["attempt"] == final_keys[(row["lineage"], row["step"])]
    ]
    per_gamma_final = {
        str(gamma): aggregate([
            row for row in final_attempts if math.isclose(
                row["gamma"], float(gamma), rel_tol=0.0, abs_tol=1.0e-8,
            )
        ])
        for gamma in config.gammas
    }
    status_counts = Counter(
        row["status"] for row in result["outcomes"].values()
    )
    return {
        "contexts": len(final_attempts),
        "retried_context_fraction": (
            None if not final_attempts else float(np.mean([
                row["attempt"] > 0 for row in final_attempts
            ]))
        ),
        "pooled": aggregate(rows),
        "pooled_first_attempt": aggregate(first_attempts),
        "pooled_final_attempt": aggregate(final_attempts),
        "per_gamma": per_gamma,
        "per_gamma_final_attempt": per_gamma_final,
        "outcome_counts": dict(sorted(status_counts.items())),
        "sample_counts": result["sample_counts"],
    }


def run(args) -> dict:
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    gpu_contract = BASE._gpu_contract(args.device, int(args.physical_gpu))
    checkpoint_sha = sha256_file(args.checkpoint)
    if checkpoint_sha != str(args.expected_checkpoint_sha256).lower():
        raise RuntimeError("checkpoint SHA256 mismatch")
    policy, payload = GPS.load_sfm_hp100_policy(
        args.checkpoint, device=args.device,
    )
    adapter = PORT.HP100ExpansionPolicy(policy).eval()
    reference_policy, _ = GPS.load_sfm_hp100_policy(
        args.checkpoint, device=args.device,
    )
    reference = PORT.HP100ExpansionPolicy(reference_policy).eval()
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    before = _model_state_sha256(adapter)

    config = PredictiveConfig(
        gammas=HYBRID._parse_gammas(args.gammas),
        lineages_per_gamma=int(args.lineages_per_gamma),
        max_steps=int(args.max_steps), max_attempts=int(args.max_attempts),
        ess_target=float(args.ess_target), seed=int(args.seed),
    )
    config.validate()
    features, calibration = BASE.calibration_features(
        reference, dataset_root=args.pretrain_dataset_root,
        expected_manifest_sha256=args.expected_pretrain_dataset_manifest_sha256,
        count=50, seed=config.seed, base_std=1.0,
        paired_noised_representation=True,
    )
    lengthscale = mean_pairwise_lengthscale(features)
    support = HYBRID._calibration_support_by_gamma(
        features, calibration, config.gammas,
        device=next(reference.parameters()).device,
    )
    task = PORT.SFMHP100ExpansionTask(
        scene_profile=args.scene_profile,
        scenario_start=int(args.scenario_start),
    ).attach_context_encoder(policy)
    keys = _lineage_keys(config)

    output.mkdir(parents=True)
    preflight = {
        "status": "SFM_HP100_PREDICTIVE_EXECUTION_PREFLIGHT_PASSED",
        "version": VERSION,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_scientific_status": payload.get("scientific_status"),
        "config": asdict(config),
        "scene": SS.scene_profile(args.scene_profile),
        "scenario_start": int(args.scenario_start),
        "gpu": gpu_contract,
        "calibration": {**calibration, "lengthscale": float(lengthscale)},
        "execution_rule": (
            "lexicographic exact-positive H10 progress, predicted clearance, "
            "selected uncertainty, candidate index"
        ),
        "prediction_note": (
            "CV collision-free is already inside exact y=1; clearance is an "
            "audit and tie-break, not an additional gate"
        ),
        "policy_trainable_parameters": 0,
    }
    _write_json(output / "PREFLIGHT.json", preflight)
    with HYBRID._OrderedSidecarVerifier(task, int(args.verifier_workers)) as verifier:
        gathered = gather_predictive(
            adapter, reference, task, keys=keys, config=config,
            lengthscale=float(lengthscale), support_by_gamma=support,
            verifier=verifier,
        )
    after = _model_state_sha256(adapter)
    if after != before:
        raise RuntimeError("no-update diagnostic changed the policy")
    summary = summarize(gathered, config)
    trace_path = output / "predictive_trace.pt"
    torch.save({
        "status": TRACE_STATUS, "version": VERSION,
        "preflight": preflight, **gathered,
    }, trace_path)
    samples_path = output / "predictive_samples.pt"
    torch.save({
        "status": "SFM_HP100_PREDICTIVE_SAMPLE_ARCHIVE",
        "semantics": (
            "one executed exact-positive per successful context unless its first "
            "action realizes collision/OOB; one nonexecuted resolved exact-negative "
            "counterfactual per exhausted lineage"
        ),
        "samples": gathered["samples"],
    }, samples_path)
    marker = {
        "status": STATUS, "version": VERSION,
        "preflight": preflight, "summary": summary,
        "policy_unchanged": True,
        "policy_state_sha256": before,
        "trace": {"path": str(trace_path), "sha256": sha256_file(trace_path)},
        "samples": {
            "path": str(samples_path), "sha256": sha256_file(samples_path),
        },
        "timers_seconds": gathered["timers_seconds"],
    }
    _write_json(output / "PREDICTIVE_EXECUTION_COMPLETE.json", marker)
    return marker


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--checkpoint", required=True)
    value.add_argument("--expected-checkpoint-sha256", required=True)
    value.add_argument("--pretrain-dataset-root", required=True)
    value.add_argument("--expected-pretrain-dataset-manifest-sha256", required=True)
    value.add_argument("--output", required=True)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--physical-gpu", type=int, default=3)
    value.add_argument("--scene-profile", default="double_density_velocity_ood")
    value.add_argument("--scenario-start", type=int, default=810_000)
    value.add_argument("--gammas", default="0.1,0.3,0.5,1.0")
    value.add_argument("--lineages-per-gamma", type=int, default=2)
    value.add_argument("--max-steps", type=int, default=180)
    value.add_argument("--max-attempts", type=int, default=32)
    value.add_argument("--ess-target", type=float, default=0.1)
    value.add_argument("--seed", type=int, default=41)
    value.add_argument("--verifier-workers", type=int, default=32)
    return value


def main(argv=None) -> int:
    result = run(parser().parse_args(argv))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
