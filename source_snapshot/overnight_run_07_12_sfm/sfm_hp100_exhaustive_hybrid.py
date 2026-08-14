"""Transactional HP100 exhaustive-hybrid qualification.

This default-off protocol preserves the canonical raw policy whenever its
single temperature-one H10 proposal is exact-positive (``P1``).  Only a raw
exact-negative proposal (``Dminus``) opens a K=64, RBF-acquired B=16 repair
search.  The exact-positive repair with maximum nominal one-step margin is
stored as ``P2`` and its first action is executed.  Repair attempts keep the
state and context fixed while their flow-base scale increases by 0.1.

Nothing enters a persistent archive or GP support until every declared
lineage passes an independent raw-only recheck.  This module is deliberately
separate from the vendored ball core: the core's K-first/NVP loop cannot
express a raw-first conditional repair transaction.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import pickle
import sys
import time
from typing import Sequence

import numpy as np
import torch

import _paths  # noqa: F401
import grid_policy_sfm_hp100 as GPS
import sfm_hp100_ball_adapter as PORT
import sfm_hp100_ball_approval_smoke as APPROVAL
from sfm_hp100_ball_core.expansion import (
    RBFPosterior,
    _counter_seed,
    calibrate_fixed_beta,
    mean_pairwise_lengthscale,
    normalized_ess,
)
import sfm_hp100_early_acquisition as ACQ
import sfm_hp100_ball_launch as BASE
import sfm_scene as SS


VERSION = "sfm_hp100_exhaustive_hybrid_v1"
TRACE_VERSION = "sfm_hp100_exhaustive_hybrid_trace_v1"
PREFLIGHT_STATUS = "SFM_HP100_EXHAUSTIVE_HYBRID_PREFLIGHT_PASSED"
COMPLETE_STATUS = "SFM_HP100_EXHAUSTIVE_HYBRID_COMMIT_COMPLETE"
INCOMPLETE_STATUS = "SFM_HP100_EXHAUSTIVE_HYBRID_QUALIFICATION_INCOMPLETE"
H = 10
K = 64
B = 16
DEFAULT_GAMMAS = (0.1, 0.3, 0.5, 1.0)


_SIDECAR_WORKER_TASK: PORT.SFMHP100ExpansionTask | None = None


def _initialize_sidecar_worker(task: PORT.SFMHP100ExpansionTask) -> None:
    global _SIDECAR_WORKER_TASK
    task.trace_sidecars = True
    _SIDECAR_WORKER_TASK = task
    torch.set_num_threads(1)


def _verify_task_with_sidecars(
    task: PORT.SFMHP100ExpansionTask,
    context: torch.Tensor,
    candidates: torch.Tensor,
    gamma: float,
) -> tuple[tuple, tuple[dict, ...]]:
    previous = task.trace_sidecars
    task.trace_sidecars = True
    try:
        results = tuple(task.verify(context, candidates, float(gamma)))
        sidecars = tuple(task.pop_trace_sidecars())
    finally:
        task.trace_sidecars = previous
    if len(results) != len(sidecars):
        raise RuntimeError("verifier result/sidecar counts differ")
    return results, sidecars


def _sidecar_worker(
    payload: tuple[np.ndarray, np.ndarray, float],
) -> tuple[tuple, tuple[dict, ...]]:
    if _SIDECAR_WORKER_TASK is None:
        raise RuntimeError("sidecar verifier worker was not initialized")
    context, candidates, gamma = payload
    return _verify_task_with_sidecars(
        _SIDECAR_WORKER_TASK,
        torch.from_numpy(context), torch.from_numpy(candidates), float(gamma),
    )


class _OrderedSidecarVerifier:
    """Ordered exact verifier returning geometry from the same solver call."""

    def __init__(self, task: PORT.SFMHP100ExpansionTask, workers: int):
        self.task = task
        self.workers = int(workers)
        self.pool: ProcessPoolExecutor | None = None
        if self.workers < 1:
            raise ValueError("verifier workers must be positive")
        if self.workers > 1:
            try:
                pickle.dumps(task)
            except Exception as error:
                raise TypeError("sidecar verifier task must be pickleable") from error

    def __enter__(self):
        return self

    def __exit__(self, *_exc_info) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None

    def verify_many(self, blocks):
        if self.workers == 1:
            return [
                _verify_task_with_sidecars(self.task, context, candidates, gamma)
                for context, candidates, gamma in blocks
            ]
        if self.pool is None:
            self.pool = ProcessPoolExecutor(
                max_workers=self.workers, mp_context=mp.get_context("spawn"),
                initializer=_initialize_sidecar_worker, initargs=(self.task,),
            )
        payloads = [
            (
                np.ascontiguousarray(context.detach().cpu().numpy()),
                np.ascontiguousarray(candidates.detach().cpu().numpy()),
                float(gamma),
            )
            for context, candidates, gamma in blocks
        ]
        return list(self.pool.map(_sidecar_worker, payloads, chunksize=1))


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _torch_save(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _sha256(path: str | Path) -> str:
    return APPROVAL.sha256_file(path)


def _load_input_checkpoint(
    path: str, expected_sha256: str, device: str, optimizer_scope: str,
):
    actual = _sha256(path)
    if actual != str(expected_sha256).lower():
        raise RuntimeError(f"checkpoint SHA256 mismatch: {actual} != {expected_sha256}")
    policy, payload = GPS.load_sfm_hp100_policy(path, device=device)
    status = payload.get("scientific_status")
    if status == "canonical_ID_promoted":
        if payload.get("pretrained_from_scratch") is not True:
            raise ValueError("canonical HP100 input is not from-scratch pretraining")
    elif status == "HP100_EXHAUSTIVE_HYBRID_ROUND1_COMMITTED":
        if payload.get("promotable") is not True:
            raise ValueError("hybrid parent checkpoint is not promotable")
        root = str(payload.get("pretrained_checkpoint_sha256", ""))
        if len(root) != 64:
            raise ValueError("hybrid parent lacks its root pretrained digest")
    else:
        raise ValueError(f"unsupported exhaustive-hybrid parent status {status!r}")
    adapter = PORT.HP100ExpansionPolicy(policy)
    if optimizer_scope == "last_block_and_head":
        trainable = adapter.expansion_optimizer_parameters(optimizer_scope)
        expected_count = 137_236
    elif optimizer_scope == "head_only":
        for parameter in adapter.parameters():
            parameter.requires_grad_(False)
        trainable = list(adapter.head.parameters())
        for parameter in trainable:
            parameter.requires_grad_(True)
        expected_count = 5_140
    else:
        raise ValueError(f"unsupported optimizer scope {optimizer_scope!r}")
    count = sum(parameter.numel() for parameter in trainable)
    if count != expected_count:
        raise RuntimeError(
            f"{optimizer_scope} parameter count drifted: {count} != {expected_count}"
        )
    names = sorted(
        name for name, parameter in adapter.named_parameters()
        if parameter.requires_grad
    )
    return adapter, payload, actual, names, count


def _model_state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _parameter_snapshot(parameters: Sequence[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [parameter.detach().cpu().clone() for parameter in parameters]


def _relative_parameter_drift(
    parameters: Sequence[torch.nn.Parameter], before: Sequence[torch.Tensor],
) -> float:
    numerator = 0.0
    denominator = 0.0
    for parameter, reference in zip(parameters, before):
        current = parameter.detach().cpu().to(torch.float64)
        reference64 = reference.to(torch.float64)
        numerator += float((current - reference64).square().sum())
        denominator += float(reference64.square().sum())
    return math.sqrt(numerator / max(denominator, 1.0e-24))


def _restore_parameters(
    parameters: Sequence[torch.nn.Parameter], before: Sequence[torch.Tensor],
) -> None:
    with torch.no_grad():
        for parameter, reference in zip(parameters, before):
            parameter.copy_(reference.to(parameter.device, parameter.dtype))


def _finite_parameters(parameters: Sequence[torch.nn.Parameter]) -> bool:
    return all(bool(torch.isfinite(parameter).all()) for parameter in parameters)


def _parse_gammas(value: str) -> tuple[float, ...]:
    output = tuple(float(item) for item in value.split(",") if item.strip())
    if not output or len(set(output)) != len(output):
        raise ValueError("gammas must be a nonempty unique comma-separated list")
    declared = {round(float(gamma), 8) for gamma in SS.GAMMAS}
    if any(round(float(gamma), 8) not in declared for gamma in output):
        raise ValueError(f"gammas must be drawn from {SS.GAMMAS}")
    return output


@dataclass(frozen=True, order=True)
class LineageKey:
    gamma: float
    replica: int

    @property
    def label(self) -> str:
        return f"g{self.gamma:.9g}:rep{self.replica:02d}"


@dataclass(frozen=True)
class HybridConfig:
    gammas: tuple[float, ...] = DEFAULT_GAMMAS
    lineages_per_gamma: int = 4
    max_steps: int = 180
    max_repair_batches: int = 32
    repair_base_std_start: float = 1.0
    repair_base_std_step: float = 0.1
    K: int = K
    B: int = B
    ess_target: float = 0.1
    rbf_noise: float = 1.0e-2
    p1_learning_rate: float = 1.0e-5
    p2_learning_rate: float = 1.0e-3
    negative_alpha_base: float = 1.0e-3
    batch_size: int = 64
    max_microcycles: int = 6
    max_relative_parameter_drift: float = 0.25
    gp_buffer_cap: int = 2_688
    seed: int = 2

    def validate(self) -> None:
        if not self.gammas or self.lineages_per_gamma < 1:
            raise ValueError("declare at least one gamma and lineage")
        if self.max_steps < 1 or self.max_repair_batches < 1:
            raise ValueError("step and repair limits must be positive")
        if (self.K, self.B) != (K, B):
            raise ValueError("exhaustive-hybrid qualification fixes K=64 and B=16")
        if not 1.0 / self.K <= self.ess_target <= 1.0:
            raise ValueError("ESS target must lie in [1/K,1]")
        if not 0.0 < self.rbf_noise:
            raise ValueError("RBF noise must be positive")
        if self.repair_base_std_start != 1.0 or self.repair_base_std_step != 0.1:
            raise ValueError("repair schedule must be 1.0, 1.1, ...")
        if not 0.0 < self.p1_learning_rate or not 0.0 < self.p2_learning_rate:
            raise ValueError("P1/P2 learning rates must be positive")
        if self.negative_alpha_base < 0.0 or self.batch_size < 1:
            raise ValueError("negative alpha and batch size are invalid")
        if self.max_microcycles < 1 or self.gp_buffer_cap < len(self.gammas):
            raise ValueError("microcycle and GP caps must be positive")
        if not 0.0 < self.max_relative_parameter_drift < 1.0:
            raise ValueError("relative drift gate must lie in (0,1)")


def _lineage_keys(config: HybridConfig) -> tuple[LineageKey, ...]:
    return tuple(
        LineageKey(float(gamma), replica)
        for gamma in config.gammas
        for replica in range(config.lineages_per_gamma)
    )


def _scene_seed(master_seed: int, replica: int) -> int:
    # Gamma is deliberately absent: matching replicas share one pedestrian scene.
    return _counter_seed(master_seed, "exhaustive_hybrid_scene", int(replica))


def _sampling_seed(
    master_seed: int,
    stream: str,
    key: LineageKey,
    step: int,
    *,
    microcycle: int | None = None,
    attempt: int | None = None,
) -> int:
    coordinates: list[int | str] = [stream, f"{key.gamma:.9g}", key.replica, step]
    if microcycle is not None:
        coordinates.append(microcycle)
    if attempt is not None:
        coordinates.append(attempt)
    return _counter_seed(master_seed, *coordinates)


def _verification_row(result) -> dict:
    return {
        "valid": bool(result.valid),
        "hp_eligible": bool(result.hp_eligible),
        "margin": float(result.margin),
        "native_cost": float(result.execution_cost),
        "H10_progress": float(result.progress),
        "progress_eligible": bool(result.progress_eligible),
        "error": bool(result.error),
        "step_margin": (
            None if result.step_margin is None else float(result.step_margin)
        ),
    }


def _validated_sidecar(
    task: PORT.SFMHP100ExpansionTask,
    context: torch.Tensor,
    candidate: torch.Tensor,
    gamma: float,
    expected,
    sidecar: dict,
) -> dict:
    exact = sidecar["result"]
    if bool(exact.get("y", 0)) != bool(expected.valid):
        raise RuntimeError("exact sidecar and verifier label differ")
    if not bool(exact.get("resolved", False)) == (not bool(expected.error)):
        raise RuntimeError("exact sidecar and verifier resolution differ")
    if not math.isfinite(float(expected.execution_cost)):
        raise RuntimeError("verifier cost is non-finite")
    _, ped_xy, ped_vel = task.decode_context(context)
    exact["pedestrian_prediction"] = PORT.VERIFY.predict_pedestrians(
        ped_xy, ped_vel, H=len(candidate),
    ).astype(np.float32, copy=False)
    return exact


def _record(
    *,
    group: str,
    key: LineageKey,
    scenario_id: int,
    microcycle: int,
    step: int,
    context: torch.Tensor,
    candidate: torch.Tensor,
    flow_base: torch.Tensor,
    result,
    base_std: float,
    repair_attempt: int | None,
) -> dict:
    if group not in {"P1", "P2", "Dminus"}:
        raise ValueError(f"unknown exhaustive-hybrid group {group!r}")
    if group in {"P1", "P2"} and not result.valid:
        raise ValueError(f"{group} must be exact-positive")
    if group == "Dminus" and (result.valid or result.error):
        raise ValueError("Dminus must be a resolved exact-negative raw proposal")
    return {
        "group": group,
        "gamma": float(key.gamma),
        "replica": int(key.replica),
        "lineage": key.label,
        "scenario_id": int(scenario_id),
        "microcycle": int(microcycle),
        "step": int(step),
        "context": context.detach().cpu().to(torch.float32).clone(),
        "candidate": candidate.detach().cpu().to(torch.float32).clone(),
        "flow_base": flow_base.detach().cpu().to(torch.float32).clone(),
        "verification": _verification_row(result),
        "base_std": float(base_std),
        "repair_attempt": (
            None if repair_attempt is None else int(repair_attempt)
        ),
    }


def _reset_states(
    task: PORT.SFMHP100ExpansionTask,
    keys: Sequence[LineageKey],
    config: HybridConfig,
) -> dict[LineageKey, PORT.SFMState]:
    return {
        key: task.reset(
            key.gamma,
            episode=int(round(key.gamma * 10_000)) * config.lineages_per_gamma
            + key.replica,
            seed=_scene_seed(config.seed, key.replica),
        )
        for key in keys
    }


def _chosen_max_margin(results: Sequence) -> int | None:
    eligible = [
        index for index, result in enumerate(results)
        if not result.error and result.valid and result.step_margin is not None
    ]
    if not eligible:
        return None
    # Exact full-H positivity is the only eligibility gate.  Native cost and
    # candidate index are deterministic tie-breakers, not extra gates.
    return min(eligible, key=lambda index: (
        -float(results[index].step_margin),
        float(results[index].execution_cost),
        int(index),
    ))


def _segment(state: PORT.SFMState, candidate: torch.Tensor) -> np.ndarray:
    return PORT.clipped_plan_states(
        state.robot, candidate.detach().cpu().numpy(),
    )[:, :2].astype(np.float32, copy=False)


def _new_trace_event(
    key: LineageKey,
    state: PORT.SFMState,
    *,
    microcycle: int,
    step: int,
    context: torch.Tensor,
    raw_candidate: torch.Tensor,
    raw_result,
    raw_sidecar: dict,
) -> dict:
    return {
        "gamma": float(key.gamma),
        "replica": int(key.replica),
        "lineage": key.label,
        "scenario_id": int(state.scenario_id),
        "microcycle": int(microcycle),
        "step": int(step),
        "context": context.detach().cpu().to(torch.float32).clone(),
        "state_before": np.asarray(state.robot, np.float32).copy(),
        "raw_controls": raw_candidate.detach().cpu().to(torch.float32).numpy(),
        "raw_segment": _segment(state, raw_candidate),
        "raw_verification": _verification_row(raw_result),
        "raw_sidecar": raw_sidecar,
        "repair_attempts": [],
        "executed_group": None,
        "state_after": np.asarray(state.robot, np.float32).copy(),
        "terminal": None,
    }


def _attach_scene_to_event(
    task: PORT.SFMHP100ExpansionTask,
    event: dict,
    context: torch.Tensor,
) -> None:
    _, ped_xy, ped_vel = task.decode_context(context)
    event["ped_xy"] = ped_xy
    event["ped_vel"] = ped_vel


def _posterior_by_gamma(
    config: HybridConfig,
    lengthscale: float,
    support_by_gamma: dict[float, torch.Tensor],
) -> dict[float, RBFPosterior]:
    output = {}
    for gamma in config.gammas:
        posterior = RBFPosterior(float(lengthscale), float(config.rbf_noise))
        support = support_by_gamma.get(float(gamma))
        if support is None or len(support) == 0:
            raise ValueError(f"gamma {gamma:g} lacks pretrained RBF support")
        posterior.set_buffer(support)
        output[float(gamma)] = posterior
    return output


def _calibration_support_by_gamma(
    features: torch.Tensor,
    calibration: dict,
    gammas: Sequence[float],
    *,
    device: torch.device,
) -> dict[float, torch.Tensor]:
    selection = list(calibration.get("selection", ()))
    if len(selection) != len(features):
        raise ValueError("calibration selection and feature rows differ")
    grouped: defaultdict[float, list[torch.Tensor]] = defaultdict(list)
    declared = {round(float(gamma), 8): float(gamma) for gamma in gammas}
    for feature, row in zip(features, selection):
        rounded = round(float(row["gamma"]), 8)
        if rounded in declared:
            grouped[declared[rounded]].append(feature)
    output = {}
    for gamma in gammas:
        rows = grouped.get(float(gamma), ())
        if not rows:
            raise ValueError(f"gamma {gamma:g} has no calibrated reference rows")
        output[float(gamma)] = torch.stack(list(rows)).to(device)
    return output


@torch.inference_mode()
def _frozen_reference_features(
    reference_adapter: PORT.HP100ExpansionPolicy,
    contexts: Sequence[torch.Tensor],
    plans_by_context: Sequence[torch.Tensor],
    bases_by_context: Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    if not (
        len(contexts) == len(plans_by_context) == len(bases_by_context)
    ):
        raise ValueError("reference feature inputs differ in length")
    return [
        reference_adapter.embed(context, plans, base=bases).detach()
        for context, plans, bases in zip(
            contexts, plans_by_context, bases_by_context,
        )
    ]


@torch.inference_mode()
def gather_hybrid(
    adapter: PORT.HP100ExpansionPolicy,
    reference_adapter: PORT.HP100ExpansionPolicy,
    task: PORT.SFMHP100ExpansionTask,
    *,
    keys: Sequence[LineageKey],
    config: HybridConfig,
    lengthscale: float,
    support_by_gamma: dict[float, torch.Tensor],
    microcycle: int,
    verifier: _OrderedSidecarVerifier,
) -> dict:
    """Gather one transactional hybrid rollout for every requested lineage."""
    states = _reset_states(task, keys, config)
    active = set(keys)
    samples: list[dict] = []
    events: list[dict] = []
    repair_batch_histogram: Counter[int] = Counter()
    gp = _posterior_by_gamma(config, lengthscale, support_by_gamma)
    timers: defaultdict[str, float] = defaultdict(float)

    for step in range(config.max_steps):
        ordered = sorted(active)
        if not ordered:
            break
        contexts = [task.context(states[key], key.gamma) for key in ordered]
        raw_seeds = [
            _sampling_seed(
                config.seed, "raw_gather", key, step, microcycle=microcycle,
            )
            for key in ordered
        ]
        started = time.perf_counter()
        raw_plans, raw_bases, _, _ = ACQ._sample_blocks(
            adapter, contexts, raw_seeds, K=1, flow_base_std=1.0,
        )
        timers["raw_flow"] += time.perf_counter() - started
        started = time.perf_counter()
        raw_results = verifier.verify_many([
            (context, plans, key.gamma)
            for key, context, plans in zip(ordered, contexts, raw_plans)
        ])
        timers["raw_verify"] += time.perf_counter() - started

        repairs: list[dict] = []
        chosen: dict[LineageKey, tuple[torch.Tensor, str]] = {}
        for key, context, plans, bases, verified_block in zip(
            ordered, contexts, raw_plans, raw_bases, raw_results,
        ):
            result_tuple, sidecar_tuple = verified_block
            if len(result_tuple) != 1:
                raise RuntimeError("raw verifier must return exactly one proposal")
            result = result_tuple[0]
            if result.error:
                raise RuntimeError("raw exact verifier error is unresolved")
            candidate = plans[0]
            sidecar = _validated_sidecar(
                task, context, candidate, key.gamma, result, sidecar_tuple[0],
            )
            event = _new_trace_event(
                key, states[key], microcycle=microcycle, step=step,
                context=context, raw_candidate=candidate,
                raw_result=result, raw_sidecar=sidecar,
            )
            _attach_scene_to_event(task, event, context)
            if result.valid:
                samples.append(_record(
                    group="P1", key=key, scenario_id=states[key].scenario_id,
                    microcycle=microcycle, step=step, context=context,
                    candidate=candidate, flow_base=bases[0], result=result,
                    base_std=1.0, repair_attempt=None,
                ))
                chosen[key] = (candidate, "P1")
            else:
                samples.append(_record(
                    group="Dminus", key=key, scenario_id=states[key].scenario_id,
                    microcycle=microcycle, step=step, context=context,
                    candidate=candidate, flow_base=bases[0], result=result,
                    base_std=1.0, repair_attempt=None,
                ))
                repairs.append({
                    "key": key, "context": context, "event": event,
                    "state": states[key],
                })
            events.append(event)

        pending = repairs
        for attempt in range(config.max_repair_batches):
            if not pending:
                break
            base_std = config.repair_base_std_start + (
                config.repair_base_std_step * attempt
            )
            repair_contexts = [row["context"] for row in pending]
            repair_seeds = [
                _sampling_seed(
                    config.seed, "repair", row["key"], step,
                    microcycle=microcycle, attempt=attempt,
                )
                for row in pending
            ]
            started = time.perf_counter()
            plans_by_context, bases_by_context, _, generators = (
                ACQ._sample_blocks(
                    adapter, repair_contexts, repair_seeds,
                    K=config.K, flow_base_std=base_std,
                )
            )
            features_by_context = _frozen_reference_features(
                reference_adapter, repair_contexts,
                plans_by_context, bases_by_context,
            )
            timers["repair_flow_and_phi"] += time.perf_counter() - started
            selected_rows = []
            for row, plans, bases, features, generator in zip(
                pending, plans_by_context, bases_by_context,
                features_by_context, generators,
            ):
                posterior = gp[float(row["key"].gamma)]
                marginal_sigma = posterior.sigma(features)
                beta_used = calibrate_fixed_beta(
                    [marginal_sigma], target=config.ess_target,
                )
                selected, selected_sigma, conditional_ess = posterior.acquire(
                    features, config.B, beta_used, generator,
                )
                selected_rows.append({
                    **row,
                    "plans": plans,
                    "bases": bases,
                    "features": features,
                    "selected": selected,
                    "selected_sigma": selected_sigma,
                    "conditional_ess": conditional_ess,
                    "marginal_sigma": marginal_sigma,
                    "beta_used": beta_used,
                    "queried": plans[selected],
                })
            started = time.perf_counter()
            verified = verifier.verify_many([
                (row["context"], row["queried"], row["key"].gamma)
                for row in selected_rows
            ])
            timers["repair_verify"] += time.perf_counter() - started
            next_pending = []
            for row, verified_block in zip(selected_rows, verified):
                results, sidecars = verified_block
                if len(results) != config.B or any(result.error for result in results):
                    raise RuntimeError("repair verifier returned unresolved B rows")
                selected = row["selected"]
                queried = row["queried"]
                queried_bases = row["bases"][selected]
                selected_local = _chosen_max_margin(results)
                attempt_trace = {
                    "attempt": int(attempt),
                    "base_std": float(base_std),
                    "candidate_ids": list(map(int, selected)),
                    "segments": np.stack([
                        _segment(row["state"], candidate)
                        for candidate in queried
                    ]).astype(np.float32, copy=False),
                    "verification": [
                        _verification_row(result) for result in results
                    ],
                    "selected_local": (
                        None if selected_local is None else int(selected_local)
                    ),
                    "selected_sigma": list(map(float, row["selected_sigma"])),
                    "conditional_ess": list(map(float, row["conditional_ess"])),
                    "beta_used": float(row["beta_used"]),
                    "marginal_ESS_over_K": float(normalized_ess(
                        row["marginal_sigma"], row["beta_used"],
                    )),
                    "uncertainty_uplift": float(
                        np.mean(row["selected_sigma"])
                        - float(row["marginal_sigma"].mean())
                    ),
                }
                row["event"]["repair_attempts"].append(attempt_trace)
                if selected_local is None:
                    next_pending.append({
                        key: row[key] for key in ("key", "context", "event", "state")
                    })
                    continue
                candidate = queried[selected_local]
                result = results[selected_local]
                sidecar = _validated_sidecar(
                    task, row["context"], candidate, row["key"].gamma,
                    result, sidecars[selected_local],
                )
                attempt_trace["selected_sidecar"] = sidecar
                samples.append(_record(
                    group="P2", key=row["key"],
                    scenario_id=row["state"].scenario_id,
                    microcycle=microcycle, step=step,
                    context=row["context"], candidate=candidate,
                    flow_base=queried_bases[selected_local], result=result,
                    base_std=base_std, repair_attempt=attempt,
                ))
                chosen[row["key"]] = (candidate, "P2")
                repair_batch_histogram[attempt + 1] += 1
            pending = next_pending

        exhausted = {row["key"] for row in pending}
        for key in ordered:
            event = next(
                row for row in reversed(events)
                if row["lineage"] == key.label
                and int(row["microcycle"]) == microcycle
                and int(row["step"]) == step
            )
            if key in exhausted:
                event["terminal"] = "repair_exhausted"
                active.remove(key)
                continue
            candidate, group = chosen[key]
            states[key] = task.advance(states[key], candidate)
            event["executed_group"] = group
            event["state_after"] = np.asarray(states[key].robot, np.float32).copy()
            terminal = task.terminal(states[key])
            if terminal is not None:
                event["terminal"] = str(terminal).lower()
                active.remove(key)

    for key in sorted(active):
        active.remove(key)
        # The absence of an early cutoff is explicit: T=180 is a real timeout.
        matches = [event for event in events if event["lineage"] == key.label]
        if matches:
            matches[-1]["terminal"] = "timeout"

    outcomes = {}
    for key in keys:
        rows = [event for event in events if event["lineage"] == key.label]
        terminal = rows[-1]["terminal"] if rows else "empty"
        outcomes[key.label] = {
            "gamma": float(key.gamma), "replica": int(key.replica),
            "scenario_id": int(states[key].scenario_id),
            "status": terminal,
            "executed_steps": int(states[key].steps),
            "net_goal_progress": float(
                np.linalg.norm(SS.GOAL) - np.linalg.norm(
                    np.asarray(states[key].robot, np.float32)[:2] - SS.GOAL
                )
            ),
        }
    counts = Counter(row["group"] for row in samples)
    return {
        "microcycle": int(microcycle),
        "outcomes": outcomes,
        "all_hybrid_success": all(
            row["status"] == "success" for row in outcomes.values()
        ),
        "samples": samples,
        "events": events,
        "sample_counts": dict(sorted(counts.items())),
        "repair_batches_to_success": dict(sorted(repair_batch_histogram.items())),
        "timers_seconds": dict(sorted(timers.items())),
        "gp_semantics": {
            "round_samples_committed_during_transaction": 0,
            "pretrained_reference_rows": {
                str(gamma): int(len(support_by_gamma[float(gamma)]))
                for gamma in config.gammas
            },
            "representation": "frozen pretrained paired-noised phi at s=0.9",
            "adaptive_marginal_ESS_target": float(config.ess_target),
            "acquisition_role": (
                "first draw is tilted by posterior sigma against immutable "
                "gamma-matched pretrained support; later B slots also use "
                "conditional RBF diversity without replacement"
            ),
        },
    }


def _stack_rows(rows: Sequence[dict], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.stack([row["context"] for row in rows]).to(device),
        torch.stack([row["candidate"] for row in rows]).to(device),
    )


def _gradient_norm(
    gradients: Sequence[torch.Tensor | None], device: torch.device,
) -> torch.Tensor:
    return torch.sqrt(sum(
        (
            gradient.square().sum() for gradient in gradients
            if gradient is not None
        ),
        torch.zeros((), device=device),
    ).clamp_min(0.0))


def _set_step_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


def phased_update(
    adapter: PORT.HP100ExpansionPolicy,
    samples: Sequence[dict],
    config: HybridConfig,
    *,
    microcycle: int,
) -> dict:
    """Apply P2/Dminus repair first, then a low-LR P1 retention pass.

    Dminus uses the existing normalized signed-gradient convention.  The
    requested lineage count ratio is logged and applied literally to alpha;
    because group losses are means, this is an aggressive ablation rather than
    a necessary count correction.
    """
    parameters = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("exhaustive-hybrid update has no trainable parameters")
    device = parameters[0].device
    before = _parameter_snapshot(parameters)
    by_lineage: defaultdict[str, list[dict]] = defaultdict(list)
    for row in samples:
        by_lineage[row["lineage"]].append(row)

    p2_optimizer = torch.optim.Adam(parameters, lr=config.p2_learning_rate)
    p2_losses, negative_losses, alpha_rows = [], [], []
    paired_rows = []
    p2_steps = 0
    for lineage in sorted(by_lineage):
        rows = by_lineage[lineage]
        p1 = [row for row in rows if row["group"] == "P1"]
        p2 = [row for row in rows if row["group"] == "P2"]
        negative = [row for row in rows if row["group"] == "Dminus"]
        if not p2:
            if negative:
                raise RuntimeError("successful hybrid lineage has Dminus without P2")
            continue
        if len(p2) != len(negative):
            raise RuntimeError("each raw Dminus must have exactly one selected P2 repair")
        count_ratio = (len(p1) + len(p2)) / len(negative)
        alpha_effective = config.negative_alpha_base * count_ratio
        p2_by_step = {
            (int(row["microcycle"]), int(row["step"])): row for row in p2
        }
        negative_by_step = {
            (int(row["microcycle"]), int(row["step"])): row
            for row in negative
        }
        if len(p2_by_step) != len(p2) or len(negative_by_step) != len(negative):
            raise RuntimeError("P2/Dminus contains duplicate lineage-context rows")
        if p2_by_step.keys() != negative_by_step.keys():
            raise RuntimeError("P2 and Dminus do not pair at the same contexts")
        paired_rows.extend(
            (p2_by_step[key], negative_by_step[key], float(alpha_effective))
            for key in sorted(p2_by_step)
        )
        alpha_rows.append({
            "lineage": lineage,
            "P1": len(p1), "P2": len(p2), "Dminus": len(negative),
            "positive_to_negative_ratio": float(count_ratio),
            "negative_alpha_effective": float(alpha_effective),
        })

    p2_order = np.random.default_rng(_counter_seed(
        config.seed, "P2_Dminus_order", microcycle,
    )).permutation(len(paired_rows))
    paired_rows = [paired_rows[int(index)] for index in p2_order]
    for batch_index, start in enumerate(range(0, len(paired_rows), config.batch_size)):
        batch = paired_rows[start:start + config.batch_size]
        positive_rows = [row[0] for row in batch]
        negative_rows = [row[1] for row in batch]
        alpha = torch.as_tensor(
            [row[2] for row in batch], device=device, dtype=torch.float32,
        )
        positive_context, positive_candidate = _stack_rows(positive_rows, device)
        negative_context, negative_candidate = _stack_rows(negative_rows, device)
        _set_step_seed(_counter_seed(
            config.seed, "P2_Dminus", microcycle, batch_index,
        ), device)
        positive_per_sample = adapter.cfm_loss(
            positive_context, positive_candidate, reduction="none",
        )
        negative_per_sample = adapter.cfm_loss(
            negative_context, negative_candidate, reduction="none",
        )
        positive_loss = positive_per_sample.mean()
        mean_alpha = alpha.mean()
        if float(mean_alpha) > 0.0:
            negative_objective = (
                negative_per_sample * (alpha / mean_alpha)
            ).mean()
        else:
            negative_objective = negative_per_sample.mean()
        positive_grad = torch.autograd.grad(
            positive_loss, parameters, allow_unused=True,
        )
        negative_grad = torch.autograd.grad(
            negative_objective, parameters, allow_unused=True,
        )
        positive_norm = _gradient_norm(positive_grad, device)
        negative_norm = _gradient_norm(negative_grad, device)
        rho = mean_alpha * positive_norm / (negative_norm + 1.0e-12)
        p2_optimizer.zero_grad()
        for parameter, pos, neg in zip(parameters, positive_grad, negative_grad):
            if pos is None and neg is None:
                parameter.grad = None
            elif pos is None:
                parameter.grad = -rho * neg.detach()
            elif neg is None:
                parameter.grad = pos.detach()
            else:
                parameter.grad = pos.detach() - rho * neg.detach()
        p2_optimizer.step()
        p2_steps += 1
        p2_losses.extend(map(float, positive_per_sample.detach().cpu()))
        negative_losses.extend(map(float, negative_per_sample.detach().cpu()))

    p1 = [row for row in samples if row["group"] == "P1"]
    p1_order = np.random.default_rng(_counter_seed(
        config.seed, "P1_order", microcycle,
    )).permutation(len(p1))
    p1 = [p1[int(index)] for index in p1_order]
    p1_optimizer = torch.optim.Adam(parameters, lr=config.p1_learning_rate)
    p1_losses = []
    p1_steps = 0
    for batch_index, start in enumerate(range(0, len(p1), config.batch_size)):
        rows = p1[start:start + config.batch_size]
        contexts, candidates = _stack_rows(rows, device)
        _set_step_seed(_counter_seed(
            config.seed, "P1_anchor", microcycle, batch_index,
        ), device)
        loss = adapter.cfm_loss(contexts, candidates, reduction="mean")
        p1_optimizer.zero_grad(); loss.backward(); p1_optimizer.step()
        p1_steps += 1
        p1_losses.append(float(loss.detach()))

    drift = _relative_parameter_drift(parameters, before)
    finite = _finite_parameters(parameters)
    accepted = bool(finite and drift <= config.max_relative_parameter_drift)
    if not accepted:
        _restore_parameters(parameters, before)
    return {
        "accepted": accepted,
        "finite": finite,
        "relative_parameter_drift": float(drift),
        "drift_gate": float(config.max_relative_parameter_drift),
        "P2_Dminus_steps": int(p2_steps),
        "P1_steps": int(p1_steps),
        "P2_loss": (float(np.mean(p2_losses)) if p2_losses else None),
        "Dminus_loss": (
            float(np.mean(negative_losses)) if negative_losses else None
        ),
        "P1_loss": (float(np.mean(p1_losses)) if p1_losses else None),
        "lineage_negative_scaling": alpha_rows,
        "phase_order": ["P2_plus_signed_Dminus", "P1_retention_anchor"],
        "optimizer_state": "separate fresh Adam state per phase and microcycle",
        "sample_order": (
            "global deterministic shuffle within P2/Dminus pairs and P1; "
            "each row exposed exactly once"
        ),
    }


@torch.inference_mode()
def raw_only_recheck(
    adapter: PORT.HP100ExpansionPolicy,
    task: PORT.SFMHP100ExpansionTask,
    *,
    keys: Sequence[LineageKey],
    config: HybridConfig,
    microcycle: int,
    verifier: _OrderedSidecarVerifier,
) -> dict:
    """Fixed development-gate raw check: no K/B, tilt, repair, or training."""
    states = _reset_states(task, keys, config)
    active = set(keys)
    events = []
    outcomes = {}
    for step in range(config.max_steps):
        ordered = sorted(active)
        if not ordered:
            break
        contexts = [task.context(states[key], key.gamma) for key in ordered]
        seeds = [
            _sampling_seed(config.seed, "fixed_raw_development_gate", key, step)
            for key in ordered
        ]
        plans_by_context, bases_by_context, _, _ = ACQ._sample_blocks(
            adapter, contexts, seeds, K=1, flow_base_std=1.0,
        )
        results_by_context = verifier.verify_many([
            (context, plans, key.gamma)
            for key, context, plans in zip(ordered, contexts, plans_by_context)
        ])
        for key, context, plans, bases, verified_block in zip(
            ordered, contexts, plans_by_context, bases_by_context,
            results_by_context,
        ):
            results, sidecars = verified_block
            if len(results) != 1 or len(sidecars) != 1:
                raise RuntimeError("raw development gate must return one exact row")
            result = results[0]
            if result.error:
                raise RuntimeError("fixed raw development gate verifier error")
            candidate = plans[0]
            exact = _validated_sidecar(
                task, context, candidate, key.gamma, result, sidecars[0],
            )
            before = np.asarray(states[key].robot, np.float32).copy()
            event = {
                "gamma": float(key.gamma), "replica": int(key.replica),
                "lineage": key.label, "scenario_id": int(states[key].scenario_id),
                "microcycle": int(microcycle), "step": int(step),
                "context": context.detach().cpu().to(torch.float32).clone(),
                "state_before": before,
                "raw_controls": candidate.detach().cpu().to(torch.float32).clone(),
                "flow_base": bases[0].detach().cpu().to(torch.float32).clone(),
                "raw_segment": _segment(states[key], candidate),
                "raw_verification": _verification_row(result),
                "raw_sidecar": exact,
                "valid": bool(result.valid), "terminal": None,
            }
            _attach_scene_to_event(task, event, context)
            if not result.valid:
                event["state_after"] = before.copy()
                event["terminal"] = "raw_red"
                outcomes[key.label] = {
                    "status": "raw_red", "clear": False,
                    "steps": int(states[key].steps),
                }
                active.remove(key)
                events.append(event)
                continue
            states[key] = task.advance(states[key], candidate)
            event["state_after"] = np.asarray(states[key].robot, np.float32).copy()
            terminal = task.terminal(states[key])
            if terminal is not None:
                status = str(terminal).lower()
                event["terminal"] = status
                outcomes[key.label] = {
                    "status": status, "clear": status == "success",
                    "steps": int(states[key].steps),
                }
                active.remove(key)
            events.append(event)
    for key in sorted(active):
        outcomes[key.label] = {
            "status": "timeout", "clear": False,
            "steps": int(states[key].steps),
        }
    clear = sorted(label for label, row in outcomes.items() if row["clear"])
    return {
        "microcycle": int(microcycle),
        "outcomes": outcomes,
        "clear_lineages": clear,
        "all_clear": len(clear) == len(keys),
        "events": events,
        "semantics": (
            "adaptively reused fixed development CRN on the declared lineages; "
            "raw temp1 singleton per context; stop on first red; CLEAR iff all "
            "executed H10 windows are exact-positive and terminal SUCCESS; this "
            "is not a fresh held-out evaluation bank"
        ),
    }


def _balanced_gp_indices(samples: Sequence[dict], cap: int) -> list[int]:
    positive = [
        (index, row) for index, row in enumerate(samples)
        if row["group"] in {"P1", "P2"}
    ]
    groups: defaultdict[tuple[float, int], list[tuple[int, dict]]] = defaultdict(list)
    for index, row in positive:
        groups[(float(row["gamma"]), int(row["replica"]))].append((index, row))
    for rows in groups.values():
        rows.sort(key=lambda item: (
            item[1]["microcycle"], item[1]["step"], item[1]["group"],
        ))
    quotas = {key: 0 for key in groups}
    while sum(quotas.values()) < min(int(cap), len(positive)):
        changed = False
        for key in sorted(groups):
            if (
                quotas[key] < len(groups[key])
                and sum(quotas.values()) < min(int(cap), len(positive))
            ):
                quotas[key] += 1
                changed = True
        if not changed:
            break
    ordered_by_group = {}
    for key, rows in groups.items():
        quota = quotas[key]
        if quota == 0:
            ordered_by_group[key] = []
        elif quota == len(rows):
            ordered_by_group[key] = [row[0] for row in rows]
        else:
            positions = np.linspace(0, len(rows) - 1, quota)
            chosen = np.rint(positions).astype(np.int64)
            if len(set(map(int, chosen))) != quota:
                raise RuntimeError("temporal-uniform GP selector duplicated a row")
            ordered_by_group[key] = [rows[int(position)][0] for position in chosen]
    selected = []
    for rank in range(max(quotas.values(), default=0)):
        for key in sorted(groups):
            if rank < len(ordered_by_group[key]):
                selected.append(ordered_by_group[key][rank])
    return selected


def _checkpoint_payload(
    adapter: PORT.HP100ExpansionPolicy,
    *,
    parent_checkpoint_sha: str,
    root_pretrained_sha: str,
    optimizer_scope: str,
    microcycle: int,
    promotable: bool,
) -> dict:
    return {
        "scientific_status": (
            "HP100_EXHAUSTIVE_HYBRID_ROUND1_COMMITTED"
            if promotable else "HP100_EXHAUSTIVE_HYBRID_DIAGNOSTIC_ONLY"
        ),
        # Keep the strict HP100 checkpoint schema so the canonical raw
        # evaluator can load both diagnostic and committed states directly.
        "state_dict": {
            key: value.detach().cpu().clone()
            for key, value in adapter.policy.state_dict().items()
        },
        "config": adapter.policy.config(),
        "parent_checkpoint_sha256": str(parent_checkpoint_sha),
        "pretrained_checkpoint_sha256": str(root_pretrained_sha),
        "optimizer_scope": str(optimizer_scope),
        "microcycle": int(microcycle),
        "promotable": bool(promotable),
    }


def run_arm(args) -> dict:
    config = HybridConfig(
        gammas=_parse_gammas(args.gammas),
        lineages_per_gamma=int(args.lineages_per_gamma),
        max_steps=int(args.max_steps),
        max_repair_batches=int(args.max_repair_batches),
        ess_target=float(args.ess_target), rbf_noise=float(args.rbf_noise),
        p1_learning_rate=float(args.p1_learning_rate),
        p2_learning_rate=float(args.p2_learning_rate),
        negative_alpha_base=float(args.negative_alpha_base),
        batch_size=int(args.batch_size),
        max_microcycles=int(args.max_microcycles),
        max_relative_parameter_drift=float(args.max_relative_parameter_drift),
        gp_buffer_cap=int(args.gp_buffer_cap), seed=int(args.seed),
    )
    config.validate()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing exhaustive-hybrid output: {output}")
    adapter, checkpoint_payload, checkpoint_sha, trainable_names, trainable_count = (
        _load_input_checkpoint(
            args.checkpoint, args.expected_checkpoint_sha256,
            args.device, args.optimizer_scope,
        )
    )
    reference_adapter, _, reference_sha, _, _ = _load_input_checkpoint(
        args.checkpoint, args.expected_checkpoint_sha256,
        args.device, "head_only",
    )
    if reference_sha != checkpoint_sha:
        raise RuntimeError("frozen acquisition reference checkpoint changed")
    root_pretrained_sha = (
        checkpoint_sha if checkpoint_payload.get("scientific_status") == "canonical_ID_promoted"
        else str(checkpoint_payload["pretrained_checkpoint_sha256"])
    )
    for parameter in reference_adapter.parameters():
        parameter.requires_grad_(False)
    reference_adapter.eval()
    features, calibration = BASE.calibration_features(
        reference_adapter, dataset_root=args.pretrain_dataset_root,
        expected_manifest_sha256=args.expected_pretrain_dataset_manifest_sha256,
        count=50, seed=config.seed, base_std=1.0,
        paired_noised_representation=True,
    )
    lengthscale = mean_pairwise_lengthscale(features)
    support_by_gamma = _calibration_support_by_gamma(
        features, calibration, config.gammas,
        device=next(reference_adapter.parameters()).device,
    )
    task = PORT.SFMHP100ExpansionTask(
        scene_profile=args.scene_profile,
        scenario_start=int(args.scenario_start),
    ).attach_context_encoder(adapter.policy)
    keys = _lineage_keys(config)
    preflight = {
        "status": PREFLIGHT_STATUS, "version": VERSION,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_scientific_status": checkpoint_payload.get("scientific_status"),
        "optimizer_scope": str(args.optimizer_scope),
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_count": trainable_count,
        "config": asdict(config),
        "scene": SS.scene_profile(args.scene_profile),
        "scenario_start": int(args.scenario_start),
        "gpu": BASE._gpu_contract(args.device, args.physical_gpu),
        "calibration": {**calibration, "lengthscale": float(lengthscale)},
        "acquisition_reference": {
            "checkpoint_sha256": reference_sha,
            "policy_state_sha256": _model_state_sha256(reference_adapter),
            "feature_space": "frozen pretrained paired-noised phi at s=0.9",
            "support_rows_by_gamma": {
                str(gamma): int(len(support_by_gamma[float(gamma)]))
                for gamma in config.gammas
            },
        },
        "transaction": {
            "commit_gate": (
                "all lineages CLEAR in the adaptively reused fixed raw-only "
                "development gate; final performance still requires a disjoint bank"
            ),
            "staged_data_enters_GP": False,
            "retry_cap": config.max_repair_batches,
            "maximum_repair_base_std": (
                config.repair_base_std_start
                + config.repair_base_std_step * (config.max_repair_batches - 1)
            ),
            "mathematical_guarantee": (
                "none: finite-H constant-velocity certification is not recursive "
                "feasibility or liveness under reactive SFM dynamics"
            ),
        },
    }
    preflight_path = output.with_suffix(output.suffix + ".preflight.json")
    preflight_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(preflight_path, preflight)
    if args.mode == "preflight":
        return preflight

    output.mkdir(parents=True)
    _write_json(output / "PREFLIGHT.json", preflight)

    staged_samples: list[dict] = []
    cycles = []
    trace = {
        "status": "SFM_HP100_EXHAUSTIVE_HYBRID_TRACE",
        "version": TRACE_VERSION, "config": asdict(config),
        "optimizer_scope": str(args.optimizer_scope),
        "checkpoint_sha256": checkpoint_sha,
        "events": [], "raw_rechecks": [], "cycles": [],
    }
    failing = set(keys)
    with _OrderedSidecarVerifier(task, int(args.verifier_workers)) as verifier:
        for microcycle in range(1, config.max_microcycles + 1):
            gather = gather_hybrid(
                adapter, reference_adapter, task,
                keys=sorted(failing), config=config,
                lengthscale=lengthscale, microcycle=microcycle,
                support_by_gamma=support_by_gamma,
                verifier=verifier,
            )
            staged_samples.extend(gather["samples"])
            trace["events"].extend(gather["events"])
            cycle = {
                key: value for key, value in gather.items()
                if key not in {"samples", "events"}
            }
            if not gather["all_hybrid_success"]:
                cycle["update"] = None
                cycle["recheck"] = None
                cycles.append(cycle); trace["cycles"].append(cycle)
                break
            update = phased_update(
                adapter, gather["samples"], config, microcycle=microcycle,
            )
            cycle["update"] = update
            diagnostic_checkpoint = output / f"checkpoint_micro_{microcycle:02d}.pt"
            _torch_save(diagnostic_checkpoint, _checkpoint_payload(
                adapter, parent_checkpoint_sha=checkpoint_sha,
                root_pretrained_sha=root_pretrained_sha,
                optimizer_scope=args.optimizer_scope,
                microcycle=microcycle, promotable=False,
            ))
            if not update["accepted"]:
                cycle["recheck"] = None
                cycles.append(cycle); trace["cycles"].append(cycle)
                break
            recheck = raw_only_recheck(
                adapter, task, keys=keys, config=config,
                microcycle=microcycle, verifier=verifier,
            )
            trace["raw_rechecks"].extend(recheck["events"])
            cycle["recheck"] = {
                key: value for key, value in recheck.items() if key != "events"
            }
            cycles.append(cycle); trace["cycles"].append(cycle)
            if recheck["all_clear"]:
                failing = set()
                break
            failing = {
                key for key in keys
                if not recheck["outcomes"][key.label]["clear"]
            }

    trace_path = output / "exhaustive_hybrid_trace.pt"
    _torch_save(trace_path, trace)
    all_clear = not failing and bool(cycles) and bool(
        cycles[-1].get("recheck", {}).get("all_clear", False)
    )
    if all_clear:
        gp_indices = _balanced_gp_indices(staged_samples, config.gp_buffer_cap)
        archive = {
            "status": "SFM_HP100_EXHAUSTIVE_HYBRID_ARCHIVE_COMMITTED",
            "samples": staged_samples,
            "gp_indices": gp_indices,
            "gp_balance": (
                "equal gamma-lineage quotas with temporal-uniform linspace "
                "inside each lineage"
            ),
        }
        archive_path = output / "query_archive.pt"
        _torch_save(archive_path, archive)
        checkpoint_path = output / "checkpoint_round_01.pt"
        _torch_save(checkpoint_path, _checkpoint_payload(
            adapter, parent_checkpoint_sha=checkpoint_sha,
            root_pretrained_sha=root_pretrained_sha,
            optimizer_scope=args.optimizer_scope,
            microcycle=int(cycles[-1]["microcycle"]), promotable=True,
        ))
        marker = {
            "status": COMPLETE_STATUS, "version": VERSION,
            "preflight": preflight, "cycles": cycles,
            "trace": str(trace_path), "trace_sha256": _sha256(trace_path),
            "archive": str(archive_path), "archive_sha256": _sha256(archive_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "policy_state_sha256": _model_state_sha256(adapter),
            "clear_microcycle": int(cycles[-1]["microcycle"]),
            "sample_counts": dict(sorted(Counter(
                row["group"] for row in staged_samples
            ).items())),
        }
        _write_json(output / "COMMIT_COMPLETE.json", marker)
    else:
        diagnostic_samples = output / "staged_samples_diagnostic_only.pt"
        _torch_save(diagnostic_samples, {
            "status": "SFM_HP100_EXHAUSTIVE_HYBRID_STAGED_DIAGNOSTIC_ONLY",
            "persistent_archive_committed": False,
            "enters_GP": False,
            "samples": staged_samples,
        })
        marker = {
            "status": INCOMPLETE_STATUS, "version": VERSION,
            "preflight": preflight, "cycles": cycles,
            "trace": str(trace_path), "trace_sha256": _sha256(trace_path),
            "persistent_archive_committed": False,
            "promotable_checkpoint_written": False,
            "staged_samples_diagnostic": str(diagnostic_samples),
            "staged_samples_diagnostic_sha256": _sha256(diagnostic_samples),
            "remaining_lineages": sorted(key.label for key in failing),
            "sample_counts_staged_only": dict(sorted(Counter(
                row["group"] for row in staged_samples
            ).items())),
        }
        _write_json(output / "QUALIFICATION_INCOMPLETE.json", marker)
    return marker


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--mode", choices=("preflight", "run"), default="preflight")
    value.add_argument("--checkpoint", required=True)
    value.add_argument("--expected-checkpoint-sha256", required=True)
    value.add_argument("--pretrain-dataset-root", required=True)
    value.add_argument("--expected-pretrain-dataset-manifest-sha256", required=True)
    value.add_argument("--output", required=True)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--physical-gpu", type=int, required=True)
    value.add_argument("--optimizer-scope", choices=(
        "head_only", "last_block_and_head",
    ), required=True)
    value.add_argument("--scene-profile", default=PORT.DEFAULT_SCENE_PROFILE,
                       choices=SS.SCIENTIFIC_EVAL_PROFILES)
    value.add_argument("--scenario-start", type=int, default=500_000)
    value.add_argument("--gammas", default=",".join(map(str, DEFAULT_GAMMAS)))
    value.add_argument("--lineages-per-gamma", type=int, default=4)
    value.add_argument("--max-steps", type=int, default=180)
    value.add_argument("--max-repair-batches", type=int, default=32)
    value.add_argument("--max-microcycles", type=int, default=6)
    value.add_argument("--verifier-workers", type=int, default=16)
    value.add_argument("--ess-target", type=float, default=0.1)
    value.add_argument("--rbf-noise", type=float, default=1.0e-2)
    value.add_argument("--p1-learning-rate", type=float, default=1.0e-5)
    value.add_argument("--p2-learning-rate", type=float, default=1.0e-3)
    value.add_argument("--negative-alpha-base", type=float, default=1.0e-3)
    value.add_argument("--batch-size", type=int, default=64)
    value.add_argument("--max-relative-parameter-drift", type=float, default=0.25)
    value.add_argument("--gp-buffer-cap", type=int, default=2_688)
    value.add_argument("--seed", type=int, default=2)
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    marker = run_arm(args)
    print(json.dumps({
        "status": marker["status"], "output": str(Path(args.output).resolve()),
    }), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
