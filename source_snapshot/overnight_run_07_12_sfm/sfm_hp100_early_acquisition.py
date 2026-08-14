"""Early-horizon, no-update acquisition diagnostic for HP100 SFM.

This runner deliberately stops before any CFM replay.  It asks whether a larger
proposal/query budget can keep the frozen pretrained policy out of early NVP
states.  All seven gammas share the same scenario and flow-noise coordinates.
Only selected-B exact full-H queries may be executed; an optional K-B check at
NVP is audit-only and can never alter execution or enter a buffer.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable, Sequence

import numpy as np
import torch

import _paths  # noqa: F401
import sfm_hp100_ball_adapter as PORT
import sfm_hp100_ball_approval_smoke as APPROVAL
from sfm_hp100_ball_core.expansion import RBFPosterior, _OrderedVerifier, _counter_seed
import sfm_scene as SS


VERSION = "sfm_hp100_early_acquisition_v2"
TRACE_VERSION = "sfm_hp100_early_branch_trace_v2"
STATUS = "SFM_HP100_EARLY_ACQUISITION_COMPLETE"
ROUND = 1
RETRY_BATCH = 0
H = 10
EXPECTED_PARALLEL_EPISODES = 16


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def model_state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _embedded_dataset_provenance(payload: dict) -> dict:
    dataset_manifest_sha = payload.get("dataset", {}).get("manifest_sha256")
    provenance_manifest_sha = payload.get("provenance", {}).get(
        "dataset_manifest_sha256"
    )
    if (
        not isinstance(dataset_manifest_sha, str)
        or len(dataset_manifest_sha) != 64
        or dataset_manifest_sha != provenance_manifest_sha
    ):
        raise ValueError("checkpoint pretraining-dataset provenance is incomplete")
    return {
        "pretrain_dataset_manifest_sha256": dataset_manifest_sha,
        "pretrained_scientific_status": payload.get("scientific_status"),
    }


def resolve_checkpoint_provenance(
    payload: dict,
    *,
    checkpoint_path: str,
    checkpoint_sha256: str,
    pretrained_provenance_checkpoint: str | None,
    expansion_delivery_marker: str | None,
) -> dict:
    """Resolve an embedded or externally anchored pretraining-data chain.

    Canonical r0 checkpoints carry their dataset chain directly.  Portable
    expansion-evaluation checkpoints intentionally do not duplicate that large
    metadata object, so tracing one requires both the canonical r0 checkpoint
    and the immutable expansion delivery marker.  Validation happens before
    rollout generation so an incomplete chain fails cheaply.
    """
    try:
        embedded = _embedded_dataset_provenance(payload)
    except ValueError:
        embedded = None
    if embedded is not None:
        if pretrained_provenance_checkpoint or expansion_delivery_marker:
            raise ValueError(
                "embedded-provenance checkpoint does not accept external anchors"
            )
        return {"mode": "embedded", **embedded}

    if payload.get("scientific_status") != "HP100_BALL_PORT_EVALUATION_READY":
        raise ValueError("checkpoint lacks an accepted HP100 provenance mode")
    if not pretrained_provenance_checkpoint or not expansion_delivery_marker:
        raise ValueError(
            "expanded checkpoint requires --pretrained-provenance-checkpoint "
            "and --expansion-delivery-marker"
        )

    anchor_path = Path(pretrained_provenance_checkpoint).resolve()
    anchor_sha = APPROVAL.sha256_file(anchor_path)
    if payload.get("pretrained_checkpoint_sha256") != anchor_sha:
        raise ValueError("expanded checkpoint does not descend from provenance anchor")
    anchor_payload = torch.load(anchor_path, map_location="cpu", weights_only=False)
    if (
        not isinstance(anchor_payload, dict)
        or anchor_payload.get("scientific_status") != "canonical_ID_promoted"
        or anchor_payload.get("pretrained_from_scratch") is not True
    ):
        raise ValueError("pretraining provenance anchor is not canonical HP100 r0")
    anchor = _embedded_dataset_provenance(anchor_payload)

    marker_path = Path(expansion_delivery_marker).resolve()
    marker = json.loads(marker_path.read_text())
    if marker.get("status") != "SFM_HP100_EARLY_EXPANSION_COMPLETE":
        raise ValueError("expansion delivery marker is incomplete")
    preflight = marker.get("preflight", {})
    if preflight.get("checkpoint_sha256") != anchor_sha:
        raise ValueError("expansion marker does not pin the provenance anchor")
    if preflight.get("calibration", {}).get("manifest_sha256") != anchor[
        "pretrain_dataset_manifest_sha256"
    ]:
        raise ValueError("expansion marker dataset does not match provenance anchor")

    resolved_checkpoint = str(Path(checkpoint_path).resolve())
    matching = [
        row for row in marker.get("evaluation_checkpoints", {}).get("rows", [])
        if str(Path(row.get("path", "")).resolve()) == resolved_checkpoint
        and row.get("sha256") == checkpoint_sha256
    ]
    if len(matching) != 1:
        raise ValueError("expanded checkpoint is not uniquely pinned by delivery marker")
    source_path = Path(payload.get("source_generic_checkpoint", "")).resolve()
    source_sha = payload.get("source_generic_checkpoint_sha256")
    if (
        not source_path.is_file()
        or APPROVAL.sha256_file(source_path) != source_sha
        or matching[0].get("source_sha256") != source_sha
    ):
        raise ValueError("expanded checkpoint source-generic hash chain is broken")
    return {
        "mode": "anchored_expansion",
        **anchor,
        "pretrained_checkpoint": str(anchor_path),
        "pretrained_checkpoint_sha256": anchor_sha,
        "expansion_delivery_marker": str(marker_path),
        "expansion_delivery_marker_sha256": APPROVAL.sha256_file(marker_path),
        "source_generic_checkpoint": str(source_path),
        "source_generic_checkpoint_sha256": source_sha,
    }


def _candidate_sha256(candidate: torch.Tensor) -> str:
    value = candidate.detach().cpu().to(torch.float32).contiguous().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def _attach_exact_chosen_sidecar(
    task: PORT.SFMHP100ExpansionTask,
    event: dict,
) -> dict:
    """Reverify one retained blue branch and attach its exact GREEN geometry."""
    chosen = event.get("chosen_local")
    context = event.pop("context")
    if chosen is None:
        event["chosen_verifier_sidecar"] = None
        return event
    candidate = torch.from_numpy(
        np.asarray(event["queried_controls"][int(chosen)], np.float32)
    )
    result, sidecar = task._verify_one(context, candidate, float(event["gamma"]))
    original = event["verification"][int(chosen)]
    if result.error or not result.valid or not original["valid"]:
        raise RuntimeError("retained blue branch did not reproduce its exact positive label")
    if not math.isclose(
        float(result.execution_cost), float(original["execution_cost"]),
        rel_tol=0.0, abs_tol=1.0e-7,
    ):
        raise RuntimeError("serial GREEN recheck changed the blue branch cost")
    exact = sidecar["result"]
    _, ped_xy, ped_vel = task.decode_context(context)
    exact["pedestrian_prediction"] = PORT.VERIFY.predict_pedestrians(
        ped_xy, ped_vel, H=len(candidate),
    ).astype(np.float32, copy=False)
    artificial = [
        face for face in exact["faces"]
        if face["kind"] == "artificial" and bool(face["feasible"])
    ]
    if (
        not exact.get("resolved") or int(exact.get("y", 0)) != 1
        or not exact.get("full_h")
        or exact["diagnostics"].get("solver")
        != "paper_static_exact_2d_angular_interval_socp"
        or int(exact["diagnostics"].get("K_artificial", -1)) != 16
        or len(artificial) != 16
    ):
        raise RuntimeError("retained blue branch lacks the exact 16-face GREEN certificate")
    expected_segment = np.asarray(
        event["queried_segments"][int(chosen)], np.float32,
    )
    if not np.array_equal(np.asarray(exact["segment"], np.float32), expected_segment):
        raise RuntimeError("serial GREEN segment differs from the retained blue branch")
    event["chosen_verifier_sidecar"] = exact
    return event


def _terminal_status(value: str | None) -> str | None:
    if value is None:
        return None
    mapping = {"SUCCESS": "success", "COLLISION": "collision", "OOB": "oob"}
    if value not in mapping:
        raise RuntimeError(f"unexpected terminal status {value!r}")
    return mapping[value]


def _select(
    results: Sequence,
    *,
    rule: str,
    step_margin_weight: float,
) -> tuple[int | None, list[float | None]]:
    eligible = [
        index for index, result in enumerate(results)
        if not result.error and result.valid and result.progress_eligible
    ]
    scores: list[float | None] = [None] * len(results)
    if not eligible:
        return None, scores
    if rule == "max_step_margin":
        if any(results[index].step_margin is None for index in eligible):
            raise ValueError("max_step_margin requires a margin for every exact positive")
        chosen = min(
            eligible,
            key=lambda index: (
                -float(results[index].step_margin),
                float(results[index].execution_cost),
                index,
            ),
        )
        for index in eligible:
            scores[index] = -float(results[index].step_margin)
        return chosen, scores
    if rule != "weighted_cost":
        raise ValueError(f"unknown execution rule {rule!r}")
    return APPROVAL.select_exact_positive(
        results, step_margin_weight=step_margin_weight,
    )


def _progress_rank(results: Sequence, chosen: int | None) -> int | None:
    """One-based H10-progress rank among execution-eligible B queries."""
    if chosen is None:
        return None
    eligible = [
        index for index, result in enumerate(results)
        if not result.error and result.valid and result.progress_eligible
    ]
    ranked = sorted(
        eligible,
        key=lambda index: (-float(results[index].progress), index),
    )
    return ranked.index(int(chosen)) + 1


@torch.inference_mode()
def _sample_blocks(
    adapter: PORT.HP100ExpansionPolicy,
    contexts: Sequence[torch.Tensor],
    seeds: Sequence[int],
    *,
    K: int,
    flow_base_std: float,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Generator]]:
    """Sample all active contexts in one GPU flow solve while preserving CRN."""
    if len(contexts) != len(seeds):
        raise ValueError("contexts and seeds differ")
    device = next(adapter.parameters()).device
    dtype = adapter.policy.head.weight.dtype
    bases, tokens, generators = [], [], []
    for context, seed in zip(contexts, seeds):
        generator = torch.Generator(device=device).manual_seed(int(seed))
        base = torch.randn(
            K, adapter.policy.d, generator=generator, device=device, dtype=dtype,
        ) * float(flow_base_std)
        token = adapter._policy_context(context).to(device=device, dtype=dtype)
        bases.append(base)
        tokens.append(token.reshape(1, -1).expand(K, -1))
        generators.append(generator)
    flat_base = torch.cat(bases)
    flat_token = torch.cat(tokens)
    flat_plans = adapter.policy.sample(
        len(flat_base), flat_token, nfe=adapter.nfe, temp=1.0,
        initial_noise=flat_base,
    ).detach()
    flat_features = adapter.policy.phi_s_from_x0(
        flat_plans, flat_token, flat_base, s=0.9,
    ).detach()
    plans = list(flat_plans.split(K))
    features = list(flat_features.split(K))
    plan_bases = [base.reshape(K, H, 2) for base in bases]
    return plans, plan_bases, features, generators


def _plan_diagnostics(candidates: torch.Tensor, u_max: float) -> dict:
    values = candidates.detach().cpu().to(torch.float32).reshape(len(candidates), -1)
    saturation = float((values.abs() >= float(u_max) - 1.0e-6).float().mean())
    rounded = np.round(values.numpy(), decimals=5)
    unique_ratio = float(len(np.unique(rounded, axis=0)) / len(rounded))
    distance = (
        float(torch.pdist(values).mean() / math.sqrt(values.shape[1]))
        if len(values) > 1 else 0.0
    )
    return {
        "component_saturation_fraction": saturation,
        "unique_plan_fraction": unique_ratio,
        "mean_pairwise_control_distance": distance,
    }


def _summary(
    lineages: Sequence[dict], query_rows: Sequence[dict], context_rows: Sequence[dict],
    max_steps: int,
) -> dict:
    def one(
        rows: Sequence[dict], queries: Sequence[dict],
        contexts_for_group: Sequence[dict],
    ) -> dict:
        selection_horizon = min(30, int(max_steps))
        nvp_times = [row["nvp_step"] for row in rows if row["status"] == "nvp"]
        # Success/cutoff are favorable right-censoring. Collision/OOB terminate
        # gathering at their actual executed lifetime and must not rank as survival.
        gather_clock = [
            max_steps if row["status"] in {"success", "cutoff"}
            else row["executed_steps"]
            for row in rows
        ]
        nvp_clock = [
            row["nvp_step"] if row["status"] == "nvp" else max_steps
            for row in rows
        ]
        rmst_30_clock = [
            selection_horizon if row["status"] in {"success", "cutoff"}
            else min(selection_horizon, int(row["executed_steps"]))
            for row in rows
        ]
        positive = sum(row["y"] == 1 for row in queries)
        negative = sum(row["y"] == 0 and not row["error"] for row in queries)
        errors = sum(bool(row["error"]) for row in queries)
        contexts = {(row["seed"], row["gamma"], row["replica"], row["step"]) for row in queries}
        positive_contexts = {
            (row["seed"], row["gamma"], row["replica"], row["step"])
            for row in queries if row["y"] == 1
        }
        eligible_queries = [row for row in queries if row["execution_eligible"]]
        chosen_queries = [row for row in queries if row["chosen"]]
        prefix_contexts = [
            row for row in contexts_for_group
            if int(row["step"]) < selection_horizon
        ]
        one_step_progress = [
            row["chosen_one_step_goal_progress"] for row in contexts_for_group
            if row["chosen_one_step_goal_progress"] is not None
        ]
        net_progress = [
            row["net_goal_progress_to_stop_or_cutoff"] for row in rows
        ]
        progress_ranks = [
            row["chosen_H10_progress_rank"] for row in contexts_for_group
            if row["chosen_H10_progress_rank"] is not None
        ]
        performance_reference = [
            row["performance_reference_H10_goal_progress"]
            for row in contexts_for_group
            if row["performance_reference_H10_goal_progress"] is not None
        ]
        chosen_minus_reference = [
            row["chosen_minus_performance_reference_H10_progress"]
            for row in contexts_for_group
            if row["chosen_minus_performance_reference_H10_progress"] is not None
        ]
        lineage_keys = [
            (int(row["seed"]), float(row["gamma"]), int(row["replica"]))
            for row in rows
        ]

        def mean_or_none(values: Sequence[float]) -> float | None:
            return float(np.mean(values)) if values else None

        def lineage_macro_mean(field: str, *, sum_values: bool = False) -> float | None:
            grouped: dict[tuple[int, float, int], list[float]] = defaultdict(list)
            for row in prefix_contexts:
                value = row.get(field)
                if value is not None:
                    grouped[(
                        int(row["seed"]), float(row["gamma"]), int(row["replica"]),
                    )].append(float(value))
            values = []
            for key in lineage_keys:
                samples = grouped.get(key, [])
                if sum_values:
                    values.append(float(np.sum(samples)) if samples else 0.0)
                elif samples:
                    values.append(float(np.mean(samples)))
            return float(np.mean(values)) if values else None

        survived_30 = sum(
            row["status"] == "success" or row["executed_steps"] >= 30
            for row in rows
        )
        return {
            "lineages": len(rows),
            "status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
            "S30": float(survived_30 / len(rows)) if rows else None,
            "survived_30_count": int(survived_30),
            "early_gather_RMST_to_max_steps": (
                float(np.mean(gather_clock)) if rows else None
            ),
            "early_gather_RMST_at_30": (
                float(np.mean(rmst_30_clock)) if rows else None
            ),
            "lineage_macro_net_goal_progress_at_30": lineage_macro_mean(
                "chosen_one_step_goal_progress", sum_values=True,
            ),
            "lineage_net_goal_progress_to_stop_or_cutoff_mean": (
                float(np.mean(net_progress)) if net_progress else None
            ),
            "executed_steps": {
                "mean": float(np.mean([row["executed_steps"] for row in rows])),
                "median": float(np.median([row["executed_steps"] for row in rows])),
            } if rows else {"mean": None, "median": None},
            "right_censored_nvp_clock_median": (
                float(np.median(nvp_clock)) if rows else None
            ),
            "nvp_step": {
                "count": len(nvp_times),
                "q25": (float(np.quantile(nvp_times, .25)) if nvp_times else None),
                "median": (float(np.median(nvp_times)) if nvp_times else None),
                "q75": (float(np.quantile(nvp_times, .75)) if nvp_times else None),
            },
            "selected_B": {
                "contexts": len(contexts), "Dplus": positive, "Dminus": negative,
                "unresolved": errors,
                "positive_fraction": (
                    float(positive / (positive + negative))
                    if positive + negative else None
                ),
                "positive_context_fraction": (
                    float(len(positive_contexts) / len(contexts)) if contexts else None
                ),
            },
            "goal_progress": {
                "eligible_B_H10_mean": mean_or_none([
                    row["H10_goal_progress"] for row in eligible_queries
                ]),
                "chosen_H10_mean": mean_or_none([
                    row["H10_goal_progress"] for row in chosen_queries
                ]),
                "chosen_one_step_mean": mean_or_none(one_step_progress),
                "chosen_H10_progress_rank_mean": mean_or_none(progress_ranks),
                "lineage_macro_chosen_H10_mean_at_30": lineage_macro_mean(
                    "chosen_H10_goal_progress"
                ),
                "lineage_macro_chosen_one_step_mean_at_30": lineage_macro_mean(
                    "chosen_one_step_goal_progress"
                ),
                "chosen_H10_progress_percentile_mean_at_30": mean_or_none([
                    row["chosen_H10_progress_percentile"] for row in prefix_contexts
                    if row.get("chosen_H10_progress_percentile") is not None
                ]),
                "chosen_H10_progress_percentile_ge_0p75_fraction_at_30": (
                    float(np.mean([
                        row["chosen_H10_progress_percentile"] >= .75
                        for row in prefix_contexts
                        if row.get("chosen_H10_progress_percentile") is not None
                    ])) if any(
                        row.get("chosen_H10_progress_percentile") is not None
                        for row in prefix_contexts
                    ) else None
                ),
                "performance_reference_H10_mean": mean_or_none(
                    performance_reference
                ),
                "chosen_minus_performance_reference_H10_mean": mean_or_none(
                    chosen_minus_reference
                ),
                "chosen_native_cost_mean": mean_or_none([
                    row["native_cost"] for row in chosen_queries
                ]),
                "chosen_step_margin_mean": mean_or_none([
                    row["step_margin"] for row in chosen_queries
                ]),
            },
        }

    pooled = one(lineages, query_rows, context_rows)
    by_gamma = {}
    for gamma in map(float, SS.GAMMAS):
        rows = [row for row in lineages if float(row["gamma"]) == gamma]
        queries = [row for row in query_rows if float(row["gamma"]) == gamma]
        contexts_for_gamma = [
            row for row in context_rows if float(row["gamma"]) == gamma
        ]
        by_gamma[f"{gamma:g}"] = one(rows, queries, contexts_for_gamma)
    by_seed = {}
    for seed in sorted({int(row["seed"]) for row in lineages}):
        rows = [row for row in lineages if int(row["seed"]) == seed]
        queries = [row for row in query_rows if int(row["seed"]) == seed]
        contexts_for_seed = [
            row for row in context_rows if int(row["seed"]) == seed
        ]
        by_seed[str(seed)] = one(rows, queries, contexts_for_seed)
    return {"pooled": pooled, "per_gamma": by_gamma, "per_seed": by_seed}


@torch.inference_mode()
def run_diagnostic(
    adapter: PORT.HP100ExpansionPolicy,
    task: PORT.SFMHP100ExpansionTask,
    *,
    seeds: Sequence[int],
    max_steps: int,
    K: int,
    B: int,
    flow_base_std: float,
    beta: float,
    rbf_lengthscale: float,
    rbf_noise: float,
    execution_rule: str,
    step_margin_weight: float,
    parallel_episodes: int,
    verifier_workers: int,
    audit_unselected_at_nvp: bool,
    event_callback: Callable[[dict], None] | None = None,
) -> tuple[dict, list[dict], list[dict]]:
    if parallel_episodes != EXPECTED_PARALLEL_EPISODES:
        raise ValueError("canonical diagnostic requires exactly 16 lineages per gamma")
    if max_steps < 30:
        raise ValueError("max_steps must be at least 30")
    if K < 1 or B < 1 or B > K:
        raise ValueError("require 1 <= B <= K")
    if not math.isfinite(flow_base_std) or flow_base_std <= 0.0:
        raise ValueError("flow_base_std must be finite and positive")
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be finite and positive")
    if verifier_workers < 1 or not seeds:
        raise ValueError("verifier_workers and seeds must be nonempty/positive")
    if execution_rule == "weighted_cost" and (
        not math.isfinite(step_margin_weight) or step_margin_weight < 0.0
    ):
        raise ValueError("step-margin weight must be finite and nonnegative")

    before_hash = model_state_sha256(adapter)
    posterior = RBFPosterior(rbf_lengthscale, rbf_noise)
    posterior.set_buffer(None)
    device = next(adapter.parameters()).device
    episodes: list[dict[str, Any]] = []
    for seed in map(int, seeds):
        for gamma in map(float, SS.GAMMAS):
            for replica in range(parallel_episodes):
                reset_seed = _counter_seed(seed, "reset", ROUND, RETRY_BATCH, replica)
                state = task.reset(gamma, replica, reset_seed)
                initial_goal_distance = float(
                    np.linalg.norm(np.asarray(state.robot, np.float32)[:2] - SS.GOAL)
                )
                episodes.append({
                    "seed": seed, "gamma": gamma, "replica": replica,
                    "scenario_id": int(state.scenario_id), "state": state,
                    "status": None, "nvp_step": None, "executed_steps": 0,
                    "initial_goal_distance": initial_goal_distance,
                })

    query_rows: list[dict] = []
    context_rows: list[dict] = []
    plan_stats: list[dict] = []
    timers = defaultdict(float)
    verifier_candidates = 0
    audit_candidates = 0
    with _OrderedVerifier(task, verifier_workers) as verifier:
        for step in range(max_steps):
            active = [row for row in episodes if row["status"] is None]
            if not active:
                break
            started = time.perf_counter()
            contexts = [
                task.context(row["state"], row["gamma"]).detach().to(device)
                for row in active
            ]
            gather_seeds = [
                _counter_seed(
                    row["seed"], "gather", ROUND, RETRY_BATCH,
                    row["replica"], step,
                )
                for row in active
            ]
            candidate_blocks, base_blocks, feature_blocks, generators = _sample_blocks(
                adapter, contexts, gather_seeds, K=K, flow_base_std=flow_base_std,
            )
            timers["flow_and_phi"] += time.perf_counter() - started

            prepared = []
            started = time.perf_counter()
            for row, context, candidates, bases, features, generator in zip(
                active, contexts, candidate_blocks, base_blocks, feature_blocks, generators,
            ):
                covariance = posterior.covariance(features)
                sigma = (
                    torch.diagonal(covariance) - posterior.noise
                ).clamp_min(0.0).sqrt().detach().cpu()
                selected, selected_sigma, conditional_ess = (
                    posterior.acquire_from_covariance(
                        covariance, B, beta, generator,
                        sampling_device=features.device,
                    )
                )
                prepared.append({
                    "episode": row, "context": context, "candidates": candidates,
                    "bases": bases, "sigma": sigma, "selected": selected,
                    "selected_sigma": selected_sigma,
                    "conditional_ess": conditional_ess,
                    "queried": candidates[selected],
                })
                plan_stats.append(_plan_diagnostics(candidates, adapter.policy.u_max))
            timers["empty_rbf_acquisition"] += time.perf_counter() - started

            started = time.perf_counter()
            verified = verifier.verify_many([
                (row["context"], row["queried"], row["episode"]["gamma"])
                for row in prepared
            ])
            timers["selected_B_verifier"] += time.perf_counter() - started
            verifier_candidates += sum(len(row["queried"]) for row in prepared)

            nvp_audits = []
            for index, (prepared_row, results) in enumerate(zip(prepared, verified)):
                if len(results) != B:
                    raise RuntimeError("verifier returned the wrong selected-B size")
                if any(result.error for result in results):
                    raise RuntimeError("verifier error is unresolved, not Dminus")
                chosen, scores = _select(
                    results, rule=execution_rule,
                    step_margin_weight=step_margin_weight,
                )
                performance_reference, _ = _select(
                    results, rule="weighted_cost", step_margin_weight=0.0,
                )
                prepared_row["results"] = results
                prepared_row["chosen"] = chosen
                prepared_row["scores"] = scores
                prepared_row["performance_reference"] = performance_reference
                if chosen is None and audit_unselected_at_nvp and B < K:
                    selected_set = set(prepared_row["selected"])
                    remaining = [candidate for candidate in range(K) if candidate not in selected_set]
                    prepared_row["audit_ids"] = remaining
                    nvp_audits.append((index, remaining))

            audit_results: dict[int, Sequence] = {}
            if nvp_audits:
                started = time.perf_counter()
                rows = verifier.verify_many([
                    (
                        prepared[index]["context"],
                        prepared[index]["candidates"][remaining],
                        prepared[index]["episode"]["gamma"],
                    )
                    for index, remaining in nvp_audits
                ])
                timers["nvp_KminusB_audit"] += time.perf_counter() - started
                for (index, remaining), results in zip(nvp_audits, rows):
                    if any(result.error for result in results):
                        raise RuntimeError("NVP K-B audit encountered verifier error")
                    audit_results[index] = results
                    audit_candidates += len(remaining)

            for index, prepared_row in enumerate(prepared):
                episode = prepared_row["episode"]
                results = prepared_row["results"]
                selected = prepared_row["selected"]
                chosen = prepared_row["chosen"]
                performance_reference = prepared_row["performance_reference"]
                identity = {
                    "seed": episode["seed"], "gamma": episode["gamma"],
                    "replica": episode["replica"], "scenario_id": episode["scenario_id"],
                    "step": step,
                }
                positive_count = 0
                queried_cpu = prepared_row["queried"].detach().cpu()
                for local, (candidate_id, result) in enumerate(zip(selected, results)):
                    y = int(bool(result.valid))
                    positive_count += y
                    query_rows.append(identity | {
                        "candidate_id": int(candidate_id), "selected_slot": local,
                        "y": y, "error": bool(result.error),
                        "execution_eligible": bool(
                            result.valid and result.progress_eligible
                        ),
                        "chosen": bool(local == chosen),
                        "performance_reference": bool(
                            local == performance_reference
                        ),
                        "native_cost": float(result.execution_cost),
                        "step_margin": float(result.step_margin),
                        "H10_goal_progress": float(result.progress),
                        "selection_score": prepared_row["scores"][local],
                        "sigma": float(prepared_row["selected_sigma"][local]),
                        "candidate_sha256": _candidate_sha256(queried_cpu[local]),
                        "first_action": queried_cpu[local, 0].tolist(),
                    })

                oracle_positive = None
                nvp_cause = None
                eligible_progress = [
                    float(result.progress) for result in results
                    if result.valid and result.progress_eligible
                ]
                chosen_H10_progress = (
                    None if chosen is None else float(results[chosen].progress)
                )
                chosen_one_step_progress = None
                performance_reference_one_step_progress = None
                robot_before, ped_xy, ped_vel = task.decode_context(
                    prepared_row["context"]
                )
                robot_after = np.asarray(robot_before, np.float32).copy()
                if performance_reference is not None:
                    reference_after = PORT.clipped_plan_states(
                        robot_before,
                        queried_cpu[int(performance_reference)].numpy(),
                    )[1]
                    performance_reference_one_step_progress = float(
                        np.linalg.norm(np.asarray(robot_before[:2]) - SS.GOAL)
                        - np.linalg.norm(reference_after[:2] - SS.GOAL)
                    )
                archived_negative_local = None
                if chosen is None:
                    if index in audit_results:
                        oracle_positive = any(
                            result.valid and result.progress_eligible
                            for result in audit_results[index]
                        )
                        nvp_cause = "acquisition_miss" if oracle_positive else "proposal_failure"
                    elif B == K:
                        oracle_positive = False
                        nvp_cause = "proposal_failure"
                    else:
                        nvp_cause = "selected_B_all_negative"
                    episode["status"] = "nvp"
                    episode["nvp_step"] = step
                    rejected = [
                        local for local, result in enumerate(results)
                        if not result.error and not result.valid
                    ]
                    if rejected:
                        # Match the production executed_plus_nvp_negative rule.
                        # This is a counterfactual hard negative, never an
                        # executed branch: NVP means selected-B had no positive.
                        archived_negative_local = min(
                            rejected,
                            key=lambda local: (
                                float(results[local].execution_cost)
                                - float(step_margin_weight)
                                * float(results[local].step_margin or 0.0),
                                local,
                            ),
                        )
                else:
                    episode["state"] = task.advance(
                        episode["state"], prepared_row["queried"][chosen],
                    )
                    robot_after = np.asarray(
                        episode["state"].robot, np.float32,
                    ).copy()
                    chosen_one_step_progress = float(
                        np.linalg.norm(np.asarray(robot_before[:2]) - SS.GOAL)
                        - np.linalg.norm(robot_after[:2] - SS.GOAL)
                    )
                    episode["executed_steps"] += 1
                    episode["status"] = _terminal_status(task.terminal(episode["state"]))
                context_rows.append(identity | {
                    "selected_B_positive": positive_count,
                    "selected_B_all_negative": bool(chosen is None),
                    "oracle_K_positive_at_nvp": oracle_positive,
                    "nvp_cause": nvp_cause,
                    "chosen_candidate_id": (
                        None if chosen is None else int(selected[chosen])
                    ),
                    "performance_reference_candidate_id": (
                        None if performance_reference is None
                        else int(selected[performance_reference])
                    ),
                    "eligible_B_H10_progress_mean": (
                        float(np.mean(eligible_progress)) if eligible_progress else None
                    ),
                    "eligible_B_H10_progress_max": (
                        float(np.max(eligible_progress)) if eligible_progress else None
                    ),
                    "chosen_H10_goal_progress": chosen_H10_progress,
                    "chosen_one_step_goal_progress": chosen_one_step_progress,
                    "chosen_H10_progress_rank": _progress_rank(results, chosen),
                    "chosen_H10_progress_percentile": (
                        None if chosen is None else (
                            1.0 if len(eligible_progress) == 1 else
                            1.0 - (
                                float(_progress_rank(results, chosen) - 1)
                                / float(len(eligible_progress) - 1)
                            )
                        )
                    ),
                    "performance_reference_H10_goal_progress": (
                        None if performance_reference is None
                        else float(results[performance_reference].progress)
                    ),
                    "performance_reference_one_step_goal_progress": (
                        performance_reference_one_step_progress
                    ),
                    "chosen_minus_performance_reference_H10_progress": (
                        None if chosen is None or performance_reference is None
                        else float(
                            results[chosen].progress
                            - results[performance_reference].progress
                        )
                    ),
                    "marginal_sigma_mean": float(prepared_row["sigma"].mean()),
                    "marginal_ESS_over_K": 1.0,
                    "conditional_ESS_over_remaining": list(map(float, prepared_row["conditional_ess"])),
                })
                if event_callback is not None:
                    queried_array = queried_cpu.numpy()
                    trace_status = episode["status"]
                    if trace_status is None and step + 1 == max_steps:
                        trace_status = "EARLY_CUTOFF"
                    event_callback({
                        **identity,
                        "K": int(K), "B": int(B),
                        "context": prepared_row["context"].detach().cpu(),
                        "state_before": np.asarray(robot_before, np.float32).copy(),
                        "state_after": robot_after,
                        "ped_xy": np.asarray(ped_xy, np.float32).copy(),
                        "ped_vel": np.asarray(ped_vel, np.float32).copy(),
                        "queried_candidate_ids": [int(value) for value in selected],
                        "queried_controls": queried_array.copy(),
                        "queried_segments": np.stack([
                            PORT.clipped_plan_states(robot_before, candidate)[:, :2]
                            for candidate in queried_array
                        ]).astype(np.float32, copy=False),
                        "verification": [
                            {
                                "valid": bool(result.valid),
                                "progress_eligible": bool(result.progress_eligible),
                                "error": bool(result.error),
                                "execution_cost": float(result.execution_cost),
                                "progress": float(result.progress),
                                "step_margin": (
                                    None if result.step_margin is None
                                    else float(result.step_margin)
                                ),
                            }
                            for result in results
                        ],
                        "chosen_local": None if chosen is None else int(chosen),
                        "chosen_H10_progress_rank": _progress_rank(results, chosen),
                        "eligible_progress_candidates": len(eligible_progress),
                        "performance_reference_local": (
                            None if performance_reference is None
                            else int(performance_reference)
                        ),
                        "archived_negative_local": archived_negative_local,
                        "status": trace_status,
                        "nvp_cause": nvp_cause,
                    })
            if step + 1 == max_steps:
                for row in episodes:
                    if row["status"] is None:
                        row["status"] = "cutoff"

    after_hash = model_state_sha256(adapter)
    if before_hash != after_hash:
        raise RuntimeError("no-update diagnostic changed the policy state")
    if any(row["status"] is None for row in episodes):
        raise RuntimeError("diagnostic left an unterminated lineage")
    lineages = [
        {
            "seed": row["seed"], "gamma": row["gamma"], "replica": row["replica"],
            "scenario_id": row["scenario_id"], "status": row["status"],
            "nvp_step": row["nvp_step"], "executed_steps": row["executed_steps"],
            "net_goal_progress_to_stop_or_cutoff": float(
                row["initial_goal_distance"]
                - np.linalg.norm(np.asarray(row["state"].robot, np.float32)[:2] - SS.GOAL)
            ),
        }
        for row in episodes
    ]
    causes = Counter(
        row["nvp_cause"] for row in context_rows if row["nvp_cause"] is not None
    )
    payload = {
        "status": STATUS, "version": VERSION,
        "scientific_role": "acquisition-only; no replay, optimizer, or checkpoint write",
        "config": {
            "round": ROUND, "max_steps": max_steps, "K": K, "B": B, "H": H,
            "scene_profile": str(task.scene_profile),
            "scene_profile_contract": dict(task.profile),
            "scenario_start": int(task.scenario_start),
            "flow_base_std": flow_base_std, "beta": beta,
            "rbf": {"buffer_rows": 0, "lengthscale": rbf_lengthscale, "noise": rbf_noise},
            "round1_note": (
                "empty GP: marginal sigma is constant, marginal ESS/K=1, uplift=0; "
                "only sequential within-B RBF conditioning diversifies later draws"
            ),
            "batched_sampling_numerics": (
                "CRN bases are exact; one batched GPU flow solve is mathematically "
                "equivalent to production per-context solves but floating-point GEMM "
                "batch shape may change plans/phi at approximately 1e-6"
            ),
            "execution_rule": execution_rule,
            "execution_step_margin_weight": step_margin_weight,
            "parallel_episodes_per_gamma": parallel_episodes,
            "gammas": list(map(float, SS.GAMMAS)), "seeds": list(map(int, seeds)),
            "audit_unselected_KminusB_at_NVP": audit_unselected_at_nvp,
        },
        "policy_state_sha256_before": before_hash,
        "policy_state_sha256_after": after_hash,
        "policy_unchanged": True,
        "summary": _summary(lineages, query_rows, context_rows, max_steps),
        "nvp_audit": dict(sorted(causes.items())),
        "counts": {
            "selected_B_verifier_queries": verifier_candidates,
            "audit_only_KminusB_queries": audit_candidates,
            "contexts": len(context_rows),
        },
        "proposal_diagnostics": {
            key: float(np.mean([row[key] for row in plan_stats]))
            for key in plan_stats[0]
        },
        "timers_seconds": dict(sorted(timers.items())),
        "lineages": lineages,
        "scene_ledger": list(task.scene_ledger),
    }
    return payload, query_rows, context_rows


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--checkpoint", required=True)
    value.add_argument("--expected-checkpoint-sha256", required=True)
    value.add_argument(
        "--pretrained-provenance-checkpoint",
        help="canonical HP100 r0 anchor required only for an expanded checkpoint",
    )
    value.add_argument(
        "--expansion-delivery-marker",
        help="EARLY_EXPANSION_COMPLETE.json required only for an expanded checkpoint",
    )
    value.add_argument("--output", required=True)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--scene-profile", default=PORT.DEFAULT_SCENE_PROFILE,
                       choices=SS.SCIENTIFIC_EVAL_PROFILES)
    value.add_argument("--scenario-start", type=int, default=300_000)
    value.add_argument("--seeds", default="2")
    value.add_argument("--max-steps", type=int, default=40)
    value.add_argument("--K", type=int, default=64)
    value.add_argument("--B", type=int, default=16)
    value.add_argument("--flow-base-std", type=float, default=2.0)
    value.add_argument("--beta", type=float, default=5.0e-4)
    value.add_argument("--rbf-lengthscale", type=float, default=0.8932)
    value.add_argument("--rbf-noise", type=float, default=1.0e-2)
    value.add_argument("--execution-rule", choices=("max_step_margin", "weighted_cost"),
                       default="max_step_margin")
    value.add_argument("--execution-step-margin-weight", type=float, default=700_000.0)
    value.add_argument("--parallel-episodes", type=int, default=16)
    value.add_argument("--verifier-workers", type=int, default=16)
    value.add_argument("--no-audit-unselected-at-nvp", action="store_true")
    value.add_argument(
        "--trace-output",
        help=(
            "optional compact torch trace containing only replica 0 at "
            "gamma 0.1/0.5/1.0; sufficient for offline branch rendering"
        ),
    )
    value.add_argument(
        "--trace-replica", type=int, default=0,
        help="paired lineage replica retained in the optional compact trace",
    )
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    seed_values = tuple(int(item) for item in args.seeds.split(","))
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output root: {output}")
    output.mkdir(parents=True)
    adapter, checkpoint_payload, checkpoint_sha, trainable = APPROVAL._load_checkpoint(
        args.checkpoint, args.expected_checkpoint_sha256, args.device,
    )
    checkpoint_provenance = resolve_checkpoint_provenance(
        checkpoint_payload,
        checkpoint_path=args.checkpoint,
        checkpoint_sha256=checkpoint_sha,
        pretrained_provenance_checkpoint=args.pretrained_provenance_checkpoint,
        expansion_delivery_marker=args.expansion_delivery_marker,
    )
    task = PORT.SFMHP100ExpansionTask(
        scene_profile=args.scene_profile, scenario_start=args.scenario_start,
    ).attach_context_encoder(adapter.policy)
    started = time.time()
    trace_events: list[dict] = []
    if not 0 <= int(args.trace_replica) < int(args.parallel_episodes):
        raise ValueError("trace-replica must index one declared parallel lineage")

    def retain_trace(event: dict) -> None:
        if (
            int(event["seed"]) == seed_values[0]
            and int(event["replica"]) == int(args.trace_replica)
            and any(
                math.isclose(float(event["gamma"]), gamma, abs_tol=1.0e-7)
                for gamma in (0.1, 0.5, 1.0)
            )
        ):
            trace_events.append(_attach_exact_chosen_sidecar(task, event))

    payload, queries, contexts = run_diagnostic(
        adapter, task, seeds=seed_values,
        max_steps=args.max_steps, K=args.K, B=args.B,
        flow_base_std=args.flow_base_std, beta=args.beta,
        rbf_lengthscale=args.rbf_lengthscale, rbf_noise=args.rbf_noise,
        execution_rule=args.execution_rule,
        step_margin_weight=args.execution_step_margin_weight,
        parallel_episodes=args.parallel_episodes,
        verifier_workers=args.verifier_workers,
        audit_unselected_at_nvp=not args.no_audit_unselected_at_nvp,
        event_callback=(retain_trace if args.trace_output else None),
    )
    payload["wall_seconds"] = time.time() - started
    payload["checkpoint"] = {
        "path": str(Path(args.checkpoint).resolve()), "sha256": checkpoint_sha,
        "pretrain_dataset_manifest_sha256": checkpoint_provenance[
            "pretrain_dataset_manifest_sha256"
        ],
        "scientific_status": checkpoint_payload.get("scientific_status"),
        "trainable_parameter_names": trainable,
        "provenance": checkpoint_provenance,
    }
    query_path = output / "selected_B.jsonl"
    context_path = output / "contexts.jsonl"
    _write_jsonl(query_path, queries)
    _write_jsonl(context_path, contexts)
    payload["artifacts"] = {
        "selected_B": {"path": str(query_path), "sha256": APPROVAL.sha256_file(query_path)},
        "contexts": {"path": str(context_path), "sha256": APPROVAL.sha256_file(context_path)},
    }
    if args.trace_output:
        trace_path = Path(args.trace_output).resolve()
        if trace_path.exists():
            raise FileExistsError(f"refusing existing trace output: {trace_path}")
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        observed_gammas = sorted({float(row["gamma"]) for row in trace_events})
        if observed_gammas != [0.1, 0.5, 1.0]:
            raise RuntimeError(
                f"trace does not cover the three declared gammas: {observed_gammas}"
            )
        trace_bundle = {
            "status": "SFM_HP100_EARLY_BRANCH_TRACE_COMPLETE",
            "version": TRACE_VERSION,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_provenance": checkpoint_provenance,
            "policy_state_sha256": payload["policy_state_sha256_after"],
            "config": payload["config"],
            "summary": payload["summary"],
            "lineage_filter": {
                "seed": seed_values[0], "replica": int(args.trace_replica),
                "gammas": [0.1, 0.5, 1.0],
            },
            "semantics": {
                "green": "all and only the selected B exact-verifier queries",
                "purple_dashed": (
                    "lambda-zero native-cost reference among the same exact-positive B; "
                    "never an additional execution"
                ),
                "blue": (
                    "exact-positive selected proposal whose clipped first action "
                    "was executed; exact candidate-specific GREEN verifier faces "
                    "and ten level sets are retained"
                ),
                "red": (
                    "terminal NVP lowest J_native-lambda*m_step resolved-negative "
                    "counterfactual archived for Dminus; never executed"
                ),
                "black": "actual closed-loop first-action state path",
            },
            "events": trace_events,
        }
        torch.save(trace_bundle, trace_path)
        payload["artifacts"]["early_branch_trace"] = {
            "path": str(trace_path),
            "sha256": APPROVAL.sha256_file(trace_path),
            "events": len(trace_events),
        }
    marker = output / "EARLY_ACQUISITION_COMPLETE.json"
    _write_json(marker, payload)
    print(json.dumps({
        "status": STATUS, "output": str(output),
        "S30": payload["summary"]["pooled"]["S30"],
        "marker": str(marker),
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
