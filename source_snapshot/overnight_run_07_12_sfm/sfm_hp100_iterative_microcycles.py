"""Default-off iterative continuation of an authenticated Ncausal winner.

The selected one-step treatment is fixed by the completed disaster-prefix
sweep.  Every microcycle resets and gathers the same canonical 4x4 lineages,
derives a fresh N=3 causal-negative view from that cycle only, applies the
selected P2/P1/Ncausal-only dose, and evaluates all 16 lineages with one fixed
raw temperature-one CRN.  Candidate updates are transactional: a candidate
is retained only when it strictly increases raw temperature-one CLEAR
lineages, preserves every previously CLEAR lineage, and does not decrease
exact-positive prefix coverage on lineages that remain non-CLEAR or aggregate
goal progress.  Full 16/16 CLEAR is an early-stop condition, not an acceptance
requirement.

No microcycle sample enters GP support or a replay archive.  Checkpoints from
this module remain diagnostic-only until a separate disjoint evaluation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

import numpy as np
import torch

import _paths  # noqa: F401
import grid_policy_sfm_hp100 as GPS
import sfm_hp100_ball_adapter as PORT
from sfm_hp100_ball_core.expansion import (
    mean_pairwise_lengthscale,
)
import sfm_hp100_ball_launch as BASE
import sfm_hp100_disaster_prefix_continuation as CONT
import sfm_hp100_disaster_prefix_sweep as SWEEP
import sfm_hp100_exhaustive_hybrid as HYBRID
import sfm_scene as SS


VERSION = "sfm_hp100_iterative_microcycles_v1"
PREFLIGHT_STATUS = "SFM_HP100_ITERATIVE_MICROCYCLES_PREFLIGHT_PASSED"
ALL_CLEAR_STATUS = "SFM_HP100_ITERATIVE_MICROCYCLES_ALL_CLEAR"
PLATEAU_STATUS = "SFM_HP100_ITERATIVE_MICROCYCLES_PLATEAU"
MAX_CYCLES_STATUS = "SFM_HP100_ITERATIVE_MICROCYCLES_MAX_CYCLES"
CAUSAL_N = 3
EXPECTED_LINEAGES = 16


@dataclass(frozen=True)
class FixedDose:
    name: str
    p1_learning_rate: float
    p1_passes: int
    p2_learning_rate: float
    p2_passes: int
    negative_alpha: float
    negative_passes: int
    negative_learning_rate: float = SWEEP.NEGATIVE_LEARNING_RATE

    @classmethod
    def from_marker(cls, marker: dict) -> "FixedDose":
        value = dict(marker["selected_dose"])
        required = {
            "name", "p1_learning_rate", "p1_passes", "p2_learning_rate",
            "p2_passes", "negative_alpha", "negative_passes",
        }
        if set(value) != required:
            raise ValueError("selected dose schema drifted")
        result = cls(**value)
        if min(result.p1_passes, result.p2_passes, result.negative_passes) < 1:
            raise ValueError("selected dose passes must be positive")
        if result.p1_learning_rate <= 0.0 or result.p2_learning_rate <= 0.0:
            raise ValueError("selected P1/P2 learning rates must be positive")
        if result.negative_alpha < 0.0:
            raise ValueError("selected negative alpha must be nonnegative")
        return result


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _hash_update(digest, value: Any) -> None:
    """Canonical semantic hash; never relies on torch.save container bytes."""
    if isinstance(value, np.generic):
        _hash_update(digest, value.item())
    elif value is None:
        digest.update(b"none")
    elif isinstance(value, bool):
        digest.update(b"bool1" if value else b"bool0")
    elif isinstance(value, int):
        digest.update(b"int" + str(value).encode() + b";")
    elif isinstance(value, float):
        if math.isnan(value):
            encoded = b"nan"
        elif value == math.inf:
            encoded = b"+inf"
        elif value == -math.inf:
            encoded = b"-inf"
        else:
            encoded = value.hex().encode()
        digest.update(b"float" + encoded + b";")
    elif isinstance(value, str):
        encoded = value.encode()
        digest.update(b"str" + len(encoded).to_bytes(8, "little") + encoded)
    elif isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor" + str(tensor.dtype).encode())
        _hash_update(digest, tuple(tensor.shape))
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"ndarray" + str(array.dtype).encode())
        _hash_update(digest, tuple(array.shape))
        digest.update(array.tobytes())
    elif isinstance(value, dict):
        digest.update(b"dict")
        for key in sorted(value, key=lambda item: str(item)):
            _hash_update(digest, key)
            _hash_update(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(b"list" + len(value).to_bytes(8, "little"))
        for item in value:
            _hash_update(digest, item)
    else:
        raise TypeError(f"unsupported canonical-hash type {type(value)!r}")


def canonical_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _hash_update(digest, value)
    return digest.hexdigest()


def _validate_selection_marker(marker: dict) -> FixedDose:
    status = marker.get("status")
    if status == "SFM_HP100_DISASTER_PREFIX_SWEEP_COMPLETE":
        if marker.get("eligible_for_phase2") is not True:
            raise ValueError("selected sweep is not eligible_for_phase2")
        if marker.get("disjoint_M50_incremental_gate_confirmed") is not True:
            raise ValueError("selected sweep lacks the disjoint incremental gate")
        if marker.get("disjoint_M50_strict_r0_win") is not True:
            raise ValueError("selected sweep lacks the disjoint strict-r0 gate")
    elif status == "SFM_HP100_SECONDARY_COMBINED_R0_CONFIRMATION_COMPLETE":
        if marker.get("eligible_for_iterative_microcycles") is not True:
            raise ValueError("secondary combined confirmation is not eligible")
        if marker.get("combined_M50_strict_r0_win") is not True:
            raise ValueError("secondary marker lacks the strict-r0 M50 gate")
        if marker.get("primary_incremental_null_preserved") is not True:
            raise ValueError("secondary marker does not preserve the primary null")
        if marker.get("primary_incremental_claim") is not False:
            raise ValueError("secondary marker makes an incremental Ncausal claim")
    else:
        raise ValueError("selection marker is not a supported completed confirmation")
    if marker.get("selected_scope") not in {
        "head_only", "last_block_and_head",
    }:
        raise ValueError("selected optimizer scope is invalid")
    checkpoint = marker.get("selected_treatment_checkpoint", {})
    if set(checkpoint) != {"path", "sha256", "policy_state_sha256"}:
        raise ValueError("selected treatment checkpoint schema drifted")
    for field in ("sha256", "policy_state_sha256"):
        if len(str(checkpoint[field])) != 64:
            raise ValueError(f"selected checkpoint {field} is invalid")
    return FixedDose.from_marker(marker)


def load_selection(path: str | Path, expected_sha256: str) -> tuple[dict, FixedDose, dict]:
    path = Path(path).resolve()
    actual = HYBRID._sha256(path)
    if actual != str(expected_sha256).lower():
        raise RuntimeError("selected sweep marker SHA256 mismatch")
    marker = json.loads(path.read_text())
    dose = _validate_selection_marker(marker)
    checkpoint = dict(marker["selected_treatment_checkpoint"])
    checkpoint["path"] = str(Path(checkpoint["path"]).resolve())
    if HYBRID._sha256(checkpoint["path"]) != checkpoint["sha256"]:
        raise RuntimeError("selected treatment checkpoint SHA256 mismatch")
    return marker, dose, {
        "path": str(path), "sha256": actual, "checkpoint": checkpoint,
    }


def _configure_trainable(adapter: PORT.HP100ExpansionPolicy, scope: str):
    if scope == "last_block_and_head":
        parameters = adapter.expansion_optimizer_parameters(scope)
        expected = 137_236
    elif scope == "head_only":
        for parameter in adapter.parameters():
            parameter.requires_grad_(False)
        parameters = list(adapter.head.parameters())
        for parameter in parameters:
            parameter.requires_grad_(True)
        expected = 5_140
    else:
        raise ValueError(f"unsupported optimizer scope {scope!r}")
    count = sum(parameter.numel() for parameter in parameters)
    if count != expected:
        raise RuntimeError(f"{scope} parameter count drifted: {count} != {expected}")
    return parameters


def _load_selected_adapter(
    selection: dict, checkpoint: dict, *, device: str,
) -> tuple[PORT.HP100ExpansionPolicy, dict, list[torch.nn.Parameter]]:
    policy, payload = GPS.load_sfm_hp100_policy(checkpoint["path"], device=device)
    if payload.get("scientific_status") != "HP100_DISASTER_PREFIX_DIAGNOSTIC_ONLY":
        raise ValueError("selected treatment is not the declared diagnostic state")
    if payload.get("promotable") is not False or payload.get("enters_GP") is not False:
        raise ValueError("selected treatment falsely claims promotion or GP entry")
    if payload.get("optimizer_scope") != selection["selected_scope"]:
        raise ValueError("selected checkpoint optimizer scope differs from marker")
    if "Ncausal" not in str(payload.get("role", "")):
        raise ValueError("selected treatment role is not Ncausal continuation")
    adapter = PORT.HP100ExpansionPolicy(policy)
    parameters = _configure_trainable(adapter, selection["selected_scope"])
    if HYBRID._model_state_sha256(adapter) != checkpoint["policy_state_sha256"]:
        raise RuntimeError("selected treatment policy-state SHA256 mismatch")
    return adapter, payload, parameters


def _lineage_contract(
    task: PORT.SFMHP100ExpansionTask,
    keys: Sequence[HYBRID.LineageKey],
    config: HYBRID.HybridConfig,
) -> dict:
    states = HYBRID._reset_states(task, keys, config)
    rows = []
    for key in sorted(keys):
        state = states[key]
        humans = []
        for human in state.humans:
            humans.append({
                "state": np.asarray(human.state, np.float64).tolist(),
                "control": np.asarray(human.control, np.float64).tolist(),
                "goal": np.asarray(human.goal, np.float64).tolist(),
                "radius": float(human.radius),
                "desired_speed": float(human.sfm_des_speed),
                "dt": float(human.dt),
            })
        rows.append({
            "lineage": key.label, "gamma": float(key.gamma),
            "replica": int(key.replica),
            "scenario_id": int(state.scenario_id),
            "robot": np.asarray(state.robot, np.float64).tolist(),
            "humans": humans,
        })
    semantic = {
        "seed": int(config.seed), "scenario_start": int(task.scenario_start),
        "scene_profile": task.scene_profile, "rows": rows,
    }
    return {"semantic": semantic, "canonical_sha256": canonical_sha256(semantic)}


def _cycle_training_view(gather: dict, cycle: int) -> dict:
    samples = list(gather["samples"])
    events = list(gather["events"])
    if any(int(row["microcycle"]) != int(cycle) for row in samples):
        raise RuntimeError("current microcycle received a prior-cycle sample")
    if any(int(row["microcycle"]) != int(cycle) for row in events):
        raise RuntimeError("current microcycle received a prior-cycle event")
    exhausted = sum(
        str(event.get("terminal", "")).lower() == "repair_exhausted"
        for event in events
    )
    short_prefix = [
        {
            "lineage": str(event["lineage"]),
            "microcycle": int(event["microcycle"]),
            "disaster_step": int(event["step"]),
        }
        for event in events
        if str(event.get("terminal", "")).lower() == "repair_exhausted"
        and int(event["step"]) < CAUSAL_N
    ]
    if short_prefix:
        raw_counts = {
            group: sum(row["group"] == group for row in samples)
            for group in ("P1", "P2", "Dminus")
        }
        return {
            "admissible": False, "view": None,
            "repair_exhausted": exhausted,
            "inadmissible_reason": "repair_exhausted_before_full_N3_prefix",
            "short_prefix_disasters": short_prefix,
            "counts": {
                "raw_source_counts": raw_counts,
                "mutually_exclusive_training_view_counts": {
                    "retained_P1": 0,
                    "retained_P2": 0,
                    "source_Dminus": 0,
                    "Ncausal": 0,
                },
                "source_rows": int(sum(raw_counts.values())),
                "training_view_rows": 0,
                "selected_training_exposures": {
                    "retained_P1": 0, "retained_P2": 0,
                    "Dminus": 0, "Ncausal": 0,
                },
            },
        }
    view = CONT.extract_disaster_training_view(
        {"events": events}, samples, causal_n=CAUSAL_N,
        expected_exhausted=exhausted, require_full_prefix=True,
    )
    for group in ("retained_P1", "retained_P2", "Dminus", "Ncausal"):
        if any(int(row["microcycle"]) != int(cycle) for row in view[group]):
            raise RuntimeError("training view leaked a prior microcycle")
    counts = CONT._exclusive_training_counts(view)
    counts["selected_training_exposures"] = {
        "retained_P1": len(view["retained_P1"]),
        "retained_P2": len(view["retained_P2"]),
        "Dminus": 0,
        "Ncausal": len(view["Ncausal"]),
    }
    return {
        "admissible": True, "view": view, "counts": counts,
        "repair_exhausted": exhausted, "inadmissible_reason": None,
        "short_prefix_disasters": [],
    }


def _raw_gate_summary(
    recheck: dict, keys: Sequence[HYBRID.LineageKey],
) -> dict:
    labels = {key.label for key in keys}
    outcomes = dict(recheck["outcomes"])
    if set(outcomes) != labels:
        raise RuntimeError("raw P1-only gate did not evaluate all 16 lineages")
    by_lineage: dict[str, list[dict]] = {label: [] for label in labels}
    for event in recheck.get("events", ()):
        by_lineage[str(event["lineage"])].append(event)
    progress = 0.0
    for label in sorted(labels):
        events = sorted(by_lineage[label], key=lambda row: int(row["step"]))
        if not events:
            continue
        start = np.asarray(events[0]["state_before"], np.float64)[:2]
        last = events[-1]
        end_field = "state_before" if last.get("terminal") == "raw_red" else "state_after"
        end = np.asarray(last[end_field], np.float64)[:2]
        progress += float(
            np.linalg.norm(start - SS.GOAL) - np.linalg.norm(end - SS.GOAL)
        )
    clear = sorted(label for label, row in outcomes.items() if row["clear"])
    return {
        "clear_lineages": clear, "clear_count": len(clear),
        "all_clear": len(clear) == len(keys),
        "exact_positive_prefix_sum": int(sum(
            int(row["steps"]) for row in outcomes.values()
        )),
        "aggregate_goal_progress": float(progress),
        "outcomes": outcomes,
    }


def _acceptance_decision(parent: dict, candidate: dict, *, epsilon: float = 1.0e-8) -> dict:
    parent_clear = set(parent["clear_lineages"])
    candidate_clear = set(candidate["clear_lineages"])
    clear_preserved = parent_clear <= candidate_clear
    aggregate_prefix_delta = int(candidate["exact_positive_prefix_sum"]) - int(
        parent["exact_positive_prefix_sum"]
    )
    remaining_nonclear = sorted(set(candidate["outcomes"]) - candidate_clear)
    per_lineage_prefix_delta = {
        label: int(candidate["outcomes"][label]["steps"])
        - int(parent["outcomes"][label]["steps"])
        for label in remaining_nonclear
    }
    progress_delta = float(candidate["aggregate_goal_progress"]) - float(
        parent["aggregate_goal_progress"]
    )
    prefix_nonnegative = all(
        delta >= 0 for delta in per_lineage_prefix_delta.values()
    )
    progress_nonnegative = progress_delta >= -float(epsilon)
    raw_success_delta = len(candidate_clear) - len(parent_clear)
    strict = raw_success_delta > 0
    return {
        "accepted": bool(
            clear_preserved and prefix_nonnegative
            and progress_nonnegative and strict
        ),
        "prior_CLEAR_subset_preserved": clear_preserved,
        "aggregate_exact_positive_prefix_delta_diagnostic_only": (
            aggregate_prefix_delta
        ),
        "candidate_nonclear_lineage_prefix_deltas": per_lineage_prefix_delta,
        "aggregate_goal_progress_delta": progress_delta,
        "candidate_nonclear_prefix_nondecrease": prefix_nonnegative,
        "CLEAR_lineages_exempt_from_prefix_length_gate": True,
        "goal_progress_nondecrease": progress_nonnegative,
        "raw_temperature_one_CLEAR_delta": raw_success_delta,
        "raw_temperature_one_success_strictly_increased": strict,
        "strict_improvement": strict,
        "epsilon": float(epsilon),
    }


def _diagnostic_checkpoint_payload(
    adapter: PORT.HP100ExpansionPolicy, *, parent_policy_sha256: str,
    selection_marker_sha256: str, cycle: int, optimizer_scope: str,
) -> dict:
    return {
        "scientific_status": "HP100_ITERATIVE_MICROCYCLE_DIAGNOSTIC_ONLY",
        "state_dict": {
            key: value.detach().cpu().clone()
            for key, value in adapter.policy.state_dict().items()
        },
        "config": adapter.policy.config(),
        "parent_policy_state_sha256": str(parent_policy_sha256),
        "selection_marker_sha256": str(selection_marker_sha256),
        "optimizer_scope": str(optimizer_scope), "microcycle": int(cycle),
        "promotable": False, "enters_GP": False,
    }


def _assert_update_contract(update: dict, training: dict, dose: FixedDose) -> None:
    negative = update["negative"]
    if negative["negative_source"] != "Ncausal_only":
        raise RuntimeError("iterative treatment used a non-Ncausal source")
    selected = negative["selected_source_counts"]
    if int(selected["Dminus"]) != 0:
        raise RuntimeError("original Dminus entered iterative training")
    if int(selected["Ncausal"]) != len(training["Ncausal"]):
        raise RuntimeError("Ncausal exposure count drifted")
    expected = {
        "P1": (len(training["retained_P1"]), dose.p1_passes),
        "P2": (len(training["retained_P2"]), dose.p2_passes),
        "negative": (len(training["Ncausal"]), dose.negative_passes),
    }
    for phase, (rows, passes) in expected.items():
        report = update[phase]
        if int(report["rows"]) != rows or int(report["passes"]) != passes:
            raise RuntimeError(f"{phase} row/pass contract drifted")
        if int(report["total_row_exposures"]) != rows * passes:
            raise RuntimeError(f"{phase} exact-pass accounting drifted")


def run_microcycles(
    *,
    adapter: PORT.HP100ExpansionPolicy,
    reference_adapter: PORT.HP100ExpansionPolicy,
    task: PORT.SFMHP100ExpansionTask,
    verifier,
    config: HYBRID.HybridConfig,
    dose: FixedDose,
    lengthscale: float,
    support_by_gamma: dict[float, torch.Tensor],
    output: Path,
    selection_sha256: str,
    optimizer_scope: str,
    max_cycles: int,
    plateau_patience: int,
    gather_fn: Callable = HYBRID.gather_hybrid,
    recheck_fn: Callable = HYBRID.raw_only_recheck,
    update_fn: Callable = CONT.apply_disaster_prefix_update,
    initial_checkpoint: dict | None = None,
) -> dict:
    keys = HYBRID._lineage_keys(config)
    if len(keys) != EXPECTED_LINEAGES or set(config.gammas) != set(HYBRID.DEFAULT_GAMMAS):
        raise ValueError("iterative continuation requires the canonical 4x4 lineages")
    trainable = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("iterative continuation has no trainable parameters")
    labels = {key.label for key in keys}
    baseline_raw = recheck_fn(
        adapter, task, keys=keys, config=config, microcycle=0,
        verifier=verifier,
    )
    baseline_summary = _raw_gate_summary(baseline_raw, keys)
    baseline_path = output / "raw_gate_baseline.pt"
    _torch_save(baseline_path, baseline_raw)
    baseline_canonical_sha = canonical_sha256(baseline_raw)
    current_summary = baseline_summary
    current_policy_sha = HYBRID._model_state_sha256(adapter)
    current_checkpoint = (
        None if initial_checkpoint is None else dict(initial_checkpoint)
    )
    cycles = []
    no_improvement = 0
    status = ALL_CLEAR_STATUS if current_summary["all_clear"] else None

    for cycle in range(1, int(max_cycles) + 1):
        if status is not None:
            break
        cycle_root = output / "microcycles" / f"cycle_{cycle:03d}"
        parent_snapshot = HYBRID._parameter_snapshot(trainable)
        parent_policy_sha = current_policy_sha
        gather = gather_fn(
            adapter, reference_adapter, task, keys=keys, config=config,
            lengthscale=lengthscale, support_by_gamma=support_by_gamma,
            microcycle=cycle, verifier=verifier,
        )
        if set(gather["outcomes"]) != labels:
            raise RuntimeError("gather did not reset and evaluate all 16 lineages")
        trace_payload = {
            "status": "SFM_HP100_ITERATIVE_MICROCYCLE_GATHER_TRACE",
            "microcycle": cycle, "events": gather["events"],
            "outcomes": gather["outcomes"],
        }
        sample_payload = {
            "status": "SFM_HP100_ITERATIVE_MICROCYCLE_SOURCE_SAMPLES",
            "microcycle": cycle, "samples": gather["samples"],
            "enters_GP": False, "replayed_in_later_cycles": False,
        }
        trace_path = cycle_root / "gather_trace.pt"
        samples_path = cycle_root / "source_samples.pt"
        _torch_save(trace_path, trace_payload)
        _torch_save(samples_path, sample_payload)
        derived = _cycle_training_view(gather, cycle)
        training = derived["view"]
        training_path = cycle_root / "training_view.pt"
        _torch_save(training_path, {
            "status": "SFM_HP100_ITERATIVE_MICROCYCLE_TRAINING_VIEW",
            "microcycle": cycle, "training_view": training,
            "counts": derived["counts"], "enters_GP": False,
            "source_cycles": [cycle], "admissible": derived["admissible"],
            "inadmissible_reason": derived["inadmissible_reason"],
            "short_prefix_disasters": derived["short_prefix_disasters"],
        })
        if derived["admissible"]:
            update = update_fn(
                adapter, training,
                p2_learning_rate=dose.p2_learning_rate,
                p1_learning_rate=dose.p1_learning_rate,
                negative_learning_rate=dose.negative_learning_rate,
                alpha=dose.negative_alpha, batch_size=config.batch_size,
                max_relative_parameter_drift=config.max_relative_parameter_drift,
                seed=config.seed, p1_passes=dose.p1_passes,
                p2_passes=dose.p2_passes,
                negative_passes=dose.negative_passes,
                negative_source="Ncausal_only",
            )
            update.pop("_positive_anchor_state_dict", None)
            _assert_update_contract(update, training, dose)
        else:
            update = {
                "accepted": False, "skipped": True,
                "reason": derived["inadmissible_reason"],
                "parameter_update_performed": False,
                "Dminus_exposures": 0, "Ncausal_exposures": 0,
            }
        candidate_policy_sha = HYBRID._model_state_sha256(adapter)
        checkpoint_path = cycle_root / "checkpoint_candidate.pt"
        _torch_save(checkpoint_path, _diagnostic_checkpoint_payload(
            adapter, parent_policy_sha256=parent_policy_sha,
            selection_marker_sha256=selection_sha256, cycle=cycle,
            optimizer_scope=optimizer_scope,
        ))
        candidate_raw = recheck_fn(
            adapter, task, keys=keys, config=config, microcycle=cycle,
            verifier=verifier,
        )
        candidate_summary = _raw_gate_summary(candidate_raw, keys)
        raw_path = cycle_root / "raw_gate.pt"
        _torch_save(raw_path, candidate_raw)
        decision = _acceptance_decision(current_summary, candidate_summary)
        accepted = bool(update["accepted"] and decision["accepted"])
        if not accepted:
            HYBRID._restore_parameters(trainable, parent_snapshot)
            if HYBRID._model_state_sha256(adapter) != parent_policy_sha:
                raise RuntimeError("rejected microcycle rollback is not bitwise exact")
            no_improvement += 1
        else:
            current_summary = candidate_summary
            current_policy_sha = candidate_policy_sha
            current_checkpoint = {
                "path": str(checkpoint_path),
                "sha256": HYBRID._sha256(checkpoint_path),
                "policy_state_sha256": candidate_policy_sha,
            }
            no_improvement = 0

        cycle_marker = {
            "status": (
                "SFM_HP100_ITERATIVE_MICROCYCLE_ACCEPTED"
                if accepted else "SFM_HP100_ITERATIVE_MICROCYCLE_REJECTED"
            ),
            "microcycle": cycle, "parent_policy_state_sha256": parent_policy_sha,
            "candidate_policy_state_sha256": candidate_policy_sha,
            "active_policy_state_sha256_after_gate": current_policy_sha,
            "gather_all_16_lineages": True,
            "gather_counts": gather["sample_counts"],
            "training_counts": derived["counts"],
            "repair_exhausted": derived["repair_exhausted"],
            "training_view_admissible": derived["admissible"],
            "training_view_inadmissible_reason": derived["inadmissible_reason"],
            "update": update, "raw_gate": candidate_summary,
            "active_raw_gate_after_gate": current_summary,
            "acceptance": decision | {"atomic_update_accepted": update["accepted"]},
            "accepted": accepted, "consecutive_no_improvement": no_improvement,
            "artifacts": {
                "trace": str(trace_path), "trace_sha256": HYBRID._sha256(trace_path),
                "trace_canonical_sha256": canonical_sha256(trace_payload),
                "samples": str(samples_path), "samples_sha256": HYBRID._sha256(samples_path),
                "samples_canonical_sha256": canonical_sha256(sample_payload),
                "training_view": str(training_path),
                "training_view_sha256": HYBRID._sha256(training_path),
                "training_view_canonical_sha256": canonical_sha256({
                    "microcycle": cycle, "training_view": training,
                    "admissible": derived["admissible"],
                }),
                "candidate_checkpoint": str(checkpoint_path),
                "candidate_checkpoint_sha256": HYBRID._sha256(checkpoint_path),
                "raw_gate": str(raw_path), "raw_gate_sha256": HYBRID._sha256(raw_path),
                "raw_gate_canonical_sha256": canonical_sha256(candidate_raw),
            },
            "GP_updated": False, "archive_committed": False,
            "previous_cycle_samples_replayed": False,
        }
        marker_path = cycle_root / "CYCLE_COMPLETE.json"
        _write_json(marker_path, cycle_marker)
        cycles.append({
            "microcycle": cycle, "accepted": accepted,
            "candidate_clear_count": candidate_summary["clear_count"],
            "candidate_exact_positive_prefix_sum": candidate_summary[
                "exact_positive_prefix_sum"
            ],
            "candidate_aggregate_goal_progress": candidate_summary[
                "aggregate_goal_progress"
            ],
            "active_clear_count": current_summary["clear_count"],
            "active_exact_positive_prefix_sum": current_summary[
                "exact_positive_prefix_sum"
            ],
            "active_aggregate_goal_progress": current_summary[
                "aggregate_goal_progress"
            ],
            "P1": derived["counts"]["raw_source_counts"]["P1"],
            "P2": derived["counts"]["raw_source_counts"]["P2"],
            "Dminus": derived["counts"]["raw_source_counts"]["Dminus"],
            "Ncausal": derived["counts"][
                "mutually_exclusive_training_view_counts"
            ]["Ncausal"],
            "marker": str(marker_path), "marker_sha256": HYBRID._sha256(marker_path),
        })
        if accepted and current_summary["all_clear"]:
            status = ALL_CLEAR_STATUS
        elif no_improvement >= int(plateau_patience):
            status = PLATEAU_STATUS

    if status is None:
        status = MAX_CYCLES_STATUS
    marker = {
        "status": status, "version": VERSION,
        "selection_marker_sha256": selection_sha256,
        "initial_selected_checkpoint": initial_checkpoint,
        "dose": asdict(dose), "optimizer_scope": optimizer_scope,
        "baseline_raw_gate": baseline_summary,
        "baseline_raw_gate_artifact": str(baseline_path),
        "baseline_raw_gate_sha256": HYBRID._sha256(baseline_path),
        "baseline_raw_gate_canonical_sha256": baseline_canonical_sha,
        "cycles": cycles, "final_raw_gate": current_summary,
        "final_policy_state_sha256": current_policy_sha,
        "final_accepted_checkpoint": current_checkpoint,
        "all_clear": bool(current_summary["all_clear"]),
        "plateau_patience": int(plateau_patience),
        "max_cycles": int(max_cycles),
        "promotable": False, "GP_updated": False,
        "archive_committed": False,
        "disjoint_evaluation_required_before_promotion": True,
    }
    _write_json(output / "ITERATIVE_COMPLETE.json", marker)
    return marker


def run(args) -> dict:
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing iterative output: {output}")
    source_gate = SWEEP.source_gate(
        Path(args.repo_root).resolve(), args.expected_source_commit,
    )
    if args.scene_profile != SWEEP.SCENE_PROFILE:
        raise ValueError("Phase 2 must reuse the selected sweep scene profile")
    if int(args.scenario_start) != 500_000:
        raise ValueError("Phase 2 fixes the selected sweep's canonical scenario start")
    if int(args.batch_size) != 64 or int(args.seed) != 2:
        raise ValueError("Phase 2 fixes the selected dose batch size and seed")
    selection, dose, selection_provenance = load_selection(
        args.selection_marker, args.expected_selection_marker_sha256,
    )
    adapter, selected_payload, trainable = _load_selected_adapter(
        selection, selection_provenance["checkpoint"], device=args.device,
    )
    reference_adapter, reference_payload, reference_sha, _, _ = (
        HYBRID._load_input_checkpoint(
            args.reference_checkpoint, args.expected_reference_checkpoint_sha256,
            args.device, "head_only",
        )
    )
    if selected_payload.get("parent_checkpoint_sha256") != reference_sha:
        raise RuntimeError("selected treatment parent is not the declared r0")
    if reference_payload.get("scientific_status") != "canonical_ID_promoted":
        raise ValueError("acquisition reference must be canonical HP100 r0")
    for parameter in reference_adapter.parameters():
        parameter.requires_grad_(False)
    reference_adapter.eval()
    selected_state = adapter.state_dict()
    reference_state = reference_adapter.state_dict()
    trainable_names = {
        name for name, parameter in adapter.policy.named_parameters()
        if parameter.requires_grad
    }
    for name, tensor in selected_state.items():
        if name not in trainable_names and not torch.equal(
            tensor.detach().cpu(), reference_state[name].detach().cpu()
        ):
            raise RuntimeError(f"selected checkpoint changed frozen state {name}")

    config = HYBRID.HybridConfig(
        max_microcycles=int(args.max_microcycles),
        max_relative_parameter_drift=float(args.max_relative_parameter_drift),
        batch_size=int(args.batch_size), seed=int(args.seed),
    )
    config.validate()
    keys = HYBRID._lineage_keys(config)
    if len(keys) != EXPECTED_LINEAGES:
        raise RuntimeError("canonical 4x4 lineage count drifted")
    features, calibration = BASE.calibration_features(
        reference_adapter, dataset_root=args.pretrain_dataset_root,
        expected_manifest_sha256=args.expected_pretrain_dataset_manifest_sha256,
        count=50, seed=config.seed, base_std=1.0,
        paired_noised_representation=True,
    )
    lengthscale = mean_pairwise_lengthscale(features)
    support_by_gamma = HYBRID._calibration_support_by_gamma(
        features, calibration, config.gammas,
        device=next(reference_adapter.parameters()).device,
    )
    task = PORT.SFMHP100ExpansionTask(
        scene_profile=args.scene_profile,
        scenario_start=int(args.scenario_start),
    ).attach_context_encoder(reference_adapter.policy)
    lineage_contract = _lineage_contract(task, keys, config)
    preflight = {
        "status": PREFLIGHT_STATUS, "version": VERSION,
        "source_gate": source_gate,
        "source_components": {
            Path(path).name: HYBRID._sha256(path)
            for path in (__file__, HYBRID.__file__, CONT.__file__, SWEEP.__file__)
        },
        "selection": selection_provenance, "selected_scope": selection["selected_scope"],
        "selected_checkpoint_payload": {
            key: selected_payload.get(key) for key in (
                "scientific_status", "parent_checkpoint_sha256",
                "optimizer_scope", "role", "promotable", "enters_GP",
            )
        },
        "reference_checkpoint": str(Path(args.reference_checkpoint).resolve()),
        "reference_checkpoint_sha256": reference_sha,
        "pretrain_dataset": {
            "root": str(Path(args.pretrain_dataset_root).resolve()),
            "manifest": str(
                (Path(args.pretrain_dataset_root) / "manifest.json").resolve()
            ),
            "manifest_sha256": str(
                args.expected_pretrain_dataset_manifest_sha256
            ).lower(),
        },
        "dose": asdict(dose), "config": asdict(config),
        "lineage_contract": lineage_contract,
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "scene": SS.scene_profile(args.scene_profile),
        "scenario_start": int(args.scenario_start),
        "acquisition": {
            "reference": "immutable canonical-r0 paired-noised phi at s=0.9",
            "lengthscale": float(lengthscale),
            "support_rows_by_gamma": {
                str(gamma): int(len(support_by_gamma[float(gamma)]))
                for gamma in config.gammas
            },
            "microcycle_samples_enter_GP": False,
        },
        "transaction": {
            "same_16_lineages_reset_every_cycle": True,
            "Ncausal": "fresh current-cycle N=3 repair-exhausted prefixes only",
            "Dminus_training_exposures": 0,
            "previous_cycle_replay": False,
            "plateau_patience": int(args.plateau_patience),
            "max_cycles": int(args.max_microcycles),
        },
        "gpu": BASE._gpu_contract(args.device, args.physical_gpu),
    }
    output.mkdir(parents=True)
    preflight_path = output / "PREFLIGHT.json"
    _write_json(preflight_path, preflight)
    with HYBRID._OrderedSidecarVerifier(task, int(args.verifier_workers)) as verifier:
        marker = run_microcycles(
            adapter=adapter, reference_adapter=reference_adapter, task=task,
            verifier=verifier, config=config, dose=dose,
            lengthscale=lengthscale, support_by_gamma=support_by_gamma,
            output=output, selection_sha256=selection_provenance["sha256"],
            optimizer_scope=selection["selected_scope"],
            max_cycles=int(args.max_microcycles),
            plateau_patience=int(args.plateau_patience),
            initial_checkpoint=selection_provenance["checkpoint"],
        )
    marker["preflight"] = str(preflight_path)
    marker["preflight_sha256"] = HYBRID._sha256(preflight_path)
    marker["source_HEAD"] = source_gate["HEAD"]
    _write_json(output / "ITERATIVE_COMPLETE.json", marker)
    return marker


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--repo-root", required=True)
    value.add_argument("--expected-source-commit", required=True)
    value.add_argument("--selection-marker", required=True)
    value.add_argument("--expected-selection-marker-sha256", required=True)
    value.add_argument("--reference-checkpoint", required=True)
    value.add_argument("--expected-reference-checkpoint-sha256", required=True)
    value.add_argument("--pretrain-dataset-root", required=True)
    value.add_argument("--expected-pretrain-dataset-manifest-sha256", required=True)
    value.add_argument("--output", required=True)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--physical-gpu", type=int, required=True)
    value.add_argument(
        "--scene-profile", default=PORT.DEFAULT_SCENE_PROFILE,
        choices=SS.SCIENTIFIC_EVAL_PROFILES,
    )
    value.add_argument("--scenario-start", type=int, default=500_000)
    value.add_argument("--verifier-workers", type=int, default=16)
    value.add_argument("--max-microcycles", type=int, default=10)
    value.add_argument("--plateau-patience", type=int, default=3)
    value.add_argument("--batch-size", type=int, default=64)
    value.add_argument("--max-relative-parameter-drift", type=float, default=0.10)
    value.add_argument("--seed", type=int, default=2)
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if min(
        args.verifier_workers, args.max_microcycles,
        args.plateau_patience, args.batch_size,
    ) < 1:
        raise ValueError("workers, cycles, patience, and batch size must be positive")
    if args.plateau_patience > args.max_microcycles:
        raise ValueError("plateau patience cannot exceed max microcycles")
    if not 0.0 < args.max_relative_parameter_drift < 1.0:
        raise ValueError("relative drift gate must lie in (0,1)")
    marker = run(args)
    print(json.dumps({"status": marker["status"], "output": args.output}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
