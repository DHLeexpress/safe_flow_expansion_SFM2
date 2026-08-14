"""Default-off disaster-prefix continuation for the HP100 hybrid trace.

This diagnostic consumes, but never mutates, an authenticated incomplete
``sfm_hp100_exhaustive_hybrid`` transaction.  For each repair-exhausted
lineage it reclassifies the three *executed* windows immediately preceding the
dead end as a negative training view (``Ncausal``).  Their original rows,
exact labels, candidates, flow bases, and provenance remain unchanged in the
source artifact.

The treatment is deliberately isolated from the alpha-zero control:

1. retained P2 attraction, one pass by default, Adam at 1e-3;
2. retained P1 retention, one pass by default, fresh Adam at 1e-5;
3. all original Dminus plus Ncausal, one negative-ascent pass by default,
   fresh Adam at ``negative_lr * alpha`` (1e-4 * .01 = 1e-6 by default).

Each phase may make multiple exact passes while retaining one Adam instance
within that phase.  ``Ncausal_only`` is an optional negative source; the
default remains the original Dminus-plus-Ncausal treatment.

The alpha-zero control is exactly the model after phases 1--2.  A targeted
four-step hybrid audit starts from each reconstructed t-3 state.  Only if at
least four of six dead ends are recovered does the updated model receive a
diagnostic-only full 4x4 hybrid rerun.  No row enters GP support, no second
update occurs, and no checkpoint is promoted by this module.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import torch

import _paths  # noqa: F401
import sfm_hp100_ball_adapter as PORT
from sfm_hp100_ball_core.expansion import (
    RBFPosterior,
    _counter_seed,
    calibrate_fixed_beta,
    mean_pairwise_lengthscale,
    normalized_ess,
)
import sfm_hp100_ball_launch as BASE
import sfm_hp100_early_acquisition as ACQ
import sfm_hp100_exhaustive_hybrid as HYBRID
import sfm_scene as SS


VERSION = "sfm_hp100_disaster_prefix_continuation_v1"
STATUS = "SFM_HP100_DISASTER_PREFIX_DIAGNOSTIC_COMPLETE"
REJECTED_STATUS = "SFM_HP100_DISASTER_PREFIX_UPDATE_REJECTED"
CAUSAL_N = 3
AUDIT_M = 64
TARGETED_STEPS = CAUSAL_N + 1
EXPECTED_EXHAUSTED = 6
NEGATIVE_SOURCES = ("Ncausal_only", "Dminus_plus_Ncausal")


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


def _model_state_sha256(module: torch.nn.Module) -> str:
    return HYBRID._model_state_sha256(module)


def _sample_identity(row: dict) -> tuple[str, int, int, str]:
    return (
        str(row["lineage"]), int(row["microcycle"]), int(row["step"]),
        str(row["group"]),
    )


def _event_identity(event: dict) -> tuple[str, int, int]:
    return (
        str(event["lineage"]), int(event["microcycle"]), int(event["step"]),
    )


def load_authenticated_inputs(
    *,
    trace_path: str | Path,
    expected_trace_sha256: str,
    staged_path: str | Path,
    expected_staged_sha256: str,
    checkpoint_sha256: str,
    optimizer_scope: str,
) -> tuple[dict, list[dict], dict]:
    trace_path = Path(trace_path).resolve()
    staged_path = Path(staged_path).resolve()
    actual_trace_sha = HYBRID._sha256(trace_path)
    actual_staged_sha = HYBRID._sha256(staged_path)
    if actual_trace_sha != str(expected_trace_sha256).lower():
        raise RuntimeError("prior trace SHA256 mismatch")
    if actual_staged_sha != str(expected_staged_sha256).lower():
        raise RuntimeError("prior staged-sample SHA256 mismatch")
    trace = torch.load(trace_path, map_location="cpu", weights_only=False)
    staged = torch.load(staged_path, map_location="cpu", weights_only=False)
    if trace.get("version") != HYBRID.TRACE_VERSION:
        raise ValueError("prior trace is not the frozen exhaustive-hybrid trace")
    if trace.get("checkpoint_sha256") != str(checkpoint_sha256).lower():
        raise ValueError("prior trace was not gathered from the declared r0")
    if trace.get("optimizer_scope") != optimizer_scope:
        raise ValueError("prior trace optimizer scope differs from this arm")
    if staged.get("status") != (
        "SFM_HP100_EXHAUSTIVE_HYBRID_STAGED_DIAGNOSTIC_ONLY"
    ):
        raise ValueError("prior samples are not an incomplete diagnostic stage")
    if staged.get("persistent_archive_committed") is not False:
        raise ValueError("prior incomplete rows unexpectedly entered an archive")
    if staged.get("enters_GP") is not False:
        raise ValueError("prior incomplete rows unexpectedly entered GP support")
    config = HYBRID.HybridConfig(**trace["config"])
    config.validate()
    if tuple(config.gammas) != HYBRID.DEFAULT_GAMMAS:
        raise ValueError("disaster-prefix study requires the canonical 4x4 trace")
    if config.lineages_per_gamma != 4:
        raise ValueError("disaster-prefix study requires four lineages/gamma")
    return trace, list(staged["samples"]), {
        "trace": str(trace_path), "trace_sha256": actual_trace_sha,
        "staged_samples": str(staged_path),
        "staged_samples_sha256": actual_staged_sha,
    }


def extract_disaster_training_view(
    trace: dict,
    samples: Sequence[dict],
    *,
    causal_n: int = CAUSAL_N,
    expected_exhausted: int = EXPECTED_EXHAUSTED,
    require_full_prefix: bool = True,
) -> dict:
    """Build Ncausal without rewriting any source row or exact label."""
    if int(causal_n) != CAUSAL_N:
        raise ValueError("the declared diagnostic fixes N=3")
    events = list(trace.get("events", ()))
    exhausted = [
        event for event in events
        if str(event.get("terminal", "")).lower() == "repair_exhausted"
    ]
    if len(exhausted) != int(expected_exhausted):
        raise ValueError(
            f"expected {expected_exhausted} repair-exhausted events, "
            f"found {len(exhausted)}"
        )
    labels = [str(event["lineage"]) for event in exhausted]
    if len(set(labels)) != len(labels):
        raise ValueError("a lineage has multiple repair-exhausted terminals")
    event_index = {_event_identity(event): event for event in events}
    if len(event_index) != len(events):
        raise ValueError("prior trace contains duplicate event identities")
    sample_index: defaultdict[tuple[str, int, int, str], list[dict]] = defaultdict(list)
    for row in samples:
        sample_index[_sample_identity(row)].append(row)

    causal_source_ids: set[tuple[str, int, int, str]] = set()
    causal = []
    disasters = []
    for terminal in sorted(exhausted, key=_event_identity):
        lineage, microcycle, terminal_step = _event_identity(terminal)
        if terminal_step < causal_n and require_full_prefix:
            raise ValueError("repair exhaustion occurred before a length-3 prefix")
        prefix_start = max(0, terminal_step - causal_n)
        prefix = []
        for step in range(prefix_start, terminal_step):
            event_key = (lineage, microcycle, step)
            if event_key not in event_index:
                raise ValueError(f"missing executed prefix event {event_key}")
            event = event_index[event_key]
            source_group = str(event.get("executed_group"))
            if source_group not in {"P1", "P2"}:
                raise ValueError("causal prefix must consist of executed P1/P2 rows")
            identity = (lineage, microcycle, step, source_group)
            matches = sample_index.get(identity, ())
            if len(matches) != 1:
                raise ValueError(f"expected one archived executed row for {identity}")
            source = matches[0]
            if not bool(source["verification"]["valid"]):
                raise ValueError("executed causal source lost its true positive label")
            causal_source_ids.add(identity)
            view = dict(source)
            view.update({
                "group": "Ncausal",
                "training_role": "negative_ascent_disaster_prefix",
                "source_group": source_group,
                "source_identity": identity,
                "causal_N": int(causal_n),
                "causal_offset": int(step - terminal_step),
                "disaster_step": int(terminal_step),
                "disaster_terminal": "repair_exhausted",
                "true_verifier_label_preserved": True,
            })
            # Clone tensors so a training view can never mutate the source row.
            for field in ("context", "candidate", "flow_base"):
                view[field] = source[field].detach().cpu().clone()
            causal.append(view)
            prefix.append(identity)
        disasters.append({
            "lineage": lineage, "microcycle": int(microcycle),
            "disaster_step": int(terminal_step),
            "prefix_start_step": int(prefix_start),
            "causal_source_identities": prefix,
        })

    retained_p1 = [
        row for row in samples
        if row["group"] == "P1" and _sample_identity(row) not in causal_source_ids
    ]
    retained_p2 = [
        row for row in samples
        if row["group"] == "P2" and _sample_identity(row) not in causal_source_ids
    ]
    dminus = [row for row in samples if row["group"] == "Dminus"]
    expected_causal = int(expected_exhausted) * int(causal_n)
    if require_full_prefix and len(causal) != expected_causal:
        raise RuntimeError("Ncausal extraction count drifted")
    return {
        "retained_P1": retained_p1,
        "retained_P2": retained_p2,
        "Dminus": dminus,
        "Ncausal": causal,
        "disasters": disasters,
        "causal_source_identities": sorted(causal_source_ids),
        "counts": {
            "source_P1": sum(row["group"] == "P1" for row in samples),
            "source_P2": sum(row["group"] == "P2" for row in samples),
            "source_Dminus": len(dminus),
            "retained_P1": len(retained_p1),
            "retained_P2": len(retained_p2),
            "Ncausal": len(causal),
            "repair_exhausted_lineages": len(disasters),
        },
        "semantics": (
            "Ncausal is a negative training view of the exactly N=3 actually "
            "executed P1/P2 windows preceding each dead end; original rows and "
            "their true exact-positive verifier labels remain unchanged; the "
            "same identities are removed from P1/P2 attraction; a diagnostic "
            "rerun may have fewer than N predecessors only when exhaustion "
            "occurs before step N"
        ),
    }


def _trainable_parameters(adapter: PORT.HP100ExpansionPolicy):
    parameters = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("disaster-prefix continuation has no trainable parameters")
    return parameters


def _one_exposure_attraction(
    adapter: PORT.HP100ExpansionPolicy,
    rows: Sequence[dict],
    *,
    learning_rate: float,
    batch_size: int,
    seed: int,
    phase: str,
    passes: int = 1,
) -> dict:
    if int(passes) < 1:
        raise ValueError("attraction passes must be positive")
    parameters = _trainable_parameters(adapter)
    device = parameters[0].device
    optimizer = torch.optim.Adam(parameters, lr=float(learning_rate))
    losses = []
    steps = 0
    order_seeds = []
    for pass_index in range(int(passes)):
        # Pass zero preserves the original one-pass CRN exactly. Additional
        # passes receive distinct counter coordinates without resetting Adam.
        order_seed = (
            _counter_seed(seed, phase, "order")
            if pass_index == 0 else
            _counter_seed(seed, phase, "pass", pass_index, "order")
        )
        order_seeds.append(int(order_seed))
        order = np.random.default_rng(order_seed).permutation(len(rows))
        ordered = [rows[int(index)] for index in order]
        for batch_index, start in enumerate(
            range(0, len(ordered), int(batch_size))
        ):
            batch = ordered[start:start + int(batch_size)]
            contexts, candidates = HYBRID._stack_rows(batch, device)
            HYBRID._set_step_seed(
                (
                    _counter_seed(seed, phase, "cfm", batch_index)
                    if pass_index == 0 else
                    _counter_seed(
                        seed, phase, "pass", pass_index, "cfm", batch_index,
                    )
                ),
                device,
            )
            loss = adapter.cfm_loss(contexts, candidates, reduction="mean")
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
            steps += 1
    return {
        "phase": phase, "role": "CFM attraction", "rows": len(rows),
        "unique_source_identities": len({_sample_identity(row) for row in rows}),
        "passes": int(passes), "exposures_per_row": int(passes),
        "total_row_exposures": int(len(rows) * int(passes)),
        "adam_steps": int(steps),
        "learning_rate": float(learning_rate),
        "mean_batch_loss": float(np.mean(losses)) if losses else None,
        "fresh_adam": True, "optimizer_instances": 1,
        "deterministic_fresh_order_seed_per_pass": order_seeds,
    }


def _one_exposure_negative_ascent(
    adapter: PORT.HP100ExpansionPolicy,
    rows: Sequence[dict],
    *,
    negative_learning_rate: float,
    alpha: float,
    batch_size: int,
    seed: int,
    passes: int = 1,
    phase: str = "Dminus_plus_Ncausal_negative_ascent",
) -> dict:
    if negative_learning_rate < 0.0 or alpha < 0.0:
        raise ValueError("negative learning rate and alpha must be nonnegative")
    if int(passes) < 1:
        raise ValueError("negative passes must be positive")
    parameters = _trainable_parameters(adapter)
    device = parameters[0].device
    effective_lr = float(negative_learning_rate) * float(alpha)
    optimizer = torch.optim.Adam(parameters, lr=effective_lr)
    losses = []
    steps = 0
    order_seeds = []
    seed_namespace = (
        "Dminus_Ncausal_negative_ascent"
        if phase == "Dminus_plus_Ncausal_negative_ascent" else phase
    )
    for pass_index in range(int(passes)):
        order_seed = (
            _counter_seed(seed, seed_namespace, "order")
            if pass_index == 0 else
            _counter_seed(
                seed, seed_namespace, "pass", pass_index, "order",
            )
        )
        order_seeds.append(int(order_seed))
        order = np.random.default_rng(order_seed).permutation(len(rows))
        ordered = [rows[int(index)] for index in order]
        for batch_index, start in enumerate(
            range(0, len(ordered), int(batch_size))
        ):
            batch = ordered[start:start + int(batch_size)]
            contexts, candidates = HYBRID._stack_rows(batch, device)
            HYBRID._set_step_seed(
                (
                    _counter_seed(seed, seed_namespace, batch_index)
                    if pass_index == 0 else
                    _counter_seed(
                        seed, seed_namespace, "pass", pass_index,
                        "cfm", batch_index,
                    )
                ),
                device,
            )
            cfm_loss = adapter.cfm_loss(contexts, candidates, reduction="mean")
            objective = -cfm_loss
            optimizer.zero_grad(); objective.backward(); optimizer.step()
            losses.append(float(cfm_loss.detach()))
            steps += 1
    return {
        "phase": phase,
        "role": "maximize CFM loss on declared negative training views",
        "rows": len(rows), "passes": int(passes),
        "exposures_per_row": int(passes),
        "total_row_exposures": int(len(rows) * int(passes)),
        "adam_steps": int(steps),
        "requested_negative_learning_rate": float(negative_learning_rate),
        "alpha": float(alpha), "effective_learning_rate": effective_lr,
        "mean_CFM_loss_before_ascent_step": (
            float(np.mean(losses)) if losses else None
        ),
        "fresh_adam": True, "optimizer_instances": 1,
        "deterministic_fresh_order_seed_per_pass": order_seeds,
    }


def apply_disaster_prefix_update(
    adapter: PORT.HP100ExpansionPolicy,
    training_view: dict,
    *,
    p2_learning_rate: float,
    p1_learning_rate: float,
    negative_learning_rate: float,
    alpha: float,
    batch_size: int,
    max_relative_parameter_drift: float,
    seed: int,
    p1_passes: int = 1,
    p2_passes: int = 1,
    negative_passes: int = 1,
    negative_source: str = "Dminus_plus_Ncausal",
) -> dict:
    """Create one shared positive anchor, then treatment-only negative ascent."""
    if negative_source not in NEGATIVE_SOURCES:
        raise ValueError(f"negative_source must be one of {NEGATIVE_SOURCES}")
    if min(int(p1_passes), int(p2_passes), int(negative_passes)) < 1:
        raise ValueError("P1, P2, and negative passes must be positive")
    parameters = _trainable_parameters(adapter)
    before = HYBRID._parameter_snapshot(parameters)
    before_sha = _model_state_sha256(adapter)
    p2 = _one_exposure_attraction(
        adapter, training_view["retained_P2"],
        learning_rate=p2_learning_rate, batch_size=batch_size,
        seed=seed, phase="retained_P2_attraction", passes=p2_passes,
    )
    p1 = _one_exposure_attraction(
        adapter, training_view["retained_P1"],
        learning_rate=p1_learning_rate, batch_size=batch_size,
        seed=seed, phase="retained_P1_retention", passes=p1_passes,
    )
    positive_anchor_sha = _model_state_sha256(adapter)
    positive_anchor_state = {
        key: value.detach().cpu().clone()
        for key, value in adapter.state_dict().items()
    }
    # The alpha-zero control is literally this state, not a separately rerun
    # stochastic optimization.  This makes the isolation bitwise by design.
    alpha0_control_sha = positive_anchor_sha
    negative_source_rows = {
        "Dminus": list(training_view["Dminus"]),
        "Ncausal": list(training_view["Ncausal"]),
    }
    negative_rows = (
        negative_source_rows["Ncausal"]
        if negative_source == "Ncausal_only"
        else [*negative_source_rows["Dminus"], *negative_source_rows["Ncausal"]]
    )
    negative_phase = (
        "Ncausal_negative_ascent"
        if negative_source == "Ncausal_only"
        else "Dminus_plus_Ncausal_negative_ascent"
    )
    negative = _one_exposure_negative_ascent(
        adapter, negative_rows,
        negative_learning_rate=negative_learning_rate, alpha=alpha,
        batch_size=batch_size, seed=seed, passes=negative_passes,
        phase=negative_phase,
    )
    negative.update({
        "negative_source": negative_source,
        "available_source_counts": {
            key: len(value) for key, value in negative_source_rows.items()
        },
        "selected_source_counts": {
            "Dminus": (
                len(negative_source_rows["Dminus"])
                if negative_source == "Dminus_plus_Ncausal" else 0
            ),
            "Ncausal": len(negative_source_rows["Ncausal"]),
        },
    })
    treatment_sha_before_gate = _model_state_sha256(adapter)
    drift = HYBRID._relative_parameter_drift(parameters, before)
    finite = HYBRID._finite_parameters(parameters)
    accepted = bool(finite and drift <= float(max_relative_parameter_drift))
    if not accepted:
        HYBRID._restore_parameters(parameters, before)
    return {
        "accepted": accepted, "finite": finite,
        "relative_parameter_drift": float(drift),
        "drift_gate": float(max_relative_parameter_drift),
        "model_sha256_before": before_sha,
        "positive_anchor_model_sha256": positive_anchor_sha,
        "alpha0_control_model_sha256": alpha0_control_sha,
        "positive_anchor_bitwise_equal_to_alpha0_control": (
            positive_anchor_sha == alpha0_control_sha
        ),
        "treatment_model_sha256_before_gate": treatment_sha_before_gate,
        "treatment_model_sha256_after_gate": _model_state_sha256(adapter),
        "phase_order": [
            "retained_P2_attraction", "retained_P1_retention",
            negative_phase,
        ],
        "P2": p2, "P1": p1, "negative": negative,
        "atomic_rollback": not accepted,
        # Internal handoff to the artifact writer.  ``run`` removes this
        # tensor mapping before serializing the JSON report.
        "_positive_anchor_state_dict": positive_anchor_state,
    }


def _events_by_lineage(trace: dict) -> dict[str, dict[int, dict]]:
    output: defaultdict[str, dict[int, dict]] = defaultdict(dict)
    for event in trace["events"]:
        if int(event["microcycle"]) != 1:
            raise ValueError("prior qualification trace must contain only microcycle 1")
        step = int(event["step"])
        if step in output[str(event["lineage"])]:
            raise ValueError("prior trace has duplicate lineage steps")
        output[str(event["lineage"])][step] = event
    return dict(output)


def _samples_by_identity(samples: Sequence[dict]) -> dict[tuple, dict]:
    output = {}
    for row in samples:
        identity = _sample_identity(row)
        if identity in output:
            raise ValueError(f"duplicate sample identity {identity}")
        output[identity] = row
    return output


def _assert_context_equal(actual: torch.Tensor, archived: torch.Tensor, label: str) -> None:
    actual = actual.detach().cpu().to(torch.float32)
    archived = archived.detach().cpu().to(torch.float32)
    if not torch.equal(actual, archived):
        maximum = float((actual - archived).abs().max())
        raise RuntimeError(f"archived context mismatch at {label}; max_abs={maximum:g}")


def reconstruct_prefix_state(
    task: PORT.SFMHP100ExpansionTask,
    *,
    key: HYBRID.LineageKey,
    target_step: int,
    config: HYBRID.HybridConfig,
    events_by_lineage: dict[str, dict[int, dict]],
    samples_by_identity: dict[tuple, dict],
) -> tuple[PORT.SFMState, torch.Tensor, dict]:
    """Reset/replay to target_step with exactly one context call per step."""
    state = HYBRID._reset_states(task, (key,), config)[key]
    rows = events_by_lineage[key.label]
    context_calls = 0
    for step in range(int(target_step) + 1):
        if step not in rows:
            raise ValueError(f"missing archived event for {key.label} step {step}")
        event = rows[step]
        if not np.array_equal(
            np.asarray(state.robot, np.float32),
            np.asarray(event["state_before"], np.float32),
        ):
            raise RuntimeError(f"state-before replay mismatch at {key.label}:{step}")
        context = task.context(state, key.gamma)
        context_calls += 1
        _assert_context_equal(context, event["context"], f"{key.label}:{step}")
        if step == int(target_step):
            return state, context, {
                "lineage": key.label, "target_step": int(target_step),
                "context_calls": int(context_calls),
                "expected_context_calls": int(target_step) + 1,
                "archived_context_bitwise_equal": True,
            }
        group = str(event.get("executed_group"))
        if group not in {"P1", "P2"}:
            raise RuntimeError("cannot replay a non-executed archived event")
        identity = (key.label, 1, step, group)
        if identity not in samples_by_identity:
            raise ValueError(f"missing archived executed sample {identity}")
        state = task.advance(state, samples_by_identity[identity]["candidate"])
        if not np.array_equal(
            np.asarray(state.robot, np.float32),
            np.asarray(event["state_after"], np.float32),
        ):
            raise RuntimeError(f"state-after replay mismatch at {key.label}:{step}")
        if task.terminal(state) is not None:
            raise RuntimeError("archived prefix terminated before its disaster")
    raise AssertionError("unreachable prefix reconstruction")


def archived_raw_base_sequence(
    *,
    disaster: dict,
    events_by_lineage: dict[str, dict[int, dict]],
    samples_by_identity: dict[tuple, dict],
) -> list[torch.Tensor]:
    """Return fixed raw x0 for t-3..t, including raw-negative P2 contexts."""
    lineage = str(disaster["lineage"])
    start = int(disaster["prefix_start_step"])
    terminal = int(disaster["disaster_step"])
    if terminal - start != CAUSAL_N:
        raise ValueError("disaster prefix does not have the declared N=3 span")
    output = []
    for step in range(start, terminal + 1):
        event = events_by_lineage[lineage][step]
        # At a P2 step, the raw proposal was negative; its Dminus row owns the
        # raw flow_base.  P2's base belongs to the repair and is not the CRN.
        group = "P1" if event.get("executed_group") == "P1" else "Dminus"
        identity = (lineage, 1, step, group)
        if identity not in samples_by_identity:
            raise ValueError(f"missing archived raw flow base {identity}")
        output.append(
            samples_by_identity[identity]["flow_base"].detach().cpu().clone()
        )
    if len(output) != TARGETED_STEPS:
        raise RuntimeError("fixed targeted raw-base sequence is not N+1=4")
    return output


@torch.inference_mode()
def _sample_from_bases(
    adapter: PORT.HP100ExpansionPolicy,
    context: torch.Tensor,
    bases: torch.Tensor,
) -> torch.Tensor:
    device = next(adapter.parameters()).device
    dtype = adapter.policy.head.weight.dtype
    flat = bases.reshape(len(bases), adapter.policy.d).to(device=device, dtype=dtype)
    token = adapter._policy_context(context).to(device=device, dtype=dtype)
    tokens = token.reshape(1, -1).expand(len(flat), -1)
    return adapter.policy.sample(
        len(flat), tokens, nfe=adapter.nfe, temp=1.0,
        initial_noise=flat,
    ).detach()


def _fixed_base_bank(
    adapter: PORT.HP100ExpansionPolicy,
    *,
    key: HYBRID.LineageKey,
    disaster_step: int,
    seed: int,
) -> torch.Tensor:
    device = next(adapter.parameters()).device
    generator = torch.Generator(device=device).manual_seed(_counter_seed(
        seed, "disaster_fixed_raw_M64", key.label, int(disaster_step),
    ))
    return torch.randn(
        AUDIT_M, adapter.policy.d, generator=generator, device=device,
        dtype=adapter.policy.head.weight.dtype,
    ).reshape(AUDIT_M, HYBRID.H, 2)


def _verification_summary(results: Sequence) -> dict:
    if any(result.error for result in results):
        raise RuntimeError("fixed-raw M64 audit contains an unresolved verifier row")
    resolved = [result for result in results if not result.error]
    positives = [result for result in resolved if result.valid]
    return {
        "M": len(results), "resolved": len(resolved),
        "exact_positive": len(positives),
        "exact_positive_fraction": float(len(positives) / len(results)),
        "mean_H10_progress": float(np.mean([r.progress for r in resolved])),
        "mean_step_margin": float(np.mean([
            r.step_margin for r in resolved if r.step_margin is not None
        ])),
    }


def _summary_delta(left: dict, right: dict) -> dict:
    fields = (
        "exact_positive_fraction", "mean_H10_progress", "mean_step_margin",
    )
    return {field: float(left[field] - right[field]) for field in fields}


@torch.inference_mode()
def fixed_raw_m64_audit(
    r0_adapter: PORT.HP100ExpansionPolicy,
    alpha0_adapter: PORT.HP100ExpansionPolicy,
    treatment_adapter: PORT.HP100ExpansionPolicy,
    task: PORT.SFMHP100ExpansionTask,
    verifier: HYBRID._OrderedSidecarVerifier,
    reconstructed: Sequence[dict],
    *,
    seed: int,
) -> tuple[list[dict], dict[str, torch.Tensor]]:
    rows, bank_by_lineage = [], {}
    blocks = []
    metadata = []
    for item in reconstructed:
        key = item["key"]
        bases = _fixed_base_bank(
            r0_adapter, key=key, disaster_step=item["disaster_step"], seed=seed,
        )
        bank_by_lineage[key.label] = bases.detach().cpu().clone()
        r0 = _sample_from_bases(r0_adapter, item["context"], bases)
        alpha0 = _sample_from_bases(alpha0_adapter, item["context"], bases)
        treatment = _sample_from_bases(treatment_adapter, item["context"], bases)
        blocks.extend([
            (item["context"], r0, key.gamma),
            (item["context"], alpha0, key.gamma),
            (item["context"], treatment, key.gamma),
        ])
        metadata.append((item, bases, r0, alpha0, treatment))
    verified = verifier.verify_many(blocks)
    for index, (item, bases, r0, alpha0, treatment) in enumerate(metadata):
        r0_results, _ = verified[3 * index]
        alpha0_results, _ = verified[3 * index + 1]
        treatment_results, _ = verified[3 * index + 2]
        r0_summary = _verification_summary(r0_results)
        alpha0_summary = _verification_summary(alpha0_results)
        treatment_summary = _verification_summary(treatment_results)
        rows.append({
            "lineage": item["key"].label,
            "gamma": float(item["key"].gamma),
            "disaster_step": int(item["disaster_step"]),
            "prefix_start_step": int(item["prefix_start_step"]),
            "fixed_base_sha256": hashlib.sha256(
                bases.detach().cpu().contiguous().numpy().tobytes()
            ).hexdigest(),
            "r0": r0_summary,
            "alpha0_positive_anchor": alpha0_summary,
            "treatment": treatment_summary,
            "anchor_minus_r0": _summary_delta(alpha0_summary, r0_summary),
            "treatment_minus_alpha0": _summary_delta(
                treatment_summary, alpha0_summary,
            ),
            "r0_plans": r0.detach().cpu(),
            "alpha0_plans": alpha0.detach().cpu(),
            "treatment_plans": treatment.detach().cpu(),
            "fixed_bases": bases.detach().cpu(),
        })
    return rows, bank_by_lineage


def _key_from_label(label: str) -> HYBRID.LineageKey:
    gamma_text, replica_text = label.split(":")
    return HYBRID.LineageKey(
        float(gamma_text.removeprefix("g")),
        int(replica_text.removeprefix("rep")),
    )


def _full_rerun_decision(
    recovered: int, minimum: int, render_diagnostic_regardless: bool,
) -> tuple[bool, bool]:
    gate_passed = int(recovered) >= int(minimum)
    return gate_passed, bool(gate_passed or render_diagnostic_regardless)


def _targeted_recovery_attribution(
    alpha0_rows: Sequence[dict], treatment_rows: Sequence[dict],
    r0_rows: Sequence[dict] | None = None,
) -> dict:
    alpha0 = {row["lineage"]: row for row in alpha0_rows}
    treatment = {row["lineage"]: row for row in treatment_rows}
    if alpha0.keys() != treatment.keys():
        raise ValueError("alpha0 and treatment targeted lineages differ")
    r0 = None if r0_rows is None else {
        row["lineage"]: row for row in r0_rows
    }
    if r0 is not None and r0.keys() != alpha0.keys():
        raise ValueError("r0 and alpha0 targeted lineages differ")
    deltas = []
    for lineage in sorted(treatment):
        alpha0_row = alpha0[lineage]
        treatment_row = treatment[lineage]
        deltas.append({
            "lineage": lineage,
            "r0_recovered": (
                None if r0 is None else bool(r0[lineage]["recovered"])
            ),
            "r0_status": None if r0 is None else r0[lineage]["status"],
            "alpha0_recovered": bool(alpha0_row["recovered"]),
            "treatment_recovered": bool(treatment_row["recovered"]),
            "treatment_minus_alpha0_recovered": int(
                treatment_row["recovered"]
            ) - int(alpha0_row["recovered"]),
            "alpha0_status": alpha0_row["status"],
            "treatment_status": treatment_row["status"],
            "treatment_minus_alpha0_sample_counts": {
                group: int(treatment_row["sample_counts"].get(group, 0))
                - int(alpha0_row["sample_counts"].get(group, 0))
                for group in ("P1", "P2", "Dminus")
            },
            "alpha0_minus_r0_recovered": (
                None if r0 is None else
                int(alpha0_row["recovered"]) - int(r0[lineage]["recovered"])
            ),
        })
    r0_recovered = None if r0 is None else sum(
        row["recovered"] for row in r0_rows
    )
    alpha0_recovered = sum(row["recovered"] for row in alpha0_rows)
    treatment_recovered = sum(row["recovered"] for row in treatment_rows)
    incremental = sum(
        row["treatment_recovered"] and not row["alpha0_recovered"]
        for row in deltas
    )
    regressions = sum(
        row["alpha0_recovered"] and not row["treatment_recovered"]
        for row in deltas
    )
    return {
        "r0_recovered": None if r0_recovered is None else int(r0_recovered),
        "alpha0_recovered": int(alpha0_recovered),
        "treatment_recovered": int(treatment_recovered),
        "incremental_recoveries_treatment_over_alpha0": int(incremental),
        "incremental_regressions_treatment_vs_alpha0": int(regressions),
        "net_incremental_recoveries": int(incremental - regressions),
        "alpha_effect_supported_by_recovery_count": bool(
            treatment_recovered > alpha0_recovered
        ),
        "positive_anchor_minus_r0_recovered": (
            None if r0_recovered is None else
            int(alpha0_recovered - r0_recovered)
        ),
        "rows": deltas,
    }


def _exclusive_training_counts(training_view: dict) -> dict:
    """Report a non-overlapping training partition and raw source separately."""
    raw_source = {
        "P1": int(training_view["counts"]["source_P1"]),
        "P2": int(training_view["counts"]["source_P2"]),
        "Dminus": int(training_view["counts"]["source_Dminus"]),
    }
    mutually_exclusive = {
        "retained_P1": int(training_view["counts"]["retained_P1"]),
        "retained_P2": int(training_view["counts"]["retained_P2"]),
        "source_Dminus": int(training_view["counts"]["source_Dminus"]),
        "Ncausal": int(training_view["counts"]["Ncausal"]),
    }
    if sum(raw_source.values()) != sum(mutually_exclusive.values()):
        raise RuntimeError("exclusive training partition does not conserve source rows")
    return {
        "raw_source_counts": raw_source,
        "mutually_exclusive_training_view_counts": mutually_exclusive,
        "source_rows": int(sum(raw_source.values())),
        "training_view_rows": int(sum(mutually_exclusive.values())),
    }


def _treatment_trace_provenance(
    *,
    treatment_checkpoint: str | Path,
    treatment_policy_state_sha256: str,
    parent_r0_checkpoint_sha256: str,
    source_incomplete_trace_sha256: str,
) -> dict:
    treatment_checkpoint = Path(treatment_checkpoint).resolve()
    treatment_sha = HYBRID._sha256(treatment_checkpoint)
    if treatment_sha == str(parent_r0_checkpoint_sha256).lower():
        raise RuntimeError("treatment trace cannot masquerade as its r0 parent")
    return {
        "checkpoint_sha256": treatment_sha,
        "generating_checkpoint": str(treatment_checkpoint),
        "generating_checkpoint_sha256": treatment_sha,
        "generating_policy_state_sha256": str(treatment_policy_state_sha256),
        "parent_r0_checkpoint_sha256": str(parent_r0_checkpoint_sha256).lower(),
        "source_incomplete_trace_sha256": str(
            source_incomplete_trace_sha256
        ).lower(),
    }
@torch.inference_mode()
def _targeted_repair(
    adapter: PORT.HP100ExpansionPolicy,
    reference_adapter: PORT.HP100ExpansionPolicy,
    task: PORT.SFMHP100ExpansionTask,
    verifier: HYBRID._OrderedSidecarVerifier,
    *,
    state: PORT.SFMState,
    context: torch.Tensor,
    key: HYBRID.LineageKey,
    step: int,
    config: HYBRID.HybridConfig,
    posterior: RBFPosterior,
    microcycle: int,
) -> tuple[torch.Tensor | None, dict | None, list[dict]]:
    attempts = []
    for attempt in range(config.max_repair_batches):
        base_std = config.repair_base_std_start + config.repair_base_std_step * attempt
        seed = HYBRID._sampling_seed(
            config.seed, "disaster_targeted_repair", key, step,
            microcycle=microcycle, attempt=attempt,
        )
        plans, bases, _, generators = ACQ._sample_blocks(
            adapter, (context,), (seed,), K=config.K, flow_base_std=base_std,
        )
        features = HYBRID._frozen_reference_features(
            reference_adapter, (context,), plans, bases,
        )[0]
        sigma = posterior.sigma(features)
        beta = calibrate_fixed_beta([sigma], target=config.ess_target)
        selected, selected_sigma, conditional_ess = posterior.acquire(
            features, config.B, beta, generators[0],
        )
        queried = plans[0][selected]
        results, sidecars = verifier.verify_many([
            (context, queried, key.gamma),
        ])[0]
        if any(result.error for result in results):
            raise RuntimeError("targeted repair has an unresolved exact query")
        chosen = HYBRID._chosen_max_margin(results)
        attempt_trace = {
            "attempt": int(attempt), "base_std": float(base_std),
            "candidate_ids": list(map(int, selected)),
            "segments": np.stack([
                HYBRID._segment(state, candidate) for candidate in queried
            ]).astype(np.float32, copy=False),
            "verification": [HYBRID._verification_row(result) for result in results],
            "selected_local": None if chosen is None else int(chosen),
            "selected_sigma": list(map(float, selected_sigma)),
            "conditional_ess": list(map(float, conditional_ess)),
            "beta_used": float(beta),
            "marginal_ESS_over_K": float(normalized_ess(sigma, beta)),
            "uncertainty_uplift": float(
                np.mean(selected_sigma) - float(sigma.mean())
            ),
        }
        attempts.append(attempt_trace)
        if chosen is None:
            continue
        candidate = queried[chosen]
        attempt_trace["selected_sidecar"] = HYBRID._validated_sidecar(
            task, context, candidate, key.gamma, results[chosen], sidecars[chosen],
        )
        return candidate, HYBRID._record(
            group="P2", key=key, scenario_id=state.scenario_id,
            microcycle=microcycle, step=step, context=context,
            candidate=candidate, flow_base=bases[0][selected][chosen],
            result=results[chosen], base_std=base_std,
            repair_attempt=attempt,
        ), attempts
    return None, None, attempts


@torch.inference_mode()
def targeted_four_step_recovery(
    adapter: PORT.HP100ExpansionPolicy,
    reference_adapter: PORT.HP100ExpansionPolicy,
    task: PORT.SFMHP100ExpansionTask,
    verifier: HYBRID._OrderedSidecarVerifier,
    *,
    reconstructed: dict,
    archived_raw_bases: Sequence[torch.Tensor],
    config: HYBRID.HybridConfig,
    lengthscale: float,
    support: torch.Tensor,
    microcycle: int = 2,
) -> dict:
    key = reconstructed["key"]
    state = reconstructed["state"]
    first_context = reconstructed["context"]
    posterior = RBFPosterior(float(lengthscale), float(config.rbf_noise))
    posterior.set_buffer(support)
    events, samples = [], []
    status = None
    raw_archived_bases_consumed = 0
    for offset in range(TARGETED_STEPS):
        step = int(reconstructed["prefix_start_step"]) + offset
        context = first_context if offset == 0 else task.context(state, key.gamma)
        if len(archived_raw_bases) != TARGETED_STEPS:
            raise ValueError("targeted audit requires four archived raw bases")
        raw_bases = archived_raw_bases[offset].reshape(1, HYBRID.H, 2).to(
            next(adapter.parameters()).device
        )
        raw_archived_bases_consumed += 1
        raw = _sample_from_bases(adapter, context, raw_bases)
        results, sidecars = verifier.verify_many([
            (context, raw, key.gamma),
        ])[0]
        if len(results) != 1 or results[0].error:
            raise RuntimeError("targeted raw proposal did not resolve exactly")
        result, candidate = results[0], raw[0]
        raw_sidecar = HYBRID._validated_sidecar(
            task, context, candidate, key.gamma, result, sidecars[0],
        )
        event = HYBRID._new_trace_event(
            key, state, microcycle=microcycle, step=step,
            context=context, raw_candidate=candidate,
            raw_result=result, raw_sidecar=raw_sidecar,
        )
        HYBRID._attach_scene_to_event(task, event, context)
        if result.valid:
            event["executed_group"] = "P1"
            samples.append(HYBRID._record(
                group="P1", key=key, scenario_id=state.scenario_id,
                microcycle=microcycle, step=step, context=context,
                candidate=candidate, flow_base=raw_bases[0], result=result,
                base_std=1.0, repair_attempt=None,
            ))
        else:
            samples.append(HYBRID._record(
                group="Dminus", key=key, scenario_id=state.scenario_id,
                microcycle=microcycle, step=step, context=context,
                candidate=candidate, flow_base=raw_bases[0], result=result,
                base_std=1.0, repair_attempt=None,
            ))
            candidate, p2, attempts = _targeted_repair(
                adapter, reference_adapter, task, verifier,
                state=state, context=context, key=key, step=step,
                config=config, posterior=posterior, microcycle=microcycle,
            )
            event["repair_attempts"] = attempts
            if candidate is None:
                event["terminal"] = "repair_exhausted"
                events.append(event)
                status = "repair_exhausted"
                break
            samples.append(p2)
            event["executed_group"] = "P2"
        state = task.advance(state, candidate)
        event["state_after"] = np.asarray(state.robot, np.float32).copy()
        terminal = task.terminal(state)
        if terminal is not None:
            event["terminal"] = str(terminal).lower()
            status = str(terminal).lower()
            events.append(event)
            break
        events.append(event)
    if status is None:
        status = "survived_N_plus_1"
    recovered = status in {"success", "survived_N_plus_1"}
    return {
        "lineage": key.label, "gamma": float(key.gamma),
        "disaster_step": int(reconstructed["disaster_step"]),
        "prefix_start_step": int(reconstructed["prefix_start_step"]),
        "targeted_steps": TARGETED_STEPS,
        "status": status, "recovered": bool(recovered),
        "raw_archived_bases_available": len(archived_raw_bases),
        "raw_archived_bases_consumed": int(raw_archived_bases_consumed),
        "executed_steps": sum(event.get("executed_group") in {"P1", "P2"} for event in events),
        "sample_counts": dict(sorted(Counter(row["group"] for row in samples).items())),
        "events": events, "samples": samples,
    }


def _diagnostic_checkpoint_payload(
    adapter: PORT.HP100ExpansionPolicy,
    *,
    checkpoint_sha256: str,
    optimizer_scope: str,
    role: str,
) -> dict:
    return {
        "scientific_status": "HP100_DISASTER_PREFIX_DIAGNOSTIC_ONLY",
        "state_dict": {
            key: value.detach().cpu().clone()
            for key, value in adapter.policy.state_dict().items()
        },
        "config": adapter.policy.config(),
        "parent_checkpoint_sha256": str(checkpoint_sha256),
        "optimizer_scope": str(optimizer_scope),
        "role": str(role), "promotable": False,
        "enters_GP": False,
    }


def run(args) -> dict:
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing disaster-prefix output: {output}")
    checkpoint_sha = HYBRID._sha256(args.checkpoint)
    if checkpoint_sha != str(args.expected_checkpoint_sha256).lower():
        raise RuntimeError("r0 checkpoint SHA256 mismatch")
    trace, source_samples, provenance = load_authenticated_inputs(
        trace_path=args.prior_trace,
        expected_trace_sha256=args.expected_prior_trace_sha256,
        staged_path=args.prior_staged_samples,
        expected_staged_sha256=args.expected_prior_staged_samples_sha256,
        checkpoint_sha256=checkpoint_sha,
        optimizer_scope=args.optimizer_scope,
    )
    training = extract_disaster_training_view(trace, source_samples)
    config = HYBRID.HybridConfig(**trace["config"])
    config = replace(
        config, max_relative_parameter_drift=float(args.max_relative_parameter_drift),
    )
    adapter, checkpoint_payload, _, trainable_names, trainable_count = (
        HYBRID._load_input_checkpoint(
            args.checkpoint, checkpoint_sha, args.device, args.optimizer_scope,
        )
    )
    pre_adapter, _, _, _, _ = HYBRID._load_input_checkpoint(
        args.checkpoint, checkpoint_sha, args.device, args.optimizer_scope,
    )
    reference_adapter, _, _, _, _ = HYBRID._load_input_checkpoint(
        args.checkpoint, checkpoint_sha, args.device, "head_only",
    )
    for parameter in pre_adapter.parameters():
        parameter.requires_grad_(False)
    for parameter in reference_adapter.parameters():
        parameter.requires_grad_(False)
    pre_adapter.eval(); reference_adapter.eval()
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
    ).attach_context_encoder(pre_adapter.policy)
    preflight = {
        "status": "SFM_HP100_DISASTER_PREFIX_PREFLIGHT_PASSED",
        "version": VERSION, "source": provenance,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_scientific_status": checkpoint_payload.get("scientific_status"),
        "optimizer_scope": args.optimizer_scope,
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_count": trainable_count,
        "config": asdict(config),
        "training_view": _exclusive_training_counts(training),
        "update": {
            "P2_lr": float(args.p2_learning_rate),
            "P2_passes": int(args.p2_passes),
            "P1_lr": float(args.p1_learning_rate),
            "P1_passes": int(args.p1_passes),
            "negative_lr": float(args.negative_learning_rate),
            "negative_alpha": float(args.negative_alpha),
            "negative_passes": int(args.negative_passes),
            "negative_source": str(args.negative_source),
            "negative_effective_lr": (
                float(args.negative_learning_rate) * float(args.negative_alpha)
            ),
            "N": CAUSAL_N,
            "one_exposure_per_row": bool(
                int(args.p1_passes) == int(args.p2_passes)
                == int(args.negative_passes) == 1
            ),
        },
        "audit": {
            "fixed_raw_base_M": AUDIT_M,
            "targeted_steps": TARGETED_STEPS,
            "minimum_recovered_for_full_rerun": int(args.min_recovered),
            "render_diagnostic_regardless": bool(args.render_diagnostic_regardless),
        },
        "gpu": BASE._gpu_contract(args.device, args.physical_gpu),
    }
    preflight_path = output.with_suffix(output.suffix + ".preflight.json")
    preflight_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(preflight_path, preflight)
    if args.mode == "preflight":
        return preflight
    output.mkdir(parents=True)
    _write_json(output / "PREFLIGHT.json", preflight)

    update = apply_disaster_prefix_update(
        adapter, training,
        p2_learning_rate=float(args.p2_learning_rate),
        p1_learning_rate=float(args.p1_learning_rate),
        negative_learning_rate=float(args.negative_learning_rate),
        alpha=float(args.negative_alpha), batch_size=int(args.batch_size),
        max_relative_parameter_drift=float(args.max_relative_parameter_drift),
        seed=config.seed, p1_passes=int(args.p1_passes),
        p2_passes=int(args.p2_passes),
        negative_passes=int(args.negative_passes),
        negative_source=str(args.negative_source),
    )
    positive_anchor_state = update.pop("_positive_anchor_state_dict")
    anchor_checkpoint = output / "checkpoint_alpha0_positive_anchor.pt"
    # The alpha-zero control is the one shared positive anchor snapshot.  It
    # is not a second stochastic rerun of phases 1--2.
    anchor_adapter, _, _, _, _ = HYBRID._load_input_checkpoint(
        args.checkpoint, checkpoint_sha, args.device, args.optimizer_scope,
    )
    anchor_adapter.load_state_dict(positive_anchor_state, strict=True)
    if _model_state_sha256(anchor_adapter) != update["positive_anchor_model_sha256"]:
        raise RuntimeError("alpha0 anchor replay is not bitwise identical")
    _torch_save(anchor_checkpoint, _diagnostic_checkpoint_payload(
        anchor_adapter, checkpoint_sha256=checkpoint_sha,
        optimizer_scope=args.optimizer_scope, role="alpha0_positive_anchor",
    ))
    if not update["accepted"]:
        marker = {
            "status": REJECTED_STATUS, "version": VERSION,
            "preflight": preflight, "update": update,
            "persistent_archive_committed": False,
            "enters_GP": False, "full_rerun_performed": False,
        }
        _write_json(output / "UPDATE_REJECTED.json", marker)
        return marker
    treatment_checkpoint = output / "checkpoint_treatment_diagnostic.pt"
    _torch_save(treatment_checkpoint, _diagnostic_checkpoint_payload(
        adapter, checkpoint_sha256=checkpoint_sha,
        optimizer_scope=args.optimizer_scope,
        role=(
            "positive_anchor_plus_Dminus_Ncausal_negative_ascent"
            if args.negative_source == "Dminus_plus_Ncausal" else
            "positive_anchor_plus_Ncausal_negative_ascent"
        ),
    ))

    events_by_lineage = _events_by_lineage(trace)
    samples_by_identity = _samples_by_identity(source_samples)
    reconstructed_audit, reconstruction_reports = [], []
    archived_raw_bases = {}
    for disaster in training["disasters"]:
        key = _key_from_label(disaster["lineage"])
        state, context, report = reconstruct_prefix_state(
            task, key=key, target_step=disaster["prefix_start_step"],
            config=config, events_by_lineage=events_by_lineage,
            samples_by_identity=samples_by_identity,
        )
        reconstructed_audit.append({
            **disaster, "key": key, "state": state, "context": context,
        })
        reconstruction_reports.append(report)
        archived_raw_bases[key.label] = archived_raw_base_sequence(
            disaster=disaster, events_by_lineage=events_by_lineage,
            samples_by_identity=samples_by_identity,
        )

    with HYBRID._OrderedSidecarVerifier(task, int(args.verifier_workers)) as verifier:
        m64, _fixed_m64_bases = fixed_raw_m64_audit(
            pre_adapter, anchor_adapter, adapter, task, verifier,
            reconstructed_audit,
            seed=config.seed,
        )
        targeted_r0 = []
        targeted_alpha0 = []
        targeted_treatment = []
        targeted_events = []
        targeted_samples = []
        # R0, alpha0, and treatment each reconstruct fresh mutable SFM state
        # and use identical archived raw x0 at t-3..t. Repair seeds are also
        # identical, so recovery changes are attributable to model changes.
        for model_label, model, destination in (
            ("r0", pre_adapter, targeted_r0),
            ("alpha0_positive_anchor", anchor_adapter, targeted_alpha0),
            ("treatment", adapter, targeted_treatment),
        ):
            for disaster in training["disasters"]:
                key = _key_from_label(disaster["lineage"])
                state, context, report = reconstruct_prefix_state(
                    task, key=key, target_step=disaster["prefix_start_step"],
                    config=config, events_by_lineage=events_by_lineage,
                    samples_by_identity=samples_by_identity,
                )
                reconstruction_reports.append({
                    **report, "purpose": "targeted_recovery",
                    "model": model_label,
                })
                result = targeted_four_step_recovery(
                    model, reference_adapter, task, verifier,
                    reconstructed={
                        **disaster, "key": key, "state": state,
                        "context": context,
                    },
                    archived_raw_bases=archived_raw_bases[key.label],
                    config=config, lengthscale=lengthscale,
                    support=support_by_gamma[float(key.gamma)],
                )
                events = result.pop("events")
                samples = result.pop("samples")
                for event in events:
                    event["audit_model"] = model_label
                for row in samples:
                    row["audit_model"] = model_label
                targeted_events.extend(events)
                targeted_samples.extend(samples)
                destination.append({**result, "model": model_label})
        attribution = _targeted_recovery_attribution(
            targeted_alpha0, targeted_treatment, targeted_r0,
        )
        r0_recovered = attribution["r0_recovered"]
        alpha0_recovered = attribution["alpha0_recovered"]
        recovered = attribution["treatment_recovered"]
        incremental_recoveries = attribution[
            "incremental_recoveries_treatment_over_alpha0"
        ]
        incremental_regressions = attribution[
            "incremental_regressions_treatment_vs_alpha0"
        ]
        alpha_effect_supported = attribution[
            "alpha_effect_supported_by_recovery_count"
        ]
        gate_passed, full_rerun_performed = _full_rerun_decision(
            recovered, int(args.min_recovered),
            bool(args.render_diagnostic_regardless),
        )
        full = None
        if full_rerun_performed:
            full = HYBRID.gather_hybrid(
                adapter, reference_adapter, task,
                keys=HYBRID._lineage_keys(config), config=config,
                lengthscale=lengthscale, support_by_gamma=support_by_gamma,
                # Reuse the original round-1 sampling coordinates for a direct
                # treatment-vs-r0 trace comparison.
                microcycle=1, verifier=verifier,
            )

    m64_path = output / "fixed_raw_M64_pre_post.pt"
    _torch_save(m64_path, {
        "status": "SFM_HP100_DISASTER_PREFIX_FIXED_RAW_M64_AUDIT",
        "M": AUDIT_M, "rows": m64,
    })
    targeted_trace_path = output / "targeted_Nplus1_trace.pt"
    _torch_save(targeted_trace_path, {
        "status": "SFM_HP100_DISASTER_PREFIX_TARGETED_TRACE",
        "N": CAUSAL_N, "N_plus_1": TARGETED_STEPS,
        "CRN": "archived raw flow_base at original t-3..t; shared repair seeds",
        "events": targeted_events, "samples": targeted_samples,
    })
    training_view_path = output / "disaster_training_view.pt"
    _torch_save(training_view_path, {
        "status": "SFM_HP100_DISASTER_PREFIX_TRAINING_VIEW",
        "enters_GP": False, "source_rows_unchanged": True,
        "training_view": training,
    })

    full_trace_path = None
    full_samples_path = None
    full_summary = None
    if full is not None:
        full_exhausted = sum(
            str(event.get("terminal", "")).lower() == "repair_exhausted"
            for event in full["events"]
        )
        full_training = extract_disaster_training_view(
            {"events": full["events"]}, full["samples"],
            expected_exhausted=full_exhausted, require_full_prefix=False,
        )
        causal_by_event = {
            (identity[0], int(identity[1]), int(identity[2])): identity
            for identity in full_training["causal_source_identities"]
        }
        for event in full["events"]:
            event_key = _event_identity(event)
            identity = causal_by_event.get(event_key)
            event["causal_negative"] = identity is not None
            if identity is not None:
                event["causal_source_group"] = identity[3]
                event["causal_N"] = CAUSAL_N
        full_count_report = _exclusive_training_counts(full_training)
        treatment_trace_provenance = _treatment_trace_provenance(
            treatment_checkpoint=treatment_checkpoint,
            treatment_policy_state_sha256=update[
                "treatment_model_sha256_after_gate"
            ],
            parent_r0_checkpoint_sha256=checkpoint_sha,
            source_incomplete_trace_sha256=provenance["trace_sha256"],
        )
        full_trace = {
            "status": "SFM_HP100_DISASTER_PREFIX_FULL_RERUN_TRACE_DIAGNOSTIC_ONLY",
            "version": HYBRID.TRACE_VERSION, "config": asdict(config),
            "optimizer_scope": args.optimizer_scope,
            # ``checkpoint_sha256`` names the policy that generated this trace,
            # never the r0 parent. This prevents legacy consumers from
            # authenticating treatment behavior as pretrained behavior.
            **treatment_trace_provenance,
            "events": full["events"], "raw_rechecks": [],
            "cycles": [{
                "microcycle": 1,
                "raw_source_counts": full_count_report["raw_source_counts"],
                "mutually_exclusive_training_view_counts": full_count_report[
                    "mutually_exclusive_training_view_counts"
                ],
                "outcomes": full["outcomes"],
                "all_hybrid_success": full["all_hybrid_success"],
                "update": "none_diagnostic_rerun_only", "recheck": None,
            }],
            "scientific_label": (
                "treatment_behavior_gate_passed_diagnostic_full_rerun"
                if gate_passed else
                "forced_diagnostic_below_treatment_behavior_gate"
            ),
            "alpha_effect_supported_by_targeted_recovery_count": bool(
                alpha_effect_supported
            ),
        }
        full_trace_path = output / "exhaustive_hybrid_trace.pt"
        _torch_save(full_trace_path, full_trace)
        full_samples_path = output / "full_rerun_samples_diagnostic_only.pt"
        _torch_save(full_samples_path, {
            "status": "SFM_HP100_DISASTER_PREFIX_FULL_RERUN_SAMPLES_DIAGNOSTIC_ONLY",
            "enters_GP": False, "second_update_performed": False,
            "samples": full["samples"],
            "Ncausal_training_view": full_training["Ncausal"],
            "raw_source_counts": full_count_report["raw_source_counts"],
            "mutually_exclusive_training_view_counts": full_count_report[
                "mutually_exclusive_training_view_counts"
            ],
        })
        full_summary = {
            key: value for key, value in full.items()
            if key not in {"events", "samples"}
        }
        full_summary["raw_source_counts"] = full_count_report["raw_source_counts"]
        full_summary["mutually_exclusive_training_view_counts"] = (
            full_count_report["mutually_exclusive_training_view_counts"]
        )
        full_summary["N_semantics"] = full_training["semantics"]

    marker = {
        "status": STATUS, "version": VERSION, "preflight": preflight,
        "training_view": _exclusive_training_counts(training), "update": update,
        "reconstruction": reconstruction_reports,
        "fixed_raw_M64": {
            "artifact": str(m64_path), "sha256": HYBRID._sha256(m64_path),
            "rows": [{
                key: value for key, value in row.items()
                if key not in {
                    "r0_plans", "alpha0_plans", "treatment_plans",
                    "fixed_bases",
                }
            } for row in m64],
        },
        "targeted_Nplus1": {
            "N": CAUSAL_N, "steps": TARGETED_STEPS,
            "r0_recovered": int(r0_recovered),
            "alpha0_recovered": int(alpha0_recovered),
            "positive_anchor_minus_r0_recovered": int(
                attribution["positive_anchor_minus_r0_recovered"]
            ),
            "treatment_recovered": int(recovered),
            "incremental_recoveries_treatment_over_alpha0": int(
                incremental_recoveries
            ),
            "incremental_regressions_treatment_vs_alpha0": int(
                incremental_regressions
            ),
            "net_incremental_recoveries": int(
                incremental_recoveries - incremental_regressions
            ),
            "alpha_effect_supported_by_recovery_count": bool(
                alpha_effect_supported
            ),
            "alpha_effect_claim_rule": (
                "true only when treatment recovered count is strictly greater "
                "than the shared positive-anchor alpha0 control"
            ),
            "total_per_model": len(targeted_treatment),
            "minimum_for_full_rerun": int(args.min_recovered),
            "treatment_behavior_gate_passed": bool(gate_passed),
            "full_rerun_gate_is_not_an_alpha_attribution_claim": True,
            "r0_rows": targeted_r0,
            "alpha0_rows": targeted_alpha0,
            "treatment_rows": targeted_treatment,
            "treatment_minus_alpha0": attribution["rows"],
            "trace": str(targeted_trace_path),
            "trace_sha256": HYBRID._sha256(targeted_trace_path),
        },
        "full_rerun": {
            "performed": bool(full is not None),
            "forced_below_gate": bool(full is not None and not gate_passed),
            "summary": full_summary,
            "trace": None if full_trace_path is None else str(full_trace_path),
            "trace_sha256": (
                None if full_trace_path is None else HYBRID._sha256(full_trace_path)
            ),
            "samples": None if full_samples_path is None else str(full_samples_path),
            "samples_sha256": (
                None if full_samples_path is None else HYBRID._sha256(full_samples_path)
            ),
            "second_update_performed": False, "enters_GP": False,
        },
        "checkpoints": {
            "alpha0_positive_anchor": str(anchor_checkpoint),
            "alpha0_positive_anchor_sha256": HYBRID._sha256(anchor_checkpoint),
            "treatment_diagnostic": str(treatment_checkpoint),
            "treatment_diagnostic_sha256": HYBRID._sha256(treatment_checkpoint),
            "promotable": False,
        },
        "training_view": str(training_view_path),
        "training_view_sha256": HYBRID._sha256(training_view_path),
        "persistent_archive_committed": False, "GP_committed": False,
    }
    _write_json(output / "DIAGNOSTIC_COMPLETE.json", marker)
    return marker


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--mode", choices=("preflight", "run"), default="preflight")
    value.add_argument("--checkpoint", required=True)
    value.add_argument("--expected-checkpoint-sha256", required=True)
    value.add_argument("--prior-trace", required=True)
    value.add_argument("--expected-prior-trace-sha256", required=True)
    value.add_argument("--prior-staged-samples", required=True)
    value.add_argument("--expected-prior-staged-samples-sha256", required=True)
    value.add_argument("--pretrain-dataset-root", required=True)
    value.add_argument("--expected-pretrain-dataset-manifest-sha256", required=True)
    value.add_argument("--output", required=True)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--physical-gpu", type=int, required=True)
    value.add_argument(
        "--optimizer-scope", choices=("head_only", "last_block_and_head"),
        required=True,
    )
    value.add_argument(
        "--scene-profile", default=PORT.DEFAULT_SCENE_PROFILE,
        choices=SS.SCIENTIFIC_EVAL_PROFILES,
    )
    value.add_argument("--scenario-start", type=int, default=500_000)
    value.add_argument("--verifier-workers", type=int, default=16)
    value.add_argument("--p2-learning-rate", type=float, default=1.0e-3)
    value.add_argument("--p2-passes", type=int, default=1)
    value.add_argument("--p1-learning-rate", type=float, default=1.0e-5)
    value.add_argument("--p1-passes", type=int, default=1)
    value.add_argument("--negative-learning-rate", type=float, default=1.0e-4)
    value.add_argument("--negative-alpha", type=float, default=0.01)
    value.add_argument("--negative-passes", type=int, default=1)
    value.add_argument(
        "--negative-source", choices=NEGATIVE_SOURCES,
        default="Dminus_plus_Ncausal",
    )
    value.add_argument("--batch-size", type=int, default=64)
    value.add_argument("--max-relative-parameter-drift", type=float, default=0.25)
    value.add_argument("--min-recovered", type=int, default=4)
    value.add_argument("--render-diagnostic-regardless", action="store_true")
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if not 0 <= int(args.min_recovered) <= EXPECTED_EXHAUSTED:
        raise ValueError("min-recovered must lie in [0,6]")
    if args.p1_learning_rate <= 0.0 or args.p2_learning_rate <= 0.0:
        raise ValueError("P1/P2 learning rates must be positive")
    if args.negative_learning_rate < 0.0 or args.negative_alpha < 0.0:
        raise ValueError("negative learning rate and alpha must be nonnegative")
    if min(args.p1_passes, args.p2_passes, args.negative_passes) < 1:
        raise ValueError("P1, P2, and negative passes must be positive")
    if args.batch_size < 1 or args.verifier_workers < 1:
        raise ValueError("batch size and verifier workers must be positive")
    marker = run(args)
    print(json.dumps({
        "status": marker["status"], "output": str(Path(args.output).resolve()),
    }), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
