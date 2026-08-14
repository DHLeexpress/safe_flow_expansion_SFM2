"""Five-round cumulative HP100 exhaustive-hybrid expansion cell.

This default-off study is the cumulative counterpart of
``sfm_hp100_iterative_microcycles``.  It deliberately keeps one optimizer and
one policy lineage alive for five micro-rounds: there is no scientific
rollback between rounds.  Gathering remains the exact exhaustive-hybrid
mechanism (raw P1 first, uncertainty-selected exact-positive P2 repair after a
raw Dminus), while replay uses the declared sparse-group intervention

    mean(P1) + w2 mean(P2) - alpha mean(Ncausal) - alpha mean(Dminus).

The signed negative CFM terms are an ablation, not a normalized likelihood
objective.  Their rows retain their exact verifier labels and provenance.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

import _paths  # noqa: F401
import grid_policy_sfm_hp100 as GPS
import sfm_hp100_ball_adapter as PORT
from sfm_hp100_ball_core.expansion import _counter_seed, mean_pairwise_lengthscale
import sfm_hp100_ball_launch as BASE
import sfm_hp100_disaster_prefix_continuation as PREFIX
import sfm_hp100_disaster_prefix_sweep as SOURCE
import sfm_hp100_exhaustive_hybrid as HYBRID
import sfm_hp100_iterative_microcycles as ITER


VERSION = "sfm_hp100_cumulative_microblocks_v1"
STATUS = "SFM_HP100_CUMULATIVE_MICROBLOCK_ARM_COMPLETE"
GAMMAS = (0.1, 0.3, 0.5, 1.0)
QUOTAS = {"P1": 32, "P2": 16, "Ncausal": 12, "Dminus": 4}
EXPECTED_COUNTS = {
    "last_block_and_head": 137_236,
    "last_two_blocks_and_head": 269_332,
}


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def _torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _cpu_optimizer_state(optimizer: torch.optim.Optimizer) -> dict:
    state = optimizer.state_dict()
    return {
        "state": {
            key: {
                name: value.detach().cpu().clone()
                if isinstance(value, torch.Tensor) else value
                for name, value in row.items()
            }
            for key, row in state["state"].items()
        },
        "param_groups": state["param_groups"],
    }


def _checkpoint_payload(
    adapter: PORT.HP100ExpansionPolicy,
    optimizer: torch.optim.Optimizer,
    *,
    parent_checkpoint_sha256: str,
    reference_checkpoint_sha256: str,
    optimizer_scope: str,
    learning_rate: float,
    p2_weight: float,
    negative_alpha: float,
    micro_round: int,
) -> dict:
    return {
        "scientific_status": "HP100_CUMULATIVE_MICROBLOCK_DIAGNOSTIC_ONLY",
        "state_dict": {
            key: value.detach().cpu().clone()
            for key, value in adapter.policy.state_dict().items()
        },
        "config": adapter.policy.config(),
        "parent_checkpoint_sha256": str(parent_checkpoint_sha256),
        "pretrained_checkpoint_sha256": str(reference_checkpoint_sha256),
        "optimizer_scope": str(optimizer_scope),
        "learning_rate": float(learning_rate),
        "p2_weight": float(p2_weight),
        "negative_alpha": float(negative_alpha),
        "micro_round": int(micro_round),
        "optimizer_state_dict": _cpu_optimizer_state(optimizer),
        "persistent_optimizer": True,
        "promotable": False,
        "disjoint_evaluation_required": True,
    }


def _configure_trainable(
    adapter: PORT.HP100ExpansionPolicy, scope: str,
) -> tuple[list[torch.nn.Parameter], list[str]]:
    parameters = adapter.expansion_optimizer_parameters(scope)
    count = sum(parameter.numel() for parameter in parameters)
    if count != EXPECTED_COUNTS.get(scope):
        raise RuntimeError(
            f"{scope} parameter count drifted: {count} != "
            f"{EXPECTED_COUNTS.get(scope)}"
        )
    names = sorted(
        name for name, parameter in adapter.policy.named_parameters()
        if parameter.requires_grad
    )
    block_ids = (1,) if scope == "last_block_and_head" else (0, 1)
    if not names or any(
        not (
            name.startswith("head.")
            or any(name.startswith(f"trunk.blocks.{index}.") for index in block_ids)
        )
        for name in names
    ):
        raise RuntimeError("conditioning or trunk-input parameters became trainable")
    return parameters, names


def _load_inputs(args):
    marker, _, selection = ITER.load_selection(
        args.selection_marker, args.expected_selection_marker_sha256,
    )
    selected = selection["checkpoint"]
    selected_path = str(Path(args.checkpoint).resolve())
    if selected_path != selected["path"]:
        raise RuntimeError("checkpoint is not the authenticated selected checkpoint")
    if HYBRID._sha256(selected_path) != str(args.expected_checkpoint_sha256).lower():
        raise RuntimeError("selected checkpoint SHA256 mismatch")
    if selected["sha256"] != str(args.expected_checkpoint_sha256).lower():
        raise RuntimeError("selection marker points to a different checkpoint SHA256")
    policy, payload = GPS.load_sfm_hp100_policy(selected_path, device=args.device)
    adapter = PORT.HP100ExpansionPolicy(policy)
    parameters, trainable_names = _configure_trainable(
        adapter, args.optimizer_scope,
    )
    if HYBRID._model_state_sha256(adapter) != selected["policy_state_sha256"]:
        raise RuntimeError("selected policy-state SHA256 mismatch")

    reference_path = str(Path(args.reference_checkpoint).resolve())
    reference_sha = HYBRID._sha256(reference_path)
    if reference_sha != str(args.expected_reference_checkpoint_sha256).lower():
        raise RuntimeError("reference checkpoint SHA256 mismatch")
    reference_policy, reference_payload = GPS.load_sfm_hp100_policy(
        reference_path, device=args.device,
    )
    if reference_payload.get("scientific_status") != "canonical_ID_promoted":
        raise ValueError("reference checkpoint is not canonical HP100 r0")
    reference = PORT.HP100ExpansionPolicy(reference_policy)
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    reference.eval()

    reference_state = reference.state_dict()
    trainable_set = set(trainable_names)
    for name, value in adapter.policy.state_dict().items():
        if name not in trainable_set and not torch.equal(
            value.detach().cpu(), reference_state[name].detach().cpu(),
        ):
            raise RuntimeError(f"frozen parameter differs from canonical r0: {name}")
    return (
        marker, selection, adapter, payload, parameters, trainable_names,
        reference, reference_payload, reference_sha,
    )


def _embed_rows(
    adapter: PORT.HP100ExpansionPolicy,
    rows: Sequence[dict],
    *,
    chunk_size: int = 512,
) -> torch.Tensor:
    device = next(adapter.parameters()).device
    output = []
    with torch.inference_mode():
        for start in range(0, len(rows), int(chunk_size)):
            block = rows[start:start + int(chunk_size)]
            contexts = torch.stack([row["context"] for row in block]).to(device)
            candidates = torch.stack([row["candidate"] for row in block]).to(device)
            bases = torch.stack([row["flow_base"] for row in block]).to(device)
            output.append(
                adapter.embed(contexts, candidates, base=bases).detach()
            )
    if not output:
        return torch.empty((0, 256), device=device)
    return torch.cat(output)


def gp_support_for_round(
    *,
    reference: PORT.HP100ExpansionPolicy,
    calibration_support: dict[float, torch.Tensor],
    previous_support: dict[float, torch.Tensor],
    history: Sequence[Sequence[dict]],
    cap: int,
    replay_rounds: int,
) -> tuple[dict[float, torch.Tensor], dict]:
    """Build frozen-r0 support from exact-positive rows in recent rounds."""
    if int(replay_rounds) != 2:
        raise ValueError("the cumulative protocol fixes two GP replay rounds")
    recent = [
        row for round_rows in history[-int(replay_rounds):]
        for row in round_rows if row["group"] in {"P1", "P2"}
    ]
    selected = (
        [recent[index] for index in HYBRID._balanced_gp_indices(recent, int(cap))]
        if recent else []
    )
    grouped: defaultdict[float, list[dict]] = defaultdict(list)
    for row in selected:
        grouped[float(row["gamma"])].append(row)
    support = {}
    fallback = {}
    for gamma in GAMMAS:
        rows = grouped.get(float(gamma), [])
        if rows:
            support[float(gamma)] = _embed_rows(reference, rows)
            fallback[str(gamma)] = None
        else:
            inherited = previous_support.get(
                float(gamma), calibration_support[float(gamma)],
            )
            support[float(gamma)] = inherited
            fallback[str(gamma)] = (
                "previous_or_calibration_support_no_recent_positive"
            )
    report = {
        "candidate_recent_positive_rows": len(recent),
        "selected_rows": len(selected),
        "literal_total_cap": int(cap),
        "active_gammas": len(GAMMAS),
        "nominal_cap_per_active_gamma": int(cap) // len(GAMMAS),
        "rows_by_gamma": {
            str(gamma): len(grouped.get(float(gamma), ())) for gamma in GAMMAS
        },
        "support_rows_by_gamma": {
            str(gamma): int(len(support[float(gamma)])) for gamma in GAMMAS
        },
        "fallback_by_gamma": fallback,
        "selector": "gamma-lineage balanced, temporal-uniform within lineage",
        "representation": "canonical-r0 paired-noised phi(s=0.9)",
        "source_round_window": int(replay_rounds),
    }
    return support, report


def training_view(gather: dict) -> dict:
    samples = list(gather["samples"])
    exhausted = sum(
        str(event.get("terminal", "")).lower() == "repair_exhausted"
        for event in gather["events"]
    )
    view = PREFIX.extract_disaster_training_view(
        {"events": gather["events"]}, samples,
        causal_n=3, expected_exhausted=exhausted,
        require_full_prefix=False,
    )
    groups = {
        "P1": list(view["retained_P1"]),
        "P2": list(view["retained_P2"]),
        "Ncausal": list(view["Ncausal"]),
        "Dminus": list(view["Dminus"]),
    }
    counts = {name: len(rows) for name, rows in groups.items()}
    return {
        "groups": groups,
        "counts": counts,
        "source_counts": dict(sorted(Counter(
            row["group"] for row in samples
        ).items())),
        "repair_exhausted_lineages": int(exhausted),
        "short_prefix_lengths": [
            len(row["causal_source_identities"]) for row in view["disasters"]
        ],
        "causal_source_identities": view["causal_source_identities"],
        "semantics": view["semantics"],
    }


def _cyclic_epoch_indices(
    count: int, draws: int, *, seed: int,
) -> np.ndarray:
    if count <= 0:
        return np.empty(0, np.int64)
    if draws < count:
        raise ValueError("epoch capacity cannot omit a unique source row")
    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(int(count))
    return np.resize(permutation, int(draws)).astype(np.int64, copy=False)


def stratified_signed_update(
    adapter: PORT.HP100ExpansionPolicy,
    optimizer: torch.optim.Optimizer,
    groups: dict[str, Sequence[dict]],
    *,
    p2_weight: float,
    negative_alpha: float,
    passes: int,
    seed: int,
    micro_round: int,
) -> dict:
    if set(groups) != set(QUOTAS):
        raise ValueError("training groups differ from the declared four streams")
    if int(passes) != 10:
        raise ValueError("the cumulative protocol fixes ten effective passes")
    counts = {name: len(groups[name]) for name in QUOTAS}
    batch_requirements = [
        math.ceil(counts[name] / quota)
        for name, quota in QUOTAS.items() if counts[name]
    ]
    n_batches = max(batch_requirements, default=0)
    parameters = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("cumulative update has no trainable parameters")
    device = parameters[0].device
    before = HYBRID._parameter_snapshot(parameters)
    exposures = {name: np.zeros(counts[name], np.int64) for name in QUOTAS}
    loss_sums = Counter()
    objective_values = []
    gradient_norms = []

    for pass_index in range(int(passes)):
        epoch_indices = {}
        for name, quota in QUOTAS.items():
            epoch_indices[name] = _cyclic_epoch_indices(
                counts[name], n_batches * quota,
                seed=_counter_seed(
                    int(seed), "cumulative_stratified_order", micro_round,
                    pass_index, name,
                ),
            )
        for batch_index in range(n_batches):
            optimizer.zero_grad(set_to_none=True)
            means = {}
            for group_index, (name, quota) in enumerate(QUOTAS.items()):
                indices = epoch_indices[name][
                    batch_index * quota:(batch_index + 1) * quota
                ]
                if not len(indices):
                    means[name] = torch.zeros((), device=device)
                    continue
                np.add.at(exposures[name], indices, 1)
                rows = [groups[name][int(index)] for index in indices]
                contexts, candidates = HYBRID._stack_rows(rows, device)
                HYBRID._set_step_seed(_counter_seed(
                    int(seed), "cumulative_cfm_noise", micro_round,
                    pass_index, batch_index, group_index,
                ), device)
                means[name] = adapter.cfm_loss(
                    contexts, candidates, reduction="none",
                ).mean()
                loss_sums[name] += float(means[name].detach())
            objective = (
                means["P1"] + float(p2_weight) * means["P2"]
                - float(negative_alpha) * means["Ncausal"]
                - float(negative_alpha) * means["Dminus"]
            )
            if not bool(torch.isfinite(objective)):
                raise RuntimeError("non-finite signed CFM objective")
            objective.backward()
            squared = torch.zeros((), device=device)
            for parameter in parameters:
                if parameter.grad is not None:
                    squared += parameter.grad.square().sum()
            gradient_norm = float(torch.sqrt(squared).detach())
            if not math.isfinite(gradient_norm):
                raise RuntimeError("non-finite cumulative gradient")
            optimizer.step()
            if not HYBRID._finite_parameters(parameters):
                raise RuntimeError("non-finite model parameters after Adam step")
            objective_values.append(float(objective.detach()))
            gradient_norms.append(gradient_norm)

    total_steps = int(passes) * n_batches
    exposure_report = {}
    for name, values in exposures.items():
        exposure_report[name] = {
            "unique_rows": int(len(values)),
            "total_draws": int(values.sum()),
            "minimum": (None if not len(values) else int(values.min())),
            "maximum": (None if not len(values) else int(values.max())),
            "all_unique_rows_covered_each_pass": (
                True if not len(values) else bool(values.min() >= int(passes))
            ),
        }
    return {
        "counts": counts,
        "quotas_per_batch": dict(QUOTAS),
        "effective_passes": int(passes),
        "N_batches": int(n_batches),
        "N_Adam_steps": int(total_steps),
        "expected_N_Adam_steps": int(passes) * int(n_batches),
        "persistent_optimizer": True,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "p2_weight": float(p2_weight),
        "negative_alpha_Ncausal": float(negative_alpha),
        "negative_alpha_Dminus": float(negative_alpha),
        "objective": (
            "mean_CFM(P1)+w2*mean_CFM(P2)-alpha*mean_CFM(Ncausal)"
            "-alpha*mean_CFM(Dminus)"
        ),
        "group_loss_mean_over_steps": {
            name: (None if total_steps == 0 else float(loss_sums[name] / total_steps))
            for name in QUOTAS
        },
        "signed_objective_mean": (
            None if not objective_values else float(np.mean(objective_values))
        ),
        "gradient_norm": {
            "mean": (None if not gradient_norms else float(np.mean(gradient_norms))),
            "maximum": (None if not gradient_norms else float(max(gradient_norms))),
        },
        "exposure": exposure_report,
        "relative_parameter_drift_this_round": HYBRID._relative_parameter_drift(
            parameters, before,
        ),
        "blind_spots": [
            "signed negative CFM ascent is unbounded and is not a normalized likelihood suppression objective",
            "fixed 32/16/12/4 quotas deliberately replace empirical group prevalence",
            "stored flow_base pairs phi/GP only; CFM draws fresh x0 and tau",
        ],
    }


def _raw_summary(recheck: dict, keys) -> dict:
    return ITER._raw_gate_summary(recheck, keys)


def _transition_rows(previous: dict, current: dict, micro_round: int) -> list[dict]:
    rows = []
    for lineage in sorted(current["outcomes"]):
        before = previous["outcomes"][lineage]
        after = current["outcomes"][lineage]
        if not bool(before["clear"]) and bool(after["clear"]):
            rows.append({
                "lineage": lineage, "micro_round": int(micro_round),
                "before_status": str(before["status"]),
                "after_status": str(after["status"]),
                "before_steps": int(before["steps"]),
                "after_steps": int(after["steps"]),
            })
    return rows


def run(args) -> dict:
    output = Path(args.output_root).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing cell output: {output}")
    if int(args.micro_rounds) != 5 or int(args.replicas_per_gamma) != 8:
        raise ValueError("declared block is exactly 5 rounds and 8 replicas/gamma")
    if int(args.passes) != 10 or int(args.batch_size) != 64:
        raise ValueError("declared replay fixes 10 passes and batch size 64")
    if int(args.gp_buffer_cap) != 2_688 or int(args.gp_replay_rounds) != 2:
        raise ValueError("declared GP contract fixes cap 2688 and two rounds")
    if float(args.p2_weight) != 2.0:
        raise ValueError("declared C1-C4 cells fix P2 weight 2")
    source_gate = SOURCE.source_gate(
        Path(args.source_root).resolve(), args.expected_source_commit,
    )
    gpu = BASE._gpu_contract(args.device, int(args.physical_gpu))
    (
        selection_marker, selection, adapter, selected_payload, trainable,
        trainable_names, reference, reference_payload, reference_sha,
    ) = _load_inputs(args)

    config = HYBRID.HybridConfig(
        gammas=GAMMAS,
        lineages_per_gamma=int(args.replicas_per_gamma),
        max_steps=int(args.max_steps),
        max_repair_batches=int(args.max_repair_batches),
        ess_target=float(args.ess_target),
        rbf_noise=float(args.rbf_noise),
        batch_size=int(args.batch_size),
        max_microcycles=int(args.micro_rounds),
        gp_buffer_cap=int(args.gp_buffer_cap),
        seed=int(args.seed),
    )
    config.validate()
    keys = HYBRID._lineage_keys(config)
    if len(keys) != 32:
        raise RuntimeError("cumulative block must contain 32 gamma-lineages")

    features, calibration = BASE.calibration_features(
        reference,
        dataset_root=args.pretrain_dataset_root,
        expected_manifest_sha256=args.expected_pretrain_dataset_manifest_sha256,
        count=50, seed=int(args.seed), base_std=1.0,
        paired_noised_representation=True,
    )
    lengthscale = mean_pairwise_lengthscale(features)
    calibration_support = HYBRID._calibration_support_by_gamma(
        features, calibration, config.gammas,
        device=next(reference.parameters()).device,
    )
    previous_support = dict(calibration_support)
    task = PORT.SFMHP100ExpansionTask(
        scene_profile=args.scene_profile,
        scenario_start=int(args.scenario_start),
    ).attach_context_encoder(reference.policy)
    lineage_contract = ITER._lineage_contract(task, keys, config)
    training_scenario_ids = sorted({
        int(row["scenario_id"])
        for row in lineage_contract["semantic"]["rows"]
    })
    if len(training_scenario_ids) != int(args.replicas_per_gamma):
        raise RuntimeError("gamma-paired lineages did not produce eight scenes")

    preflight = {
        "status": "SFM_HP100_CUMULATIVE_MICROBLOCK_PREFLIGHT_PASSED",
        "version": VERSION,
        "source": source_gate,
        "source_components": {
            Path(path).name: HYBRID._sha256(path)
            for path in (
                __file__, PORT.__file__, HYBRID.__file__, PREFIX.__file__,
                ITER.__file__,
            )
        },
        "gpu": gpu,
        "selection_marker": {
            "path": str(Path(args.selection_marker).resolve()),
            "sha256": str(args.expected_selection_marker_sha256).lower(),
            "status": selection_marker.get("status"),
        },
        "start_checkpoint": selection["checkpoint"],
        "start_checkpoint_payload": {
            key: selected_payload.get(key) for key in (
                "scientific_status", "optimizer_scope", "role", "promotable",
                "enters_GP",
            )
        },
        "reference_checkpoint": str(Path(args.reference_checkpoint).resolve()),
        "reference_checkpoint_sha256": reference_sha,
        "reference_status": reference_payload.get("scientific_status"),
        "trainable_scope": args.optimizer_scope,
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_count": sum(p.numel() for p in trainable),
        "conditioning_encoders_frozen": True,
        "config": asdict(config),
        "replay": {
            "quotas": dict(QUOTAS), "passes": int(args.passes),
            "learning_rate": float(args.learning_rate),
            "p2_weight": float(args.p2_weight),
            "negative_alpha": float(args.negative_alpha),
            "persistent_Adam_across_all_micro_rounds": True,
            "rollback_within_five_round_block": False,
        },
        "GP": {
            "lengthscale": float(lengthscale),
            "rbf_noise": float(config.rbf_noise),
            "total_cap": int(config.gp_buffer_cap),
            "active_gamma_cap_interpretation": "2688 = 4 x 672",
            "replay_rounds": int(args.gp_replay_rounds),
            "fixed_within_each_micro_round": True,
        },
        "lineage_contract": lineage_contract,
        "training_scenario_ids": training_scenario_ids,
        "M50_selection_warning": (
            "this cell requires a shared disjoint M50 screen; selecting among "
            "eight cells requires a later new M100 for an unbiased paper claim"
        ),
    }
    output.mkdir(parents=True)
    _write_json(output / "PREFLIGHT.json", preflight)

    optimizer = torch.optim.Adam(trainable, lr=float(args.learning_rate))
    initial_parameters = HYBRID._parameter_snapshot(trainable)
    history: list[list[dict]] = []
    rounds = []
    transitions = []
    with HYBRID._OrderedSidecarVerifier(
        task, int(args.verifier_workers),
    ) as verifier:
        baseline_raw = HYBRID.raw_only_recheck(
            adapter, task, keys=keys, config=config, microcycle=0,
            verifier=verifier,
        )
        baseline_summary = _raw_summary(baseline_raw, keys)
        _torch_save(output / "raw_development_r00.pt", baseline_raw)
        previous_raw_summary = baseline_summary

        for micro_round in range(1, int(args.micro_rounds) + 1):
            round_root = output / "micro_rounds" / f"round_{micro_round:02d}"
            support, gp_report = gp_support_for_round(
                reference=reference,
                calibration_support=calibration_support,
                previous_support=previous_support,
                history=history,
                cap=int(args.gp_buffer_cap),
                replay_rounds=int(args.gp_replay_rounds),
            )
            previous_support = support
            gather = HYBRID.gather_hybrid(
                adapter, reference, task, keys=keys, config=config,
                lengthscale=float(lengthscale), support_by_gamma=support,
                microcycle=micro_round, verifier=verifier,
            )
            history.append(list(gather["samples"]))
            trace_payload = {
                "status": "SFM_HP100_CUMULATIVE_GATHER_TRACE",
                "version": VERSION, "micro_round": micro_round,
                "events": gather["events"], "outcomes": gather["outcomes"],
                "sample_counts": gather["sample_counts"],
            }
            source_payload = {
                "status": "SFM_HP100_CUMULATIVE_SOURCE_SAMPLES",
                "micro_round": micro_round, "samples": gather["samples"],
                "GP_evidence_if_exact_positive": True,
            }
            trace_path = round_root / "gather_trace.pt"
            source_path = round_root / "source_samples.pt"
            _torch_save(trace_path, trace_payload)
            _torch_save(source_path, source_payload)

            view = training_view(gather)
            training_path = round_root / "training_view.pt"
            _torch_save(training_path, {
                "status": "SFM_HP100_CUMULATIVE_TRAINING_VIEW",
                "micro_round": micro_round, **view,
            })
            update = stratified_signed_update(
                adapter, optimizer, view["groups"],
                p2_weight=float(args.p2_weight),
                negative_alpha=float(args.negative_alpha),
                passes=int(args.passes), seed=int(args.seed),
                micro_round=micro_round,
            )
            candidate_sha = HYBRID._model_state_sha256(adapter)
            checkpoint_path = round_root / "checkpoint.pt"
            _torch_save(checkpoint_path, _checkpoint_payload(
                adapter, optimizer,
                parent_checkpoint_sha256=str(args.expected_checkpoint_sha256),
                reference_checkpoint_sha256=reference_sha,
                optimizer_scope=args.optimizer_scope,
                learning_rate=float(args.learning_rate),
                p2_weight=float(args.p2_weight),
                negative_alpha=float(args.negative_alpha),
                micro_round=micro_round,
            ))
            raw = HYBRID.raw_only_recheck(
                adapter, task, keys=keys, config=config,
                microcycle=micro_round, verifier=verifier,
            )
            raw_summary = _raw_summary(raw, keys)
            raw_path = round_root / "raw_development.pt"
            _torch_save(raw_path, raw)
            new_transitions = _transition_rows(
                previous_raw_summary, raw_summary, micro_round,
            )
            transitions.extend(new_transitions)
            previous_raw_summary = raw_summary
            marker = {
                "status": "SFM_HP100_CUMULATIVE_MICRO_ROUND_COMPLETE",
                "micro_round": micro_round,
                "policy_state_sha256": candidate_sha,
                "optimizer_state_persistent_from_round_1": True,
                "no_scientific_rollback": True,
                "gather": {
                    "outcomes": gather["outcomes"],
                    "sample_counts": gather["sample_counts"],
                    "all_hybrid_success": gather["all_hybrid_success"],
                    "timers_seconds": gather["timers_seconds"],
                },
                "GP": gp_report,
                "training": {
                    key: value for key, value in view.items() if key != "groups"
                },
                "update": update,
                "raw_development": raw_summary,
                "new_failure_to_CLEAR_transitions": new_transitions,
                "relative_parameter_drift_from_block_start": (
                    HYBRID._relative_parameter_drift(trainable, initial_parameters)
                ),
                "artifacts": {
                    "trace": str(trace_path),
                    "trace_sha256": HYBRID._sha256(trace_path),
                    "source_samples": str(source_path),
                    "source_samples_sha256": HYBRID._sha256(source_path),
                    "training_view": str(training_path),
                    "training_view_sha256": HYBRID._sha256(training_path),
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": HYBRID._sha256(checkpoint_path),
                    "raw_development": str(raw_path),
                    "raw_development_sha256": HYBRID._sha256(raw_path),
                },
            }
            marker_path = round_root / "ROUND_COMPLETE.json"
            _write_json(marker_path, marker)
            rounds.append({
                "micro_round": micro_round,
                "P1": view["counts"]["P1"],
                "P2": view["counts"]["P2"],
                "Ncausal": view["counts"]["Ncausal"],
                "Dminus": view["counts"]["Dminus"],
                "N_Adam_steps": update["N_Adam_steps"],
                "repair_exhausted_lineages": view["repair_exhausted_lineages"],
                "raw_CLEAR": raw_summary["clear_count"],
                "raw_prefix": raw_summary["exact_positive_prefix_sum"],
                "raw_goal_progress": raw_summary["aggregate_goal_progress"],
                "new_CLEAR": len(new_transitions),
                "marker": str(marker_path),
            })

    final_checkpoint_path = output / "checkpoint_round_05.pt"
    _torch_save(final_checkpoint_path, _checkpoint_payload(
        adapter, optimizer,
        parent_checkpoint_sha256=str(args.expected_checkpoint_sha256),
        reference_checkpoint_sha256=reference_sha,
        optimizer_scope=args.optimizer_scope,
        learning_rate=float(args.learning_rate),
        p2_weight=float(args.p2_weight),
        negative_alpha=float(args.negative_alpha),
        micro_round=int(args.micro_rounds),
    ))
    marker = {
        "status": STATUS, "version": VERSION,
        "preflight": preflight,
        "cell": {
            "optimizer_scope": args.optimizer_scope,
            "learning_rate": float(args.learning_rate),
            "p2_weight": float(args.p2_weight),
            "negative_alpha": float(args.negative_alpha),
        },
        "rounds": rounds,
        "baseline_raw_development": baseline_summary,
        "final_raw_development": previous_raw_summary,
        "failure_to_CLEAR_transitions": transitions,
        "final_checkpoint": {
            "path": str(final_checkpoint_path),
            "sha256": HYBRID._sha256(final_checkpoint_path),
            "policy_state_sha256": HYBRID._model_state_sha256(adapter),
        },
        "training_scenario_ids": training_scenario_ids,
        "scene_ledger": task.scene_ledger,
        "persistent_Adam": True,
        "micro_rounds_retained_without_rollback": int(args.micro_rounds),
        "relative_parameter_drift_from_block_start": (
            HYBRID._relative_parameter_drift(trainable, initial_parameters)
        ),
        "requires_shared_disjoint_M50": True,
        "promotable": False,
    }
    _write_json(output / "ARM_COMPLETE.json", marker)
    print(json.dumps({
        "status": STATUS, "output": str(output), "cell": marker["cell"],
        "final_raw_CLEAR": previous_raw_summary["clear_count"],
        "transitions": len(transitions),
        "checkpoint": marker["final_checkpoint"],
    }), flush=True)
    return marker


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--source-root", required=True)
    value.add_argument("--expected-source-commit", required=True)
    value.add_argument("--selection-marker", required=True)
    value.add_argument("--expected-selection-marker-sha256", required=True)
    value.add_argument("--checkpoint", required=True)
    value.add_argument("--expected-checkpoint-sha256", required=True)
    value.add_argument("--reference-checkpoint", required=True)
    value.add_argument("--expected-reference-checkpoint-sha256", required=True)
    value.add_argument("--pretrain-dataset-root", required=True)
    value.add_argument("--expected-pretrain-dataset-manifest-sha256", required=True)
    value.add_argument("--output-root", required=True)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--physical-gpu", type=int, required=True)
    value.add_argument(
        "--optimizer-scope", required=True, choices=tuple(EXPECTED_COUNTS),
    )
    value.add_argument("--learning-rate", required=True, type=float)
    value.add_argument("--p2-weight", default=2.0, type=float)
    value.add_argument("--negative-alpha", required=True, type=float)
    value.add_argument("--micro-rounds", default=5, type=int)
    value.add_argument("--replicas-per-gamma", default=8, type=int)
    value.add_argument("--passes", default=10, type=int)
    value.add_argument("--batch-size", default=64, type=int)
    value.add_argument("--gp-buffer-cap", default=2_688, type=int)
    value.add_argument("--gp-replay-rounds", default=2, type=int)
    value.add_argument("--scene-profile", default="double_density_velocity_ood")
    value.add_argument("--scenario-start", default=640_000, type=int)
    value.add_argument("--seed", default=17, type=int)
    value.add_argument("--max-steps", default=180, type=int)
    value.add_argument("--max-repair-batches", default=32, type=int)
    value.add_argument("--ess-target", default=0.1, type=float)
    value.add_argument("--rbf-noise", default=1.0e-2, type=float)
    value.add_argument("--verifier-workers", default=8, type=int)
    return value


def main(argv=None) -> int:
    run(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
