"""SFM2 expansion archive gathering around the frozen acquisition contract.

This runner drives the unmodified ``sfm_hp100_predictive_execution``
K=64/B=32 always-on transaction across declared scene blocks and enriches
every archived D+/D- row with the acquisition provenance the handoff demands:
round, K/B indices, flow-base std, beta, ESS, selected uncertainty, and
checkpoint/source digests.  Sample rows alone do not carry those fields; the
runner joins each sample to its trace event fail-closed and refuses any row it
cannot reproduce from the event log.

The sampling policy may advance across rounds; the acquisition reference
representation and RBF calibration stay pinned to the canonical r0 checkpoint.
Nothing here updates a parameter.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np
import torch

import _paths  # noqa: F401
import grid_policy_sfm_hp100 as GPS
import sfm_hp100_ball_adapter as PORT
import sfm_hp100_ball_launch as BASE
from sfm_hp100_ball_core.expansion import (
    _counter_seed,
    mean_pairwise_lengthscale,
)
import sfm_hp100_exhaustive_hybrid as HYBRID
import sfm_hp100_predictive_execution as PRED
import sfm_scene as SS

from sfm_hp100_expansion_funnel import (
    DECLARED_EVAL_BANKS,
    acquisition_scenario_start,
    assert_declared_banks_static,
    bank_range,
)
from sfm_hp100_expansion_status import Heartbeat


VERSION = "sfm_hp100_expansion_archive_v1"
ARCHIVE_STATUS = "SFM2_EXPANSION_ARCHIVE"
COMPLETE_STATUS = "SFM2_EXPANSION_ARCHIVE_COMPLETE"
SEMANTICS = (
    "D+: selected executed exact full-H10 positives without an immediately "
    "realized collision/OOB; D-: exactly one nonexecuted resolved exact-negative "
    "counterfactual per retry-exhausted lineage, plus each executed y=1 row "
    "whose immediate live transition realized collision/OOB"
)


@dataclass(frozen=True)
class ArchiveConfig:
    gammas: tuple[float, ...] = SS.GAMMAS
    lineages_per_gamma: int = 8
    blocks: int = 1
    max_steps: int = 180
    max_attempts: int = 32
    ess_target: float = 0.1
    seed: int = 41
    round_index: int = 1
    scene_profile: str = "double_density_velocity_ood"

    def validate(self) -> None:
        if self.blocks < 1:
            raise ValueError("declare at least one archive scene block")
        if self.round_index < 1:
            raise ValueError("expansion round indices start at one")
        # Everything else is enforced by the frozen PredictiveConfig contract.
        self.block_config(0).validate()

    def block_seed(self, block: int) -> int:
        return _counter_seed(
            self.seed, "sfm2_expansion_archive_block", self.round_index, int(block),
        )

    def block_config(self, block: int) -> PRED.PredictiveConfig:
        return PRED.PredictiveConfig(
            gammas=self.gammas,
            lineages_per_gamma=self.lineages_per_gamma,
            max_steps=self.max_steps,
            max_attempts=self.max_attempts,
            ess_target=self.ess_target,
            seed=self.block_seed(block),
        )


def dematerialize(row: dict) -> dict:
    """Clone archived tensors outside ``torch.inference_mode``.

    ``gather_predictive`` runs under inference mode, so its stored tensors are
    inference tensors and would raise inside a later ``cfm_loss`` backward.
    Cloning here (in normal mode) clears that flag without changing values.
    """
    if torch.is_inference_mode_enabled():
        raise RuntimeError("dematerialize must run outside inference mode")
    for field in ("context", "candidate", "flow_base"):
        value = row.get(field)
        if isinstance(value, torch.Tensor):
            row[field] = value.detach().clone()
    return row


def enrich_samples(
    samples: list[dict],
    events: list[dict],
    *,
    config: ArchiveConfig,
    block: int,
    provenance: dict,
) -> list[dict]:
    """Join each archived sample to its trace attempt row, fail-closed."""
    event_by_key = {}
    for event in events:
        key = (str(event["lineage"]), int(event["step"]))
        if key in event_by_key:
            raise RuntimeError(f"duplicate trace event for {key}")
        event_by_key[key] = event

    block_config = config.block_config(block)
    rows = []
    for sample in samples:
        key = (str(sample["lineage"]), int(sample["step"]))
        event = event_by_key.get(key)
        if event is None:
            raise RuntimeError(f"archived sample lacks its trace event: {key}")
        attempt = int(sample["attempt"])
        attempts = event["attempts"]
        if attempt >= len(attempts):
            raise RuntimeError(f"archived sample lacks its attempt row: {key}")
        attempt_row = attempts[attempt]
        if int(attempt_row["attempt"]) != attempt:
            raise RuntimeError(f"trace attempt index drifted at {key}")
        expected_std = 1.0 + 0.1 * attempt
        if not math.isclose(
            float(attempt_row["base_std"]), expected_std,
            rel_tol=0.0, abs_tol=1.0e-9,
        ):
            raise RuntimeError(f"retry std schedule drifted at {key}")
        if sample["negative_reason"] == "all_negative_nvp":
            local = attempt_row.get("negative_counterfactual_local")
        else:
            local = attempt_row.get("predictive_local")
        if local is None:
            raise RuntimeError(f"archived sample lacks its selected local at {key}")
        local = int(local)
        if attempt_row["verification"][local] != sample["verification"]:
            raise RuntimeError(
                f"trace verification and sample verification differ at {key}"
            )
        lineage_key = HYBRID.LineageKey(float(sample["gamma"]), int(sample["replica"]))
        row = dematerialize(dict(sample))
        row.update({
            "round": int(config.round_index),
            "block": int(block),
            "block_seed": int(block_config.seed),
            "K_index": int(attempt_row["candidate_ids"][local]),
            "B_local": local,
            "base_std": float(attempt_row["base_std"]),
            "beta": float(attempt_row["beta"]),
            "selected_sigma": float(attempt_row["selected_sigma"][local]),
            "marginal_sigma": float(
                attempt_row["marginal_sigma"][int(attempt_row["candidate_ids"][local])]
            ),
            "conditional_ess": float(attempt_row["conditional_ess"][local]),
            "marginal_ESS_over_K": float(attempt_row["marginal_ESS_over_K"]),
            "positive_first16": int(attempt_row["positive_first16"]),
            "positive_B32": int(attempt_row["positive_B32"]),
            "attempts_used": len(attempts),
            "scene_profile": str(config.scene_profile),
            "scenario_start": int(provenance["scenario_start"]),
            "sampling_seed": HYBRID._sampling_seed(
                block_config.seed, "predictive_always_on", lineage_key,
                int(sample["step"]), microcycle=0, attempt=attempt,
            ),
            "checkpoint_sha256": str(provenance["checkpoint_sha256"]),
            "reference_checkpoint_sha256": str(
                provenance["reference_checkpoint_sha256"]
            ),
            "source_sha256": str(provenance["source_sha256"]),
        })
        rows.append(row)
    return rows


def assert_trace_invariants(samples: list[dict], events: list[dict]) -> None:
    """Prove the frozen retry/execution/archive semantics on a gathered block.

    - the retry schedule is exactly ``base_std = 1.0 + 0.1 * attempt``;
    - a retry never advances state: an NVP lineage ends with its replan state
      untouched and nothing executed;
    - only exact positives execute; every nonexecuted archived row is the one
      resolved ``all_negative_nvp`` counterfactual of an exhausted lineage;
    - a realized-failure row keeps its verifier ``y=1`` while its event
      records the realized collision/OOB terminal.
    """
    event_by_key = {}
    for event in events:
        key = (str(event["lineage"]), int(event["step"]))
        if key in event_by_key:
            raise RuntimeError(f"duplicate trace event for {key}")
        event_by_key[key] = event
        for index, attempt in enumerate(event["attempts"]):
            if int(attempt["attempt"]) != index:
                raise RuntimeError(f"attempt indices are not contiguous at {key}")
            if not math.isclose(
                float(attempt["base_std"]), 1.0 + 0.1 * index,
                rel_tol=0.0, abs_tol=1.0e-9,
            ):
                raise RuntimeError(f"retry std schedule drifted at {key}")
        if event.get("terminal") == "nvp":
            if event.get("executed_role") is not None:
                raise RuntimeError(f"an NVP lineage executed a candidate at {key}")
            if not np.array_equal(
                np.asarray(event["state_before"]), np.asarray(event["state_after"]),
            ):
                raise RuntimeError(f"retry exhaustion advanced state at {key}")

    nvp_events = {
        key for key, event in event_by_key.items()
        if event.get("terminal") == "nvp"
    }
    nvp_samples = [
        row for row in samples
        if row.get("negative_reason") == "all_negative_nvp"
    ]
    nvp_keys = {(str(row["lineage"]), int(row["step"])) for row in nvp_samples}
    if len(nvp_samples) != len(nvp_keys) or nvp_keys != nvp_events:
        raise RuntimeError(
            "exhausted lineages and archived all_negative_nvp rows do not pair 1:1"
        )
    for row in samples:
        key = (str(row["lineage"]), int(row["step"]))
        if key not in event_by_key:
            raise RuntimeError(f"archived sample lacks its trace event: {key}")
        valid = bool(row["verification"]["valid"])
        if row["executed"]:
            if not valid:
                raise RuntimeError(f"an exact negative was executed at {key}")
            expected_role = (
                "positive" if row["role"] == "positive" else row["negative_reason"]
            )
            if event_by_key[key].get("executed_role") != expected_role:
                raise RuntimeError(f"executed role disagrees with the trace at {key}")
        else:
            if row.get("negative_reason") != "all_negative_nvp" or valid:
                raise RuntimeError(
                    f"a nonexecuted row is not the declared NVP counterfactual at {key}"
                )


def assert_bank_disjoint(rows: list[dict], banks=None) -> None:
    """Fail closed if any acquisition scenario id hits a same-profile bank."""
    if banks is None:
        banks = [
            bank for stage in DECLARED_EVAL_BANKS.values() for bank in stage
        ]
    for row in rows:
        for bank in banks:
            if str(bank["scene_profile"]) != str(row.get("scene_profile")):
                continue
            low, high = bank_range(bank)
            if low <= int(row["scenario_id"]) < high:
                raise RuntimeError(
                    f"acquisition scenario {row['scenario_id']} collides with "
                    f"declared evaluation bank {bank['stage']}"
                )


def _assert_blocks_disjoint(scenarios_by_block: dict[int, set[int]]) -> None:
    seen: dict[int, int] = {}
    for block in sorted(scenarios_by_block):
        for scenario in scenarios_by_block[block]:
            previous = seen.get(int(scenario))
            if previous is not None and previous != int(block):
                raise RuntimeError(
                    f"scenario {scenario} appears in blocks {previous} and {block}"
                )
            seen[int(scenario)] = int(block)


def gather_round(
    adapter: PORT.HP100ExpansionPolicy,
    reference: PORT.HP100ExpansionPolicy,
    task: PORT.SFMHP100ExpansionTask,
    *,
    config: ArchiveConfig,
    lengthscale: float,
    support_by_gamma: dict[float, torch.Tensor],
    verifier,
    provenance: dict,
    trace_dir: Path | None = None,
    heartbeat: Heartbeat | None = None,
) -> dict:
    """Gather one round of enriched D+/D- rows across declared scene blocks."""
    config.validate()
    rows: list[dict] = []
    block_summaries = []
    outcomes = {}
    scenarios_by_block: dict[int, set[int]] = {}
    shards = []
    for block in range(config.blocks):
        if heartbeat is not None:
            heartbeat.beat(
                status="running", phase="gather", round=config.round_index,
                block=block, blocks=config.blocks,
                samples=dict(Counter(row["role"] for row in rows)),
            )
        block_config = config.block_config(block)
        keys = PRED._lineage_keys(block_config)
        gathered = PRED.gather_predictive(
            adapter, reference, task, keys=keys, config=block_config,
            lengthscale=float(lengthscale), support_by_gamma=support_by_gamma,
            verifier=verifier,
        )
        assert_trace_invariants(gathered["samples"], gathered["events"])
        rows.extend(enrich_samples(
            gathered["samples"], gathered["events"],
            config=config, block=block, provenance=provenance,
        ))
        block_summaries.append({
            "block": int(block),
            "block_seed": int(block_config.seed),
            "summary": PRED.summarize(gathered, block_config),
            "timers_seconds": gathered["timers_seconds"],
        })
        scenarios_by_block[block] = {
            int(outcome["scenario_id"])
            for outcome in gathered["outcomes"].values()
        }
        for label, outcome in gathered["outcomes"].items():
            outcomes[f"block{block:03d}:{label}"] = outcome
        if trace_dir is not None:
            shard = Path(trace_dir) / f"trace_block_{block:03d}.pt"
            HYBRID._torch_save(shard, {
                "status": PRED.TRACE_STATUS,
                "version": VERSION,
                "round": int(config.round_index),
                "block": int(block),
                "block_seed": int(block_config.seed),
                **gathered,
            })
            shards.append({"path": str(shard), "sha256": PRED.sha256_file(shard)})
        # Events dominate memory (K/B segments per attempt); the shard on disk
        # is now the only holder of this block's event log.
        del gathered
    _assert_blocks_disjoint(scenarios_by_block)
    assert_bank_disjoint(rows)
    return {
        "rows": rows,
        "block_summaries": block_summaries,
        "outcomes": outcomes,
        "trace_shards": shards,
        "sample_counts": _sample_counts(rows),
    }


def _sample_counts(rows: list[dict]) -> dict:
    return {
        "positive": sum(row["role"] == "positive" for row in rows),
        "all_negative_nvp": sum(
            row["negative_reason"] == "all_negative_nvp" for row in rows
        ),
        "realized_collision": sum(
            row["negative_reason"] == "realized_collision" for row in rows
        ),
        "realized_oob": sum(
            row["negative_reason"] == "realized_oob" for row in rows
        ),
    }


def load_archive(path: str | Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("status") != ARCHIVE_STATUS:
        raise ValueError(f"not an SFM2 expansion archive: {path}")
    return payload


def _load_frozen(checkpoint: str, expected_sha: str, device: str):
    actual = PRED.sha256_file(checkpoint)
    if actual != str(expected_sha).lower():
        raise RuntimeError(f"checkpoint SHA256 mismatch: {actual} != {expected_sha}")
    policy, payload = GPS.load_sfm_hp100_policy(checkpoint, device=device)
    adapter = PORT.HP100ExpansionPolicy(policy).eval()
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    return adapter, payload, actual


def run(args) -> dict:
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing archive output: {output}")
    assert_declared_banks_static()
    gpu_contract = BASE._gpu_contract(args.device, int(args.physical_gpu))
    adapter, payload, checkpoint_sha = _load_frozen(
        args.checkpoint, args.expected_checkpoint_sha256, args.device,
    )
    reference, reference_payload, reference_sha = _load_frozen(
        args.reference_checkpoint, args.expected_reference_checkpoint_sha256,
        args.device,
    )
    config = ArchiveConfig(
        gammas=HYBRID._parse_gammas(args.gammas),
        lineages_per_gamma=int(args.lineages_per_gamma),
        blocks=int(args.blocks),
        max_steps=int(args.max_steps),
        max_attempts=int(args.max_attempts),
        ess_target=float(args.ess_target),
        seed=int(args.seed),
        round_index=int(args.round_index),
        scene_profile=str(args.scene_profile),
    )
    config.validate()
    scenario_start = (
        acquisition_scenario_start(config.round_index)
        if args.scenario_start is None else int(args.scenario_start)
    )
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
        scene_profile=config.scene_profile,
        scenario_start=scenario_start,
    ).attach_context_encoder(adapter.policy)

    output.mkdir(parents=True)
    provenance = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_scientific_status": payload.get("scientific_status"),
        "reference_checkpoint": str(Path(args.reference_checkpoint).resolve()),
        "reference_checkpoint_sha256": reference_sha,
        "reference_scientific_status": reference_payload.get("scientific_status"),
        "source_sha256": PRED.sha256_file(__file__),
        "acquisition_source_sha256": PRED.sha256_file(PRED.__file__),
        "scenario_start": int(scenario_start),
        "gpu": gpu_contract,
        "calibration": {**calibration, "lengthscale": float(lengthscale)},
    }
    heartbeat = Heartbeat(
        args.status_json, interval_seconds=float(args.heartbeat_seconds),
    )
    before = PRED._model_state_sha256(adapter)
    with HYBRID._OrderedSidecarVerifier(task, int(args.verifier_workers)) as verifier:
        gathered = gather_round(
            adapter, reference, task, config=config,
            lengthscale=float(lengthscale), support_by_gamma=support,
            verifier=verifier, provenance=provenance, trace_dir=output,
            heartbeat=heartbeat,
        )
    if PRED._model_state_sha256(adapter) != before:
        raise RuntimeError("archive gathering changed the sampling policy")
    archive_path = output / f"expansion_archive_r{config.round_index}.pt"
    HYBRID._torch_save(archive_path, {
        "status": ARCHIVE_STATUS,
        "version": VERSION,
        "semantics": SEMANTICS,
        "config": asdict(config),
        "provenance": provenance,
        "rows": gathered["rows"],
    })
    marker = {
        "status": COMPLETE_STATUS,
        "version": VERSION,
        "config": asdict(config),
        "provenance": provenance,
        "sample_counts": gathered["sample_counts"],
        "outcome_counts": dict(sorted(Counter(
            outcome["status"] for outcome in gathered["outcomes"].values()
        ).items())),
        "block_summaries": gathered["block_summaries"],
        "trace_shards": gathered["trace_shards"],
        "archive": {
            "path": str(archive_path),
            "sha256": PRED.sha256_file(archive_path),
        },
        "policy_unchanged": True,
        "policy_state_sha256": before,
    }
    HYBRID._write_json(output / "ARCHIVE_COMPLETE.json", marker)
    heartbeat.beat(
        status="complete", phase="gather", round=config.round_index,
        samples=gathered["sample_counts"],
    )
    return marker


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--checkpoint", required=True)
    value.add_argument("--expected-checkpoint-sha256", required=True)
    value.add_argument("--reference-checkpoint", required=True)
    value.add_argument("--expected-reference-checkpoint-sha256", required=True)
    value.add_argument("--pretrain-dataset-root", required=True)
    value.add_argument("--expected-pretrain-dataset-manifest-sha256", required=True)
    value.add_argument("--output", required=True)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--physical-gpu", type=int, default=3)
    value.add_argument("--scene-profile", default="double_density_velocity_ood")
    value.add_argument("--scenario-start", type=int, default=None)
    value.add_argument("--round-index", type=int, default=1)
    value.add_argument("--blocks", type=int, default=1)
    value.add_argument("--gammas", default="0.1,0.2,0.3,0.4,0.5,0.7,1.0")
    value.add_argument("--lineages-per-gamma", type=int, default=8)
    value.add_argument("--max-steps", type=int, default=180)
    value.add_argument("--max-attempts", type=int, default=32)
    value.add_argument("--ess-target", type=float, default=0.1)
    value.add_argument("--seed", type=int, default=41)
    value.add_argument("--verifier-workers", type=int, default=32)
    value.add_argument("--status-json", default=None)
    value.add_argument("--heartbeat-seconds", type=float, default=30.0)
    return value


def main(argv=None) -> int:
    marker = run(parser().parse_args(argv))
    print(json.dumps(
        {"status": marker["status"], "sample_counts": marker["sample_counts"]},
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
