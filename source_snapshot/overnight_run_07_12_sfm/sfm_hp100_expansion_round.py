"""Round orchestrator for SFM2 positive-minus-alpha-negative expansion.

Each round gathers an enriched acquisition archive with the *current* sampling
policy (round 1 samples from r0), applies one declared update, and saves a
strict raw-evaluable checkpoint.  The acquisition reference representation,
RBF calibration support, and every condition encoder stay pinned to the
canonical r0 checkpoint for every round: there is no round-to-round GP support
accumulation, no buffer rule, and no historical replay role.

The first-round archive may be shared between arms (gather once, train per
alpha) through ``--reuse-archive``; the archive provenance is then verified
against the declared r0 digest fail-closed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import _paths  # noqa: F401
import grid_policy_sfm_hp100 as GPS
import sfm_hp100_ball_adapter as PORT
import sfm_hp100_ball_launch as BASE
from sfm_hp100_ball_core.expansion import mean_pairwise_lengthscale
import sfm_hp100_exhaustive_hybrid as HYBRID
import sfm_hp100_predictive_execution as PRED

import sfm_hp100_expansion_archive as ARCH
from sfm_hp100_expansion_funnel import acquisition_scenario_start
from sfm_hp100_expansion_status import Heartbeat
import sfm_hp100_expansion_update as UPD


VERSION = "sfm_hp100_expansion_round_v1"
ROUND_STATUS = "SFM2_EXPANSION_ROUND_COMPLETE"
RECIPE_STATUS = "SFM2_EXPANSION_RECIPE_COMPLETE"


def split_roles(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    positives = [row for row in rows if row["role"] == "positive"]
    negatives = [row for row in rows if row["role"] == "negative"]
    if len(positives) + len(negatives) != len(rows):
        raise ValueError("archive rows carry an undeclared role")
    return positives, negatives


def _load_reused_archive(path: Path, *, expected_checkpoint_sha: str) -> dict:
    payload = ARCH.load_archive(path)
    provenance = payload["provenance"]
    if provenance["checkpoint_sha256"] != str(expected_checkpoint_sha).lower():
        raise RuntimeError(
            "reused archive was not gathered from the declared sampling checkpoint"
        )
    if int(payload["config"]["round_index"]) != 1:
        raise RuntimeError("only a round-1 archive may be reused across arms")
    return payload


def run_recipe(args) -> dict:
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing recipe output: {output}")
    rounds = int(args.rounds)
    if rounds < 1:
        raise ValueError("declare at least one expansion round")
    gpu_contract = BASE._gpu_contract(args.device, int(args.physical_gpu))
    r0_sha = PRED.sha256_file(args.checkpoint)
    if r0_sha != str(args.expected_checkpoint_sha256).lower():
        raise RuntimeError("r0 checkpoint SHA256 mismatch")
    policy, payload = GPS.load_sfm_hp100_policy(args.checkpoint, device=args.device)
    adapter = PORT.HP100ExpansionPolicy(policy).eval()
    reference_policy, _ = GPS.load_sfm_hp100_policy(
        args.checkpoint, device=args.device,
    )
    reference = PORT.HP100ExpansionPolicy(reference_policy).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    encoder_r0 = UPD.encoder_state_sha256(reference)

    update_config = UPD.UpdateConfig(
        alpha=float(args.alpha),
        learning_rate=float(args.learning_rate),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        grad_clip_norm=float(args.grad_clip_norm),
        max_relative_parameter_drift=float(args.max_relative_parameter_drift),
        train_mode=str(args.train_mode),
        seed=int(args.update_seed),
    )
    update_config.validate()

    features, calibration = BASE.calibration_features(
        reference, dataset_root=args.pretrain_dataset_root,
        expected_manifest_sha256=args.expected_pretrain_dataset_manifest_sha256,
        count=50, seed=int(args.seed), base_std=1.0,
        paired_noised_representation=True,
    )
    lengthscale = mean_pairwise_lengthscale(features)
    gammas = HYBRID._parse_gammas(args.gammas)
    support = HYBRID._calibration_support_by_gamma(
        features, calibration, gammas,
        device=next(reference.parameters()).device,
    )

    output.mkdir(parents=True)
    heartbeat = Heartbeat(
        args.status_json, interval_seconds=float(args.heartbeat_seconds),
    )
    recipe_provenance = {
        "recipe_id": str(args.recipe_id),
        "r0_checkpoint": str(Path(args.checkpoint).resolve()),
        "r0_checkpoint_sha256": r0_sha,
        "r0_scientific_status": payload.get("scientific_status"),
        "gpu": gpu_contract,
        "calibration": {**calibration, "lengthscale": float(lengthscale)},
        "update_config": asdict(update_config),
        "rounds": rounds,
        "gp_support_rule": (
            "frozen 50-row balanced pretrained preflight for every round; "
            "no round-to-round accumulation"
        ),
        "source_sha256": PRED.sha256_file(__file__),
    }
    HYBRID._write_json(output / "RECIPE_DECLARED.json", recipe_provenance)

    parent_sha = r0_sha
    round_markers = []
    checkpoints = []
    for round_index in range(1, rounds + 1):
        round_dir = output / f"round_{round_index:02d}"
        current_sha = PRED._model_state_sha256(adapter)
        archive_config = ARCH.ArchiveConfig(
            gammas=gammas,
            lineages_per_gamma=int(args.lineages_per_gamma),
            blocks=int(args.blocks),
            max_steps=int(args.max_steps),
            max_attempts=int(args.max_attempts),
            ess_target=float(args.ess_target),
            seed=int(args.seed),
            round_index=round_index,
            scene_profile=str(args.scene_profile),
        )
        archive_config.validate()
        if round_index == 1 and args.reuse_archive is not None:
            archive_payload = _load_reused_archive(
                Path(args.reuse_archive), expected_checkpoint_sha=r0_sha,
            )
            rows = archive_payload["rows"]
            archive_record = {
                "reused": True,
                "path": str(Path(args.reuse_archive).resolve()),
                "sha256": PRED.sha256_file(args.reuse_archive),
                "sample_counts": ARCH._sample_counts(rows),
            }
        else:
            scenario_start = acquisition_scenario_start(round_index)
            task = PORT.SFMHP100ExpansionTask(
                scene_profile=archive_config.scene_profile,
                scenario_start=scenario_start,
            ).attach_context_encoder(adapter.policy)
            for parameter in adapter.parameters():
                parameter.requires_grad_(False)
            round_dir.mkdir(parents=True)
            provenance = {
                "checkpoint_sha256": (
                    r0_sha if round_index == 1 else parent_sha
                ),
                "policy_state_sha256": current_sha,
                "reference_checkpoint_sha256": r0_sha,
                "source_sha256": PRED.sha256_file(ARCH.__file__),
                "scenario_start": int(scenario_start),
            }
            with HYBRID._OrderedSidecarVerifier(
                task, int(args.verifier_workers),
            ) as verifier:
                gathered = ARCH.gather_round(
                    adapter, reference, task, config=archive_config,
                    lengthscale=float(lengthscale), support_by_gamma=support,
                    verifier=verifier, provenance=provenance,
                    trace_dir=round_dir, heartbeat=heartbeat,
                )
            rows = gathered["rows"]
            ARCH.assert_bank_disjoint(rows)
            archive_path = round_dir / f"expansion_archive_r{round_index}.pt"
            HYBRID._torch_save(archive_path, {
                "status": ARCH.ARCHIVE_STATUS,
                "version": ARCH.VERSION,
                "semantics": ARCH.SEMANTICS,
                "config": asdict(archive_config),
                "provenance": provenance,
                "rows": rows,
            })
            archive_record = {
                "reused": False,
                "path": str(archive_path),
                "sha256": PRED.sha256_file(archive_path),
                "sample_counts": gathered["sample_counts"],
                "block_summaries": gathered["block_summaries"],
                "trace_shards": gathered["trace_shards"],
            }

        positives, negatives = split_roles(rows)
        heartbeat.beat(
            status="running", phase="update", round=round_index,
            samples=ARCH._sample_counts(rows),
        )
        update_metrics = UPD.expansion_update(
            adapter, positives, negatives, update_config,
            round_index=round_index,
        )
        if update_metrics["encoder_state_sha256_after"] != encoder_r0:
            raise RuntimeError("condition encoders drifted from the r0 digest")
        checkpoint_path = output / f"checkpoint_r{round_index}.pt"
        HYBRID._torch_save(checkpoint_path, UPD.checkpoint_payload(
            adapter,
            parent_checkpoint_sha256=parent_sha,
            pretrained_checkpoint_sha256=r0_sha,
            round_index=round_index,
            alpha=update_config.alpha,
        ))
        checkpoint_sha = PRED.sha256_file(checkpoint_path)
        marker = {
            "status": ROUND_STATUS,
            "version": VERSION,
            "recipe_id": str(args.recipe_id),
            "round": round_index,
            "archive": archive_record,
            "update": update_metrics,
            "checkpoint": {
                "path": str(checkpoint_path), "sha256": checkpoint_sha,
                "policy_state_sha256": PRED._model_state_sha256(adapter),
            },
            "parent_checkpoint_sha256": parent_sha,
        }
        HYBRID._write_json(output / f"ROUND_{round_index}_COMPLETE.json", marker)
        round_markers.append(marker)
        checkpoints.append({"round": round_index, "sha256": checkpoint_sha})
        parent_sha = checkpoint_sha
        if not update_metrics["accepted"]:
            break

    final = {
        "status": RECIPE_STATUS,
        "version": VERSION,
        "provenance": recipe_provenance,
        "rounds_completed": len(round_markers),
        "rounds_declared": rounds,
        "checkpoints": checkpoints,
        "accepted": all(marker["update"]["accepted"] for marker in round_markers),
    }
    HYBRID._write_json(output / "RECIPE_COMPLETE.json", final)
    heartbeat.beat(
        status="complete", phase="recipe", rounds=len(round_markers),
    )
    return final


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--checkpoint", required=True)
    value.add_argument("--expected-checkpoint-sha256", required=True)
    value.add_argument("--pretrain-dataset-root", required=True)
    value.add_argument("--expected-pretrain-dataset-manifest-sha256", required=True)
    value.add_argument("--output", required=True)
    value.add_argument("--recipe-id", required=True)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--physical-gpu", type=int, default=3)
    value.add_argument("--verifier-workers", type=int, default=32)
    value.add_argument("--rounds", type=int, default=1)
    value.add_argument("--alpha", type=float, default=0.0)
    value.add_argument("--learning-rate", type=float, default=1.0e-6)
    value.add_argument("--batch-size", type=int, default=64)
    value.add_argument("--epochs", type=int, default=1)
    value.add_argument("--grad-clip-norm", type=float, default=1.0)
    value.add_argument(
        "--max-relative-parameter-drift", type=float, default=0.25,
    )
    value.add_argument("--train-mode", default="eval", choices=("eval", "train"))
    value.add_argument("--update-seed", type=int, default=2)
    value.add_argument("--scene-profile", default="double_density_velocity_ood")
    value.add_argument("--gammas", default="0.1,0.2,0.3,0.4,0.5,0.7,1.0")
    value.add_argument("--lineages-per-gamma", type=int, default=8)
    value.add_argument("--blocks", type=int, default=1)
    value.add_argument("--max-steps", type=int, default=180)
    value.add_argument("--max-attempts", type=int, default=32)
    value.add_argument("--ess-target", type=float, default=0.1)
    value.add_argument("--seed", type=int, default=41)
    value.add_argument("--reuse-archive", default=None)
    value.add_argument("--status-json", default=None)
    value.add_argument("--heartbeat-seconds", type=float, default=30.0)
    return value


def main(argv=None) -> int:
    final = run_recipe(parser().parse_args(argv))
    print(json.dumps(
        {"status": final["status"], "rounds": final["rounds_completed"],
         "accepted": final["accepted"]},
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
