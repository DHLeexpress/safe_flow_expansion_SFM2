"""Authenticated three-round HP100 early executed-window expansion.

This protocol is intentionally narrower than :mod:`sfm_hp100_ball_launch`:
every gamma receives exactly 16 paired OOD lineages, each lineage stops at its
first NVP or at the declared 30-step acquisition cutoff, and no terminal-
success retry/selection is performed.  D+ contains only the sampled H10
proposal whose first action was actually executed.  At NVP, D- receives one
resolved-invalid counterfactual proposal; that row is never described as
executed and never enters the GP.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import torch

import _paths  # noqa: F401
import grid_policy_sfm_hp100 as GPS
import sfm_hp100_ball_adapter as PORT
import sfm_hp100_ball_launch as BASE
from sfm_hp100_ball_core import ExpansionConfig, assert_vendored_core, run_safe_expansion
import sfm_scene as SS


VERSION = "sfm_hp100_early_executed_v2"
K = 64
B = 16
T_GATHER = 30
PARALLEL_EPISODES = 16
GP_CAP = 2_688
GP_REFERENCE_ROUNDS = 2
REPLAY_ROUNDS = 2
FLOW_BASE_STD = 1.4
EXECUTION_STEP_MARGIN_WEIGHT = 50_000.0
LEARNING_RATE = 1.0e-6
MICROBATCH_REPEATS = 10
NEGATIVE_ALPHA = 1.0e-3
ESS_TARGET = 0.1
INITIAL_BETA = 5.0e-4


def _load_checkpoint(
    path: str, expected_sha256: str, device: str, optimizer_scope: str,
):
    actual = BASE.sha256_file(path)
    if actual != str(expected_sha256).lower():
        raise RuntimeError(f"checkpoint SHA256 mismatch: {actual} != {expected_sha256}")
    policy, payload = GPS.load_sfm_hp100_policy(path, device=device)
    if payload.get("scientific_status") != "canonical_ID_promoted":
        raise ValueError("early expansion requires the canonical ID-promoted HP100 model")
    if payload.get("pretrained_from_scratch") is not True:
        raise ValueError("early expansion requires the from-scratch HP100 checkpoint")
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
        raise ValueError(f"unsupported HP100 optimizer scope {optimizer_scope!r}")
    names = sorted(name for name, value in adapter.named_parameters() if value.requires_grad)
    count = sum(value.numel() for value in trainable)
    if count != expected_count:
        raise RuntimeError(
            f"{optimizer_scope} parameter count drifted: {count} != {expected_count}"
        )
    return adapter, payload, actual, names, count


def protocol_config(args, *, lengthscale: float) -> ExpansionConfig:
    config = ExpansionConfig(
        rounds=int(args.rounds), gammas=tuple(map(float, SS.GAMMAS)),
        parallel_episodes=PARALLEL_EPISODES,
        verifier_workers=int(args.verifier_workers),
        max_retry_batches=1, max_steps=T_GATHER,
        max_steps_status="EARLY_CUTOFF", K=K, B=B,
        batch_size=int(args.batch_size), inner_steps=None,
        microbatch_repeats=int(args.microbatch_repeats),
        learning_rate=float(args.learning_rate),
        freeze_visual_encoder=True, head_only_update=False,
        optimizer_scope=str(args.optimizer_scope), replay_rounds=REPLAY_ROUNDS,
        gp_buffer_cap=GP_CAP, gp_noise=1.0e-2,
        rbf_lengthscale=float(lengthscale), beta=INITIAL_BETA,
        adaptive_beta=True, ess_target=ESS_TARGET,
        negative_alpha=float(args.negative_alpha), replay_top_fraction=1.0,
        replay_selector="uniform", flow_base_std=float(args.flow_base_std),
        candidate_perturb_std=0.0,
        candidate_perturb_scope="coherent_horizon",
        execution_rule="min_cost",
        execution_step_margin_weight=float(args.execution_step_margin_weight),
        archive_rule="executed_plus_nvp_negative",
        successful_trajectory_selector="lowest_episode_id",
        successful_trajectories_per_gamma=0,
        replay_acceptance="execution_eligible",
        paired_noised_representation=True, seed=int(args.seed),
        acquisition_feature="learned_phi", coverage_replay="none",
        replay_augmentation="none",
        gp_reference_mode="sliding_executed_positive_per_gamma_frozen_phi",
        gp_reference_rounds=GP_REFERENCE_ROUNDS,
        gp_sliding_row_selector="trajectory_uniform",
    )
    config.validate()
    return config


def build_contract(
    args, adapter, payload, checkpoint_sha, trainable_names, trainable_count,
    features, calibration_provenance,
):
    core = assert_vendored_core()
    lengthscale = float(__import__(
        "sfm_hp100_ball_core.expansion", fromlist=["mean_pairwise_lengthscale"]
    ).mean_pairwise_lengthscale(features))
    config = protocol_config(args, lengthscale=lengthscale)
    return {
        "status": "SFM_HP100_EARLY_EXPANSION_PREFLIGHT_PASSED",
        "version": VERSION,
        "source": {
            "launcher": str(Path(__file__).resolve()),
            "launcher_sha256": BASE.sha256_file(__file__),
            "adapter": str(Path(PORT.__file__).resolve()),
            "adapter_sha256": BASE.sha256_file(PORT.__file__),
        },
        "core": core,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "architecture": payload["config"],
        "optimizer_scope": {
            "name": str(args.optimizer_scope),
            "trainable_parameter_names": trainable_names,
            "trainable_parameter_count": trainable_count,
            "learning_rate": float(args.learning_rate),
            "microbatch_repeats": int(args.microbatch_repeats),
            "dropout_mode": "eval/off",
        },
        "scene_profile": SS.scene_profile(args.scene_profile),
        "scenario_bank": {
            "start": int(args.scenario_start),
            "lineages_per_gamma_round": PARALLEL_EPISODES,
            "retry_batches": 1,
            "paired_across_gamma": True,
        },
        "calibration": {**calibration_provenance, "lengthscale": lengthscale},
        "expansion_config": asdict(config),
        "semantics": {
            "lineage": (
                "all 16 lineages/gamma are retained; each stops independently at "
                "NVP, physical terminal, or EARLY_CUTOFF=30; no success quota or retry"
            ),
            "D_plus": (
                "one exact-positive sampled H10 proposal per non-NVP context; "
                "only its first action is executed"
            ),
            "D_minus": (
                "one resolved-invalid terminal counterfactual proposal per NVP, "
                f"ranked by J_native-{float(args.execution_step_margin_weight):g}"
                "*m_step; never executed"
            ),
            "GP": (
                "D+ only, frozen pretrained paired-noised phi, prior two rounds, "
                "seven independent 384-row trajectory/time-uniform supports"
            ),
            "replay": (
                "current plus previous round D+; each eligible row is exposed "
                f"exactly {int(args.microbatch_repeats)} times; "
                "D- contributes only through signed negative_alpha"
            ),
            "evaluation": (
                "not acquisition: exported checkpoints are evaluated full T=180 by "
                "raw temperature-one sfm_hp100_eval.py on a disjoint CRN bank"
            ),
        },
        "gpu": BASE._gpu_contract(args.device, args.physical_gpu),
        "output_root": str(Path(args.output).resolve()),
    }, config


class RoundPrinter:
    def __call__(self, row):
        print(json.dumps({
            "stage": "round_complete", "round": row["round"],
            "D_round": row["queries"], "D_plus_round": row["positives"],
            "D_minus_round": row["negatives"], "NVP": row["NVP"],
            "success": row["success"], "collision": row["collision"],
            "oob": row["oob"], "timeout": row["timeout"],
            "early_cutoff": row.get("early_cutoff", 0),
            "gp_buffer": row["gp_buffer"], "beta": row["beta"],
            "ESS_over_K": row["ESS_over_K"],
            "uncertainty_uplift": row["uncertainty_uplift"],
            "adam_steps": row["steps"], "round_total_s": row["round_total_s"],
        }, allow_nan=False), flush=True)


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
    value.add_argument("--scene-profile", default=PORT.DEFAULT_SCENE_PROFILE,
                       choices=SS.SCIENTIFIC_EVAL_PROFILES)
    value.add_argument("--scenario-start", type=int, default=400_000)
    value.add_argument("--rounds", type=int, default=3)
    value.add_argument("--verifier-workers", type=int, default=32)
    value.add_argument("--batch-size", type=int, default=64)
    value.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    value.add_argument("--microbatch-repeats", type=int, default=MICROBATCH_REPEATS)
    value.add_argument("--flow-base-std", type=float, default=FLOW_BASE_STD)
    value.add_argument(
        "--execution-step-margin-weight", type=float,
        default=EXECUTION_STEP_MARGIN_WEIGHT,
    )
    value.add_argument(
        "--optimizer-scope", choices=("head_only", "last_block_and_head"),
        default="last_block_and_head",
    )
    value.add_argument("--negative-alpha", type=float, default=NEGATIVE_ALPHA)
    value.add_argument("--seed", type=int, default=2)
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output root: {output}")
    adapter, payload, checkpoint_sha, names, count = _load_checkpoint(
        args.checkpoint, args.expected_checkpoint_sha256, args.device,
        args.optimizer_scope,
    )
    task = PORT.SFMHP100ExpansionTask(
        scene_profile=args.scene_profile, scenario_start=args.scenario_start,
    ).attach_context_encoder(adapter.policy)
    features, calibration = BASE.calibration_features(
        adapter, dataset_root=args.pretrain_dataset_root,
        expected_manifest_sha256=args.expected_pretrain_dataset_manifest_sha256,
        count=50, seed=args.seed, base_std=float(args.flow_base_std),
        paired_noised_representation=True,
    )
    contract, config = build_contract(
        args, adapter, payload, checkpoint_sha, names, count, features, calibration,
    )
    preflight = output.with_suffix(output.suffix + ".preflight.json")
    preflight.parent.mkdir(parents=True, exist_ok=True)
    preflight.write_text(json.dumps(contract, indent=2, allow_nan=False) + "\n")
    if args.mode == "preflight":
        print(json.dumps({
            "status": contract["status"], "preflight": str(preflight),
            "lengthscale": contract["calibration"]["lengthscale"],
            "trainable_parameters": count,
        }))
        return 0

    score_partial = output.with_suffix(output.suffix + ".execution_scores.jsonl.partial")
    acquisition_partial = output.with_suffix(output.suffix + ".acquisition.jsonl.partial")
    scores = BASE.ScoreLog(
        score_partial, acquisition_partial,
        float(args.execution_step_margin_weight),
    )
    try:
        manifest = run_safe_expansion(
            adapter, task, output, config=config, calibration_features=None,
            event_callback=scores, round_callback=RoundPrinter(),
        )
    finally:
        scores.close()
    score_path = output / "execution_scores.jsonl"
    acquisition_path = output / "acquisition_contexts.jsonl"
    score_partial.replace(score_path)
    acquisition_partial.replace(acquisition_path)
    acquisition_audit = BASE.acquisition_audit(acquisition_path, manifest)
    acquisition_audit_path = output / "acquisition_per_gamma.json"
    acquisition_audit_path.write_text(
        json.dumps(acquisition_audit, indent=2, allow_nan=False) + "\n"
    )
    evaluation_checkpoints = BASE.export_evaluation_checkpoints(
        output, architecture=payload["config"], pretrained_sha256=checkpoint_sha,
    )
    delivery = {
        "status": "SFM_HP100_EARLY_EXPANSION_COMPLETE",
        "version": VERSION, "preflight": contract,
        "expansion_manifest_status": manifest["status"],
        "execution_scores": str(score_path),
        "acquisition_contexts": str(acquisition_path),
        "acquisition_per_gamma": str(acquisition_audit_path),
        "evaluation_checkpoints": evaluation_checkpoints,
        "scene_ledger": task.scene_ledger,
    }
    marker = output / "EARLY_EXPANSION_COMPLETE.json"
    marker.write_text(json.dumps(delivery, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": delivery["status"], "output": str(output)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
