"""Authenticated Helios launcher for the HP100/3-D-ball protocol port.

``preflight`` performs no gathering or optimization.  ``run`` uses distinct
OOD pedestrian scenarios for every parallel replica/retry/round, paired across
gamma by the vendored core's reset seed.  A fixed scenario is accepted only by
preflight for audit-video planning and can never enter replay.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

import _paths  # noqa: F401
import grid_policy_sfm_hp100 as GPS
import sfm_hp100_ball_adapter as PORT
from sfm_hp100_ball_core import ExpansionConfig, assert_vendored_core, run_safe_expansion
from sfm_hp100_ball_core.expansion import normalized_ess
import sfm_hp100_features as HPF
import sfm_metrics2 as VERIFY
import sfm_scene as SS
import stage3_hp100_pretrain as PRETRAIN


VERSION = "sfm_hp100_ball_launch_v1"


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _gpu_contract(device: str, physical_gpu: int) -> dict:
    if not str(device).startswith("cuda"):
        if int(physical_gpu) != -1:
            raise ValueError("CPU preflight requires --physical-gpu -1")
        return dict(device=str(device), physical_gpu=None, visible=None)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(int(physical_gpu)):
        raise RuntimeError(
            "set CUDA_VISIBLE_DEVICES to exactly --physical-gpu before launch; "
            f"got {visible!r} versus {physical_gpu}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda is unavailable")
    return dict(
        device=str(device), physical_gpu=int(physical_gpu), visible=visible,
        logical_device_name=torch.cuda.get_device_name(0),
    )


def _checkpoint(checkpoint: str, expected_sha256: str, device: str):
    actual = sha256_file(checkpoint)
    if actual != str(expected_sha256).lower():
        raise RuntimeError(f"checkpoint SHA256 mismatch: {actual} != {expected_sha256}")
    policy, payload = GPS.load_sfm_hp100_policy(checkpoint, device=device)
    GPS.configure_head_only_expansion(policy)
    adapter = PORT.HP100ExpansionPolicy(policy)
    PORT.assert_head_only(adapter)
    return adapter, payload, actual


@torch.no_grad()
def calibration_features(
    adapter, *, dataset_root: str, expected_manifest_sha256: str,
    count: int, seed: int, base_std: float,
    paired_noised_representation: bool,
):
    """Sample authenticated pretrained phi features for RBF calibration.

    The 50 contexts are gamma-balanced (8 for one rotating gamma and 7 for
    each other gamma) and trajectory-balanced: each selected context comes
    from a distinct successful training lineage within its gamma.
    """
    if int(count) != 50:
        raise ValueError("the declared RBF preflight uses exactly 50 embeddings")
    train, _, dataset_meta = PRETRAIN.load_split(
        dataset_root, SS.GAMMAS, val_frac=0.1, seed=PRETRAIN.DEFAULT_SEED,
        expected_manifest_sha256=str(expected_manifest_sha256),
        require_canonical=True,
    )
    gamma_rows = torch.as_tensor(train.gamma_rows, dtype=torch.int64)
    episodes = torch.as_tensor(train.episodes, dtype=torch.int64)
    source_rows = torch.as_tensor(train.source_rows, dtype=torch.int64)
    if not (len(train) == len(gamma_rows) == len(episodes) == len(source_rows)):
        raise RuntimeError("HP100 training split row maps differ in length")

    generator = torch.Generator().manual_seed(int(seed))
    per_gamma = [int(count) // len(SS.GAMMAS)] * len(SS.GAMMAS)
    per_gamma[int(seed) % len(SS.GAMMAS)] += int(count) % len(SS.GAMMAS)
    selected_indices: list[int] = []
    selection: list[dict] = []
    for gamma_index, required in enumerate(per_gamma):
        local = torch.nonzero(gamma_rows == gamma_index, as_tuple=False).flatten()
        lineage_ids = torch.unique(episodes[local], sorted=True)
        if len(lineage_ids) < required:
            raise RuntimeError(
                f"gamma {SS.GAMMAS[gamma_index]} has only {len(lineage_ids)} "
                f"eligible training lineages; need {required}"
            )
        chosen_lineages = lineage_ids[
            torch.randperm(len(lineage_ids), generator=generator)[:required]
        ]
        for lineage in chosen_lineages.tolist():
            candidates = local[episodes[local] == int(lineage)]
            offset = int(torch.randint(
                len(candidates), (1,), generator=generator,
            ).item())
            dataset_index = int(candidates[offset])
            selected_indices.append(dataset_index)
            selection.append(dict(
                calibration_index=len(selection),
                gamma=float(SS.GAMMAS[gamma_index]),
                gamma_index=int(gamma_index), episode=int(lineage),
                train_dataset_index=dataset_index,
                source_row=int(source_rows[dataset_index]),
            ))
    if len(selected_indices) != int(count):
        raise RuntimeError("RBF calibration did not select exactly 50 contexts")

    device = next(adapter.parameters()).device
    rows = []
    was_training = adapter.training
    adapter.eval()
    try:
        for index, dataset_index in enumerate(selected_indices):
            hp100, low5, hist, _, episode, gamma_index = train[dataset_index]
            expected = selection[index]
            if int(episode) != expected["episode"] or int(gamma_index) != expected["gamma_index"]:
                raise RuntimeError("HP100 training split item disagrees with its row map")
            if tuple(hp100.shape) != GPS.GRID_SHAPE:
                raise RuntimeError(f"calibration HP100 history has shape {tuple(hp100.shape)}")
            if tuple(low5.shape) != (5,) or tuple(hist.shape) != (GPS.K_HIST, 2):
                raise RuntimeError("calibration low5/control-history shape differs")
            context = adapter.policy.ctx_from(
                hp100.to(device), low5.to(device), hist.to(device),
            )[0]
            plan_generator = torch.Generator(device=device).manual_seed(
                int(seed) + 1009 * index
            )
            if paired_noised_representation:
                plan, base = adapter.sample_with_base(
                    context, 1, plan_generator, base_std=float(base_std),
                )
                feature = adapter.embed(context, plan, base=base)
            else:
                plan = adapter.sample(
                    context, 1, plan_generator, base_std=float(base_std),
                )
                feature = adapter.embed(context, plan)
            rows.append(feature[0].detach().cpu())
    finally:
        adapter.train(was_training)
    features = torch.stack(rows)
    feature_sha256 = hashlib.sha256(
        features.contiguous().numpy().tobytes()
    ).hexdigest()
    provenance = dict(
        dataset_root=str(Path(dataset_root).resolve()),
        manifest=str(Path(dataset_meta["manifest"]).resolve()),
        manifest_sha256=str(dataset_meta["manifest_sha256"]),
        expected_manifest_sha256=str(expected_manifest_sha256),
        split_seed=int(dataset_meta["split_seed"]), selection_seed=int(seed),
        split_semantics=dataset_meta["split_semantics"],
        count=int(count), per_gamma={
            str(float(gamma)): int(per_gamma[index])
            for index, gamma in enumerate(SS.GAMMAS)
        },
        trajectory_balance="distinct successful training lineage within each gamma",
        representation=(
            "paired noised phi at s=0.9 using the authoritative sampled x0"
            if paired_noised_representation
            else "endpoint phi at s=0.9; no paired x0"
        ),
        paired_noised_representation=bool(paired_noised_representation),
        base_std=float(base_std), feature_sha256=feature_sha256,
        selection=selection,
    )
    return features, provenance


def protocol_config(args, *, lengthscale: float) -> ExpansionConfig:
    gp_cap = int(args.gp_buffer_cap)
    if gp_cap % len(SS.GAMMAS):
        raise ValueError("gp-buffer-cap must be divisible by seven SFM gammas")
    config = ExpansionConfig(
        rounds=int(args.rounds), gammas=tuple(map(float, SS.GAMMAS)),
        parallel_episodes=int(args.parallel_episodes),
        verifier_workers=int(args.verifier_workers),
        max_retry_batches=int(args.max_retry_batches), max_steps=180,
        K=PORT.K, B=PORT.B, batch_size=int(args.batch_size),
        inner_steps=args.optimizer_steps_per_round,
        microbatch_repeats=int(args.microbatch_repeats),
        learning_rate=float(args.learning_rate), freeze_visual_encoder=True,
        head_only_update=True, replay_rounds=3, gp_buffer_cap=gp_cap,
        gp_noise=1.0e-2, rbf_lengthscale=float(lengthscale),
        beta=float(args.initial_beta), adaptive_beta=True,
        ess_target=float(args.ess_target),
        negative_alpha=float(args.negative_alpha),
        replay_top_fraction=1.0, replay_selector="uniform",
        flow_base_std=float(args.flow_base_std), candidate_perturb_std=0.0,
        candidate_perturb_scope="coherent_horizon", execution_rule="min_cost",
        execution_step_margin_weight=(
            0.0 if args.execution_step_margin_weight is None
            else float(args.execution_step_margin_weight)
        ),
        archive_rule="preterminal_resolved_queries",
        successful_trajectory_selector="random_success",
        successful_trajectories_per_gamma=1,
        replay_acceptance="safety_valid",
        # Every selected-B query is one original full H=10 plan, so its sampled
        # flow base is authoritative. No cross-replan x0 stitching occurs.
        paired_noised_representation=bool(args.paired_noised_representation),
        seed=int(args.seed),
        acquisition_feature="learned_phi", coverage_replay="none",
        replay_augmentation="none",
        # Head-only expansion fixes phi. State that explicitly instead of
        # pretending that current-model re-embedding changes the features.
        gp_reference_mode="sliding_positive_per_gamma_frozen_phi",
        gp_sliding_row_selector="trajectory_uniform",
    )
    config.validate()
    return config


def build_contract(
    args, adapter, checkpoint_payload, checkpoint_sha, features,
    calibration_provenance,
):
    core = assert_vendored_core()
    lengthscale = float(
        __import__(
            "sfm_hp100_ball_core.expansion", fromlist=["mean_pairwise_lengthscale"]
        ).mean_pairwise_lengthscale(features)
    )
    config = protocol_config(args, lengthscale=lengthscale)
    fixed = args.audit_fixed_scenario_id
    return dict(
        status="SFM_HP100_BALL_PORT_PREFLIGHT_PASSED",
        version=VERSION,
        source=dict(
            launcher=str(Path(__file__).resolve()), launcher_sha256=sha256_file(__file__),
            adapter=str(Path(PORT.__file__).resolve()), adapter_sha256=sha256_file(PORT.__file__),
        ),
        core=core, checkpoint=str(Path(args.checkpoint).resolve()),
        checkpoint_sha256=checkpoint_sha, architecture=checkpoint_payload["config"],
        head_only_trainable=PORT.assert_head_only(adapter),
        scene_profile=SS.scene_profile(args.scene_profile),
        production_scenario_bank=dict(
            start=int(args.scenario_start),
            mapping=(
                "scenario_start + counter_hash(seed,round,retry,replica) mod 1e9; "
                "paired across gamma, distinct across replica/retry/round"
            ),
            declared_nonoverlap=dict(
                hp100_pretraining="approximately 0..529",
                matched_id_eval="150000..150049",
                severe_ood_eval="250000..250049",
            ),
        ),
        diagnostic_fixed_scenario=(
            None if fixed is None else dict(
                scenario_id=int(fixed), role="audit video only",
                enters_replay=False,
            )
        ),
        verifier=VERIFY.verifier_manifest(), observation=HPF.contract(),
        calibration=dict(
            **calibration_provenance, lengthscale=lengthscale,
        ),
        expansion_config=asdict(config),
        semantics=dict(
            query="K=16 flow plans; RBF acquisition selects B=4 without replacement",
            execution_eligibility=(
                "resolved exact candidate-specific 2-D full-H y=1 only; no nominal-Hp "
                "gate and no progress gate"
            ),
            selector="J_native - lambda * nominal_one_step_Hp_margin",
            failure=(
                "if B contains no exact positive, only that lineage terminates "
                "NVP; its resolved terminal B queries remain truthfully labeled"
            ),
            archive=(
                "every resolved selected-B query from every parallel lineage through "
                "its NVP/success/collision/timeout decision; D+ is exact-positive and "
                "D- is exact-negative, with no terminal-success selection"
            ),
            representation=(
                "paired noised phi from each query's original full-plan x0; no "
                "first-action latent stitching"
            ),
            gp=(
                "strictly prior-round bounded sliding exact-positive support; equal "
                "gamma capacity, then near-equal lineage and departure-to-tail "
                "time-stage coverage"
            ),
            optimizer=(
                "optimizer_steps_per_round caps distinct microbatches; microbatch_repeats "
                "re-exposes the same batch and is not a one-use setting; negative_alpha "
                "sets the negative-gradient norm relative to the positive gradient"
            ),
        ),
        gpu=_gpu_contract(args.device, args.physical_gpu),
        output_root=str(Path(args.output).resolve()),
    ), config


class ScoreLog:
    """Stream flat lambda rows plus one acquisition diagnostic per context."""

    def __init__(self, path: Path, acquisition_path: Path, weight: float):
        self.path = path
        self.acquisition_path = acquisition_path
        self.weight = float(weight)
        self.stream = path.open("x")
        self.acquisition_stream = acquisition_path.open("x")

    def __call__(self, event):
        for local, result in enumerate(event["verification"]):
            margin = result.get("step_margin")
            cost = float(result["execution_cost"])
            score = None if margin is None else cost - self.weight * float(margin)
            json_cost = cost if np.isfinite(cost) else None
            json_margin = (
                None if margin is None or not np.isfinite(float(margin))
                else float(margin)
            )
            json_score = score if score is not None and np.isfinite(score) else None
            progress = result.get("progress")
            json_progress = (
                None if progress is None or not np.isfinite(float(progress))
                else float(progress)
            )
            row = dict(
                round=int(event["round"]), gamma=float(event["gamma"]),
                context_id=int(event["context_id"]),
                episode=int(event["episode"]),
                retry_batch=int(event["retry_batch"]),
                replica=int(event["replica"]), step=int(event["step"]),
                candidate_id=int(event["selected"][local]),
                exact_positive=bool(result["valid"] and not result["error"]),
                execution_eligible=bool(
                    result["valid"] and not result["error"]
                    and result["progress_eligible"]
                    and result.get("target_eligible", True)
                ),
                native_cost=json_cost, step_margin=json_margin,
                H10_goal_progress=json_progress, combined_score=json_score,
                chosen=bool(event["chosen_local"] == local),
                archived_terminal_negative=bool(
                    event.get("archived_negative_local") == local
                ),
                nvp_reason=event["nvp_reason"],
            )
            self.stream.write(json.dumps(row, allow_nan=False) + "\n")
        acquisition = dict(
            round=int(event["round"]), gamma=float(event["gamma"]),
            context_id=int(event["context_id"]), episode=int(event["episode"]),
            retry_batch=int(event["retry_batch"]),
            replica=int(event["replica"]), step=int(event["step"]),
            sigma_K=[float(value) for value in event["sigma_K"]],
            selected_ids=[int(value) for value in event["selected"]],
            selected_sigma=[float(value) for value in event["selected_sigma"]],
        )
        self.acquisition_stream.write(
            json.dumps(acquisition, allow_nan=False) + "\n"
        )
        self.stream.flush()
        self.acquisition_stream.flush()

    def close(self):
        self.stream.close()
        self.acquisition_stream.close()


def acquisition_audit(path: Path, manifest: dict) -> dict:
    """Compute realized marginal ESS/uplift separately for every gamma/round."""
    beta_by_round = {
        int(row["round"]): float(row["beta"]) for row in manifest["rounds"]
    }
    grouped: dict[tuple[int, float], list[tuple[float, float]]] = {}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            sigma = torch.as_tensor(row["sigma_K"], dtype=torch.float64)
            round_index = int(row["round"])
            beta = beta_by_round[round_index]
            selected = torch.as_tensor(row["selected_sigma"], dtype=torch.float64)
            grouped.setdefault((round_index, float(row["gamma"])), []).append((
                normalized_ess(sigma, beta),
                float(selected.mean() - sigma.mean()),
            ))
    cells = {}
    for (round_index, gamma), values in sorted(grouped.items()):
        key = f"r{round_index:03d}_g{gamma:.9g}"
        cells[key] = dict(
            round=round_index, gamma=gamma, contexts=len(values),
            beta=beta_by_round[round_index],
            marginal_ESS_over_K=float(np.mean([value[0] for value in values])),
            uncertainty_uplift=float(np.mean([value[1] for value in values])),
        )
    return dict(
        status="SFM_HP100_PER_GAMMA_ACQUISITION_AUDIT_COMPLETE",
        definition=(
            "mean over context-level normalized marginal ESS and "
            "selected-minus-pool sigma; no gamma pooling"
        ),
        cells=cells,
    )


def export_evaluation_checkpoints(
    output: Path, *, architecture: dict, pretrained_sha256: str,
) -> dict:
    """Repackage generic core states for the strict canonical HP100 evaluator.

    The vendored task-agnostic core stores its ExpansionConfig under ``config``.
    ``sfm_hp100_eval.py`` correctly expects the model architecture there, so an
    authenticated adapter export is required rather than teaching the evaluator
    to guess an architecture from tensor shapes.
    """
    destination = output / "evaluation_checkpoints"
    destination.mkdir()
    rows = []
    for source in sorted(output.glob("checkpoint_*.pt")):
        payload = torch.load(source, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "model" not in payload:
            raise RuntimeError(f"generic expansion checkpoint is malformed: {source}")
        round_index = int(payload["round"])
        target = destination / f"checkpoint_{round_index:03d}.pt"
        torch.save(dict(
            state_dict=payload["model"], config=architecture,
            scientific_status="HP100_BALL_PORT_EVALUATION_READY",
            expansion_round=round_index,
            expansion_pretrained=bool(payload.get("pretrained", False)),
            source_generic_checkpoint=str(source),
            source_generic_checkpoint_sha256=sha256_file(source),
            pretrained_checkpoint_sha256=str(pretrained_sha256),
        ), target)
        rows.append(dict(
            round=round_index, path=str(target), sha256=sha256_file(target),
            source=str(source), source_sha256=sha256_file(source),
        ))
    if not rows:
        raise RuntimeError("expansion produced no generic checkpoints to export")
    marker = destination / "EVALUATION_CHECKPOINTS.json"
    marker.write_text(json.dumps(dict(
        status="SFM_HP100_EVALUATION_CHECKPOINT_EXPORT_COMPLETE",
        architecture=architecture, pretrained_checkpoint_sha256=pretrained_sha256,
        checkpoints=rows,
        evaluator=(
            "sfm_hp100_eval.py; raw temperature=1, no acquisition, verifier "
            "selection, guidance, or fallback"
        ),
    ), indent=2, allow_nan=False) + "\n")
    return dict(marker=str(marker), marker_sha256=sha256_file(marker), rows=rows)


def parser():
    value = argparse.ArgumentParser()
    value.add_argument("--mode", choices=("preflight", "run"), default="preflight")
    value.add_argument("--checkpoint", required=True)
    value.add_argument("--expected-checkpoint-sha256", required=True)
    value.add_argument("--pretrain-dataset-root", required=True)
    value.add_argument(
        "--expected-pretrain-dataset-manifest-sha256", required=True,
    )
    value.add_argument("--output", required=True)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--physical-gpu", type=int, required=True)
    value.add_argument("--scene-profile", default=PORT.DEFAULT_SCENE_PROFILE,
                       choices=SS.SCIENTIFIC_EVAL_PROFILES)
    value.add_argument("--scenario-start", type=int, default=PORT.DEFAULT_SCENARIO_START)
    value.add_argument("--audit-fixed-scenario-id", type=int)
    value.add_argument("--rounds", type=int, default=10)
    value.add_argument("--parallel-episodes", type=int, default=16)
    value.add_argument("--verifier-workers", type=int, default=8)
    value.add_argument("--max-retry-batches", type=int, default=32)
    value.add_argument(
        "--successful-trajectories-per-gamma", type=int, default=1,
        help=(
            "compatibility knob; preterminal mode requires 1 because it archives "
            "all parallel lineages without terminal-success selection"
        ),
    )
    value.add_argument("--batch-size", type=int, default=64)
    value.add_argument("--optimizer-steps-per-round", type=int)
    value.add_argument("--microbatch-repeats", type=int, default=10)
    value.add_argument("--learning-rate", type=float, default=1.0e-6)
    value.add_argument("--flow-base-std", type=float, default=1.0)
    # Matches the ball runner's fallback when the promoted pretrain manifest
    # has no stored beta.  Round 1 has an empty GP (uniform sigma); the value
    # first matters once successful support exists, before adaptive calibration
    # supplies the following round's beta.
    value.add_argument("--initial-beta", type=float, default=5.0e-4)
    value.add_argument("--ess-target", type=float, default=0.1)
    # The 3-D run used 768 / 4 = 192 rows per gamma. SFM has seven
    # gammas, so its capacity-preserving total is 7 * 192 = 1344.
    value.add_argument("--gp-buffer-cap", type=int, default=1344)
    value.add_argument("--negative-alpha", type=float, default=0.0)
    value.add_argument("--execution-step-margin-weight", type=float)
    value.add_argument("--paired-noised-representation", action="store_true")
    value.add_argument("--seed", type=int, default=2)
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    if args.mode == "run" and not args.paired_noised_representation:
        raise ValueError(
            "canonical preterminal run requires --paired-noised-representation; "
            "each selected-B query has an authoritative full-plan x0"
        )
    if int(args.successful_trajectories_per_gamma) != 1:
        raise ValueError(
            "preterminal archive has no terminal-success selector; "
            "--successful-trajectories-per-gamma must remain 1"
        )
    if args.mode == "run" and args.execution_step_margin_weight is None:
        raise ValueError(
            "run mode requires an explicit --execution-step-margin-weight "
            "selected from the task-specific no-update lambda preflight"
        )
    if args.audit_fixed_scenario_id is not None and args.mode != "preflight":
        raise ValueError("fixed-scenario mode is diagnostic-only and may not run/replay")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output root: {output}")
    adapter, checkpoint_payload, checkpoint_sha = _checkpoint(
        args.checkpoint, args.expected_checkpoint_sha256, args.device,
    )
    task = PORT.SFMHP100ExpansionTask(
        scene_profile=args.scene_profile, scenario_start=args.scenario_start,
        fixed_scenario_id=args.audit_fixed_scenario_id,
    ).attach_context_encoder(adapter.policy)
    features, calibration_provenance = calibration_features(
        adapter, dataset_root=args.pretrain_dataset_root,
        expected_manifest_sha256=args.expected_pretrain_dataset_manifest_sha256,
        count=50, seed=args.seed, base_std=args.flow_base_std,
        paired_noised_representation=bool(args.paired_noised_representation),
    )
    contract, config = build_contract(
        args, adapter, checkpoint_payload, checkpoint_sha, features,
        calibration_provenance,
    )
    preflight_path = output.with_suffix(output.suffix + ".preflight.json")
    preflight_path.parent.mkdir(parents=True, exist_ok=True)
    preflight_path.write_text(json.dumps(contract, indent=2, allow_nan=False) + "\n")
    if args.mode == "preflight":
        print(json.dumps(dict(status=contract["status"], preflight=str(preflight_path),
                              lengthscale=contract["calibration"]["lengthscale"])))
        return 0
    score_path = output.with_suffix(output.suffix + ".execution_scores.jsonl.partial")
    acquisition_path = output.with_suffix(
        output.suffix + ".acquisition.jsonl.partial"
    )
    scores = ScoreLog(
        score_path, acquisition_path, args.execution_step_margin_weight,
    )
    try:
        manifest = run_safe_expansion(
            adapter, task, output, config=config, calibration_features=None,
            event_callback=scores,
        )
    finally:
        scores.close()
    final_scores = output / "execution_scores.jsonl"
    final_acquisition = output / "acquisition_contexts.jsonl"
    score_path.replace(final_scores)
    acquisition_path.replace(final_acquisition)
    per_gamma_acquisition = acquisition_audit(final_acquisition, manifest)
    acquisition_audit_path = output / "acquisition_per_gamma.json"
    acquisition_audit_path.write_text(
        json.dumps(per_gamma_acquisition, indent=2, allow_nan=False) + "\n"
    )
    evaluation_checkpoints = export_evaluation_checkpoints(
        output, architecture=checkpoint_payload["config"],
        pretrained_sha256=checkpoint_sha,
    )
    delivery = dict(
        status="SFM_HP100_BALL_PORT_RUN_COMPLETE", preflight=contract,
        expansion_manifest_status=manifest["status"],
        execution_scores=str(final_scores),
        acquisition_contexts=str(final_acquisition),
        acquisition_per_gamma=str(acquisition_audit_path),
        evaluation_checkpoints=evaluation_checkpoints,
        adapter_semantics=dict(
            replay_label=(
                "full H=10 exact positive only; terminal suffixes shorter than H "
                "are excluded rather than zero-padded into D+"
            ),
        ),
        scene_ledger=task.scene_ledger,
    )
    (output / "SFM_HP100_BALL_PORT_COMPLETE.json").write_text(
        json.dumps(delivery, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(dict(status=delivery["status"], output=str(output))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
