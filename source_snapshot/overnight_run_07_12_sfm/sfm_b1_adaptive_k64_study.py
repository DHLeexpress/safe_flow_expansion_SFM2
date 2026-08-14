"""Five-round qualification of adaptive exact-verifier acquisition.

This is deliberately one arm, not a sweep.  It keeps the selected B1 update
(``alpha=.001``, 16 chunks x 16 replay epochs) and changes only acquisition:
64 learned-flow proposals are sampled sequentially without replacement under
the RBF uncertainty Gibbs tilt, then the exact verifier is queried in batches
of four until an admissible max-margin action is found or all 64 proposals are
exhausted.

Checkpoint selection uses the fixed canonical-temperature-one M10 bank that
is already evaluated by the trainer.  After selection, r0 and the selected
checkpoint are evaluated on the same untouched M100 bank at temperature one.
No temperature search, action filter, fallback, expert proposal, or trajectory
asset is introduced here.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import sfm_b1_alpha_steps_sweep as AS
import sfm_b1_curve_eval as CE
import sfm_b1_sweep as SW
import sfm_protocol as SP


ROUNDS = 5
ARM_NAME = "margin_alpha0p001_inner016_adaptiveK64"
SCENE_PROFILE = "double_density_velocity_ood"


def _write_json(path, payload):
    path = os.fspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    os.replace(temporary, path)


def _run_logged(command, *, path):
    started = time.perf_counter()
    with open(path, "w") as stream:
        completed = subprocess.run(
            command, cwd=SW.ROOT, stdout=stream, stderr=subprocess.STDOUT,
            text=True, check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}; inspect {path}"
        )
    return time.perf_counter() - started


def sanity_rows(arm_dir, *, expected_rounds=ROUNDS):
    """Load the trainer's fixed-bank, canonical-temperature M10 records."""
    path = os.path.join(arm_dir, "method_manifest.json")
    with open(path) as stream:
        manifest = json.load(stream)
    if (manifest.get("status") != "ARM_COMPLETE"
            or int(manifest.get("rounds", -1)) != int(expected_rounds)):
        raise RuntimeError("adaptive arm manifest is incomplete")
    recipe = manifest.get("recipe", {})
    expected_recipe = dict(
        acquisition_mode="adaptive_k64", K=SP.ADAPTIVE_PROPOSAL_K,
        B=SP.ADAPTIVE_QUERY_BATCH, max_queries=SP.ADAPTIVE_MAX_QUERIES,
        selector="margin", alpha=0.001, optimizer_steps=16,
        inner_epochs=16, lr=1e-4, sanity_M=10,
    )
    for key, expected in expected_recipe.items():
        if recipe.get(key) != expected:
            raise RuntimeError(f"adaptive recipe mismatch for {key}: {recipe.get(key)!r}")
    payloads = [manifest.get("baseline_sanity")] + [
        record.get("sanity") for record in manifest.get("history", [])
    ]
    if len(payloads) != int(expected_rounds) + 1:
        raise RuntimeError("adaptive arm lacks a complete r0:r5 sanity curve")
    canonical = CE.temperature_vector(1.0)
    rows = []
    for expected_round, payload in enumerate(payloads):
        if payload is None or int(payload.get("round", -1)) != expected_round:
            raise RuntimeError("adaptive sanity rounds are incomplete or out of order")
        if payload.get("temperature_by_gamma") != canonical:
            raise RuntimeError("adaptive sanity evaluation is not canonical temperature one")
        rows.append(dict(
            round=expected_round, summary=payload["summary"],
            temperature_by_gamma=canonical,
            checkpoint=os.path.abspath(
                os.path.join(arm_dir, f"round_{expected_round:02d}.pt")
            ),
        ))
    return rows, manifest


def select_round(rows):
    """Select once on fixed raw M10; earlier round wins an exact tie."""
    if [int(row["round"]) for row in rows] != list(range(ROUNDS + 1)):
        raise ValueError("selection requires exactly the declared r0:r5 records")
    return min(
        rows,
        key=lambda row: (
            *CE.temperature_selection_key(
                row["summary"], row["temperature_by_gamma"]
            ),
            int(row["round"]),
        ),
    )


def validate_paired_confirmations(r0, selected, *, expected_selected_round):
    for name, payload, expected_round in (
            ("r0", r0, 0), ("selected", selected, int(expected_selected_round))):
        if payload.get("status") != "SFM_B1_SINGLE_CONFIRMATION_COMPLETE":
            raise RuntimeError(f"{name} confirmation is incomplete")
        if int(payload.get("round", -1)) != expected_round:
            raise RuntimeError(f"{name} confirmation round mismatch")
        if payload.get("temperature_by_gamma") != CE.temperature_vector(1.0):
            raise RuntimeError(f"{name} confirmation is not canonical temperature one")
        bank = payload.get("bank", {})
        if (int(bank.get("ep0", -1)), int(bank.get("M", -1))) != (
                int(SP.ADAPTIVE_CONFIRM_EP0), 100):
            raise RuntimeError(f"{name} does not use the declared M100 bank")
    if r0["bank"] != selected["bank"]:
        raise RuntimeError("r0 and selected confirmations are not paired on one exact bank")
    return True


def _metric_delta(selected, baseline):
    metrics = ("SR", "CR", "timeout", "V_safe")
    return {
        metric: float(selected[metric]) - float(baseline[metric])
        for metric in metrics
    }


def run(args):
    output = AS._validate_output_root(args.outdir)
    if os.path.exists(output):
        raise FileExistsError("qualification output root must not already exist")
    if int(args.rounds) != ROUNDS:
        raise ValueError("this qualification is locked to exactly five rounds")
    if args.scene_profile != SCENE_PROFILE:
        raise ValueError(f"this qualification is locked to {SCENE_PROFILE}")
    if int(args.workers) < 1:
        raise ValueError("workers must be positive")
    observed_checkpoint_sha = SW.sha256_file(args.checkpoint)
    if observed_checkpoint_sha != args.expected_checkpoint_sha256:
        raise RuntimeError("checkpoint SHA-256 mismatch")
    preflight = AS._load_preflight(
        args.preflight, args.checkpoint, args.expected_preflight_sha256,
    )
    observed_gpu_sha = SW.sha256_file(args.gpu_provenance)
    if observed_gpu_sha != args.expected_gpu_provenance_sha256:
        raise RuntimeError("GPU provenance SHA-256 mismatch")
    selected_rbf = preflight["sweep_selected"]
    if int(selected_rbf["cap"]) != SP.GP_CAP:
        raise RuntimeError("adaptive study requires the reviewed cap-512 RBF row")
    source = SW.git_frozen_source()

    os.makedirs(output)
    gpu_record = os.path.join(output, "gpu_provenance.txt")
    shutil.copyfile(args.gpu_provenance, gpu_record)
    if SW.sha256_file(gpu_record) != observed_gpu_sha:
        raise RuntimeError("copied GPU provenance changed")
    logs = os.path.join(output, "logs")
    os.makedirs(logs)
    arm_dir = os.path.join(output, "arm")
    plan = dict(
        status="ADAPTIVE_K64_STUDY_PLANNED", source=source,
        checkpoint=dict(path=os.path.abspath(args.checkpoint),
                        sha256=observed_checkpoint_sha),
        preflight=dict(path=os.path.abspath(args.preflight),
                       sha256=args.expected_preflight_sha256,
                       ell=float(selected_rbf["ell"]), cap=int(selected_rbf["cap"]),
                       lambda_=float(preflight["lambda_"])),
        gpu_provenance=dict(path=os.path.abspath(gpu_record),
                            sha256=observed_gpu_sha),
        arm=dict(
            name=ARM_NAME, rounds=ROUNDS, scene_profile=SCENE_PROFILE,
            acquisition=("K=64 learned proposals; sequential RBF-uncertainty Gibbs "
                         "sampling without replacement; exact-verifier queries in "
                         "batches of four; stop at first admissible batch; NVP only "
                         "after all 64"),
            storage="only actually queried resolved candidates enter D/D+",
            execution="max one-step H_P margin among all queried admissible candidates",
            alpha=0.001, optimizer_chunks=16, inner_epochs=16,
            learning_rate=1e-4, W=2, sanity="raw temperature-one M10 every round",
        ),
        selection=("one checkpoint selected by the predeclared canonical-temperature-one "
                   "M10 key; earlier round breaks exact ties"),
        confirmation=("paired r0/selected raw M100 on the exact same untouched bank at "
                      "temperature one; no tuning and no trajectory artifacts"),
    )
    _write_json(os.path.join(output, "STUDY_PLAN.json"), plan)

    trainer_command = [
        sys.executable, os.path.join(SW.HERE, "sfm_b1_expand.py"),
        "--checkpoint", os.path.abspath(args.checkpoint),
        "--outdir", arm_dir,
        "--custom-name", ARM_NAME,
        "--selector", "margin", "--alpha", "0.001",
        "--optimizer-steps", "16", "--inner-epochs", "16",
        "--lr", "1e-4", "--sanity-M", "10", "--adaptive-k64",
        "--ell", str(float(selected_rbf["ell"])),
        "--cap", str(int(selected_rbf["cap"])),
        "--rounds", str(ROUNDS), "--device", args.device,
        "--verifier-workers", str(int(args.workers)),
        "--scene-profile", SCENE_PROFILE,
    ]
    train_seconds = _run_logged(
        trainer_command, path=os.path.join(logs, "train.log"),
    )
    if SW.sha256_file(args.checkpoint) != observed_checkpoint_sha:
        raise RuntimeError("source checkpoint changed during training")
    rows, manifest = sanity_rows(arm_dir)
    selected = select_round(rows)

    confirmation_results = {}
    confirmation_seconds = {}
    for label, row in (("r0", rows[0]), ("selected", selected)):
        directory = os.path.join(output, "confirmation", label)
        command = [
            sys.executable, os.path.join(SW.HERE, "sfm_b1_curve_eval.py"), "confirm",
            "--checkpoint", row["checkpoint"], "--round", str(row["round"]),
            "--temperature", "1.0", "--single-vector-only",
            "--scene-profile", SCENE_PROFILE, "--outdir", directory,
            "--device", args.device, "--workers", str(int(args.workers)),
            "--ep0", str(SP.ADAPTIVE_CONFIRM_EP0), "--M", "100",
        ]
        confirmation_seconds[label] = _run_logged(
            command, path=os.path.join(logs, f"confirm_{label}.log"),
        )
        with open(os.path.join(directory, "COMPLETE.json")) as stream:
            confirmation_results[label] = json.load(stream)
    validate_paired_confirmations(
        confirmation_results["r0"], confirmation_results["selected"],
        expected_selected_round=selected["round"],
    )

    r0_pooled = confirmation_results["r0"]["summary"]["pooled"]
    selected_pooled = confirmation_results["selected"]["summary"]["pooled"]
    complete = dict(
        status="ADAPTIVE_K64_STUDY_COMPLETE", source=source,
        checkpoint=plan["checkpoint"], preflight=plan["preflight"],
        gpu_provenance=plan["gpu_provenance"], arm=plan["arm"],
        selection=dict(
            bank="fixed raw canonical-temperature-one M10",
            selected_round=int(selected["round"]), selected_checkpoint=selected["checkpoint"],
            selected_checkpoint_sha256=SW.sha256_file(selected["checkpoint"]),
            all_rounds=[dict(round=row["round"], summary=row["summary"]) for row in rows],
        ),
        paired_confirmation=dict(
            bank=confirmation_results["r0"]["bank"],
            r0=r0_pooled, selected=selected_pooled,
            selected_minus_r0=_metric_delta(selected_pooled, r0_pooled),
            no_temperature_tuning=True, no_trajectory_artifacts=True,
        ),
        acquisition_by_round=[dict(
            round=record["round"],
            adaptive_acquisition=record["gather"].get("adaptive_acquisition"),
        ) for record in manifest["history"]],
        runtime_seconds=dict(training=train_seconds, confirmation=confirmation_seconds,
                             total=train_seconds + sum(confirmation_seconds.values())),
        artifacts=dict(
            plan=os.path.abspath(os.path.join(output, "STUDY_PLAN.json")),
            arm_manifest=os.path.abspath(os.path.join(arm_dir, "method_manifest.json")),
            confirmation_r0=os.path.abspath(
                os.path.join(output, "confirmation", "r0", "COMPLETE.json")
            ),
            confirmation_selected=os.path.abspath(
                os.path.join(output, "confirmation", "selected", "COMPLETE.json")
            ),
        ),
    )
    _write_json(os.path.join(output, "ADAPTIVE_K64_STUDY_COMPLETE.json"), complete)
    return complete


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--expected-preflight-sha256", required=True)
    parser.add_argument("--gpu-provenance", required=True)
    parser.add_argument("--expected-gpu-provenance-sha256", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--scene-profile", default=SCENE_PROFILE, choices=(SCENE_PROFILE,))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=32)
    return parser


def main(argv=None):
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
