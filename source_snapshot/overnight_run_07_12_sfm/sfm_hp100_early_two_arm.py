"""Frozen two-GPU launcher for the selected HP100 early-expansion lambda.

The lambda is read only from a completed, training/evaluation-blind
``sfm_hp100_early_lambda_report`` artifact.  Both optimizer-scope arms first
run the launcher's preflight and their shared scientific contracts must match
before either training process starts.  There are no retries.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


VERSION = "sfm_hp100_early_two_arm_v1"
SELECTION_STATUS = "SFM_HP100_EARLY_LAMBDA_CALIBRATION_COMPLETE"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ("git", "-C", str(root), *args), text=True,
    ).strip()


def source_gate(root: Path, expected_commit: str) -> dict:
    head = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain")
    if head != expected_commit:
        raise RuntimeError(f"HEAD mismatch: {head} != {expected_commit}")
    if status:
        raise RuntimeError("frozen two-arm run requires a clean worktree")
    return {"root": str(root), "HEAD": head, "clean": True}


def load_selected_lambda(
    path: Path, *, expected_sha256: str, checkpoint_sha256: str,
) -> tuple[dict, float]:
    actual = sha256_file(path)
    if actual != expected_sha256.lower():
        raise RuntimeError(
            f"lambda-report SHA256 mismatch: {actual} != {expected_sha256}"
        )
    report = json.loads(path.read_text())
    if report.get("status") != SELECTION_STATUS or report.get("selected") is None:
        raise ValueError("lambda report did not pass its fail-closed selection gate")
    if report.get("selection_blind_to_training_and_evaluation") is not True:
        raise ValueError("lambda selection was not declared blind to training/evaluation")
    if report.get("checkpoint_sha256") != checkpoint_sha256.lower():
        raise ValueError("lambda report and launch checkpoint hashes differ")
    selected = report["selected"]
    if selected.get("passes") is not True or float(selected["lambda"]) <= 0.0:
        raise ValueError("selected lambda is not a positive passing calibration cell")
    return report, float(selected["lambda"])


@dataclass(frozen=True)
class Arm:
    name: str
    optimizer_scope: str
    physical_gpu: int


def declared_arms(gpus: tuple[int, ...]) -> tuple[Arm, Arm]:
    if len(gpus) != 2 or len(set(gpus)) != 2:
        raise ValueError("exactly two distinct physical GPUs are required")
    return (
        Arm("head_only", "head_only", int(gpus[0])),
        Arm("last_block_and_head", "last_block_and_head", int(gpus[1])),
    )


def expansion_command(args, arm: Arm, *, selected_lambda: float, mode: str):
    launcher = (
        Path(args.source_root).resolve()
        / "source_snapshot/overnight_run_07_12_sfm/sfm_hp100_early_expand.py"
    )
    output = Path(args.output_root).resolve() / "arms" / arm.name / "run"
    return [
        sys.executable, "-u", str(launcher), "--mode", mode,
        "--checkpoint", str(Path(args.checkpoint).resolve()),
        "--expected-checkpoint-sha256", args.expected_checkpoint_sha256,
        "--pretrain-dataset-root",
        str(Path(args.pretrain_dataset_root).resolve()),
        "--expected-pretrain-dataset-manifest-sha256",
        args.expected_pretrain_dataset_manifest_sha256,
        "--output", str(output), "--device", "cuda:0",
        "--physical-gpu", str(arm.physical_gpu),
        "--scene-profile", args.scene_profile,
        "--scenario-start", str(args.scenario_start),
        "--rounds", str(args.rounds),
        "--verifier-workers", str(args.verifier_workers),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--microbatch-repeats", str(args.microbatch_repeats),
        "--flow-base-std", str(args.flow_base_std),
        "--execution-step-margin-weight", str(selected_lambda),
        "--optimizer-scope", arm.optimizer_scope,
        "--negative-alpha", str(args.negative_alpha),
        "--seed", str(args.seed),
    ]


def shared_preflight_signature(contract: dict) -> dict:
    """Return every scientific field that must agree across scope arms."""
    config = dict(contract["expansion_config"])
    config.pop("optimizer_scope")
    return {
        "version": contract["version"],
        "checkpoint_sha256": contract["checkpoint_sha256"],
        "architecture": contract["architecture"],
        "scene_profile": contract["scene_profile"],
        "scenario_bank": contract["scenario_bank"],
        "calibration": contract["calibration"],
        "expansion_config_except_optimizer_scope": config,
    }


def _gpu_environment(physical_gpu: int) -> dict:
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES=str(int(physical_gpu)),
        CUDA_DEVICE_ORDER="PCI_BUS_ID", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
    )
    return environment


def _run_logged(command: list[str], *, environment: dict, log: Path) -> dict:
    started = time.time()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x") as stream:
        stream.write("COMMAND " + json.dumps(command) + "\n")
        stream.flush()
        result = subprocess.run(
            command, stdout=stream, stderr=subprocess.STDOUT,
            env=environment, text=True,
        )
    return {
        "command": command, "log": str(log), "returncode": result.returncode,
        "wall_seconds": time.time() - started, "log_sha256": sha256_file(log),
    }


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--source-root", required=True)
    value.add_argument("--expected-source-commit", required=True)
    value.add_argument("--checkpoint", required=True)
    value.add_argument("--expected-checkpoint-sha256", required=True)
    value.add_argument("--pretrain-dataset-root", required=True)
    value.add_argument("--expected-pretrain-dataset-manifest-sha256", required=True)
    value.add_argument("--lambda-report", required=True)
    value.add_argument("--expected-lambda-report-sha256", required=True)
    value.add_argument("--output-root", required=True)
    value.add_argument("--gpus", default="1,3")
    value.add_argument("--scene-profile", default="double_density_velocity_ood")
    value.add_argument("--scenario-start", type=int, default=400_000)
    value.add_argument("--rounds", type=int, default=3)
    value.add_argument("--verifier-workers", type=int, default=32)
    value.add_argument("--batch-size", type=int, default=64)
    value.add_argument("--learning-rate", type=float, default=1.0e-6)
    value.add_argument("--microbatch-repeats", type=int, default=10)
    value.add_argument("--flow-base-std", type=float, default=1.4)
    value.add_argument("--negative-alpha", type=float, default=1.0e-3)
    value.add_argument("--seed", type=int, default=2)
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    root = Path(args.source_root).resolve()
    output = Path(args.output_root).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output root: {output}")
    if output == root or root in output.parents:
        raise ValueError("output root must be outside the frozen source worktree")
    if args.rounds < 1:
        raise ValueError("rounds must be positive")
    if args.verifier_workers < 1 or args.batch_size < 1:
        raise ValueError("worker and batch counts must be positive")
    if args.microbatch_repeats < 1 or args.learning_rate <= 0.0:
        raise ValueError("learning dose must be positive")
    if not abs(float(args.flow_base_std) - 1.4) <= 1e-8:
        raise ValueError("this frozen protocol requires flow_base_std=1.4")
    source = source_gate(root, args.expected_source_commit)
    checkpoint_sha = sha256_file(args.checkpoint)
    if checkpoint_sha != args.expected_checkpoint_sha256.lower():
        raise RuntimeError("checkpoint SHA256 mismatch")
    lambda_report_path = Path(args.lambda_report).resolve()
    lambda_report, selected_lambda = load_selected_lambda(
        lambda_report_path,
        expected_sha256=args.expected_lambda_report_sha256,
        checkpoint_sha256=checkpoint_sha,
    )
    gpus = tuple(int(item) for item in args.gpus.split(","))
    arms = declared_arms(gpus)

    output.mkdir(parents=True)
    contract = {
        "status": "SFM_HP100_EARLY_TWO_ARM_DECLARED", "version": VERSION,
        "source": source,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "pretrain_dataset_root": str(Path(args.pretrain_dataset_root).resolve()),
        "pretrain_dataset_manifest_sha256":
            args.expected_pretrain_dataset_manifest_sha256,
        "lambda_report": str(lambda_report_path),
        "lambda_report_sha256": args.expected_lambda_report_sha256.lower(),
        "selected_lambda": selected_lambda,
        "selection_row": lambda_report["selected"],
        "arms": [asdict(arm) for arm in arms],
        "shared_recipe": {
            "rounds": args.rounds, "scenario_start": args.scenario_start,
            "seed": args.seed, "flow_base_std": args.flow_base_std,
            "learning_rate": args.learning_rate,
            "microbatch_repeats": args.microbatch_repeats,
            "negative_alpha": args.negative_alpha,
        },
    }
    _write_json(output / "TWO_ARM_CONTRACT.json", contract)

    preflights = []
    signatures = []
    for arm in arms:
        arm_root = output / "arms" / arm.name
        command = expansion_command(
            args, arm, selected_lambda=selected_lambda, mode="preflight",
        )
        result = _run_logged(
            command, environment=_gpu_environment(arm.physical_gpu),
            log=arm_root / "preflight.log",
        )
        if result["returncode"] != 0:
            raise RuntimeError(json.dumps(result))
        preflight_path = arm_root / "run.preflight.json"
        if not preflight_path.is_file():
            raise RuntimeError(f"missing preflight artifact for {arm.name}")
        payload = json.loads(preflight_path.read_text())
        if payload.get("status") != "SFM_HP100_EARLY_EXPANSION_PREFLIGHT_PASSED":
            raise RuntimeError(f"invalid preflight status for {arm.name}")
        signatures.append(shared_preflight_signature(payload))
        result.update(
            arm=arm.name, physical_gpu=arm.physical_gpu,
            preflight=str(preflight_path),
            preflight_sha256=sha256_file(preflight_path),
        )
        preflights.append(result)
    if signatures[0] != signatures[1]:
        raise RuntimeError("optimizer-scope arms have different shared preflight contracts")
    if float(signatures[0]["expansion_config_except_optimizer_scope"][
        "execution_step_margin_weight"
    ]) != selected_lambda:
        raise RuntimeError("preflight did not use the selected lambda")
    _write_json(output / "PREFLIGHT_COMPLETE.json", {
        "status": "SFM_HP100_EARLY_TWO_ARM_PREFLIGHT_COMPLETE",
        "shared_signature": signatures[0], "arms": preflights,
    })

    def train(arm: Arm) -> dict:
        arm_root = output / "arms" / arm.name
        command = expansion_command(
            args, arm, selected_lambda=selected_lambda, mode="run",
        )
        result = _run_logged(
            command, environment=_gpu_environment(arm.physical_gpu),
            log=arm_root / "train.log",
        )
        result.update(arm=arm.name, physical_gpu=arm.physical_gpu)
        if result["returncode"] != 0:
            raise RuntimeError(json.dumps(result))
        marker = arm_root / "run" / "EARLY_EXPANSION_COMPLETE.json"
        if not marker.is_file():
            raise RuntimeError(f"missing delivery marker for {arm.name}")
        delivery = json.loads(marker.read_text())
        if delivery.get("status") != "SFM_HP100_EARLY_EXPANSION_COMPLETE":
            raise RuntimeError(f"invalid delivery marker for {arm.name}")
        result.update(marker=str(marker), marker_sha256=sha256_file(marker))
        return result

    results = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(train, arm): arm for arm in arms}
        for future in as_completed(futures):
            results.append(future.result())
    final_source = source_gate(root, args.expected_source_commit)
    _write_json(output / "TWO_ARM_COMPLETE.json", {
        "status": "SFM_HP100_EARLY_TWO_ARM_COMPLETE", "version": VERSION,
        "selected_lambda": selected_lambda, "source_after": final_source,
        "arms": sorted(results, key=lambda row: row["arm"]),
    })
    print(json.dumps({
        "status": "SFM_HP100_EARLY_TWO_ARM_COMPLETE",
        "output": str(output), "selected_lambda": selected_lambda,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
