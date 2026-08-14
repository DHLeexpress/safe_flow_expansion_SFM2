"""Frozen two-GPU sweep for the paper-SOCP HP100 preterminal protocol.

Training is a declared 3x3 grid over the execution margin weight and signed
negative-gradient strength.  It never retries a failed arm.  A fixed raw-M10
bank screens r1/r2/r5/r10, then a disjoint raw-M50 bank evaluates the three
shortlisted cells plus the untouched pretrained and locked Kazuki baselines.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time


VERSION = "sfm_hp100_preterminal_sweep_v1"
LAMBDAS = (0.0, 70_000.0, 140_000.0)
ALPHAS = (0.0, 0.001, 0.01)
SCREEN_ROUNDS = (1, 2, 5, 10)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _slug(value: float) -> str:
    if value == 0:
        return "0"
    return f"{value:g}".replace(".", "p").replace("-", "m")


@dataclass(frozen=True)
class Arm:
    lambda_weight: float
    negative_alpha: float

    @property
    def name(self) -> str:
        return f"lambda{int(self.lambda_weight):06d}_alpha{_slug(self.negative_alpha)}"


def declared_arms() -> tuple[Arm, ...]:
    return tuple(Arm(weight, alpha) for weight in LAMBDAS for alpha in ALPHAS)


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
        raise RuntimeError("frozen sweep requires a clean worktree")
    return {"root": str(root), "HEAD": head, "clean": True}


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


def _gpu_environment(physical_gpu: int) -> dict:
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES=str(int(physical_gpu)),
        CUDA_DEVICE_ORDER="PCI_BUS_ID", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
    )
    return environment


def _parallel_gpu_jobs(jobs, *, gpus, jobs_per_gpu, worker):
    slots: queue.Queue[int] = queue.Queue()
    for gpu in gpus:
        for _ in range(jobs_per_gpu):
            slots.put(int(gpu))
    results = []

    def wrapped(job):
        gpu = slots.get()
        try:
            return worker(job, gpu)
        finally:
            slots.put(gpu)

    with ThreadPoolExecutor(max_workers=slots.qsize()) as executor:
        futures = {executor.submit(wrapped, job): job for job in jobs}
        for future in as_completed(futures):
            results.append(future.result())
    return results


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _pooled(path: Path) -> dict:
    return json.loads(path.read_text())["summary"]["pooled"]


def _successful_clearance(metrics: dict) -> float:
    value = metrics.get("successful_clearance")
    return float("-inf") if value is None else float(value)


def _successful_time(metrics: dict) -> float:
    value = metrics.get("successful_time_to_goal")
    return float("inf") if value is None else float(value)


def _clearance_gain(metrics: dict, baseline: dict) -> float:
    clearance = _successful_clearance(metrics)
    baseline_clearance = _successful_clearance(baseline)
    if baseline_clearance == float("-inf"):
        return 0.0 if clearance == float("-inf") else float("inf")
    return clearance - baseline_clearance


def development_key(metrics: dict, baseline: dict, *, round_index: int):
    clearance = _successful_clearance(metrics)
    baseline_clearance = _successful_clearance(baseline)
    safety_improvements = sum((
        metrics["CR"] < baseline["CR"],
        metrics["Validity"] > baseline["Validity"],
        clearance > baseline_clearance,
    ))
    admissible = (
        metrics["SR"] >= baseline["SR"] - 0.10
        and metrics["timeout"] <= baseline["timeout"] + 0.10
    )
    return (
        int(admissible), int(safety_improvements),
        baseline["CR"] - metrics["CR"],
        metrics["Validity"] - baseline["Validity"],
        _clearance_gain(metrics, baseline),
        metrics["SR"], -_successful_time(metrics), -int(round_index),
    )


def strict_win(metrics: dict, baseline: dict) -> bool:
    return bool(
        metrics["CR"] < baseline["CR"]
        and metrics["Validity"] > baseline["Validity"]
        and _successful_clearance(metrics) > _successful_clearance(baseline)
        and metrics["SR"] >= baseline["SR"] - 0.03
        and metrics["timeout"] <= baseline["timeout"] + 0.03
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--source-root", required=True)
    value.add_argument("--expected-source-commit", required=True)
    value.add_argument("--checkpoint", required=True)
    value.add_argument("--expected-checkpoint-sha256", required=True)
    value.add_argument("--pretrain-dataset-root", required=True)
    value.add_argument("--expected-pretrain-dataset-manifest-sha256", required=True)
    value.add_argument("--output-root", required=True)
    value.add_argument("--gpus", default="1,3")
    value.add_argument("--jobs-per-gpu", type=int, default=2)
    value.add_argument("--verifier-workers", type=int, default=8)
    value.add_argument("--eval-workers", type=int, default=4)
    value.add_argument("--rounds", type=int, default=10)
    value.add_argument("--screen-ep0", type=int, default=350_000)
    value.add_argument("--screen-M", type=int, default=10)
    value.add_argument("--confirm-ep0", type=int, default=360_000)
    value.add_argument("--confirm-M", type=int, default=50)
    value.add_argument("--scenario-start", type=int, default=300_000)
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    root = Path(args.source_root).resolve()
    output = Path(args.output_root).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output root: {output}")
    if args.rounds < max(SCREEN_ROUNDS):
        raise ValueError("declared screening requires at least ten rounds")
    if args.jobs_per_gpu < 1 or args.verifier_workers < 1 or args.eval_workers < 1:
        raise ValueError("worker counts must be positive")
    gpus = tuple(int(item) for item in args.gpus.split(","))
    if not gpus:
        raise ValueError("at least one GPU is required")
    source = source_gate(root, args.expected_source_commit)
    if sha256_file(args.checkpoint) != args.expected_checkpoint_sha256:
        raise RuntimeError("checkpoint SHA256 mismatch")
    output.mkdir(parents=True)
    arms = declared_arms()
    contract = {
        "status": "SFM_HP100_PRETERMINAL_SWEEP_DECLARED", "version": VERSION,
        "source": source, "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "scene_profile": "double_density_velocity_ood",
        "mechanism": {
            "rounds": args.rounds, "K": 16, "B": 4,
            "parallel_episodes_per_gamma": 16, "ESS_target": 0.1,
            "RBF_cap_total": 1344, "RBF_cap_per_gamma": 192,
            "GP_balance": "gamma -> lineage -> unique time stage",
            "archive": "all resolved selected-B queries through lineage terminal decision",
            "scenario_batches_per_gamma_round": 1,
            "success_retry": False,
            "positive": "exact full-H paper-SOCP y=1",
            "negative": "resolved exact y=0; never GP support",
            "paired_noised_representation": True,
            "head_only": True, "learning_rate": 1e-6,
            "microbatch_repeats": 10, "replay_rounds": 3,
        },
        "arms": [asdict(arm) | {"name": arm.name} for arm in arms],
        "screen": {
            "bank": [args.screen_ep0, args.screen_ep0 + args.screen_M - 1],
            "M_per_gamma": args.screen_M, "rounds": list(SCREEN_ROUNDS),
            "temperature": 1.0, "shortlist": 3,
            "ranking": (
                "liveness gate; number of improved CR/Validity/clearance; then "
                "CR, Validity, clearance, SR, time, earlier round"
            ),
        },
        "confirmation": {
            "bank": [args.confirm_ep0, args.confirm_ep0 + args.confirm_M - 1],
            "M_per_gamma": args.confirm_M, "temperature": 1.0,
            "comparators": ["untouched pretrained", "locked Kazuki"],
        },
        "execution": {"gpus": list(gpus), "jobs_per_gpu": args.jobs_per_gpu,
                      "verifier_workers_per_arm": args.verifier_workers},
    }
    _write_json(output / "SWEEP_CONTRACT.json", contract)

    launcher = root / "source_snapshot/overnight_run_07_12_sfm/sfm_hp100_ball_launch.py"
    evaluator = root / "source_snapshot/overnight_run_07_12_sfm/sfm_hp100_eval.py"
    kazuki = root / "source_snapshot/overnight_run_07_12_sfm/sfm_hp100_kazuki_eval.py"

    def train(arm: Arm, gpu: int):
        arm_root = output / "arms" / arm.name
        run_root = arm_root / "run"
        command = [
            sys.executable, "-u", str(launcher), "--mode", "run",
            "--checkpoint", str(Path(args.checkpoint).resolve()),
            "--expected-checkpoint-sha256", args.expected_checkpoint_sha256,
            "--pretrain-dataset-root", str(Path(args.pretrain_dataset_root).resolve()),
            "--expected-pretrain-dataset-manifest-sha256",
            args.expected_pretrain_dataset_manifest_sha256,
            "--output", str(run_root), "--device", "cuda:0",
            "--physical-gpu", str(gpu), "--scene-profile",
            "double_density_velocity_ood", "--scenario-start", str(args.scenario_start),
            "--rounds", str(args.rounds), "--parallel-episodes", "16",
            "--verifier-workers", str(args.verifier_workers),
            "--max-retry-batches", "1", "--successful-trajectories-per-gamma", "1",
            "--batch-size", "64", "--microbatch-repeats", "10",
            "--learning-rate", "1e-6", "--flow-base-std", "1.0",
            "--initial-beta", "0.0005", "--ess-target", "0.1",
            "--gp-buffer-cap", "1344", "--negative-alpha", str(arm.negative_alpha),
            "--execution-step-margin-weight", str(arm.lambda_weight),
            "--paired-noised-representation", "--seed", "2",
        ]
        result = _run_logged(
            command, environment=_gpu_environment(gpu), log=arm_root / "train.log",
        )
        result.update(arm=arm.name, physical_gpu=gpu)
        if result["returncode"] != 0:
            raise RuntimeError(json.dumps(result))
        marker = run_root / "SFM_HP100_BALL_PORT_COMPLETE.json"
        if not marker.is_file():
            raise RuntimeError(f"missing training marker for {arm.name}")
        delivery = json.loads(marker.read_text())
        scenario_ids = {
            int(row["scenario_id"]) for row in delivery.get("scene_ledger", ())
        }
        reserved = set(range(args.screen_ep0, args.screen_ep0 + args.screen_M))
        reserved.update(range(args.confirm_ep0, args.confirm_ep0 + args.confirm_M))
        overlap = sorted(scenario_ids & reserved)
        if overlap:
            raise RuntimeError(
                f"training/evaluation scenario-bank overlap for {arm.name}: {overlap}"
            )
        result["marker"] = str(marker)
        result["marker_sha256"] = sha256_file(marker)
        result["distinct_training_scenarios"] = len(scenario_ids)
        return result

    training = _parallel_gpu_jobs(
        arms, gpus=gpus, jobs_per_gpu=args.jobs_per_gpu, worker=train,
    )
    _write_json(output / "TRAINING_COMPLETE.json", {
        "status": "SFM_HP100_PRETERMINAL_TRAINING_COMPLETE", "runs": training,
    })

    eval_jobs = [("pretrained", 0, Path(args.checkpoint).resolve())]
    for arm in arms:
        for round_index in SCREEN_ROUNDS:
            checkpoint = output / "arms" / arm.name / "run" / "evaluation_checkpoints" / f"checkpoint_{round_index:03d}.pt"
            if not checkpoint.is_file():
                raise RuntimeError(f"missing screening checkpoint: {checkpoint}")
            eval_jobs.append((arm.name, round_index, checkpoint))

    def screen(job, gpu: int):
        arm_name, round_index, checkpoint = job
        destination = output / "screening" / arm_name / f"r{round_index:03d}_m10.json"
        command = [
            sys.executable, "-u", str(evaluator), "--checkpoint", str(checkpoint),
            "--scene-profile", "double_density_velocity_ood",
            "--ep0", str(args.screen_ep0), "--M", str(args.screen_M),
            "--device", "cuda:0", "--noise-seed", "20260803",
            "--verifier-workers", str(args.eval_workers), "--out", str(destination),
        ]
        result = _run_logged(
            command, environment=_gpu_environment(gpu),
            log=destination.with_suffix(".log"),
        )
        if result["returncode"] != 0 or not destination.is_file():
            raise RuntimeError(json.dumps(result))
        return {"arm": arm_name, "round": round_index, "path": str(destination),
                "sha256": sha256_file(destination), "metrics": _pooled(destination),
                "physical_gpu": gpu}

    screening = _parallel_gpu_jobs(
        eval_jobs, gpus=gpus, jobs_per_gpu=args.jobs_per_gpu, worker=screen,
    )
    baseline = next(row["metrics"] for row in screening if row["arm"] == "pretrained")
    candidates = [row for row in screening if row["arm"] != "pretrained"]
    candidates.sort(
        key=lambda row: development_key(
            row["metrics"], baseline, round_index=row["round"],
        ), reverse=True,
    )
    shortlist = candidates[:3]
    _write_json(output / "SCREENING_COMPLETE.json", {
        "status": "SFM_HP100_PRETERMINAL_M10_SCREENING_COMPLETE",
        "baseline": baseline, "shortlist": shortlist, "cells": screening,
    })

    confirm_jobs = [("pretrained", 0, Path(args.checkpoint).resolve())]
    for row in shortlist:
        checkpoint = output / "arms" / row["arm"] / "run" / "evaluation_checkpoints" / f"checkpoint_{row['round']:03d}.pt"
        confirm_jobs.append((row["arm"], row["round"], checkpoint))

    def confirm(job, gpu: int):
        arm_name, round_index, checkpoint = job
        destination = output / "confirmation" / arm_name / f"r{round_index:03d}_m50.json"
        command = [
            sys.executable, "-u", str(evaluator), "--checkpoint", str(checkpoint),
            "--scene-profile", "double_density_velocity_ood",
            "--ep0", str(args.confirm_ep0), "--M", str(args.confirm_M),
            "--device", "cuda:0", "--noise-seed", "20260804",
            "--verifier-workers", str(args.eval_workers), "--out", str(destination),
        ]
        result = _run_logged(command, environment=_gpu_environment(gpu),
                             log=destination.with_suffix(".log"))
        if result["returncode"] != 0 or not destination.is_file():
            raise RuntimeError(json.dumps(result))
        return {"arm": arm_name, "round": round_index, "path": str(destination),
                "sha256": sha256_file(destination), "metrics": _pooled(destination),
                "physical_gpu": gpu}

    confirmation = _parallel_gpu_jobs(
        confirm_jobs, gpus=gpus, jobs_per_gpu=1, worker=confirm,
    )
    confirm_baseline = next(
        row["metrics"] for row in confirmation if row["arm"] == "pretrained"
    )
    finalists = [row for row in confirmation if row["arm"] != "pretrained"]
    finalists.sort(
        key=lambda row: development_key(
            row["metrics"], confirm_baseline, round_index=row["round"],
        ), reverse=True,
    )
    observed_best = finalists[0]

    kazuki_out = output / "confirmation" / "locked_kazuki_m50.json"
    kazuki_result = _run_logged([
        sys.executable, "-u", str(kazuki), "--checkpoint",
        str(Path(args.checkpoint).resolve()), "--expected-checkpoint-sha256",
        args.expected_checkpoint_sha256, "--expected-source-commit",
        args.expected_source_commit, "--scene-profile", "double_density_velocity_ood",
        "--ep0", str(args.confirm_ep0), "--M", str(args.confirm_M),
        "--device", "cuda:0", "--workers", str(args.eval_workers),
        "--out", str(kazuki_out),
    ], environment=_gpu_environment(gpus[0]), log=kazuki_out.with_suffix(".log"))
    if kazuki_result["returncode"] != 0 or not kazuki_out.is_file():
        raise RuntimeError(json.dumps(kazuki_result))
    kazuki_metrics = _pooled(kazuki_out)

    delivery = {
        "status": "SFM_HP100_PRETERMINAL_SWEEP_COMPLETE",
        "pretrained_M50": confirm_baseline, "locked_kazuki_M50": kazuki_metrics,
        "locked_kazuki_M50_artifact": {
            "path": str(kazuki_out), "sha256": sha256_file(kazuki_out),
            "execution": kazuki_result,
        },
        "observed_best_M50": observed_best,
        "strict_pretrained_win": strict_win(observed_best["metrics"], confirm_baseline),
        "strict_win_definition": (
            "CR lower, Validity and successful clearance higher, SR no more than "
            "0.03 below pretrained, timeout no more than 0.03 above pretrained"
        ),
        "confirmation_cells": confirmation,
        "scientific_note": (
            "M10 selected arm/round; disjoint M50 confirms without temperature "
            "tuning. No M100 claim is made by this driver."
        ),
    }
    _write_json(output / "SWEEP_COMPLETE.json", delivery)
    print(json.dumps({
        "status": delivery["status"], "output": str(output),
        "strict_pretrained_win": delivery["strict_pretrained_win"],
        "observed_best": observed_best,
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
