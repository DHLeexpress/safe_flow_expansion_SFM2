"""Authenticated compact Ncausal-only HP100 disaster-prefix sweep.

Stage 0 runs one identical reference dose for ``head_only`` and
``last_block_and_head`` and selects a scope on one fixed raw-M10 bank without
making that reference dose an incremental-treatment gate.  Stage 1 runs a
small declared dose table on that one scope and enforces the incremental
gates, deduplicating raw evaluations of scientifically identical alpha-zero
anchors.  The selected treatment, its matched anchor, and r0 are then
evaluated on a disjoint raw-M50 bank.

This driver never exposes original Dminus rows to negative ascent.  Its only
negative training rows are the authenticated N=3 causal prefixes selected by
``sfm_hp100_disaster_prefix_continuation.py``.
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


VERSION = "sfm_hp100_disaster_prefix_sweep_v1"
SCENE_PROFILE = "double_density_velocity_ood"
OPTIMIZER_SCOPES = ("head_only", "last_block_and_head")
SCREEN_EP0 = 610_000
SCREEN_M = 10
SCREEN_NOISE_SEED = 20_260_805
CONFIRM_EP0 = 620_000
CONFIRM_M = 50
CONFIRM_NOISE_SEED = 20_260_806
NEGATIVE_LEARNING_RATE = 1.0e-4
NEGATIVE_SOURCE = "Ncausal_only"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Dose:
    name: str
    p1_learning_rate: float
    p1_passes: int
    p2_learning_rate: float
    p2_passes: int
    negative_alpha: float
    negative_passes: int

    @property
    def anchor_key(self) -> tuple[float, int, float, int]:
        return (
            self.p1_learning_rate, self.p1_passes,
            self.p2_learning_rate, self.p2_passes,
        )


REFERENCE_DOSE = Dose(
    "current_p2_a0p1_n4", 1.0e-5, 1, 1.0e-3, 1, 0.1, 4,
)


def declared_stage1_doses() -> tuple[Dose, ...]:
    """Return the frozen compact dose table, including the Stage-0 reference."""
    return (
        REFERENCE_DOSE,
        Dose("current_p2_a0p1_n16", 1.0e-5, 1, 1.0e-3, 1, 0.1, 16),
        Dose("current_p2_a0p3_n4", 1.0e-5, 1, 1.0e-3, 1, 0.3, 4),
        Dose("soft_p2_a0p1_n4", 1.0e-5, 1, 3.0e-4, 1, 0.1, 4),
        Dose("soft_p2_a0p1_n16", 1.0e-5, 1, 3.0e-4, 1, 0.1, 16),
        Dose("retain_p1_a0p1_n4", 3.0e-5, 1, 3.0e-4, 1, 0.1, 4),
        Dose("gentle4_a0p1_n4", 1.0e-5, 4, 1.0e-4, 4, 0.1, 4),
    )


@dataclass(frozen=True)
class ScopeInput:
    optimizer_scope: str
    trace: Path
    trace_sha256: str
    staged_samples: Path
    staged_samples_sha256: str


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


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


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


def _pooled(path: Path) -> dict:
    return json.loads(path.read_text())["summary"]["pooled"]


def _evaluation_cache_key(record: dict) -> str:
    return str(record.get("policy_state_sha256", record["sha256"]))


def _verify_input(path: str | Path, expected_sha256: str, label: str) -> dict:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    actual = sha256_file(resolved)
    if actual != str(expected_sha256).lower():
        raise RuntimeError(f"{label} SHA256 mismatch: {actual} != {expected_sha256}")
    return {"path": str(resolved), "sha256": actual}


def _aggregate_m64(marker: dict, model: str) -> dict:
    rows = marker["fixed_raw_M64"]["rows"]
    if not rows:
        raise ValueError("fixed raw M64 audit contains no rows")
    summaries = [row[model] for row in rows]
    total = sum(int(row["M"]) for row in summaries)
    if total <= 0:
        raise ValueError("fixed raw M64 audit has no candidates")

    def weighted(field: str) -> float:
        return sum(float(row[field]) * int(row["M"]) for row in summaries) / total

    exact_positive = sum(int(row["exact_positive"]) for row in summaries)
    return {
        "M_total": int(total), "lineages": len(summaries),
        "exact_positive": int(exact_positive),
        "Validity": float(exact_positive / total),
        "mean_H10_progress": weighted("mean_H10_progress"),
        "mean_step_margin": weighted("mean_step_margin"),
    }


def _candidate_summary(
    *, scope: str, dose: Dose, marker: dict,
    r0_raw: dict, anchor_raw: dict, treatment_raw: dict,
    anchor_checkpoint: dict, treatment_checkpoint: dict,
) -> dict:
    r0_m64 = _aggregate_m64(marker, "r0")
    anchor_m64 = _aggregate_m64(marker, "alpha0_positive_anchor")
    treatment_m64 = _aggregate_m64(marker, "treatment")
    targeted = marker["targeted_Nplus1"]
    delta_sr = float(treatment_raw["SR"] - anchor_raw["SR"])
    delta_validity = float(treatment_m64["Validity"] - anchor_m64["Validity"])
    progress_drop = float(
        anchor_m64["mean_H10_progress"] - treatment_m64["mean_H10_progress"]
    )
    gates = {
        "incremental_raw_SR_strictly_positive": delta_sr > 0.0,
        "incremental_M64_Validity_strictly_positive": delta_validity > 0.0,
        "timeout_nonincrease": treatment_raw["timeout"] <= anchor_raw["timeout"],
        "target_recovery_nonregression": (
            int(targeted["treatment_recovered"])
            >= int(targeted["alpha0_recovered"])
        ),
        "mean_H10_progress_drop_at_most_0p02": progress_drop <= 0.02,
        "r0_raw_SR_nonregression": treatment_raw["SR"] >= r0_raw["SR"],
        "r0_raw_Validity_nonregression": (
            treatment_raw["Validity"] >= r0_raw["Validity"]
        ),
        "r0_timeout_nonincrease": treatment_raw["timeout"] <= r0_raw["timeout"],
        "r0_M64_Validity_nonregression": (
            treatment_m64["Validity"] >= r0_m64["Validity"]
        ),
    }
    return {
        "scope": scope, "dose": asdict(dose),
        "anchor_checkpoint": anchor_checkpoint,
        "treatment_checkpoint": treatment_checkpoint,
        "r0_raw_M10": r0_raw, "anchor_raw_M10": anchor_raw,
        "treatment_raw_M10": treatment_raw,
        "r0_M64": r0_m64, "anchor_M64": anchor_m64,
        "treatment_M64": treatment_m64,
        "targeted_recovery": {
            "r0": int(targeted["r0_recovered"]),
            "anchor": int(targeted["alpha0_recovered"]),
            "treatment": int(targeted["treatment_recovered"]),
        },
        "relative_parameter_drift": float(
            marker["update"]["relative_parameter_drift"]
        ),
        "deltas": {
            "raw_SR": delta_sr, "M64_Validity": delta_validity,
            "raw_Validity": float(
                treatment_raw["Validity"] - anchor_raw["Validity"]
            ),
            "timeout": float(treatment_raw["timeout"] - anchor_raw["timeout"]),
            "mean_H10_progress": float(
                treatment_m64["mean_H10_progress"]
                - anchor_m64["mean_H10_progress"]
            ),
        },
        "gates": gates, "eligible": bool(all(gates.values())),
    }


def candidate_key(row: dict) -> tuple:
    """Rank only after eligibility: bottleneck gain, recovery, validity, drift."""
    return (
        min(row["deltas"]["raw_SR"], row["deltas"]["M64_Validity"]),
        row["targeted_recovery"]["treatment"],
        row["treatment_raw_M10"]["Validity"],
        -row["relative_parameter_drift"],
    )


def confirmation_decision(
    treatment: dict, anchor: dict, r0: dict,
) -> tuple[dict, dict, bool, bool]:
    incremental = {
        "raw_SR": float(treatment["SR"] - anchor["SR"]),
        "raw_Validity": float(treatment["Validity"] - anchor["Validity"]),
        "timeout": float(treatment["timeout"] - anchor["timeout"]),
    }
    versus_r0 = {
        "raw_SR": float(treatment["SR"] - r0["SR"]),
        "raw_Validity": float(treatment["Validity"] - r0["Validity"]),
        "timeout": float(treatment["timeout"] - r0["timeout"]),
    }
    incremental_confirmed = bool(
        incremental["raw_SR"] > 0.0
        and incremental["raw_Validity"] > 0.0
        and incremental["timeout"] <= 0.0
    )
    strict_r0_win = bool(
        versus_r0["raw_SR"] > 0.0
        and versus_r0["raw_Validity"] > 0.0
        and versus_r0["timeout"] <= 0.0
    )
    return incremental, versus_r0, incremental_confirmed, strict_r0_win


def choose_stage0_scope(rows: list[dict]) -> tuple[dict, dict]:
    """Choose capacity only when last-block liveness matches head-only.

    The one-reference-dose incremental eligibility is diagnostic at Stage 0;
    it cannot stop the later dose search.  Head-only wins immediately if the
    larger scope loses at least one success out of the pooled 70-rollout M10
    bank, or increases timeout, for either its anchor or its treatment.
    """
    by_scope = {row["scope"]: row for row in rows}
    if set(by_scope) != set(OPTIMIZER_SCOPES):
        raise ValueError("Stage 0 requires exactly one cell per optimizer scope")
    head = by_scope["head_only"]
    last = by_scope["last_block_and_head"]
    tolerance = 1.0 / 70.0
    comparisons = {}
    liveness_regression = False
    for role in ("anchor_raw_M10", "treatment_raw_M10"):
        sr_delta = float(last[role]["SR"] - head[role]["SR"])
        timeout_delta = float(last[role]["timeout"] - head[role]["timeout"])
        comparisons[role] = {
            "last_minus_head_SR": sr_delta,
            "last_minus_head_timeout": timeout_delta,
        }
        if sr_delta <= -tolerance + 1.0e-12 or timeout_delta > 0.0:
            liveness_regression = True
    if liveness_regression:
        return head, {
            "rule": "head_only_due_to_last_block_liveness_regression",
            "one_success_tolerance": tolerance, "comparisons": comparisons,
        }
    selected = max(
        (head, last),
        key=lambda row: (
            row["treatment_M64"]["Validity"],
            row["anchor_M64"]["Validity"],
            row["treatment_raw_M10"]["Validity"],
            row["anchor_raw_M10"]["Validity"],
            int(row["scope"] == "last_block_and_head"),
        ),
    )
    return selected, {
        "rule": "liveness_matched_then_M64_raw_validity_capacity_tiebreak",
        "one_success_tolerance": tolerance, "comparisons": comparisons,
    }


def _scope_inputs(args) -> dict[str, ScopeInput]:
    return {
        "head_only": ScopeInput(
            "head_only", Path(args.head_trace).resolve(),
            args.expected_head_trace_sha256,
            Path(args.head_staged_samples).resolve(),
            args.expected_head_staged_samples_sha256,
        ),
        "last_block_and_head": ScopeInput(
            "last_block_and_head", Path(args.last_block_trace).resolve(),
            args.expected_last_block_trace_sha256,
            Path(args.last_block_staged_samples).resolve(),
            args.expected_last_block_staged_samples_sha256,
        ),
    }


def _continuation_command(
    *, python: str, continuation: Path, checkpoint: Path,
    checkpoint_sha256: str, dataset_root: Path, dataset_manifest_sha256: str,
    scope_input: ScopeInput, dose: Dose, output: Path, gpu: int,
    verifier_workers: int, scenario_start: int, max_drift: float,
) -> list[str]:
    return [
        python, "-u", str(continuation), "--mode", "run",
        "--checkpoint", str(checkpoint),
        "--expected-checkpoint-sha256", checkpoint_sha256,
        "--prior-trace", str(scope_input.trace),
        "--expected-prior-trace-sha256", scope_input.trace_sha256,
        "--prior-staged-samples", str(scope_input.staged_samples),
        "--expected-prior-staged-samples-sha256",
        scope_input.staged_samples_sha256,
        "--pretrain-dataset-root", str(dataset_root),
        "--expected-pretrain-dataset-manifest-sha256", dataset_manifest_sha256,
        "--output", str(output), "--device", "cuda:0",
        "--physical-gpu", str(gpu), "--optimizer-scope",
        scope_input.optimizer_scope, "--scene-profile", SCENE_PROFILE,
        "--scenario-start", str(scenario_start),
        "--verifier-workers", str(verifier_workers),
        "--p1-learning-rate", str(dose.p1_learning_rate),
        "--p1-passes", str(dose.p1_passes),
        "--p2-learning-rate", str(dose.p2_learning_rate),
        "--p2-passes", str(dose.p2_passes),
        "--negative-learning-rate", str(NEGATIVE_LEARNING_RATE),
        "--negative-alpha", str(dose.negative_alpha),
        "--negative-passes", str(dose.negative_passes),
        "--negative-source", NEGATIVE_SOURCE, "--batch-size", "64",
        "--max-relative-parameter-drift", str(max_drift),
        "--min-recovered", "6",
    ]


def _validate_run_marker(
    marker: dict, dose: Dose, scope_input: ScopeInput,
) -> None:
    preflight = marker["preflight"]
    if preflight["optimizer_scope"] != scope_input.optimizer_scope:
        raise RuntimeError("continuation optimizer scope drifted")
    source = preflight["source"]
    if source["trace_sha256"] != scope_input.trace_sha256.lower():
        raise RuntimeError("continuation trace provenance drifted")
    if source["staged_samples_sha256"] != scope_input.staged_samples_sha256.lower():
        raise RuntimeError("continuation staged-sample provenance drifted")
    update_contract = preflight["update"]
    expected = {
        "P1_lr": dose.p1_learning_rate, "P1_passes": dose.p1_passes,
        "P2_lr": dose.p2_learning_rate, "P2_passes": dose.p2_passes,
        "negative_lr": NEGATIVE_LEARNING_RATE,
        "negative_alpha": dose.negative_alpha,
        "negative_passes": dose.negative_passes,
        "negative_source": NEGATIVE_SOURCE,
    }
    for key, value in expected.items():
        if update_contract[key] != value:
            raise RuntimeError(f"continuation contract drifted at {key}")
    negative = marker["update"]["negative"]
    if negative["negative_source"] != NEGATIVE_SOURCE:
        raise RuntimeError("continuation used an undeclared negative source")
    if int(negative["selected_source_counts"]["Dminus"]) != 0:
        raise RuntimeError("Dminus entered an Ncausal-only treatment")
    if int(negative["selected_source_counts"]["Ncausal"]) <= 0:
        raise RuntimeError("Ncausal-only treatment contains no Ncausal rows")


def _rejected_cell(run: dict) -> dict:
    return {
        "scope": run["scope"], "dose": run["dose"], "run": run,
        "eligible": False, "gates": {"atomic_update_accepted": False},
        "ineligible_reason": "relative-parameter-drift or finite update gate",
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--source-root", required=True)
    value.add_argument("--expected-source-commit", required=True)
    value.add_argument("--checkpoint", required=True)
    value.add_argument("--expected-checkpoint-sha256", required=True)
    value.add_argument("--pretrain-dataset-root", required=True)
    value.add_argument("--expected-pretrain-dataset-manifest-sha256", required=True)
    value.add_argument("--head-trace", required=True)
    value.add_argument("--expected-head-trace-sha256", required=True)
    value.add_argument("--head-staged-samples", required=True)
    value.add_argument("--expected-head-staged-samples-sha256", required=True)
    value.add_argument("--last-block-trace", required=True)
    value.add_argument("--expected-last-block-trace-sha256", required=True)
    value.add_argument("--last-block-staged-samples", required=True)
    value.add_argument("--expected-last-block-staged-samples-sha256", required=True)
    value.add_argument("--output-root", required=True)
    value.add_argument("--gpus", default="1,3")
    value.add_argument("--jobs-per-gpu", type=int, default=1)
    value.add_argument("--verifier-workers", type=int, default=16)
    value.add_argument("--eval-workers", type=int, default=4)
    value.add_argument("--scenario-start", type=int, default=500_000)
    value.add_argument("--max-relative-parameter-drift", type=float, default=0.10)
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    root = Path(args.source_root).resolve()
    output = Path(args.output_root).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output root: {output}")
    if min(args.jobs_per_gpu, args.verifier_workers, args.eval_workers) < 1:
        raise ValueError("worker counts must be positive")
    gpus = tuple(int(item) for item in args.gpus.split(",") if item.strip())
    if not gpus:
        raise ValueError("at least one GPU is required")
    if args.max_relative_parameter_drift <= 0.0:
        raise ValueError("max relative parameter drift must be positive")

    source = source_gate(root, args.expected_source_commit)
    checkpoint = _verify_input(
        args.checkpoint, args.expected_checkpoint_sha256, "r0 checkpoint",
    )
    dataset_manifest = _verify_input(
        Path(args.pretrain_dataset_root) / "manifest.json",
        args.expected_pretrain_dataset_manifest_sha256, "pretrain dataset manifest",
    )
    scope_inputs = _scope_inputs(args)
    authenticated_scopes = {}
    for scope, item in scope_inputs.items():
        authenticated_scopes[scope] = {
            "trace": _verify_input(item.trace, item.trace_sha256, f"{scope} trace"),
            "staged_samples": _verify_input(
                item.staged_samples, item.staged_samples_sha256,
                f"{scope} staged samples",
            ),
        }
    continuation = (
        root / "source_snapshot/overnight_run_07_12_sfm/"
        "sfm_hp100_disaster_prefix_continuation.py"
    )
    evaluator = (
        root / "source_snapshot/overnight_run_07_12_sfm/sfm_hp100_eval.py"
    )
    source_components = {
        "continuation": _verify_input(
            continuation, sha256_file(continuation), "continuation source",
        ),
        "raw_evaluator": _verify_input(
            evaluator, sha256_file(evaluator), "raw evaluator source",
        ),
    }
    output.mkdir(parents=True)
    doses = declared_stage1_doses()
    contract = {
        "status": "SFM_HP100_DISASTER_PREFIX_SWEEP_DECLARED",
        "version": VERSION, "source": source, "checkpoint": checkpoint,
        "source_components": source_components,
        "pretrain_dataset": {
            "root": str(Path(args.pretrain_dataset_root).resolve()),
            "manifest": dataset_manifest,
        },
        "authenticated_scope_inputs": authenticated_scopes,
        "negative_contract": {
            "source": NEGATIVE_SOURCE, "Dminus_exposures": 0,
            "learning_rate_before_alpha": NEGATIVE_LEARNING_RATE,
        },
        "stage0": {
            "scopes": list(OPTIMIZER_SCOPES),
            "reference_dose": asdict(REFERENCE_DOSE),
            "screen": {
                "ep0": SCREEN_EP0, "M_per_gamma": SCREEN_M,
                "noise_seed": SCREEN_NOISE_SEED, "temperature": 1.0,
            },
            "selection": (
                "reference eligibility is diagnostic only; choose head if the "
                "last-block anchor or treatment loses >=1/70 SR or gains timeout; "
                "otherwise M64/raw Validity and capacity break ties"
            ),
        },
        "stage1": {
            "selected_scope_only": True,
            "doses": [asdict(dose) for dose in doses],
            "anchor_eval_cache": "policy-state SHA256",
        },
        "eligibility": {
            "raw_SR": "treatment > matched alpha0 anchor",
            "M64_Validity": "treatment > matched alpha0 anchor",
            "timeout": "treatment <= matched alpha0 anchor",
            "targeted_recovery": "treatment >= matched alpha0 anchor",
            "mean_H10_progress_drop_max": 0.02,
            "r0_nonregression": (
                "treatment raw SR/Validity/M64 Validity >= r0 and timeout <= r0"
            ),
        },
        "ranking": (
            "min(delta raw SR, delta M64 Validity), targeted recovery, "
            "raw Validity, lower parameter drift"
        ),
        "confirmation": {
            "ep0": CONFIRM_EP0, "M_per_gamma": CONFIRM_M,
            "noise_seed": CONFIRM_NOISE_SEED, "temperature": 1.0,
            "cells": ["r0", "selected treatment", "matched alpha0 anchor"],
        },
        "execution": {
            "gpus": list(gpus), "jobs_per_gpu": args.jobs_per_gpu,
            "verifier_workers_per_job": args.verifier_workers,
            "eval_workers_per_job": args.eval_workers,
        },
    }
    _write_json(output / "SWEEP_CONTRACT.json", contract)

    def run_dose(job, gpu: int):
        stage, scope, dose = job
        arm_root = output / stage / "arms" / scope / dose.name
        run_root = arm_root / "run"
        command = _continuation_command(
            python=sys.executable, continuation=continuation,
            checkpoint=Path(checkpoint["path"]),
            checkpoint_sha256=checkpoint["sha256"],
            dataset_root=Path(args.pretrain_dataset_root).resolve(),
            dataset_manifest_sha256=dataset_manifest["sha256"],
            scope_input=scope_inputs[scope], dose=dose, output=run_root,
            gpu=gpu, verifier_workers=args.verifier_workers,
            scenario_start=args.scenario_start,
            max_drift=args.max_relative_parameter_drift,
        )
        execution = _run_logged(
            command, environment=_gpu_environment(gpu), log=arm_root / "train.log",
        )
        if execution["returncode"] != 0:
            raise RuntimeError(json.dumps(execution))
        complete_path = run_root / "DIAGNOSTIC_COMPLETE.json"
        rejected_path = run_root / "UPDATE_REJECTED.json"
        markers = [path for path in (complete_path, rejected_path) if path.is_file()]
        if len(markers) != 1:
            raise RuntimeError(f"expected one continuation marker in {run_root}")
        marker_path = markers[0]
        marker = json.loads(marker_path.read_text())
        _validate_run_marker(marker, dose, scope_inputs[scope])
        base_result = {
            "stage": stage, "scope": scope, "dose": asdict(dose),
            "root": str(run_root), "marker": str(marker_path),
            "marker_sha256": sha256_file(marker_path), "execution": execution,
            "rejected": marker_path == rejected_path,
        }
        if marker_path == rejected_path:
            if marker.get("status") != "SFM_HP100_DISASTER_PREFIX_UPDATE_REJECTED":
                raise RuntimeError("unexpected continuation rejection status")
            return base_result
        if marker.get("status") != "SFM_HP100_DISASTER_PREFIX_DIAGNOSTIC_COMPLETE":
            raise RuntimeError("unexpected continuation completion status")
        anchor = Path(marker["checkpoints"]["alpha0_positive_anchor"])
        treatment = Path(marker["checkpoints"]["treatment_diagnostic"])
        anchor_sha = sha256_file(anchor)
        treatment_sha = sha256_file(treatment)
        if anchor_sha != marker["checkpoints"]["alpha0_positive_anchor_sha256"]:
            raise RuntimeError("anchor checkpoint digest disagrees with marker")
        if treatment_sha != marker["checkpoints"]["treatment_diagnostic_sha256"]:
            raise RuntimeError("treatment checkpoint digest disagrees with marker")
        return base_result | {
            "anchor_checkpoint": {
                "path": str(anchor), "sha256": anchor_sha,
                "policy_state_sha256": marker["update"][
                    "positive_anchor_model_sha256"
                ],
            },
            "treatment_checkpoint": {
                "path": str(treatment), "sha256": treatment_sha,
                "policy_state_sha256": marker["update"][
                    "treatment_model_sha256_after_gate"
                ],
            },
        }

    def evaluate_unique(
        records: list[dict], *, stage: str,
        existing: dict[str, dict] | None = None,
    ) -> dict[str, dict]:
        existing = {} if existing is None else existing
        unique = {}
        for record in records:
            cache_key = _evaluation_cache_key(record)
            unique.setdefault(cache_key, record)
        cached = {key: existing[key] for key in unique if key in existing}
        missing = {key: record for key, record in unique.items() if key not in cached}

        def evaluate(item, gpu: int):
            cache_key, record = item
            destination = (
                output / "evaluation" / stage / f"{cache_key[:16]}_m10.json"
            )
            command = [
                sys.executable, "-u", str(evaluator), "--checkpoint", record["path"],
                "--scene-profile", SCENE_PROFILE, "--ep0", str(SCREEN_EP0),
                "--M", str(SCREEN_M), "--device", "cuda:0",
                "--noise-seed", str(SCREEN_NOISE_SEED),
                "--verifier-workers", str(args.eval_workers), "--out",
                str(destination),
            ]
            execution = _run_logged(
                command, environment=_gpu_environment(gpu),
                log=destination.with_suffix(".log"),
            )
            if execution["returncode"] != 0 or not destination.is_file():
                raise RuntimeError(json.dumps(execution))
            return cache_key, {
                "checkpoint": record, "path": str(destination),
                "sha256": sha256_file(destination), "metrics": _pooled(destination),
                "execution": execution, "physical_gpu": gpu,
                "cache_key": cache_key,
            }

        rows = _parallel_gpu_jobs(
            list(missing.items()), gpus=gpus, jobs_per_gpu=args.jobs_per_gpu,
            worker=evaluate,
        )
        return cached | dict(rows)

    r0_record = checkpoint
    stage0_runs = _parallel_gpu_jobs(
        [("stage0", scope, REFERENCE_DOSE) for scope in OPTIMIZER_SCOPES],
        gpus=gpus, jobs_per_gpu=1, worker=run_dose,
    )
    stage0_eval_records = [r0_record]
    for run in stage0_runs:
        if not run["rejected"]:
            stage0_eval_records.extend((
                run["anchor_checkpoint"], run["treatment_checkpoint"],
            ))
    stage0_evaluations = evaluate_unique(stage0_eval_records, stage="stage0")
    r0_raw_m10 = stage0_evaluations[_evaluation_cache_key(r0_record)]["metrics"]
    stage0_cells = []
    for run in stage0_runs:
        if run["rejected"]:
            stage0_cells.append(_rejected_cell(run))
            continue
        marker = json.loads(Path(run["marker"]).read_text())
        anchor = run["anchor_checkpoint"]
        treatment = run["treatment_checkpoint"]
        cell = _candidate_summary(
            scope=run["scope"], dose=REFERENCE_DOSE, marker=marker,
            r0_raw=r0_raw_m10,
            anchor_raw=stage0_evaluations[_evaluation_cache_key(anchor)]["metrics"],
            treatment_raw=stage0_evaluations[
                _evaluation_cache_key(treatment)
            ]["metrics"],
            anchor_checkpoint=anchor, treatment_checkpoint=treatment,
        )
        cell["run"] = run
        stage0_cells.append(cell)
    accepted_scope_cells = [cell for cell in stage0_cells if not cell["run"]["rejected"]]
    if len(accepted_scope_cells) == 2:
        selected_scope_cell, scope_selection = choose_stage0_scope(
            accepted_scope_cells
        )
    elif len(accepted_scope_cells) == 1:
        selected_scope_cell = accepted_scope_cells[0]
        scope_selection = {
            "rule": "only_scope_with_atomic_update_accepted",
            "rejected_scope": next(
                cell["scope"] for cell in stage0_cells if cell["run"]["rejected"]
            ),
        }
    else:
        selected_scope_cell = None
        scope_selection = {"rule": "both_scope_reference_updates_rejected"}
    stage0_marker = {
        "status": (
            "SFM_HP100_DISASTER_PREFIX_STAGE0_COMPLETE"
            if selected_scope_cell is not None else
            "SFM_HP100_DISASTER_PREFIX_STAGE0_NO_ACCEPTED_SCOPE"
        ),
        "r0_raw_M10": stage0_evaluations[_evaluation_cache_key(r0_record)],
        "cells": stage0_cells,
        "selected_scope": (
            None if selected_scope_cell is None else selected_scope_cell["scope"]
        ),
        "selected_reference_cell": selected_scope_cell,
        "selection_audit": scope_selection,
        "reference_incremental_eligibility_is_not_a_stage1_gate": True,
        "unique_checkpoint_evaluations": len(stage0_evaluations),
    }
    _write_json(output / "STAGE0_COMPLETE.json", stage0_marker)
    if selected_scope_cell is None:
        final = {
            "status": "SFM_HP100_DISASTER_PREFIX_SWEEP_NO_ACCEPTED_SCOPE",
            "stage0": str(output / "STAGE0_COMPLETE.json"),
            "stage1_performed": False, "confirmation_performed": False,
        }
        _write_json(output / "SWEEP_COMPLETE.json", final)
        print(json.dumps(final), flush=True)
        return 0

    selected_scope = selected_scope_cell["scope"]
    selected_stage0_run = next(
        run for run in stage0_runs if run["scope"] == selected_scope
    )
    remaining = [dose for dose in doses if dose != REFERENCE_DOSE]
    stage1_new_runs = _parallel_gpu_jobs(
        [("stage1", selected_scope, dose) for dose in remaining],
        gpus=gpus, jobs_per_gpu=args.jobs_per_gpu, worker=run_dose,
    )
    stage1_runs = [selected_stage0_run, *stage1_new_runs]
    stage1_eval_records = []
    for run in stage1_runs:
        if not run["rejected"]:
            stage1_eval_records.extend((
                run["anchor_checkpoint"], run["treatment_checkpoint"],
            ))
    stage1_evaluations = evaluate_unique(
        stage1_eval_records, stage="stage1", existing=stage0_evaluations,
    )
    stage1_cells = []
    for run in stage1_runs:
        if run["rejected"]:
            stage1_cells.append(_rejected_cell(run))
            continue
        dose = Dose(**run["dose"])
        marker = json.loads(Path(run["marker"]).read_text())
        anchor = run["anchor_checkpoint"]
        treatment = run["treatment_checkpoint"]
        cell = _candidate_summary(
            scope=selected_scope, dose=dose, marker=marker,
            r0_raw=r0_raw_m10,
            anchor_raw=stage1_evaluations[_evaluation_cache_key(anchor)]["metrics"],
            treatment_raw=stage1_evaluations[
                _evaluation_cache_key(treatment)
            ]["metrics"],
            anchor_checkpoint=anchor, treatment_checkpoint=treatment,
        )
        cell["run"] = run
        stage1_cells.append(cell)
    eligible = [cell for cell in stage1_cells if cell["eligible"]]
    winner = max(eligible, key=candidate_key) if eligible else None
    stage1_marker = {
        "status": (
            "SFM_HP100_DISASTER_PREFIX_STAGE1_COMPLETE"
            if winner is not None else
            "SFM_HP100_DISASTER_PREFIX_STAGE1_NO_ELIGIBLE_DOSE"
        ),
        "selected_scope": selected_scope, "cells": stage1_cells,
        "winner": winner,
        "unique_checkpoint_evaluations": len(stage1_evaluations),
        "anchor_evaluation_cache_savings": (
            sum(not run["rejected"] for run in stage1_runs) - len({
                _evaluation_cache_key(run["anchor_checkpoint"])
                for run in stage1_runs if not run["rejected"]
            })
        ),
    }
    _write_json(output / "STAGE1_COMPLETE.json", stage1_marker)
    if winner is None:
        final = {
            "status": "SFM_HP100_DISASTER_PREFIX_SWEEP_NO_ELIGIBLE_DOSE",
            "stage0": str(output / "STAGE0_COMPLETE.json"),
            "stage1": str(output / "STAGE1_COMPLETE.json"),
            "confirmation_performed": False,
        }
        _write_json(output / "SWEEP_COMPLETE.json", final)
        print(json.dumps(final), flush=True)
        return 0

    confirmation_records = [
        ("r0", r0_record),
        ("matched_alpha0", winner["anchor_checkpoint"]),
        ("selected_treatment", winner["treatment_checkpoint"]),
    ]

    def confirm(job, gpu: int):
        label, record = job
        destination = output / "confirmation" / f"{label}_m50.json"
        command = [
            sys.executable, "-u", str(evaluator), "--checkpoint", record["path"],
            "--scene-profile", SCENE_PROFILE, "--ep0", str(CONFIRM_EP0),
            "--M", str(CONFIRM_M), "--device", "cuda:0",
            "--noise-seed", str(CONFIRM_NOISE_SEED),
            "--verifier-workers", str(args.eval_workers), "--out", str(destination),
        ]
        execution = _run_logged(
            command, environment=_gpu_environment(gpu),
            log=destination.with_suffix(".log"),
        )
        if execution["returncode"] != 0 or not destination.is_file():
            raise RuntimeError(json.dumps(execution))
        return {
            "label": label, "checkpoint": record, "path": str(destination),
            "sha256": sha256_file(destination), "metrics": _pooled(destination),
            "execution": execution, "physical_gpu": gpu,
        }

    confirmation = _parallel_gpu_jobs(
        confirmation_records, gpus=gpus, jobs_per_gpu=1, worker=confirm,
    )
    by_label = {row["label"]: row for row in confirmation}
    treatment_metrics = by_label["selected_treatment"]["metrics"]
    anchor_metrics = by_label["matched_alpha0"]["metrics"]
    r0_metrics = by_label["r0"]["metrics"]
    (
        confirmation_incremental, confirmation_vs_r0,
        incremental_confirmed, strict_r0_win,
    ) = confirmation_decision(
        treatment_metrics, anchor_metrics, r0_metrics,
    )
    confirmation_marker = {
        "status": "SFM_HP100_DISASTER_PREFIX_DISJOINT_M50_COMPLETE",
        "bank": {
            "ep0": CONFIRM_EP0, "M_per_gamma": CONFIRM_M,
            "noise_seed": CONFIRM_NOISE_SEED, "temperature": 1.0,
        },
        "winner_from_M10_only": winner, "cells": confirmation,
        "selected_treatment_minus_matched_anchor": confirmation_incremental,
        "selected_treatment_minus_r0": confirmation_vs_r0,
        "incremental_gate_confirmed": incremental_confirmed,
        "strict_r0_win": strict_r0_win,
        "eligible_for_phase2": bool(incremental_confirmed and strict_r0_win),
    }
    _write_json(output / "CONFIRMATION_COMPLETE.json", confirmation_marker)
    final = {
        "status": "SFM_HP100_DISASTER_PREFIX_SWEEP_COMPLETE",
        "selected_scope": selected_scope, "selected_dose": winner["dose"],
        "selected_treatment_checkpoint": winner["treatment_checkpoint"],
        "matched_alpha0_checkpoint": winner["anchor_checkpoint"],
        "stage0": str(output / "STAGE0_COMPLETE.json"),
        "stage1": str(output / "STAGE1_COMPLETE.json"),
        "confirmation": str(output / "CONFIRMATION_COMPLETE.json"),
        "disjoint_M50_incremental_gate_confirmed": incremental_confirmed,
        "disjoint_M50_strict_r0_win": strict_r0_win,
        "eligible_for_phase2": confirmation_marker["eligible_for_phase2"],
    }
    _write_json(output / "SWEEP_COMPLETE.json", final)
    print(json.dumps(final), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
