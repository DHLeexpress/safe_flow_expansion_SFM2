"""Run and screen the eight-cell cumulative HP100 micro-block study.

All cells use the same selected starting checkpoint, four gathering gammas,
eight paired pedestrian scenarios, and sampling seed.  They differ only in
optimizer scope and the declared learning configuration.  The driver launches
all eight cell subprocesses on physical GPU 1, waits for authenticated
``ARM_COMPLETE.json`` markers, then evaluates r0, the selected start, and all
eight final checkpoints on one new shared raw temperature-one M50 bank.  The
locked Kazuki comparator uses the same r0 prior and scenario bank.

The M50 result selects a development candidate among eight cells; it is not an
unbiased paper confirmation.  A later disjoint M100 remains required.
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

import sfm_hp100_preterminal_sweep as RANK


VERSION = "sfm_hp100_cumulative_microblock_sweep_v1"
SCENE_PROFILE = "double_density_velocity_ood"
GATHER_GAMMAS = (0.1, 0.3, 0.5, 1.0)
OPTIMIZER_SCOPES = ("last_block_and_head", "last_two_blocks_and_head")
MICRO_ROUNDS = 5
LINEAGES_PER_GAMMA = 8
BATCH_SIZE = 64
EFFECTIVE_PASSES = 10
GP_BUFFER_CAP = 2_688
GP_REPLAY_ROUNDS = 2
QUOTAS = {"P1": 32, "P2": 16, "Ncausal": 12, "Dminus": 4}
DEFAULT_EVAL_EP0 = 700_000
DEFAULT_EVAL_NOISE_SEED = 20_260_810


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class LearningConfig:
    name: str
    learning_rate: float
    p2_weight: float
    negative_alpha: float


@dataclass(frozen=True)
class Cell:
    optimizer_scope: str
    config: LearningConfig

    @property
    def name(self) -> str:
        return f"{self.optimizer_scope}__{self.config.name}"


def declared_configs() -> tuple[LearningConfig, ...]:
    return (
        LearningConfig("C1", 3.0e-5, 2.0, 0.01),
        LearningConfig("C2", 3.0e-5, 2.0, 0.05),
        LearningConfig("C3", 1.0e-4, 2.0, 0.01),
        LearningConfig("C4", 1.0e-4, 2.0, 0.05),
    )


def declared_cells() -> tuple[Cell, ...]:
    return tuple(
        Cell(scope, config)
        for scope in OPTIMIZER_SCOPES
        for config in declared_configs()
    )


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ("git", "-C", str(root), *args), text=True,
    ).strip()


def source_gate(root: Path, expected_commit: str) -> dict:
    head = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain")
    if head != str(expected_commit):
        raise RuntimeError(f"HEAD mismatch: {head} != {expected_commit}")
    if status:
        raise RuntimeError("cumulative sweep requires a clean frozen worktree")
    remote = _git(root, "rev-parse", "origin/master")
    if remote != head:
        raise RuntimeError(f"origin/master mismatch: {remote} != {head}")
    return {"root": str(root), "HEAD": head, "origin_master": remote, "clean": True}


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def _verified_file(path: str | Path, expected_sha256: str, role: str) -> dict:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {role}: {resolved}")
    actual = sha256_file(resolved)
    if actual != str(expected_sha256).lower():
        raise RuntimeError(f"{role} SHA256 mismatch: {actual} != {expected_sha256}")
    return {"path": str(resolved), "sha256": actual}


def _selection_mentions_checkpoint(selection: Path, checkpoint_sha256: str) -> None:
    payload = json.loads(selection.read_text())

    def contains(value) -> bool:
        if isinstance(value, dict):
            return any(contains(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(contains(item) for item in value)
        return str(value).lower() == checkpoint_sha256.lower()

    if not contains(payload):
        raise RuntimeError("selection marker does not authenticate the start checkpoint")


def _gpu_environment(physical_gpu: int) -> dict:
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES=str(int(physical_gpu)),
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
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
        "command": command,
        "log": str(log),
        "returncode": int(result.returncode),
        "wall_seconds": float(time.time() - started),
        "log_sha256": sha256_file(log),
    }


def _pooled(path: Path) -> dict:
    return json.loads(path.read_text())["summary"]["pooled"]


def _checkpoint_record(value, *, marker_path: Path) -> dict:
    if not isinstance(value, dict) or "path" not in value or "sha256" not in value:
        raise RuntimeError("ARM_COMPLETE final_checkpoint must contain path and sha256")
    path = Path(value["path"])
    if not path.is_absolute():
        path = (marker_path.parent / path).resolve()
    record = _verified_file(path, value["sha256"], "cell final checkpoint")
    if "policy_state_sha256" in value:
        record["policy_state_sha256"] = str(value["policy_state_sha256"])
    return record


def _arm_delivery(
    arm_root: Path, *, cell: Cell, expected_start_sha256: str,
) -> tuple[dict, dict, tuple[int, ...]]:
    marker_path = arm_root / "ARM_COMPLETE.json"
    if not marker_path.is_file():
        raise RuntimeError(f"missing ARM_COMPLETE.json: {marker_path}")
    payload = json.loads(marker_path.read_text())
    if not str(payload.get("status", "")).endswith("ARM_COMPLETE"):
        raise RuntimeError(f"invalid arm status in {marker_path}")
    expected_cell = {
        "optimizer_scope": cell.optimizer_scope,
        "learning_rate": float(cell.config.learning_rate),
        "p2_weight": float(cell.config.p2_weight),
        "negative_alpha": float(cell.config.negative_alpha),
    }
    if payload.get("cell") != expected_cell:
        raise RuntimeError(f"cell contract drift in {marker_path}")
    start = payload.get("preflight", {}).get("start_checkpoint", {})
    if str(start.get("sha256", "")).lower() != expected_start_sha256.lower():
        raise RuntimeError(f"cell did not use the declared start checkpoint: {marker_path}")
    if payload.get("persistent_Adam") is not True:
        raise RuntimeError(f"cell did not preserve Adam state: {marker_path}")
    if int(payload.get("micro_rounds_retained_without_rollback", -1)) != MICRO_ROUNDS:
        raise RuntimeError(f"cell did not retain all five micro-rounds: {marker_path}")
    rounds = payload.get("rounds", ())
    if [int(row.get("micro_round", -1)) for row in rounds] != list(
        range(1, MICRO_ROUNDS + 1)
    ):
        raise RuntimeError(f"cell has an incomplete micro-round sequence: {marker_path}")
    checkpoint = _checkpoint_record(
        payload.get("final_checkpoint"), marker_path=marker_path,
    )
    scenario_ids = payload.get("training_scenario_ids")
    if not isinstance(scenario_ids, list) or not scenario_ids:
        raise RuntimeError("ARM_COMPLETE must declare nonempty training_scenario_ids")
    scenario_ids = tuple(sorted({int(value) for value in scenario_ids}))
    return (
        {"path": str(marker_path), "sha256": sha256_file(marker_path)},
        checkpoint,
        scenario_ids,
    )


def _cell_command(args, root: Path, cell: Cell, arm_root: Path) -> list[str]:
    runner = root / (
        "source_snapshot/overnight_run_07_12_sfm/"
        "sfm_hp100_cumulative_microblocks.py"
    )
    return [
        sys.executable, "-u", str(runner),
        "--source-root", str(root),
        "--expected-source-commit", args.expected_source_commit,
        "--selection-marker", str(Path(args.selection_marker).resolve()),
        "--expected-selection-marker-sha256",
        args.expected_selection_marker_sha256,
        "--checkpoint", str(Path(args.checkpoint).resolve()),
        "--expected-checkpoint-sha256", args.expected_checkpoint_sha256,
        "--reference-checkpoint", str(Path(args.r0_checkpoint).resolve()),
        "--expected-reference-checkpoint-sha256", args.expected_r0_checkpoint_sha256,
        "--pretrain-dataset-root", str(Path(args.pretrain_dataset_root).resolve()),
        "--expected-pretrain-dataset-manifest-sha256",
        args.expected_pretrain_dataset_manifest_sha256,
        "--output-root", str(arm_root),
        "--device", "cuda:0",
        "--physical-gpu", str(args.physical_gpu),
        "--optimizer-scope", cell.optimizer_scope,
        "--learning-rate", str(cell.config.learning_rate),
        "--p2-weight", str(cell.config.p2_weight),
        "--negative-alpha", str(cell.config.negative_alpha),
        "--micro-rounds", str(MICRO_ROUNDS),
        "--replicas-per-gamma", str(LINEAGES_PER_GAMMA),
        "--passes", str(EFFECTIVE_PASSES),
        "--batch-size", str(BATCH_SIZE),
        "--gp-buffer-cap", str(GP_BUFFER_CAP),
        "--gp-replay-rounds", str(GP_REPLAY_ROUNDS),
        "--scene-profile", args.scene_profile,
        "--scenario-start", str(args.scenario_start),
        "--verifier-workers", str(args.verifier_workers_per_cell),
        "--max-steps", "180",
        "--max-repair-batches", str(args.max_repair_batches),
        "--ess-target", "0.1",
        "--rbf-noise", "0.01",
        "--seed", str(args.seed),
    ]


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--source-root", required=True)
    value.add_argument("--expected-source-commit", required=True)
    value.add_argument("--selection-marker", required=True)
    value.add_argument("--expected-selection-marker-sha256", required=True)
    value.add_argument("--checkpoint", required=True)
    value.add_argument("--expected-checkpoint-sha256", required=True)
    value.add_argument("--r0-checkpoint", required=True)
    value.add_argument("--expected-r0-checkpoint-sha256", required=True)
    value.add_argument("--pretrain-dataset-root", required=True)
    value.add_argument("--expected-pretrain-dataset-manifest-sha256", required=True)
    value.add_argument("--output-root", required=True)
    value.add_argument("--physical-gpu", type=int, default=1)
    value.add_argument("--scene-profile", default=SCENE_PROFILE)
    value.add_argument("--scenario-start", type=int, default=800_000)
    value.add_argument("--seed", type=int, default=17)
    value.add_argument("--verifier-workers-per-cell", type=int, default=8)
    value.add_argument("--max-repair-batches", type=int, default=32)
    value.add_argument("--eval-workers", type=int, default=8)
    value.add_argument("--eval-jobs", type=int, default=4)
    value.add_argument("--eval-ep0", type=int, default=DEFAULT_EVAL_EP0)
    value.add_argument("--eval-M", type=int, default=50)
    value.add_argument("--eval-noise-seed", type=int, default=DEFAULT_EVAL_NOISE_SEED)
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    root = Path(args.source_root).resolve()
    output = Path(args.output_root).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output root: {output}")
    if output == root or root in output.parents:
        raise ValueError("output root must remain outside the frozen source worktree")
    if args.physical_gpu != 1:
        raise ValueError("this declared study must run on physical GPU 1")
    if args.eval_M != 50:
        raise ValueError("the declared disjoint screening bank is M=50 per gamma")
    if min(
        args.verifier_workers_per_cell, args.max_repair_batches,
        args.eval_workers, args.eval_jobs,
    ) < 1:
        raise ValueError("worker, repair, and evaluation counts must be positive")

    source = source_gate(root, args.expected_source_commit)
    cell_runner = root / (
        "source_snapshot/overnight_run_07_12_sfm/"
        "sfm_hp100_cumulative_microblocks.py"
    )
    evaluator = root / "source_snapshot/overnight_run_07_12_sfm/sfm_hp100_eval.py"
    kazuki = root / "source_snapshot/overnight_run_07_12_sfm/sfm_hp100_kazuki_eval.py"
    for role, path in (
        ("cumulative cell runner", cell_runner),
        ("raw evaluator", evaluator),
        ("locked Kazuki evaluator", kazuki),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing {role}: {path}")

    selection = _verified_file(
        args.selection_marker, args.expected_selection_marker_sha256,
        "selection marker",
    )
    start = _verified_file(
        args.checkpoint, args.expected_checkpoint_sha256,
        "selected starting checkpoint",
    )
    r0 = _verified_file(
        args.r0_checkpoint, args.expected_r0_checkpoint_sha256,
        "canonical r0 checkpoint",
    )
    _selection_mentions_checkpoint(Path(selection["path"]), start["sha256"])
    dataset_manifest = _verified_file(
        Path(args.pretrain_dataset_root).resolve() / "manifest.json",
        args.expected_pretrain_dataset_manifest_sha256,
        "pretrain dataset manifest",
    )
    cells = declared_cells()
    output.mkdir(parents=True)
    contract = {
        "status": "SFM_HP100_CUMULATIVE_MICROBLOCK_SWEEP_DECLARED",
        "version": VERSION,
        "source": source,
        "selection_marker": selection,
        "start_checkpoint": start,
        "r0_checkpoint": r0,
        "pretrain_dataset": {
            "root": str(Path(args.pretrain_dataset_root).resolve()),
            "manifest": dataset_manifest,
        },
        "cells": [
            {"name": cell.name, "optimizer_scope": cell.optimizer_scope,
             "config": asdict(cell.config)}
            for cell in cells
        ],
        "shared_training": {
            "micro_rounds": MICRO_ROUNDS,
            "gammas": list(GATHER_GAMMAS),
            "lineages_per_gamma": LINEAGES_PER_GAMMA,
            "gamma_lineages": len(GATHER_GAMMAS) * LINEAGES_PER_GAMMA,
            "distinct_pedestrian_scenarios": LINEAGES_PER_GAMMA,
            "same_scenarios_across_gamma_cells_and_micro_rounds": True,
            "batch_size": BATCH_SIZE,
            "quotas": QUOTAS,
            "effective_passes": EFFECTIVE_PASSES,
            "N_Adam": "10 * max(ceil(nP1/32),ceil(nP2/16),ceil(nN/12),ceil(nD-/4))",
            "persistent_Adam_across_micro_rounds": True,
            "scientific_rollback_inside_five_round_block": False,
            "gp_buffer_cap_total": GP_BUFFER_CAP,
            "gp_cap_per_active_gamma": GP_BUFFER_CAP // len(GATHER_GAMMAS),
            "gp_reference_rounds": GP_REPLAY_ROUNDS,
            "signed_negative_cfm_is_unbounded_ablation": True,
        },
        "execution": {
            "physical_gpu": args.physical_gpu,
            "simultaneous_cell_processes": len(cells),
            "verifier_workers_per_cell": args.verifier_workers_per_cell,
        },
        "screen": {
            "scene_profile": args.scene_profile,
            "bank": [args.eval_ep0, args.eval_ep0 + args.eval_M - 1],
            "M_per_gamma": args.eval_M,
            "noise_seed": args.eval_noise_seed,
            "all_seven_gammas": True,
            "temperature": 1.0,
            "NFE": 8,
            "comparators": ["canonical r0", "selected start", "locked Kazuki"],
            "selection_is_development_screen_not_final_confirmation": True,
        },
    }
    _write_json(output / "SWEEP_CONTRACT.json", contract)

    def train(cell: Cell) -> dict:
        arm_root = output / "arms" / cell.name
        result = _run_logged(
            _cell_command(args, root, cell, arm_root),
            environment=_gpu_environment(args.physical_gpu),
            log=output / "logs" / f"{cell.name}.train.log",
        )
        result.update(cell=cell.name, physical_gpu=args.physical_gpu)
        if result["returncode"] != 0:
            raise RuntimeError(json.dumps(result))
        marker, checkpoint, scenario_ids = _arm_delivery(
            arm_root, cell=cell, expected_start_sha256=start["sha256"],
        )
        result.update(
            marker=marker, final_checkpoint=checkpoint,
            training_scenario_ids=list(scenario_ids),
        )
        return result

    training = []
    with ThreadPoolExecutor(max_workers=len(cells)) as executor:
        futures = {executor.submit(train, cell): cell for cell in cells}
        for future in as_completed(futures):
            training.append(future.result())
    training.sort(key=lambda row: row["cell"])
    scenario_sets = {tuple(row["training_scenario_ids"]) for row in training}
    if len(scenario_sets) != 1:
        raise RuntimeError("cells did not gather the identical paired scenario bank")
    training_scenarios = next(iter(scenario_sets))
    eval_scenarios = set(range(args.eval_ep0, args.eval_ep0 + args.eval_M))
    overlap = sorted(set(training_scenarios) & eval_scenarios)
    if overlap:
        raise RuntimeError(f"training/evaluation scenario overlap: {overlap}")
    _write_json(output / "TRAINING_COMPLETE.json", {
        "status": "SFM_HP100_CUMULATIVE_MICROBLOCK_TRAINING_COMPLETE",
        "runs": training,
        "shared_training_scenario_ids": list(training_scenarios),
    })

    checkpoints = [
        {"name": "r0", "kind": "baseline", "checkpoint": r0},
        {"name": "selected_start", "kind": "baseline", "checkpoint": start},
    ]
    by_cell = {row["cell"]: row for row in training}
    checkpoints.extend({
        "name": cell.name,
        "kind": "cell",
        "optimizer_scope": cell.optimizer_scope,
        "config": asdict(cell.config),
        "checkpoint": by_cell[cell.name]["final_checkpoint"],
    } for cell in cells)

    def evaluate(record: dict) -> dict:
        destination = output / "m50" / f"{record['name']}.json"
        command = [
            sys.executable, "-u", str(evaluator),
            "--checkpoint", record["checkpoint"]["path"],
            "--scene-profile", args.scene_profile,
            "--ep0", str(args.eval_ep0),
            "--M", str(args.eval_M),
            "--device", "cuda:0",
            "--noise-seed", str(args.eval_noise_seed),
            "--verifier-workers", str(args.eval_workers),
            "--out", str(destination),
        ]
        result = _run_logged(
            command, environment=_gpu_environment(args.physical_gpu),
            log=output / "logs" / f"{record['name']}.m50.log",
        )
        if result["returncode"] != 0 or not destination.is_file():
            raise RuntimeError(json.dumps(result))
        return {
            **record,
            "artifact": {"path": str(destination), "sha256": sha256_file(destination)},
            "metrics": _pooled(destination),
            "execution": result,
        }

    evaluations = []
    with ThreadPoolExecutor(max_workers=args.eval_jobs) as executor:
        futures = {executor.submit(evaluate, row): row for row in checkpoints}
        for future in as_completed(futures):
            evaluations.append(future.result())
    evaluations.sort(key=lambda row: row["name"])
    r0_metrics = next(row["metrics"] for row in evaluations if row["name"] == "r0")
    start_metrics = next(
        row["metrics"] for row in evaluations if row["name"] == "selected_start"
    )
    candidates = [row for row in evaluations if row["kind"] == "cell"]
    candidates.sort(
        key=lambda row: (
            RANK.development_key(
                row["metrics"], start_metrics, round_index=MICRO_ROUNDS,
            ),
            row["name"],
        ),
        reverse=True,
    )
    strict_candidates = [
        row for row in candidates if RANK.strict_win(row["metrics"], start_metrics)
    ]
    selected = strict_candidates[0] if strict_candidates else candidates[0]

    kazuki_out = output / "m50" / "locked_kazuki.json"
    kazuki_execution = _run_logged([
        sys.executable, "-u", str(kazuki),
        "--checkpoint", r0["path"],
        "--expected-checkpoint-sha256", r0["sha256"],
        "--expected-source-commit", args.expected_source_commit,
        "--scene-profile", args.scene_profile,
        "--ep0", str(args.eval_ep0),
        "--M", str(args.eval_M),
        "--device", "cuda:0",
        "--workers", str(args.eval_workers),
        "--out", str(kazuki_out),
    ], environment=_gpu_environment(args.physical_gpu),
       log=output / "logs" / "locked_kazuki.m50.log")
    if kazuki_execution["returncode"] != 0 or not kazuki_out.is_file():
        raise RuntimeError(json.dumps(kazuki_execution))
    kazuki_metrics = _pooled(kazuki_out)

    final_source = source_gate(root, args.expected_source_commit)
    delivery = {
        "status": "SFM_HP100_CUMULATIVE_MICROBLOCK_SWEEP_COMPLETE",
        "version": VERSION,
        "source_after": final_source,
        "r0_M50": r0_metrics,
        "selected_start_M50": start_metrics,
        "locked_kazuki_M50": kazuki_metrics,
        "locked_kazuki_artifact": {
            "path": str(kazuki_out), "sha256": sha256_file(kazuki_out),
            "execution": kazuki_execution,
        },
        "selected_cell": selected,
        "selection_status": (
            "strict_win_over_selected_start"
            if strict_candidates else "observed_best_no_strict_win"
        ),
        "selected_strict_win_over_start": RANK.strict_win(
            selected["metrics"], start_metrics,
        ),
        "selected_strict_win_over_r0": RANK.strict_win(
            selected["metrics"], r0_metrics,
        ),
        "selected_strict_win_over_locked_kazuki": RANK.strict_win(
            selected["metrics"], kazuki_metrics,
        ),
        "strict_win_definition": (
            "CR lower, Validity and successful clearance higher, SR no more "
            "than 0.03 lower, timeout no more than 0.03 higher"
        ),
        "ranked_cells": candidates,
        "raw_M50_evaluations": evaluations,
        "scientific_note": (
            "One new shared raw temp=1 M50 bank screened eight final cells. "
            "This is a multiple-comparison development selection; a fresh "
            "disjoint M100 is required before a final paper claim."
        ),
    }
    _write_json(output / "SWEEP_COMPLETE.json", delivery)
    print(json.dumps({
        "status": delivery["status"],
        "output": str(output),
        "selection_status": delivery["selection_status"],
        "selected_cell": selected["name"],
        "selected_metrics": selected["metrics"],
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
