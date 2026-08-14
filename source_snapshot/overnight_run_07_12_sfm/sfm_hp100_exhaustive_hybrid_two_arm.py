"""Frozen two-GPU launcher for the HP100 exhaustive-hybrid qualification."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import sfm_hp100_exhaustive_hybrid as HYBRID


VERSION = "sfm_hp100_exhaustive_hybrid_two_arm_v1"


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


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(("git", "-C", str(root), *args), text=True).strip()


def source_gate(root: Path, expected_commit: str, required_ref: str) -> dict:
    head = _git(root, "rev-parse", "HEAD")
    remote = _git(root, "rev-parse", str(required_ref))
    status = _git(root, "status", "--porcelain")
    if head != str(expected_commit):
        raise RuntimeError(f"HEAD mismatch: {head} != {expected_commit}")
    if status:
        raise RuntimeError("exhaustive-hybrid launch requires a clean frozen worktree")
    if remote != str(expected_commit):
        raise RuntimeError(
            f"required source ref mismatch: {required_ref}={remote} != {expected_commit}"
        )
    return {
        "root": str(root), "HEAD": head, "required_ref": str(required_ref),
        "required_ref_commit": remote, "clean": True,
    }


def gpu_identity(physical_gpu: int) -> dict:
    output = subprocess.check_output([
        "nvidia-smi", f"--id={int(physical_gpu)}",
        "--query-gpu=index,uuid,name",
        "--format=csv,noheader,nounits",
    ], text=True).strip()
    rows = [row.strip() for row in output.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"GPU {physical_gpu} identity is not unique: {rows}")
    index, uuid, name = (value.strip() for value in rows[0].split(",", 2))
    if int(index) != int(physical_gpu):
        raise RuntimeError(f"nvidia-smi remapped physical GPU {physical_gpu} to {index}")
    return {"physical_gpu": int(index), "uuid": uuid, "name": name}


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _gpu_environment(physical_gpu: int) -> dict:
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES=str(int(physical_gpu)),
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
    )
    return environment


def command(args, arm: Arm, *, mode: str) -> list[str]:
    script = (
        Path(args.source_root).resolve()
        / "source_snapshot/overnight_run_07_12_sfm/sfm_hp100_exhaustive_hybrid.py"
    )
    output = Path(args.output_root).resolve() / "arms" / arm.name / "run"
    return [
        sys.executable, "-u", str(script), "--mode", mode,
        "--checkpoint", str(Path(args.checkpoint).resolve()),
        "--expected-checkpoint-sha256", args.expected_checkpoint_sha256,
        "--pretrain-dataset-root", str(Path(args.pretrain_dataset_root).resolve()),
        "--expected-pretrain-dataset-manifest-sha256",
        args.expected_pretrain_dataset_manifest_sha256,
        "--output", str(output), "--device", "cuda:0",
        "--physical-gpu", str(arm.physical_gpu),
        "--optimizer-scope", arm.optimizer_scope,
        "--scene-profile", args.scene_profile,
        "--scenario-start", str(args.scenario_start),
        "--gammas", args.gammas,
        "--lineages-per-gamma", str(args.lineages_per_gamma),
        "--max-steps", str(args.max_steps),
        "--max-repair-batches", str(args.max_repair_batches),
        "--max-microcycles", str(args.max_microcycles),
        "--verifier-workers", str(args.verifier_workers),
        "--ess-target", str(args.ess_target),
        "--rbf-noise", str(args.rbf_noise),
        "--p1-learning-rate", str(args.p1_learning_rate),
        "--p2-learning-rate", str(args.p2_learning_rate),
        "--negative-alpha-base", str(args.negative_alpha_base),
        "--batch-size", str(args.batch_size),
        "--max-relative-parameter-drift", str(args.max_relative_parameter_drift),
        "--gp-buffer-cap", str(args.gp_buffer_cap), "--seed", str(args.seed),
    ]


def _run_logged(command_value: list[str], *, env: dict, log: Path) -> dict:
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log.open("x") as stream:
        stream.write("COMMAND " + json.dumps(command_value) + "\n")
        stream.flush()
        result = subprocess.run(
            command_value, stdout=stream, stderr=subprocess.STDOUT,
            env=env, text=True,
        )
    return {
        "command": command_value, "log": str(log),
        "returncode": int(result.returncode),
        "wall_seconds": float(time.time() - started),
        "log_sha256": HYBRID._sha256(log),
    }


def shared_signature(preflight: dict) -> dict:
    return {
        "version": preflight["version"],
        "checkpoint_sha256": preflight["checkpoint_sha256"],
        "config": preflight["config"],
        "scene": preflight["scene"],
        "scenario_start": preflight["scenario_start"],
        "calibration": preflight["calibration"],
        "acquisition_reference": preflight["acquisition_reference"],
        "transaction": preflight["transaction"],
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--source-root", required=True)
    value.add_argument("--expected-source-commit", required=True)
    value.add_argument("--required-source-ref", default="origin/master")
    value.add_argument("--checkpoint", required=True)
    value.add_argument("--expected-checkpoint-sha256", required=True)
    value.add_argument("--pretrain-dataset-root", required=True)
    value.add_argument("--expected-pretrain-dataset-manifest-sha256", required=True)
    value.add_argument("--output-root", required=True)
    value.add_argument("--gpus", default="1,3")
    value.add_argument("--scene-profile", default="double_density_velocity_ood")
    value.add_argument("--scenario-start", type=int, default=500_000)
    value.add_argument("--gammas", default="0.1,0.3,0.5,1.0")
    value.add_argument("--lineages-per-gamma", type=int, default=4)
    value.add_argument("--max-steps", type=int, default=180)
    value.add_argument("--max-repair-batches", type=int, default=32)
    value.add_argument("--max-microcycles", type=int, default=6)
    value.add_argument("--verifier-workers", type=int, default=16)
    value.add_argument("--ess-target", type=float, default=0.1)
    value.add_argument("--rbf-noise", type=float, default=1.0e-2)
    value.add_argument("--p1-learning-rate", type=float, default=1.0e-5)
    value.add_argument("--p2-learning-rate", type=float, default=1.0e-3)
    value.add_argument("--negative-alpha-base", type=float, default=1.0e-3)
    value.add_argument("--batch-size", type=int, default=64)
    value.add_argument("--max-relative-parameter-drift", type=float, default=0.25)
    value.add_argument("--gp-buffer-cap", type=int, default=2_688)
    value.add_argument("--seed", type=int, default=2)
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    root = Path(args.source_root).resolve()
    output = Path(args.output_root).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output root: {output}")
    if output == root or root in output.parents:
        raise ValueError("outputs must remain outside the frozen worktree")
    source = source_gate(
        root, args.expected_source_commit, args.required_source_ref,
    )
    checkpoint_sha = HYBRID._sha256(args.checkpoint)
    if checkpoint_sha != args.expected_checkpoint_sha256.lower():
        raise RuntimeError("checkpoint SHA256 mismatch")
    arms = declared_arms(tuple(int(item) for item in args.gpus.split(",")))
    gpu_identities = {
        arm.name: gpu_identity(arm.physical_gpu) for arm in arms
    }
    output.mkdir(parents=True)
    _write_json(output / "TWO_ARM_CONTRACT.json", {
        "status": "SFM_HP100_EXHAUSTIVE_HYBRID_TWO_ARM_DECLARED",
        "version": VERSION, "source": source,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "arms": [asdict(arm) for arm in arms],
        "gpu_identities": gpu_identities,
    })

    preflight_rows, signatures = [], []
    for arm in arms:
        arm_root = output / "arms" / arm.name
        result = _run_logged(
            command(args, arm, mode="preflight"),
            env=_gpu_environment(arm.physical_gpu),
            log=arm_root / "preflight.log",
        )
        if result["returncode"]:
            raise RuntimeError(json.dumps(result))
        preflight_path = arm_root / "run.preflight.json"
        preflight = json.loads(preflight_path.read_text())
        if preflight.get("status") != HYBRID.PREFLIGHT_STATUS:
            raise RuntimeError(f"invalid preflight status for {arm.name}")
        signatures.append(shared_signature(preflight))
        preflight_rows.append({
            **result, "arm": arm.name,
            "preflight": str(preflight_path),
            "preflight_sha256": HYBRID._sha256(preflight_path),
        })
    if signatures[0] != signatures[1]:
        raise RuntimeError("scope arms differ outside their optimizer scope/GPU")
    _write_json(output / "PREFLIGHT_COMPLETE.json", {
        "status": "SFM_HP100_EXHAUSTIVE_HYBRID_TWO_ARM_PREFLIGHT_COMPLETE",
        "shared_signature": signatures[0], "arms": preflight_rows,
    })

    def launch(arm: Arm) -> dict:
        arm_root = output / "arms" / arm.name
        result = _run_logged(
            command(args, arm, mode="run"),
            env=_gpu_environment(arm.physical_gpu),
            log=arm_root / "run.log",
        )
        if result["returncode"]:
            raise RuntimeError(json.dumps(result))
        run_root = arm_root / "run"
        candidates = [
            run_root / "COMMIT_COMPLETE.json",
            run_root / "QUALIFICATION_INCOMPLETE.json",
        ]
        markers = [path for path in candidates if path.is_file()]
        if len(markers) != 1:
            raise RuntimeError(f"{arm.name} has no unique terminal marker")
        marker = json.loads(markers[0].read_text())
        expected_status = (
            HYBRID.COMPLETE_STATUS
            if markers[0].name == "COMMIT_COMPLETE.json"
            else HYBRID.INCOMPLETE_STATUS
        )
        if marker.get("status") != expected_status:
            raise RuntimeError(
                f"{arm.name} terminal marker/status mismatch: "
                f"{markers[0].name} contains {marker.get('status')!r}"
            )
        result.update(
            arm=arm.name, physical_gpu=arm.physical_gpu,
            terminal_marker=str(markers[0]),
            terminal_status=marker["status"],
            terminal_marker_sha256=HYBRID._sha256(markers[0]),
        )
        return result

    results = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(launch, arm): arm for arm in arms}
        for future in as_completed(futures):
            results.append(future.result())
    final_source = source_gate(
        root, args.expected_source_commit, args.required_source_ref,
    )
    all_arms_committed = all(
        row["terminal_status"] == HYBRID.COMPLETE_STATUS for row in results
    )
    marker = {
        "status": "SFM_HP100_EXHAUSTIVE_HYBRID_TWO_ARM_COMPLETE",
        "version": VERSION, "source_after": final_source,
        "all_arms_committed": bool(all_arms_committed),
        "scientific_outcome": (
            "both_scopes_passed_CLEAR_commit_gate"
            if all_arms_committed else "at_least_one_scope_incomplete"
        ),
        "arms": sorted(results, key=lambda row: row["arm"]),
    }
    _write_json(output / "TWO_ARM_COMPLETE.json", marker)
    print(json.dumps({"status": marker["status"], "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
