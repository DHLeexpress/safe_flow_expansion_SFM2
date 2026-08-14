"""Staged raw evaluation funnel for SFM2 predictive expansion checkpoints.

The funnel never re-implements evaluation: every stage subprocess-invokes the
fixed ``sfm_hp100_eval.py`` raw temperature-one evaluator on a declared frozen
episode bank.  Stages are strictly ordered:

- ``dev-m10``: fixed disjoint development screen for every saved checkpoint;
- ``shortlist-m50``: fresh bank for the declared shortlist only;
- ``confirm-m100``: untouched bank for the single locked winner (plus r0),
  refused unless a search contract has already locked that winner.

Acquisition traces and expansion archives are training data, never funnel
inputs; the admission check refuses them fail-closed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

import _paths  # noqa: F401
import sfm_hp100_ball_adapter as PORT
import sfm_protocol as PROTO
import sfm_scene as SS

from sfm_hp100_expansion_status import Heartbeat


VERSION = "sfm_hp100_expansion_funnel_v1"
STAGE_STATUS = "SFM2_EXPANSION_FUNNEL_STAGE_COMPLETE"
EVALUATOR = Path(__file__).resolve().with_name("sfm_hp100_eval.py")

OOD_PROFILE = "double_density_velocity_ood"
ID_PROFILE = "matched_id"

# Frozen funnel banks.  Episode ranges are pairwise disjoint, disjoint from
# every historical protocol bank, and disjoint from the declared acquisition
# scenario starts below.  ``assert_declared_banks_static`` proves this at
# import-independent test time; ``assert_bank_disjoint`` in the archive runner
# additionally proves no realized acquisition scenario id hit a bank episode.
DECLARED_EVAL_BANKS = {
    "dev_m10": (
        dict(stage="dev_m10", scene_profile=OOD_PROFILE,
             ep0=900_000, M=10, noise_seed=20_260_814),
        dict(stage="dev_m10", scene_profile=ID_PROFILE,
             ep0=910_000, M=10, noise_seed=20_260_814),
    ),
    "shortlist_m50": (
        dict(stage="shortlist_m50", scene_profile=OOD_PROFILE,
             ep0=920_000, M=50, noise_seed=20_260_815),
        dict(stage="shortlist_m50", scene_profile=ID_PROFILE,
             ep0=930_000, M=50, noise_seed=20_260_815),
    ),
    "confirm_m100": (
        dict(stage="confirm_m100", scene_profile=OOD_PROFILE,
             ep0=940_000, M=100, noise_seed=20_260_816),
        dict(stage="confirm_m100", scene_profile=ID_PROFILE,
             ep0=950_000, M=100, noise_seed=20_260_816),
    ),
}

ACQUISITION_SCENARIO_START_BASE = 860_000
ACQUISITION_SCENARIO_START_STRIDE = 10_000

# Historical episode-bank starts that the new funnel banks must avoid.  Each
# start is given a conservative width far above any bank actually drawn there.
_KNOWN_EXTERNAL_EP0 = (
    PROTO.PRETRAIN_GATE_EP0, PROTO.PRETRAIN_CONFIRM_EP0, PROTO.EXPANSION_EP0,
    PROTO.SCREEN_EP0, PROTO.CONFIRM_EP0, PROTO.KAZUKI_CONFIRM_EP0,
    PROTO.SMOKE_EP0, PROTO.SMOKE_EVAL_EP0, PROTO.DEPLOY_ID_EP0,
    PROTO.DEPLOY_OOD_EP0, PROTO.QUERY_DIAGNOSTIC_EP0,
    PROTO.DEPLOY_DENSITY_OOD_EP0, PROTO.DEPLOY_DOUBLE_SHIFT_EP0,
    PROTO.TEMPERATURE_SELECT_EP0, PROTO.CURVE_SCREEN_EP0,
    PROTO.FINAL_CONFIRM_EP0, PROTO.ADAPTIVE_CONFIRM_EP0,
)
_KNOWN_EXTERNAL_WIDTH = 10_000


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def bank_range(bank: dict) -> tuple[int, int]:
    return int(bank["ep0"]), int(bank["ep0"]) + int(bank["M"])


def acquisition_scenario_start(round_index: int) -> int:
    if int(round_index) < 1:
        raise ValueError("expansion round indices start at one")
    return (
        ACQUISITION_SCENARIO_START_BASE
        + ACQUISITION_SCENARIO_START_STRIDE * (int(round_index) - 1)
    )


def assert_declared_banks_static() -> None:
    """Fail closed unless every declared funnel bank is pairwise disjoint and
    clear of all historical protocol banks and legacy deploy ranges."""
    banks = [bank for stage in DECLARED_EVAL_BANKS.values() for bank in stage]
    for bank in banks:
        if bank["scene_profile"] not in SS.SCIENTIFIC_EVAL_PROFILES:
            raise ValueError(f"unknown funnel scene profile {bank['scene_profile']!r}")
        if int(bank["M"]) < 1 or int(bank["ep0"]) < 0:
            raise ValueError("funnel bank must declare a positive episode range")
    for index, first in enumerate(banks):
        lo_a, hi_a = bank_range(first)
        for second in banks[index + 1:]:
            lo_b, hi_b = bank_range(second)
            if lo_a < hi_b and lo_b < hi_a:
                raise ValueError(
                    f"funnel banks overlap: {first['stage']} and {second['stage']}"
                )
        for start in _KNOWN_EXTERNAL_EP0:
            if lo_a < start + _KNOWN_EXTERNAL_WIDTH and start < hi_a:
                raise ValueError(
                    f"funnel bank {first['stage']} collides with historical ep0 {start}"
                )
        for lo_b, hi_b in PORT.DECLARED_EVAL_RANGES:
            if lo_a < hi_b and lo_b < hi_a:
                raise ValueError(
                    f"funnel bank {first['stage']} collides with a declared legacy range"
                )


def assert_evaluable_checkpoint(path: str | Path) -> dict:
    """Refuse anything but a strict ``{state_dict, config}`` HP100 checkpoint.

    Acquisition traces, sample archives, and expansion archives carry a
    ``status``/``samples`` payload instead; evaluating one would silently
    conflate training data with independent raw evaluation.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"funnel input is not a checkpoint payload: {path}")
    status = str(payload.get("status", ""))
    if "TRACE" in status or "ARCHIVE" in status or "samples" in payload:
        raise ValueError(
            f"funnel refuses acquisition/training data as an evaluation input: {path}"
        )
    if "state_dict" not in payload or "config" not in payload:
        raise ValueError(f"funnel input lacks the strict HP100 schema: {path}")
    return payload


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


def evaluate_checkpoint(
    *,
    label: str,
    checkpoint: Path,
    bank: dict,
    output: Path,
    device: str,
    physical_gpu: int,
    verifier_workers: int,
    python_bin: str = sys.executable,
) -> dict:
    assert_evaluable_checkpoint(checkpoint)
    destination = output / f"{label}_{bank['scene_profile']}.json"
    if destination.exists():
        raise FileExistsError(f"refusing existing funnel evaluation: {destination}")
    command = [
        python_bin, "-u", str(EVALUATOR),
        "--checkpoint", str(checkpoint),
        "--scene-profile", str(bank["scene_profile"]),
        "--ep0", str(int(bank["ep0"])),
        "--M", str(int(bank["M"])),
        "--device", device,
        "--noise-seed", str(int(bank["noise_seed"])),
        "--verifier-workers", str(int(verifier_workers)),
        "--out", str(destination),
    ]
    execution = _run_logged(
        command, environment=_gpu_environment(physical_gpu),
        log=destination.with_suffix(".log"),
    )
    if execution["returncode"] != 0 or not destination.is_file():
        raise RuntimeError(
            f"funnel evaluation failed: {json.dumps(execution, sort_keys=True)}"
        )
    payload = json.loads(destination.read_text())
    if payload.get("status") != "SFM_HP100_RAW_EVAL_COMPLETE":
        raise RuntimeError(f"funnel evaluation is incomplete: {destination}")
    if int(payload["ep0"]) != int(bank["ep0"]) or int(payload["M_per_gamma"]) != int(bank["M"]):
        raise RuntimeError("funnel evaluation drifted from its declared bank")
    return {
        "label": str(label),
        "bank": dict(bank),
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "path": str(destination),
        "sha256": sha256_file(destination),
        "pooled": payload["summary"]["pooled"],
        "per_gamma": payload["summary"]["per_gamma"],
        "execution": execution,
    }


def _parse_checkpoints(values: list[str]) -> list[tuple[str, Path]]:
    rows = []
    for value in values:
        label, _, path = value.partition("=")
        if not label or not path:
            raise ValueError(
                f"checkpoint arguments must be label=path, got {value!r}"
            )
        rows.append((label, Path(path).resolve()))
    if len({label for label, _ in rows}) != len(rows):
        raise ValueError("funnel checkpoint labels must be unique")
    return rows


def run_stage(args) -> dict:
    stage = str(args.stage).replace("-", "_")
    if stage not in DECLARED_EVAL_BANKS:
        raise ValueError(f"unknown funnel stage {args.stage!r}")
    assert_declared_banks_static()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing funnel stage output: {output}")
    checkpoints = _parse_checkpoints(args.checkpoint)

    contract = None
    if stage == "confirm_m100":
        if args.search_contract is None:
            raise ValueError("confirm-m100 requires the locked search contract")
        contract = json.loads(Path(args.search_contract).read_text())
        winner = contract.get("locked_winner")
        if not isinstance(winner, dict) or len(str(winner.get("sha256", ""))) != 64:
            raise ValueError("confirm-m100 requires a locked single winner")
        allowed = {str(winner["sha256"]), str(contract.get("r0_sha256", ""))}
        for label, checkpoint in checkpoints:
            actual = sha256_file(checkpoint)
            if actual not in allowed:
                raise ValueError(
                    f"confirm-m100 refuses unlocked checkpoint {label} ({actual})"
                )

    output.mkdir(parents=True)
    heartbeat = Heartbeat(
        args.status_json, interval_seconds=float(args.heartbeat_seconds),
    )
    results = []
    for index, (label, checkpoint) in enumerate(checkpoints):
        for bank in DECLARED_EVAL_BANKS[stage]:
            heartbeat.beat(
                status="running", phase=stage, checkpoint=label,
                bank=bank["scene_profile"], completed=len(results),
                total=len(checkpoints) * len(DECLARED_EVAL_BANKS[stage]),
            )
            results.append(evaluate_checkpoint(
                label=label, checkpoint=checkpoint, bank=bank, output=output,
                device=args.device, physical_gpu=int(args.physical_gpu),
                verifier_workers=int(args.verifier_workers),
            ))
    marker = {
        "status": STAGE_STATUS,
        "version": VERSION,
        "stage": stage,
        "banks": [dict(bank) for bank in DECLARED_EVAL_BANKS[stage]],
        "evaluator": {"path": str(EVALUATOR), "sha256": sha256_file(EVALUATOR)},
        "search_contract": (
            None if args.search_contract is None
            else {
                "path": str(Path(args.search_contract).resolve()),
                "sha256": sha256_file(args.search_contract),
            }
        ),
        "results": results,
    }
    _write_json(output / "STAGE_COMPLETE.json", marker)
    heartbeat.beat(status="complete", phase=stage, completed=len(results))
    return marker


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "stage", choices=("dev-m10", "shortlist-m50", "confirm-m100"),
    )
    value.add_argument(
        "--checkpoint", action="append", required=True,
        help="label=path; repeat per checkpoint",
    )
    value.add_argument("--output", required=True)
    value.add_argument("--device", default="cuda")
    value.add_argument("--physical-gpu", type=int, default=3)
    value.add_argument("--verifier-workers", type=int, default=32)
    value.add_argument("--search-contract", default=None)
    value.add_argument("--status-json", default=None)
    value.add_argument("--heartbeat-seconds", type=float, default=30.0)
    return value


def main(argv=None) -> int:
    marker = run_stage(parser().parse_args(argv))
    print(json.dumps(
        {"status": marker["status"], "stage": marker["stage"],
         "results": len(marker["results"])},
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
