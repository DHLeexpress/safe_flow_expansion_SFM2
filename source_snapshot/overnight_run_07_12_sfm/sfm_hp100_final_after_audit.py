"""Audit real HP100 checkpoints on fixed disaster lineages for paper videos.

This utility never trains or edits a checkpoint.  It replays each candidate
with the exhaustive-hybrid raw-only gate (temperature one, one proposal per
context, exact full-H10 verification) and emits renderable traces only for
lineages that reach the goal without a red window.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import grid_policy_sfm_hp100 as GPS
import sfm_hp100_ball_adapter as PORT
import sfm_hp100_exhaustive_hybrid as HYBRID


STATUS = "SFM_HP100_FINAL_AFTER_AUDIT_COMPLETE"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_replicas(value: str) -> tuple[int, ...]:
    replicas = tuple(int(item) for item in value.split(",") if item.strip())
    if not replicas or len(set(replicas)) != len(replicas):
        raise ValueError("replicas must be a nonempty unique list")
    return replicas


def _render_case(result: dict, key: HYBRID.LineageKey) -> dict:
    outcome = result["outcomes"][key.label]
    events = [
        row for row in result["events"]
        if str(row["lineage"]) == key.label
    ]
    traces = []
    for row in events:
        positive = bool(row["valid"])
        traces.append({
            "step": int(row["step"]),
            "state": np.asarray(row["state_before"], np.float32),
            "next_state": np.asarray(row["state_after"], np.float32),
            "gamma": float(row["gamma"]),
            "ped_xy": np.asarray(row["ped_xy"], np.float32),
            "ped_vel": np.asarray(row["ped_vel"], np.float32),
            "proposal_controls": row["raw_controls"],
            "proposal_result": row["raw_sidecar"],
            "proposal_label": (
                "full_h_positive" if positive else "full_h_negative"
            ),
        })
    positives = sum(
        row["proposal_label"] == "full_h_positive" for row in traces
    )
    scenario_ids = {int(row["scenario_id"]) for row in events}
    if len(scenario_ids) != 1:
        raise RuntimeError(f"{key.label} does not have one scenario id")
    return {
        "scene_profile": "double_density_velocity_ood",
        "episode": scenario_ids.pop(),
        "rollout_index": int(key.replica),
        "gamma": float(key.gamma),
        "outcome": str(outcome["status"]),
        "traces": traces,
        "terminal_snapshot": None,
        "proposal_full_h_positive": int(positives),
        "proposal_full_h_negative": int(len(traces) - positives),
        "proposal_positive_fraction": (
            float(positives / len(traces)) if traces else None
        ),
        "trajectory_validity": (
            float(positives / len(traces)) if traces else 0.0
        ),
        "metrics_row": None,
        "audit_semantics": (
            "same scenario and deterministic sampling lineage as the before "
            "trace; raw temperature-one singleton at every context; no K/B, "
            "uncertainty tilting, repair, or controller wrapper"
        ),
    }


def audit(
    checkpoints: list[str], output: str | Path, *, device: str,
    gamma: float, replicas: tuple[int, ...], seed: int,
    scenario_start: int, max_steps: int, workers: int,
) -> dict:
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    keys = tuple(HYBRID.LineageKey(float(gamma), replica) for replica in replicas)
    config = HYBRID.HybridConfig(
        gammas=(float(gamma),), lineages_per_gamma=max(replicas) + 1,
        max_steps=int(max_steps), max_microcycles=1, seed=int(seed),
    )
    rows = []
    results = []
    for checkpoint_path in checkpoints:
        policy, _payload = GPS.load_sfm_hp100_policy(checkpoint_path, device=device)
        adapter = PORT.HP100ExpansionPolicy(policy).eval()
        task = PORT.SFMHP100ExpansionTask(
            scene_profile="double_density_velocity_ood",
            scenario_start=int(scenario_start),
        ).attach_context_encoder(policy)
        with HYBRID._OrderedSidecarVerifier(task, int(workers)) as verifier:
            result = HYBRID.raw_only_recheck(
                adapter, task, keys=keys, config=config,
                microcycle=0, verifier=verifier,
            )
        clear = tuple(sorted(result["clear_lineages"]))
        rows.append({
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "clear_count": len(clear),
            "requested_count": len(keys),
            "clear_lineages": clear,
            "outcomes": result["outcomes"],
        })
        results.append(result)

    order = sorted(
        range(len(rows)),
        key=lambda index: (
            -int(rows[index]["clear_count"]),
            str(rows[index]["checkpoint"]),
        ),
    )
    winner_index = order[0]
    winner = rows[winner_index]
    winner_result = results[winner_index]
    cases = {
        key.label: _render_case(winner_result, key)
        for key in keys if key.label in winner_result["clear_lineages"]
    }
    bundle = {
        "status": STATUS,
        "kind": "fixed_disaster_raw_only_checkpoint_audit",
        "gamma": float(gamma),
        "replicas": list(map(int, replicas)),
        "seed": int(seed),
        "scenario_start": int(scenario_start),
        "max_steps": int(max_steps),
        "selection_rule": "most fixed lineages cleared; path tie-break only",
        "winner": winner,
        "all_requested_lineages_solved": (
            int(winner["clear_count"]) == len(keys)
        ),
        "rows": rows,
        "cases": cases,
    }
    trace_path = output / "after_audit_trace.pt"
    torch.save(bundle, trace_path)
    marker = {
        key: value for key, value in bundle.items()
        if key not in {"rows", "cases"}
    }
    marker["rows"] = rows
    marker["trace"] = str(trace_path)
    marker["trace_sha256"] = _sha256(trace_path)
    (output / "AFTER_AUDIT_COMPLETE.json").write_text(
        json.dumps(marker, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return marker


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--checkpoint", nargs="+", required=True)
    value.add_argument("--output", required=True)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--gamma", type=float, default=0.2)
    value.add_argument("--replicas", default="2,12,4")
    value.add_argument("--seed", type=int, default=17)
    value.add_argument("--scenario-start", type=int, default=620000)
    value.add_argument("--max-steps", type=int, default=180)
    value.add_argument("--workers", type=int, default=3)
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    result = audit(
        args.checkpoint, args.output, device=args.device,
        gamma=args.gamma, replicas=_parse_replicas(args.replicas),
        seed=args.seed, scenario_start=args.scenario_start,
        max_steps=args.max_steps, workers=args.workers,
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
