"""Declared first-pass grid over the v2 MPC execution-rule parameters.

Runs the v2 no-update diagnostic (``sfm_hp100_predictive_execution_v2``) once
per declared parameter combo on the NVP-heavy mid gammas, then aggregates
per-combo and per-gamma controller outcomes.  These are acquisition-controller
statistics on training-side scenarios -- they are NEVER raw evaluation.

Grid (first pass): lam x r_eff with rho and sigma_len fixed; the declared
second pass refines rho/sigma_len around the first-pass winner.  Ranking
follows the user's criterion: among combos with the fewest collisions and the
largest NVP reduction, prefer the lowest mean success time-to-goal.

Scenario contract: defaults reproduce the archive_r1 lineages exactly
(``--scenario-start 860000 --seed 41 --lineages-per-gamma 8``); gammas default
to the measured difficulty peak 0.2,0.3,0.4.  Combos run sequentially on one
GPU; existing combo directories with a completion marker are skipped, so a
crashed sweep resumes by re-invocation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time

from sfm_hp100_expansion_status import Heartbeat
import sfm_hp100_predictive_execution_v2 as V2


VERSION = "sfm_hp100_mpc_rule_search_v1"
STATUS = "SFM_HP100_MPC_RULE_SEARCH_COMPLETE"
COMBO_MARKER = "PREDICTIVE_EXECUTION_V2_COMPLETE.json"
ROBOT_DT = 0.1

# First-pass declared grid: 3 x 3 over the cost weight and the activation
# radius; decay base and softness are fixed and refined in a declared second
# pass around the first-pass winner.
GRID_LAM = (0.5, 1.0, 2.0)
GRID_R_EFF = (0.20, 0.30, 0.45)
GRID_RHO = 1.1
GRID_SIGMA_LEN = 0.10


def declared_grid() -> tuple[V2.MPCRuleParams, ...]:
    return tuple(
        V2.MPCRuleParams(
            lam=lam, rho=GRID_RHO, r_eff=r_eff, sigma_len=GRID_SIGMA_LEN,
        )
        for lam in GRID_LAM
        for r_eff in GRID_R_EFF
    )


def combo_id(params: V2.MPCRuleParams) -> str:
    def token(value: float) -> str:
        return f"{value:g}".replace(".", "p").replace("-", "m")

    return (
        f"lam{token(params.lam)}_reff{token(params.r_eff)}"
        f"_rho{token(params.rho)}_sig{token(params.sigma_len)}"
    )


def combo_statistics(marker: dict) -> dict:
    """Per-combo and per-gamma controller outcomes from one v2 marker."""
    outcomes = marker["outcomes"]
    per_gamma: dict[str, dict] = {}
    for row in outcomes.values():
        cell = per_gamma.setdefault(f"{float(row['gamma']):g}", {
            "success": 0, "collision": 0, "nvp": 0, "timeout": 0, "oob": 0,
            "success_steps": [],
        })
        status = str(row["status"])
        if status not in cell:
            raise ValueError(f"unsupported terminal status {status!r}")
        cell[status] += 1
        if status == "success":
            cell["success_steps"].append(int(row["executed_steps"]))

    def finish(cell: dict) -> dict:
        steps = cell.pop("success_steps")
        return {
            **cell,
            "mean_success_ttg_seconds": (
                None if not steps
                else float(sum(steps) * ROBOT_DT / len(steps))
            ),
        }

    pooled = {
        "success": 0, "collision": 0, "nvp": 0, "timeout": 0, "oob": 0,
        "success_steps": [],
    }
    for cell in per_gamma.values():
        for name in ("success", "collision", "nvp", "timeout", "oob"):
            pooled[name] += cell[name]
        pooled["success_steps"].extend(cell["success_steps"])
    summary = marker["summary"]
    return {
        "pooled": finish(pooled),
        "per_gamma": {
            gamma: finish(cell) for gamma, cell in sorted(per_gamma.items())
        },
        "mean_executed_step_clearance": (
            summary["pooled_final_attempt"]["predictive_clearance"]
        ),
        "selector_shadow": marker["selector_shadow"],
    }


def rank_combos(rows: list[dict]) -> list[str]:
    """Fewest collisions, then fewest NVP, then lowest success TtG, then id."""
    def key(row: dict):
        pooled = row["statistics"]["pooled"]
        ttg = pooled["mean_success_ttg_seconds"]
        return (
            int(pooled["collision"]) + int(pooled["oob"]),
            int(pooled["nvp"]),
            float("inf") if ttg is None else float(ttg),
            str(row["combo_id"]),
        )

    return [row["combo_id"] for row in sorted(rows, key=key)]


def run(args) -> dict:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    heartbeat = Heartbeat(
        args.status_json, interval_seconds=float(args.heartbeat_seconds),
    )
    grid = declared_grid()
    rows = []
    for index, params in enumerate(grid):
        identity = combo_id(params)
        combo_output = output / identity
        marker_path = combo_output / COMBO_MARKER
        heartbeat.beat(
            status="running", phase="grid", combo=identity,
            completed=index, total=len(grid),
        )
        if marker_path.is_file():
            marker = json.loads(marker_path.read_text())
        else:
            if combo_output.exists():
                raise FileExistsError(
                    f"combo directory exists without a marker: {combo_output}"
                )
            command = [
                sys.executable, "-u",
                str(Path(V2.__file__).resolve()),
                "--checkpoint", str(args.checkpoint),
                "--expected-checkpoint-sha256",
                str(args.expected_checkpoint_sha256),
                "--pretrain-dataset-root", str(args.pretrain_dataset_root),
                "--expected-pretrain-dataset-manifest-sha256",
                str(args.expected_pretrain_dataset_manifest_sha256),
                "--output", str(combo_output),
                "--device", str(args.device),
                "--physical-gpu", str(int(args.physical_gpu)),
                "--scene-profile", str(args.scene_profile),
                "--scenario-start", str(int(args.scenario_start)),
                "--gammas", str(args.gammas),
                "--lineages-per-gamma", str(int(args.lineages_per_gamma)),
                "--max-steps", str(int(args.max_steps)),
                "--max-attempts", str(int(args.max_attempts)),
                "--ess-target", str(float(args.ess_target)),
                "--seed", str(int(args.seed)),
                "--verifier-workers", str(int(args.verifier_workers)),
                "--mpc-lam", str(params.lam),
                "--mpc-rho", str(params.rho),
                "--mpc-r-eff", str(params.r_eff),
                "--mpc-sigma-len", str(params.sigma_len),
            ]
            log_path = output / f"{identity}.log"
            started = time.time()
            with log_path.open("w") as stream:
                stream.write("COMMAND " + json.dumps(command) + "\n")
                stream.flush()
                completed = subprocess.run(
                    command, stdout=stream, stderr=subprocess.STDOUT,
                )
            if completed.returncode != 0 or not marker_path.is_file():
                raise RuntimeError(
                    f"combo {identity} failed rc={completed.returncode}; "
                    f"log: {log_path}"
                )
            marker = json.loads(marker_path.read_text())
            marker.setdefault("wall_seconds", time.time() - started)
        rows.append({
            "combo_id": identity,
            "mpc_params": dict(marker["mpc_params"]),
            "statistics": combo_statistics(marker),
            "marker": str(marker_path),
        })

    ranked = rank_combos(rows)
    summary = {
        "status": STATUS,
        "version": VERSION,
        "execution_rule": V2.EXECUTION_RULE,
        "criterion": (
            "fewest collisions, then fewest NVP, then lowest mean success "
            "time-to-goal; acquisition-controller statistics only, never "
            "raw evaluation"
        ),
        "declared_grid": [asdict(params) for params in grid],
        "scenario_contract": {
            "scene_profile": str(args.scene_profile),
            "scenario_start": int(args.scenario_start),
            "gammas": str(args.gammas),
            "lineages_per_gamma": int(args.lineages_per_gamma),
            "seed": int(args.seed),
        },
        "rows": rows,
        "ranked": ranked,
        "recommended": ranked[0] if ranked else None,
    }
    V2.PRED._write_json(output / "GRID_SUMMARY.json", summary)
    heartbeat.beat(
        status="complete", phase="done", completed=len(grid), total=len(grid),
    )
    return summary


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--checkpoint", required=True)
    value.add_argument("--expected-checkpoint-sha256", required=True)
    value.add_argument("--pretrain-dataset-root", required=True)
    value.add_argument("--expected-pretrain-dataset-manifest-sha256", required=True)
    value.add_argument("--output", required=True)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--physical-gpu", type=int, default=3)
    value.add_argument("--scene-profile", default="double_density_velocity_ood")
    value.add_argument("--scenario-start", type=int, default=860_000)
    value.add_argument("--gammas", default="0.2,0.3,0.4")
    value.add_argument("--lineages-per-gamma", type=int, default=8)
    value.add_argument("--max-steps", type=int, default=180)
    value.add_argument("--max-attempts", type=int, default=32)
    value.add_argument("--ess-target", type=float, default=0.1)
    value.add_argument("--seed", type=int, default=41)
    value.add_argument("--verifier-workers", type=int, default=32)
    value.add_argument("--status-json", default=None)
    value.add_argument("--heartbeat-seconds", type=float, default=30.0)
    return value


def main(argv=None) -> int:
    summary = run(parser().parse_args(argv))
    print(json.dumps({
        "status": summary["status"],
        "recommended": summary["recommended"],
        "ranked": summary["ranked"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
