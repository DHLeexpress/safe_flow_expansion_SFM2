"""Gamma-balanced allocation planner for the extended raw-obs collection.

The archive CLI accepts a single ``--lineages-per-gamma`` for all its gammas,
so exact per-gamma balance requires splitting gammas across jobs and giving
each job the episode count its neediest gamma requires.  This planner groups
gammas whose episode needs are similar (sorted, chunked), which keeps the
overshoot small and the per-job D+ workload (the wall-clock driver, since
collection is verifier-throughput-bound) roughly even.  It only prints the
launch commands — it never launches anything.

Measured D+/episode yields come from the first mega collection
(1,008 episodes, jobs 1-3, 2026-08-16).
"""
from __future__ import annotations

import argparse
import json
import math

MEASURED_DPLUS_PER_EPISODE = {
    "0.1": 74.0, "0.2": 54.1, "0.3": 53.5, "0.4": 56.1,
    "0.5": 57.1, "0.7": 58.8, "1.0": 62.1,
}
DEFAULT_TARGET = 100_000
DEFAULT_JOBS = 4
SCENARIO_START_BASE = 1_400_000
SCENARIO_START_STEP = 100_000
SEED_BASE = 61


def plan(target=DEFAULT_TARGET, jobs=DEFAULT_JOBS,
         yields=MEASURED_DPLUS_PER_EPISODE):
    per_gamma_target = target / len(yields)
    needs = {
        gamma: math.ceil(per_gamma_target / rate)
        for gamma, rate in yields.items()
    }
    ordered = sorted(needs, key=lambda gamma: needs[gamma])
    chunk = math.ceil(len(ordered) / int(jobs))
    groups = [ordered[i:i + chunk] for i in range(0, len(ordered), chunk)]
    rows = []
    for index, group in enumerate(groups):
        lineages = max(needs[gamma] for gamma in group)
        rows.append({
            "job": index + 1,
            "gammas": sorted(group, key=float),
            "lineages_per_gamma": lineages,
            "episodes": lineages * len(group),
            "expected_dplus": round(sum(lineages * yields[g] for g in group)),
            "scenario_start": SCENARIO_START_BASE + index * SCENARIO_START_STEP,
            "seed": SEED_BASE + index,
        })
    return {
        "target_dplus": int(target),
        "per_gamma_target": round(per_gamma_target),
        "episode_needs": needs,
        "jobs": rows,
        "expected_total_dplus": sum(row["expected_dplus"] for row in rows),
    }


def launch_commands(allocation, *, output_root, driver, checkpoint_args):
    commands = []
    for row in allocation["jobs"]:
        gammas = ",".join(row["gammas"])
        commands.append(
            f"nohup $PY {driver} {checkpoint_args} "
            f"--output {output_root}/job{row['job']} --round-index 1 "
            f"--scenario-start {row['scenario_start']} "
            f"--gammas {gammas} --lineages-per-gamma {row['lineages_per_gamma']} "
            f"--max-steps 180 --max-attempts 32 --ess-target 0.1 "
            f"--seed {row['seed']} --device cuda:0 --physical-gpu 3 "
            f"--verifier-workers 24 "
            f"--status-json {output_root}/job{row['job']}/STATUS.json "
            f"--heartbeat-seconds 30 "
            f"> {output_root}/job{row['job']}.log 2>&1 &"
        )
    return commands


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    parser.add_argument(
        "--output-root",
        default="/data3/research1/claude_sfm2_predictive_cfc09ad/extended_collect",
    )
    args = parser.parse_args(argv)
    allocation = plan(target=args.target, jobs=args.jobs)
    print(json.dumps(allocation, indent=2, sort_keys=True))
    print()
    print("# launch commands (PY / checkpoint args as in the ops chains):")
    for command in launch_commands(
        allocation,
        output_root=args.output_root,
        driver="scripts/mpc_expansion/mpc_collect_extended.py",
        checkpoint_args="<CHECKPOINT_AND_DATASET_SHA_ARGS>",
    ):
        print(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
