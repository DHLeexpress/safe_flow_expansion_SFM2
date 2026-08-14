"""Secondary disjoint confirmation of the combined P1+P2+Ncausal policy.

The primary disaster-prefix sweep asks whether Ncausal improves over its
matched alpha-zero P1+P2 anchor.  This additive analysis preserves that null
result and asks the distinct user-requested question: does the complete
P1+P2+Ncausal treatment improve over canonical r0?  Training is never rerun.
Exactly one M10-selected treatment, its matched anchor, and r0 are evaluated
on the already-declared untouched M50 bank.  No runner-up substitution is
allowed after that bank is read.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import sfm_hp100_disaster_prefix_sweep as SWEEP


VERSION = "sfm_hp100_combined_r0_confirmation_v1"
CONTRACT_STATUS = "SFM_HP100_SECONDARY_COMBINED_R0_CONTRACT_DECLARED"
COMPLETE_STATUS = "SFM_HP100_SECONDARY_COMBINED_R0_CONFIRMATION_COMPLETE"
NO_M10_STATUS = "SFM_HP100_SECONDARY_COMBINED_R0_NO_ELIGIBLE_M10"


def _count(metric: dict, field: str) -> int:
    value = float(metric[field]) * int(metric["n"])
    rounded = int(round(value))
    if abs(value - rounded) > 1.0e-8:
        raise ValueError(f"{field} does not map to an integer outcome count")
    return rounded


def _authenticate_cell(cell: dict) -> dict:
    if cell.get("run", {}).get("rejected"):
        raise ValueError("secondary analysis cannot use a rejected update")
    dose = SWEEP.Dose(**cell["dose"])
    if dose not in SWEEP.declared_stage1_doses():
        raise ValueError("cell dose is outside the frozen Stage-1 table")
    run = cell["run"]
    marker_path = Path(run["marker"]).resolve()
    if SWEEP.sha256_file(marker_path) != run["marker_sha256"]:
        raise RuntimeError("continuation marker SHA256 mismatch")
    marker = json.loads(marker_path.read_text())
    if marker.get("status") != "SFM_HP100_DISASTER_PREFIX_DIAGNOSTIC_COMPLETE":
        raise ValueError("continuation cell is not complete")
    if marker["update"].get("accepted") is not True:
        raise ValueError("continuation atomic update was not accepted")
    negative = marker["update"]["negative"]
    if negative["negative_source"] != SWEEP.NEGATIVE_SOURCE:
        raise ValueError("cell is not Ncausal-only")
    selected = negative["selected_source_counts"]
    if int(selected["Dminus"]) != 0 or int(selected["Ncausal"]) <= 0:
        raise ValueError("cell has invalid Ncausal/Dminus exposure sources")
    expected_exposures = int(selected["Ncausal"]) * int(dose.negative_passes)
    if int(negative["total_row_exposures"]) != expected_exposures:
        raise ValueError("Ncausal exact-pass exposure accounting drifted")
    for phase, passes in (("P1", dose.p1_passes), ("P2", dose.p2_passes)):
        report = marker["update"][phase]
        if int(report["passes"]) != int(passes):
            raise ValueError(f"{phase} pass count drifted")
        if int(report["total_row_exposures"]) != int(report["rows"]) * int(passes):
            raise ValueError(f"{phase} exact-pass exposure accounting drifted")
    for role in ("anchor_checkpoint", "treatment_checkpoint"):
        record = cell[role]
        if SWEEP.sha256_file(record["path"]) != record["sha256"]:
            raise RuntimeError(f"{role} SHA256 mismatch")
    if (
        cell["treatment_checkpoint"]["policy_state_sha256"]
        != marker["update"]["treatment_model_sha256_after_gate"]
    ):
        raise RuntimeError("treatment policy-state SHA256 mismatch")
    return {"dose": dose, "marker": marker}


def combined_cell(cell: dict, dose_index: int) -> dict:
    authenticated = _authenticate_cell(cell)
    r0 = cell["r0_raw_M10"]
    treatment = cell["treatment_raw_M10"]
    r0_m64 = cell["r0_M64"]
    treatment_m64 = cell["treatment_M64"]
    targeted = cell["targeted_recovery"]
    gates = {
        "M10_success_count_at_least_r0_plus_1": (
            _count(treatment, "SR") >= _count(r0, "SR") + 1
        ),
        "M10_raw_Validity_strictly_above_r0": (
            float(treatment["Validity"]) > float(r0["Validity"])
        ),
        "M64_exact_positive_at_least_r0_plus_1": (
            int(treatment_m64["exact_positive"])
            >= int(r0_m64["exact_positive"]) + 1
        ),
        "M10_timeout_count_nonincrease": (
            _count(treatment, "timeout") <= _count(r0, "timeout")
        ),
        "targeted_recovery_nonregression_vs_r0": (
            int(targeted["treatment"]) >= int(targeted["r0"])
        ),
        "M64_mean_H10_progress_drop_at_most_0p02_vs_r0": (
            float(treatment_m64["mean_H10_progress"])
            >= float(r0_m64["mean_H10_progress"]) - 0.02
        ),
    }
    deltas = {
        "M10_SR": float(treatment["SR"] - r0["SR"]),
        "M10_raw_Validity": float(treatment["Validity"] - r0["Validity"]),
        "M64_Validity": float(treatment_m64["Validity"] - r0_m64["Validity"]),
        "M64_mean_H10_progress": float(
            treatment_m64["mean_H10_progress"] - r0_m64["mean_H10_progress"]
        ),
    }
    return {
        "scope": cell["scope"], "dose": asdict(authenticated["dose"]),
        "dose_index": int(dose_index), "eligible": bool(all(gates.values())),
        "gates": gates, "deltas_vs_r0": deltas,
        "r0_raw_M10": r0, "treatment_raw_M10": treatment,
        "matched_anchor_raw_M10": cell["anchor_raw_M10"],
        "r0_M64": r0_m64, "treatment_M64": treatment_m64,
        "matched_anchor_M64": cell["anchor_M64"],
        "targeted_recovery": targeted,
        "relative_parameter_drift": float(cell["relative_parameter_drift"]),
        "treatment_checkpoint": cell["treatment_checkpoint"],
        "matched_alpha0_checkpoint": cell["anchor_checkpoint"],
        "continuation_marker": cell["run"]["marker"],
        "continuation_marker_sha256": cell["run"]["marker_sha256"],
        "primary_incremental_gates": cell["gates"],
        "primary_incremental_eligible": bool(cell["eligible"]),
    }


def selection_key(row: dict) -> tuple:
    delta = row["deltas_vs_r0"]
    return (
        min(delta["M10_SR"], delta["M64_Validity"]),
        delta["M10_SR"], delta["M64_Validity"],
        delta["M10_raw_Validity"],
        int(row["targeted_recovery"]["treatment"]),
        float(row["treatment_M64"]["mean_H10_progress"]),
        -float(row["relative_parameter_drift"]), -int(row["dose_index"]),
    )


def m50_gates(treatment: dict, r0: dict) -> dict:
    return {
        "M50_success_count_at_least_r0_plus_1": (
            _count(treatment, "SR") >= _count(r0, "SR") + 1
        ),
        "M50_collision_count_at_most_r0_minus_1": (
            _count(treatment, "CR") <= _count(r0, "CR") - 1
        ),
        "M50_timeout_count_nonincrease": (
            _count(treatment, "timeout") <= _count(r0, "timeout")
        ),
        "M50_raw_Validity_strictly_above_r0": (
            float(treatment["Validity"]) > float(r0["Validity"])
        ),
    }


def _verify_json(path: Path, expected_sha256: str, label: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = SWEEP.sha256_file(path)
    if actual != str(expected_sha256).lower():
        raise RuntimeError(f"{label} SHA256 mismatch")
    return {"path": str(path), "sha256": actual}


def run(args) -> dict:
    repo = Path(args.repo_root).resolve()
    output = Path(args.output).resolve()
    primary = Path(args.primary_root).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing secondary output: {output}")
    source = SWEEP.source_gate(repo, args.expected_source_commit)
    contract_record = _verify_json(
        primary / "SWEEP_CONTRACT.json", args.expected_primary_contract_sha256,
        "primary sweep contract",
    )
    stage1_record = _verify_json(
        primary / "STAGE1_COMPLETE.json", args.expected_primary_stage1_sha256,
        "primary Stage-1 marker",
    )
    sweep_record = _verify_json(
        primary / "SWEEP_COMPLETE.json", args.expected_primary_sweep_sha256,
        "primary sweep marker",
    )
    stage1 = json.loads(Path(stage1_record["path"]).read_text())
    primary_sweep = json.loads(Path(sweep_record["path"]).read_text())
    primary_contract = json.loads(Path(contract_record["path"]).read_text())
    if stage1.get("status") != "SFM_HP100_DISASTER_PREFIX_STAGE1_NO_ELIGIBLE_DOSE":
        raise ValueError("secondary analysis requires the preserved primary null")
    if primary_sweep.get("status") != "SFM_HP100_DISASTER_PREFIX_SWEEP_NO_ELIGIBLE_DOSE":
        raise ValueError("primary sweep null status drifted")
    confirmation = primary_contract["confirmation"]
    expected_bank = {
        "ep0": SWEEP.CONFIRM_EP0, "M_per_gamma": SWEEP.CONFIRM_M,
        "noise_seed": SWEEP.CONFIRM_NOISE_SEED, "temperature": 1.0,
    }
    for key, value in expected_bank.items():
        if confirmation[key] != value:
            raise ValueError(f"frozen M50 bank drifted at {key}")
    if (primary / "CONFIRMATION_COMPLETE.json").exists():
        raise ValueError("primary confirmation unexpectedly exists")
    if list(primary.rglob("*m50*")):
        raise ValueError("primary M50 output is not untouched")
    m10_artifacts = []
    for path in sorted(primary.glob("evaluation/*/*_m10.json")):
        payload = json.loads(path.read_text())
        if (
            payload.get("status") != "SFM_HP100_RAW_EVAL_COMPLETE"
            or int(payload["ep0"]) != SWEEP.SCREEN_EP0
            or int(payload["M_per_gamma"]) != SWEEP.SCREEN_M
            or int(payload["noise_seed"]) != SWEEP.SCREEN_NOISE_SEED
            or float(payload["temperature"]) != 1.0
        ):
            raise ValueError(f"primary M10 evaluator contract drifted: {path}")
        m10_artifacts.append({
            "path": str(path.resolve()), "sha256": SWEEP.sha256_file(path),
        })
    if not m10_artifacts:
        raise ValueError("primary sweep has no authenticated M10 artifacts")
    r0 = SWEEP._verify_input(
        args.r0_checkpoint, args.expected_r0_checkpoint_sha256, "r0 checkpoint",
    )
    if r0["sha256"] != primary_contract["checkpoint"]["sha256"]:
        raise ValueError("secondary r0 differs from the primary contract")

    declared = {dose.name: index for index, dose in enumerate(
        SWEEP.declared_stage1_doses()
    )}
    rows = []
    rejected_cells = []
    for cell in stage1["cells"]:
        name = cell["dose"]["name"]
        if name not in declared:
            raise ValueError("Stage-1 cell has undeclared dose name")
        if cell.get("run", {}).get("rejected"):
            rejected_cells.append({
                "scope": cell.get("scope"), "dose": cell["dose"],
                "reason": cell.get("ineligible_reason", "atomic update rejected"),
            })
            continue
        rows.append(combined_cell(cell, declared[name]))
    eligible = [row for row in rows if row["eligible"]]
    selected = max(eligible, key=selection_key) if eligible else None
    output.mkdir(parents=True)
    source_files = {
        "secondary_driver": {
            "path": str(Path(__file__).resolve()),
            "sha256": SWEEP.sha256_file(__file__),
        },
        "raw_evaluator": {
            "path": str(
                repo / "source_snapshot/overnight_run_07_12_sfm/sfm_hp100_eval.py"
            ),
            "sha256": SWEEP.sha256_file(
                repo / "source_snapshot/overnight_run_07_12_sfm/sfm_hp100_eval.py"
            ),
        },
    }
    contract = {
        "status": CONTRACT_STATUS, "version": VERSION, "source": source,
        "primary_incremental_null_preserved": True,
        "analysis_timing": (
            "post-hoc secondary M10 screen defined after observing the primary "
            "null; the frozen disjoint M50 bank remained untouched"
        ),
        "primary": {
            "contract": contract_record, "stage1": stage1_record,
            "sweep": sweep_record, "M10_artifacts": m10_artifacts,
        },
        "source_files": source_files, "r0_checkpoint": r0,
        "M10_rule": (
            "combined treatment vs r0; success +1, raw Validity strict, "
            "M64 exact-positive +1, timeout nonincrease, recovery nonregression, "
            "mean H10 progress drop <=0.02"
        ),
        "selection_key": (
            "min(delta SR,delta M64), delta SR, delta M64, delta raw Validity, "
            "recovery, H10 progress, lower drift, earlier declared dose"
        ),
        "M50_bank": expected_bank,
        "M50_cells": ["r0", "selected_treatment", "matched_alpha0"],
        "no_runner_up_substitution": True,
        "M10_rows": rows, "rejected_primary_cells": rejected_cells,
        "selected_M10": selected,
    }
    SWEEP._write_json(output / "SECONDARY_CONTRACT.json", contract)
    if selected is None:
        marker = {
            "status": NO_M10_STATUS, "eligible_for_iterative_microcycles": False,
            "contract": str(output / "SECONDARY_CONTRACT.json"),
            "primary_incremental_null_preserved": True,
        }
        SWEEP._write_json(output / "SECONDARY_COMPLETE.json", marker)
        return marker

    evaluator = Path(source_files["raw_evaluator"]["path"])
    gpus = tuple(int(item) for item in args.gpus.split(",") if item.strip())
    if not gpus:
        raise ValueError("at least one GPU is required")
    jobs = [
        ("r0", r0),
        ("selected_treatment", selected["treatment_checkpoint"]),
        ("matched_alpha0", selected["matched_alpha0_checkpoint"]),
    ]

    def evaluate(job, gpu: int) -> dict:
        label, checkpoint = job
        destination = output / "m50" / f"{label}.json"
        command = [
            sys.executable, "-u", str(evaluator), "--checkpoint", checkpoint["path"],
            "--scene-profile", SWEEP.SCENE_PROFILE,
            "--ep0", str(SWEEP.CONFIRM_EP0), "--M", str(SWEEP.CONFIRM_M),
            "--device", "cuda:0", "--noise-seed", str(SWEEP.CONFIRM_NOISE_SEED),
            "--verifier-workers", str(args.eval_workers), "--out", str(destination),
        ]
        execution = SWEEP._run_logged(
            command, environment=SWEEP._gpu_environment(gpu),
            log=destination.with_suffix(".log"),
        )
        if execution["returncode"] != 0 or not destination.is_file():
            raise RuntimeError(json.dumps(execution))
        return {
            "label": label, "checkpoint": checkpoint,
            "path": str(destination), "sha256": SWEEP.sha256_file(destination),
            "metrics": SWEEP._pooled(destination), "execution": execution,
            "physical_gpu": int(gpu),
        }

    evaluated = SWEEP._parallel_gpu_jobs(
        jobs, gpus=gpus, jobs_per_gpu=1, worker=evaluate,
    )
    by_label = {row["label"]: row for row in evaluated}
    treatment = by_label["selected_treatment"]["metrics"]
    r0_metrics = by_label["r0"]["metrics"]
    gates = m50_gates(treatment, r0_metrics)
    eligible_iterative = bool(all(gates.values()) and selected["eligible"])
    marker = {
        "status": COMPLETE_STATUS, "version": VERSION,
        "contract": str(output / "SECONDARY_CONTRACT.json"),
        "contract_sha256": SWEEP.sha256_file(output / "SECONDARY_CONTRACT.json"),
        "primary_incremental_null_preserved": True,
        "primary_incremental_claim": False,
        "selected_scope": selected["scope"], "selected_dose": selected["dose"],
        "selected_treatment_checkpoint": selected["treatment_checkpoint"],
        "matched_alpha0_checkpoint": selected["matched_alpha0_checkpoint"],
        "selected_M10": selected, "M50_bank": expected_bank,
        "M50_cells": evaluated, "M50_gates_vs_r0": gates,
        "combined_M50_strict_r0_win": bool(all(gates.values())),
        "eligible_for_iterative_microcycles": eligible_iterative,
        "no_runner_up_substitution": True,
    }
    SWEEP._write_json(output / "SECONDARY_COMPLETE.json", marker)
    return marker


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--repo-root", required=True)
    value.add_argument("--expected-source-commit", required=True)
    value.add_argument("--primary-root", required=True)
    value.add_argument("--expected-primary-contract-sha256", required=True)
    value.add_argument("--expected-primary-stage1-sha256", required=True)
    value.add_argument("--expected-primary-sweep-sha256", required=True)
    value.add_argument("--r0-checkpoint", required=True)
    value.add_argument("--expected-r0-checkpoint-sha256", required=True)
    value.add_argument("--output", required=True)
    value.add_argument("--gpus", default="1,3")
    value.add_argument("--eval-workers", type=int, default=8)
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.eval_workers < 1:
        raise ValueError("eval workers must be positive")
    marker = run(args)
    print(json.dumps({
        "status": marker["status"],
        "eligible_for_iterative_microcycles": marker.get(
            "eligible_for_iterative_microcycles", False
        ),
        "output": str(Path(args.output).resolve()),
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
