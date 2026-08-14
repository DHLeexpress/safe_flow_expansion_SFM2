"""Select an HP100 early-acquisition lambda without using training or eval results."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import sfm_hp100_early_acquisition as ACQ
import sfm_scene as SS


STATUS = "SFM_HP100_EARLY_LAMBDA_CALIBRATION_COMPLETE"
NO_GO = "SFM_HP100_EARLY_LAMBDA_CALIBRATION_NO_GO"
VERSION = "sfm_hp100_early_lambda_report_v2"
EXPECTED_SEEDS = (2, 3, 5)
EXPECTED_SCENE_PROFILE = "double_density_velocity_ood"
EXPECTED_SCENARIO_START = 350_000
EXPECTED_CHECKPOINT_SHA256 = (
    "258999ae8ccee8aec5aab92a6f751221d3c15583ac26e0a7ec8311f13316ec44"
)
EXPECTED_DATASET_MANIFEST_SHA256 = (
    "44f2bfa8afbb2318376ae9e188b1b622f102253a4a91f5c8ca0f9634d5041c94"
)
EXPECTED_BETA = 5.0e-4
EXPECTED_RBF_NOISE = 1.0e-2


def _metric(summary: dict, key: str) -> float:
    if key.startswith("goal_progress."):
        value = summary["goal_progress"][key.split(".", 1)[1]]
    else:
        value = summary[key]
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"missing/nonfinite calibration metric {key}")
    return float(value)


def _cell(marker: Path) -> tuple[dict, dict]:
    payload = json.loads(marker.read_text())
    if payload.get("status") != ACQ.STATUS or payload.get("version") != ACQ.VERSION:
        raise ValueError(f"incomplete acquisition marker: {marker}")
    config = payload["config"]
    expected = {
        "round": 1, "max_steps": 40, "K": 64, "B": 16, "H": 10,
        "scene_profile": EXPECTED_SCENE_PROFILE,
        "scene_profile_contract": SS.scene_profile(EXPECTED_SCENE_PROFILE),
        "scenario_start": EXPECTED_SCENARIO_START,
        "flow_base_std": 1.4, "execution_rule": "weighted_cost",
        "parallel_episodes_per_gamma": 16, "seeds": list(EXPECTED_SEEDS),
        "gammas": list(map(float, SS.GAMMAS)),
        "audit_unselected_KminusB_at_NVP": False,
    }
    for key, wanted in expected.items():
        actual = config.get(key)
        if isinstance(wanted, float):
            equal = math.isclose(float(actual), wanted, rel_tol=0.0, abs_tol=1e-8)
        else:
            equal = actual == wanted
        if not equal:
            raise ValueError(f"{marker}: {key}={actual!r}, expected {wanted!r}")
    if not math.isclose(
        float(config.get("beta", float("nan"))), EXPECTED_BETA,
        rel_tol=0.0, abs_tol=1e-12,
    ):
        raise ValueError(f"{marker}: beta is outside the frozen calibration contract")
    rbf = config.get("rbf", {})
    lengthscale = float(rbf.get("lengthscale", float("nan")))
    if (
        int(rbf.get("buffer_rows", -1)) != 0
        or not math.isfinite(lengthscale) or lengthscale <= 0.0
        or not math.isclose(
            float(rbf.get("noise", float("nan"))), EXPECTED_RBF_NOISE,
            rel_tol=0.0, abs_tol=1e-12,
        )
    ):
        raise ValueError(f"{marker}: invalid frozen empty-GP/RBF contract")
    if payload.get("policy_unchanged") is not True:
        raise ValueError(f"calibration changed the policy: {marker}")
    checkpoint = payload.get("checkpoint", {})
    if checkpoint.get("sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"{marker}: wrong canonical checkpoint SHA256")
    if (
        checkpoint.get("pretrain_dataset_manifest_sha256")
        != EXPECTED_DATASET_MANIFEST_SHA256
    ):
        raise ValueError(f"{marker}: wrong pretraining dataset ledger")
    if (
        payload.get("policy_state_sha256_before")
        != payload.get("policy_state_sha256_after")
    ):
        raise ValueError(f"{marker}: frozen policy hashes differ")
    if len(payload["lineages"]) != len(EXPECTED_SEEDS) * 7 * 16:
        raise ValueError(f"calibration cell does not contain 336 lineages: {marker}")
    summary = payload["summary"]["pooled"]
    row = {
        "lambda": float(config["execution_step_margin_weight"]),
        "S30": _metric(summary, "S30"),
        "RMST30": _metric(summary, "early_gather_RMST_at_30"),
        "RMST40": _metric(summary, "early_gather_RMST_to_max_steps"),
        "net_progress30": _metric(
            summary, "lineage_macro_net_goal_progress_at_30"
        ),
        "H10_progress30": _metric(
            summary, "goal_progress.lineage_macro_chosen_H10_mean_at_30"
        ),
        "one_step_progress30": _metric(
            summary, "goal_progress.lineage_macro_chosen_one_step_mean_at_30"
        ),
        "progress_percentile30": _metric(
            summary, "goal_progress.chosen_H10_progress_percentile_mean_at_30"
        ),
        "progress_top_quartile_fraction30": _metric(
            summary,
            "goal_progress.chosen_H10_progress_percentile_ge_0p75_fraction_at_30",
        ),
        "rbf_lengthscale": lengthscale,
        "marker": str(marker.resolve()),
    }
    return payload, row


def _scenario_ledger(payload: dict) -> list[tuple]:
    return sorted(
        (
            int(row["seed"]), float(row["gamma"]), int(row["replica"]),
            int(row["scenario_id"]),
        )
        for row in payload["lineages"]
    )


def _require_positive_baseline(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"lambda=0 baseline {label} must be finite and positive")
    return value


def build_report(markers: list[Path]) -> tuple[dict, list[dict]]:
    cells = [_cell(path) for path in markers]
    payloads = [item[0] for item in cells]
    rows = [item[1] for item in cells]
    if len({row["lambda"] for row in rows}) != len(rows):
        raise ValueError("lambda cells are not unique")
    baseline_rows = [row for row in rows if math.isclose(row["lambda"], 0.0)]
    if len(baseline_rows) != 1:
        raise ValueError("calibration requires exactly one lambda=0 control")
    checkpoint_hashes = {row["checkpoint"]["sha256"] for row in payloads}
    dataset_hashes = {
        row["checkpoint"]["pretrain_dataset_manifest_sha256"] for row in payloads
    }
    policy_hashes = {row["policy_state_sha256_after"] for row in payloads}
    scene_contracts = {
        json.dumps(row["config"]["scene_profile_contract"], sort_keys=True)
        for row in payloads
    }
    rbf_lengthscales = {row["rbf_lengthscale"] for row in rows}
    ledgers = [_scenario_ledger(row) for row in payloads]
    recorded_scene_ledgers = [row["scene_ledger"] for row in payloads]
    if (
        len(checkpoint_hashes) != 1 or len(dataset_hashes) != 1
        or len(policy_hashes) != 1 or len(scene_contracts) != 1
        or len(rbf_lengthscales) != 1
    ):
        raise ValueError("calibration cells do not share one frozen policy")
    if any(ledger != ledgers[0] for ledger in ledgers[1:]):
        raise ValueError("calibration cells do not share one CRN scenario ledger")
    if any(ledger != recorded_scene_ledgers[0] for ledger in recorded_scene_ledgers[1:]):
        raise ValueError("calibration cells do not share one full scene ledger")

    baseline = baseline_rows[0]
    baseline_net = _require_positive_baseline(
        baseline["net_progress30"], "net_progress30",
    )
    baseline_h10 = _require_positive_baseline(
        baseline["H10_progress30"], "H10_progress30",
    )
    baseline_step = _require_positive_baseline(
        baseline["one_step_progress30"], "one_step_progress30",
    )
    per_seed_baseline = payloads[rows.index(baseline)]["summary"]["per_seed"]
    for payload, row in cells:
        row["S30_delta"] = row["S30"] - baseline["S30"]
        row["net_progress_retention"] = (
            row["net_progress30"] / baseline_net
        )
        row["H10_progress_retention"] = (
            row["H10_progress30"] / baseline_h10
        )
        row["one_step_progress_retention"] = (
            row["one_step_progress30"] / baseline_step
        )
        seed_improvements = 0
        minimum_seed_net_retention = math.inf
        for seed in map(str, EXPECTED_SEEDS):
            current = payload["summary"]["per_seed"][seed]
            control = per_seed_baseline[seed]
            seed_improvements += int(float(current["S30"]) > float(control["S30"]))
            minimum_seed_net_retention = min(
                minimum_seed_net_retention,
                float(current["lineage_macro_net_goal_progress_at_30"])
                / _require_positive_baseline(
                    control["lineage_macro_net_goal_progress_at_30"],
                    f"seed {seed} net_progress30",
                ),
            )
        row["seed_S30_improvement_count"] = seed_improvements
        row["minimum_seed_net_progress_retention"] = minimum_seed_net_retention
        row["passes_aspirational_S30_0p70"] = bool(
            row["lambda"] > 0.0 and row["S30"] >= .70
        )
        row["passes"] = bool(
            row["lambda"] > 0.0
            and row["S30_delta"] >= .10
            and row["RMST30"] >= baseline["RMST30"]
            and row["RMST40"] >= baseline["RMST40"]
            and row["net_progress_retention"] >= .75
            and row["H10_progress_retention"] >= .90
            and row["one_step_progress_retention"] >= .70
            and row["progress_percentile30"] >= .70
            and seed_improvements >= 2
            and minimum_seed_net_retention >= .60
        )

    passing = [row for row in rows if row["passes"]]
    passing.sort(
        key=lambda row: (
            -row["S30"], -row["net_progress30"], row["lambda"],
        )
    )
    selected = passing[0] if passing else None
    report = {
        "status": STATUS if selected is not None else NO_GO,
        "version": VERSION,
        "selection_blind_to_training_and_evaluation": True,
        "selection_rule": "maximize S30 within a progress-preserving Pareto gate",
        "selection_stage_disclosure": (
            "calibration-stage adaptation fixed after the strict S30>=0.70 gate "
            "returned NO-GO and before any expansion training or raw evaluation"
        ),
        "checkpoint_sha256": next(iter(checkpoint_hashes)),
        "pretrain_dataset_manifest_sha256": next(iter(dataset_hashes)),
        "rbf_lengthscale": next(iter(rbf_lengthscales)),
        "seeds": list(EXPECTED_SEEDS),
        "gate": {
            "S30_delta_vs_lambda0_min": .10,
            "RMST30_vs_lambda0": ">=1", "RMST40_vs_lambda0": ">=1",
            "net_progress_retention_min": .75,
            "H10_progress_retention_min": .90,
            "one_step_progress_retention_min": .70,
            "progress_percentile_min": .70,
            "seed_S30_improvement_count_min": 2,
            "minimum_seed_net_progress_retention": .60,
            "aspirational_S30_min_reported_not_required": .70,
        },
        "baseline": baseline,
        "selected": selected,
        "rows": sorted(rows, key=lambda row: row["lambda"]),
    }
    return report, report["rows"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marker", action="append", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args(argv)
    outdir = Path(args.outdir).resolve()
    if outdir.exists():
        raise FileExistsError(f"refusing existing output directory: {outdir}")
    report, rows = build_report([Path(value).resolve() for value in args.marker])
    outdir.mkdir(parents=True)
    json_path = outdir / "lambda_calibration.json"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    csv_path = outdir / "lambda_calibration.csv"
    with csv_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({
        "status": report["status"], "selected": report["selected"],
        "json": str(json_path), "csv": str(csv_path),
    }, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
