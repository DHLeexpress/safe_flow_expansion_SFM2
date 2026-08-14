import csv
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/sfm_hp100_early_eval_report.py"
SPEC = importlib.util.spec_from_file_location("sfm_hp100_early_eval_report", SCRIPT)
REPORT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(REPORT)


def _cell(rows):
    n = len(rows)
    successes = [row for row in rows if row["success"]]
    return {
        "n": n,
        "SR": len(successes) / n,
        "CR": sum(row["collision"] for row in rows) / n,
        "timeout": sum(row["timeout"] for row in rows) / n,
        "Validity": sum(row["validity"] for row in rows) / n,
        "successful_clearance": (
            sum(row["successful_clearance"] for row in successes) / len(successes)
            if successes else None
        ),
        "successful_time_to_goal": (
            sum(row["time_to_goal"] for row in successes) / len(successes)
            if successes else None
        ),
    }


def _payload(round_i):
    gammas = (0.1, 1.0)
    ep0, m = 9000, 3
    rows = []
    for gamma_index, gamma in enumerate(gammas):
        for rollout_index in range(m):
            outcome = (round_i + gamma_index + rollout_index) % 3
            success = outcome == 0
            collision = outcome == 1
            rows.append({
                "episode": ep0 + rollout_index,
                "gamma": gamma,
                "success": success,
                "collision": collision,
                "timeout": outcome == 2,
                "validity": 0.2 + 0.1 * round_i + 0.02 * gamma_index,
                "successful_clearance": 0.1 + 0.01 * round_i if success else None,
                "time_to_goal": 8.0 + round_i if success else None,
            })
    per_gamma = {
        str(gamma): _cell([row for row in rows if row["gamma"] == gamma])
        for gamma in gammas
    }
    return {
        "status": "SFM_HP100_RAW_EVAL_COMPLETE",
        "version": "sfm_hp100_raw_clipped_v1",
        "evaluator_source": f"/frozen/worktree/r{round_i}/sfm_hp100_eval.py",
        "evaluator_sha256": "eval-sha",
        "checkpoint": f"/checkpoints/round_{round_i:02d}.pt",
        "checkpoint_sha256": f"checkpoint-{round_i}",
        "architecture": {"arch": "v3-sfm-hp100-residual", "grid_shape": [10, 32, 100]},
        "dynamics": {"name": "componentwise_capped_double_integrator_v1"},
        "verifier": {
            "contract": {"solver": "exact_2d_angular_interval_socp"},
            "evaluator_source": f"/frozen/worktree/r{round_i}/sfm_metrics2.py",
            "evaluator_sha256": "verifier-sha",
            "polytope_source": f"/frozen/worktree/r{round_i}/verifier_polytope.py",
            "polytope_sha256": "polytope-sha",
        },
        "observation": {"tensor_shape": [10, 32, 100], "predict_gain": 0.0},
        "scene": {"scene_profile": "double_density_velocity_ood", "n_ped": 40},
        "ep0": ep0,
        "M_per_gamma": m,
        "temperature": 1.0,
        "NFE": 8,
        "noise_seed": 20260802,
        "noise_bank": {
            "seed": 20260802,
            "shape": [len(gammas), m, 180, 20],
            "dtype": "float32",
            "sha256": "same-crn-bank",
        },
        "summary": {"pooled": _cell(rows), "per_gamma": per_gamma},
        "rows": rows,
        "semantics": "unguided raw flow; temp=1",
    }


def _inputs(tmp_path):
    inputs = {}
    for round_i in REPORT.ROUNDS:
        path = tmp_path / f"r{round_i}.json"
        path.write_text(json.dumps(_payload(round_i)))
        inputs[round_i] = path
    return inputs


def test_build_report_validates_and_writes_compact_artifacts(tmp_path):
    inputs = _inputs(tmp_path)
    outputs = REPORT.build_report(inputs, tmp_path / "report")

    assert set(outputs) == {"csv", "json", "png", "pdf"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs.values())
    payload = json.loads(outputs["json"].read_text())
    assert payload["status"] == "SFM_HP100_EARLY_RAW_REPORT_COMPLETE"
    assert payload["rounds"] == [0, 1, 2, 3]
    assert payload["gammas"] == [0.1, 1.0]
    assert len(payload["rows"]) == 4 * 3
    assert payload["crn_contract"]["noise_bank"]["sha256"] == "same-crn-bank"

    with outputs["csv"].open(newline="") as stream:
        table = list(csv.DictReader(stream))
    assert len(table) == 12
    assert table[0]["scope"] == "pooled"
    assert table[0]["SR"] == str(_payload(0)["summary"]["pooled"]["SR"])


@pytest.mark.parametrize("field", ["noise", "source"])
def test_build_report_rejects_mixed_crn_or_source_contract(tmp_path, field):
    inputs = _inputs(tmp_path)
    broken = json.loads(inputs[2].read_text())
    if field == "noise":
        broken["noise_bank"]["sha256"] = "different-bank"
        match = "CRN contract"
    else:
        broken["evaluator_sha256"] = "different-evaluator"
        match = "source contract"
    inputs[2].write_text(json.dumps(broken))

    with pytest.raises(ValueError, match=match):
        REPORT.build_report(inputs, tmp_path / "report")
    assert not (tmp_path / "report").exists()


def test_build_report_rejects_stale_summary(tmp_path):
    inputs = _inputs(tmp_path)
    broken = json.loads(inputs[3].read_text())
    broken["summary"]["pooled"]["Validity"] += 0.1
    inputs[3].write_text(json.dumps(broken))

    with pytest.raises(ValueError, match="stored Validity is inconsistent"):
        REPORT.build_report(inputs, tmp_path / "report")
