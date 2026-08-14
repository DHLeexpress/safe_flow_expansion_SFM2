import json

import pytest

import execution_lambda_preflight as P


def _records():
    return [
        dict(gamma=.30000001, context_id="c1", candidate_id="a", exact_positive=True,
             execution_eligible=True, native_cost=10., step_margin=0., note="preserve-me"),
        dict(gamma=.3, context_id="c1", candidate_id="b", exact_positive=True,
             execution_eligible=True, native_cost=14., step_margin=2.),
        dict(gamma=.3, context_id="c2", candidate_id="a", exact_positive=True,
             execution_eligible=True, native_cost=20., step_margin=1.),
        dict(gamma=.3, context_id="c2", candidate_id="b", exact_positive=True,
             execution_eligible=True, native_cost=22., step_margin=2.),
        dict(gamma=.3, context_id="nvp", candidate_id="a", exact_positive=False,
             execution_eligible=False, native_cost=1., step_margin=9.),
    ]


def test_robust_scale_sweep_and_nvp_are_reported_without_fallback():
    records = _records()
    report = P.build_report(
        records,
        raw_input={"format": "memory", "payload": records},
        declared_gammas=[.3],
    )
    calibration = report["calibration"]
    assert calibration["median_native_cost_span"] == 3.
    assert calibration["median_nonzero_step_margin_span"] == 1.5
    assert calibration["lambda0"] == 2.

    by_multiplier = {row["multiplier"]: row for row in report["sweep"]}
    assert [row["candidate_id"] for row in by_multiplier[0.0]["selections"]] == ["a", "a"]
    assert [row["candidate_id"] for row in by_multiplier[2.0]["selections"]] == ["b", "b"]
    assert by_multiplier[0.0]["agreement_with_pure_cost"] == 1.
    assert by_multiplier[2.0]["agreement_with_pure_margin"] == 1.
    assert by_multiplier[1.0]["nvp"] == [{"gamma": .3, "context_id": "nvp"}]
    gamma = by_multiplier[1.0]["per_gamma"]["0.3"]
    assert gamma["contexts"] == 3
    assert gamma["nvp"] == 1
    assert gamma["eligible_candidates"] == 4
    assert report["raw_input"]["payload"][0]["note"] == "preserve-me"


def test_jsonl_loader_preserves_rows_and_cli_field_mapping(tmp_path):
    rows = []
    for row in _records():
        rows.append({
            "g": row["gamma"], "ctx": row["context_id"], "cand": row["candidate_id"],
            "y": row["exact_positive"], "eligible": row["execution_eligible"],
            "J": row["native_cost"], "m": row["step_margin"],
        })
    source = tmp_path / "input.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    output = tmp_path / "report.json"
    assert P.main([
        str(source), "--output", str(output), "--declared-gammas", "0.3",
        "--gamma-field", "g", "--context-id-field", "ctx",
        "--candidate-id-field", "cand", "--exact-positive-field", "y",
        "--execution-eligible-field", "eligible", "--native-cost-field", "J",
        "--step-margin-field", "m",
    ]) == 0
    result = json.loads(output.read_text())
    assert result["status"] == "EXECUTION_LAMBDA_PREFLIGHT_COMPLETE"
    assert result["raw_input"]["format"] == "jsonl"
    assert result["raw_input"]["payload"] == rows


@pytest.mark.parametrize("field,value,match", [
    ("exact_positive", 1, "JSON boolean"),
    ("native_cost", float("inf"), "finite"),
    ("context_id", None, "string or integer"),
])
def test_bad_schema_fails_closed(field, value, match):
    records = _records()
    records[0] = {**records[0], field: value}
    with pytest.raises(ValueError, match=match):
        P.normalize_candidates(records, declared_gammas=[.3])


def test_inconsistent_gate_duplicate_candidate_and_gamma_mismatch_fail_closed():
    records = _records()
    records[0] = {**records[0], "exact_positive": False, "execution_eligible": True}
    with pytest.raises(ValueError, match="inconsistent gate"):
        P.normalize_candidates(records, declared_gammas=[.3])

    records = _records()
    records[1] = {**records[1], "candidate_id": "a"}
    with pytest.raises(ValueError, match="duplicate candidate_id"):
        P.normalize_candidates(records, declared_gammas=[.3])

    with pytest.raises(ValueError, match="matches 0 declared"):
        P.normalize_candidates(_records(), declared_gammas=[.5])


def test_lambda_is_not_identifiable_without_margin_or_cost_contrast():
    records = _records()[:2]
    for row in records:
        row["step_margin"] = 1.
    candidates = P.normalize_candidates(records, declared_gammas=[.3])
    with pytest.raises(ValueError, match="no nonzero"):
        P.calibrate_lambda(candidates)

    records = _records()[:2]
    for row in records:
        row["native_cost"] = 1.
    candidates = P.normalize_candidates(records, declared_gammas=[.3])
    with pytest.raises(ValueError, match="cost span is zero"):
        P.calibrate_lambda(candidates)
