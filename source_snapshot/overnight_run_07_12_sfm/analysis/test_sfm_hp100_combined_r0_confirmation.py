import sfm_hp100_combined_r0_confirmation as SEC
import sfm_hp100_iterative_microcycles as ITER


def _metrics(success, collision, timeout, validity, n=70):
    return {
        "n": n, "SR": success / n, "CR": collision / n,
        "timeout": timeout / n, "Validity": validity,
    }


def test_m50_gate_uses_integer_outcomes_and_strict_validity():
    r0 = _metrics(140, 205, 5, 0.60, n=350)
    treatment = _metrics(141, 204, 5, 0.61, n=350)
    assert all(SEC.m50_gates(treatment, r0).values())
    treatment["Validity"] = 0.60
    assert not SEC.m50_gates(treatment, r0)[
        "M50_raw_Validity_strictly_above_r0"
    ]


def test_selection_key_prefers_liveness_after_shared_m64_bottleneck():
    base = {
        "deltas_vs_r0": {
            "M10_SR": 5 / 70, "M64_Validity": 1 / 384,
            "M10_raw_Validity": 0.03,
        },
        "targeted_recovery": {"treatment": 1},
        "treatment_M64": {"mean_H10_progress": 0.68},
        "relative_parameter_drift": 0.01, "dose_index": 5,
    }
    lower_sr = {
        **base, "deltas_vs_r0": {
            **base["deltas_vs_r0"], "M10_SR": 3 / 70,
        }, "dose_index": 4,
    }
    assert SEC.selection_key(base) > SEC.selection_key(lower_sr)


def test_iterative_validator_accepts_secondary_without_false_incremental_claim():
    marker = {
        "status": SEC.COMPLETE_STATUS,
        "eligible_for_iterative_microcycles": True,
        "combined_M50_strict_r0_win": True,
        "primary_incremental_null_preserved": True,
        "primary_incremental_claim": False,
        "selected_scope": "head_only",
        "selected_dose": {
            "name": "retain_p1_a0p1_n4", "p1_learning_rate": 3e-5,
            "p1_passes": 1, "p2_learning_rate": 3e-4, "p2_passes": 1,
            "negative_alpha": 0.1, "negative_passes": 4,
        },
        "selected_treatment_checkpoint": {
            "path": "/tmp/x", "sha256": "a" * 64,
            "policy_state_sha256": "b" * 64,
        },
    }
    dose = ITER._validate_selection_marker(marker)
    assert dose.name == "retain_p1_a0p1_n4"
    marker["primary_incremental_claim"] = True
    try:
        ITER._validate_selection_marker(marker)
    except ValueError as error:
        assert "incremental" in str(error)
    else:
        raise AssertionError("false incremental claim was accepted")
