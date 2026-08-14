from __future__ import annotations

import sfm_hp100_disaster_prefix_sweep as S


def _metrics(*, sr=.5, timeout=.1, validity=.6):
    return {
        "SR": sr, "CR": 1.0 - sr - timeout, "timeout": timeout,
        "Validity": validity, "successful_clearance": .1,
        "successful_time_to_goal": 10.0,
    }


def _marker(*, anchor_positive=30, treatment_positive=31,
            anchor_progress=.5, treatment_progress=.49,
            anchor_recovered=2, treatment_recovered=2, drift=.01):
    def audit(positive, progress):
        return {
            "M": 64, "exact_positive": positive,
            "exact_positive_fraction": positive / 64,
            "mean_H10_progress": progress, "mean_step_margin": .2,
        }

    return {
        "fixed_raw_M64": {"rows": [{
            "r0": audit(29, anchor_progress),
            "alpha0_positive_anchor": audit(anchor_positive, anchor_progress),
            "treatment": audit(treatment_positive, treatment_progress),
        }]},
        "targeted_Nplus1": {
            "r0_recovered": 0, "alpha0_recovered": anchor_recovered,
            "treatment_recovered": treatment_recovered,
        },
        "update": {"relative_parameter_drift": drift},
    }


def _candidate(**overrides):
    values = dict(
        scope="head_only", dose=S.REFERENCE_DOSE, marker=_marker(),
        r0_raw=_metrics(sr=.49, timeout=.1, validity=.59),
        anchor_raw=_metrics(sr=.5, timeout=.1, validity=.6),
        treatment_raw=_metrics(sr=.51, timeout=.1, validity=.61),
        anchor_checkpoint={"path": "anchor", "sha256": "a" * 64},
        treatment_checkpoint={"path": "treatment", "sha256": "b" * 64},
    )
    values.update(overrides)
    return S._candidate_summary(**values)


def test_declared_table_is_compact_and_exercises_anchor_and_negative_doses():
    doses = S.declared_stage1_doses()
    assert len(doses) == 7
    assert doses[0] == S.REFERENCE_DOSE
    assert len({dose.name for dose in doses}) == len(doses)
    assert len({dose.anchor_key for dose in doses}) == 4
    assert {dose.negative_passes for dose in doses} == {4, 16}
    assert {dose.negative_alpha for dose in doses} == {.1, .3}
    assert any(dose.p1_passes == 4 and dose.p2_passes == 4 for dose in doses)


def test_continuation_command_is_hard_coded_to_ncausal_only(tmp_path):
    scope = S.ScopeInput(
        "head_only", tmp_path / "trace.pt", "1" * 64,
        tmp_path / "staged.pt", "2" * 64,
    )
    command = S._continuation_command(
        python="python", continuation=tmp_path / "continue.py",
        checkpoint=tmp_path / "r0.pt", checkpoint_sha256="3" * 64,
        dataset_root=tmp_path / "dataset", dataset_manifest_sha256="4" * 64,
        scope_input=scope, dose=S.REFERENCE_DOSE,
        output=tmp_path / "out", gpu=1, verifier_workers=16,
        scenario_start=500_000, max_drift=.1,
    )
    assert command[command.index("--negative-source") + 1] == "Ncausal_only"
    assert "Dminus_plus_Ncausal" not in command
    assert command[command.index("--negative-passes") + 1] == "4"
    assert command[command.index("--optimizer-scope") + 1] == "head_only"


def test_marker_validation_rejects_any_dminus_exposure(tmp_path):
    scope = S.ScopeInput(
        "head_only", tmp_path / "trace.pt", "1" * 64,
        tmp_path / "staged.pt", "2" * 64,
    )
    dose = S.REFERENCE_DOSE
    marker = {
        "status": "SFM_HP100_DISASTER_PREFIX_DIAGNOSTIC_COMPLETE",
        "preflight": {
            "optimizer_scope": "head_only",
            "source": {
                "trace_sha256": "1" * 64,
                "staged_samples_sha256": "2" * 64,
            },
            "update": {
                "P1_lr": dose.p1_learning_rate, "P1_passes": dose.p1_passes,
                "P2_lr": dose.p2_learning_rate, "P2_passes": dose.p2_passes,
                "negative_lr": S.NEGATIVE_LEARNING_RATE,
                "negative_alpha": dose.negative_alpha,
                "negative_passes": dose.negative_passes,
                "negative_source": S.NEGATIVE_SOURCE,
            },
        },
        "update": {"negative": {
            "negative_source": S.NEGATIVE_SOURCE,
            "selected_source_counts": {"Dminus": 0, "Ncausal": 18},
        }},
    }
    S._validate_run_marker(marker, dose, scope)
    marker["update"]["negative"]["selected_source_counts"]["Dminus"] = 1
    try:
        S._validate_run_marker(marker, dose, scope)
    except RuntimeError as error:
        assert "Dminus" in str(error)
    else:
        raise AssertionError("Dminus exposure must fail closed")


def test_candidate_gate_requires_strict_sr_and_m64_validity_gain():
    assert _candidate()["eligible"]
    no_sr_gain = _candidate(
        treatment_raw=_metrics(sr=.5, timeout=.1, validity=.61),
    )
    assert not no_sr_gain["eligible"]
    assert not no_sr_gain["gates"]["incremental_raw_SR_strictly_positive"]
    no_m64_gain = _candidate(marker=_marker(treatment_positive=30))
    assert not no_m64_gain["eligible"]
    assert not no_m64_gain["gates"][
        "incremental_M64_Validity_strictly_positive"
    ]


def test_candidate_gate_rejects_timeout_recovery_and_progress_regressions():
    timeout = _candidate(
        treatment_raw=_metrics(sr=.51, timeout=.11, validity=.61),
    )
    recovery = _candidate(marker=_marker(treatment_recovered=1))
    progress = _candidate(
        marker=_marker(anchor_progress=.5, treatment_progress=.479),
    )
    assert not timeout["gates"]["timeout_nonincrease"]
    assert not recovery["gates"]["target_recovery_nonregression"]
    assert not progress["gates"]["mean_H10_progress_drop_at_most_0p02"]


def test_stage1_candidate_must_not_regress_r0():
    row = _candidate(
        r0_raw=_metrics(sr=.52, timeout=.09, validity=.62),
    )
    assert not row["eligible"]
    assert not row["gates"]["r0_raw_SR_nonregression"]
    assert not row["gates"]["r0_raw_Validity_nonregression"]
    assert not row["gates"]["r0_timeout_nonincrease"]


def test_m50_phase2_gate_requires_incremental_and_strict_r0_win():
    r0 = _metrics(sr=.50, timeout=.10, validity=.60)
    anchor = _metrics(sr=.51, timeout=.09, validity=.61)
    treatment = _metrics(sr=.52, timeout=.09, validity=.62)
    _, _, incremental, strict_r0 = S.confirmation_decision(
        treatment, anchor, r0,
    )
    assert incremental and strict_r0

    _, _, incremental, strict_r0 = S.confirmation_decision(
        _metrics(sr=.52, timeout=.09, validity=.605), anchor, r0,
    )
    assert not incremental and strict_r0


def test_candidate_ranking_uses_bottleneck_gain_before_recovery_and_drift():
    left = _candidate()
    right = _candidate(
        treatment_raw=_metrics(sr=.52, timeout=.1, validity=.61),
        marker=_marker(treatment_positive=34, treatment_recovered=6, drift=.001),
    )
    assert S.candidate_key(right) > S.candidate_key(left)


def test_scope_selection_uses_capacity_only_after_liveness_matches():
    head = _candidate(scope="head_only")
    last = _candidate(scope="last_block_and_head")
    selected, audit = S.choose_stage0_scope([head, last])
    assert selected["scope"] == "last_block_and_head"
    assert "capacity_tiebreak" in audit["rule"]

    lower_sr_last = _candidate(
        scope="last_block_and_head",
        anchor_raw=_metrics(sr=.48, timeout=.1, validity=.6),
        treatment_raw=_metrics(sr=.49, timeout=.1, validity=.61),
    )
    selected, audit = S.choose_stage0_scope([head, lower_sr_last])
    assert selected["scope"] == "head_only"
    assert "liveness_regression" in audit["rule"]


def test_stage0_reference_incremental_failure_does_not_block_scope_selection():
    head = _candidate(
        scope="head_only",
        treatment_raw=_metrics(sr=.5, timeout=.1, validity=.6),
        marker=_marker(treatment_positive=30),
    )
    last = _candidate(
        scope="last_block_and_head",
        treatment_raw=_metrics(sr=.5, timeout=.1, validity=.6),
        marker=_marker(treatment_positive=30),
    )
    assert not head["eligible"] and not last["eligible"]
    selected, _ = S.choose_stage0_scope([head, last])
    assert selected["scope"] == "last_block_and_head"


def test_evaluation_cache_uses_policy_state_not_torch_container_bytes():
    first = {
        "path": "first.pt", "sha256": "1" * 64,
        "policy_state_sha256": "a" * 64,
    }
    second = {
        "path": "second.pt", "sha256": "2" * 64,
        "policy_state_sha256": "a" * 64,
    }
    assert S._evaluation_cache_key(first) == S._evaluation_cache_key(second)


def test_parser_pins_declared_banks_and_two_gpus():
    parser = S.parser()
    defaults = {
        action.dest: action.default for action in parser._actions
        if action.dest != "help"
    }
    assert defaults["gpus"] == "1,3"
    assert S.SCREEN_EP0 == 610_000
    assert S.SCREEN_M == 10
    assert S.SCREEN_NOISE_SEED == 20_260_805
    assert S.CONFIRM_EP0 == 620_000
    assert S.CONFIRM_M == 50
    assert S.CONFIRM_NOISE_SEED == 20_260_806
