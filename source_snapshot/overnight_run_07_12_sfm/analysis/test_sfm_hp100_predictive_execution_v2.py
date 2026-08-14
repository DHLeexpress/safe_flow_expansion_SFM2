"""Tests for the experimental MPC-cost execution rule (v2 selector)."""
from dataclasses import dataclass

import numpy as np
import pytest

import sfm_hp100_ball_adapter as PORT
import sfm_hp100_mpc_rule_search as GRID
import sfm_hp100_predictive_execution as PRED
import sfm_hp100_predictive_execution_v2 as V2
import sfm_scene as SS


@dataclass
class Result:
    valid: bool
    progress: float
    step_margin: float = 0.0
    hp_eligible: bool = True
    margin: float = 0.0
    execution_cost: float = 0.0
    progress_eligible: bool = True
    error: bool = False


def audit(progress, clearance, horizon=None, collision_free=True):
    value = {
        "H10_goal_progress": float(progress),
        "predicted_min_clearance": float(clearance),
        "predicted_collision_free": bool(collision_free),
    }
    if horizon is not None:
        value[V2.HORIZON_KEY] = [float(h) for h in horizon]
    return value


PARAMS = V2.MPCRuleParams()


def test_eligibility_matches_the_frozen_rule():
    results = [
        Result(False, 2.0),                      # exact negative
        Result(True, 0.9, error=True),           # verifier error
        Result(True, 0.8),                       # only eligible row
    ]
    audits = [
        audit(2.0, 0.5, [0.5] * 10),
        audit(0.9, 0.5, [0.5] * 10),
        audit(0.8, 0.5, [0.5] * 10),
    ]
    assert V2.select_predictive_mpc(
        results, audits, [0.1, 0.2, 0.3], params=PARAMS,
    ) == 2
    all_negative = [Result(False, 0.5), Result(False, 0.7)]
    negative_audits = [audit(0.5, -0.1, [0.1] * 10), audit(0.7, -0.2, [0.1] * 10)]
    assert V2.select_predictive_mpc(
        all_negative, negative_audits, [0.1, 0.2], params=PARAMS,
    ) is None
    assert PRED.select_predictive_progress(
        all_negative, negative_audits, [0.1, 0.2],
    ) is None
    with pytest.raises(RuntimeError, match="duplicate CV collision"):
        V2.select_predictive_mpc(
            [Result(True, 0.5)],
            [audit(0.5, -0.01, [0.5] * 10, collision_free=False)],
            [0.5], params=PARAMS,
        )
    with pytest.raises(RuntimeError, match="H10 progress differ"):
        V2.select_predictive_mpc(
            [Result(True, 0.5)],
            [audit(0.9, 0.5, [0.5] * 10)],
            [0.5], params=PARAMS,
        )


def test_near_horizon_violation_flips_the_progress_choice():
    # A wins on progress but dips to 0.05 m at h=1; B stays wide open.
    results = [Result(True, 1.5), Result(True, 1.4)]
    audits = [
        audit(1.5, 0.05, [0.05] + [0.5] * 9),
        audit(1.4, 0.50, [0.5] * 10),
    ]
    assert PRED.select_predictive_progress(results, audits, [0.1, 0.1]) == 0
    assert V2.select_predictive_mpc(
        results, audits, [0.1, 0.1], params=PARAMS,
    ) == 1
    # Costs were recorded per candidate.
    assert audits[0][V2.COST_KEY] > audits[1][V2.COST_KEY]


def test_rho_weights_the_same_violation_more_at_near_horizon():
    near = [0.1] + [1.0] * 9      # violation at h=1
    late = [1.0] * 8 + [0.1, 1.0]  # identical violation at h=9
    cost_near = V2.mpc_cost(1.0, near, PARAMS)
    cost_late = V2.mpc_cost(1.0, late, PARAMS)
    assert cost_near > cost_late
    results = [Result(True, 1.0), Result(True, 1.0)]
    audits = [audit(1.0, 0.1, near), audit(1.0, 0.1, late)]
    assert V2.select_predictive_mpc(
        results, audits, [0.5, 0.5], params=PARAMS,
    ) == 1


def test_determinism_and_index_tiebreak():
    results = [Result(True, 1.0), Result(True, 1.0)]
    horizon = [0.4] * 10
    for _ in range(3):
        audits = [audit(1.0, 0.4, horizon), audit(1.0, 0.4, horizon)]
        assert V2.select_predictive_mpc(
            results, audits, [0.9, 0.9], params=PARAMS,
        ) == 0


def test_infinite_clearance_contributes_zero_cost():
    empty_crowd = [float("inf")] * 10
    assert V2.mpc_cost(1.0, empty_crowd, PARAMS) == pytest.approx(-1.0)


def test_missing_horizon_clearances_fail_closed():
    with pytest.raises(ValueError, match="requires per-horizon clearances"):
        V2.select_predictive_mpc(
            [Result(True, 0.5)], [audit(0.5, 0.5)], [0.5], params=PARAMS,
        )


def test_horizon_clearances_match_the_frozen_cv_audit():
    class StubTask:
        def decode_context(self, context):
            robot = np.array([0.0, 0.0, 1.0, 0.5], np.float32)
            ped_xy = np.array([[1.0, 0.2], [2.0, -1.0]], np.float32)
            ped_vel = np.array([[-0.5, 0.0], [0.0, 0.8]], np.float32)
            return robot, ped_xy, ped_vel

    import torch
    task = StubTask()
    context = torch.zeros(1)
    candidate = torch.linspace(-1.0, 1.0, 20).reshape(10, 2)
    vector = V2.horizon_clearances(task, context, candidate)
    reference = PRED.prediction_metrics(task, context, candidate)
    assert len(vector) == 10
    assert min(vector) == pytest.approx(
        reference["predicted_min_clearance"], abs=1.0e-6,
    )
    components = V2.mpc_cost_components(task, context, candidate)
    assert components["horizon_clearances"] == vector
    assert components["min_clearance"] == pytest.approx(min(vector))


def test_install_v2_patches_shadow_records_and_restores():
    original_selector = PRED.select_predictive_progress
    original_metrics = PRED.prediction_metrics
    shadow = []
    with V2.install_v2_selector(PARAMS, shadow):
        assert PRED.select_predictive_progress is not original_selector
        assert PRED.prediction_metrics is not original_metrics
        results = [Result(True, 1.5), Result(True, 1.4)]
        audits = [
            audit(1.5, 0.05, [0.05] + [0.5] * 9),
            audit(1.4, 0.50, [0.5] * 10),
        ]
        chosen = PRED.select_predictive_progress(results, audits, [0.1, 0.1])
        assert chosen == 1
        assert shadow[-1]["old_rule_local"] == 0
        assert shadow[-1]["new_rule_local"] == 1
        assert shadow[-1]["changed"] is True
        assert shadow[-1]["eligible"] == 2
        assert audits[0][V2.SHADOW_KEY]["new_rule_local"] == 1
    assert PRED.select_predictive_progress is original_selector
    assert PRED.prediction_metrics is original_metrics
    assert V2.shadow_summary(shadow) == {
        "selector_calls": 1, "comparable_calls": 1, "changed_calls": 1,
        "change_fraction": 1.0,
    }


def test_patched_metrics_appends_the_horizon_vector():
    class StubTask:
        def decode_context(self, context):
            robot = np.array([0.0, 0.0, 0.5, 0.5], np.float32)
            ped_xy = np.array([[1.5, 1.5]], np.float32)
            ped_vel = np.array([[0.0, 0.0]], np.float32)
            return robot, ped_xy, ped_vel

    import torch
    with V2.install_v2_selector(PARAMS, []):
        value = PRED.prediction_metrics(
            StubTask(), torch.zeros(1), torch.zeros(10, 2),
        )
    assert len(value[V2.HORIZON_KEY]) == 10
    assert min(value[V2.HORIZON_KEY]) == pytest.approx(
        value["predicted_min_clearance"], abs=1.0e-6,
    )


def test_v2_provenance_is_unmistakable():
    value = V2.v2_provenance(PARAMS)
    assert value["execution_rule"] == "predictive_mpc_v2"
    assert value["authoritative"] is False
    assert value["mpc_params"] == {
        "lam": 1.0, "rho": 1.1, "r_eff": 0.30, "sigma_len": 0.10,
    }
    assert len(value["v2_source_sha256"]) == 64
    assert value["base_module"]["version"] == PRED.VERSION


def test_declared_grid_is_nine_unique_combos_and_ranking_criterion():
    grid = GRID.declared_grid()
    assert len(grid) == 9
    identities = [GRID.combo_id(params) for params in grid]
    assert len(set(identities)) == 9
    for params in grid:
        params.validate()
        assert params.rho == GRID.GRID_RHO
        assert params.sigma_len == GRID.GRID_SIGMA_LEN

    def row(identity, collision, nvp, ttg, oob=0):
        return {
            "combo_id": identity,
            "statistics": {"pooled": {
                "collision": collision, "nvp": nvp, "oob": oob,
                "mean_success_ttg_seconds": ttg,
            }},
        }

    ranked = GRID.rank_combos([
        row("slow_safe", 0, 2, 9.0),
        row("fast_safe", 0, 2, 7.0),
        row("fast_nvp", 0, 9, 6.0),
        row("colliding", 1, 0, 5.0),
        row("oob_arm", 0, 0, 5.0, oob=2),
    ])
    assert ranked == [
        "fast_safe", "slow_safe", "fast_nvp", "colliding", "oob_arm",
    ]


def test_combo_statistics_aggregates_outcomes_and_ttg():
    marker = {
        "outcomes": {
            "g0.2:rep00": {"gamma": 0.2, "status": "success",
                           "executed_steps": 60},
            "g0.2:rep01": {"gamma": 0.2, "status": "nvp",
                           "executed_steps": 12},
            "g0.3:rep00": {"gamma": 0.3, "status": "success",
                           "executed_steps": 80},
            "g0.3:rep01": {"gamma": 0.3, "status": "collision",
                           "executed_steps": 30},
        },
        "summary": {"pooled_final_attempt": {"predictive_clearance": 0.21}},
        "selector_shadow": {"change_fraction": 0.4},
    }
    statistics = GRID.combo_statistics(marker)
    assert statistics["pooled"]["success"] == 2
    assert statistics["pooled"]["nvp"] == 1
    assert statistics["pooled"]["collision"] == 1
    assert statistics["pooled"]["mean_success_ttg_seconds"] == pytest.approx(7.0)
    assert statistics["per_gamma"]["0.2"]["nvp"] == 1
    assert statistics["per_gamma"]["0.3"][
        "mean_success_ttg_seconds"
    ] == pytest.approx(8.0)
    assert statistics["mean_executed_step_clearance"] == 0.21
    assert statistics["selector_shadow"]["change_fraction"] == 0.4
