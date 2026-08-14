from dataclasses import dataclass

import pytest

import sfm_hp100_predictive_execution as P


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


def audit(progress, clearance, collision_free=True):
    return {
        "H10_goal_progress": float(progress),
        "predicted_min_clearance": float(clearance),
        "predicted_collision_free": bool(collision_free),
    }


def test_predictive_selector_prioritizes_progress_over_margin_and_clearance():
    results = [
        Result(True, 0.4, step_margin=100.0),
        Result(True, 0.8, step_margin=0.01),
        Result(False, 2.0, step_margin=1000.0),
    ]
    chosen = P.select_predictive_progress(
        results,
        [audit(.4, .5), audit(.8, .1), audit(2.0, 1.0)],
        [.9, .1, 10.0],
    )
    assert chosen == 1


def test_predictive_selector_uses_clearance_then_uncertainty_as_ties():
    results = [Result(True, .8), Result(True, .8), Result(True, .8)]
    audits = [audit(.8, .1), audit(.8, .2), audit(.8, .2)]
    assert P.select_predictive_progress(results, audits, [.9, .1, .8]) == 2


def test_exact_positive_must_reproduce_constant_velocity_collision_audit():
    with pytest.raises(RuntimeError, match="duplicate CV collision"):
        P.select_predictive_progress(
            [Result(True, .5)], [audit(.5, -.01, False)], [.5],
        )


def test_no_exact_positive_returns_none_and_archives_best_progress_negative():
    results = [Result(False, .2), Result(False, .7), Result(False, .4)]
    audits = [audit(.2, -.1), audit(.7, -.2), audit(.4, -.05)]
    assert P.select_predictive_progress(results, audits, [.1, .2, .3]) is None
    assert P._negative_counterfactual(results, audits) == 1


def test_declared_budget_and_retry_schedule_are_frozen():
    config = P.PredictiveConfig()
    config.validate()
    assert (config.K, config.B) == (64, 32)
    assert config.base_std_start == 1.0
    assert config.base_std_step == 0.1


def test_sample_record_allows_only_declared_realized_failure_override():
    result = Result(True, .5)
    common = dict(
        key=type("Key", (), {"gamma": .3, "replica": 0, "label": "g.3:r0"})(),
        state=type("State", (), {"scenario_id": 7})(), step=2, attempt=0,
        context=P.torch.zeros(3), candidate=P.torch.zeros(10, 2),
        flow_base=P.torch.zeros(10, 2), result=result,
        audit=audit(.5, .1), executed=True,
    )
    row = P._sample_record(
        role="negative", negative_reason="realized_collision", **common,
    )
    assert row["role"] == "negative"
    assert row["verification"]["valid"] is True
    with pytest.raises(ValueError, match="failure reason"):
        P._sample_record(role="negative", negative_reason=None, **common)
