import numpy as np

import sfm_hp100_predictive_execution_viz as V


def event(
    *, gamma, scenario, step, nearest, positives=8,
    predictive=1, reference=0, predictive_progress=.6,
    reference_progress=.3, predictive_clearance=.1,
):
    ped_xy = np.array([[nearest + .2, 0.0]], np.float32)
    verification = [
        {"valid": True, "H10_progress": reference_progress, "step_margin": .5},
        {"valid": True, "H10_progress": predictive_progress, "step_margin": .2},
    ]
    audits = [
        {"predicted_collision_free": True, "predicted_min_clearance": .2},
        {
            "predicted_collision_free": predictive_clearance >= 0.0,
            "predicted_min_clearance": predictive_clearance,
        },
    ]
    return {
        "gamma": gamma,
        "lineage": f"g{gamma}:rep00",
        "scenario_id": scenario,
        "step": step,
        "state_before": np.zeros(4, np.float32),
        "ped_xy": ped_xy,
        "attempts": [{
            "attempt": 0,
            "base_std": 1.0,
            "positive_B32": positives,
            "predictive_local": predictive,
            "max_margin_reference_local": reference,
            "verification": verification,
            "prediction_audits": audits,
            "selected_sigma": [.2, .8],
        }],
    }


def test_event_case_stats_reports_selector_tradeoff():
    stats = V.event_case_stats(event(
        gamma=.3, scenario=10, step=7, nearest=.12,
        predictive_progress=.9, reference_progress=.2,
    ))
    assert stats["selector_disagreement"]
    assert np.isclose(stats["current_nearest_pedestrian_clearance"], .12)
    assert np.isclose(stats["H10_progress_gain"], .7)
    assert stats["predictive_sigma_rank"] == 1


def test_case_screen_enforces_thresholds_and_gamma_diversity():
    rows = [
        event(gamma=.1, scenario=1, step=1, nearest=.1,
              predictive_progress=.5, reference_progress=.2),
        event(gamma=.1, scenario=1, step=2, nearest=.1,
              predictive_progress=1.0, reference_progress=.2),
        event(gamma=.3, scenario=2, step=3, nearest=.2,
              predictive_progress=.7, reference_progress=.2),
        event(gamma=.5, scenario=3, step=4, nearest=.2,
              predictive_progress=.8, reference_progress=.2),
        event(gamma=1.0, scenario=4, step=5, nearest=.2,
              predictive_progress=.9, reference_progress=.2),
        # Each of these fails one declared criterion.
        event(gamma=.7, scenario=5, step=6, nearest=.8),
        event(gamma=.7, scenario=5, step=7, nearest=.2, positives=2),
        event(gamma=.7, scenario=5, step=8, nearest=.2,
              predictive_progress=.25, reference_progress=.2),
        event(gamma=.7, scenario=5, step=9, nearest=.2,
              predictive=0, reference=0),
        event(gamma=.7, scenario=5, step=10, nearest=.2,
              predictive_clearance=-.01),
    ]
    selected = V.select_case_events(rows, count=4)
    assert {stats["gamma"] for _, stats in selected} == {.1, .3, .5, 1.0}
    # The strongest gamma-.1 event wins that gamma's diversity slot.
    assert next(stats for _, stats in selected if stats["gamma"] == .1)["step"] == 2


def test_case_screen_fills_fifth_slot_by_global_rank():
    rows = [
        event(gamma=.1, scenario=1, step=1, nearest=.1,
              predictive_progress=.5, reference_progress=.2),
        event(gamma=.1, scenario=1, step=2, nearest=.1,
              predictive_progress=1.0, reference_progress=.2),
        event(gamma=.3, scenario=2, step=3, nearest=.2,
              predictive_progress=.7, reference_progress=.2),
        event(gamma=.5, scenario=3, step=4, nearest=.2,
              predictive_progress=.8, reference_progress=.2),
        event(gamma=1.0, scenario=4, step=5, nearest=.2,
              predictive_progress=.9, reference_progress=.2),
    ]
    selected = V.select_case_events(rows, count=5)
    assert len(selected) == 5
    assert {stats["step"] for _, stats in selected} == {1, 2, 3, 4, 5}


def test_case_screen_fails_closed_when_too_few_events():
    rows = [event(gamma=.1, scenario=1, step=1, nearest=.1)]
    try:
        V.select_case_events(rows, count=4)
    except RuntimeError as error:
        assert "only 1 events" in str(error)
    else:
        raise AssertionError("case selection must fail closed")
