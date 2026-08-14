import numpy as np

import sfm_metrics2 as M


def test_label_is_only_task_collision_and_moving_certificate():
    # A stationary, non-progressing plan is still y=1 when the three label clauses pass.
    result = M.verify_query(
        np.zeros(4), np.zeros((10, 2)), np.zeros((0, 2)), np.zeros((0, 2)), .5
    )
    assert result["resolved"] and result["y"] == 1
    assert result["taskspace"] and result["collision_free"] and result["certificate"]
    assert "progress" not in result and "cost" not in result


def test_predicted_goal_crossing_still_certifies_full_h():
    state = np.array([5.4, 6.0, 1.0, 0.0], np.float32)
    result = M.verify_query(
        state, np.zeros((10, 2)), np.zeros((0, 2)), np.zeros((0, 2)), .5
    )
    assert np.min(np.linalg.norm(result["segment"] - M.SS.GOAL[None], axis=1)) < .5
    assert result["resolved"] and result["terminal_step"] == 10
    assert result["full_h"] and result["y"] == 1 and result["train_eligible"]


def test_post_goal_tail_violation_rejects_the_full_window():
    state = np.array([5.7, 6.0, 2.0, 0.0], np.float32)
    result = M.verify_query(
        state, np.zeros((10, 2)), np.zeros((0, 2)), np.zeros((0, 2)), .5
    )
    assert np.min(np.linalg.norm(result["segment"] - M.SS.GOAL[None], axis=1)) < .5
    assert result["resolved"] and result["terminal_step"] == 10 and result["full_h"]
    assert result["y"] == 0 and not result["taskspace"] and not result["train_eligible"]


def test_worker_contract_has_no_legacy_theta_grid_argument():
    payload = (3, 7, np.zeros(4), np.zeros((10, 2)),
               np.zeros((0, 2)), np.zeros((0, 2)), .5)
    context, candidate, result = M.verify_in_worker(payload)
    assert (context, candidate) == (3, 7)
    assert result["diagnostics"]["solver"] == "exact_2d_angular_interval_socp"
    assert result["diagnostics"]["K_artificial"] == 16
