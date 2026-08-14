import numpy as np
import torch

import sfm_hp100_expert_mechanism_viz as V
import stage2_hp100_data as DATA


def test_contraction_audit_marks_the_first_violating_state():
    polytope = dict(
        A=np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]], np.float32),
        b=np.ones(4, np.float32), margins=np.ones(4, np.float32),
    )
    states = np.zeros((2, 4, 4), np.float32)
    states[0, :, 0] = [0.0, 0.1, 0.2, 0.3]
    states[1, :, 0] = [0.0, 0.1, 0.5, 0.6]
    audit = V.contraction_audit(states, polytope, gamma=.2)
    np.testing.assert_array_equal(audit["feasible"], [True, False])
    np.testing.assert_array_equal(audit["first_violation"], [-1, 2])


def test_proportional_subset_is_deterministic_and_keeps_both_classes():
    feasible = np.array([True] * 2 + [False] * 8)
    first = V.proportional_subset(feasible, 5)
    second = V.proportional_subset(feasible, 5)
    np.testing.assert_array_equal(first, second)
    assert len(first) == 5
    assert feasible[first].any()
    assert (~feasible[first]).any()


def test_return_rollouts_exposes_exact_weights_and_rejection_locations():
    config = DATA.locked_expert_config()
    config.update(num_samples=16, debug_max_rollouts=16, horizon=3)
    planner = DATA.CappedSafeMPPIAdapter(**config)
    state = torch.zeros(4)
    goal = torch.tensor([6.0, 6.0])
    obstacles = torch.tensor([[1.2, .4, .2]], dtype=torch.float32)
    velocities = torch.zeros((1, 2), dtype=torch.float32)
    _, info = planner.plan(
        state, goal, obstacles, gamma=.5, obstacle_velocities=velocities,
        seed=17, return_rollouts=True,
    )
    debug = info["debug_rollouts"]
    np.testing.assert_array_equal(debug["sample_indices"], np.arange(16))
    assert debug["states"].shape == (16, 4, 4)
    assert debug["controls"].shape == (16, 3, 2)
    assert debug["weights"].shape == (16,)
    assert debug["first_violation_step"].shape == (16,)
    assert np.isclose(debug["weights"].sum(), 1.0)
    np.testing.assert_array_equal(
        np.asarray(debug["first_violation_step"]) < 0,
        np.asarray(debug["feasible"]),
    )
    assert info["num_candidates"] == 16
    assert info["num_accepted"] + info["num_rejected"] == 16
    assert info["selection_semantics"] in {
        "accepted_temperature_weighted_mean",
        "all_rejected_safest_fallback_weighted_mean",
    }


def test_return_rollouts_does_not_change_the_planner_output():
    config = DATA.locked_expert_config()
    config.update(num_samples=32, debug_max_rollouts=32, horizon=3)
    inputs = dict(
        state=torch.zeros(4), goal=torch.tensor([6.0, 6.0]),
        obstacles=torch.tensor([[1.2, .4, .2]], dtype=torch.float32),
        gamma=.5, obstacle_velocities=torch.zeros((1, 2)), seed=23,
    )
    plain = DATA.CappedSafeMPPIAdapter(**config)
    traced = DATA.CappedSafeMPPIAdapter(**config)
    action_plain, info_plain = plain.plan(**inputs, return_rollouts=False)
    action_traced, info_traced = traced.plan(**inputs, return_rollouts=True)
    torch.testing.assert_close(action_plain, action_traced, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(info_plain["mean_sequence"], info_traced["mean_sequence"])


def test_all_rejected_trace_is_explicitly_a_fallback():
    config = DATA.locked_expert_config()
    config.update(num_samples=32, debug_max_rollouts=32, horizon=3)
    planner = DATA.CappedSafeMPPIAdapter(**config)
    _, info = planner.plan(
        torch.tensor([0.0, 0.0, 2.0, 0.0]), torch.tensor([6.0, 0.0]),
        torch.tensor([[.25, 0.0, .2]], dtype=torch.float32), gamma=1.0,
        obstacle_velocities=torch.zeros((1, 2)), seed=29, return_rollouts=True,
    )
    assert info["num_accepted"] == 0
    assert info["num_rejected"] == 32
    assert info["selection_semantics"] == "all_rejected_safest_fallback_weighted_mean"
    debug = info["debug_rollouts"]
    assert np.all(np.asarray(debug["first_violation_step"]) == 1)
    assert np.isclose(np.asarray(debug["weights"]).sum(), 1.0)
