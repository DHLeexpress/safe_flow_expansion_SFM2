from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
import pytest

import sfm_hp100_eval as E


def test_clipped_verifier_rollout_uses_shared_velocity_cap():
    state = np.array([0.0, 0.0, 1.95, 0.0], np.float32)
    controls = np.tile(np.array([2.0, 0.0], np.float32), (10, 1))
    positions = E.clipped_rollout_positions(state, controls)
    assert positions.shape == (11, 2)
    # First step uses v=1.95; every later step uses the capped v=2.0.
    np.testing.assert_allclose(positions[1, 0], 0.205, atol=1.0e-6)
    np.testing.assert_allclose(np.diff(positions[1:, 0]), 0.21, atol=1.0e-6)


def test_summary_uses_only_successes_for_clearance_and_time():
    rows = []
    for gamma in E.SS.GAMMAS:
        rows.extend([
            dict(
                gamma=gamma, success=True, collision=False, timeout=False,
                validity=0.75, successful_clearance=0.2, time_to_goal=9.0,
            ),
            dict(
                gamma=gamma, success=False, collision=True, timeout=False,
                validity=0.25, successful_clearance=None, time_to_goal=None,
            ),
        ])
    summary = E.summarize(rows)
    assert summary["pooled"]["SR"] == 0.5
    assert summary["pooled"]["CR"] == 0.5
    assert summary["pooled"]["Validity"] == 0.5
    assert summary["pooled"]["successful_clearance"] == pytest.approx(0.2)
    assert summary["pooled"]["successful_time_to_goal"] == 9.0


def test_id_gate_is_raw_unit_temperature_and_matched_id(monkeypatch):
    called = {}

    def fake_evaluate(policy, **kwargs):
        called.update(kwargs)
        cells = {
            str(gamma): {
                "SR": 0.8, "CR": 0.2, "timeout": 0.0, "Validity": 0.7,
                "successful_clearance": 0.1, "successful_time_to_goal": 8.0,
            }
            for gamma in E.SS.GAMMAS
        }
        return [], {"pooled": next(iter(cells.values())), "per_gamma": cells}

    monkeypatch.setattr(E, "evaluate", fake_evaluate)
    sentinel = object()
    result = E.id_raw_gate(
        object(), M=3, ep0=12000, device="cpu", seed=99,
        validity_executor=sentinel,
    )
    assert called["scene_profile"] == "matched_id"
    assert called["with_validity"] is True
    assert called["validity_executor"] is sentinel
    assert called["seed"] == result["noise_seed"] == 99
    assert result["temperature"] == 1.0
    assert result["NFE"] == 8
    assert result["pooled"]["Validity"] == 0.7
    assert set(result["per_gamma"]) == {str(gamma) for gamma in E.SS.GAMMAS}


def _episode_row(episode, *, steps=2):
    controls = np.zeros((steps, 2), np.float32)
    states = np.zeros((steps + 1, 4), np.float32)
    peds = np.full((steps, 1, 2), 20.0, np.float32)
    velocities = np.zeros_like(peds)
    return dict(
        episode=int(episode), gamma=0.5, status="timeout", success=False,
        collision=False, timeout=True, steps=int(steps), time_to_goal=None,
        min_clearance=1.0, successful_clearance=None,
        states=states, controls=controls, ped_xy=peds, ped_vel=velocities,
    )


def test_parallel_validity_is_ordered_and_identical_to_serial():
    rows = [_episode_row(7), _episode_row(3), _episode_row(9, steps=0)]
    serial = E.attach_validity(rows)
    with ProcessPoolExecutor(
        max_workers=2, mp_context=mp.get_context("spawn")
    ) as executor:
        parallel = E.attach_validity(rows, executor=executor)
    assert parallel == serial
    assert [row["episode"] for row in parallel] == [7, 3, 9]


def test_parallel_and_serial_validity_both_fail_closed():
    malformed = _episode_row(7, steps=1)
    malformed["controls"] = np.zeros((0, 2), np.float32)
    with pytest.raises(RuntimeError, match="executed HP100 window"):
        E.attach_validity([malformed])
    with ProcessPoolExecutor(
        max_workers=2, mp_context=mp.get_context("spawn")
    ) as executor:
        with pytest.raises(RuntimeError, match="executed HP100 window"):
            E.attach_validity([malformed], executor=executor)


def test_retained_proposals_keep_declared_ranks_for_zero_step_cells(monkeypatch):
    def initial_collision(episode, pedestrian_xy):
        episode.status = "collision"
        episode.minimum_clearance = -0.01
        return True

    monkeypatch.setattr(E, "_terminal_check", initial_collision)
    policy = type("Policy", (), {"d": 4})()
    rows = E.run_batched_raw(
        policy, scene_profile="matched_id", ep0=910_000, M=1,
        noise=E.noise_bank(M=1, d=4, seed=7), device="cpu",
        retain_proposals=True,
    )
    assert len(rows) == len(E.SS.GAMMAS)
    for row in rows:
        assert row["steps"] == 0
        assert row["states"].shape == (1, 4)
        assert row["controls"].shape == (0, 2)
        assert row["ped_xy"].shape == (0, 20, 2)
        assert row["ped_vel"].shape == (0, 20, 2)
        assert row["proposals"].shape == (0, E.H, 2)
