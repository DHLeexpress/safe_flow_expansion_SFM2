"""Gate 1 (admission) and gate 2 (dynamics/verifier equivalence) tests."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import grid_policy_sfm_hp100 as GPS
import sfm_hp100_ball_adapter as PORT
import sfm_hp100_eval as EVAL
import sfm_hp100_expansion_archive as ARCH
import sfm_hp100_expansion_funnel as FUNNEL
import sfm_scene as SS


REPO = Path(__file__).resolve().parents[3]


def test_checkpoint_sha_admission_fails_closed(tmp_path):
    junk = tmp_path / "junk.pt"
    junk.write_bytes(b"not a checkpoint")
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        ARCH._load_frozen(str(junk), "0" * 64, "cpu")


def test_strict_config_mismatch_refuses_to_load(tmp_path):
    policy = GPS.build_sfm_hp100_policy()
    payload = {"state_dict": policy.state_dict(), "config": policy.config()}
    payload["config"]["width"] = 128
    path = tmp_path / "drifted.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="strict SFM HP100 checkpoint"):
        GPS.load_sfm_hp100_policy(str(path), device="cpu")


def test_cached_r0_m50_provenance_matches_locked_baseline():
    ood = json.loads(
        (REPO / "provenance/hp100_pretrain_20260802/ood_m50.json").read_text()
    )
    matched = json.loads(
        (REPO / "provenance/hp100_pretrain_20260802/id_m50.json").read_text()
    )
    for payload in (ood, matched):
        assert payload["status"] == "SFM_HP100_RAW_EVAL_COMPLETE"
        assert float(payload["temperature"]) == 1.0
        assert int(payload["NFE"]) == 8
    assert np.isclose(ood["summary"]["pooled"]["SR"], 0.56)
    assert np.isclose(ood["summary"]["pooled"]["CR"], 0.4371, atol=5.0e-5)
    assert np.isclose(matched["summary"]["pooled"]["SR"], 0.9571, atol=5.0e-5)
    assert np.isclose(matched["summary"]["pooled"]["Validity"], 0.7906, atol=5.0e-5)


def test_funnel_refuses_trace_and_archive_payloads(tmp_path):
    trace = tmp_path / "trace.pt"
    torch.save({"status": "SFM_HP100_PREDICTIVE_EXECUTION_TRACE", "events": []}, trace)
    with pytest.raises(ValueError, match="refuses acquisition"):
        FUNNEL.assert_evaluable_checkpoint(trace)
    archive = tmp_path / "archive.pt"
    torch.save({"status": ARCH.ARCHIVE_STATUS, "rows": []}, archive)
    with pytest.raises(ValueError, match="refuses acquisition"):
        FUNNEL.assert_evaluable_checkpoint(archive)
    samples = tmp_path / "samples.pt"
    torch.save({"samples": []}, samples)
    with pytest.raises(ValueError, match="refuses acquisition"):
        FUNNEL.assert_evaluable_checkpoint(samples)


def test_funnel_accepts_the_strict_checkpoint_schema(tmp_path):
    policy = GPS.build_sfm_hp100_policy()
    path = tmp_path / "strict.pt"
    GPS.save_sfm_hp100_policy(policy, str(path))
    payload = FUNNEL.assert_evaluable_checkpoint(path)
    assert set(payload) >= {"state_dict", "config"}


def test_clipped_task_dynamics_match_the_raw_evaluator_rollout():
    rng = np.random.default_rng(7)
    state = rng.normal(size=4).astype(np.float32)
    controls = rng.normal(scale=3.0, size=(10, 2)).astype(np.float32)
    task_states = PORT.clipped_plan_states(state, controls)
    eval_positions = EVAL.clipped_rollout_positions(state, controls)
    assert np.array_equal(task_states[:, :2], eval_positions)


def _packed_context(robot, ped_xy, ped_vel):
    token = np.zeros(GPS.LOW_TOKEN + GPS.VISUAL_TOKEN, np.float32)
    return torch.from_numpy(np.concatenate([
        token, np.asarray(robot, np.float32),
        np.asarray(ped_xy, np.float32).reshape(-1),
        np.asarray(ped_vel, np.float32).reshape(-1),
    ]))


def test_task_verifier_matches_raw_evaluator_verifier_on_full_windows():
    profile = SS.scene_profile("double_density_velocity_ood")
    n_ped = int(profile["n_ped"])
    task = PORT.SFMHP100ExpansionTask(fixed_scenario_id=7)
    rng = np.random.default_rng(11)
    for trial in range(4):
        robot = np.array([1.0, 1.0, 0.0, 0.0], np.float32)
        # Keep the current position strictly outside every pedestrian disc so
        # the nominal current-tangent polytope stays feasible.
        ped_xy = rng.uniform(2.5, 6.0, size=(n_ped, 2)).astype(np.float32)
        ped_vel = rng.uniform(-1.0, 1.0, size=(n_ped, 2)).astype(np.float32)
        controls = rng.normal(scale=1.0, size=(10, 2)).astype(np.float32)
        context = _packed_context(robot, ped_xy, ped_vel)
        result, sidecar = task._verify_one(
            context, torch.from_numpy(controls), 0.3,
        )
        reference = EVAL.verify_executed_window(
            robot, controls, ped_xy, ped_vel, 0.3,
        )
        assert reference["resolved"]
        assert bool(result.valid) == bool(reference["y"])
        assert sidecar["result"]["taskspace"] == reference["taskspace"]
        assert sidecar["result"]["collision_free"] == reference["collision_free"]
        assert sidecar["result"]["certificate"] == reference["certificate"]


def test_task_verifier_never_promotes_short_windows():
    profile = SS.scene_profile("double_density_velocity_ood")
    n_ped = int(profile["n_ped"])
    task = PORT.SFMHP100ExpansionTask(fixed_scenario_id=7)
    robot = np.array([1.0, 1.0, 0.0, 0.0], np.float32)
    ped_xy = np.full((n_ped, 2), 5.5, np.float32)
    ped_vel = np.zeros((n_ped, 2), np.float32)
    context = _packed_context(robot, ped_xy, ped_vel)
    short = torch.zeros(4, 2)
    result, _ = task._verify_one(context, short, 0.3)
    assert not result.valid
    reference = EVAL.verify_executed_window(
        robot, short.numpy(), ped_xy, ped_vel, 0.3,
    )
    # The raw evaluator accepts executed suffixes shorter than H; the
    # expansion task deliberately does not.  Gate 2 compares them on H=10 only.
    assert reference["resolved"] and reference["window_horizon"] == 4
