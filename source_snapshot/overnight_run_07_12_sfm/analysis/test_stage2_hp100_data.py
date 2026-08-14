import json

import numpy as np
import pytest
import torch

import sfm_hp100_dynamics as D
import stage2_hp100_data as S

_ORIGINAL_HP100_FRAME = S.HPF.hp100_frame


def test_hp100_expert_overrides_retreat_without_mutating_historical_comparator():
    assert S.EXPERT.demonstration_config().predict_gain == 0.25
    assert S.locked_expert_config()["predict_gain"] == 0.0
    assert S.HPF.PREDICT_GAIN == 0.0


class _FakePlanner:
    def __init__(self, *, accepted=1, violating=False):
        self.accepted = int(accepted)
        self.violating = bool(violating)
        self.calls = 0

    def plan(self, state, goal, obstacles, **kwargs):
        _, geometry = _ORIGINAL_HP100_FRAME(
            state[:2],
            obstacles,
            sensing=S.SS.R_SENSE,
            n_base=S.N_BASE,
            obstacle_velocities=kwargs["obstacle_velocities"],
            robot_velocity=state[2:4],
            predict_gain=S.locked_expert_config()["predict_gain"],
            predict_tau=S.HORIZON * D.DT,
            return_geometry=True,
        )
        polytope = tuple(geometry[key] for key in ("A", "b", "ref", "margins"))
        sequence = np.zeros((S.HORIZON, 2), np.float32)
        sequence[0, 0] = 0.2 if self.calls == 0 else -0.2
        if self.violating:
            sequence[:, 0] = 2.0
        self.calls += 1
        return torch.from_numpy(sequence[0]), {
            "polytope": polytope,
            "mean_sequence": sequence,
            "num_candidates": 2048,
            "num_accepted": self.accepted,
            "num_rejected": 2048 - self.accepted,
        }


def _planner_tuple_from_feature_geometry(geometry):
    A = torch.as_tensor(geometry["A"], dtype=torch.float32)
    b = torch.as_tensor(geometry["b"], dtype=torch.float32)
    ref = torch.as_tensor(geometry["ref"], dtype=torch.float32)
    margins = torch.clamp(b - A @ ref, min=1.0e-3)
    return tuple(
        value.detach().cpu().numpy()
        for value in (A, b, ref, margins)
    )


def test_planner_geometry_audit_uses_derived_float32_margin_envelope():
    center = np.array([3.434889, 3.678803], np.float32)
    obstacles = np.array([[3.64, 3.68, 0.2]], np.float32)
    frame, geometry = S.HPF.hp100_frame(
        center,
        obstacles,
        obstacle_velocities=np.array([[-2.0, 0.0]], np.float32),
        robot_velocity=np.array([2.0, 0.0], np.float32),
        return_geometry=True,
    )
    planner = _planner_tuple_from_feature_geometry(geometry)
    S._assert_feature_matches_planner(
        frame, geometry, planner, center, provenance="narrow-face-test"
    )

    bad_A = planner[0].copy()
    bad_A.flat[0] = np.nextafter(bad_A.flat[0], np.float32(np.inf))
    with pytest.raises(RuntimeError, match="mismatch at A"):
        S._assert_feature_matches_planner(
            frame, geometry, (bad_A, *planner[1:]), center,
            provenance="one-ulp-test",
        )

    bad_margins = planner[3].copy()
    bad_margins[0] += np.float32(1.0e-3)
    with pytest.raises(RuntimeError, match="roundoff envelope"):
        S._assert_feature_matches_planner(
            frame, geometry, (*planner[:3], bad_margins), center,
            provenance="bad-margin-test",
        )


def test_rollout_collects_current_certified_weighted_plan(monkeypatch):
    monkeypatch.setattr(S.SS, "make_humans", lambda *args, **kwargs: [object()] * S.N_PED)
    monkeypatch.setattr(
        S.SS,
        "collect_humans",
        lambda humans: (
            np.full((S.N_PED, 2), 20.0, np.float32),
            np.zeros((S.N_PED, 2), np.float32),
        ),
    )
    monkeypatch.setattr(S.SS, "advance_humans", lambda humans, state: None)
    hp_calls = []

    original_hp = S.HPF.hp100_frame

    def fresh_hp(robot_xy, obstacles, **kwargs):
        hp_calls.append((
            np.asarray(robot_xy).copy(), np.asarray(obstacles).copy(),
            kwargs["sensing"], kwargs["n_base"],
            np.asarray(kwargs["obstacle_velocities"]).copy(),
            np.asarray(kwargs["robot_velocity"]).copy(),
            kwargs["predict_gain"],
        ))
        return original_hp(robot_xy, obstacles, **kwargs)

    monkeypatch.setattr(S.HPF, "hp100_frame", fresh_hp)
    records, summary = S.rollout_episode(
        3, 0.5, device="cpu", planner=_FakePlanner(), T_max=2
    )
    assert summary["timeout"] and len(records) == 2
    assert len(hp_calls) == 2
    assert all(call[3] == 16 for call in hp_calls)
    assert all(call[4].shape == (S.N_PED, 2) for call in hp_calls)
    assert all(call[6] == 0.0 for call in hp_calls)
    assert records[0]["hp"].shape == (32, 100)
    assert records[0]["hp"].dtype == np.float32
    np.testing.assert_allclose(records[0]["executed_action"], [0.2, 0.0])
    np.testing.assert_allclose(records[1]["state"][2:], [0.02, 0.0])
    assert records[0]["U"].shape == (10, 2)
    np.testing.assert_allclose(records[0]["U"][0], [0.2, 0.0])
    np.testing.assert_allclose(records[0]["U"][1:], 0.0)
    assert records[0]["target_eligible"]
    assert records[0]["target_reason_code"] == S.TARGET_ELIGIBLE
    assert np.max(np.abs(records[0]["U"])) <= D.U_MAX


def test_all_rejected_weighted_plan_remains_ledger_only(monkeypatch):
    monkeypatch.setattr(S.SS, "make_humans", lambda *args, **kwargs: [object()] * S.N_PED)
    monkeypatch.setattr(
        S.SS, "collect_humans",
        lambda humans: (
            np.full((S.N_PED, 2), 20.0, np.float32),
            np.zeros((S.N_PED, 2), np.float32),
        ),
    )
    monkeypatch.setattr(S.SS, "advance_humans", lambda humans, state: None)
    records, _ = S.rollout_episode(
        3, 0.5, device="cpu", planner=_FakePlanner(accepted=0), T_max=1
    )
    assert len(records) == 1
    assert not records[0]["target_eligible"]
    assert records[0]["target_reason_code"] == S.TARGET_ALL_REJECTED


def test_accepted_weighted_plan_that_fails_recheck_is_excluded(monkeypatch):
    monkeypatch.setattr(S.SS, "make_humans", lambda *args, **kwargs: [object()] * S.N_PED)
    monkeypatch.setattr(
        S.SS, "collect_humans",
        lambda humans: (
            np.full((S.N_PED, 2), 20.0, np.float32),
            np.zeros((S.N_PED, 2), np.float32),
        ),
    )
    monkeypatch.setattr(S.SS, "advance_humans", lambda humans, state: None)
    records, _ = S.rollout_episode(
        3, 0.1, device="cpu", planner=_FakePlanner(accepted=1, violating=True), T_max=1
    )
    assert len(records) == 1
    assert not records[0]["target_eligible"]
    assert records[0]["target_reason_code"] == S.TARGET_WEIGHTED_H10_FAILED
    assert records[0]["plan_first_violation"] >= 1


def _record(episode=0, step=0):
    return dict(
        hp=np.zeros((32, 100), np.float32),
        low5=np.zeros(5, np.float32),
        hist=np.zeros((16, 2), np.float32),
        U=np.zeros((10, 2), np.float32),
        episode=np.int64(episode),
        step=np.int64(step),
        state=np.zeros(4, np.float32),
        ped_xy=np.zeros((20, 2), np.float32),
        ped_vel=np.zeros((20, 2), np.float32),
        executed_action=np.zeros(2, np.float32),
        target_eligible=True,
        target_reason_code=np.int8(S.TARGET_ELIGIBLE),
        plan_candidate_count=np.int32(2048),
        plan_accepted_count=np.int32(1),
        plan_rejected_count=np.int32(2047),
        plan_weighted_h=np.ones(11, np.float32),
        plan_first_violation=np.int16(-1),
        action_mean_max_abs_error=np.float32(0.0),
    )


def test_small_cpu_dataset_smoke_writes_auditable_manifest(tmp_path):
    def fake_rollout(episode, gamma, **kwargs):
        success = episode == 1
        return [_record(episode, 0)], dict(
            episode=episode, gamma=gamma, success=success, collision=not success,
            timeout=False, steps=1, min_clearance=(1.0 if success else -0.1),
        )

    manifest = S.generate_dataset(
        tmp_path,
        episode_start=0,
        successes_per_gamma=1,
        max_attempts_per_gamma=2,
        gammas=(0.5,),
        device="cpu",
        T_max=2,
        rollout_fn=fake_rollout,
    )
    assert manifest["status"] == "HP100_ID_DATASET_COMPLETE"
    assert not manifest["canonical_full_run"]
    assert manifest["dynamics"]["action_cap"]["maximum"] == 2.0
    assert manifest["dynamics"]["velocity_cap"]["maximum"] == 2.0
    assert manifest["feature"]["nominal_polytope_n_base"] == 16
    assert manifest["feature"]["velocity_aware"] is False
    assert manifest["feature"]["current_position_tangent"] is True
    assert manifest["feature"]["predict_gain"] == 0.0
    assert manifest["expert"]["name"] == S.HP100_EXPERT_NAME
    assert manifest["feature"]["predict_tau"] == 1.0
    assert manifest["total_successful_lineages"] == 1
    assert manifest["total_eligible_windows"] == 1
    assert manifest["target_contract"] == S.target_contract()
    assert manifest["expert"]["supervised_target"] == S.SUPERVISED_TARGET
    assert "old-grid upsample" in manifest["feature"]["construction"]
    assert manifest["files"][0]["successful_episodes"] == [1]
    assert manifest["files"][0]["rejected_episodes"] == [0]
    assert manifest["files"][0]["episode_range"] == [0, 2]
    progress_path = tmp_path / manifest["files"][0]["progress_file"]
    progress = json.loads(progress_path.read_text())
    assert progress["status"] == "HP100_GAMMA_COLLECTION_COMPLETE"
    assert progress["accepted_successes"] == progress["target_successes"] == 1
    assert progress["attempted_episodes"] == 2
    assert manifest["files"][0]["progress_sha256"] == S.sha256_file(progress_path)
    data_path = tmp_path / "sfm_hp100_windows_g0.5.pt"
    payload = torch.load(data_path, map_location="cpu", weights_only=False)
    assert payload["hp"].shape == (1, 32, 100)
    assert payload["state"].shape == (1, 4)
    assert payload["ped_xy"].shape == (1, 20, 2)
    assert payload["ped_vel"].shape == (1, 20, 2)
    assert payload["target_eligible"].tolist() == [True]
    assert manifest["files"][0]["eligible_windows"] == 1
    assert manifest["files"][0]["sha256"] == S.sha256_file(data_path)
    assert manifest["runtime"]["requested_device"] == "cpu"
    assert {"human_agent", "human_advance", "nominal_polytope", "safemppi_barrier"} <= set(
        manifest["source_hashes"]
    )
    disk_manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert disk_manifest["source_hashes"]["dynamics"]["sha256"]


def test_success_quota_fails_closed_at_attempt_cap(tmp_path):
    def rejected(episode, gamma, **kwargs):
        return [_record(episode, 0)], dict(
            episode=episode, gamma=gamma, success=False, collision=True,
            timeout=False, steps=1, min_clearance=-0.1,
        )

    with pytest.raises(RuntimeError, match="0/1 eligible successful trajectories in 2 attempts"):
        S.generate_dataset(
            tmp_path,
            successes_per_gamma=1,
            max_attempts_per_gamma=2,
            gammas=(0.5,),
            T_max=2,
            rollout_fn=rejected,
        )


def test_completion_provenance_fails_closed_on_source_change(monkeypatch):
    initial_git = {"root": "/repo", "head": "abc", "clean": True, "status": ""}
    initial_hashes = {"generator": {"path": "/repo/g.py", "sha256": "old"}}
    monkeypatch.setattr(S, "_git_provenance", lambda: dict(initial_git))
    monkeypatch.setattr(
        S, "_source_hashes",
        lambda: {"generator": {"path": "/repo/g.py", "sha256": "new"}},
    )
    with pytest.raises(RuntimeError, match="source file hashes changed"):
        S._assert_provenance_unchanged(initial_git, initial_hashes)


def test_device_assignment_is_round_robin_in_one_parent_manifest(tmp_path):
    seen = []

    def fake_rollout(episode, gamma, **kwargs):
        seen.append((float(gamma), kwargs["device"]))
        return [_record(episode, 0)], dict(
            episode=episode, gamma=gamma, success=True, collision=False,
            timeout=False, steps=1, min_clearance=1.0,
        )

    manifest = S.generate_dataset(
        tmp_path, successes_per_gamma=1, max_attempts_per_gamma=1,
        gammas=(0.1, 0.2, 0.3), devices=("dev0", "dev1"),
        T_max=1, rollout_fn=fake_rollout,
    )
    assert seen == [(0.1, "dev0"), (0.2, "dev1"), (0.3, "dev0")]
    assert manifest["parallelism"]["gamma_device_map"] == {
        "0.1": "dev0", "0.2": "dev1", "0.3": "dev0",
    }
