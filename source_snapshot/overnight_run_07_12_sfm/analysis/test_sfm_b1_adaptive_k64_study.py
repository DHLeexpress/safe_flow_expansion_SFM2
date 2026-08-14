import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import sfm_b1_adaptive_k64_study as S
import sfm_b1_curve_eval as CE
import sfm_b1_expand as X
import sfm_b1_store as BS
import sfm_protocol as SP


def _summary(cr):
    per_gamma = {}
    for gamma in SP.GAMMAS:
        per_gamma[str(gamma)] = dict(
            CR=float(cr), V_safe=.4, timeout=0.0, SR=1.0 - float(cr),
            successful_clearance=dict(mean=.2),
            successful_time_to_goal=dict(mean=8.0),
        )
    return dict(
        per_gamma=per_gamma,
        pooled=dict(
            CR=float(cr), V_safe=.4, timeout=0.0, SR=1.0 - float(cr),
            successful_clearance=dict(mean=.2),
            successful_time_to_goal=dict(mean=8.0),
        ),
    )


def test_round_selection_is_canonical_and_prefers_earliest_exact_tie():
    canonical = CE.temperature_vector(1.0)
    rows = [
        dict(round=round_i, summary=_summary(.3), temperature_by_gamma=canonical)
        for round_i in range(S.ROUNDS + 1)
    ]
    rows[2]["summary"] = _summary(.1)
    rows[4]["summary"] = copy.deepcopy(rows[2]["summary"])
    assert S.select_round(rows)["round"] == 2


def _confirmation(round_i):
    return dict(
        status="SFM_B1_SINGLE_CONFIRMATION_COMPLETE", round=round_i,
        temperature_by_gamma=CE.temperature_vector(1.0),
        bank=dict(ep0=SP.ADAPTIVE_CONFIRM_EP0, M=100, sha256="same-bank"),
    )


def test_paired_confirmation_requires_exact_same_canonical_m100_bank():
    baseline, selected = _confirmation(0), _confirmation(3)
    assert S.validate_paired_confirmations(
        baseline, selected, expected_selected_round=3,
    )
    selected["bank"] = dict(ep0=SP.ADAPTIVE_CONFIRM_EP0, M=100, sha256="different")
    with pytest.raises(RuntimeError, match="not paired"):
        S.validate_paired_confirmations(
            baseline, selected, expected_selected_round=3,
        )


def test_paired_confirmation_rejects_temperature_tuning():
    baseline, selected = _confirmation(0), _confirmation(1)
    selected["temperature_by_gamma"][str(SP.GAMMAS[0])] = .95
    with pytest.raises(RuntimeError, match="not canonical"):
        S.validate_paired_confirmations(
            baseline, selected, expected_selected_round=1,
        )


def test_adaptive_gather_stops_after_first_admissible_batch_and_stores_queried_only(monkeypatch):
    class _GP:
        @staticmethod
        def acquisition_sigma_batched(features):
            return torch.ones(features.shape[:2])

        @staticmethod
        def sequential_acquire_batched(features, steps, beta, generator=None):
            order = list(range(int(steps)))
            trace = [dict(chosen_sigma=.8, ess_norm=.5) for _ in order]
            return [order], [trace]

    class _Executor:
        @staticmethod
        def map(function, tasks):
            return [function(task) for task in tasks]

    cfg = SimpleNamespace(
        acquisition_mode="adaptive_k64", T=1, H=10, K=8, B=4,
        max_queries=8, nfe=1, temp=1.0, phi_s=.9, selector="margin",
    )
    prepared = dict(
        state=np.zeros(4, np.float32), hp10=torch.zeros(10, 16, 12),
        low=torch.zeros(5), hist=torch.zeros(16, 2),
        ped_xy=np.zeros((1, 2), np.float32), ped_vel=np.zeros((1, 2), np.float32),
    )
    replica = SimpleNamespace(
        alive=True, prepared=prepared, scenario_id=1, gamma=.5,
        humans=[], state=np.zeros(4, np.float32), states=[np.zeros(4, np.float32)],
        controls=[], peds=[], minimum_clearance=1.0, status=None,
    )
    windows = torch.zeros(1, cfg.K, cfg.H, 2)
    monkeypatch.setattr(X, "policy_sha256", lambda policy: "frozen")
    monkeypatch.setattr(X, "_stack_prepared", lambda replicas, device: (
        replicas, {"hp10": None, "low": None, "hist": None},
    ))
    monkeypatch.setattr(X.BE, "generate_windows", lambda *args, **kwargs: windows)
    monkeypatch.setattr(X, "_features", lambda *args, **kwargs: torch.zeros(1, cfg.K, 2))
    monkeypatch.setattr(X.SM, "predict_pedestrians", lambda *args, **kwargs: np.zeros((10, 1, 2)))
    monkeypatch.setattr(X.SM, "rollout_positions", lambda state, controls: np.zeros((11, 2)))
    monkeypatch.setattr(X.BE, "classify_candidate", lambda *args, **kwargs: "yield")

    def verify(task):
        context, candidate = task[:2]
        positive = int(candidate) == 5
        result = dict(
            resolved=True, y=int(positive), full_h=True, terminal_step=10,
            taskspace=True, collision_free=positive, certificate=positive,
            train_eligible=positive, segment=np.zeros((11, 2), np.float32),
            pedestrian_prediction=np.zeros((10, 1, 2), np.float32), diagnostics={},
        )
        return context, candidate, result

    monkeypatch.setattr(X.SM, "verify_in_worker", verify)

    def select(rows, **kwargs):
        positive = [row for row in rows if row["result"]["y"] == 1]
        if not positive:
            return None
        positive[0].update(hp_margin=.1, step_progress=.01)
        return positive[0]

    monkeypatch.setattr(X.BC, "select_admissible", select)

    def advance(value, action):
        value.controls.append(np.asarray(action))
        value.status = "success"
        value.alive = False

    monkeypatch.setattr(X, "_advance", advance)
    shard = BS.RoundShard(1)
    result = X.gather_macro_round_adaptive(
        object(), object(), _GP(), .1, [replica], cfg, shard, "cpu",
        _Executor(), None, record_all_traces=True,
    )
    assert len(shard.D) == 8
    assert {row["candidate_id"] for row in shard.D} == set(range(8))
    assert shard.queries[5]["executed"]
    assert result["adaptive_acquisition"]["outcomes"] == {"rescued_5_16": 1}
    assert result["adaptive_acquisition"]["realized_queries"]["mean"] == 8.0
    assert result["realized_ess_over_remaining"] == pytest.approx(.5)
    assert result["adaptive_acquisition"]["ess_by_rank"] == {
        "1_4": pytest.approx(.5), "5_16": pytest.approx(.5),
        "17_32": None, "33_64": None,
    }
    assert result["sigma"]["base_uplift"] == pytest.approx(0.0)
    assert result["sigma"]["uplift"] == pytest.approx(0.0)
    assert result["sigma"]["pending_conditioned_uplift"] == pytest.approx(-.2)
    assert "pre-acquisition posterior" in result["sigma"]["comparison_note"]

    def verify_with_error(task):
        context, candidate = task[:2]
        if int(candidate) == 0:
            return context, candidate, dict(resolved=False, error="solver failed")
        return context, candidate, dict(
            resolved=True, y=0, full_h=True, terminal_step=10,
            taskspace=True, collision_free=False, certificate=False,
            train_eligible=False, segment=np.zeros((11, 2), np.float32),
            pedestrian_prediction=np.zeros((10, 1, 2), np.float32), diagnostics={},
        )

    monkeypatch.setattr(X.SM, "verify_in_worker", verify_with_error)
    failed = SimpleNamespace(
        alive=True, prepared=prepared, scenario_id=2, gamma=.5,
        humans=[], state=np.zeros(4, np.float32), states=[np.zeros(4, np.float32)],
        controls=[], peds=[], minimum_clearance=1.0, status=None,
    )
    failed_shard = BS.RoundShard(1)
    failed_result = X.gather_macro_round_adaptive(
        object(), object(), _GP(), .1, [failed], cfg, failed_shard, "cpu",
        _Executor(), None, record_all_traces=True,
    )
    assert failed.status == "verifier_error"
    assert len(failed_shard.D) == 3
    assert failed_result["adaptive_acquisition"]["realized_queries"]["mean"] == 4.0
    assert failed_result["adaptive_acquisition"]["outcomes"] == {
        "verifier_error_fail_closed": 1
    }
    assert failed_result["outcomes"][0]["verifier_error"]

    def verify_with_error_and_positive(task):
        context, candidate = task[:2]
        if int(candidate) == 0:
            return context, candidate, dict(resolved=False, error="solver failed")
        positive = int(candidate) == 1
        return context, candidate, dict(
            resolved=True, y=int(positive), full_h=True, terminal_step=10,
            taskspace=True, collision_free=positive, certificate=positive,
            train_eligible=positive, segment=np.zeros((11, 2), np.float32),
            pedestrian_prediction=np.zeros((10, 1, 2), np.float32), diagnostics={},
        )

    monkeypatch.setattr(X.SM, "verify_in_worker", verify_with_error_and_positive)
    mixed = SimpleNamespace(
        alive=True, prepared=prepared, scenario_id=3, gamma=.5,
        humans=[], state=np.zeros(4, np.float32), states=[np.zeros(4, np.float32)],
        controls=[], peds=[], minimum_clearance=1.0, status=None,
    )
    mixed_shard = BS.RoundShard(1)
    mixed_result = X.gather_macro_round_adaptive(
        object(), object(), _GP(), .1, [mixed], cfg, mixed_shard, "cpu",
        _Executor(), None, record_all_traces=True,
    )
    assert mixed.status == "verifier_error"
    assert not any(row["executed"] for row in mixed_shard.D)
    assert mixed_result["adaptive_acquisition"]["outcomes"] == {
        "verifier_error_fail_closed": 1
    }
