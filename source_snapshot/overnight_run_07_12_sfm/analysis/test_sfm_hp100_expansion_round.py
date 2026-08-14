"""Acquisition health gate, fallback diagnosis, and exact resume state."""
import argparse
import contextlib
import json

import numpy as np
import pytest
import torch

import grid_policy_sfm_hp100 as GPS
import sfm_hp100_ball_adapter as PORT
import sfm_hp100_expansion_round as ROUND
import sfm_hp100_expansion_update as UPD


CTX = GPS.LOW_TOKEN + GPS.VISUAL_TOKEN
GAMMAS = (0.1, 1.0)


def _adapter(seed=0):
    torch.manual_seed(seed)
    return PORT.HP100ExpansionPolicy(GPS.build_sfm_hp100_policy()).eval()


def _row(seed, *, gamma=0.1, role="positive", negative_reason=None):
    generator = torch.Generator().manual_seed(seed)
    return {
        "role": role,
        "gamma": gamma, "replica": 0, "lineage": f"g{gamma:g}:rep00",
        "scenario_id": 860_000 + seed, "step": seed, "attempt": 0,
        "executed": role == "positive",
        "negative_reason": negative_reason,
        "context": torch.randn(CTX + 4, generator=generator),
        "candidate": torch.randn(10, 2, generator=generator),
        "flow_base": torch.randn(10, 2, generator=generator),
        "verification": {"valid": role == "positive"},
        "prediction_audit": {},
        "positive_first16": 8, "positive_B32": 20, "base_std": 1.0,
        "attempts_used": 1, "beta": 2.0, "marginal_ESS_over_K": 0.1,
        "selected_sigma": 0.5,
    }


def _healthy_rows():
    rows = [
        _row(seed, gamma=gamma)
        for gamma in GAMMAS for seed in range(10)
    ]
    rows.append(_row(
        99, gamma=0.1, role="negative", negative_reason="all_negative_nvp",
    ))
    return rows


def _summaries(*, contexts=40, attempts=60, nvp=1, success=3):
    return [{
        "summary": {
            "contexts": contexts,
            "retried_context_fraction": 0.1,
            "outcome_counts": {"success": success, "nvp": nvp},
            "pooled": {"attempts": attempts},
            "per_gamma": {},
        },
    }]


def test_health_gate_passes_on_a_healthy_archive():
    stats = ROUND.acquisition_statistics(_healthy_rows(), _summaries())
    health = ROUND.evaluate_health_gate(stats, gammas=GAMMAS)
    assert health["passed"]
    assert stats["dplus_per_gamma"] == {"0.1": 10, "1": 10}
    assert stats["uncertainty_tilt_ratio"] == pytest.approx(
        (20 / 32) / (8 / 16),
    )
    assert stats["mean_attempts"] == pytest.approx(1.5)
    assert stats["nvp_lineage_fraction"] == pytest.approx(0.25)


def test_health_gate_fails_on_degenerate_archives():
    # A gamma with zero positives fails the absolute floor.
    rows = [row for row in _healthy_rows() if float(row["gamma"]) != 1.0]
    stats = ROUND.acquisition_statistics(rows, _summaries())
    health = ROUND.evaluate_health_gate(stats, gammas=GAMMAS)
    assert not health["passed"]
    failed = {
        row["criterion"] for row in health["criteria"] if not row["passed"]
    }
    assert "min_dplus_gamma_1" in failed
    # An NVP-heavy round fails the lineage and zero-positive criteria.
    stats = ROUND.acquisition_statistics(
        _healthy_rows(), _summaries(contexts=10, nvp=8, success=1,
                                    attempts=200),
    )
    health = ROUND.evaluate_health_gate(stats, gammas=GAMMAS)
    failed = {
        row["criterion"] for row in health["criteria"] if not row["passed"]
    }
    assert "max_nvp_lineage_fraction" in failed
    assert "max_zero_positive_context_fraction" in failed
    assert "max_mean_attempts" in failed
    # Collapse against the round-1 baseline fails the relative floor even
    # when the absolute floor holds.
    shrunk = [
        row for index, row in enumerate(_healthy_rows())
        if row["role"] == "negative" or index % 2 == 0
    ]
    stats = ROUND.acquisition_statistics(shrunk, _summaries())
    health = ROUND.evaluate_health_gate(
        stats, gammas=GAMMAS,
        baseline_dplus_per_gamma={"0.1": 40, "1": 40},
    )
    failed = {
        row["criterion"] for row in health["criteria"] if not row["passed"]
    }
    assert "min_dplus_baseline_fraction_gamma_0.1" in failed


def test_health_gate_treats_missing_statistics_as_unevaluable():
    stats = ROUND.acquisition_statistics(_healthy_rows(), [])
    assert stats["contexts"] is None
    health = ROUND.evaluate_health_gate(stats, gammas=GAMMAS)
    assert health["passed"]
    by_name = {row["criterion"]: row for row in health["criteria"]}
    assert not by_name["max_mean_attempts"]["evaluable"]


def test_fallback_diagnosis_schema(tmp_path):
    adapter = _adapter(seed=1)
    reference = _adapter(seed=1)
    rows = [_row(
        7, gamma=0.1, role="negative", negative_reason="all_negative_nvp",
    )]
    stats = ROUND.acquisition_statistics(rows, _summaries())
    health = ROUND.evaluate_health_gate(stats, gammas=GAMMAS)
    parent = tmp_path / "parent.pt"
    torch.save(
        {"state_dict": reference.policy.state_dict()}, parent,
    )
    diagnosis = ROUND.fallback_diagnosis(
        adapter=adapter, reference=reference, round_index=2, health=health,
        rows=rows, archive_ledger=[{"round": 2, "path": "unused"}],
        parent_checkpoint_path=parent, probe_seed=2,
    )
    assert diagnosis["status"] == ROUND.FALLBACK_STATUS
    assert diagnosis["halted_round"] == 2
    drift = diagnosis["policy_drift"]
    assert drift["relative_trainable_drift_vs_r0"] == pytest.approx(0.0)
    assert drift["relative_trainable_drift_vs_previous_round"] == (
        pytest.approx(0.0)
    )
    assert diagnosis["failure_typology"]["exact_negative_nvp_after_retries"] == 1
    assert "0.1" in diagnosis["per_gamma_positive_vanishing"]
    assert diagnosis["recommended_followup"]["action"].startswith("schedule")
    assert "no automatic retraining" in diagnosis["protocol"]
    # No positives anywhere -> no CFM probes, but the field must exist.
    assert diagnosis["dplus_cfm_probes"] == []


def _degenerate_run_args(tmp_path, checkpoint, sha):
    return ROUND.parser().parse_args([
        "--checkpoint", str(checkpoint),
        "--expected-checkpoint-sha256", sha,
        "--pretrain-dataset-root", str(tmp_path / "unused"),
        "--expected-pretrain-dataset-manifest-sha256", "0" * 64,
        "--output", str(tmp_path / "arm"),
        "--recipe-id", "E1",
        "--device", "cpu", "--rounds", "2", "--gammas", "0.1,1.0",
        "--lineages-per-gamma", "1", "--seed", "3",
    ])


def test_failed_gate_halts_before_any_training(tmp_path, monkeypatch):
    torch.manual_seed(0)
    policy_seed = {"count": 0}

    def _fake_load(path, device="cpu"):
        policy_seed["count"] += 1
        torch.manual_seed(11)
        return GPS.build_sfm_hp100_policy(), {"scientific_status": "test"}

    checkpoint = tmp_path / "r0.pt"
    torch.save({"state_dict": _adapter(11).policy.state_dict()}, checkpoint)
    import sfm_hp100_predictive_execution as PRED
    sha = PRED.sha256_file(checkpoint)

    monkeypatch.setattr(ROUND.GPS, "load_sfm_hp100_policy", _fake_load)
    monkeypatch.setattr(
        ROUND.BASE, "_gpu_contract", lambda device, gpu: {"device": device},
    )
    monkeypatch.setattr(
        ROUND.BASE, "calibration_features",
        lambda *a, **k: (torch.randn(50, 8), {"count": 50}),
    )
    monkeypatch.setattr(ROUND, "mean_pairwise_lengthscale", lambda f: 1.0)
    monkeypatch.setattr(
        ROUND.HYBRID, "_calibration_support_by_gamma",
        lambda *a, **k: {},
    )

    class _FakeTask:
        def __init__(self, **kwargs):
            pass

        def attach_context_encoder(self, policy):
            return self

    monkeypatch.setattr(ROUND.PORT, "SFMHP100ExpansionTask", _FakeTask)

    @contextlib.contextmanager
    def _fake_verifier(task, workers):
        yield object()

    monkeypatch.setattr(ROUND.HYBRID, "_OrderedSidecarVerifier", _fake_verifier)

    degenerate_rows = [_row(
        5, gamma=0.1, role="negative", negative_reason="all_negative_nvp",
    )]
    monkeypatch.setattr(
        ROUND.ARCH, "gather_round",
        lambda *a, **k: {
            "rows": list(degenerate_rows),
            "sample_counts": ROUND.ARCH._sample_counts(degenerate_rows),
            "block_summaries": _summaries(contexts=5, nvp=2, success=0),
            "trace_shards": [],
        },
    )
    monkeypatch.setattr(ROUND.ARCH, "assert_bank_disjoint", lambda rows: None)

    def _no_training(*args, **kwargs):
        raise AssertionError("training ran on a failed acquisition gate")

    monkeypatch.setattr(ROUND.UPD, "expansion_update", _no_training)

    final = ROUND.run_recipe(
        _degenerate_run_args(tmp_path, checkpoint, sha),
    )
    assert final["halted_at_round"] == 1
    assert final["rounds_completed"] == 0
    assert not final["accepted"]
    output = tmp_path / "arm"
    health = json.loads((output / "ROUND_1_HEALTH.json").read_text())
    assert not health["passed"]
    diagnosis = json.loads((output / "FALLBACK_DIAGNOSIS.json").read_text())
    assert diagnosis["status"] == ROUND.FALLBACK_STATUS
    assert final["dplus_yield_trend"][0]["dplus_total"] == 0
    assert not (output / "checkpoint_r1.pt").exists()
    audit = json.loads((output / "QUALIFICATION_AUDIT.json").read_text())
    assert audit["rounds"] == []


def test_rng_snapshot_restores_exactly():
    torch.manual_seed(7)
    np.random.seed(7)
    snapshot = ROUND._rng_snapshot()
    a_torch = torch.randn(4)
    a_np = np.random.rand(4)
    ROUND._rng_restore(snapshot)
    assert torch.equal(torch.randn(4), a_torch)
    assert np.allclose(np.random.rand(4), a_np)


def test_resume_payload_round_trips_model_optimizer_and_provenance(tmp_path):
    adapter = _adapter(seed=2)
    parameters, _ = UPD.configure_trainable(adapter)
    optimizer = torch.optim.Adam(parameters, lr=1.0e-5)
    UPD.expansion_update(
        adapter, [_row(seed) for seed in range(4)], [],
        UPD.UpdateConfig(seed=2), round_index=1, optimizer=optimizer,
    )
    payload = ROUND.resume_payload(
        adapter=adapter, optimizer=optimizer, round_index=1,
        parent_checkpoint_sha256="c" * 64, r0_checkpoint_sha256="a" * 64,
        recipe_provenance={"recipe_id": "E1"},
        archive_ledger=[{"round": 1, "path": "x"}],
        gp_state={"rule": "frozen", "features": torch.randn(50, 8),
                  "calibration": {}, "lengthscale": 1.0},
        baseline_dplus_per_gamma={"0.1": 4},
    )
    path = tmp_path / "resume_r1.pt"
    torch.save(payload, path)
    loaded = ROUND.load_resume(path, expected_r0_sha256="a" * 64)
    assert loaded["round_index"] == 1
    assert loaded["baseline_dplus_per_gamma"] == {"0.1": 4}
    restored = _adapter(seed=9)
    restored.policy.load_state_dict(loaded["model_state_dict"])
    for (name_a, a), (name_b, b) in zip(
        restored.named_parameters(), adapter.named_parameters(),
    ):
        assert name_a == name_b and torch.equal(a, b)
    restored_parameters, _ = UPD.configure_trainable(restored)
    restored_optimizer = torch.optim.Adam(restored_parameters, lr=1.0e-5)
    restored_optimizer.load_state_dict(loaded["optimizer_state_dict"])
    states = list(restored_optimizer.state.values())
    assert states and all("exp_avg" in state for state in states)
    with pytest.raises(RuntimeError, match="declared r0"):
        ROUND.load_resume(path, expected_r0_sha256="b" * 64)


def test_qualification_audit_collects_the_declared_evidence():
    marker = {
        "round": 1,
        "archive": {
            "sample_counts": {"positive": 3, "all_negative_nvp": 1,
                              "realized_collision": 0, "realized_oob": 0},
            "block_summaries": _summaries(),
        },
        "update": {
            "negative_counts_by_reason": {"all_negative_nvp": 1,
                                          "realized_collision": 0,
                                          "realized_oob": 0},
            "exposure_passes_declared": 4, "exposure_passes_completed": 4,
            "unique_positive_samples": 3, "unique_negative_samples": 1,
            "positive_exposures": 12, "negative_exposures": 4,
            "duplicate_exposures": 9, "in_archive_duplicate_rows": 0,
            "adam_steps": 4,
            "positive_loss_mean": 0.5, "negative_loss_mean": 0.6,
            "objective_mean": 0.5, "per_pass_positive_loss": [0.5] * 4,
            "grad_norm_pre_clip": [1.0], "grad_norm_post_clip": [1.0],
            "per_pass_grad_norm": [1.0] * 4, "clipped_fraction": 0.0,
            "positive_grad_norm": 1.0, "negative_grad_norm": 0.5,
            "grad_cosine": 0.1, "relative_parameter_drift": 0.01,
            "drift_gate": 0.25, "finite": True, "abort_reason": None,
            "accepted": True,
        },
    }
    audit = ROUND.qualification_audit(
        [marker], {"recipe_id": "QUAL-A0", "update_config": {"alpha": 0.0}},
    )
    assert audit["status"] == "SFM2_EXPANSION_QUALIFICATION_AUDIT"
    round_audit = audit["rounds"][0]
    assert round_audit["exposure"]["duplicate_exposures"] == 9
    assert round_audit["retry"]["terminal_nvp"] == 1
    assert round_audit["loss"]["per_pass_positive_loss"] == [0.5] * 4
    assert round_audit["drift"]["accepted"] is True
    assert "never count as evaluation" in audit["note"]


def _patched_healthy_environment(tmp_path, monkeypatch):
    def _fake_load(path, device="cpu"):
        torch.manual_seed(11)
        return GPS.build_sfm_hp100_policy(), {"scientific_status": "test"}

    checkpoint = tmp_path / "r0.pt"
    torch.save({"state_dict": _adapter(11).policy.state_dict()}, checkpoint)
    import sfm_hp100_predictive_execution as PRED
    sha = PRED.sha256_file(checkpoint)

    monkeypatch.setattr(ROUND.GPS, "load_sfm_hp100_policy", _fake_load)
    monkeypatch.setattr(
        ROUND.BASE, "_gpu_contract", lambda device, gpu: {"device": device},
    )
    monkeypatch.setattr(
        ROUND.BASE, "calibration_features",
        lambda *a, **k: (torch.randn(50, 8), {"count": 50}),
    )
    monkeypatch.setattr(ROUND, "mean_pairwise_lengthscale", lambda f: 1.0)
    monkeypatch.setattr(
        ROUND.HYBRID, "_calibration_support_by_gamma", lambda *a, **k: {},
    )

    class _FakeTask:
        def __init__(self, **kwargs):
            pass

        def attach_context_encoder(self, policy):
            return self

    monkeypatch.setattr(ROUND.PORT, "SFMHP100ExpansionTask", _FakeTask)

    @contextlib.contextmanager
    def _fake_verifier(task, workers):
        yield object()

    monkeypatch.setattr(
        ROUND.HYBRID, "_OrderedSidecarVerifier", _fake_verifier,
    )
    healthy = _healthy_rows()
    monkeypatch.setattr(
        ROUND.ARCH, "gather_round",
        lambda *a, **k: {
            "rows": list(healthy),
            "sample_counts": ROUND.ARCH._sample_counts(healthy),
            "block_summaries": _summaries(),
            "trace_shards": [],
        },
    )
    monkeypatch.setattr(ROUND.ARCH, "assert_bank_disjoint", lambda rows: None)
    return checkpoint, sha


def _healthy_run_args(tmp_path, checkpoint, sha, *extra):
    return ROUND.parser().parse_args([
        "--checkpoint", str(checkpoint),
        "--expected-checkpoint-sha256", sha,
        "--pretrain-dataset-root", str(tmp_path / "unused"),
        "--expected-pretrain-dataset-manifest-sha256", "0" * 64,
        "--output", str(tmp_path / "arm"),
        "--recipe-id", "E1",
        "--device", "cpu", "--gammas", "0.1,1.0",
        "--lineages-per-gamma", "1", "--seed", "3",
        *extra,
    ])


def test_resume_refuses_an_optimizer_scope_mismatch(tmp_path, monkeypatch):
    checkpoint, sha = _patched_healthy_environment(tmp_path, monkeypatch)
    final = ROUND.run_recipe(
        _healthy_run_args(tmp_path, checkpoint, sha, "--rounds", "1"),
    )
    assert final["accepted"] and final["rounds_completed"] == 1
    resume = tmp_path / "arm" / "resume_r1.pt"
    assert resume.is_file()
    with pytest.raises(RuntimeError, match="identical declared update recipe"):
        ROUND.run_recipe(_healthy_run_args(
            tmp_path, checkpoint, sha, "--rounds", "2",
            "--resume-from", str(resume),
            "--optimizer-scope", UPD.REDUCED_OPTIMIZER_SCOPE,
        ))
    # The identical scope resumes past the identity check and completes r2.
    resumed = ROUND.run_recipe(_healthy_run_args(
        tmp_path, checkpoint, sha, "--rounds", "2",
        "--resume-from", str(resume),
    ))
    assert resumed["rounds_completed"] == 2
    marker = json.loads(
        (tmp_path / "arm" / "ROUND_2_COMPLETE.json").read_text()
    )
    assert marker["update"]["optimizer_scope"] == UPD.OPTIMIZER_SCOPE
