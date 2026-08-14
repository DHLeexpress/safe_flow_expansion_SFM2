"""Gates 3-9 and 12: acquisition budget, retry, selector, roles, round trip."""
import numpy as np
import pytest
import torch

from sfm_hp100_ball_core.expansion import RBFPosterior, calibrate_fixed_beta
import sfm_hp100_exhaustive_hybrid as HYBRID
import sfm_hp100_expansion_archive as ARCH
import sfm_hp100_predictive_execution as PRED


class Result:
    def __init__(self, valid, progress, error=False):
        self.valid = valid
        self.progress = progress
        self.error = error
        self.hp_eligible = True
        self.margin = 0.0
        self.execution_cost = 0.0
        self.progress_eligible = True
        self.step_margin = 0.0


def audit(progress, clearance, collision_free=True):
    return {
        "H10_goal_progress": float(progress),
        "predicted_min_clearance": float(clearance),
        "predicted_collision_free": bool(collision_free),
    }


def test_frozen_acquisition_budget_and_ess_target():
    config = PRED.PredictiveConfig()
    config.validate()
    assert (config.K, config.B, config.ess_target) == (64, 32, 0.1)
    with pytest.raises(ValueError, match="K=64 and B=32"):
        PRED.PredictiveConfig(K=32).validate()
    with pytest.raises(ValueError, match="K=64 and B=32"):
        PRED.PredictiveConfig(B=16).validate()
    with pytest.raises(ValueError, match="1.0, 1.1"):
        PRED.PredictiveConfig(base_std_step=0.2).validate()


def test_rbf_acquisition_is_without_replacement_at_the_declared_budget():
    torch.manual_seed(3)
    posterior = RBFPosterior(1.0, 1.0e-2)
    posterior.set_buffer(torch.randn(50, 8, dtype=torch.float64))
    features = torch.randn(64, 8, dtype=torch.float64)
    sigma = posterior.sigma(features)
    beta = calibrate_fixed_beta([sigma], target=0.1)
    generator = torch.Generator().manual_seed(5)
    selected, selected_sigma, conditional_ess = posterior.acquire(
        features, 32, beta, generator,
    )
    assert len(selected) == 32
    assert len(set(map(int, selected))) == 32
    assert len(selected_sigma) == 32 and len(conditional_ess) == 32


def test_selector_orders_progress_clearance_sigma_then_index():
    results = [Result(True, 0.8) for _ in range(5)] + [Result(True, 0.9)]
    audits = [
        audit(0.8, 0.1), audit(0.8, 0.3), audit(0.8, 0.3),
        audit(0.8, 0.3), audit(0.8, 0.3), audit(0.9, 0.0),
    ]
    sigma = [0.9, 0.2, 0.7, 0.7, 0.1, 0.0]
    # Progress dominates every other key.
    assert PRED.select_predictive_progress(results, audits, sigma) == 5
    # Without the progress leader: clearance, then sigma, then lowest index.
    assert PRED.select_predictive_progress(
        results[:5], audits[:5], sigma[:5],
    ) == 2


def _event(lineage, step, attempts, *, terminal=None, executed_role=None,
           state_before=None, state_after=None, scenario_id=101):
    zeros = np.zeros(4, np.float32)
    return {
        "gamma": 0.3, "replica": 0, "lineage": lineage,
        "scenario_id": scenario_id, "step": step,
        "state_before": zeros if state_before is None else state_before,
        "state_after": zeros if state_after is None else state_after,
        "attempts": attempts,
        "executed_role": executed_role,
        "terminal": terminal,
    }


def _attempt(index, *, predictive_local=None, negative_local=None,
             base_std=None, valid_locals=()):
    verification = [
        {"valid": local in valid_locals, "H10_progress": 0.5,
         "step_margin": 0.1, "hp_eligible": True, "margin": 0.0,
         "native_cost": 0.0, "progress_eligible": True, "error": False}
        for local in range(2)
    ]
    row = {
        "attempt": index,
        "base_std": (1.0 + 0.1 * index) if base_std is None else base_std,
        "candidate_ids": [7, 21],
        "verification": verification,
        "prediction_audits": [audit(0.5, 0.2), audit(0.4, 0.1)],
        "selected_sigma": [0.3, 0.6],
        "marginal_sigma": [0.1] * 64,
        "conditional_ess": [0.2, 0.4],
        "beta": 2.5,
        "marginal_ESS_over_K": 0.1,
        "positive_first16": 1,
        "positive_B32": len(valid_locals),
        "predictive_local": predictive_local,
    }
    if negative_local is not None:
        row["negative_counterfactual_local"] = negative_local
    return row


def _sample(lineage, step, attempt, *, role, executed, negative_reason=None,
            valid=None, scenario_id=101):
    if valid is None:
        valid = role == "positive" or negative_reason != "all_negative_nvp"
    return {
        "role": role, "gamma": 0.3, "replica": 0, "lineage": lineage,
        "scenario_id": scenario_id, "step": step, "attempt": attempt,
        "executed": executed, "negative_reason": negative_reason,
        "context": torch.zeros(4), "candidate": torch.zeros(10, 2),
        "flow_base": torch.zeros(10, 2),
        "verification": {"valid": valid, "H10_progress": 0.5,
                         "step_margin": 0.1, "hp_eligible": True,
                         "margin": 0.0, "native_cost": 0.0,
                         "progress_eligible": True, "error": False},
        "prediction_audit": audit(0.5, 0.2),
    }


def test_trace_invariants_accept_the_declared_roles():
    events = [
        _event("g0.3:rep00", 0,
               [_attempt(0, predictive_local=0, valid_locals=(0,))],
               executed_role="positive",
               state_after=np.ones(4, np.float32)),
        _event("g0.3:rep01", 0,
               [_attempt(0), _attempt(1, negative_local=1)],
               terminal="nvp"),
        _event("g0.3:rep02", 0,
               [_attempt(0, predictive_local=0, valid_locals=(0,))],
               executed_role="realized_collision", terminal="collision",
               state_after=np.ones(4, np.float32)),
    ]
    samples = [
        _sample("g0.3:rep00", 0, 0, role="positive", executed=True),
        _sample("g0.3:rep01", 0, 1, role="negative", executed=False,
                negative_reason="all_negative_nvp"),
        _sample("g0.3:rep02", 0, 0, role="negative", executed=True,
                negative_reason="realized_collision"),
    ]
    ARCH.assert_trace_invariants(samples, events)


def test_trace_invariants_reject_retry_schedule_drift():
    events = [_event("g0.3:rep00", 0, [
        _attempt(0), _attempt(1, base_std=1.3, negative_local=0),
    ], terminal="nvp")]
    samples = [_sample("g0.3:rep00", 0, 1, role="negative", executed=False,
                       negative_reason="all_negative_nvp")]
    with pytest.raises(RuntimeError, match="retry std schedule"):
        ARCH.assert_trace_invariants(samples, events)


def test_trace_invariants_reject_state_advance_during_retry_exhaustion():
    events = [_event(
        "g0.3:rep00", 0, [_attempt(0, negative_local=0)],
        terminal="nvp", state_after=np.ones(4, np.float32),
    )]
    samples = [_sample("g0.3:rep00", 0, 0, role="negative", executed=False,
                       negative_reason="all_negative_nvp")]
    with pytest.raises(RuntimeError, match="advanced state"):
        ARCH.assert_trace_invariants(samples, events)


def test_trace_invariants_reject_an_executed_exact_negative():
    events = [_event("g0.3:rep00", 0,
                     [_attempt(0, predictive_local=0, valid_locals=(0,))],
                     executed_role="positive")]
    samples = [_sample("g0.3:rep00", 0, 0, role="positive", executed=True,
                       valid=False)]
    with pytest.raises(RuntimeError, match="exact negative was executed"):
        ARCH.assert_trace_invariants(samples, events)


def test_trace_invariants_require_exactly_one_nvp_counterfactual():
    events = [_event("g0.3:rep00", 0, [_attempt(0, negative_local=0)],
                     terminal="nvp")]
    with pytest.raises(RuntimeError, match="pair 1:1"):
        ARCH.assert_trace_invariants([], events)


def test_realized_failure_keeps_verifier_label_and_trace_role():
    events = [_event("g0.3:rep00", 0,
                     [_attempt(0, predictive_local=0, valid_locals=(0,))],
                     executed_role="positive", terminal="collision")]
    samples = [_sample("g0.3:rep00", 0, 0, role="negative", executed=True,
                       negative_reason="realized_collision")]
    # The archived row keeps y=1 but the trace recorded the wrong role.
    with pytest.raises(RuntimeError, match="disagrees with the trace"):
        ARCH.assert_trace_invariants(samples, events)


def test_enrichment_joins_provenance_and_fails_closed(tmp_path):
    del tmp_path
    config = ARCH.ArchiveConfig(
        gammas=(0.3,), lineages_per_gamma=1, round_index=2, seed=41,
    )
    provenance = {
        "scenario_start": 860_000,
        "checkpoint_sha256": "a" * 64,
        "reference_checkpoint_sha256": "b" * 64,
        "source_sha256": "c" * 64,
    }
    events = [_event("g0.3:rep00", 0,
                     [_attempt(0, predictive_local=1, valid_locals=(1,))],
                     executed_role="positive")]
    samples = [_sample("g0.3:rep00", 0, 0, role="positive", executed=True)]
    rows = ARCH.enrich_samples(
        samples, events, config=config, block=0, provenance=provenance,
    )
    row = rows[0]
    assert row["round"] == 2
    assert row["K_index"] == 21 and row["B_local"] == 1
    assert row["base_std"] == 1.0 and row["beta"] == 2.5
    assert row["selected_sigma"] == 0.6
    assert row["marginal_ESS_over_K"] == 0.1
    assert row["sampling_seed"] == HYBRID._sampling_seed(
        config.block_seed(0), "predictive_always_on",
        HYBRID.LineageKey(0.3, 0), 0, microcycle=0, attempt=0,
    )
    # A sample whose verification cannot be reproduced from the trace fails.
    drifted = dict(samples[0])
    drifted["verification"] = dict(
        drifted["verification"], H10_progress=9.0,
    )
    with pytest.raises(RuntimeError, match="verification differ"):
        ARCH.enrich_samples(
            [drifted], events, config=config, block=0, provenance=provenance,
        )
    orphan = _sample("g0.3:rep00", 5, 0, role="positive", executed=True)
    with pytest.raises(RuntimeError, match="lacks its trace event"):
        ARCH.enrich_samples(
            [orphan], events, config=config, block=0, provenance=provenance,
        )


def test_dematerialize_clears_inference_tensors_for_backward():
    with torch.inference_mode():
        row = {
            "context": torch.zeros(4),
            "candidate": torch.ones(10, 2),
            "flow_base": torch.ones(10, 2),
        }
        assert row["candidate"].is_inference()
    with pytest.raises(RuntimeError, match="outside inference mode"):
        with torch.inference_mode():
            ARCH.dematerialize(dict(row))
    cleared = ARCH.dematerialize(dict(row))
    assert not cleared["candidate"].is_inference()
    assert torch.equal(cleared["candidate"], row["candidate"])
    weight = torch.nn.Parameter(torch.ones(1))
    loss = (cleared["candidate"] * weight).sum()
    loss.backward()
    assert weight.grad is not None


def test_archive_rows_round_trip_bitwise(tmp_path):
    row = _sample("g0.3:rep00", 0, 0, role="positive", executed=True)
    row["context"] = torch.randn(340)
    row["candidate"] = torch.randn(10, 2)
    row["flow_base"] = torch.randn(10, 2)
    path = tmp_path / "rows.pt"
    torch.save({"rows": [row]}, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)["rows"][0]
    for field in ("context", "candidate", "flow_base"):
        assert torch.equal(loaded[field], row[field])
    assert loaded["verification"] == row["verification"]
    assert loaded["prediction_audit"] == row["prediction_audit"]


def test_bank_disjointness_fails_closed_on_a_collision():
    row = {
        "scenario_id": 900_004,
        "scene_profile": "double_density_velocity_ood",
    }
    with pytest.raises(RuntimeError, match="collides"):
        ARCH.assert_bank_disjoint([row])
    clear = {
        "scenario_id": 860_123,
        "scene_profile": "double_density_velocity_ood",
    }
    ARCH.assert_bank_disjoint([clear])


def test_blocks_may_not_share_a_scenario_id():
    with pytest.raises(RuntimeError, match="blocks 0 and 1"):
        ARCH._assert_blocks_disjoint({0: {5}, 1: {5}})
    ARCH._assert_blocks_disjoint({0: {5}, 1: {6}})
