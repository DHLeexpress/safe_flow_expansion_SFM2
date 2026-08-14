"""Gate 13: rendering follows the committed event log without resampling."""
import inspect

import numpy as np
import pytest
import torch

import sfm_hp100_early_acquisition as ACQ
import sfm_hp100_predictive_execution as PRED
import sfm_hp100_predictive_execution_viz as VIZ


def _stub_event(step, *, executed=True):
    verification = [
        {"valid": executed, "H10_progress": 0.3, "step_margin": 0.5},
        {"valid": executed, "H10_progress": 0.8, "step_margin": 0.2},
    ]
    audits = [
        {"predicted_collision_free": True, "predicted_min_clearance": 0.4},
        {"predicted_collision_free": True, "predicted_min_clearance": 0.2},
    ]
    return {
        "gamma": 0.3,
        "replica": 0,
        "lineage": "g0.3:rep00",
        "scenario_id": 42,
        "step": step,
        "state_before": np.zeros(4, np.float32),
        "state_after": np.array([0.1, 0.1, 1.0, 1.0], np.float32),
        "ped_xy": np.array([[0.3, 0.0]], np.float32),
        "ped_vel": np.zeros((1, 2), np.float32),
        "attempts": [{
            "attempt": 0,
            "base_std": 1.0,
            "candidate_ids": [3, 9],
            "B_segments": np.zeros((2, 11, 2), np.float32),
            "verification": verification,
            "prediction_audits": audits,
            "predictive_local": 1 if executed else None,
            "negative_counterfactual_local": None if executed else 0,
            "max_margin_reference_local": 0,
            "selected_sigma": [0.2, 0.8],
            "positive_B32": 6,
        }],
        "executed_role": "positive" if executed else None,
        "terminal": None if executed else "nvp",
    }


@pytest.fixture()
def no_sampling(monkeypatch):
    def poisoned(*args, **kwargs):
        raise AssertionError("rendering must never resample the flow policy")

    monkeypatch.setattr(ACQ, "_sample_blocks", poisoned)
    monkeypatch.setattr(PRED, "gather_predictive", poisoned)
    return poisoned


def test_viz_module_never_imports_a_sampling_path():
    source = inspect.getsource(VIZ)
    for banned in (
        "sfm_hp100_early_acquisition",
        "sfm_hp100_exhaustive_hybrid",
        "gather_predictive",
        "sample_with_base",
    ):
        assert banned not in source


def test_case_stats_and_selection_replay_the_stored_event_log(no_sampling):
    events = [_stub_event(step) for step in range(3)]
    stats = VIZ.event_case_stats(events[0])
    assert stats["selector_disagreement"]
    assert stats["exact_positive_B32"] == 6
    selected_segment, replayed, _ = VIZ._selected_segment(events[0])
    assert selected_segment.shape == (11, 2)
    assert replayed


def test_lineage_replay_reads_only_the_committed_trace(no_sampling, tmp_path):
    trace = {
        "status": PRED.TRACE_STATUS,
        "events": [_stub_event(step) for step in range(3)],
    }
    path = tmp_path / "trace.pt"
    torch.save(trace, path)
    loaded = VIZ._load(path)
    rows = VIZ._events_for_lineage(loaded, "g0.3:rep00")
    assert [int(row["step"]) for row in rows] == [0, 1, 2]
    history = VIZ._history(rows, 2, reveal_current=True)
    assert history.shape == (4, 2)
    nvp = _stub_event(0, executed=False)
    segment, replayed, sidecar = VIZ._selected_segment(nvp)
    assert not replayed and sidecar is None


def test_noncontiguous_lineage_steps_fail_closed(no_sampling):
    trace = {"events": [_stub_event(0), _stub_event(2)]}
    with pytest.raises(ValueError, match="contiguous"):
        VIZ._events_for_lineage(trace, "g0.3:rep00")
