from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import sfm_hp100_exhaustive_hybrid as HYBRID
import sfm_hp100_iterative_microcycles as ITER


def _row(group, key, cycle, step=0):
    return {
        "group": group, "gamma": float(key.gamma),
        "replica": int(key.replica), "lineage": key.label,
        "scenario_id": 123, "microcycle": int(cycle), "step": int(step),
        "context": torch.tensor([float(cycle), float(step)]),
        "candidate": torch.full((10, 2), float(step + 1)),
        "flow_base": torch.zeros(10, 2),
        "verification": {
            "valid": group in {"P1", "P2"}, "error": False,
            "step_margin": 0.1 if group in {"P1", "P2"} else -0.1,
        },
    }


def _event(key, cycle, step, *, executed="P1", terminal=None):
    return {
        "lineage": key.label, "gamma": float(key.gamma),
        "replica": int(key.replica), "microcycle": int(cycle),
        "step": int(step), "executed_group": executed, "terminal": terminal,
        "state_before": np.zeros(4, np.float32),
        "state_after": np.array([0.1, 0.1, 0.0, 0.0], np.float32),
    }


def _raw_result(keys, clear_labels=(), *, steps=1, progress=0.1):
    clear_labels = set(clear_labels)
    outcomes, events = {}, []
    for key in keys:
        clear = key.label in clear_labels
        outcomes[key.label] = {
            "status": "success" if clear else "raw_red",
            "clear": clear, "steps": int(steps),
        }
        events.append({
            "lineage": key.label, "step": 0,
            "state_before": np.zeros(4, np.float32),
            "state_after": np.array([progress, 0.0, 0.0, 0.0], np.float32),
            "terminal": "success" if clear else "raw_red",
        })
    return {"outcomes": outcomes, "events": events}


class _FakePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def config(self):
        return {"kind": "fake"}


class _FakeAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy = _FakePolicy()


def _dose():
    return ITER.FixedDose(
        "winner", p1_learning_rate=1e-5, p1_passes=1,
        p2_learning_rate=3e-4, p2_passes=1,
        negative_alpha=0.1, negative_passes=4,
    )


def _gather_factory(calls):
    def gather(_adapter, _reference, _task, *, keys, microcycle, **_kwargs):
        calls.append((int(microcycle), tuple(key.label for key in keys)))
        samples, events, outcomes = [], [], {}
        for key in keys:
            samples.extend((
                _row("P1", key, microcycle),
                _row("P2", key, microcycle, 1),
                _row("Dminus", key, microcycle, 1),
            ))
            events.append(_event(key, microcycle, 0, terminal="success"))
            outcomes[key.label] = {"status": "success"}
        return {
            "samples": samples, "events": events, "outcomes": outcomes,
            "sample_counts": {"P1": len(keys), "P2": len(keys),
                              "Dminus": len(keys)},
        }
    return gather


def _update_factory(cycles, *, dminus_exposures=0):
    def update(adapter, view, **kwargs):
        present = {
            int(row["microcycle"])
            for group in ("retained_P1", "retained_P2", "Dminus", "Ncausal")
            for row in view[group]
        }
        assert len(present) == 1
        cycles.append(present.pop())
        with torch.no_grad():
            adapter.policy.weight.add_(0.25)
        reports = {}
        for phase, group, passes in (
            ("P1", "retained_P1", kwargs["p1_passes"]),
            ("P2", "retained_P2", kwargs["p2_passes"]),
            ("negative", "Ncausal", kwargs["negative_passes"]),
        ):
            rows = len(view[group])
            reports[phase] = {
                "rows": rows, "passes": passes,
                "total_row_exposures": rows * passes,
            }
        reports["negative"].update({
            "negative_source": kwargs["negative_source"],
            "selected_source_counts": {
                "Dminus": int(dminus_exposures),
                "Ncausal": len(view["Ncausal"]),
            },
        })
        return {"accepted": True, **reports}
    return update


def test_selection_must_be_explicitly_eligible_for_phase2():
    marker = {
        "status": "SFM_HP100_DISASTER_PREFIX_SWEEP_COMPLETE",
        "eligible_for_phase2": False,
    }
    with pytest.raises(ValueError, match="not eligible_for_phase2"):
        ITER._validate_selection_marker(marker)


def test_current_cycle_ncausal_replaces_sources_and_rejects_prior_rows():
    key = HYBRID.LineageKey(0.1, 0)
    samples, events = [], []
    for step, group in enumerate(("P1", "P2", "P1")):
        events.append(_event(key, 2, step, executed=group))
        if group == "P2":
            samples.append(_row("Dminus", key, 2, step))
        samples.append(_row(group, key, 2, step))
    events.append(_event(
        key, 2, 3, executed=None, terminal="repair_exhausted",
    ))
    samples.append(_row("Dminus", key, 2, 3))
    gather = {"samples": samples, "events": events}

    derived = ITER._cycle_training_view(gather, 2)

    assert derived["repair_exhausted"] == 1
    assert len(derived["view"]["Ncausal"]) == 3
    assert not derived["view"]["retained_P1"]
    assert not derived["view"]["retained_P2"]
    assert derived["counts"]["selected_training_exposures"] == {
        "retained_P1": 0, "retained_P2": 0,
        "Dminus": 0, "Ncausal": 3,
    }
    contaminated = {**gather, "samples": [*samples, _row("P1", key, 1, 9)]}
    with pytest.raises(RuntimeError, match="prior-cycle sample"):
        ITER._cycle_training_view(contaminated, 2)


def test_acceptance_requires_clear_preservation_and_pareto_nondecrease():
    parent = {
        "clear_lineages": ["a"], "exact_positive_prefix_sum": 10,
        "aggregate_goal_progress": 2.0,
        "outcomes": {
            "a": {"steps": 4, "clear": True},
            "b": {"steps": 6, "clear": False},
        },
    }
    improved = {
        "clear_lineages": ["a", "b"], "exact_positive_prefix_sum": 11,
        "aggregate_goal_progress": 2.1,
        "outcomes": {
            "a": {"steps": 4, "clear": True},
            "b": {"steps": 7, "clear": True},
        },
    }
    assert ITER._acceptance_decision(parent, improved)["accepted"] is True
    lost_clear = {**improved, "clear_lineages": ["b"]}
    assert ITER._acceptance_decision(parent, lost_clear)["accepted"] is False
    conservative = {**improved, "aggregate_goal_progress": 1.9}
    assert ITER._acceptance_decision(parent, conservative)["accepted"] is False
    unchanged = dict(parent)
    assert ITER._acceptance_decision(parent, unchanged)["accepted"] is False


def test_new_faster_success_is_exempt_from_global_prefix_nondecrease():
    parent = {
        "clear_lineages": [], "exact_positive_prefix_sum": 30,
        "aggregate_goal_progress": 1.0,
        "outcomes": {
            "new_success": {"steps": 20, "clear": False},
            "still_open": {"steps": 10, "clear": False},
        },
    }
    candidate = {
        "clear_lineages": ["new_success"], "exact_positive_prefix_sum": 18,
        "aggregate_goal_progress": 1.2,
        "outcomes": {
            "new_success": {"steps": 8, "clear": True},
            "still_open": {"steps": 10, "clear": False},
        },
    }

    decision = ITER._acceptance_decision(parent, candidate)

    assert decision["aggregate_exact_positive_prefix_delta_diagnostic_only"] == -12
    assert decision["candidate_nonclear_lineage_prefix_deltas"] == {
        "still_open": 0,
    }
    assert decision["accepted"] is True


def test_prefix_or_progress_gain_without_raw_success_is_rejected():
    parent = {
        "clear_lineages": [], "exact_positive_prefix_sum": 10,
        "aggregate_goal_progress": 1.0,
        "outcomes": {"open": {"steps": 10, "clear": False}},
    }
    candidate = {
        "clear_lineages": [], "exact_positive_prefix_sum": 12,
        "aggregate_goal_progress": 1.2,
        "outcomes": {"open": {"steps": 12, "clear": False}},
    }

    decision = ITER._acceptance_decision(parent, candidate)

    assert decision["candidate_nonclear_prefix_nondecrease"] is True
    assert decision["goal_progress_nondecrease"] is True
    assert decision["raw_temperature_one_CLEAR_delta"] == 0
    assert decision["raw_temperature_one_success_strictly_increased"] is False
    assert decision["accepted"] is False


def test_loop_gathers_all_16_then_accepts_all_clear_without_replay(tmp_path):
    config = HYBRID.HybridConfig(max_microcycles=5)
    keys = HYBRID._lineage_keys(config)
    gather_calls, update_cycles = [], []
    raw_rows = iter((
        _raw_result(keys, clear_labels=(), steps=1, progress=0.1),
        _raw_result(keys, clear_labels={key.label for key in keys},
                    steps=2, progress=0.2),
    ))

    marker = ITER.run_microcycles(
        adapter=_FakeAdapter(), reference_adapter=object(), task=object(),
        verifier=object(), config=config, dose=_dose(), lengthscale=1.0,
        support_by_gamma={}, output=tmp_path,
        selection_sha256="a" * 64, optimizer_scope="head_only",
        max_cycles=5, plateau_patience=3,
        gather_fn=_gather_factory(gather_calls),
        recheck_fn=lambda *_args, **_kwargs: next(raw_rows),
        update_fn=_update_factory(update_cycles),
    )

    assert marker["status"] == ITER.ALL_CLEAR_STATUS
    assert marker["all_clear"] is True
    assert update_cycles == [1]
    assert gather_calls == [(1, tuple(key.label for key in keys))]
    assert len(gather_calls[0][1]) == 16
    cycle = json_load(tmp_path / "microcycles/cycle_001/CYCLE_COMPLETE.json")
    assert cycle["training_counts"]["selected_training_exposures"]["Dminus"] == 0
    assert cycle["previous_cycle_samples_replayed"] is False
    assert cycle["GP_updated"] is False


def json_load(path: Path):
    import json
    return json.loads(path.read_text())


def test_rejected_updates_rollback_bitwise_and_plateau_after_three(tmp_path):
    config = HYBRID.HybridConfig(max_microcycles=5)
    keys = HYBRID._lineage_keys(config)
    parent_raw = _raw_result(keys, clear_labels=(), steps=1, progress=0.1)
    raw_rows = iter((parent_raw, parent_raw, parent_raw, parent_raw))
    adapter = _FakeAdapter()
    original = adapter.policy.weight.detach().clone()
    gather_calls, update_cycles = [], []

    marker = ITER.run_microcycles(
        adapter=adapter, reference_adapter=object(), task=object(),
        verifier=object(), config=config, dose=_dose(), lengthscale=1.0,
        support_by_gamma={}, output=tmp_path,
        selection_sha256="b" * 64, optimizer_scope="head_only",
        max_cycles=5, plateau_patience=3,
        initial_checkpoint={
            "path": "/selected.pt", "sha256": "e" * 64,
            "policy_state_sha256": HYBRID._model_state_sha256(adapter),
        },
        gather_fn=_gather_factory(gather_calls),
        recheck_fn=lambda *_args, **_kwargs: next(raw_rows),
        update_fn=_update_factory(update_cycles),
    )

    assert marker["status"] == ITER.PLATEAU_STATUS
    assert marker["all_clear"] is False
    assert update_cycles == [1, 2, 3]
    assert [row[0] for row in gather_calls] == [1, 2, 3]
    assert torch.equal(adapter.policy.weight, original)
    assert all(not row["accepted"] for row in marker["cycles"])
    assert marker["final_accepted_checkpoint"] == {
        "path": "/selected.pt", "sha256": "e" * 64,
        "policy_state_sha256": HYBRID._model_state_sha256(adapter),
    }


def test_loop_fails_closed_if_update_reports_any_dminus_exposure(tmp_path):
    config = HYBRID.HybridConfig(max_microcycles=1)
    keys = HYBRID._lineage_keys(config)
    raw_rows = iter((
        _raw_result(keys, clear_labels=()),
        _raw_result(keys, clear_labels=()),
    ))
    with pytest.raises(RuntimeError, match="Dminus entered"):
        ITER.run_microcycles(
            adapter=_FakeAdapter(), reference_adapter=object(), task=object(),
            verifier=object(), config=config, dose=_dose(), lengthscale=1.0,
            support_by_gamma={}, output=tmp_path,
            selection_sha256="c" * 64, optimizer_scope="head_only",
            max_cycles=1, plateau_patience=1,
            gather_fn=_gather_factory([]),
            recheck_fn=lambda *_args, **_kwargs: next(raw_rows),
            update_fn=_update_factory([], dminus_exposures=1),
        )


def test_short_disaster_prefix_rejects_cycle_without_update_then_plateaus(tmp_path):
    config = HYBRID.HybridConfig(max_microcycles=1)
    keys = HYBRID._lineage_keys(config)
    raw_rows = iter((
        _raw_result(keys, clear_labels=()),
        _raw_result(keys, clear_labels=()),
    ))
    update_called = []

    def gather(_adapter, _reference, _task, *, keys, microcycle, **_kwargs):
        samples, events, outcomes = [], [], {}
        for index, key in enumerate(keys):
            samples.append(_row(
                "Dminus" if index == 0 else "P1", key, microcycle, 0,
            ))
            events.append(_event(
                key, microcycle, 0, executed=None if index == 0 else "P1",
                terminal="repair_exhausted" if index == 0 else "success",
            ))
            outcomes[key.label] = {
                "status": "repair_exhausted" if index == 0 else "success"
            }
        return {
            "samples": samples, "events": events, "outcomes": outcomes,
            "sample_counts": {"P1": len(keys) - 1, "Dminus": 1},
        }

    marker = ITER.run_microcycles(
        adapter=_FakeAdapter(), reference_adapter=object(), task=object(),
        verifier=object(), config=config, dose=_dose(), lengthscale=1.0,
        support_by_gamma={}, output=tmp_path,
        selection_sha256="d" * 64, optimizer_scope="head_only",
        max_cycles=1, plateau_patience=1, gather_fn=gather,
        recheck_fn=lambda *_args, **_kwargs: next(raw_rows),
        update_fn=lambda *_args, **_kwargs: update_called.append(True),
    )

    assert marker["status"] == ITER.PLATEAU_STATUS
    assert update_called == []
    cycle = json_load(tmp_path / "microcycles/cycle_001/CYCLE_COMPLETE.json")
    assert cycle["training_view_admissible"] is False
    assert cycle["update"]["parameter_update_performed"] is False
    assert cycle["training_counts"]["selected_training_exposures"] == {
        "Dminus": 0, "Ncausal": 0, "retained_P1": 0, "retained_P2": 0,
    }
    assert cycle["training_counts"]["training_view_rows"] == 0
