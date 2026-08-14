from dataclasses import replace
import json

import numpy as np
import pytest
import torch
from torch import nn

import sfm_hp100_disaster_prefix_continuation as CONT
import sfm_hp100_exhaustive_hybrid as HYBRID
from sfm_hp100_ball_core.expansion import Verification


def _row(group, lineage, step, target):
    return {
        "group": group,
        "gamma": float(lineage.split(":")[0][1:]),
        "replica": int(lineage.split("rep")[1]),
        "lineage": lineage,
        "scenario_id": 123,
        "microcycle": 1,
        "step": int(step),
        "context": torch.tensor([float(step), 0.25]),
        "candidate": torch.full((10, 2), float(target)),
        "flow_base": torch.full((10, 2), -float(target)),
        "verification": {
            "valid": group in {"P1", "P2"}, "error": False,
            "step_margin": 0.1 if group in {"P1", "P2"} else -0.1,
        },
    }


def _incomplete_trace_and_samples():
    events, samples = [], []
    for replica in range(6):
        gamma = (0.1, 0.3, 0.5, 1.0)[replica % 4]
        lineage = f"g{gamma:g}:rep{replica:02d}"
        for step in range(3):
            group = "P1" if step != 1 else "P2"
            events.append({
                "lineage": lineage, "microcycle": 1, "step": step,
                "executed_group": group, "terminal": None,
            })
            if group == "P2":
                samples.append(_row("Dminus", lineage, step, -1.0))
            samples.append(_row(group, lineage, step, step + 1.0))
        events.append({
            "lineage": lineage, "microcycle": 1, "step": 3,
            "executed_group": None, "terminal": "repair_exhausted",
        })
        samples.append(_row("Dminus", lineage, 3, -2.0))
    return {"events": events}, samples


def test_extracts_exactly_three_executed_windows_per_disaster_without_relabeling():
    trace, samples = _incomplete_trace_and_samples()
    source_candidates = {
        CONT._sample_identity(row): row["candidate"].clone() for row in samples
    }

    view = CONT.extract_disaster_training_view(trace, samples)

    assert view["counts"] == {
        "source_P1": 12, "source_P2": 6, "source_Dminus": 12,
        "retained_P1": 0, "retained_P2": 0, "Ncausal": 18,
        "repair_exhausted_lineages": 6,
    }
    assert len(view["causal_source_identities"]) == 18
    assert {row["source_group"] for row in view["Ncausal"]} == {"P1", "P2"}
    assert all(row["group"] == "Ncausal" for row in view["Ncausal"])
    assert all(row["verification"]["valid"] for row in view["Ncausal"])
    assert all(row["true_verifier_label_preserved"] for row in view["Ncausal"])
    assert sorted(row["causal_offset"] for row in view["Ncausal"][:3]) == [-3, -2, -1]
    # The source artifact is immutable and the view owns cloned tensors.
    for row in samples:
        assert row["group"] in {"P1", "P2", "Dminus"}
        assert torch.equal(row["candidate"], source_candidates[CONT._sample_identity(row)])
    source = next(row for row in samples if row["group"] == "P1")
    derived = next(
        row for row in view["Ncausal"]
        if tuple(row["source_identity"]) == CONT._sample_identity(source)
    )
    assert derived["candidate"].data_ptr() != source["candidate"].data_ptr()


def test_extraction_fails_closed_on_wrong_disaster_count_or_nonexecuted_prefix():
    trace, samples = _incomplete_trace_and_samples()
    with pytest.raises(ValueError, match="expected 5 repair-exhausted"):
        CONT.extract_disaster_training_view(trace, samples, expected_exhausted=5)

    trace["events"][0]["executed_group"] = None
    with pytest.raises(ValueError, match="executed P1/P2"):
        CONT.extract_disaster_training_view(trace, samples)


class _TinyAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = nn.Parameter(torch.tensor(7.0), requires_grad=False)
        self.head = nn.Parameter(torch.tensor(1.0))

    def cfm_loss(self, contexts, candidates, reduction="mean"):
        target = candidates.reshape(len(candidates), -1).mean(dim=1)
        prediction = self.head.expand_as(target)
        loss = (prediction - target).square()
        return loss if reduction == "none" else loss.mean()


def _training_view():
    def row(group, step, target):
        return {
            "group": group, "lineage": "g0.1:rep00", "microcycle": 1,
            "step": step, "context": torch.tensor([0.1]),
            "candidate": torch.full((10, 2), float(target)),
        }

    return {
        "retained_P2": [row("P2", 0, 2.0)],
        "retained_P1": [row("P1", 1, 0.5)],
        "Dminus": [row("Dminus", 2, -1.0)],
        "Ncausal": [row("Ncausal", 3, 3.0)],
    }


def test_update_uses_one_shared_positive_anchor_then_treatment_only_ascent():
    adapter = _TinyAdapter()
    frozen_before = adapter.frozen.detach().clone()

    report = CONT.apply_disaster_prefix_update(
        adapter, _training_view(), p2_learning_rate=1.0e-3,
        p1_learning_rate=1.0e-5, negative_learning_rate=1.0e-4,
        alpha=0.01, batch_size=64, max_relative_parameter_drift=0.5,
        seed=2,
    )

    anchor = report.pop("_positive_anchor_state_dict")
    assert report["accepted"] is True
    assert report["positive_anchor_bitwise_equal_to_alpha0_control"] is True
    assert report["positive_anchor_model_sha256"] == report[
        "alpha0_control_model_sha256"
    ]
    assert report["negative"]["effective_learning_rate"] == pytest.approx(1.0e-6)
    assert report["P2"]["rows"] == report["P1"]["rows"] == 1
    assert report["negative"]["rows"] == 2
    assert report["negative"]["negative_source"] == "Dminus_plus_Ncausal"
    assert report["negative"]["selected_source_counts"] == {
        "Dminus": 1, "Ncausal": 1,
    }
    assert report["P2"]["passes"] == report["P1"]["passes"] == 1
    assert report["negative"]["passes"] == 1
    assert report["P2"]["deterministic_fresh_order_seed_per_pass"] == [
        CONT._counter_seed(2, "retained_P2_attraction", "order")
    ]
    assert report["negative"]["deterministic_fresh_order_seed_per_pass"] == [
        CONT._counter_seed(2, "Dminus_Ncausal_negative_ascent", "order")
    ]
    assert report["P2"]["exposures_per_row"] == 1
    assert report["P1"]["exposures_per_row"] == 1
    assert report["negative"]["exposures_per_row"] == 1
    assert not torch.equal(adapter.head.detach().cpu(), anchor["head"])
    assert torch.equal(adapter.frozen, frozen_before)


def test_ncausal_only_uses_18_rows_and_exact_multi_pass_accounting():
    trace, samples = _incomplete_trace_and_samples()
    view = CONT.extract_disaster_training_view(trace, samples)
    second_trace, second_samples = _incomplete_trace_and_samples()
    second_view = CONT.extract_disaster_training_view(
        second_trace, second_samples,
    )
    for row in second_view["Dminus"]:
        row["candidate"] = torch.full_like(row["candidate"], 1.0e6)
    first = _TinyAdapter()
    second = _TinyAdapter()

    kwargs = dict(
        p2_learning_rate=1.0e-3, p1_learning_rate=1.0e-5,
        negative_learning_rate=1.0e-4, alpha=0.01, batch_size=7,
        max_relative_parameter_drift=1.0, seed=19,
        p1_passes=2, p2_passes=3, negative_passes=4,
        negative_source="Ncausal_only",
    )
    first_report = CONT.apply_disaster_prefix_update(first, view, **kwargs)
    second_report = CONT.apply_disaster_prefix_update(second, second_view, **kwargs)
    first_report.pop("_positive_anchor_state_dict")
    second_report.pop("_positive_anchor_state_dict")

    negative = first_report["negative"]
    assert negative["negative_source"] == "Ncausal_only"
    assert negative["available_source_counts"] == {
        "Dminus": 12, "Ncausal": 18,
    }
    assert negative["selected_source_counts"] == {
        "Dminus": 0, "Ncausal": 18,
    }
    assert negative["rows"] == 18
    assert negative["passes"] == negative["exposures_per_row"] == 4
    assert negative["total_row_exposures"] == 72
    assert negative["adam_steps"] == 12  # 4 * ceil(18 / 7)
    assert negative["optimizer_instances"] == 1
    assert len(set(negative["deterministic_fresh_order_seed_per_pass"])) == 4
    # Empty retained-positive pools still report their declared passes exactly.
    assert first_report["P1"]["rows"] == first_report["P2"]["rows"] == 0
    assert first_report["P1"]["passes"] == 2
    assert first_report["P2"]["passes"] == 3
    assert first_report["P1"]["adam_steps"] == 0
    assert first_report["P2"]["adam_steps"] == 0
    # Dminus has exactly zero loss/exposure influence in Ncausal-only mode:
    # replacing every Dminus action by an extreme value leaves the update
    # bitwise identical. Counter-derived ordering/CFM seeds are repeatable too.
    assert first_report["treatment_model_sha256_after_gate"] == second_report[
        "treatment_model_sha256_after_gate"
    ]
    assert torch.equal(first.head, second.head)


def test_positive_multi_passes_use_one_optimizer_per_phase():
    adapter = _TinyAdapter()
    report = CONT.apply_disaster_prefix_update(
        adapter, _training_view(), p2_learning_rate=1.0e-3,
        p1_learning_rate=1.0e-5, negative_learning_rate=1.0e-4,
        alpha=0.0, batch_size=64, max_relative_parameter_drift=1.0,
        seed=2, p1_passes=5, p2_passes=3, negative_passes=2,
    )
    report.pop("_positive_anchor_state_dict")
    assert report["P1"]["rows"] == 1
    assert report["P1"]["passes"] == report["P1"]["exposures_per_row"] == 5
    assert report["P1"]["total_row_exposures"] == 5
    assert report["P1"]["adam_steps"] == 5
    assert report["P1"]["optimizer_instances"] == 1
    assert report["P2"]["passes"] == 3
    assert report["P2"]["adam_steps"] == 3
    assert report["negative"]["passes"] == 2
    assert report["negative"]["adam_steps"] == 2


def test_pass_validation_happens_before_any_parameter_update():
    adapter = _TinyAdapter()
    before = {key: value.clone() for key, value in adapter.state_dict().items()}
    with pytest.raises(ValueError, match="passes must be positive"):
        CONT.apply_disaster_prefix_update(
            adapter, _training_view(), p2_learning_rate=1.0e-3,
            p1_learning_rate=1.0e-5, negative_learning_rate=1.0e-4,
            alpha=0.01, batch_size=64, max_relative_parameter_drift=1.0,
            seed=2, p1_passes=0,
        )
    for key, value in adapter.state_dict().items():
        assert torch.equal(value, before[key])


def test_cli_defaults_preserve_one_pass_combined_negative_behavior():
    args = CONT.parser().parse_args([
        "--checkpoint", "r0.pt",
        "--expected-checkpoint-sha256", "a" * 64,
        "--prior-trace", "trace.pt",
        "--expected-prior-trace-sha256", "b" * 64,
        "--prior-staged-samples", "samples.pt",
        "--expected-prior-staged-samples-sha256", "c" * 64,
        "--pretrain-dataset-root", "dataset",
        "--expected-pretrain-dataset-manifest-sha256", "d" * 64,
        "--output", "out",
        "--physical-gpu", "1",
        "--optimizer-scope", "head_only",
    ])
    assert (args.p1_passes, args.p2_passes, args.negative_passes) == (1, 1, 1)
    assert args.negative_source == "Dminus_plus_Ncausal"


def test_update_rolls_back_all_three_phases_atomically_on_drift():
    adapter = _TinyAdapter()
    before = {key: value.clone() for key, value in adapter.state_dict().items()}
    report = CONT.apply_disaster_prefix_update(
        adapter, _training_view(), p2_learning_rate=0.1,
        p1_learning_rate=0.1, negative_learning_rate=0.1, alpha=1.0,
        batch_size=1, max_relative_parameter_drift=1.0e-12, seed=2,
    )
    report.pop("_positive_anchor_state_dict")
    assert report["accepted"] is False
    assert report["atomic_rollback"] is True
    for key, value in adapter.state_dict().items():
        assert torch.equal(value, before[key])


class _FakeState:
    def __init__(self, step=0):
        self.robot = np.array([float(step), 0.0, 0.0, 0.0], np.float32)
        self.steps = int(step)
        self.scenario_id = 11


class _FakeTask:
    def __init__(self):
        self.context_calls = 0

    def reset(self, gamma, episode, seed):
        del gamma, episode, seed
        return _FakeState(0)

    def context(self, state, gamma):
        self.context_calls += 1
        return torch.tensor([float(state.steps), float(gamma)])

    def advance(self, state, candidate):
        del candidate
        return _FakeState(state.steps + 1)

    def terminal(self, state):
        del state
        return None


def _replay_fixture():
    lineage = "g0.1:rep00"
    events, samples = {}, {}
    for step in range(3):
        event = {
            "lineage": lineage, "microcycle": 1, "step": step,
            "context": torch.tensor([float(step), 0.1]),
            "state_before": np.array([step, 0, 0, 0], np.float32),
            "state_after": np.array([step + 1, 0, 0, 0], np.float32),
            "executed_group": "P1",
        }
        events[step] = event
        row = _row("P1", lineage, step, 0.0)
        samples[CONT._sample_identity(row)] = row
    return {lineage: events}, samples


def test_reconstruction_calls_context_once_per_step_and_asserts_archive():
    task = _FakeTask()
    events, samples = _replay_fixture()
    config = replace(HYBRID.HybridConfig(), gammas=(0.1,), lineages_per_gamma=1)
    state, context, report = CONT.reconstruct_prefix_state(
        task, key=HYBRID.LineageKey(0.1, 0), target_step=2,
        config=config, events_by_lineage=events, samples_by_identity=samples,
    )
    assert state.steps == 2
    assert torch.equal(context, torch.tensor([2.0, 0.1]))
    assert task.context_calls == 3
    assert report["context_calls"] == report["expected_context_calls"] == 3
    assert report["archived_context_bitwise_equal"] is True

    events["g0.1:rep00"][1]["context"][0] = 99.0
    with pytest.raises(RuntimeError, match="archived context mismatch"):
        CONT.reconstruct_prefix_state(
            _FakeTask(), key=HYBRID.LineageKey(0.1, 0), target_step=2,
            config=config, events_by_lineage=events,
            samples_by_identity=samples,
        )


def test_targeted_crn_uses_raw_p1_or_dminus_base_never_p2_repair_base():
    lineage = "g0.1:rep00"
    events = {lineage: {
        0: {"executed_group": "P1"},
        1: {"executed_group": "P2"},
        2: {"executed_group": "P1"},
        3: {"executed_group": None},
    }}
    rows = [
        _row("P1", lineage, 0, 10.0),
        _row("Dminus", lineage, 1, 20.0),
        _row("P2", lineage, 1, 999.0),
        _row("P1", lineage, 2, 30.0),
        _row("Dminus", lineage, 3, 40.0),
    ]
    samples = {CONT._sample_identity(row): row for row in rows}
    bases = CONT.archived_raw_base_sequence(
        disaster={
            "lineage": lineage, "prefix_start_step": 0,
            "disaster_step": 3,
        },
        events_by_lineage=events, samples_by_identity=samples,
    )
    # _row stores flow_base=-target. The P2 repair base (-999) is excluded.
    assert [float(base[0, 0]) for base in bases] == [-10.0, -20.0, -30.0, -40.0]


def test_diagnostic_rerun_reports_available_prefix_when_failure_is_before_N():
    lineage = "g0.1:rep00"
    trace = {"events": [
        {"lineage": lineage, "microcycle": 1, "step": 0,
         "executed_group": "P1", "terminal": None},
        {"lineage": lineage, "microcycle": 1, "step": 1,
         "executed_group": None, "terminal": "repair_exhausted"},
    ]}
    view = CONT.extract_disaster_training_view(
        trace, [_row("P1", lineage, 0, 1.0), _row("Dminus", lineage, 1, -1.0)],
        expected_exhausted=1, require_full_prefix=False,
    )
    assert view["counts"]["Ncausal"] == 1
    assert view["disasters"][0]["prefix_start_step"] == 0

    counts = CONT._exclusive_training_counts(view)
    assert counts["raw_source_counts"] == {"P1": 1, "P2": 0, "Dminus": 1}
    assert counts["mutually_exclusive_training_view_counts"] == {
        "retained_P1": 0, "retained_P2": 0,
        "source_Dminus": 1, "Ncausal": 1,
    }
    assert counts["source_rows"] == counts["training_view_rows"] == 2
    # The count/attribution block used by delivery markers is strict JSON.
    assert json.loads(json.dumps(counts)) == counts


class _TargetAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))


class _TargetTask:
    def context(self, state, gamma):
        return torch.tensor([float(state.steps), float(gamma)])

    def advance(self, state, candidate):
        del candidate
        return _FakeState(state.steps + 1)

    def terminal(self, state):
        del state
        return None


class _AlwaysPositiveVerifier:
    def verify_many(self, blocks):
        output = []
        for _context, candidates, _gamma in blocks:
            result = Verification(
                True, True, 0.5, 1.0, progress=0.2, step_margin=0.1,
            )
            output.append(((result,) * len(candidates), ({},) * len(candidates)))
        return output


def test_targeted_recovery_consumes_all_four_archived_raw_bases(monkeypatch):
    consumed = []

    def fake_sample(_adapter, _context, bases):
        consumed.append(float(bases.reshape(-1)[0]))
        return torch.zeros(len(bases), 10, 2)

    monkeypatch.setattr(CONT, "_sample_from_bases", fake_sample)
    monkeypatch.setattr(HYBRID, "_validated_sidecar", lambda *args: {})
    monkeypatch.setattr(HYBRID, "_attach_scene_to_event", lambda *args: None)
    monkeypatch.setattr(HYBRID, "_segment", lambda state, candidate: np.zeros((11, 2)))
    adapter = _TargetAdapter()
    state = _FakeState(0)
    state.scenario_id = 11
    bases = [torch.full((10, 2), float(index)) for index in (1, 2, 3, 4)]
    result = CONT.targeted_four_step_recovery(
        adapter, adapter, _TargetTask(), _AlwaysPositiveVerifier(),
        reconstructed={
            "key": HYBRID.LineageKey(0.1, 0), "state": state,
            "context": torch.tensor([0.0, 0.1]),
            "prefix_start_step": 0, "disaster_step": 3,
        },
        archived_raw_bases=bases,
        config=replace(
            HYBRID.HybridConfig(), gammas=(0.1,), lineages_per_gamma=1,
        ),
        lengthscale=1.0, support=torch.zeros(1, 2),
    )
    assert result["status"] == "survived_N_plus_1"
    assert result["raw_archived_bases_available"] == 4
    assert result["raw_archived_bases_consumed"] == 4
    assert consumed == [1.0, 2.0, 3.0, 4.0]


def test_full_rerun_gate_is_scientific_and_force_is_diagnostic_only():
    assert CONT._full_rerun_decision(4, 4, False) == (True, True)
    assert CONT._full_rerun_decision(3, 4, False) == (False, False)
    assert CONT._full_rerun_decision(3, 4, True) == (False, True)


def test_behavior_gate_and_alpha_effect_attribution_are_separate():
    def row(label, recovered):
        return {
            "lineage": label, "recovered": recovered,
            "status": "survived_N_plus_1" if recovered else "repair_exhausted",
            "sample_counts": {},
        }

    alpha0 = [row(f"L{index}", index < 4) for index in range(6)]
    # Same aggregate recovery, but one incremental repair and one regression:
    # behavior gate passes; alpha-effect claim must remain false.
    treatment = [row(f"L{index}", index in {0, 1, 2, 4}) for index in range(6)]
    r0 = [row(f"L{index}", index < 2) for index in range(6)]
    attribution = CONT._targeted_recovery_attribution(alpha0, treatment, r0)
    assert CONT._full_rerun_decision(
        attribution["treatment_recovered"], 4, False,
    ) == (True, True)
    assert attribution["incremental_recoveries_treatment_over_alpha0"] == 1
    assert attribution["incremental_regressions_treatment_vs_alpha0"] == 1
    assert attribution["alpha_effect_supported_by_recovery_count"] is False
    assert attribution["r0_recovered"] == 2
    assert attribution["positive_anchor_minus_r0_recovered"] == 2
    assert attribution["rows"][0]["r0_status"] == "survived_N_plus_1"

    treatment[5] = row("L5", True)
    attribution = CONT._targeted_recovery_attribution(alpha0, treatment, r0)
    assert attribution["treatment_recovered"] == 5
    assert attribution["alpha_effect_supported_by_recovery_count"] is True


def test_full_trace_authenticates_treatment_separately_from_r0(tmp_path):
    treatment = tmp_path / "treatment.pt"
    treatment.write_bytes(b"diagnostic treatment checkpoint")
    provenance = CONT._treatment_trace_provenance(
        treatment_checkpoint=treatment,
        treatment_policy_state_sha256="b" * 64,
        parent_r0_checkpoint_sha256="a" * 64,
        source_incomplete_trace_sha256="c" * 64,
    )
    assert provenance["checkpoint_sha256"] == provenance[
        "generating_checkpoint_sha256"
    ]
    assert provenance["checkpoint_sha256"] != provenance[
        "parent_r0_checkpoint_sha256"
    ]
    assert provenance["generating_policy_state_sha256"] == "b" * 64
    assert provenance["parent_r0_checkpoint_sha256"] == "a" * 64
    assert json.loads(json.dumps(provenance)) == provenance
