from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import sfm_hp100_exhaustive_hybrid as HYBRID
import sfm_hp100_iterative_dose_sweep as SWEEP
import sfm_hp100_iterative_microcycles as ITER


class _Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def config(self):
        return {"kind": "fake"}


class _Adapter(nn.Module):
    def __init__(self, scope="head_only"):
        super().__init__()
        self.policy = _Policy()
        self.scope = scope


def _row(group, cycle=1):
    return {
        "group": group,
        "lineage": "g0.1:rep00",
        "gamma": 0.1,
        "replica": 0,
        "microcycle": cycle,
        "step": 0,
        "context": torch.zeros(2),
        "candidate": torch.zeros(10, 2),
        "flow_base": torch.zeros(10, 2),
    }


def _view():
    return {
        "retained_P1": [_row("P1")],
        "retained_P2": [_row("P2")],
        "Dminus": [_row("Dminus")],
        "Ncausal": [{**_row("Ncausal"), "source_group": "P1"}],
    }


def _summary(clear=(), *, progress=1.0):
    clear = set(clear)
    return {
        "clear_lineages": sorted(clear),
        "clear_count": len(clear),
        "all_clear": len(clear) == 2,
        "exact_positive_prefix_sum": 6,
        "aggregate_goal_progress": float(progress),
        "outcomes": {
            "a": {"clear": "a" in clear, "steps": 3},
            "b": {"clear": "b" in clear, "steps": 3},
        },
    }


def _fake_update(starts, *, drift_by_scope=None, make_eligible=True):
    drift_by_scope = drift_by_scope or {
        "head_only": 0.02, "last_block_and_head": 0.03,
    }

    def update(adapter, view, **kwargs):
        starts.append((adapter.scope, float(adapter.policy.weight.detach())))
        delta = 0.25 if make_eligible else 0.01
        with torch.no_grad():
            adapter.policy.weight.add_(delta)
        reports = {}
        for phase, group, passes in (
            ("P1", "retained_P1", kwargs["p1_passes"]),
            ("P2", "retained_P2", kwargs["p2_passes"]),
            ("negative", "Ncausal", kwargs["negative_passes"]),
        ):
            reports[phase] = {
                "rows": len(view[group]),
                "passes": int(passes),
                "total_row_exposures": len(view[group]) * int(passes),
            }
        reports["negative"].update({
            "negative_source": kwargs["negative_source"],
            "selected_source_counts": {
                "Dminus": 0, "Ncausal": len(view["Ncausal"]),
            },
            "effective_learning_rate": (
                kwargs["negative_learning_rate"] * kwargs["alpha"]
            ),
        })
        return {
            "accepted": True,
            "relative_parameter_drift": drift_by_scope[adapter.scope],
            "positive_anchor_model_sha256": f"{adapter.scope}-anchor",
            "_positive_anchor_state_dict": {},
            **reports,
        }
    return update


def _raw(keys):
    outcomes, events = {}, []
    for key in keys:
        outcomes[key.label] = {"clear": False, "steps": 1, "status": "raw_red"}
        events.append({
            "lineage": key.label,
            "step": 0,
            "state_before": np.zeros(4, np.float32),
            "state_after": np.zeros(4, np.float32),
            "terminal": "raw_red",
        })
    return {"outcomes": outcomes, "events": events}


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _make_iterative_root(tmp_path):
    root = tmp_path / "iterative"
    cycle_root = root / "microcycles/cycle_001"
    cycle_root.mkdir(parents=True)
    selected_path = tmp_path / "selected.pt"
    reference_path = tmp_path / "r0.pt"
    selected_path.write_bytes(b"selected")
    reference_path.write_bytes(b"r0")
    selected = {
        "path": str(selected_path.resolve()),
        "sha256": HYBRID._sha256(selected_path),
        "policy_state_sha256": "a" * 64,
    }
    config = HYBRID.HybridConfig(max_microcycles=3)
    keys = HYBRID._lineage_keys(config)
    baseline = _raw(keys)
    baseline_path = root / "raw_gate_baseline.pt"
    torch.save(baseline, baseline_path)

    view = _view()
    training_payload = {
        "status": "SFM_HP100_ITERATIVE_MICROCYCLE_TRAINING_VIEW",
        "microcycle": 1,
        "training_view": view,
        "counts": {},
        "enters_GP": False,
        "source_cycles": [1],
        "admissible": True,
        "inadmissible_reason": None,
        "short_prefix_disasters": [],
    }
    training_path = cycle_root / "training_view.pt"
    torch.save(training_payload, training_path)
    training_canonical = ITER.canonical_sha256({
        "microcycle": 1, "training_view": view, "admissible": True,
    })
    cycle_marker = {
        "status": "SFM_HP100_ITERATIVE_MICROCYCLE_REJECTED",
        "microcycle": 1,
        "parent_policy_state_sha256": selected["policy_state_sha256"],
        "training_view_admissible": True,
        "GP_updated": False,
        "archive_committed": False,
        "training_counts": {
            "selected_training_exposures": {
                "retained_P1": 1, "retained_P2": 1,
                "Dminus": 0, "Ncausal": 1,
            },
        },
        "artifacts": {
            "training_view_sha256": HYBRID._sha256(training_path),
            "training_view_canonical_sha256": training_canonical,
        },
    }
    cycle_marker_path = cycle_root / "CYCLE_COMPLETE.json"
    _write_json(cycle_marker_path, cycle_marker)

    preflight = {
        "status": ITER.PREFLIGHT_STATUS,
        "source_gate": {"HEAD": SWEEP.EXPECTED_EVIDENCE_SOURCE_COMMIT},
        "selection": {"checkpoint": selected},
        "reference_checkpoint": str(reference_path.resolve()),
        "reference_checkpoint_sha256": HYBRID._sha256(reference_path),
        "config": asdict(config),
    }
    preflight_path = root / "PREFLIGHT.json"
    _write_json(preflight_path, preflight)
    marker = {
        "status": ITER.PLATEAU_STATUS,
        "source_HEAD": SWEEP.EXPECTED_EVIDENCE_SOURCE_COMMIT,
        "promotable": False,
        "GP_updated": False,
        "archive_committed": False,
        "initial_selected_checkpoint": selected,
        "preflight_sha256": HYBRID._sha256(preflight_path),
        "baseline_raw_gate_sha256": HYBRID._sha256(baseline_path),
        "baseline_raw_gate_canonical_sha256": ITER.canonical_sha256(baseline),
        "baseline_raw_gate": ITER._raw_gate_summary(baseline, keys),
        "cycles": [{
            "microcycle": 1,
            "marker": str(cycle_marker_path),
            "marker_sha256": HYBRID._sha256(cycle_marker_path),
        }],
    }
    marker_path = root / "ITERATIVE_COMPLETE.json"
    _write_json(marker_path, marker)
    return root, HYBRID._sha256(marker_path), training_path


def test_declared_doses_have_matched_positive_controls_and_treatments():
    doses = SWEEP.declared_doses("original")
    assert len(doses) == 8
    for pair in {dose.pair for dose in doses}:
        rows = [dose for dose in doses if dose.pair == pair]
        assert len(rows) == 2
        control = next(row for row in rows if row.positive_only)
        treatment = next(row for row in rows if not row.positive_only)
        assert (
            control.p1_learning_rate, control.p1_passes,
            control.p2_learning_rate, control.p2_passes,
        ) == (
            treatment.p1_learning_rate, treatment.p1_passes,
            treatment.p2_learning_rate, treatment.p2_passes,
        )
        assert treatment.negative_alpha == 0.1


def test_p1_dominant_family_is_new_paired_and_p1_is_not_weaker():
    doses = SWEEP.declared_doses("p1_dominant")
    assert len(doses) == 8
    assert {dose.pair for dose in doses} == {
        "balanced1", "retain1", "anchor4", "p1strong",
    }
    assert not ({dose.name for dose in doses} & {
        dose.name for dose in SWEEP.declared_doses("original")
    })
    for pair in {dose.pair for dose in doses}:
        rows = [dose for dose in doses if dose.pair == pair]
        control = next(row for row in rows if row.positive_only)
        treatment = next(row for row in rows if not row.positive_only)
        assert control.p1_learning_rate >= control.p2_learning_rate
        assert (
            control.p1_learning_rate, control.p1_passes,
            control.p2_learning_rate, control.p2_passes,
        ) == (
            treatment.p1_learning_rate, treatment.p1_passes,
            treatment.p2_learning_rate, treatment.p2_passes,
        )
        assert treatment.negative_learning_rate == 3.0e-5
        assert treatment.negative_alpha == 0.1
        assert treatment.negative_passes == 4


def test_balanced_depth_changes_only_pass_depth_and_caps_negative_passes():
    doses = SWEEP.declared_doses("balanced_depth")
    assert len(doses) == 8
    expected = {
        "balanced2": (2, 2),
        "balanced4": (4, 4),
        "balanced8": (8, 4),
        "balanced16": (16, 4),
    }
    prior_names = {
        dose.name
        for family in ("original", "p1_dominant")
        for dose in SWEEP.declared_doses(family)
    }
    assert not ({dose.name for dose in doses} & prior_names)
    for pair, (positive_passes, negative_passes) in expected.items():
        rows = [dose for dose in doses if dose.pair == pair]
        control = next(row for row in rows if row.positive_only)
        treatment = next(row for row in rows if not row.positive_only)
        for row in rows:
            assert row.p1_learning_rate == row.p2_learning_rate == 1.0e-5
            assert row.p1_passes == row.p2_passes == positive_passes
        assert treatment.negative_learning_rate == 3.0e-5
        assert treatment.negative_alpha == 0.1
        assert treatment.negative_passes == negative_passes


def test_parser_selects_dose_family():
    common = [
        "--repo-root", "/repo", "--expected-source-commit", "a" * 40,
        "--iterative-root", "/iterative",
        "--expected-iterative-marker-sha256", "b" * 64,
        "--cycle", "1", "--output", "/output", "--physical-gpu", "3",
    ]
    assert SWEEP.parser().parse_args(common).dose_family == "original"
    assert SWEEP.parser().parse_args(
        [*common, "--dose-family", "p1_dominant"]
    ).dose_family == "p1_dominant"
    assert SWEEP.parser().parse_args(
        [*common, "--dose-family", "balanced_depth"]
    ).dose_family == "balanced_depth"


def test_completed_cycle_authenticates_and_tampering_fails(tmp_path):
    root, marker_sha, training_path = _make_iterative_root(tmp_path)
    loaded = SWEEP.load_completed_cycle(root, marker_sha, 1)
    assert loaded["cycle"] == 1
    assert len(loaded["training_view"]["Ncausal"]) == 1
    assert loaded["selected_checkpoint"]["policy_state_sha256"] == "a" * 64

    training_path.write_bytes(training_path.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="training view SHA256 mismatch"):
        SWEEP.load_completed_cycle(root, marker_sha, 1)


def test_cells_share_checkpoint_view_and_select_lower_drift(tmp_path):
    parent_adapter = _Adapter()
    parent = {
        "path": "/selected.pt",
        "sha256": "b" * 64,
        "policy_state_sha256": HYBRID._model_state_sha256(parent_adapter),
    }
    starts, evaluated = [], []

    def factory(scope):
        return _Adapter(scope)

    def evaluate(adapter, cell):
        evaluated.append(cell)
        weight = float(adapter.policy.weight.detach())
        clear = ("a", "b") if weight > 1.2 else ("a",)
        return {"cell": cell}, _summary(clear, progress=1.1)

    result = SWEEP.evaluate_cells(
        adapter_factory=factory,
        training_view=_view(),
        baseline_summary=_summary(("a",)),
        evaluate_fn=evaluate,
        output=tmp_path,
        parent_checkpoint=parent,
        cycle_marker_sha256="c" * 64,
        batch_size=64,
        max_relative_parameter_drift=0.1,
        seed=2,
        update_fn=_fake_update(starts),
    )

    assert result["status"] == SWEEP.COMPLETE_STATUS
    assert len(result["cells"]) == 16
    assert len(starts) == 16
    assert all(start == 1.0 for _, start in starts)
    assert len(evaluated) == 16
    assert all(cell["Dminus_training_exposures"] == 0 for cell in result["cells"])
    assert result["winner"]["scope"] == "head_only"
    assert result["winner"]["dose"]["name"] == "soft1_positive_only"
    assert result["winner"]["acceptance"][
        "raw_temperature_one_CLEAR_delta"
    ] == 1
    assert Path(result["selected_checkpoint"]["path"]).is_file()


def test_no_new_clear_produces_no_winner(tmp_path):
    adapter = _Adapter()
    parent = {
        "path": "/selected.pt",
        "sha256": "d" * 64,
        "policy_state_sha256": HYBRID._model_state_sha256(adapter),
    }

    result = SWEEP.evaluate_cells(
        adapter_factory=lambda scope: _Adapter(scope),
        training_view=_view(),
        baseline_summary=_summary(("a",)),
        evaluate_fn=lambda adapter, cell: (
            {"cell": cell}, _summary(("a",), progress=1.2)
        ),
        output=tmp_path,
        parent_checkpoint=parent,
        cycle_marker_sha256="e" * 64,
        batch_size=64,
        max_relative_parameter_drift=0.1,
        seed=2,
        doses=SWEEP.declared_doses()[:2],
        update_fn=_fake_update([], make_eligible=False),
    )

    assert result["status"] == SWEEP.NO_WINNER_STATUS
    assert result["eligible_cells"] == 0
    assert result["winner"] is None
    assert result["selected_checkpoint"] is None


def test_selection_rejects_success_substitution_despite_higher_count():
    parent = _summary(("a",), progress=1.0)
    candidate = {
        **_summary(("b",), progress=1.1),
        "clear_lineages": ["b", "c"],
        "clear_count": 2,
        "outcomes": {
            "a": {"clear": False, "steps": 3},
            "b": {"clear": True, "steps": 3},
            "c": {"clear": True, "steps": 3},
        },
    }
    parent["outcomes"]["c"] = {"clear": False, "steps": 3}

    decision = ITER._acceptance_decision(parent, candidate)

    assert decision["raw_temperature_one_CLEAR_delta"] == 1
    assert decision["prior_CLEAR_subset_preserved"] is False
    assert decision["accepted"] is False


def test_raw_crn_registry_rejects_changed_flow_base():
    registry = {}
    payload = {
        "events": [{
            "lineage": "g0.1:rep00", "step": 0,
            "flow_base": torch.zeros(10, 2),
        }],
    }
    SWEEP._register_raw_bases(registry, payload)
    payload["events"][0]["flow_base"] = torch.ones(10, 2)
    with pytest.raises(RuntimeError, match="CRN flow base drifted"):
        SWEEP._register_raw_bases(registry, payload)
