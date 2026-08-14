import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

import sfm_hp100_exhaustive_hybrid as HYBRID
import sfm_hp100_exhaustive_hybrid_two_arm as LAUNCH
import sfm_hp100_exhaustive_hybrid_viz as VIZ


LINEAGE = "g0.1:rep00"


def _event(*, microcycle=1, step=0, repairs=0, terminal=None):
    return {
        "microcycle": int(microcycle),
        "step": int(step),
        "lineage": LINEAGE,
        "repair_attempts": [
            {"attempt": index, "base_std": 1.0 + 0.1 * index}
            for index in range(repairs)
        ],
    }


def _drawable_event(*, terminal=None):
    event = _event(terminal=terminal)
    event.update({
        "ped_xy": np.empty((0, 2), dtype=np.float32),
        "ped_vel": np.empty((0, 2), dtype=np.float32),
        "state_before": np.zeros(4, dtype=np.float32),
        "state_after": np.array([0.1, 0.0, 1.0, 0.0], dtype=np.float32),
        "raw_segment": np.zeros((11, 2), dtype=np.float32),
        "raw_verification": {"valid": True},
        "raw_sidecar": {},
        "terminal": terminal,
    })
    return event


def test_frames_reveal_every_repair_attempt_then_append_post_recheck_frame():
    trace = {
        "events": [
            _event(microcycle=1, step=0, repairs=2),
            _event(microcycle=1, step=1, repairs=0),
            _event(microcycle=2, step=0, repairs=1),
        ],
        "cycles": [
            {"microcycle": 1, "recheck": {"outcomes": {}}},
            {"microcycle": 2, "recheck": {"outcomes": {}}},
        ],
    }

    assert VIZ._frames(trace) == [
        (1, 0, None, False),
        (1, 0, 0, False),
        (1, 0, 1, False),
        (1, 1, None, False),
        (1, 1, None, True),
        (2, 0, None, False),
        (2, 0, 0, False),
        (2, 0, None, True),
    ]


def test_recheck_status_is_per_cycle_and_not_permanently_latched():
    trace = {
        "cycles": [
            {
                "microcycle": 1,
                "recheck": {"outcomes": {LINEAGE: {"clear": True}}},
            },
            {
                "microcycle": 2,
                "recheck": {"outcomes": {LINEAGE: {"clear": False}}},
            },
        ],
    }

    assert VIZ._recheck_status(trace) == {
        (1, LINEAGE): True,
        (2, LINEAGE): False,
    }


def test_clear_appears_only_post_recheck_and_success_terminal_is_green(monkeypatch):
    # Geometry itself has a separate exact-solver test; keep this test focused
    # on annotation timing and colors.
    monkeypatch.setattr(VIZ, "_draw_green", lambda *args, **kwargs: None)
    grouped = {(1, LINEAGE): [_drawable_event(terminal="success")]}
    status = {(1, LINEAGE): True, (2, LINEAGE): False}
    figure, axis = plt.subplots()
    try:
        VIZ.draw_cell(
            axis, grouped, status, gamma=0.1, replica=0,
            microcycle=1, step=0, repair_attempt_index=None,
            post_recheck=False,
        )
        text = {item.get_text(): item for item in axis.texts}
        assert not any(label.startswith("CLEAR") for label in text)
        assert text["SUCCESS"].get_color() == VIZ.CLEAR_GREEN

        axis.clear()
        VIZ.draw_cell(
            axis, grouped, status, gamma=0.1, replica=0,
            microcycle=1, step=0, repair_attempt_index=None,
            post_recheck=True,
        )
        assert "CLEAR μ=1" in {item.get_text() for item in axis.texts}

        axis.clear()
        VIZ.draw_cell(
            axis, grouped, status, gamma=0.1, replica=0,
            microcycle=2, step=0, repair_attempt_index=None,
            post_recheck=True,
        )
        assert not any(
            item.get_text().startswith("CLEAR") for item in axis.texts
        )
    finally:
        plt.close(figure)


def test_optional_ncausal_overlay_is_magenta_and_default_is_unchanged(monkeypatch):
    monkeypatch.setattr(VIZ, "_draw_green", lambda *args, **kwargs: None)
    event = _drawable_event()
    event["causal_negative"] = True
    event["executed_group"] = "P1"
    grouped = {(1, LINEAGE): [event]}
    figure, axis = plt.subplots()
    try:
        VIZ.draw_cell(
            axis, grouped, {}, gamma=0.1, replica=0,
            microcycle=1, step=0, repair_attempt_index=None,
            post_recheck=False,
        )
        assert VIZ.NCAUSAL_MAGENTA not in {
            line.get_color() for line in axis.lines
        }

        axis.clear()
        VIZ.draw_cell(
            axis, grouped, {}, gamma=0.1, replica=0,
            microcycle=1, step=0, repair_attempt_index=None,
            post_recheck=False, show_causal_negative=True,
        )
        magenta = [
            line for line in axis.lines
            if line.get_color() == VIZ.NCAUSAL_MAGENTA
        ]
        assert len(magenta) == 1
        assert magenta[0].get_linewidth() == pytest.approx(2.7)
    finally:
        plt.close(figure)


def _preflight_signature_payload():
    return {
        "version": HYBRID.VERSION,
        "checkpoint_sha256": "a" * 64,
        "config": {"ess_target": 0.1},
        "scene": {"name": "double_density_velocity_ood"},
        "scenario_start": 500_000,
        "calibration": {"feature_sha256": "b" * 64},
        "acquisition_reference": {"mode": "pretrained_gamma_matched"},
        "transaction": {"commit_gate": "all_clear"},
        "optimizer_scope": "head_only",
        "gpu": {"physical_gpu": 1},
    }


def test_shared_signature_ignores_only_arm_local_scope_and_gpu():
    left = _preflight_signature_payload()
    right = copy.deepcopy(left)
    right["optimizer_scope"] = "last_block_and_head"
    right["gpu"] = {"physical_gpu": 3}
    assert LAUNCH.shared_signature(left) == LAUNCH.shared_signature(right)

    right["acquisition_reference"]["mode"] = "empty_prior"
    assert LAUNCH.shared_signature(left) != LAUNCH.shared_signature(right)


@pytest.mark.parametrize(
    "terminal_statuses, expected_committed, expected_outcome",
    [
        (
            {
                "head_only": HYBRID.COMPLETE_STATUS,
                "last_block_and_head": HYBRID.COMPLETE_STATUS,
            },
            True,
            "both_scopes_passed_CLEAR_commit_gate",
        ),
        (
            {
                "head_only": HYBRID.COMPLETE_STATUS,
                "last_block_and_head": HYBRID.INCOMPLETE_STATUS,
            },
            False,
            "at_least_one_scope_incomplete",
        ),
    ],
)
def test_two_arm_delivery_reports_aggregate_commit_outcome(
    monkeypatch, tmp_path, terminal_statuses, expected_committed,
    expected_outcome,
):
    expected_sha = "a" * 64
    output_root = tmp_path / "delivery"

    monkeypatch.setattr(
        LAUNCH, "source_gate",
        lambda root, expected_commit, required_ref: {
            "root": str(root), "HEAD": expected_commit,
            "required_ref": required_ref,
            "required_ref_commit": expected_commit, "clean": True,
        },
    )
    monkeypatch.setattr(
        LAUNCH, "gpu_identity",
        lambda physical_gpu: {
            "physical_gpu": int(physical_gpu),
            "uuid": f"GPU-{physical_gpu}", "name": "test-gpu",
        },
    )
    monkeypatch.setattr(LAUNCH.HYBRID, "_sha256", lambda path: expected_sha)

    def fake_run_logged(command_value, *, env, log):
        mode = command_value[command_value.index("--mode") + 1]
        run_root = Path(command_value[command_value.index("--output") + 1])
        arm = run_root.parent.name
        physical_gpu = command_value[
            command_value.index("--physical-gpu") + 1
        ]
        assert env["CUDA_VISIBLE_DEVICES"] == physical_gpu
        if mode == "preflight":
            payload = _preflight_signature_payload()
            payload["status"] = HYBRID.PREFLIGHT_STATUS
            payload["optimizer_scope"] = command_value[
                command_value.index("--optimizer-scope") + 1
            ]
            payload["gpu"] = {"physical_gpu": int(physical_gpu)}
            path = run_root.with_suffix(".preflight.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload))
        else:
            run_root.mkdir(parents=True, exist_ok=True)
            status = terminal_statuses[arm]
            name = (
                "COMMIT_COMPLETE.json"
                if status == HYBRID.COMPLETE_STATUS
                else "QUALIFICATION_INCOMPLETE.json"
            )
            (run_root / name).write_text(json.dumps({"status": status}))
        return {
            "command": command_value, "log": str(log), "returncode": 0,
            "wall_seconds": 0.0, "log_sha256": expected_sha,
        }

    monkeypatch.setattr(LAUNCH, "_run_logged", fake_run_logged)

    result = LAUNCH.main([
        "--source-root", str(tmp_path / "source"),
        "--expected-source-commit", "c" * 40,
        "--checkpoint", str(tmp_path / "checkpoint.pt"),
        "--expected-checkpoint-sha256", expected_sha,
        "--pretrain-dataset-root", str(tmp_path / "dataset"),
        "--expected-pretrain-dataset-manifest-sha256", "d" * 64,
        "--output-root", str(output_root),
    ])

    assert result == 0
    delivery = json.loads((output_root / "TWO_ARM_COMPLETE.json").read_text())
    assert delivery["all_arms_committed"] is expected_committed
    assert delivery["scientific_outcome"] == expected_outcome
    assert [row["arm"] for row in delivery["arms"]] == [
        "head_only", "last_block_and_head",
    ]
